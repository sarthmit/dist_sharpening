# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import gc
import multiprocessing as _mp
import os
import re
from concurrent.futures import (
    ProcessPoolExecutor,
    TimeoutError as _FuturesTimeout,
)
from concurrent.futures.process import BrokenProcessPool
from typing import Any, TypedDict

import psutil
import ray
import torch

from nemo_rl.data.datasets.response_datasets import _reasoning_gym_patches

# Apply patches before reasoning_gym is touched: score_answer for several
# envs evaluates the model's raw output via builtin eval(), which is both
# an OOM and RCE surface. _patch_unsafe_eval shadows it with literal_eval.
_reasoning_gym_patches.apply_all()

import reasoning_gym  # noqa: E402

# Use forkserver for the scoring workers. The forkserver is a single
# subprocess that pre-imports reasoning_gym + sympy *once*, then forks each
# scoring worker from itself. Forks inherit the imports COW, so respawning
# a worker after a hang costs ~ms instead of ~7s of re-import. The
# forkserver itself is single-threaded by design, so the "fork-after-
# threads" hazard doesn't apply.
_FORKSERVER_CTX = _mp.get_context("forkserver")
_FORKSERVER_CTX.set_forkserver_preload(
    [
        "nemo_rl.data.datasets.response_datasets._reasoning_gym_patches",
        "reasoning_gym",
        "reasoning_gym.factory",
    ]
)

from nemo_rl.data.interfaces import LLMMessageLogType  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.environments.interfaces import (  # noqa: E402
    EnvironmentInterface,
    EnvironmentReturn,
)


class ReasoningGymMetadata(TypedDict):
    dataset_name: str
    entry: dict[str, Any]


_ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    matches = _ANSWER_TAG_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    return None


# Instance attrs that some ProceduralDataset subclasses materialize in
# __init__ for generation (__getitem__), but that score_answer never reads.
# Stripping them after construction shrinks each cached bound method from
# potentially gigabytes to kilobytes.
_HEAVY_INSTANCE_ATTRS = (
    "_tasks",            # arc_agi: full ARC train+eval task dict
    "_task_ids",         # arc_agi
    "_generators",       # rearc: dict of re-ARC generator callables
    "get_pso_difficulty",  # rearc: bound function reference
    "puzzles",           # rush_hour, sokoban, etc.: loaded puzzle lists
)

# Class-level data caches that some datasets populate in __init__. Once we
# have the bound score_answer method, these are irrelevant for scoring.
_HEAVY_CLASS_ATTRS: dict[str, tuple[str, ...]] = {
    "CodeIODataset": ("_jsonl_data",),
}


def _malloc_trim() -> None:
    # glibc holds large freed allocations in arenas instead of returning them
    # to the OS, so RSS can climb monotonically even when Python has released
    # the references. malloc_trim(0) forces the arenas back to the OS.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _build_score_fn(dataset_name: str):
    """Build a score_answer callable for ``dataset_name`` that does NOT
    retain the full ProceduralDataset instance state.

    ``reasoning_gym.get_score_answer_fn`` returns a bound method whose
    closure keeps the entire dataset instance alive. For heavy datasets
    (arc_agi, rearc, codeio, rush_hour, ...) this pins multi-GB task
    dictionaries and generator tables. ``score_answer`` itself only needs
    a handful of config attrs, so we construct the instance, grab the
    bound method, and null out the heavy attrs it never reads.
    """
    try:
        from reasoning_gym.factory import DATASETS
    except Exception:
        return reasoning_gym.get_score_answer_fn(dataset_name)

    if dataset_name not in DATASETS:
        return reasoning_gym.get_score_answer_fn(dataset_name)

    dataset_cls, config_cls = DATASETS[dataset_name]
    instance = dataset_cls(config=config_cls())
    score_fn = instance.score_answer

    for attr in _HEAVY_INSTANCE_ATTRS:
        if attr in instance.__dict__:
            try:
                setattr(instance, attr, None)
            except AttributeError:
                pass

    for cls_attr in _HEAVY_CLASS_ATTRS.get(dataset_cls.__name__, ()):
        if hasattr(dataset_cls, cls_attr):
            try:
                setattr(dataset_cls, cls_attr, None)
            except (AttributeError, TypeError):
                pass

    return score_fn


def _worker_init() -> None:
    # ProcessPoolExecutor with mp_context="spawn" gives a fresh interpreter.
    # Two things to do up front so the per-call timeout doesn't end up
    # bounding cold imports:
    # 1. Re-apply patches (maze __getitem__ retry, unsafe-eval shadow).
    # 2. Force-import every reasoning_gym dataset class. The factory's
    #    DATASETS dict references the classes by object, so `from … import
    #    DATASETS` transitively loads all env modules (sympy, etc.). Without
    #    this, the first score call on env X pays a multi-second import as
    #    pickle resolves the bound method's class.
    _reasoning_gym_patches.apply_all()
    from reasoning_gym.factory import DATASETS  # noqa: F401


def _worker_noop() -> int:
    # Pre-warm hook: by the time this returns, _worker_init has run and
    # reasoning_gym is fully imported in this worker.
    return 0


def _worker_run(score_fn: Any, answer: str | None, entry: dict[str, Any]) -> float:
    # Stateless: the bound score_answer method is unpickled from the actor's
    # cache on each call. Unpickling triggers the dataset-class import, which
    # is cached by the worker's module system, so steady-state cost is just
    # the call itself plus a few KB of IPC.
    score = score_fn(answer=answer, entry=entry)
    return float(score) if score is not None else 0.0


@ray.remote
class ReasoningGymEnvironment(EnvironmentInterface[ReasoningGymMetadata]):
    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}
        # Per-call wall-clock cap. Legitimate scoring is sub-millisecond
        # (workers are pre-warmed in _spawn_pool so cold imports don't eat
        # the budget); anything exceeding 5s is the suspected hang and
        # gets killed (the affected pool only) + pinned to 0.0.
        self._score_timeout_s = float(self.cfg.get("score_timeout_s", 5.0))
        self._max_workers = int(self.cfg.get("score_workers", 8))
        self._step_count = 0
        self._proc = psutil.Process(os.getpid())
        self._n_timeouts = 0
        self._n_broken = 0
        # Bound score_answer cache lives in the actor: it survives worker
        # kills, so a hung-then-killed worker doesn't force us to re-build
        # ProceduralDataset instances. The worker is stateless.
        self._score_fns: dict[str, Any] = {}
        # N independent single-worker pools rather than one N-worker pool:
        # a hang on one input only kills its own pool, and the other N-1
        # pools keep their warm sympy imports + keep processing the batch
        # in parallel during the timeout.
        self._pools: list[ProcessPoolExecutor | None] = [
            None
        ] * self._max_workers
        # Construct all pools first (cheap — workers haven't spawned yet),
        # then warm them in parallel by submitting noops to all and waiting.
        # Sequential warmup would be ~7s × N; parallel is ~max(import_s).
        for i in range(self._max_workers):
            self._pools[i] = ProcessPoolExecutor(
                max_workers=1,
                mp_context=_FORKSERVER_CTX,
                initializer=_worker_init,
            )
        warmups = [
            (i, self._pools[i].submit(_worker_noop))
            for i in range(self._max_workers)
        ]
        for _, f in warmups:
            try:
                f.result(timeout=120.0)
            except Exception:
                pass

    def _get_score_fn(self, dataset_name: str):
        fn = self._score_fns.get(dataset_name)
        if fn is None:
            fn = _build_score_fn(dataset_name)
            self._score_fns[dataset_name] = fn
        return fn

    def _spawn_pool(self, i: int) -> None:
        # forkserver: each worker is forked from a long-lived forkserver
        # subprocess that has reasoning_gym + sympy already imported, so
        # respawning a single pool after a hang costs ~ms (just a fork)
        # rather than re-importing sympy (~7s).
        pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=_FORKSERVER_CTX,
            initializer=_worker_init,
        )
        # Still send one noop and wait, both to surface init errors early
        # and to ensure the forked worker has finished _worker_init before
        # the per-call timeout starts ticking on real work.
        try:
            pool.submit(_worker_noop).result(timeout=120.0)
        except Exception:
            pass
        self._pools[i] = pool

    def _kill_pool(self, i: int) -> None:
        # shutdown(wait=False) won't preempt a hung worker; kill the child
        # directly. _processes is CPython-internal but stable on 3.9+.
        pool = self._pools[i]
        if pool is None:
            return
        for p in list(getattr(pool, "_processes", {}).values()):
            try:
                p.kill()
            except Exception:
                pass
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._pools[i] = None

    def _kill_all_pools(self) -> None:
        for i in range(len(self._pools)):
            self._kill_pool(i)

    def _submit_to_pool(
        self,
        pool_idx: int,
        score_fn: Any,
        answer: str | None,
        entry: dict[str, Any],
    ) -> Any | None:
        """Submit one scoring job to ``self._pools[pool_idx]``. If the pool
        is broken, respawn it once and retry. Returns the future, or None
        if the submit ultimately failed."""
        pool = self._pools[pool_idx]
        if pool is None:
            self._spawn_pool(pool_idx)
            pool = self._pools[pool_idx]
        try:
            return pool.submit(_worker_run, score_fn, answer, entry)
        except (BrokenProcessPool, RuntimeError):
            self._n_broken += 1
            self._kill_pool(pool_idx)
            self._spawn_pool(pool_idx)
            try:
                return self._pools[pool_idx].submit(
                    _worker_run, score_fn, answer, entry
                )
            except Exception:
                return None
        except Exception:
            return None

    def _score_batch(
        self, items: list[tuple[str, str | None, dict[str, Any]]]
    ) -> list[float]:
        """Score a step's worth of (dataset_name, answer, entry) tuples.

        Items are round-robin'd across N single-worker pools so they run in
        parallel. We then walk futures in submission order with a per-call
        ``score_timeout_s`` cap on each ``fut.result()``.

        When one call hangs we kill *only* that pool, pin the item at 0.0,
        and resubmit any items behind it on the same pool (those futures
        became ``BrokenProcessPool`` when we killed the pool). The other
        N-1 pools keep their warm imports and continue processing the rest
        of the batch in parallel.
        """
        n = len(items)
        results: list[float | None] = [None] * n
        pending = list(range(n))
        n_pools = len(self._pools)

        for _pass in range(3):
            if not pending:
                break

            # Submit, distributing round-robin across pools. The dispatch
            # is deterministic (idx % n_pools) so a single bad input lands
            # on the same pool every retry — but we pin its result before
            # the next pass, so it isn't actually re-tried.
            submitted: list[tuple[Any, int, int]] = []  # (fut, idx, pool_idx)
            for i in pending:
                ds_name, ans, entry = items[i]
                try:
                    score_fn = self._get_score_fn(ds_name)
                except Exception:
                    results[i] = 0.0
                    continue
                pool_idx = i % n_pools
                fut = self._submit_to_pool(pool_idx, score_fn, ans, entry)
                if fut is None:
                    results[i] = 0.0
                    continue
                submitted.append((fut, i, pool_idx))

            new_pending: list[int] = []
            dead_pools: set[int] = set()
            for fut, idx, pool_idx in submitted:
                if pool_idx in dead_pools:
                    # Pool was killed earlier in this pass; this fut is
                    # broken (or queued behind the hang). Resubmit it next
                    # pass — the pool has been respawned by then.
                    new_pending.append(idx)
                    continue
                try:
                    results[idx] = float(
                        fut.result(timeout=self._score_timeout_s)
                    )
                except _FuturesTimeout:
                    self._n_timeouts += 1
                    results[idx] = 0.0  # pin: don't retry the hung input
                    self._kill_pool(pool_idx)
                    self._spawn_pool(pool_idx)
                    dead_pools.add(pool_idx)
                except BrokenProcessPool:
                    self._n_broken += 1
                    if pool_idx not in dead_pools:
                        self._kill_pool(pool_idx)
                        self._spawn_pool(pool_idx)
                        dead_pools.add(pool_idx)
                    new_pending.append(idx)
                except Exception:
                    results[idx] = 0.0

            pending = new_pending

        return [r if r is not None else 0.0 for r in results]

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[ReasoningGymMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[ReasoningGymMetadata]:
        observations = []
        rewards = []
        terminateds = []
        all_stop_strings: list[list[str] | None] = []
        all_metadata = []
        all_answers: list[str | None] = []

        items: list[tuple[str, str | None, dict[str, Any]]] = []
        for msg_log, meta in zip(message_log_batch, metadata):
            response = ""
            if msg_log and msg_log[-1]["role"] == "assistant":
                response = msg_log[-1]["content"].strip()

            answer = extract_answer(response) or response
            all_answers.append(answer)
            items.append((meta["dataset_name"], answer, meta["entry"]))

        scores = self._score_batch(items)
        for score in scores:
            rewards.append(score)
            terminateds.append(True)
            observations.append({"role": "environment", "content": ""})
            all_stop_strings.append(None)
            all_metadata.append(None)

        self._step_count += 1
        # Force a full GC every few steps: some reasoning_gym code paths
        # (e.g. codeio's tree edit distance) create many short-lived
        # reference cycles that the cyclic collector would otherwise
        # service lazily, causing the actor's RSS to climb.
        if self._step_count % 25 == 0:
            gc.collect()
            _malloc_trim()
            rss_gb = self._proc.memory_info().rss / 1024**3
            print(
                f"[ReasoningGymEnv] step={self._step_count} "
                f"rss={rss_gb:.2f}GB cached_fns={len(self._score_fns)} "
                f"score_timeouts={self._n_timeouts} "
                f"score_broken={self._n_broken}",
                flush=True,
            )

        return EnvironmentReturn(
            observations=observations,
            metadata=all_metadata,
            next_stop_strings=all_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=all_answers if return_extracted_answer else None,
        )

    def shutdown(self):
        self._kill_all_pools()

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        accuracy = (
            (final_rewards > 0.5).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )

        metrics: dict[str, float] = {"accuracy": accuracy}

        # Per-env breakdown: keyed by reasoning_gym env name stored in
        # extra_env_info (sample-level task_name is uniform = "reasoning_gym").
        env_infos = batch.get("extra_env_info", [])
        if env_infos:
            from nemo_rl.data.datasets.response_datasets.reasoning_gym_dataset import (
                ENV_TO_CATEGORY,
            )

            per_env: dict[str, list[float]] = {}
            per_category: dict[str, list[float]] = {}
            for i, info in enumerate(env_infos):
                if i >= len(final_rewards):
                    break
                name = info.get("dataset_name") if isinstance(info, dict) else None
                if not name:
                    continue
                reward_val = final_rewards[i].item()
                per_env.setdefault(name, []).append(reward_val)
                category = ENV_TO_CATEGORY.get(name)
                if category is not None:
                    per_category.setdefault(category, []).append(reward_val)
            for name, rews in per_env.items():
                metrics[f"accuracy/{name}"] = (
                    sum(1 for r in rews if r > 0.5) / len(rews)
                )
            for cat, rews in per_category.items():
                metrics[f"accuracy/category/{cat}"] = (
                    sum(1 for r in rews if r > 0.5) / len(rews)
                )

        return batch, metrics
