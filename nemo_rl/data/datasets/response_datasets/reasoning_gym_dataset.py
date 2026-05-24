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

"""Response dataset built from reasoning_gym category fractions."""

import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from typing import Any

from nemo_rl.data.datasets.response_datasets import _reasoning_gym_patches

_reasoning_gym_patches.apply_all()

import reasoning_gym  # noqa: E402
from datasets import Dataset, load_from_disk  # noqa: E402
from reasoning_gym.factory import DATASETS  # noqa: E402

from nemo_rl.data.datasets.raw_dataset import RawDataset  # noqa: E402

# Follows Table 6 of the Nemotron-RL ReasoningGym paper (arXiv:2505.24760):
# `arc` envs are folded into `cognition`; `probability` and `composite` are excluded.
_CATEGORY_ALIASES: dict[str, str] = {"arc": "cognition"}
_EXCLUDED_CATEGORIES: set[str] = {"composite", "probability"}


def _build_category_to_envs() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, (cls, _cfg) in DATASETS.items():
        mod_parts = cls.__module__.split(".")
        cat = mod_parts[1] if len(mod_parts) > 1 else "unknown"
        if cat in _EXCLUDED_CATEGORIES:
            continue
        cat = _CATEGORY_ALIASES.get(cat, cat)
        out.setdefault(cat, []).append(name)
    return out


CATEGORY_TO_ENVS: dict[str, list[str]] = _build_category_to_envs()
ALL_CATEGORIES: list[str] = sorted(CATEGORY_TO_ENVS.keys())
ENV_TO_CATEGORY: dict[str, str] = {
    env: cat for cat, envs in CATEGORY_TO_ENVS.items() for env in envs
}


def _env_counts(
    category_fractions: dict[str, float], size: int, rng: random.Random
) -> dict[str, int]:
    env_weights: list[tuple[str, float]] = []
    for cat, frac in category_fractions.items():
        if frac <= 0:
            continue
        if cat not in CATEGORY_TO_ENVS:
            raise ValueError(
                f"Unknown reasoning_gym category {cat!r}. "
                f"Available: {ALL_CATEGORIES}"
            )
        envs = CATEGORY_TO_ENVS[cat]
        per_env = frac / len(envs)
        for env_name in envs:
            env_weights.append((env_name, per_env))
    if not env_weights:
        raise ValueError(
            "No reasoning_gym envs selected; all category_fractions are 0."
        )

    total = sum(w for _, w in env_weights)
    # Floor allocation, then distribute the remainder deterministically by sampling
    raw = {name: size * w / total for name, w in env_weights}
    counts = {name: int(v) for name, v in raw.items()}
    remainder = size - sum(counts.values())
    if remainder > 0:
        fracs = sorted(
            ((raw[n] - counts[n], n) for n in raw),
            key=lambda x: (-x[0], x[1]),
        )
        for _, name in fracs[:remainder]:
            counts[name] += 1
    return {n: c for n, c in counts.items() if c > 0}


def _read_text_file(path: str | None) -> str | None:
    if not path:
        return None
    with open(path) as f:
        return f.read().strip()


def _file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return f"missing:{path}"


def _cache_fingerprint(
    *,
    category_fractions: dict[str, float],
    size: int,
    items_per_env: int,
    seed: int,
    split: str,
    tokenizer_name: str | None,
    max_prompt_tokens: int | None,
    prompt_file: str | None,
    system_prompt_file: str | None,
    oversample_factor: float,
) -> str:
    payload = json.dumps(
        {
            "category_fractions": sorted(category_fractions.items()),
            "size": size,
            "items_per_env": items_per_env,
            "seed": seed,
            "split": split,
            "tokenizer_name": tokenizer_name,
            "max_prompt_tokens": max_prompt_tokens,
            "prompt_file_sha": _file_sha256(prompt_file),
            "system_prompt_file_sha": _file_sha256(system_prompt_file),
            "oversample_factor": oversample_factor,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_save_dataset(ds: Dataset, dest: str) -> None:
    """Save then rename so a crash mid-save doesn't leave a half-written cache."""
    tmp = f"{dest}.tmp.{os.getpid()}"
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    ds.save_to_disk(tmp)
    try:
        os.rename(tmp, dest)
    except OSError:
        # Concurrent run won the race; drop our tmp copy.
        shutil.rmtree(tmp, ignore_errors=True)


def _report_and_filter_by_length(
    rows: list[dict[str, Any]],
    *,
    tokenizer_name: str,
    prompt_file: str | None,
    system_prompt_file: str | None,
    max_prompt_tokens: int | None,
    split: str,
) -> list[dict[str, Any]]:
    """Tokenize every row's final prompt, log percentiles, drop rows over the cap."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    prompt_tmpl = _read_text_file(prompt_file)
    system_prompt = _read_text_file(system_prompt_file)

    lengths: list[int] = []
    for row in rows:
        question = row["messages"][0]["content"]
        formatted = prompt_tmpl.format(question) if prompt_tmpl else question
        message_list = []
        if system_prompt:
            message_list.append({"role": "system", "content": system_prompt})
        message_list.append({"role": "user", "content": formatted})
        text = tok.apply_chat_template(
            message_list,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        lengths.append(n)

    def pct(xs: list[int], p: float) -> int:
        i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
        return xs[i]

    def summarize(xs: list[int]) -> str:
        return (
            f"n={len(xs):>5} min={xs[0]:>4} p25={pct(xs, 0.25):>4} "
            f"p50={pct(xs, 0.50):>4} p75={pct(xs, 0.75):>4} "
            f"p90={pct(xs, 0.90):>4} p99={pct(xs, 0.99):>4} "
            f"max={xs[-1]:>5} mean={sum(xs) / len(xs):>6.1f}"
        )

    if lengths:
        xs = sorted(lengths)
        print(
            f"[ReasoningGymDataset] split={split} tokenizer={tokenizer_name} "
            f"prompt-token percentiles (overall): {summarize(xs)}"
        )

        by_category: dict[str, list[int]] = defaultdict(list)
        for row, n in zip(rows, lengths):
            cat = ENV_TO_CATEGORY.get(row["dataset_name"], "unknown")
            by_category[cat].append(n)
        print(f"[ReasoningGymDataset] split={split} per-category percentiles:")
        for cat in sorted(by_category):
            cxs = sorted(by_category[cat])
            print(f"  {cat:<12} {summarize(cxs)}")

    if max_prompt_tokens is None:
        return rows

    kept: list[dict[str, Any]] = []
    per_env_dropped: dict[str, int] = defaultdict(int)
    for row, n in zip(rows, lengths):
        if n <= max_prompt_tokens:
            kept.append(row)
        else:
            per_env_dropped[row["dataset_name"]] += 1
    dropped = len(rows) - len(kept)
    print(
        f"[ReasoningGymDataset] split={split} dropped {dropped}/{len(rows)} rows "
        f"over max_prompt_tokens={max_prompt_tokens}"
    )
    if per_env_dropped:
        top = sorted(per_env_dropped.items(), key=lambda x: -x[1])[:5]
        print(
            "[ReasoningGymDataset]   worst-offender envs: "
            + ", ".join(f"{n}={c}" for n, c in top)
        )
    if not kept:
        raise ValueError(
            f"All rows exceeded max_prompt_tokens={max_prompt_tokens}. "
            "Increase the cap or raise policy.max_total_sequence_length."
        )
    return kept


class ReasoningGymDataset(RawDataset):
    """Materialized HuggingFace Dataset sampled from reasoning_gym envs.

    Each row has:
      - ``messages``: ``[{"role": "user", "content": <question>}]``
      - ``task_name``: always ``"reasoning_gym"`` (so one entry in task_to_env suffices)
      - ``dataset_name``: the specific reasoning_gym env (e.g. ``"chain_sum"``), used
        by ``ReasoningGymEnvironment`` for scoring and per-env metric breakdown
      - ``entry``: JSON-serialized original ``reasoning_gym`` entry dict

    Args:
        category_fractions: mapping of category -> weight. Weights are normalized;
            each env within a category gets an equal share of its category's weight.
            Follows Table 6 of arXiv:2505.24760: `arc` is folded into `cognition`;
            `probability` is not present.
        size: total number of rows to materialize.
        items_per_env: upper bound on per-env generator size; used for
            re-seeding when a requested per-env count exceeds this.
        seed: seed for both sample planning and per-env generator seeds.
        split: ``"train"`` or ``"validation"`` — offsets the seed so splits
            don't overlap when both are built from the same fractions.
        tokenizer_name: HuggingFace tokenizer identifier used to measure
            prompt token lengths. When provided, a percentile report is
            printed and (if ``max_prompt_tokens`` is set) rows whose fully
            formatted prompt exceeds the cap are dropped.
        max_prompt_tokens: hard cap on the tokenized prompt length (after
            chat-template application, matching what the processor emits).
            Requires ``tokenizer_name``.
        cache_dir: when set, materialized rows are persisted to
            ``<cache_dir>/<split>-<fp16>/`` (HuggingFace ``save_to_disk``
            format) and reloaded on subsequent runs with matching inputs.
            Pins eval samples across reasoning_gym package upgrades and
            avoids re-running the per-env generators on every eval. Cache
            key includes category fractions, size, items_per_env, seed,
            split, tokenizer, max_prompt_tokens, oversample_factor, and
            content hashes of the prompt files — change any of these and a
            fresh dataset is built and saved alongside.
        oversample_factor: multiplier on the planned per-env counts before
            length filtering. Generates ``int(size * oversample_factor)``
            rows, length-filters, then randomly subsamples down to exactly
            ``size`` rows. Use a value > 1.0 when length filtering would
            otherwise drop the final size below ``size`` (raises if even
            with oversampling fewer than ``size`` rows survive).
    """

    def __init__(
        self,
        category_fractions: dict[str, float],
        size: int = 10000,
        items_per_env: int = 5000,
        seed: int = 42,
        split: str = "train",
        tokenizer_name: str | None = None,
        max_prompt_tokens: int | None = None,
        prompt_file: str | None = None,
        system_prompt_file: str | None = None,
        cache_dir: str | None = None,
        oversample_factor: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.task_name = "reasoning_gym"
        self.val_dataset = None

        if oversample_factor < 1.0:
            raise ValueError(
                f"oversample_factor must be >= 1.0 (got {oversample_factor})."
            )

        cache_path: str | None = None
        if cache_dir:
            fp = _cache_fingerprint(
                category_fractions=dict(category_fractions),
                size=size,
                items_per_env=items_per_env,
                seed=seed,
                split=split,
                tokenizer_name=tokenizer_name,
                max_prompt_tokens=max_prompt_tokens,
                prompt_file=prompt_file,
                system_prompt_file=system_prompt_file,
                oversample_factor=oversample_factor,
            )
            cache_path = os.path.join(cache_dir, f"{split}-{fp[:16]}")
            if os.path.isdir(cache_path):
                self.dataset = load_from_disk(cache_path)
                print(
                    f"[ReasoningGymDataset] split={split} cache HIT "
                    f"size={len(self.dataset)} path={cache_path}"
                )
                return
            print(
                f"[ReasoningGymDataset] split={split} cache MISS; building "
                f"and will save to {cache_path}"
            )
            os.makedirs(cache_dir, exist_ok=True)

        split_offset = {"train": 0, "validation": 1}.get(split, 2)
        rng = random.Random(seed + split_offset)

        gen_size = max(size, int(round(size * oversample_factor)))
        counts = _env_counts(dict(category_fractions), gen_size, rng)

        rows: list[dict[str, Any]] = []
        for env_name, n in counts.items():
            pending = n
            env_seed_iter = rng.randint(0, 2**31)
            while pending > 0:
                chunk = min(pending, items_per_env)
                try:
                    ds = reasoning_gym.create_dataset(
                        env_name, seed=env_seed_iter, size=chunk
                    )
                except Exception as e:
                    print(
                        f"[ReasoningGymDataset] skipping {env_name!r}: {e}"
                    )
                    pending = 0
                    break
                for i in range(len(ds)):
                    entry = ds[i]
                    rows.append(
                        {
                            "messages": [
                                {"role": "user", "content": entry["question"]}
                            ],
                            "task_name": self.task_name,
                            "dataset_name": env_name,
                            "entry": json.dumps(entry, default=str),
                        }
                    )
                pending -= len(ds)
                env_seed_iter = rng.randint(0, 2**31)

        rng.shuffle(rows)

        if max_prompt_tokens is not None and not tokenizer_name:
            raise ValueError(
                "ReasoningGymDataset: max_prompt_tokens requires tokenizer_name."
            )

        if tokenizer_name:
            rows = _report_and_filter_by_length(
                rows,
                tokenizer_name=tokenizer_name,
                prompt_file=prompt_file,
                system_prompt_file=system_prompt_file,
                max_prompt_tokens=max_prompt_tokens,
                split=split,
            )

        if len(rows) < size:
            raise ValueError(
                f"[ReasoningGymDataset] split={split}: only {len(rows)} rows "
                f"survived after generating {gen_size} (oversample_factor="
                f"{oversample_factor}) and length filtering, but size={size} "
                f"is required. Increase oversample_factor."
            )
        if len(rows) > size:
            rng.shuffle(rows)
            rows = rows[:size]

        per_env = defaultdict(int)
        for r in rows:
            per_env[r["dataset_name"]] += 1
        print(
            f"[ReasoningGymDataset] split={split} size={len(rows)} "
            f"envs={len(per_env)}"
        )

        self.dataset = Dataset.from_list(rows)
        if cache_path:
            _atomic_save_dataset(self.dataset, cache_path)
            print(
                f"[ReasoningGymDataset] split={split} cache SAVED "
                f"size={len(self.dataset)} path={cache_path}"
            )
