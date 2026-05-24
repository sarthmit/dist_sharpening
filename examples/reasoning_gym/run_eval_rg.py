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

"""Evaluate a checkpoint on each ReasoningGym category split.

Mirrors the training-time validation path exactly:
  - same prompt template (data.default.prompt_file)
  - same tokenizer + chat template (policy.tokenizer)
  - same data processor (reasoning_gym_data_processor)
  - same environment (ReasoningGymEnvironment)
  - same rollout loop (run_multi_turn_rollout)

Each entry in ``data.validation`` is loaded independently so per-category
accuracy is reported. By default this is 10 categories (one fraction set to
1.0 each) as defined in examples/reasoning_gym/grpo_reasoning_gym.yaml.

Usage:
    python examples/reasoning_gym/run_eval_rg.py \\
        --config examples/configs/evals/reasoning_gym_eval.yaml \\
        ++eval.checkpoint=/path/to/hf_ckpt \\
        ++eval.save_path=/path/to/results

The checkpoint must be in HuggingFace format. Convert a DCP checkpoint via
``examples/converters/convert_dcp_to_hf.py`` first.

Any unrecognized CLI flag is forwarded as a Hydra override (same as
run_grpo.py), so all training overrides — e.g. ``++policy.model_name``,
``++data.default_val_size``, ``++policy.generation.vllm_cfg.tensor_parallel_size``
— work here too.
"""

import argparse
import json
import os
import pprint
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
from nemo_rl.data.datasets.utils import update_single_dataset_config
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.environments.utils import create_env
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on each ReasoningGym category split"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _category_label(entry_cfg: dict[str, Any]) -> str:
    """Use the single non-zero category fraction as the split label.

    The training config defines one validation entry per category with a
    single fraction set to 1.0; falling back to a "+"-joined label keeps
    custom multi-category splits readable.
    """
    fracs = entry_cfg.get("category_fractions") or {}
    nonzero = [c for c, f in fracs.items() if f and f > 0]
    if len(nonzero) == 1:
        return nonzero[0]
    return "+".join(sorted(nonzero)) or "unknown"


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "evals",
            "reasoning_gym_eval.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    # Plumb eval.checkpoint -> policy.model_name *before* OmegaConf resolves.
    # The training config chain is
    #   policy.tokenizer.name = ${policy.model_name}
    #   data.train.tokenizer_name = ${policy.tokenizer.name}
    #   data.validation[*].tokenizer_name = ${data.train.tokenizer_name}
    # so resolving with policy.model_name=None bakes None into every val
    # entry's tokenizer_name, which silently disables the prompt-length
    # filter that training applied. Setting it here keeps the resolved
    # config byte-identical to a training run pointed at this checkpoint.
    ckpt = OmegaConf.select(config, "eval.checkpoint", default=None)
    assert ckpt, (
        "eval.checkpoint must be set to an HF-format checkpoint directory.\n"
        "  e.g. ++eval.checkpoint=/path/to/hf_ckpt\n"
        "Convert a DCP checkpoint first via examples/converters/convert_dcp_to_hf.py."
    )
    OmegaConf.update(config, "policy.model_name", ckpt, force_add=True)

    config = OmegaConf.to_container(config, resolve=True)
    eval_cfg = config["eval"]

    set_seed(eval_cfg.get("seed", 42))
    save_path = eval_cfg.get("save_path")
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)

    print("Final config:")
    pprint.pprint(config)

    init_ray()

    # Tokenizer must match training: chat template + thinking-mode kwargs
    # are inherited from the training config.
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    # Generation config: add model_name (training does this in setup), then
    # let configure_generation_config flip load_format=auto for is_eval=True.
    gen_cfg = config["policy"]["generation"]
    gen_cfg["model_name"] = ckpt
    gen_cfg = configure_generation_config(gen_cfg, tokenizer, is_eval=True)

    # Cluster
    cluster_cfg = config["cluster"]
    cluster = RayVirtualCluster(
        name="rg_eval_cluster",
        bundle_ct_per_node_list=[cluster_cfg["gpus_per_node"]]
        * cluster_cfg["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_cfg["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    print(
        f"  ✓ Ray cluster: {cluster_cfg['num_nodes']} node(s) × "
        f"{cluster_cfg['gpus_per_node']} GPU(s)"
    )

    print(f"\n▶ Loading vLLM with weights from: {ckpt}")
    vllm_gen = VllmGeneration(cluster=cluster, config=gen_cfg)

    # Build env once; all splits reuse it.
    env_cfg = config["env"]["reasoning_gym"]
    rg_env = create_env("reasoning_gym", env_cfg)
    task_to_env = {"reasoning_gym": rg_env}

    data_cfg = config["data"]
    val_entries = data_cfg.get("validation")
    assert val_entries, (
        "data.validation must be a non-empty list (one entry per RG category)."
    )
    if isinstance(val_entries, dict):
        val_entries = [val_entries]

    max_samples = eval_cfg.get("max_samples_per_split")
    batch_size = int(
        eval_cfg.get("val_batch_size") or config["grpo"]["val_batch_size"]
    )
    max_seq_len = config["policy"]["max_total_sequence_length"]
    greedy = bool(eval_cfg.get("greedy", True))
    num_tests_per_prompt = int(eval_cfg.get("num_tests_per_prompt", 1))
    k_value = int(eval_cfg.get("k_value", 1))
    assert num_tests_per_prompt >= k_value, (
        f"num_tests_per_prompt ({num_tests_per_prompt}) must be >= k_value ({k_value})"
    )
    if num_tests_per_prompt > 1:
        assert not greedy and gen_cfg["temperature"] > 0, (
            "num_tests_per_prompt > 1 requires non-greedy decoding "
            "(set ++eval.greedy=false ++eval.temperature=>0)."
        )

    summary: dict[str, dict[str, Any]] = {}
    overall_rewards: list[float] = []
    t_total = time.perf_counter()

    for entry_cfg in val_entries:
        entry_cfg = dict(entry_cfg)
        if data_cfg.get("default"):
            update_single_dataset_config(entry_cfg, data_cfg["default"])
        label = _category_label(entry_cfg)
        if max_samples is not None:
            entry_cfg["size"] = min(int(entry_cfg["size"]), int(max_samples))

        print(
            f"\n▶ Split {label!r}: building dataset (size={entry_cfg.get('size')})",
            flush=True,
        )
        raw = load_response_dataset(entry_cfg)
        proc = AllTaskProcessedDataset(
            dataset=raw.dataset,
            tokenizer=tokenizer,
            default_task_data_spec=raw.task_spec,
            task_data_processors={raw.task_name: (raw.task_spec, raw.processor)},
            max_seq_length=data_cfg["max_input_seq_length"],
        )
        loader = DataLoader(
            proc,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=rl_collate_fn,
        )

        rewards: list[float] = []
        per_env_rewards: dict[str, list[float]] = {}
        per_sample_log: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        for batch in loader:
            if num_tests_per_prompt > 1:
                batch = batch.repeat_interleave(num_tests_per_prompt)
            batch_out, _ = run_multi_turn_rollout(
                vllm_gen,
                batch,
                tokenizer,
                task_to_env,
                max_seq_len=max_seq_len,
                max_rollout_turns=1,
                greedy=greedy,
            )
            batch_rewards = batch_out["total_reward"].tolist()
            rewards.extend(batch_rewards)
            for info, r in zip(batch_out["extra_env_info"], batch_rewards):
                env_name = (
                    info.get("dataset_name") if isinstance(info, dict) else None
                )
                if env_name:
                    per_env_rewards.setdefault(env_name, []).append(r)

            if save_path:
                for ml, info, r in zip(
                    batch_out["message_log"],
                    batch_out["extra_env_info"],
                    batch_rewards,
                ):
                    per_sample_log.append(
                        {
                            "messages": [
                                {"role": m["role"], "content": m["content"]}
                                for m in ml
                            ],
                            "dataset_name": (
                                info.get("dataset_name")
                                if isinstance(info, dict)
                                else None
                            ),
                            "category": (
                                info.get("category")
                                if isinstance(info, dict)
                                else None
                            ),
                            "reward": r,
                        }
                    )

        n = len(rewards)
        acc = sum(rewards) / n if n else 0.0
        per_env = {
            name: sum(rs) / len(rs) for name, rs in per_env_rewards.items()
        }
        elapsed = time.perf_counter() - t0
        print(
            f"  ✓ {label}: accuracy={acc:.4f}  n={n}  time={elapsed:.1f}s",
            flush=True,
        )
        for env_name, env_acc in sorted(per_env.items()):
            print(
                f"      · {env_name:<22} {env_acc:.4f}  "
                f"(n={len(per_env_rewards[env_name])})"
            )
        summary[label] = {
            "accuracy": acc,
            "n": n,
            "per_env": per_env,
            "elapsed_s": elapsed,
        }
        overall_rewards.extend(rewards)

        if save_path:
            split_path = Path(save_path) / f"split_{label}.jsonl"
            with open(split_path, "w") as f:
                for row in per_sample_log:
                    f.write(json.dumps(row) + "\n")

    overall_acc = (
        sum(overall_rewards) / len(overall_rewards) if overall_rewards else 0.0
    )
    summary["_overall"] = {
        "accuracy": overall_acc,
        "n": len(overall_rewards),
        "elapsed_s": time.perf_counter() - t_total,
    }

    print("\n" + "=" * 60)
    print(f" RG Eval — checkpoint={ckpt}")
    print(
        f" decoding: greedy={greedy} temperature={gen_cfg['temperature']} "
        f"top_p={gen_cfg['top_p']} num_tests_per_prompt={num_tests_per_prompt}"
    )
    print("=" * 60)
    for label, m in summary.items():
        if label == "_overall":
            continue
        print(f"  {label:<14}  acc={m['accuracy']:.4f}  n={m['n']}")
    print("-" * 60)
    print(
        f"  {'OVERALL':<14}  acc={overall_acc:.4f}  n={len(overall_rewards)}"
    )
    print("=" * 60)

    if save_path:
        out = {
            "checkpoint": ckpt,
            "config_path": args.config,
            "decoding": {
                "greedy": greedy,
                "temperature": gen_cfg["temperature"],
                "top_p": gen_cfg["top_p"],
                "num_tests_per_prompt": num_tests_per_prompt,
                "k_value": k_value,
            },
            "summary": summary,
        }
        out_path = Path(save_path) / "summary.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote: {out_path}")

    ray.get(rg_env.shutdown.remote())
    vllm_gen.shutdown()


if __name__ == "__main__":
    main()
