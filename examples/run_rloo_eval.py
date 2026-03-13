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

import argparse
import os
import pprint
from typing import cast

import ray
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.rloo import MasterConfig, refit_policy_generation, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.llm_message_utils import get_keys_from_message_log
from nemo_rl.data.datasets import update_single_dataset_config
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.evals.eval import eval_cons_k, eval_pass_k
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate an RLOO checkpoint with the same loading path as training"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint root dir. Defaults to config.checkpointing.checkpoint_dir",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Evaluate the best checkpoint per checkpointing.metric_name",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Evaluate the latest checkpoint (default)",
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Evaluate the base (non-finetuned) model without loading any checkpoint",
    )
    parser.add_argument(
        "--math-task-names",
        type=str,
        default=None,
        help=(
            "Comma-separated list of task names to request extracted answers from the math "
            "environment (for cons@k). If omitted, inferred from data configs with env_name=math."
        ),
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _infer_step_from_path(checkpoint_path: str) -> int:
    base = os.path.basename(os.path.normpath(checkpoint_path))
    if base.startswith("step_"):
        try:
            return int(base.split("_", 1)[1])
        except ValueError:
            return 0
    return 0


def _extract_answers_for_batch(
    val_batch, task_to_env, math_task_names: set[str]
) -> list[str | None]:
    """Run env.step with extracted answers where supported (math), and return answers in batch order."""
    to_env = [
        get_keys_from_message_log(val_batch["message_log"][i], ["role", "content"])
        for i in range(len(val_batch["message_log"]))
    ]
    task_names = val_batch["task_name"]

    task_groups: dict[str, list[tuple[int, list[dict[str, str]]]]] = {}
    for i, task_name in enumerate(task_names):
        task_groups.setdefault(task_name, []).append((i, to_env[i]))

    futures = []
    future_to_indices = {}
    for task_name, group in task_groups.items():
        indices = [idx for idx, _ in group]
        messages = [msg for _, msg in group]
        env_info = [val_batch["extra_env_info"][i] for i in indices]

        env = task_to_env[task_name]
        if task_name in math_task_names:
            future = env.step.remote(messages, env_info, True)
        else:
            future = env.step.remote(messages, env_info)
        futures.append(future)
        future_to_indices[future] = indices

    results = ray.get(futures)
    all_answers = []
    all_indices_order = []

    for future, result in zip(futures, results):
        indices = future_to_indices[future]
        answers = result.answers if hasattr(result, "answers") else None
        if answers is None:
            answers = [None] * len(indices)
        for i, idx in enumerate(indices):
            all_indices_order.append(idx)
            all_answers.append(answers[i])

    sorted_indices = sorted(
        range(len(all_indices_order)), key=lambda k: all_indices_order[k]
    )
    return [all_answers[i] for i in sorted_indices]


def _evaluate_pass_and_majority(
    policy_generation,
    val_dataloader,
    tokenizer,
    val_task_to_env,
    master_config: MasterConfig,
    math_task_names: set[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute pass@k and maj@k metrics on the validation set."""
    num_tests_per_prompt = master_config["grpo"]["num_generations_per_prompt"]
    required_k_values = [1, 2, 4, 8, 16]
    if num_tests_per_prompt < max(required_k_values):
        raise ValueError(
            "num_generations_per_prompt must be >= 16 to compute pass@k/maj@k for "
            f"{required_k_values}. Current value: {num_tests_per_prompt}"
        )
    k_values = required_k_values

    pass_sums = {k: 0.0 for k in k_values}
    maj_sums = {k: 0.0 for k in k_values}
    total_rewards = []
    total_lengths = []

    for batch in val_dataloader:
        repeated_batch = batch.repeat_interleave(num_tests_per_prompt)
        val_batch, gen_metrics = run_multi_turn_rollout(
            policy_generation,
            repeated_batch,
            tokenizer,
            val_task_to_env,
            max_seq_len=master_config["policy"]["max_total_sequence_length"],
            max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
            greedy=False,
        )

        rewards = val_batch["total_reward"].float()
        total_rewards.extend(rewards.tolist())
        total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

        extracted_answers = _extract_answers_for_batch(
            val_batch, val_task_to_env, math_task_names
        )

        for k in k_values:
            pass_sums[k] += eval_pass_k(rewards, num_tests_per_prompt, k)
            maj_sums[k] += eval_cons_k(
                rewards, num_tests_per_prompt, k, extracted_answers
            )

    dataset_size = len(val_dataloader.dataset)
    pass_metrics = {f"pass@{k}": pass_sums[k] / dataset_size for k in k_values}
    maj_metrics = {f"maj@{k}": maj_sums[k] / dataset_size for k in k_values}

    accuracy = (
        torch.tensor(total_rewards, dtype=torch.float32).mean().item()
        if total_rewards
        else 0.0
    )
    avg_length = (
        sum(total_lengths) / len(total_lengths) if total_lengths else 0.0
    )

    summary = {"accuracy": accuracy, "avg_length": avg_length}
    summary.update(pass_metrics)
    summary.update(maj_metrics)

    return summary, {}


def _infer_math_task_names(config: MasterConfig) -> set[str]:
    data_config = config.get("data", {})
    default_cfg = data_config.get("default") if isinstance(data_config, dict) else None
    task_names: set[str] = set()

    for key in ("train", "validation"):
        if key not in data_config or data_config[key] is None:
            continue
        cfg_list = data_config[key]
        if isinstance(cfg_list, dict):
            cfg_list = [cfg_list]
        for cfg in cfg_list:
            if not isinstance(cfg, dict):
                continue
            merged_cfg = dict(cfg)
            if isinstance(default_cfg, dict):
                update_single_dataset_config(merged_cfg, default_cfg)
            if merged_cfg.get("env_name") == "math":
                dataset_name = merged_cfg.get("dataset_name")
                if dataset_name:
                    task_names.add(str(dataset_name))
    # Keep backward compatibility with the hardcoded name used previously.
    task_names.add("math")
    return task_names


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "dist_sharpening",
            "rloo_math_lighteval.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = cast(MasterConfig, OmegaConf.to_container(config, resolve=True))
    print("Applied CLI overrides")

    if args.checkpoint_dir is not None:
        config["checkpointing"]["checkpoint_dir"] = args.checkpoint_dir

    checkpointer = CheckpointManager(config["checkpointing"])
    if sum([bool(args.best), bool(args.latest), bool(args.base)]) > 1:
        raise ValueError("Use only one of --best, --latest, or --base.")

    if args.base:
        checkpoint_path = None
        label = "base"
        print("📌 Evaluating base model (no checkpoint).")
    else:
        if args.best:
            checkpoint_path = checkpointer.get_best_checkpoint_path()
        else:
            checkpoint_path = checkpointer.get_latest_checkpoint_path()
        if checkpoint_path is None:
            raise FileNotFoundError(
                "No checkpoint found. Ensure checkpointing.checkpoint_dir contains step_* checkpoints."
            )
        label = "best" if args.best else "latest"
        print(f"📌 Using checkpoint ({label}): {checkpoint_path}")

    # Print config for reproducibility
    print("Final config:")
    pprint.pprint(config)

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for RLOO evaluation"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    dataset, val_dataset, _task_to_env, val_task_to_env = setup_response_data(
        tokenizer, config["data"], config["env"]
    )

    (
        policy,
        policy_generation,
        _cluster,
        _dataloader,
        val_dataloader,
        _loss_fn,
        logger,
        _checkpointer,
        grpo_state,
        master_config,
    ) = setup(
        config,
        tokenizer,
        dataset,
        val_dataset,
        checkpoint_path_override=checkpoint_path,
    )

    if val_dataloader is None:
        assert val_dataset is not None, (
            "Validation dataset is required for evaluation. "
            "Please ensure your data config specifies a validation split."
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=master_config["grpo"]["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
            num_workers=master_config["data"]["num_workers"],
        )

    need_refit = True
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        need_refit = False

    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    if need_refit:
        refit_policy_generation(policy, policy_generation, colocated_inference)
    else:
        policy_generation.prepare_for_generation()

    eval_step = grpo_state.get("total_steps", 0) or _infer_step_from_path(
        checkpoint_path
    )
    if args.math_task_names:
        math_task_names = {
            name.strip()
            for name in args.math_task_names.split(",")
            if name.strip()
        }
    else:
        math_task_names = _infer_math_task_names(config)
    print(f"Math task names for cons@k: {sorted(math_task_names)}")

    val_metrics, validation_timings = _evaluate_pass_and_majority(
        policy_generation,
        val_dataloader,
        tokenizer,
        val_task_to_env,
        master_config=master_config,
        math_task_names=math_task_names,
    )
    policy_generation.finish_generation()

    logger.log_metrics(val_metrics, eval_step, prefix="validation")
    logger.log_metrics(validation_timings, eval_step, prefix="timing/validation")


if __name__ == "__main__":
    main()
