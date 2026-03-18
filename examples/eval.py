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
import json
import os
import pprint
import tempfile
from pathlib import Path
from typing import Any, Optional, cast

import torch
from datasets import concatenate_datasets
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.rloo import MasterConfig, refit_policy_generation, setup, validate
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import (
    AllTaskProcessedDataset,
    load_eval_dataset,
    update_single_dataset_config,
)
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.evals.eval import eval_cons_k, eval_pass_k
from nemo_rl.environments.utils import create_env
from nemo_rl.experience.rollouts import calculate_rewards, generate_responses
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.logger import print_message_log_samples
from nemo_rl.utils.timer import Timer


DEFAULT_METRICS_NUM_SAMPLES = 16
DEFAULT_METRICS_K_VALUES = [1, 2, 4, 8, 16]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate an RLOO checkpoint using the training data pipeline (no training)"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint root dir. Defaults to config.checkpointing.checkpoint_dir",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to a specific step_* checkpoint directory to evaluate",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Evaluate the best checkpoint per checkpointing.metric_name",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="Evaluate the last checkpoint",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Alias for --last",
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Evaluate the base (non-finetuned) model without loading any checkpoint",
    )
    parser.add_argument(
        "--metrics-json-path",
        type=str,
        default=None,
        help="Optional path to save pass@k and maj@k metrics as JSON",
    )
    parser.add_argument(
        "--metrics-num-samples",
        type=int,
        default=None,
        help="Number of generations per prompt for pass@k / maj@k evaluation",
    )
    parser.add_argument(
        "--metrics-k-values",
        type=str,
        default=None,
        help="Comma-separated k values for pass@k / maj@k JSON output",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _infer_step_from_path(checkpoint_path: str | None) -> int:
    if not checkpoint_path:
        return 0
    base = os.path.basename(os.path.normpath(checkpoint_path))
    if base.startswith("step_"):
        try:
            return int(base.split("_", 1)[1])
        except ValueError:
            return 0
    return 0


def _get_eval_task_name(dataset_name: str) -> str:
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower in {"math-lighteval", "math_lighteval"}:
        return "MATH-lighteval"
    return dataset_name


def _setup_eval_validation_only(
    tokenizer,
    data_config: dict,
    env_configs: dict,
) -> tuple[AllTaskProcessedDataset, dict[str, object]]:
    if "validation" not in data_config or data_config["validation"] is None:
        raise ValueError(
            "Validation dataset is required for evaluation. "
            "Please ensure your data config specifies a validation split."
        )

    if isinstance(data_config["validation"], dict):
        val_configs = [data_config["validation"]]
    else:
        val_configs = data_config["validation"]

    default_cfg = data_config.get("default") if isinstance(data_config, dict) else None

    val_task_data_processors = {}
    val_task_to_env = {}
    val_data_list = []

    env_name_list = set()
    for cfg in val_configs:
        cfg = dict(cfg)
        if isinstance(default_cfg, dict):
            update_single_dataset_config(cfg, default_cfg)

        env_name = cfg.get("env_name")
        if env_name is None:
            raise ValueError(
                f"env_name is required for validation dataset config: {cfg}"
            )
        env_name_list.add(env_name)

    envs = {}
    for env_name in sorted(env_name_list):
        envs[env_name] = create_env(
            env_name=env_name,
            env_config=env_configs[env_name],
        )

    for cfg in val_configs:
        cfg = dict(cfg)
        if isinstance(default_cfg, dict):
            update_single_dataset_config(cfg, default_cfg)
        val_data = load_eval_dataset(cfg)
        task_name = _get_eval_task_name(cfg["dataset_name"])
        if "task_name" in val_data.rekeyed_ds.column_names:
            existing_task_names = set(val_data.rekeyed_ds["task_name"])
            if existing_task_names == {task_name}:
                val_dataset = val_data.rekeyed_ds
            else:
                val_dataset = val_data.rekeyed_ds.remove_columns(
                    ["task_name"]
                ).add_column("task_name", [task_name] * len(val_data.rekeyed_ds))
        else:
            val_dataset = val_data.rekeyed_ds.add_column(
                "task_name", [task_name] * len(val_data.rekeyed_ds)
            )
        val_data_list.append(val_dataset)
        val_task_data_processors[task_name] = (
            val_data.task_spec,
            val_data.processor,
        )

        env_name = cfg.get("env_name")
        val_task_to_env[task_name] = envs[env_name]

    merged_val_data = concatenate_datasets(val_data_list)
    val_dataset = AllTaskProcessedDataset(
        merged_val_data,
        tokenizer,
        None,
        val_task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    print(f"  ✓ Validation dataset loaded with {len(val_dataset)} samples.")
    return val_dataset, val_task_to_env


def _resolve_checkpoint_path(
    config: MasterConfig, args: argparse.Namespace
) -> tuple[Optional[str], Optional[str]]:
    if args.checkpoint_dir is not None:
        config["checkpointing"]["checkpoint_dir"] = args.checkpoint_dir

    checkpoint_selectors = [bool(args.best), bool(args.last or args.latest), bool(args.base)]
    if sum(checkpoint_selectors) > 1:
        raise ValueError("Use only one of --best, --last/--latest, or --base.")
    if args.checkpoint_path and any(checkpoint_selectors):
        raise ValueError(
            "Use only one of --checkpoint-path and --best/--last/--latest/--base."
        )

    if args.base:
        empty_ckpt_dir = tempfile.mkdtemp(prefix="nrl_eval_base_", dir="/tmp")
        config["checkpointing"]["checkpoint_dir"] = empty_ckpt_dir
        return None, "base"

    if args.checkpoint_path is not None:
        checkpoint_path = os.path.abspath(args.checkpoint_path)
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint path does not exist or is not a directory: {checkpoint_path}"
            )
        return checkpoint_path, "path"

    checkpointer = CheckpointManager(config["checkpointing"])
    if args.best:
        return checkpointer.get_best_checkpoint_path(), "best"

    return checkpointer.get_latest_checkpoint_path(), "last"


def _parse_k_values(k_values_arg: str, num_samples: int) -> list[int]:
    k_values = sorted(
        {
            int(raw.strip())
            for raw in k_values_arg.split(",")
            if raw.strip()
        }
    )
    if not k_values:
        raise ValueError("At least one metrics k value must be provided.")
    if k_values[0] <= 0:
        raise ValueError("All metrics k values must be positive integers.")
    if k_values[-1] > num_samples:
        raise ValueError(
            f"metrics k values must be <= metrics_num_samples ({num_samples})."
        )
    return k_values


def _build_power_of_two_k_values(max_k: int, num_samples: int) -> list[int]:
    if max_k <= 0:
        raise ValueError("metrics max_k must be a positive integer.")
    if max_k > num_samples:
        raise ValueError(f"metrics max_k must be <= metrics_num_samples ({num_samples}).")

    k_values = []
    current_k = 1
    while current_k < max_k:
        k_values.append(current_k)
        current_k *= 2

    if not k_values or k_values[-1] != max_k:
        k_values.append(max_k)

    return k_values


def _resolve_metrics_config(
    config: MasterConfig,
    args: argparse.Namespace,
) -> tuple[Optional[str], Optional[int], Optional[list[int]], Optional[int]]:
    metrics_config = config.get("eval_metrics", {})
    metrics_output_path = args.metrics_json_path

    should_compute_metrics = metrics_output_path is not None or bool(
        metrics_config.get("enabled", False)
    )
    if metrics_output_path is None and metrics_config.get("output_path") is not None:
        metrics_output_path = metrics_config["output_path"]
        should_compute_metrics = True

    if not should_compute_metrics:
        return None, None, None, None
    if metrics_output_path is None:
        raise ValueError(
            "pass@k / maj@k evaluation requires an output path. "
            "Set --metrics-json-path or eval_metrics.output_path."
        )

    num_samples = (
        args.metrics_num_samples
        if args.metrics_num_samples is not None
        else metrics_config.get("num_samples", DEFAULT_METRICS_NUM_SAMPLES)
    )

    if args.metrics_k_values is not None:
        k_values = _parse_k_values(args.metrics_k_values, num_samples)
    elif "max_k" in metrics_config:
        k_values = _build_power_of_two_k_values(metrics_config["max_k"], num_samples)
    elif "k_values" in metrics_config:
        k_values = sorted(set(int(k) for k in metrics_config["k_values"]))
        if not k_values:
            raise ValueError("eval_metrics.k_values must contain at least one value.")
        if k_values[0] <= 0:
            raise ValueError("All eval_metrics.k_values must be positive integers.")
        if k_values[-1] > num_samples:
            raise ValueError(
                f"eval_metrics.k_values must be <= metrics_num_samples ({num_samples})."
            )
    else:
        k_values = [k for k in DEFAULT_METRICS_K_VALUES if k <= num_samples]
        if not k_values or k_values[-1] != num_samples:
            k_values = _build_power_of_two_k_values(num_samples, num_samples)

    majk_num_samples = metrics_config.get("majk_num_samples")
    if majk_num_samples is not None:
        majk_num_samples = int(majk_num_samples)
        if majk_num_samples <= 0:
            raise ValueError("eval_metrics.majk_num_samples must be a positive integer.")

    return metrics_output_path, num_samples, k_values, majk_num_samples


def _compute_pass_maj_metrics(
    policy_generation,
    val_dataset: AllTaskProcessedDataset,
    tokenizer,
    val_task_to_env: dict[str, object],
    master_config: MasterConfig,
    num_samples_per_prompt: int,
    k_values: list[int],
    majk_num_samples: Optional[int],
) -> dict[str, Any]:
    if master_config["grpo"]["max_rollout_turns"] != 1:
        raise ValueError(
            "pass@k / maj@k JSON export is currently supported only for single-turn evaluation."
        )

    pass_scores = {k: 0.0 for k in k_values}
    maj_scores = {k: 0.0 for k in k_values}
    num_prompts = 0
    metrics_dataloader = StatefulDataLoader(
        val_dataset,
        batch_size=master_config["grpo"]["val_batch_size"],
        shuffle=False,
        collate_fn=rl_collate_fn,
        num_workers=master_config["data"]["num_workers"],
    )

    for val_batch in metrics_dataloader:
        expanded_batch = val_batch.repeat_interleave(num_samples_per_prompt)
        flat_messages, input_lengths = batched_message_log_to_flat_message(
            expanded_batch["message_log"],
            pad_value_dict={"token_ids": tokenizer.pad_token_id},
        )

        generation_input_data = BatchedDataDict[GenerationDatumSpec](
            {
                "input_ids": flat_messages["token_ids"],
                "input_lengths": input_lengths,
                "stop_strings": expanded_batch.get(
                    "stop_strings", [None] * len(expanded_batch["message_log"])
                ),
            }
        )
        multimodal_data = flat_messages.get_multimodal_dict(as_tensors=False)
        generation_input_data.update(multimodal_data)

        if "vllm_content" in expanded_batch:
            generation_input_data["vllm_content"] = expanded_batch["vllm_content"]
        if "vllm_images" in expanded_batch:
            generation_input_data["vllm_images"] = expanded_batch["vllm_images"]
        if "vllm_videos" in expanded_batch:
            generation_input_data["vllm_videos"] = expanded_batch["vllm_videos"]

        generated_batch, _, _ = generate_responses(
            policy_generation,
            generation_input_data,
            expanded_batch,
            tokenizer,
            input_lengths=input_lengths,
            greedy=False,
        )
        env_output = calculate_rewards(
            generated_batch,
            val_task_to_env,
            return_extracted_answer=True,
        )  # type: ignore[arg-type]
        rewards = env_output.rewards.float()
        answers = env_output.answers
        assert answers is not None

        for k in k_values:
            pass_scores[k] += eval_pass_k(rewards, num_samples_per_prompt, k)
            maj_scores[k] += eval_cons_k(
                rewards,
                num_samples_per_prompt,
                k,
                answers,
                mc_num_samples=majk_num_samples,
            )

        num_prompts += len(val_batch["message_log"])

    return {
        "num_prompts": num_prompts,
        "num_samples_per_prompt": num_samples_per_prompt,
        "k_values": k_values,
        "pass@k": {str(k): pass_scores[k] / num_prompts for k in k_values},
        "maj@k": {str(k): maj_scores[k] / num_prompts for k in k_values},
    }


def _compute_validation_and_pass_maj_metrics(
    policy_generation,
    val_dataset: AllTaskProcessedDataset,
    tokenizer,
    val_task_to_env: dict[str, object],
    master_config: MasterConfig,
    num_samples_per_prompt: int,
    k_values: list[int],
    majk_num_samples: Optional[int],
    step: int,
    logger=None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if master_config["grpo"]["max_rollout_turns"] != 1:
        raise ValueError(
            "pass@k / maj@k JSON export is currently supported only for single-turn evaluation."
        )

    pass_scores = {k: 0.0 for k in k_values}
    maj_scores = {k: 0.0 for k in k_values}
    num_prompts = 0
    total_reward_sum = 0.0
    total_reward_count = 0
    total_generated_tokens = 0.0
    total_samples = 0
    all_message_logs = []
    all_rewards_for_logs = []
    all_prompts = []
    all_generations = []

    metrics_dataloader = StatefulDataLoader(
        val_dataset,
        batch_size=master_config["grpo"]["val_batch_size"],
        shuffle=False,
        collate_fn=rl_collate_fn,
        num_workers=master_config["data"]["num_workers"],
    )

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...", flush=True)

        max_batches = (
            master_config["grpo"]["max_val_samples"]
            // master_config["grpo"]["val_batch_size"]
        )
        for batch_idx, val_batch in enumerate(metrics_dataloader):
            if batch_idx >= max_batches:
                break

            expanded_batch = val_batch.repeat_interleave(num_samples_per_prompt)
            flat_messages, input_lengths = batched_message_log_to_flat_message(
                expanded_batch["message_log"],
                pad_value_dict={"token_ids": tokenizer.pad_token_id},
            )

            generation_input_data = BatchedDataDict[GenerationDatumSpec](
                {
                    "input_ids": flat_messages["token_ids"],
                    "input_lengths": input_lengths,
                    "stop_strings": expanded_batch.get(
                        "stop_strings", [None] * len(expanded_batch["message_log"])
                    ),
                }
            )
            multimodal_data = flat_messages.get_multimodal_dict(as_tensors=False)
            generation_input_data.update(multimodal_data)

            if "vllm_content" in expanded_batch:
                generation_input_data["vllm_content"] = expanded_batch["vllm_content"]
            if "vllm_images" in expanded_batch:
                generation_input_data["vllm_images"] = expanded_batch["vllm_images"]
            if "vllm_videos" in expanded_batch:
                generation_input_data["vllm_videos"] = expanded_batch["vllm_videos"]

            generated_batch, _, gen_metrics = generate_responses(
                policy_generation,
                generation_input_data,
                expanded_batch,
                tokenizer,
                input_lengths=input_lengths,
                greedy=False,
            )
            env_output = calculate_rewards(
                generated_batch,
                val_task_to_env,
                return_extracted_answer=True,
            )  # type: ignore[arg-type]
            rewards = env_output.rewards.float()
            answers = env_output.answers
            assert answers is not None

            total_reward_sum += rewards.sum().item()
            total_reward_count += rewards.numel()
            total_generated_tokens += float(gen_metrics["total_generated_tokens"])
            total_samples += rewards.numel()

            for k in k_values:
                pass_scores[k] += eval_pass_k(rewards, num_samples_per_prompt, k)
                maj_scores[k] += eval_cons_k(
                    rewards,
                    num_samples_per_prompt,
                    k,
                    answers,
                    mc_num_samples=majk_num_samples,
                )

            num_prompts += len(val_batch["message_log"])

            if num_samples_per_prompt > 1:
                log_indices = range(
                    0, len(generated_batch["message_log"]), num_samples_per_prompt
                )
            else:
                log_indices = range(len(generated_batch["message_log"]))

            to_env = []
            for i in log_indices:
                message_log = generated_batch["message_log"][i]
                to_env.append(
                    get_keys_from_message_log(message_log, ["role", "content"])
                )

                if message_log and message_log[-1].get("role") == "assistant":
                    prompt_log = message_log[:-1]
                    generation = message_log[-1].get("content")
                else:
                    prompt_log = message_log
                    generation = None

                all_prompts.append(
                    get_keys_from_message_log(prompt_log, ["role", "content"])
                )
                all_generations.append(generation)

            all_message_logs.extend(to_env)
            all_rewards_for_logs.extend(rewards[list(log_indices)].tolist())

    accuracy = total_reward_sum / total_reward_count if total_reward_count > 0 else 0.0
    avg_length = (
        total_generated_tokens / total_samples if total_samples > 0 else 0.0
    )
    val_metrics = {"accuracy": accuracy, "avg_length": avg_length}

    try:
        print_message_log_samples(
            all_message_logs,
            all_rewards_for_logs,
            num_samples=min(
                master_config["logger"]["num_val_samples_to_print"],
                len(all_message_logs),
            ),
            step=step,
        )
    except Exception as e:
        print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
        print("  ⚠️ Continuing validation without displaying samples...", flush=True)

    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {total_reward_count}", flush=True)

    print("\n  ⏱️  Validation Timing:")
    print(f"    • Total validation time: {validation_time:.2f}s", flush=True)

    if logger is not None:
        val_log_data = {
            "content": all_message_logs,
            "prompt": all_prompts,
            "generation": all_generations,
            "rewards": all_rewards_for_logs,
        }
        logger.log_batched_dict_as_jsonl(val_log_data, f"val_data_step{step}.jsonl")

    metrics = {
        "num_prompts": num_prompts,
        "num_samples_per_prompt": num_samples_per_prompt,
        "k_values": k_values,
        "pass@k": {str(k): pass_scores[k] / num_prompts for k in k_values},
        "maj@k": {str(k): maj_scores[k] / num_prompts for k in k_values},
    }

    return val_metrics, timing_metrics, metrics


def _save_metrics_json(
    metrics_json_path: str,
    metrics: dict[str, Any],
    checkpoint_mode: Optional[str],
    checkpoint_path: Optional[str],
    eval_step: int,
    config: MasterConfig,
) -> None:
    output_path = Path(metrics_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_mode": checkpoint_mode,
        "checkpoint_path": checkpoint_path,
        "eval_step": eval_step,
        "model_name": config["policy"]["model_name"],
        "validation_dataset": config["data"]["validation"],
        **metrics,
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"📄 Saved pass@k / maj@k metrics to: {output_path}")


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "dist_sharpening",
            "eval_math_lighteval.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = cast(MasterConfig, OmegaConf.to_container(config, resolve=True))
    print("Applied CLI overrides")

    if "logger" in config:
        config["logger"]["wandb_enabled"] = False

    (
        metrics_output_path,
        metrics_num_samples,
        metrics_k_values,
        majk_num_samples,
    ) = _resolve_metrics_config(config, args)

    checkpoint_path, checkpoint_mode = _resolve_checkpoint_path(config, args)
    if checkpoint_mode == "base":
        print("📌 Evaluating base model (no checkpoint).")
    else:
        if checkpoint_path is None:
            raise FileNotFoundError(
                "No checkpoint found. Ensure checkpointing.checkpoint_dir contains step_* checkpoints, "
                "or use --base to evaluate the base model."
            )
        print(f"📌 Using checkpoint ({checkpoint_mode}): {checkpoint_path}")

    # Print config for reproducibility
    print("Final config:")
    pprint.pprint(config)

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for evaluation"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    val_dataset, val_task_to_env = _setup_eval_validation_only(
        tokenizer, config["data"], config["env"]
    )
    # Use validation dataset as a stand-in training dataset to avoid loading train splits.
    dataset = val_dataset

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
    if (
        metrics_output_path is not None
        and metrics_num_samples is not None
        and metrics_k_values is not None
    ):
        val_metrics, validation_timings, metrics = (
            _compute_validation_and_pass_maj_metrics(
                policy_generation,
                val_dataset,
                tokenizer,
                val_task_to_env,
                master_config,
                metrics_num_samples,
                metrics_k_values,
                majk_num_samples,
                eval_step,
                logger=logger,
            )
        )
        _save_metrics_json(
            metrics_output_path,
            metrics,
            checkpoint_mode,
            checkpoint_path,
            eval_step,
            master_config,
        )
    else:
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            eval_step,
            master_config,
            logger=logger,
        )

    policy_generation.finish_generation()

    logger.log_metrics(val_metrics, eval_step, prefix="validation")
    logger.log_metrics(validation_timings, eval_step, prefix="timing/validation")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
