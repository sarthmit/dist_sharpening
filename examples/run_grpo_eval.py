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
from typing import Optional, cast

from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.grpo import MasterConfig, refit_policy_generation, setup, validate
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate a GRPO checkpoint with the same validation path used in training"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to a specific checkpoint directory (e.g., .../step_1000)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint root dir. Defaults to config.checkpointing.checkpoint_dir",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Checkpoint step number (uses --checkpoint-dir)",
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


def _resolve_checkpoint_path(
    config: MasterConfig, args: argparse.Namespace
) -> Optional[str]:
    if args.checkpoint_path is not None:
        return os.path.abspath(args.checkpoint_path)

    checkpoint_dir = args.checkpoint_dir or config["checkpointing"]["checkpoint_dir"]
    config["checkpointing"]["checkpoint_dir"] = checkpoint_dir
    checkpointer = CheckpointManager(config["checkpointing"])

    if args.checkpoint_step is not None:
        return os.path.abspath(os.path.join(checkpoint_dir, f"step_{args.checkpoint_step}"))

    if args.best:
        return checkpointer.get_best_checkpoint_path()

    return checkpointer.get_latest_checkpoint_path()


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_math_1B.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = cast(MasterConfig, OmegaConf.to_container(config, resolve=True))
    print("Applied CLI overrides")

    checkpoint_path = _resolve_checkpoint_path(config, args)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Provide --checkpoint-path, --checkpoint-dir, or ensure "
            "checkpointing.checkpoint_dir contains step_* checkpoints."
        )
    print(f"📌 Using checkpoint: {checkpoint_path}")

    # Print config for reproducibility
    print("Final config:")
    pprint.pprint(config)

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO evaluation"
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

    NEED_REFIT = True
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        NEED_REFIT = False

    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    if NEED_REFIT:
        refit_policy_generation(policy, policy_generation, colocated_inference)
    else:
        policy_generation.prepare_for_generation()

    eval_step = grpo_state.get("total_steps", 0) or _infer_step_from_path(
        checkpoint_path
    )
    val_metrics, validation_timings = validate(
        policy_generation,
        val_dataloader,
        tokenizer,
        val_task_to_env,
        step=eval_step,
        master_config=master_config,
        logger=logger,
    )
    policy_generation.finish_generation()

    logger.log_metrics(val_metrics, eval_step, prefix="validation")
    logger.log_metrics(validation_timings, eval_step, prefix="timing/validation")


if __name__ == "__main__":
    main()
