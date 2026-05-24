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

"""Pre-materialize the reasoning_gym train + validation datasets into the
on-disk cache (``data.default.cache_dir``) without needing a GPU.

Dataset materialization is pure CPU work (reasoning_gym generators + prompt
tokenization). This script runs the same config-load + ``setup_response_data``
path as ``run_grpo.py`` with ``env_configs=None`` (no Ray, no environment
actors), so the cache fingerprints match exactly what a training run looks for.
After this finishes, a GPU run with the same config reloads from disk and skips
dataset generation entirely.

Usage:
    uv run python examples/reasoning_gym/prebuild_cache.py \
        [--config examples/reasoning_gym/grpo_reasoning_gym.yaml]

Run from the repo root so the relative ``prompt_file`` path resolves to the
same file the training run sees (the cache fingerprint hashes its contents).
"""

import argparse

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.data.utils import setup_response_data
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="examples/reasoning_gym/grpo_reasoning_gym.yaml",
        help="Path to the GRPO reasoning_gym config.",
    )
    args = parser.parse_args()

    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(args.config), resolve=True)

    tokenizer_name = config["policy"]["tokenizer"]["name"]
    print(f"[prebuild_cache] config={args.config} tokenizer={tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # env_configs=None -> build the train + validation datasets only; no Ray
    # environment actors are created. Each ReasoningGymDataset is constructed
    # with the configured cache_dir, so it materializes and saves to disk.
    setup_response_data(tokenizer, config["data"], None)
    print("[prebuild_cache] done — train + validation datasets cached.")


if __name__ == "__main__":
    main()
