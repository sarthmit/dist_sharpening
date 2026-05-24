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

"""Consolidate NeMo-RL's sharded async-HF checkpoint into a standard HF dir.

Input layout (written by the DTensor V2 worker's HF-format async save):
    <input>/
        .hf_metadata/
            config.json
            generation_config.json
            fqn_to_file_index_mapping.json
        shard-00001-model-00001-of-00001.safetensors
        shard-00002-model-00001-of-00001.safetensors
        ...

Output layout (loadable by AutoModel / vLLM):
    <output>/
        config.json
        generation_config.json
        model-00001-of-NNNNN.safetensors
        model.safetensors.index.json
        (+ tokenizer files copied from --tokenizer-dir if provided)

Re-uses Automodel's offline consolidation kernel — added to sys.path rather
than installed, since the full Automodel package pulls heavy deps we don't
need for a single-process consolidation.
"""

import argparse
import json
import os
import shutil
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUTOMODEL_SRC = os.path.join(REPO_ROOT, "3rdparty", "Automodel-workspace", "Automodel")
if AUTOMODEL_SRC not in sys.path:
    sys.path.insert(0, AUTOMODEL_SRC)

from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (  # noqa: E402
    consolidate_safetensors_files,
)


_TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "spiece.model",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--input-dir",
        required=True,
        help="Sharded model dir (contains .hf_metadata/ and shard-*.safetensors).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Destination dir for the consolidated HF checkpoint.",
    )
    p.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Optional dir to copy tokenizer files from (e.g. policy/tokenizer/).",
    )
    p.add_argument("--num-threads", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    hf_meta = os.path.join(args.input_dir, ".hf_metadata")
    mapping_path = os.path.join(hf_meta, "fqn_to_file_index_mapping.json")
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(
            f"Expected {mapping_path}; --input-dir must be the sharded model "
            "dir written by the DTensor V2 worker."
        )
    with open(mapping_path) as f:
        fqn_to_index = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    print(
        f"Consolidating {args.input_dir} -> {args.output_dir} "
        f"({len(fqn_to_index)} tensors, {len(set(fqn_to_index.values()))} output files)"
    )
    consolidate_safetensors_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fqn_to_index_mapping=fqn_to_index,
        num_threads=args.num_threads,
    )

    for name in os.listdir(hf_meta):
        if name == "fqn_to_file_index_mapping.json":
            continue
        shutil.copy2(os.path.join(hf_meta, name), os.path.join(args.output_dir, name))

    if args.tokenizer_dir:
        if not os.path.isdir(args.tokenizer_dir):
            raise FileNotFoundError(f"--tokenizer-dir not found: {args.tokenizer_dir}")
        copied = []
        for name in os.listdir(args.tokenizer_dir):
            if name in _TOKENIZER_FILENAMES:
                shutil.copy2(
                    os.path.join(args.tokenizer_dir, name),
                    os.path.join(args.output_dir, name),
                )
                copied.append(name)
        print(f"Copied {len(copied)} tokenizer file(s): {sorted(copied)}")

    print(f"Done. Consolidated HF checkpoint at: {args.output_dir}")


if __name__ == "__main__":
    main()
