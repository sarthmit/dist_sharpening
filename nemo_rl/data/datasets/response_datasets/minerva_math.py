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

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class MathLightEvalDataset(RawDataset):
    """Wrapper around the DigitalLearningGmbH/MATH-lighteval dataset.

    Args:
        output_key: Key for the output text, default is "solution"
        split: Split name for the dataset, default is "train"
        subset: Builder config name (e.g., "default", "algebra", "geometry"), default is "default"
        split_validation_size: Size of the validation data, default is 0
        seed: Seed for train/validation split when split_validation_size > 0, default is 42
    """

    _DATASET_NAME = "DigitalLearningGmbH/MATH-lighteval"
    _SUPPORTED_SPLITS = {"train", "test"}
    _SUPPORTED_SUBSETS = {
        "default",
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    }

    def __init__(
        self,
        output_key: str = "solution",
        split: str = "train",
        subset: Optional[str] = "default",
        split_validation_size: float = 0,
        seed: int = 42,
        **kwargs,
    ):
        if split not in self._SUPPORTED_SPLITS:
            raise ValueError(
                f"Invalid split: {split}. Please use 'train' or 'test'."
            )

        if subset is not None and subset not in self._SUPPORTED_SUBSETS:
            raise ValueError(
                f"Invalid subset: {subset}. Please use one of: {sorted(self._SUPPORTED_SUBSETS)}."
            )

        self.input_key = "problem"
        self.output_key = output_key
        self.task_name = "MATH-lighteval"

        if subset is None:
            self.dataset = load_dataset(self._DATASET_NAME, split=split)
        else:
            self.dataset = load_dataset(self._DATASET_NAME, name=subset, split=split)

        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": data[self.input_key]},
                {"role": "assistant", "content": data[self.output_key]},
            ],
            "task_name": self.task_name,
        }
