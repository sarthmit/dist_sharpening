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

"""MATH-lighteval dataset."""

from typing import Any, Optional

from datasets import load_dataset
from math_verify import parse

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


def _extract_answer(solution: str) -> str:
    parsed = parse(solution)
    if parsed:
        return parsed[-1]
    return solution


class MathLightEvalDataset:
    def __init__(
        self,
        split: str = "test",
        subset: Optional[str] = "default",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        if subset is None:
            ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split=split)
        else:
            ds = load_dataset(
                "DigitalLearningGmbH/MATH-lighteval", name=subset, split=split
            )

        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="mathlighteval",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.math_data_processor

    def _rekey(self, data: dict[str, Any]):
        return {
            "problem": data["problem"],
            "expected_answer": _extract_answer(data["solution"]),
        }
