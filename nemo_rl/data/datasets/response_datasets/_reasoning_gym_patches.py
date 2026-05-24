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

"""Monkey-patches for upstream reasoning_gym (>=0.1.25) crashes during dataset
materialization and scoring.

Three known failure modes used to kill training runs:

1. ``reasoning_gym.code.codeio``: the ``code_io`` task's bundled
   ``input_generator`` snippets sometimes return ``sympy.Integer`` (or other
   non-JSON-native types). The unguarded ``json.dumps(output_data)`` call
   raises ``TypeError: Object of type Integer is not JSON serializable``.
   We replace the module's ``json`` reference with a thin shim whose
   ``dumps`` always passes a ``default=`` that coerces sympy numerics to
   native Python types and falls back to ``str()`` for anything else.

2. ``reasoning_gym.games.maze``: the inner ``_random_floor_cell`` helper
   raises ``RuntimeError`` after exhausting its rejection-sampling budget on
   a degenerate (low ``prob_path``) maze, instead of letting the outer
   ``for _attempt in range(num_retries)`` loop in ``__getitem__`` regenerate
   a fresh maze. We wrap ``__getitem__`` so it retries with a shifted ``idx``
   (different effective per-sample RNG) when the inner exhaustion fires.

3. Four envs (``binary_matrix``, ``spiral_matrix``, ``string_insertion`` from
   ``algorithmic``; ``n_queens`` from ``games``) call ``eval(answer)`` on the
   model's raw output as a fallback when exact-string match fails. A
   completion containing ``[0]*10**9`` or similar instantly allocates many
   GB and OOMs the env actor (also a remote-code-execution surface).
   We shadow the module-level ``eval`` with ``ast.literal_eval``, which
   accepts the same valid list/tuple literals these scorers expect but
   syntactically rejects operators, function calls, comprehensions, and
   attribute access.

Importing this module applies all patches as a side effect; subsequent
``reasoning_gym.create_dataset(...)`` and ``score_answer(...)`` calls use
the fixed code paths.
"""

from __future__ import annotations

import ast
import json as _stdlib_json
from typing import Any

_MAZE_RETRY_BUDGET = 16


def _coerce_for_json(obj: Any) -> Any:
    try:
        import sympy

        if isinstance(obj, sympy.Integer):
            return int(obj)
        if isinstance(obj, (sympy.Float, sympy.Rational)):
            return float(obj)
    except ImportError:
        pass
    return str(obj)


class _SympySafeJsonShim:
    """Drop-in replacement for the stdlib ``json`` module that always passes
    ``default=_coerce_for_json`` to ``dumps``. All other attributes proxy to
    stdlib json untouched."""

    def dumps(self, obj: Any, **kwargs: Any) -> str:
        kwargs.setdefault("default", _coerce_for_json)
        return _stdlib_json.dumps(obj, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(_stdlib_json, name)


def _patch_codeio() -> None:
    try:
        from reasoning_gym.code import codeio
    except ImportError:
        return
    if getattr(codeio, "_nemo_rl_patched", False):
        return
    codeio.json = _SympySafeJsonShim()
    codeio._nemo_rl_patched = True


def _patch_maze() -> None:
    try:
        from reasoning_gym.games import maze as _maze_mod
    except ImportError:
        return
    if getattr(_maze_mod, "_nemo_rl_patched", False):
        return

    _orig_getitem = _maze_mod.MazeDataset.__getitem__

    def _patched_getitem(self: Any, idx: int) -> Any:
        last_err: Exception | None = None
        for shift in range(_MAZE_RETRY_BUDGET):
            try:
                return _orig_getitem(self, idx + shift * 100003)
            except RuntimeError as e:
                last_err = e
        raise RuntimeError(
            f"maze __getitem__({idx}) failed after {_MAZE_RETRY_BUDGET} "
            f"shifted retries; last error: {last_err}"
        )

    _maze_mod.MazeDataset.__getitem__ = _patched_getitem
    _maze_mod._nemo_rl_patched = True


def _safe_literal_eval(s: Any) -> Any:
    # Wrap in a bare callable so monkey-patched modules see the same
    # signature as builtin eval(str). literal_eval raises ValueError /
    # SyntaxError on non-literals; the upstream score_answer functions
    # already wrap their eval call in try/except, so failures convert
    # cleanly to score=0.0.
    return ast.literal_eval(s)


_UNSAFE_EVAL_MODULES = (
    "reasoning_gym.algorithmic.binary_matrix",
    "reasoning_gym.algorithmic.spiral_matrix",
    "reasoning_gym.algorithmic.string_insertion",
    "reasoning_gym.games.n_queens",
)


def _patch_unsafe_eval() -> None:
    import importlib

    for fqn in _UNSAFE_EVAL_MODULES:
        try:
            mod = importlib.import_module(fqn)
        except ImportError:
            continue
        if getattr(mod, "_nemo_rl_eval_patched", False):
            continue
        # Module-level shadow of the builtin: LOAD_GLOBAL inside the
        # module's score_answer resolves `eval` to the module's __dict__
        # before falling through to builtins.
        mod.eval = _safe_literal_eval
        mod._nemo_rl_eval_patched = True


def apply_all() -> None:
    _patch_codeio()
    _patch_maze()
    _patch_unsafe_eval()
