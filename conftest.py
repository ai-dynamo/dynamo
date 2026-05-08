# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stop the dynamo.vllm / dynamo.sglang subpackages from shadowing the
installed vllm / sglang during pytest runs.

Pytest's collection adds components/src/dynamo to sys.path (something during
test/fixture setup re-inserts it even after pytest_runtest_setup removes it).
With that path in front, `import vllm` resolves to dynamo.vllm and breaks
`from vllm.v1 ...`, both in the parent and in any subprocess (EngineCore,
sglang scheduler) that inherits sys.path.

Defense in depth:
  1. Pre-import the real vllm/sglang at conftest load so sys.modules is
     seeded with the venv copies.
  2. pytest_runtest_setup removes the bad path before each test runs.
  3. A patched ``multiprocessing.spawn.get_preparation_data`` strips the
     bad path right before the parent freezes sys.path for a spawned
     child — covers cases where something re-adds it during fixture/test
     execution (which is what was crashing EngineCore subprocesses with
     ``ModuleNotFoundError: No module named 'vllm.v1'``).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# trtllm uses the `tensorrt_llm` import name, so it does not collide.
for _name in ("vllm", "sglang"):
    try:
        importlib.import_module(_name)
    except ImportError:
        pass

# Suppress ImportPathMismatchError when pytest's Package collection later
# tries to load the local dynamo.vllm/__init__.py with name "vllm".
os.environ.setdefault("PY_IGNORE_IMPORTMISMATCH", "1")

_BAD_DYNAMO_PATH = str(
    Path(__file__).resolve().parent / "components" / "src" / "dynamo"
)


def _strip_bad_path() -> None:
    while _BAD_DYNAMO_PATH in sys.path:
        sys.path.remove(_BAD_DYNAMO_PATH)


# Patch multiprocessing.spawn so any subprocess vllm/sglang spawn during a
# test inherits a clean sys.path even if something re-added the bad path
# between pytest_runtest_setup and the spawn point.
try:
    import multiprocessing.spawn as _mps

    _orig_get_preparation_data = _mps.get_preparation_data

    def _patched_get_preparation_data(name):
        _strip_bad_path()
        return _orig_get_preparation_data(name)

    _mps.get_preparation_data = _patched_get_preparation_data
except Exception:
    # If multiprocessing internals change, fall through — the
    # pytest_runtest_setup hook still provides primary defense.
    pass


def pytest_runtest_setup(item):
    # Keep components/src/dynamo off sys.path so spawned subprocesses
    # don't inherit it and re-resolve `import vllm` to dynamo.vllm.
    _strip_bad_path()
