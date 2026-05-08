# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block dynamo.vllm/sglang from shadowing the installed vllm/sglang.

Pytest collection puts components/src/dynamo on sys.path, which makes
`import vllm` resolve to dynamo.vllm. Spawned subprocesses (EngineCore,
sglang scheduler) inherit that and crash on `from vllm.v1 ...`.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# Seed sys.modules with the venv copies before pytest collection runs.
for _name in ("vllm", "sglang"):
    try:
        importlib.import_module(_name)
    except ImportError:
        pass

# Suppress ImportPathMismatchError when pytest later loads dynamo.vllm
# under the bare name "vllm".
os.environ.setdefault("PY_IGNORE_IMPORTMISMATCH", "1")

_BAD_DYNAMO_PATH = str(
    Path(__file__).resolve().parent / "components" / "src" / "dynamo"
)


def _strip_bad_path() -> None:
    while _BAD_DYNAMO_PATH in sys.path:
        sys.path.remove(_BAD_DYNAMO_PATH)


# Strip the bad path before multiprocessing.spawn freezes sys.path for the
# child — catches re-insertions that happen during fixture/test execution.
try:
    import multiprocessing.spawn as _mps

    _orig_get_preparation_data = _mps.get_preparation_data

    def _patched_get_preparation_data(name):
        _strip_bad_path()
        return _orig_get_preparation_data(name)

    _mps.get_preparation_data = _patched_get_preparation_data
except Exception:
    pass


def pytest_runtest_setup(item):
    _strip_bad_path()
