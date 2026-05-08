# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stop the dynamo.vllm / dynamo.sglang subpackages from shadowing the
installed vllm / sglang during pytest runs.

Pytest's Package collection inserts components/src/dynamo into sys.path
so it can import each <pkg>/__init__.py. With that path in front,
`import vllm` resolves to dynamo.vllm and breaks `from vllm.v1 ...`,
both in the parent and in any subprocess (EngineCore, sglang scheduler)
that inherits sys.path.
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

_BAD_DYNAMO_PATH = str(Path(__file__).resolve().parent / "components" / "src" / "dynamo")


def pytest_runtest_setup(item):
    # Keep components/src/dynamo off sys.path so spawned subprocesses
    # don't inherit it and re-resolve `import vllm` to dynamo.vllm.
    while _BAD_DYNAMO_PATH in sys.path:
        sys.path.remove(_BAD_DYNAMO_PATH)
