# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DIAGNOSTIC conftest.py — temporarily logs sys.path at hook + spawn time.
Will revert after CI run."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

print(f"[CONFTEST_LOAD] cwd={os.getcwd()}", file=sys.stderr, flush=True)
print(f"[CONFTEST_LOAD] sys.path[:6]={sys.path[:6]}", file=sys.stderr, flush=True)

# Pre-import vllm/sglang to seed sys.modules with the real packages.
for _name in ("vllm", "sglang"):
    try:
        m = importlib.import_module(_name)
        print(f"[CONFTEST_LOAD] {_name} from {getattr(m, '__file__', '?')}", file=sys.stderr, flush=True)
    except ImportError as e:
        print(f"[CONFTEST_LOAD] {_name} import failed: {e}", file=sys.stderr, flush=True)

os.environ.setdefault("PY_IGNORE_IMPORTMISMATCH", "1")

_BAD_DYNAMO_PATH = str(
    Path(__file__).resolve().parent / "components" / "src" / "dynamo"
)
print(f"[CONFTEST_LOAD] _BAD_DYNAMO_PATH={_BAD_DYNAMO_PATH}", file=sys.stderr, flush=True)

# Intercept multiprocessing spawn to log sys.path at child-launch time.
try:
    import multiprocessing.spawn as _mps
    _orig_prep = _mps.get_preparation_data
    def _patched_prep(name):
        bad = _BAD_DYNAMO_PATH in sys.path
        print(f"[SPAWN_PREP] name={name} bad_in_path={bad}", file=sys.stderr, flush=True)
        if bad:
            idx = sys.path.index(_BAD_DYNAMO_PATH)
            print(f"[SPAWN_PREP] bad at idx={idx}, sys.path[:8]={sys.path[:8]}", file=sys.stderr, flush=True)
        return _orig_prep(name)
    _mps.get_preparation_data = _patched_prep
except Exception as e:
    print(f"[CONFTEST_LOAD] spawn patch failed: {e}", file=sys.stderr, flush=True)


def pytest_runtest_setup(item):
    bad_in = _BAD_DYNAMO_PATH in sys.path
    print(f"[CONFTEST_HOOK] {item.nodeid} bad_in={bad_in}", file=sys.stderr, flush=True)
    while _BAD_DYNAMO_PATH in sys.path:
        sys.path.remove(_BAD_DYNAMO_PATH)
