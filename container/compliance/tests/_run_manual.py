# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiny pytest-free runner for test_license_text.py (CI/local lack pytest)."""
import importlib.util
import inspect
import pathlib
import sys
import tempfile
import traceback

_here = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "test_license_text", _here / "test_license_text.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

passed = failed = 0
for name in sorted(dir(mod)):
    if not name.startswith("test_"):
        continue
    fn = getattr(mod, name)
    params = inspect.signature(fn).parameters
    try:
        if "tmp_path" in params:
            fn(pathlib.Path(tempfile.mkdtemp()))
        else:
            fn()
        passed += 1
        print(f"PASS {name}")
    except Exception as exc:  # noqa: BLE001
        failed += 1
        print(f"FAIL {name}: {exc!r}")
        traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
