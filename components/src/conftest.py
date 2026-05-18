# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-of-src conftest — installs a ``dynamo._core`` stub if needed.

This file lives **outside** the ``dynamo`` package on purpose. Pytest loads
conftest.py files top-down from rootdir, and this one runs before any
``dynamo.*`` package ``__init__.py`` is executed. That order is essential
for the Power Planner testbed: ``dynamo.planner/__init__.py`` eagerly
imports K8s connectors → ``dynamo.runtime.logging`` → ``dynamo._core``,
which is a maturin-built native binding that may not exist on a
developer laptop.

If the real ``dynamo._core`` is importable, this conftest is a no-op.
Otherwise it installs a minimal stub from
``dynamo.planner.tests.testbed._runtime_stub`` so the testbed can run
its α-class scenarios — none of which actually reach the runtime.

We deliberately do **not** import the stub installer through the
``dynamo`` package (which would trigger the import we're trying to
avoid). Instead we load it directly from its file path.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

_STUB_PATH = (
    pathlib.Path(__file__).parent
    / "dynamo"
    / "planner"
    / "tests"
    / "testbed"
    / "_runtime_stub.py"
)


def _install_dynamo_core_stub_if_needed() -> None:
    if not _STUB_PATH.is_file():
        return
    spec = importlib.util.spec_from_file_location(
        "_dynamo_core_stub_installer", _STUB_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_dynamo_core_stub_installer", mod)
    spec.loader.exec_module(mod)
    mod.install_stub_if_needed()
    # Also stub heavy ML predictor deps (pmdarima/filterpy/prophet) that
    # ``dynamo.planner.core.load.predictors`` imports at module load time.
    # The testbed only uses the ``constant`` predictor.
    if hasattr(mod, "install_predictor_deps_stub_if_needed"):
        mod.install_predictor_deps_stub_if_needed()


_install_dynamo_core_stub_if_needed()
