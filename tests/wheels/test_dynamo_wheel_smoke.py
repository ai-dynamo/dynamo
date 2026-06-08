# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Dynamo wheels shipped in the runtime image.

These run as part of the normal CPU test suite inside the dynamo test image, where the
wheels live at /opt/dynamo/wheelhouse. The tests operate on that directory and install the
core wheels into a throwaway venv (clean of the image's pre-installed dynamo). They spawn
no docker and need no configuration: the wheelhouse path, target arch, and whether the
optional CUDA wheels are expected are all inferred, with env overrides available.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path

import pytest

import smoke_install


pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.pre_merge,
    pytest.mark.wheel_smoke,
]


WHEELHOUSE_ENV = "DYNAMO_WHEEL_SMOKE_WHEELHOUSE"
EXTRAS_ENV = "DYNAMO_WHEEL_SMOKE_EXTRAS"
EXPECT_OPTIONAL_ENV = "DYNAMO_WHEEL_SMOKE_EXPECT_OPTIONAL"
PLATFORM_ENV = "DYNAMO_WHEEL_SMOKE_PLATFORM"
DEFAULT_WHEELHOUSE = "/opt/dynamo/wheelhouse"


def _target_arch() -> str:
    override = os.environ.get(PLATFORM_ENV, "").strip()
    if override:
        return override.rsplit("/", 1)[-1]
    machine = platform.machine().lower()
    return {"x86_64": "amd64", "aarch64": "arm64"}.get(machine, machine)


def _expect_optional() -> bool:
    override = os.environ.get(EXPECT_OPTIONAL_ENV, "").strip()
    if override:
        return override.lower() not in ("0", "false", "no")
    # No explicit signal: kvbm/gpu-memory-service ship only on CUDA builds.
    return bool(os.environ.get("CUDA_HOME")) or Path("/usr/local/cuda").exists()


def _extras() -> list[str]:
    extras = os.environ.get(EXTRAS_ENV, "mocker,vllm,sglang,trtllm")
    return [extra.strip() for extra in extras.split(",") if extra.strip()]


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@pytest.fixture(scope="session")
def wheelhouse() -> Path:
    raw = os.environ.get(WHEELHOUSE_ENV, "").strip() or DEFAULT_WHEELHOUSE
    path = Path(raw).resolve()
    if not path.exists():
        pytest.skip(f"wheelhouse not found at {path} (set {WHEELHOUSE_ENV})")
    print(f"wheelhouse: {path}")
    for wheel in smoke_install.all_wheels(path):
        print(" ", wheel.relative_to(path))
    return path


def test_core_install_clean_room(wheelhouse: Path) -> None:
    smoke_install.install_core(wheelhouse, sys.executable)


@pytest.mark.parametrize("extra", _extras())
def test_extra_declared_in_wheel(wheelhouse: Path, extra: str) -> None:
    smoke_install.assert_extra_declared(wheelhouse, extra)


def test_wheel_metadata_tags_auditwheel_glibc(wheelhouse: Path) -> None:
    if not (_have("auditwheel") and _have("elftools")):
        pytest.skip("auditwheel/pyelftools unavailable")
    target_arch = _target_arch()
    smoke_install.assert_core_wheel_metadata(wheelhouse, target_arch)
    smoke_install.check_optional_wheels(wheelhouse, target_arch, _expect_optional())
    smoke_install.assert_auditwheel_show(wheelhouse)
    smoke_install.assert_glibc_floor(wheelhouse)
