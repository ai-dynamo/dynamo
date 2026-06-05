# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Dynamo wheels shipped in the runtime image.

These run *inside* a clean base image (python:3.12-slim or manylinux_2_28); the CI
workflow extracts the runtime image's /opt/dynamo/wheelhouse to the host and mounts it
in. The tests never spawn docker — they operate on the wheelhouse directory pointed to
by DYNAMO_WHEEL_SMOKE_WHEELHOUSE and install into throwaway venvs in the current image.
"""

from __future__ import annotations

import os
import shutil
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


def _target_arch() -> str:
    platform = os.environ.get(PLATFORM_ENV, "linux/amd64")
    return platform.rsplit("/", 1)[-1]


def _extras() -> list[str]:
    extras = os.environ.get(EXTRAS_ENV, "mocker,vllm,sglang,trtllm")
    return [extra.strip() for extra in extras.split(",") if extra.strip()]


@pytest.fixture(scope="session")
def wheelhouse() -> Path:
    raw = os.environ.get(WHEELHOUSE_ENV, "").strip()
    if not raw:
        pytest.skip(f"{WHEELHOUSE_ENV} is not set")
    path = Path(raw).resolve()
    if not path.exists():
        pytest.fail(f"{WHEELHOUSE_ENV} points at a missing path: {path}")
    print(f"wheelhouse: {path}")
    for wheel in smoke_install.all_wheels(path):
        print(" ", wheel.relative_to(path))
    return path


def test_core_install_clean_room(wheelhouse: Path) -> None:
    smoke_install.install_core(wheelhouse, sys.executable)


@pytest.mark.parametrize("extra", _extras())
def test_extra_declared_in_wheel(wheelhouse: Path, extra: str) -> None:
    smoke_install.assert_extra_declared(wheelhouse, extra)


def test_import_on_manylinux_2_28_floor(wheelhouse: Path) -> None:
    smoke_install.install_core(wheelhouse, sys.executable)


def test_wheel_metadata_tags_auditwheel_glibc(wheelhouse: Path) -> None:
    if shutil.which("auditwheel") is None or shutil.which("readelf") is None:
        pytest.skip("auditwheel/readelf unavailable; run this scenario in manylinux")
    target_arch = _target_arch()
    expect_optional = bool(os.environ.get(EXPECT_OPTIONAL_ENV, "").strip())
    smoke_install.assert_core_wheel_metadata(wheelhouse, target_arch)
    smoke_install.check_optional_wheels(wheelhouse, target_arch, expect_optional)
    smoke_install.assert_auditwheel_show(wheelhouse)
    smoke_install.assert_glibc_floor(wheelhouse)
