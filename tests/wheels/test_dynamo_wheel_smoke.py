# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Dynamo wheels shipped in the runtime image.

These run as part of the normal CPU test suite inside the dynamo test image, where the
wheels live at /opt/dynamo/wheelhouse. The tests operate on that directory and install the
wheels into throwaway venvs (clean of the image's pre-installed dynamo). They spawn no
docker and need no configuration: the wheelhouse path and target arch are inferred, with
env overrides available. Nothing is skipped silently — a missing wheelhouse or missing
tooling is a failure.

Each test uses its own unique temp venv with no ports/sockets/services, so the suite is
xdist-safe (marked ``parallel``).
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
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.wheel_smoke,
]


WHEELHOUSE_ENV = "DYNAMO_WHEEL_SMOKE_WHEELHOUSE"
PLATFORM_ENV = "DYNAMO_WHEEL_SMOKE_PLATFORM"
PYTHONS_ENV = "DYNAMO_WHEEL_SMOKE_PYTHONS"
DEFAULT_WHEELHOUSE = "/opt/dynamo/wheelhouse"
# Core-only versions; 3.12 (image Python) is covered by the combined core+kvbm test.
DEFAULT_PYTHONS = "3.10,3.11,3.13"


def _target_arch() -> str:
    override = os.environ.get(PLATFORM_ENV, "").strip()
    if override:
        return override.rsplit("/", 1)[-1]
    machine = platform.machine().lower()
    return {"x86_64": "amd64", "aarch64": "arm64"}.get(machine, machine)


def _pythons() -> list[str]:
    raw = os.environ.get(PYTHONS_ENV, "").strip() or DEFAULT_PYTHONS
    return [spec.strip() for spec in raw.split(",") if spec.strip()]


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@pytest.fixture(scope="session")
def wheelhouse() -> Path:
    raw = os.environ.get(WHEELHOUSE_ENV, "").strip() or DEFAULT_WHEELHOUSE
    path = Path(raw).resolve()
    if not path.exists():
        pytest.fail(f"wheelhouse not found at {path} (set {WHEELHOUSE_ENV})")
    print(f"wheelhouse: {path}")
    for wheel in smoke_install.all_wheels(path):
        print(" ", wheel.relative_to(path))
    return path


@pytest.mark.parametrize("python_spec", _pythons())
def test_core_install_clean_room(wheelhouse: Path, python_spec: str) -> None:
    smoke_install.install_core(wheelhouse, python_spec)


def test_core_and_kvbm_install_clean_room(wheelhouse: Path) -> None:
    # kvbm pins nixl[cu12], which is cp312-only, so it installs on the image Python only.
    smoke_install.install_core(wheelhouse, sys.executable, also=("kvbm",))


def test_runtime_wheel_has_no_bundled_libraries(wheelhouse: Path) -> None:
    smoke_install.assert_no_bundled_libraries(wheelhouse)


def test_wheel_metadata_tags_auditwheel_glibc(wheelhouse: Path) -> None:
    missing = [m for m in ("auditwheel", "elftools") if not _have(m)]
    if missing:
        pytest.fail(f"required tooling missing in this image: {', '.join(missing)}")
    target_arch = _target_arch()
    smoke_install.assert_core_wheel_metadata(wheelhouse, target_arch)
    smoke_install.check_optional_wheels(wheelhouse, target_arch)
    smoke_install.assert_auditwheel_show(wheelhouse)
    smoke_install.assert_glibc_floor(wheelhouse)
