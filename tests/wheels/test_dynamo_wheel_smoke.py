# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.pre_merge,
    pytest.mark.wheel_smoke,
]


EXTRAS_ENV = "DYNAMO_WHEEL_SMOKE_EXTRAS"
IMAGE_ENV = "DYNAMO_WHEEL_SMOKE_IMAGE"
PYTHON_SMOKE_IMAGE = "python:3.12-slim"
MANYLINUX_IMAGES = {
    "amd64": "quay.io/pypa/manylinux_2_28_x86_64",
    "arm64": "quay.io/pypa/manylinux_2_28_aarch64",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_image() -> str:
    image = os.environ.get(IMAGE_ENV, "").strip()
    if not image:
        pytest.skip(f"{IMAGE_ENV} is not set")
    return image


def _platform() -> str:
    return os.environ.get("DYNAMO_WHEEL_SMOKE_PLATFORM", "linux/amd64")


def _target_arch() -> str:
    platform = _platform()
    return platform.rsplit("/", 1)[-1]


def _extras() -> list[str]:
    extras = os.environ.get(EXTRAS_ENV, "mocker,vllm,sglang,trtllm")
    return [extra.strip() for extra in extras.split(",") if extra.strip()]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=_repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        pytest.fail(
            "Command failed:\n" + " ".join(command) + "\n\n" + result.stdout,
        )
    return result


@pytest.fixture(scope="session")
def wheelhouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("dynamo-wheel-smoke")
    container_id = _run(
        [
            "docker",
            "create",
            "--platform",
            _platform(),
            _runtime_image(),
            "true",
        ]
    ).stdout.strip()
    try:
        _run(
            [
                "docker",
                "cp",
                f"{container_id}:/opt/dynamo/wheelhouse",
                str(destination),
            ]
        )
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    copied = destination / "wheelhouse"
    if not copied.exists():
        pytest.fail(f"runtime image did not contain /opt/dynamo/wheelhouse: {copied}")
    return copied


def _docker_run(
    image: str,
    wheelhouse: Path,
    command: list[str],
) -> None:
    script = _repo_root() / "tests/wheels/smoke_install.py"
    docker_command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        _platform(),
        "-e",
        "PIP_DISABLE_PIP_VERSION_CHECK=1",
        "-e",
        "PIP_ROOT_USER_ACTION=ignore",
        "-v",
        f"{wheelhouse}:/wheelhouse:ro",
        "-v",
        f"{script}:/opt/wheel-smoke/smoke_install.py:ro",
        image,
        *command,
    ]
    _run(docker_command)


def _python_smoke(wheelhouse: Path, scenario: str, *args: str) -> None:
    _docker_run(
        PYTHON_SMOKE_IMAGE,
        wheelhouse,
        [
            "python",
            "/opt/wheel-smoke/smoke_install.py",
            scenario,
            "--wheelhouse",
            "/wheelhouse",
            *args,
        ],
    )


def _manylinux_image() -> str:
    target_arch = _target_arch()
    try:
        return MANYLINUX_IMAGES[target_arch]
    except KeyError:
        pytest.fail(f"unsupported wheel smoke platform: {_platform()}")


def _manylinux_python_smoke(wheelhouse: Path, scenario: str, *args: str) -> None:
    _docker_run(
        _manylinux_image(),
        wheelhouse,
        [
            "/opt/python/cp312-cp312/bin/python",
            "/opt/wheel-smoke/smoke_install.py",
            scenario,
            "--wheelhouse",
            "/wheelhouse",
            "--target-arch",
            _target_arch(),
            *args,
        ],
    )


def _metadata_smoke(wheelhouse: Path) -> None:
    _docker_run(
        _manylinux_image(),
        wheelhouse,
        [
            "/opt/python/cp312-cp312/bin/python",
            "/opt/wheel-smoke/smoke_install.py",
            "metadata",
            "--wheelhouse",
            "/wheelhouse",
            "--target-arch",
            _target_arch(),
        ],
    )


def test_core_wheels_install_from_local_wheelhouse_in_clean_python_image(
    wheelhouse: Path,
) -> None:
    _python_smoke(wheelhouse, "core")


def test_core_wheels_import_on_manylinux_2_28_floor(wheelhouse: Path) -> None:
    _manylinux_python_smoke(wheelhouse, "core")


def test_wheel_tags_metadata_auditwheel_and_glibc_symbols(wheelhouse: Path) -> None:
    _metadata_smoke(wheelhouse)


@pytest.mark.parametrize("extra", _extras())
def test_extra_install_option_resolves_from_local_dynamo_wheels(
    wheelhouse: Path,
    extra: str,
) -> None:
    if _target_arch() == "arm64" and extra != "mocker":
        pytest.skip(f"ai-dynamo[{extra}] is smoke-tested on amd64 only")

    _python_smoke(wheelhouse, "extra", "--extra", extra)
