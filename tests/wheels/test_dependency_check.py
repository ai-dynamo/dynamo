# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve strict dependency checks around the declared production override."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.wheels import smoke_install

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

PYTHON = Path("/fixture/bin/python")
PILLOW_CONFLICT = (
    "aiperf 0.13.0 has requirement pillow~=12.3.0, but you have pillow 13.0.0."
)


@pytest.fixture
def installed_check(monkeypatch: pytest.MonkeyPatch) -> tuple:
    result = subprocess.CompletedProcess([], 1, stdout=PILLOW_CONFLICT)
    installed = {
        "aiperf": "0.13.0",
        "pillow": "13.0.0",
        "requires": ["pillow~=12.3.0"],
        "platform": "aarch64",
        "system": "Linux",
        "supported_tags": ["py3-none-manylinux_2_28_aarch64"],
        "platform_wheels": {},
    }
    monkeypatch.setattr(smoke_install.subprocess, "run", lambda *a, **k: result)
    monkeypatch.setattr(
        smoke_install.subprocess,
        "check_output",
        lambda *a, **k: json.dumps(installed),
    )
    return result, installed


@pytest.fixture
def declared_policy(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[2]
    policy = tmp_path / "overrides.planner.txt"
    policy.write_bytes((source / "container/deps/overrides.planner.txt").read_bytes())
    return policy


def test_exact_declared_override_checks_actual_installed_version(
    installed_check: tuple, declared_policy: Path
) -> None:
    smoke_install.pip_check(PYTHON, pillow_override=declared_policy)


def test_default_clean_room_does_not_allow_the_production_override(
    installed_check: tuple,
) -> None:
    with pytest.raises(AssertionError, match="dependency conflicts"):
        smoke_install.pip_check(PYTHON)


def test_policy_checks_a_runtime_without_aiperf(
    installed_check: tuple, declared_policy: Path
) -> None:
    result, installed = installed_check
    result.stdout = "No broken requirements found."
    result.returncode = 0
    installed["aiperf"] = None
    installed["requires"] = []
    smoke_install.pip_check(PYTHON, pillow_override=declared_policy)


@pytest.mark.parametrize(
    "damage",
    [
        "missing-policy",
        "wrong-policy",
        "wrong-edge",
        "missing-pillow",
        "out-of-range-pillow",
        "clean-output-below-policy-floor",
        "metadata-edge-differs",
        "other-conflict",
    ],
)
def test_override_cannot_hide_other_dependency_failures(
    installed_check: tuple, declared_policy: Path, damage: str
) -> None:
    result, installed = installed_check
    if damage == "missing-policy":
        declared_policy.unlink()
    elif damage == "wrong-policy":
        declared_policy.write_text("pillow>=12.3.0,<13\n")
    elif damage == "wrong-edge":
        result.stdout = PILLOW_CONFLICT.replace("~=12.3.0", "~=12.4.0")
    elif damage == "missing-pillow":
        result.stdout = "aiperf 0.13.0 requires pillow, which is not installed."
    elif damage == "out-of-range-pillow":
        result.stdout = PILLOW_CONFLICT.replace("pillow 13.0.0", "pillow 14.0.0")
        installed["pillow"] = "14.0.0"
    elif damage == "clean-output-below-policy-floor":
        result.stdout = "No broken requirements found."
        result.returncode = 0
        installed["pillow"] = "12.3.0"
    elif damage == "metadata-edge-differs":
        installed["requires"] = ["pillow~=12.4.0"]
    elif damage == "other-conflict":
        result.stdout += (
            "\ntyro 1.0.16 has requirement typeguard>=4, but you have typeguard 2.13.3."
        )
    with pytest.raises((AssertionError, FileNotFoundError)):
        smoke_install.pip_check(PYTHON, pillow_override=declared_policy)


@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "unknown-diagnostic",
        "wrong-tag",
        "wrong-architecture",
        "wrong-python",
        "wrong-glibc",
        "wrong-version",
        "missing-tag",
        "other-package",
    ],
)
def test_policy_requires_real_nvidia_arm_tag_evidence_for_platform_warnings(
    installed_check: tuple, declared_policy: Path, damage: str
) -> None:
    result, installed = installed_check
    result.stdout = "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform"
    installed["platform_wheels"] = {
        "nvidia-cusparselt-cu13": {
            "version": "0.8.1",
            "wheel": "Wheel-Version: 1.0\nTag: py3-none-manylinux_2_28_sbsa\n",
        }
    }
    if damage == "unknown-diagnostic":
        result.stdout = "pip dependency checking failed unexpectedly"
    elif damage == "wrong-tag":
        installed["platform_wheels"]["nvidia-cusparselt-cu13"]["wheel"] = (
            "Wheel-Version: 1.0\nTag: py3-none-manylinux_2_28_aarch64\n"
        )
    elif damage == "wrong-architecture":
        installed["platform"] = "x86_64"
    elif damage == "wrong-python":
        installed["platform_wheels"]["nvidia-cusparselt-cu13"]["wheel"] = (
            "Wheel-Version: 1.0\nTag: cp314-cp314-manylinux_2_28_sbsa\n"
        )
    elif damage == "wrong-glibc":
        installed["platform_wheels"]["nvidia-cusparselt-cu13"]["wheel"] = (
            "Wheel-Version: 1.0\nTag: py3-none-manylinux_2_99_sbsa\n"
        )
    elif damage == "wrong-version":
        installed["platform_wheels"]["nvidia-cusparselt-cu13"]["version"] = "0.8.0"
    elif damage == "missing-tag":
        installed["platform_wheels"]["nvidia-cusparselt-cu13"]["wheel"] = None
    elif damage == "other-package":
        result.stdout = "other-package 0.8.1 is not supported on this platform"
    if damage == "none":
        smoke_install.pip_check(PYTHON, pillow_override=declared_policy)
    else:
        with pytest.raises(AssertionError):
            smoke_install.pip_check(PYTHON, pillow_override=declared_policy)


def test_sbsa_alias_matches_full_tags_from_the_target_interpreter(
    installed_check: tuple,
    declared_policy: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, installed = installed_check
    result.stdout = "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform"
    installed["platform_wheels"] = {
        "nvidia-cusparselt-cu13": {
            "version": "0.8.1",
            "wheel": "Wheel-Version: 1.0\nTag: cp312-cp312-manylinux_2_28_sbsa\n",
        }
    }
    installed["supported_tags"] = ["cp312-cp312-manylinux_2_28_aarch64"]
    probes = []

    def probe(command: list[str], **kwargs: object) -> str:
        probes.append(command)
        return json.dumps(installed)

    monkeypatch.setattr(smoke_install.subprocess, "check_output", probe)
    smoke_install.pip_check(PYTHON, pillow_override=declared_policy)
    assert len(probes) == 1
    assert probes[0][:2] == [str(PYTHON), "-c"]
