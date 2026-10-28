# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep DGDR's hardware hints synchronized with the pinned AISimulate wheel."""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]

REPO_ROOT = Path(__file__).resolve().parents[5]
CATALOG_PATH = REPO_ROOT / "deploy/operator/internal/aisimulate/hardware_skus.txt"
API_PATH = (
    REPO_ROOT / "deploy/operator/api/v1beta2/dynamographdeploymentrequest_types.go"
)
PIN_PATH = REPO_ROOT / "container/deps/requirements.aisimulate.txt"


def _pinned_version() -> str:
    pins = [
        line.split("==", 1)[1].strip()
        for line in PIN_PATH.read_text().splitlines()
        if line.startswith("aisimulate==")
    ]
    assert len(pins) == 1, "requirements.aisimulate.txt must contain one exact pin"
    return pins[0]


def _packaged_hardware_skus() -> list[str]:
    distribution = metadata.distribution("aisimulate")
    system_files = [
        path
        for path in distribution.files or []
        if path.parent.as_posix() == "aisimulate_core/systems"
        and path.suffix == ".yaml"
    ]
    result = []
    for packaged_path in system_files:
        with distribution.locate_file(packaged_path).open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if isinstance(document, dict) and "gpu" in document and "node" in document:
            result.append(packaged_path.stem)
    return sorted(result)


def _documented_hardware_skus() -> list[str]:
    source = API_PATH.read_text()
    match = re.search(
        r"Known values in the bundled AISimulate release are (.*?)\n\s*// The field remains open",
        source,
        re.DOTALL,
    )
    assert match is not None, "GPUHardwareSpec.SKU must document the bundled catalog"
    return sorted(re.findall(r"`([^`]+)`", match.group(1)))


def _operator_hardware_skus() -> list[str]:
    return sorted(
        line.strip()
        for line in CATALOG_PATH.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def test_dgdr_hardware_catalog_matches_pinned_aisimulate() -> None:
    assert metadata.version("aisimulate") == _pinned_version()

    expected = _packaged_hardware_skus()
    assert _operator_hardware_skus() == expected
    assert _documented_hardware_skus() == expected
