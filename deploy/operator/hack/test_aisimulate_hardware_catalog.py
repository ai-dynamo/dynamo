# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep DGDR's GPU SKU API and hardware hints aligned with pinned AISimulate."""

from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "deploy/operator/internal/aisimulate/hardware_skus.txt"
API_PATH = (
    REPO_ROOT / "deploy/operator/api/v1beta2/dynamographdeploymentrequest_types.go"
)
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
DGDR_CRD_PATH = (
    REPO_ROOT
    / "deploy/operator/config/crd/bases/nvidia.com_dynamographdeploymentrequests.yaml"
)

# v1beta1 accepts these SKUs without a packaged AISimulate system definition.
# gb200_sxm remains as a deprecated API compatibility value.
DYNAMO_ONLY_GPU_SKUS = frozenset(
    {
        "gb10",
        "gb200_sxm",
        "l40",
        "mi200",
        "mi300",
        "t4",
        "v100_pcie",
        "v100_sxm",
        "vr_nvl72",
    }
)

# AISimulate packages these systems, but the v1beta1 enum does not expose them.
AISIMULATE_ONLY_SYSTEM_IDS = frozenset(
    {
        "b60",
        "gb300",
        "rtx_pro_6000_server",
    }
)


def _pinned_version() -> Version:
    with PYPROJECT_PATH.open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]
    requirements = [
        Requirement(dependency)
        for dependency in dependencies
        if canonicalize_name(Requirement(dependency).name) == "aisimulate"
    ]
    assert len(requirements) == 1, "pyproject.toml must contain one AISimulate pin"

    requirement = requirements[0]
    specifiers = list(requirement.specifier)
    assert requirement.url is None
    assert len(specifiers) == 1 and specifiers[0].operator == "=="
    return Version(specifiers[0].version)


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


def _dgdr_v1beta1_gpu_skus() -> set[str]:
    # Read the generated enum enforced by the Kubernetes API server.
    document = yaml.safe_load(DGDR_CRD_PATH.read_text(encoding="utf-8"))
    version = next(
        version
        for version in document["spec"]["versions"]
        if version["name"] == "v1beta1"
    )
    properties = version["schema"]["openAPIV3Schema"]["properties"]
    schema = properties["spec"]["properties"]["hardware"]["properties"]["gpuSku"]
    enum_constraints = [
        set(constraint["enum"])
        for constraint in schema.get("allOf", ())
        if "enum" in constraint
    ]
    if "enum" in schema:
        enum_constraints.append(set(schema["enum"]))

    assert enum_constraints, "v1beta1 spec.hardware.gpuSku has no enum constraint"
    return set.intersection(*enum_constraints)


def test_dgdr_hardware_catalog_matches_pinned_aisimulate() -> None:
    assert Version(metadata.version("aisimulate")) == _pinned_version()

    expected = _packaged_hardware_skus()
    assert _operator_hardware_skus() == expected
    assert _documented_hardware_skus() == expected


def test_dgdr_v1beta1_gpu_skus_match_aisimulate_system_ids() -> None:
    dynamo_ids = _dgdr_v1beta1_gpu_skus()
    aisimulate_ids = set(_packaged_hardware_skus())

    actual_dynamo_only = dynamo_ids - aisimulate_ids
    actual_aisimulate_only = aisimulate_ids - dynamo_ids
    unexpected_dynamo_only = actual_dynamo_only - DYNAMO_ONLY_GPU_SKUS
    stale_dynamo_only = DYNAMO_ONLY_GPU_SKUS - actual_dynamo_only
    unexpected_aisimulate_only = actual_aisimulate_only - AISIMULATE_ONLY_SYSTEM_IDS
    stale_aisimulate_only = AISIMULATE_ONLY_SYSTEM_IDS - actual_aisimulate_only

    assert not any(
        (
            unexpected_dynamo_only,
            stale_dynamo_only,
            unexpected_aisimulate_only,
            stale_aisimulate_only,
        )
    ), (
        "DGDR/AISimulate system ID contract drifted. Reconcile the API and "
        "AISimulate registries, or update the explicit exception sets.\n"
        f"Unexpected Dynamo-only IDs: {sorted(unexpected_dynamo_only)}\n"
        f"Stale Dynamo-only exceptions: {sorted(stale_dynamo_only)}\n"
        f"Unexpected AISimulate-only IDs: {sorted(unexpected_aisimulate_only)}\n"
        f"Stale AISimulate-only exceptions: {sorted(stale_aisimulate_only)}"
    )
