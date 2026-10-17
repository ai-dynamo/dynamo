# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the DGDR GPU SKU API aligned with AISimulate system identifiers."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.aiconfigurator,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

ROOT = Path(__file__).resolve().parents[2]
DGDR_CRD = (
    ROOT
    / "deploy/operator/config/crd/bases/nvidia.com_dynamographdeploymentrequests.yaml"
)

# DGDR accepts these SKUs even though the AISimulate wheel has no matching
# system definition. gb200_sxm remains as a deprecated API compatibility value.
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
    }
)

# AISimulate packages these system definitions, but DGDR does not expose them
# as selectable GPU SKUs yet.
AISIMULATE_ONLY_SYSTEM_IDS = frozenset(
    {
        "b60",
        "gb200",
        "gb300",
        "rtx_pro_6000_server",
        "vr200_hecate",
    }
)


def _dgdr_gpu_skus() -> set[str]:
    # The CRD schema is generated from the Go GPUSKUType marker and is the enum
    # enforced by the Kubernetes API server.
    document = yaml.safe_load(DGDR_CRD.read_text(encoding="utf-8"))
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


def _aisimulate_system_ids() -> set[str]:
    release = metadata.distribution("aisimulate")
    system_directory = Path("aisimulate_core/systems")
    system_ids = set()

    for relative_path in release.files or ():
        package_path = Path(str(relative_path))
        if package_path.parent != system_directory or package_path.suffix != ".yaml":
            continue

        document = yaml.safe_load(
            release.locate_file(relative_path).read_text(encoding="utf-8")
        )
        if (
            isinstance(document, dict)
            and {"data_dir", "gpu", "node"} <= document.keys()
        ):
            system_ids.add(package_path.stem)

    return system_ids


def test_dgdr_gpu_skus_match_aisimulate_system_ids() -> None:
    dynamo_ids = _dgdr_gpu_skus()
    aisimulate_ids = _aisimulate_system_ids()

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
