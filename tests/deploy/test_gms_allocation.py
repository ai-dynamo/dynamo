# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the aggregate GMS deployment's GPU allocation assertions."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.deploy.test_dgd import validate_gms_allocation

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.fixture
def gms_deployment(tmp_path):
    pod = {
        "metadata": {"uid": "worker-uid"},
        "spec": {
            "containers": [
                {
                    "name": "main",
                    "resources": {"claims": [{"name": "intrapod-shared-gpu"}]},
                }
            ],
            "initContainers": [
                {
                    "name": "gms-server",
                    "restartPolicy": "Always",
                    "resources": {"claims": [{"name": "intrapod-shared-gpu"}]},
                }
            ],
        },
        "status": {
            "initContainerStatuses": [{"name": "gms-server", "ready": True}],
            "resourceClaimStatuses": [
                {"name": "intrapod-shared-gpu", "resourceClaimName": "worker-gpu"}
            ],
        },
    }
    claim = {
        "status": {
            "allocation": {"devices": {"results": [{"driver": "gpu.nvidia.com"}]}},
            "reservedFor": [{"uid": "worker-uid"}],
        }
    }
    return SimpleNamespace(
        get_pods=MagicMock(return_value={"worker": [SimpleNamespace(raw=pod)]}),
        _custom_api=SimpleNamespace(
            get_namespaced_custom_object=AsyncMock(return_value=claim)
        ),
        namespace="gms-test",
        log_dir=str(tmp_path),
    )


async def test_gms_allocation_records_shared_gpu(gms_deployment, tmp_path):
    await validate_gms_allocation(gms_deployment)

    gms_deployment._custom_api.get_namespaced_custom_object.assert_awaited_once_with(
        "resource.k8s.io", "v1", "gms-test", "resourceclaims", "worker-gpu"
    )
    assert (tmp_path / "gms-resourceclaim.json").exists()


async def test_gms_allocation_keeps_unallocated_claim_artifact(
    gms_deployment, tmp_path
):
    claim = gms_deployment._custom_api.get_namespaced_custom_object.return_value
    claim["status"].pop("allocation")

    with pytest.raises(AssertionError, match="exactly one NVIDIA GPU"):
        await validate_gms_allocation(gms_deployment)

    assert json.loads((tmp_path / "gms-resourceclaim.json").read_text()) == claim


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("missing-worker", "exactly one worker"),
        ("missing-sidecar", "not injected"),
        ("non-native-sidecar", "native sidecar"),
        ("unready-sidecar", "not ready"),
        ("worker-without-claim", "main does not use"),
        ("sidecar-without-claim", "gms-server does not use"),
        ("unresolved-claim", "no resolved ResourceClaim"),
        ("unallocated-claim", "exactly one NVIDIA GPU"),
        ("wrong-driver", "exactly one NVIDIA GPU"),
        ("multiple-gpus", "exactly one NVIDIA GPU"),
        ("wrong-reservation", "not reserved for the worker"),
    ],
)
async def test_gms_allocation_rejects_invalid_deployment(
    gms_deployment, fault, message
):
    pod = gms_deployment.get_pods.return_value["worker"][0].raw
    claim = gms_deployment._custom_api.get_namespaced_custom_object.return_value
    if fault == "missing-worker":
        gms_deployment.get_pods.return_value["worker"] = []
    elif fault == "missing-sidecar":
        pod["spec"]["initContainers"] = []
    elif fault == "non-native-sidecar":
        pod["spec"]["initContainers"][0].pop("restartPolicy")
    elif fault == "unready-sidecar":
        pod["status"]["initContainerStatuses"][0]["ready"] = False
    elif fault == "worker-without-claim":
        pod["spec"]["containers"][0]["resources"]["claims"] = []
    elif fault == "sidecar-without-claim":
        pod["spec"]["initContainers"][0]["resources"]["claims"] = []
    elif fault == "unresolved-claim":
        pod["status"]["resourceClaimStatuses"] = []
    elif fault == "unallocated-claim":
        claim["status"].pop("allocation")
    elif fault == "wrong-driver":
        claim["status"]["allocation"]["devices"]["results"][0]["driver"] = "other"
    elif fault == "multiple-gpus":
        claim["status"]["allocation"]["devices"]["results"] *= 2
    elif fault == "wrong-reservation":
        claim["status"]["reservedFor"][0]["uid"] = "another-pod"

    with pytest.raises(AssertionError, match=message):
        await validate_gms_allocation(gms_deployment)
