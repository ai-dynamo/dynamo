# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "components" / "src"))

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]

try:
    from dynamo.profiler.utils.config import update_image
    from dynamo.profiler.utils.dgd_generation import add_planner_to_config
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        HardwareSpec,
        SLASpec,
        WorkloadSpec,
    )
    from dynamo.profiler.utils.profile_common import (
        derive_backend_image,
        derive_planner_image,
    )
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


def _make_dgdr(image: str) -> DynamoGraphDeploymentRequestSpec:
    return DynamoGraphDeploymentRequestSpec(
        model="Qwen/Qwen3-32B",
        backend="trtllm",
        image=image,
        hardware=HardwareSpec(gpuSku="h200_sxm", totalGpus=8, numGpusPerNode=8),
        workload=WorkloadSpec(isl=4000, osl=1000),
        sla=SLASpec(ttft=2000.0, itl=50.0),
    )


def _base_dgd_config(image: str) -> dict:
    return {
        "metadata": {"name": "test-dgd"},
        "spec": {
            "services": {
                "Frontend": {
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": image,
                            "args": ["serve"],
                        }
                    },
                }
            }
        },
    }


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        (
            "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.2.3",
            "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.3",
        ),
        (
            "nvcr.io/nvidia/ai-dynamo/dynamo-frontend@sha256:deadbeef",
            "nvcr.io/nvidia/ai-dynamo/dynamo-planner@sha256:deadbeef",
        ),
        (
            "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.2.3@sha256:deadbeef",
            "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.3@sha256:deadbeef",
        ),
    ],
)
def test_derive_planner_image_preserves_registry_tag_and_digest(
    image: str, expected: str
):
    assert derive_planner_image(image) == expected


@pytest.mark.parametrize(
    ("image", "backend", "expected"),
    [
        (
            "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.2.3",
            "vllm",
            "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3",
        ),
        (
            "nvcr.io/nvidia/ai-dynamo/dynamo-frontend@sha256:deadbeef",
            "sglang",
            "nvcr.io/nvidia/ai-dynamo/sglang-runtime@sha256:deadbeef",
        ),
        (
            "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.2.3@sha256:deadbeef",
            "trtllm",
            "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.2.3@sha256:deadbeef",
        ),
    ],
)
def test_derive_backend_image_preserves_registry_tag_and_digest(
    image: str, backend: str, expected: str
):
    assert derive_backend_image(image, backend) == expected


def test_add_planner_to_config_uses_dynamo_planner_image():
    image = "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.2.3"
    dgdr = _make_dgdr(image)
    config = _base_dgd_config(image)

    add_planner_to_config(dgdr, config)

    planner_image = config["spec"]["services"]["Planner"]["extraPodSpec"][
        "mainContainer"
    ]["image"]
    assert planner_image == "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.3"


def test_update_image_does_not_overwrite_planner_service_image():
    profiler_image = "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.2.3"
    worker_image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3"
    dgdr = _make_dgdr(profiler_image)
    config = _base_dgd_config(profiler_image)

    add_planner_to_config(dgdr, config)

    updated = update_image(config, worker_image)

    assert (
        updated["spec"]["services"]["Frontend"]["extraPodSpec"]["mainContainer"][
            "image"
        ]
        == worker_image
    )
    assert (
        updated["spec"]["services"]["Planner"]["extraPodSpec"]["mainContainer"]["image"]
        == "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.3"
    )
