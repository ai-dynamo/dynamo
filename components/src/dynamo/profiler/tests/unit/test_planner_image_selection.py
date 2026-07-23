# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]

try:
    from dynamo.profiler.utils.config import update_image
    from dynamo.profiler.utils.dgd_generation import (
        add_planner_to_config,
        apply_runtime_version_override,
        assemble_final_config,
    )
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        HardwareSpec,
        SLASpec,
        WorkloadSpec,
    )
    from dynamo.profiler.utils.profile_common import (
        ProfilerOperationalConfig,
        derive_backend_image,
        derive_planner_image,
    )
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


def _make_dgdr(
    image: str, runtime_version_override: str | None = None
) -> DynamoGraphDeploymentRequestSpec:
    return DynamoGraphDeploymentRequestSpec(
        model="Qwen/Qwen3-32B",
        backend="trtllm",
        image=image,
        runtimeVersionOverride=runtime_version_override,
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


def test_assemble_final_config_applies_runtime_version_override():
    image = "nvcr.io/nvidia/ai-dynamo/dynamo-planner:custom"
    dgdr = _make_dgdr(image, runtime_version_override="1.2.3")
    config = _base_dgd_config(image)

    result = assemble_final_config(dgdr, ProfilerOperationalConfig(), config)

    assert result["spec"]["services"]["Frontend"]["runtimeVersionOverride"] == "1.2.3"


def test_runtime_version_override_applies_to_injected_planner():
    image = "nvcr.io/nvidia/ai-dynamo/dynamo-planner:custom"
    dgdr = _make_dgdr(image, runtime_version_override="1.2.3")
    config = _base_dgd_config(image)
    add_planner_to_config(dgdr, config)

    apply_runtime_version_override(dgdr, config)

    assert {
        name: service["runtimeVersionOverride"]
        for name, service in config["spec"]["services"].items()
    } == {"Frontend": "1.2.3", "Planner": "1.2.3"}


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
