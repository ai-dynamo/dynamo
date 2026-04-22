# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang XPU / set_device_type paths."""

import copy

import pytest

from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS
from dynamo.profiler.utils.dgdr_v1beta1_types import DeviceType

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def _make_agg_config(
    *, gpu_count: str = "2", metadata_name: str = "sglang-agg"
) -> dict:
    """Return a minimal SGLang aggregated DGD config dict."""
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": metadata_name},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "my-registry/sglang-runtime:my-tag",
                        }
                    },
                },
                "decode": {
                    "componentType": "worker",
                    "replicas": 1,
                    "resources": {
                        "limits": {"gpu": gpu_count},
                        "requests": {"gpu": gpu_count},
                    },
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "my-registry/sglang-runtime:my-tag",
                            "args": [
                                "--model-path",
                                "Qwen/Qwen3-0.6B",
                                "--tp",
                                "1",
                            ],
                        }
                    },
                },
            }
        },
    }


def _make_disagg_config(*, gpu_count: str = "1") -> dict:
    """Return a minimal SGLang disaggregated DGD config dict with prefill + decode workers."""
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "sglang-disagg"},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "my-registry/sglang-runtime:my-tag",
                        }
                    },
                },
                "prefill": {
                    "componentType": "worker",
                    "subComponentType": "prefill",
                    "replicas": 1,
                    "resources": {
                        "limits": {"gpu": gpu_count},
                        "requests": {"gpu": gpu_count},
                    },
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "my-registry/sglang-runtime:my-tag",
                            "args": [
                                "--model-path",
                                "Qwen/Qwen3-0.6B",
                                "--tp",
                                "1",
                                "--disaggregation-mode",
                                "prefill",
                            ],
                        }
                    },
                },
                "decode": {
                    "componentType": "worker",
                    "subComponentType": "decode",
                    "replicas": 1,
                    "resources": {
                        "limits": {"gpu": gpu_count},
                        "requests": {"gpu": gpu_count},
                    },
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "my-registry/sglang-runtime:my-tag",
                            "args": [
                                "--model-path",
                                "Qwen/Qwen3-0.6B",
                                "--tp",
                                "1",
                                "--disaggregation-mode",
                                "decode",
                            ],
                        }
                    },
                },
            }
        },
    }


class TestSetDeviceTypeCudaNoop:
    """set_device_type is a no-op for CUDA (default) device type."""

    def test_cuda_string_returns_unchanged(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config()
        original = copy.deepcopy(config)
        result = modifier.set_device_type(config, "cuda")
        assert result == original

    def test_cuda_enum_returns_unchanged(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config()
        original = copy.deepcopy(config)
        result = modifier.set_device_type(config, DeviceType.Cuda)
        assert result == original


class TestSetDeviceTypeXpu:
    """set_device_type injects XPU-specific config when device_type is 'xpu'."""

    def test_removes_gpu_resource_limits_and_requests(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config(gpu_count="4")
        result = modifier.set_device_type(config, DeviceType.Xpu)

        worker = result["spec"]["services"]["decode"]
        resources = worker.get("resources") or {}
        limits = resources.get("limits") or {}
        requests = resources.get("requests") or {}
        assert "gpu" not in limits
        assert "gpu" not in requests

    def test_creates_resource_claim_template(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config()
        result = modifier.set_device_type(config, DeviceType.Xpu)

        rct = result.get("_xpu_resource_claim_template")
        assert rct is not None
        assert rct["kind"] == "ResourceClaimTemplate"
        assert rct["apiVersion"] == "resource.k8s.io/v1"

    def test_template_name_derived_from_metadata(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config(metadata_name="my-sglang-deployment")
        result = modifier.set_device_type(config, DeviceType.Xpu)

        rct = result["_xpu_resource_claim_template"]
        assert rct["metadata"]["name"] == "my-sglang-deployment-gpu-template"

        worker = result["spec"]["services"]["decode"]
        pod_claims = worker["extraPodSpec"]["resourceClaims"]
        assert any(
            rc.get("resourceClaimTemplateName") == "my-sglang-deployment-gpu-template"
            for rc in pod_claims
        )

    def test_template_name_fallback_without_metadata_name(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config(metadata_name="")
        result = modifier.set_device_type(config, DeviceType.Xpu)
        rct = result["_xpu_resource_claim_template"]
        assert rct["metadata"]["name"] == "gpu-template"

    def test_gpu_count_from_resources(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config(gpu_count="4")
        result = modifier.set_device_type(config, DeviceType.Xpu)

        rct = result["_xpu_resource_claim_template"]
        devices = rct["spec"]["spec"]["devices"]["requests"][0]
        assert devices["exactly"]["count"] == 4
        assert devices["exactly"]["deviceClassName"] == "gpu.intel.com"

    def test_gpu_count_fallback_when_no_resources(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config()
        del config["spec"]["services"]["decode"]["resources"]
        result = modifier.set_device_type(config, DeviceType.Xpu)

        rct = result["_xpu_resource_claim_template"]
        devices = rct["spec"]["spec"]["devices"]["requests"][0]
        assert devices["exactly"]["count"] == 1

    def test_gpu_count_inferred_from_tp_arg(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config()
        del config["spec"]["services"]["decode"]["resources"]
        config["spec"]["services"]["decode"]["extraPodSpec"]["mainContainer"][
            "args"
        ] = [
            "--model-path",
            "Qwen/Qwen3-0.6B",
            "--tp",
            "8",
        ]
        result = modifier.set_device_type(config, DeviceType.Xpu)

        rct = result["_xpu_resource_claim_template"]
        devices = rct["spec"]["spec"]["devices"]["requests"][0]
        assert devices["exactly"]["count"] == 8

    def test_gpu_count_inferred_from_tensor_parallel_size(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config()
        del config["spec"]["services"]["decode"]["resources"]
        config["spec"]["services"]["decode"]["extraPodSpec"]["mainContainer"][
            "args"
        ] = [
            "--model-path",
            "Qwen/Qwen3-0.6B",
            "--tensor-parallel-size",
            "4",
        ]
        result = modifier.set_device_type(config, DeviceType.Xpu)

        rct = result["_xpu_resource_claim_template"]
        devices = rct["spec"]["spec"]["devices"]["requests"][0]
        assert devices["exactly"]["count"] == 4

    def test_worker_pod_has_resource_claims(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config()
        result = modifier.set_device_type(config, DeviceType.Xpu)

        worker = result["spec"]["services"]["decode"]
        pod_claims = worker["extraPodSpec"]["resourceClaims"]
        assert any(rc.get("name") == "gpu" for rc in pod_claims)
        container_resources = worker["extraPodSpec"]["mainContainer"]["resources"]
        assert any(c.get("name") == "gpu" for c in container_resources["claims"])

    def test_frontend_not_modified(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config()
        result = modifier.set_device_type(config, DeviceType.Xpu)

        result_frontend = result["spec"]["services"]["Frontend"]
        fe_container = result_frontend.get("extraPodSpec", {}).get("mainContainer", {})
        assert "env" not in fe_container or fe_container["env"] is None
        assert "resourceClaims" not in result_frontend.get("extraPodSpec", {})

    def test_xpu_string_accepted(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config()
        result = modifier.set_device_type(config, "xpu")
        assert "_xpu_resource_claim_template" in result

    def test_disagg_both_workers_get_xpu_config(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_disagg_config()
        result = modifier.set_device_type(config, DeviceType.Xpu)

        for worker_name in ("prefill", "decode"):
            worker = result["spec"]["services"][worker_name]
            pod_claims = worker["extraPodSpec"]["resourceClaims"]
            assert any(
                rc.get("name") == "gpu" for rc in pod_claims
            ), f"{worker_name} missing resourceClaim"
            container_resources = worker["extraPodSpec"]["mainContainer"]["resources"]
            assert any(
                c.get("name") == "gpu" for c in container_resources["claims"]
            ), f"{worker_name} missing container claim"

    def test_idempotent_double_call(self) -> None:
        modifier = CONFIG_MODIFIERS["sglang"]
        config = _make_agg_config()
        result1 = modifier.set_device_type(config, DeviceType.Xpu)
        result2 = modifier.set_device_type(result1, DeviceType.Xpu)

        worker = result2["spec"]["services"]["decode"]
        pod_claims = worker["extraPodSpec"]["resourceClaims"]
        gpu_claims = [rc for rc in pod_claims if rc.get("name") == "gpu"]
        assert len(gpu_claims) == 1, "GPU resource claim should not be duplicated"
