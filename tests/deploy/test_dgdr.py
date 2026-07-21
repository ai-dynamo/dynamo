# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-cluster tests for DynamoGraphDeploymentRequest."""

from __future__ import annotations

import pytest
from kubernetes_asyncio.client import exceptions

from tests.deploy.dgdr_utils import (
    ManagedDGDR,
    build_dgdr,
    kubectl,
    run_lifecycle,
    total_worker_gpus,
    unique_name,
)

pytestmark = [
    pytest.mark.k8s,
    pytest.mark.deploy,
    pytest.mark.e2e,
    pytest.mark.integration,
    pytest.mark.nightly,
]

PLANNER_MOCKER_MODEL = "Qwen/Qwen3-32B"
PLANNER_MOCKER_PROFILE_DATA = (
    "/workspace/components/src/dynamo/planner/tests/data/profiling_results/"
    "H200_TP1P_TP1D"
)


def manifest(
    manager: ManagedDGDR,
    suffix: str,
    spec: dict | None = None,
) -> dict:
    return build_dgdr(
        manager.config,
        unique_name(manager.config, suffix),
        spec_overrides=spec,
    )


def static_planner_mocker_worker(name: str, mode: str) -> dict:
    """Use checked-in profile data until planner images include AIC runtime."""

    return {
        "name": name,
        "podTemplate": {
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "args": [
                            "--model-path",
                            PLANNER_MOCKER_MODEL,
                            "--model-name",
                            PLANNER_MOCKER_MODEL,
                            "--speedup-ratio",
                            "1.0",
                            "--planner-profile-data",
                            PLANNER_MOCKER_PROFILE_DATA,
                            "--disaggregation-mode",
                            mode,
                        ],
                    }
                ]
            }
        },
    }


@pytest.mark.gpu_0
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "case",
    [
        "missing-model",
        "thorough-auto",
        "invalid-backend",
        "invalid-search-strategy",
        "invalid-optimization-type",
    ],
)
async def test_dgdr_validation_webhook_rejects_invalid_specs(
    dgdr_manager: ManagedDGDR, case: str
) -> None:
    """Port the five negative admission tests from the Ginkgo suite."""

    dgdr = manifest(dgdr_manager, case)
    if case == "missing-model":
        dgdr["spec"]["model"] = ""
    elif case == "thorough-auto":
        dgdr["spec"].update({"backend": "auto", "searchStrategy": "thorough"})
    elif case == "invalid-backend":
        dgdr["spec"]["backend"] = "unknown_backend"
    elif case == "invalid-search-strategy":
        dgdr["spec"]["searchStrategy"] = "superfast"
    elif case == "invalid-optimization-type":
        dgdr["spec"]["sla"] = {"optimizationType": "cost"}

    with pytest.raises(exceptions.ApiException) as caught:
        await dgdr_manager.dry_run(dgdr)
    if case == "thorough-auto":
        message = str(caught.value).lower()
        assert any(term in message for term in ("auto", "backend", "thorough"))


@pytest.mark.gpu_0
@pytest.mark.timeout(120)
@pytest.mark.parametrize("fully_specified", [False, True], ids=["minimal", "full"])
async def test_dgdr_validation_webhook_accepts_valid_specs(
    dgdr_manager: ManagedDGDR, fully_specified: bool
) -> None:
    dgdr = manifest(dgdr_manager, "valid-full" if fully_specified else "valid-minimal")
    if fully_specified:
        dgdr["spec"].update(
            {
                "backend": "vllm",
                "searchStrategy": "rapid",
                "sla": {"ttft": 200.0, "itl": 20.0},
                "workload": {"isl": 3000, "osl": 150},
                "autoApply": True,
            }
        )
    response = await dgdr_manager.dry_run(dgdr)
    assert response["kind"] == "DynamoGraphDeploymentRequest"


@pytest.mark.gpu_0
@pytest.mark.timeout(120)
async def test_dgdr_validation_crd_uses_v1beta1_storage(
    dgdr_manager: ManagedDGDR,
) -> None:
    crd = await dgdr_manager.get_crd()
    assert "v1beta1" in (crd.status.stored_versions or [])
    storage_versions = [
        version.name for version in crd.spec.versions if version.storage
    ]
    assert storage_versions == ["v1beta1"]


@pytest.mark.gpu_0
@pytest.mark.timeout(120)
async def test_dgdr_validation_crd_registers_shortname(
    dgdr_manager: ManagedDGDR,
) -> None:
    crd = await dgdr_manager.get_crd()
    assert "dgdr" in (crd.spec.names.short_names or [])
    result = kubectl(
        "get", "dgdr", "-n", dgdr_manager.config.namespace, "--ignore-not-found"
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.gpu_0
@pytest.mark.timeout(120)
async def test_dgdr_validation_crd_registers_expected_columns(
    dgdr_manager: ManagedDGDR,
) -> None:
    crd = await dgdr_manager.get_crd()
    version = next(item for item in crd.spec.versions if item.name == "v1beta1")
    columns = {
        column.name.upper() for column in version.additional_printer_columns or []
    }
    assert {"MODEL", "BACKEND", "PHASE"} <= columns


@pytest.mark.gpu_0
@pytest.mark.timeout(120)
async def test_dgdr_validation_v1alpha1_server_dry_run_is_not_internal_error(
    dgdr_manager: ManagedDGDR,
) -> None:
    name = unique_name(dgdr_manager.config, "v1a1")
    old_manifest = f"""apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: {name}
spec:
  model: {dgdr_manager.config.model}
  backend: vllm
  profilingConfig:
    profilerImage: {dgdr_manager.config.image}
"""
    result = kubectl(
        "apply",
        "-n",
        dgdr_manager.config.namespace,
        "-f",
        "-",
        "--dry-run=server",
        input_=old_manifest,
    )
    assert "Internal error" not in result.stderr + result.stdout


@pytest.mark.gpu_0
@pytest.mark.timeout(120)
async def test_dgdr_validation_v1beta1_object_has_v1alpha1_view(
    dgdr_manager: ManagedDGDR,
) -> None:
    dgdr = manifest(dgdr_manager, "conversion-get", {"autoApply": False})
    await dgdr_manager.create(dgdr)
    converted = await dgdr_manager.get_versioned(dgdr["metadata"]["name"], "v1alpha1")
    assert converted["apiVersion"] == "nvidia.com/v1alpha1"


@pytest.mark.gpu_0
@pytest.mark.timeout(3660)
async def test_dgdr_profiling_reaches_ready_and_job_succeeds(
    dgdr_manager: ManagedDGDR,
) -> None:
    dgdr = manifest(dgdr_manager, "lifecycle-ready", {"autoApply": False})
    await dgdr_manager.create(dgdr)
    result = await dgdr_manager.wait_for_phase(dgdr["metadata"]["name"], "Ready")
    job_name = result.get("status", {}).get("profilingJobName")
    assert job_name
    await dgdr_manager.assert_profiling_job_succeeded(job_name)


@pytest.mark.gpu_1
@pytest.mark.timeout(4260)
async def test_dgdr_lifecycle_reaches_deployed_with_real_worker(
    dgdr_manager: ManagedDGDR,
) -> None:
    if dgdr_manager.config.mocker:
        pytest.skip("run with --dgdr-no-mocker to exercise a real deployment")
    dgdr = manifest(dgdr_manager, "lifecycle-deployed", {"autoApply": True})
    result, _ = await run_lifecycle(dgdr_manager, dgdr, verify_configmap=False)
    assert result["status"]["phase"] == "Deployed"


@pytest.mark.gpu_0
@pytest.mark.timeout(3660)
async def test_dgdr_profiling_emits_final_config(
    dgdr_manager: ManagedDGDR,
) -> None:
    dgdr = manifest(
        dgdr_manager,
        "rapid-configmap",
        {
            "autoApply": False,
            "workload": {"isl": 4000, "osl": 1000},
            "sla": {"ttft": 2000.0, "itl": 30.0},
        },
    )
    _, output = await run_lifecycle(dgdr_manager, dgdr)
    assert output and output["kind"] == "DynamoGraphDeployment"


@pytest.mark.gpu_0
@pytest.mark.timeout(3660)
async def test_dgdr_profiling_includes_planner_service(
    dgdr_manager: ManagedDGDR,
) -> None:
    dgdr = manifest(
        dgdr_manager,
        "rapid-planner",
        {
            "autoApply": False,
            "backend": "trtllm",
            "workload": {"isl": 4000, "osl": 1000},
            "sla": {"ttft": 2000.0, "itl": 30.0},
            "features": {
                "planner": {
                    "enabled": True,
                    "optimization_target": "sla",
                    "pre_deployment_sweeping_mode": "rapid",
                }
            },
        },
    )
    await run_lifecycle(dgdr_manager, dgdr, expected_services={"Planner": 1})


@pytest.mark.gpu_0
@pytest.mark.timeout(3660)
async def test_dgdr_profiling_respects_total_gpu_budget(
    dgdr_manager: ManagedDGDR,
) -> None:
    total_gpus = 32
    dgdr = manifest(
        dgdr_manager,
        "rapid-budget",
        {
            "model": "Qwen/Qwen3-235B-A22B-FP8",
            "backend": "trtllm",
            "autoApply": False,
            "workload": {"isl": 4000, "osl": 1000},
            "sla": {"ttft": 2000.0, "itl": 30.0},
            "hardware": {
                "gpuSku": "h200_sxm",
                "vramMb": 141120.0,
                "numGpusPerNode": 8,
                "totalGpus": total_gpus,
            },
        },
    )
    _, output = await run_lifecycle(dgdr_manager, dgdr)
    assert output is not None
    assert total_worker_gpus(output) <= total_gpus


@pytest.mark.gpu_0
@pytest.mark.timeout(4260)
@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
async def test_dgdr_lifecycle_rapid_for_each_backend(
    dgdr_manager: ManagedDGDR, backend: str
) -> None:
    dgdr = manifest(
        dgdr_manager,
        f"{backend}-rapid",
        {"backend": backend, "searchStrategy": "rapid", "autoApply": True},
    )
    await run_lifecycle(dgdr_manager, dgdr)


@pytest.mark.gpu_1
@pytest.mark.timeout(3660)
async def test_dgdr_profiling_thorough_without_deployment(
    dgdr_manager: ManagedDGDR,
) -> None:
    dgdr = manifest(
        dgdr_manager,
        "vllm-thorough",
        {"backend": "vllm", "searchStrategy": "thorough", "autoApply": False},
    )
    await run_lifecycle(dgdr_manager, dgdr)


@pytest.mark.gpu_0
@pytest.mark.timeout(3660)
async def test_dgdr_profiling_rapid_without_auto_apply(
    dgdr_manager: ManagedDGDR,
) -> None:
    dgdr = manifest(dgdr_manager, "no-autoapply", {"autoApply": False})
    result, _ = await run_lifecycle(dgdr_manager, dgdr)
    assert result["status"]["phase"] == "Ready"


@pytest.mark.gpu_0
@pytest.mark.timeout(4260)
async def test_dgdr_lifecycle_with_planner(
    dgdr_manager: ManagedDGDR,
) -> None:
    dgdr = manifest(
        dgdr_manager,
        "planner",
        {
            "model": PLANNER_MOCKER_MODEL,
            "backend": "trtllm",
            "searchStrategy": "rapid",
            "autoApply": True,
            "sla": {"itl": 50.0},
            "hardware": {
                "gpuSku": "h200_sxm",
                "vramMb": 141120.0,
                "numGpusPerNode": 8,
                "totalGpus": 8,
            },
            "features": {
                "planner": {
                    "enabled": True,
                    "optimization_target": "sla",
                    "pre_deployment_sweeping_mode": "rapid",
                    "enable_throughput_scaling": True,
                    "enable_load_scaling": False,
                    "mode": "disagg",
                    "backend": "trtllm",
                }
            },
            "overrides": {
                "dgd": {
                    "apiVersion": "nvidia.com/v1beta1",
                    "kind": "DynamoGraphDeployment",
                    "spec": {
                        "components": [
                            static_planner_mocker_worker("prefill", "prefill"),
                            static_planner_mocker_worker("decode", "decode"),
                        ]
                    },
                }
            },
        },
    )
    await run_lifecycle(dgdr_manager, dgdr, expected_services={"Planner": 1})


@pytest.mark.gpu_0
@pytest.mark.timeout(4260)
@pytest.mark.parametrize(
    "sla,workload",
    [
        (
            {"ttft": 2000.0, "itl": 30.0},
            {"isl": 4000, "osl": 1000},
        ),
        (
            {"ttft": 500.0, "itl": 15.0, "optimizationType": "latency"},
            None,
        ),
        (
            {"ttft": 5000.0, "itl": 100.0, "optimizationType": "throughput"},
            None,
        ),
    ],
    ids=["custom-sla-workload", "latency", "throughput"],
)
async def test_dgdr_lifecycle_with_sla_variations(
    dgdr_manager: ManagedDGDR, sla: dict, workload: dict | None
) -> None:
    spec = {"autoApply": True, "sla": sla}
    if workload:
        spec["workload"] = workload
    dgdr = manifest(dgdr_manager, f"sla-{sla.get('optimizationType', 'custom')}", spec)
    await run_lifecycle(dgdr_manager, dgdr)


@pytest.mark.gpu_0
@pytest.mark.timeout(4260)
async def test_dgdr_lifecycle_with_custom_hardware(
    dgdr_manager: ManagedDGDR,
) -> None:
    dgdr = manifest(
        dgdr_manager,
        "custom-hardware",
        {
            "autoApply": True,
            "hardware": {
                "gpuSku": "a100_sxm",
                "vramMb": 81920.0,
                "numGpusPerNode": 8,
                "totalGpus": 8,
            },
        },
    )
    await run_lifecycle(dgdr_manager, dgdr)


@pytest.mark.gpu_0
@pytest.mark.timeout(10920)
async def test_dgdr_profiling_backends_sequentially(
    dgdr_manager: ManagedDGDR,
) -> None:
    for backend in ("vllm", "sglang", "trtllm"):
        dgdr = manifest(
            dgdr_manager,
            f"multi-{backend}",
            {"backend": backend, "searchStrategy": "rapid", "autoApply": False},
        )
        await run_lifecycle(dgdr_manager, dgdr)


@pytest.mark.gpu_0
@pytest.mark.timeout(3660)
async def test_dgdr_profiling_output_remains_available(
    dgdr_manager: ManagedDGDR,
) -> None:
    dgdr = manifest(
        dgdr_manager,
        "two-step",
        {
            "autoApply": False,
            "sla": {"ttft": 2000.0, "itl": 30.0},
            "workload": {"isl": 4000, "osl": 1000},
        },
    )
    _, first_output = await run_lifecycle(dgdr_manager, dgdr)
    second_output = await dgdr_manager.get_output_dgd(dgdr["metadata"]["name"])
    assert first_output == second_output
