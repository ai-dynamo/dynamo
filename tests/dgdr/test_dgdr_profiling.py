# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Profiler-focused DGDR v1beta1 end-to-end tests.

These tests validate profiling artifacts and generated config behavior when
using searchStrategy=rapid with autoApply=false.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
import yaml

from tests.dgdr.conftest import (
    DEFAULT_PROFILING_TIMEOUT,
    PHASE_READY,
    _run_kubectl,
    build_dgdr_manifest,
    unique_dgdr_name,
)
from tests.utils.managed_deployment import ManagedDGDR

PROFILING_TEST_TIMEOUT_S = 3 * DEFAULT_PROFILING_TIMEOUT


def _get_output_configmap(namespace: str, dgdr_name: str) -> Dict[str, Any]:
    """Return dgdr-output-<name> ConfigMap as JSON dict."""
    cm_name = f"dgdr-output-{dgdr_name}"
    result = _run_kubectl(
        ["get", "configmap", cm_name, "-n", namespace, "-o", "json"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Expected output ConfigMap {cm_name!r}. stderr={result.stderr}"
        )
    return json.loads(result.stdout)


def _load_final_dgd_from_configmap(cm: Dict[str, Any]) -> Dict[str, Any]:
    """Parse final_config.yaml from output ConfigMap and return DGD object."""
    data = cm.get("data", {})
    final_config = data.get("final_config.yaml")
    if not final_config:
        raise KeyError("ConfigMap must include data.final_config.yaml")

    docs = [d for d in yaml.safe_load_all(final_config) if isinstance(d, dict)]
    if not docs:
        raise ValueError("final_config.yaml must contain at least one YAML document")

    dgd = docs[-1]
    if dgd.get("kind") != "DynamoGraphDeployment":
        raise ValueError(
            "Expected final YAML document kind to be DynamoGraphDeployment"
        )
    return dgd


def _total_worker_gpus(dgd: Dict[str, Any]) -> int:
    """Compute total GPU requests across worker services in a generated DGD."""
    services = dgd.get("spec", {}).get("services", {})
    total = 0
    for service in services.values():
        replicas = int(service.get("replicas", 0) or 0)
        gpu_raw = service.get("resources", {}).get("limits", {}).get("gpu", 0)
        gpus = int(gpu_raw or 0)
        total += replicas * gpus
    return total


@pytest.mark.gpu_0
@pytest.mark.nightly
@pytest.mark.integration
@pytest.mark.k8s
class TestDGDRProfilingRapid:
    @pytest.mark.timeout(PROFILING_TEST_TIMEOUT_S)
    def test_rapid_autoapply_false_emits_output_configmap(
        self,
        managed_dgdr: ManagedDGDR,
        dgdr_factory,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_profiling_timeout: int,
    ) -> None:
        """Rapid profiling should reach Ready and emit output ConfigMap."""
        name = unique_dgdr_name("rapid-cm")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            auto_apply=False,
            workload={"isl": 4000, "osl": 1000},
            sla={"ttft": 2000.0, "itl": 30.0},
        )
        dgdr_factory(manifest)

        managed_dgdr.run(
            managed_dgdr.wait_for_phase(
                name, PHASE_READY, timeout=dgdr_profiling_timeout
            )
        )
        cm = _get_output_configmap(dgdr_namespace, name)
        _ = _load_final_dgd_from_configmap(cm)

    @pytest.mark.timeout(PROFILING_TEST_TIMEOUT_S)
    def test_rapid_planner_feature_emits_planner_service(
        self,
        managed_dgdr: ManagedDGDR,
        dgdr_factory,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_profiling_timeout: int,
    ) -> None:
        """Planner-enabled rapid profiling should include Planner service in generated DGD."""
        name = unique_dgdr_name("rapid-planner")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="trtllm",
            search_strategy="rapid",
            auto_apply=False,
            workload={"isl": 4000, "osl": 1000},
            sla={"ttft": 2000.0, "itl": 30.0},
            features={
                "planner": {
                    "enabled": True,
                    "plannerPreDeploymentSweeping": "rapid",
                }
            },
        )
        dgdr_factory(manifest)

        managed_dgdr.run(
            managed_dgdr.wait_for_phase(
                name, PHASE_READY, timeout=dgdr_profiling_timeout
            )
        )
        cm = _get_output_configmap(dgdr_namespace, name)
        dgd = _load_final_dgd_from_configmap(cm)

        services = dgd.get("spec", {}).get("services", {})
        assert "Planner" in services, "Planner service should exist in generated DGD"

    @pytest.mark.xfail(
        reason=(
            "Known issue: rapid mode can exceed DGDR totalGpus in generated DGD. "
            "Tracked in #8583, fix proposed in PR #8617."
        ),
        strict=False,
    )
    @pytest.mark.timeout(PROFILING_TEST_TIMEOUT_S)
    def test_rapid_generated_dgd_respects_total_gpus_budget(
        self,
        managed_dgdr: ManagedDGDR,
        dgdr_factory,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_profiling_timeout: int,
    ) -> None:
        """Generated DGD worker GPU requests should not exceed spec.hardware.totalGpus."""
        name = unique_dgdr_name("rapid-budget")
        total_gpus_budget = 32
        manifest = build_dgdr_manifest(
            name,
            model="Qwen/Qwen3-235B-A22B-FP8",
            image=dgdr_image,
            backend="trtllm",
            search_strategy="rapid",
            auto_apply=False,
            workload={"isl": 4000, "osl": 1000},
            sla={"ttft": 2000.0, "itl": 30.0},
            hardware={
                "gpuSku": "h200_sxm",
                "vramMb": 141120,
                "numGpusPerNode": 8,
                "totalGpus": total_gpus_budget,
            },
        )
        dgdr_factory(manifest)

        managed_dgdr.run(
            managed_dgdr.wait_for_phase(
                name, PHASE_READY, timeout=dgdr_profiling_timeout
            )
        )
        cm = _get_output_configmap(dgdr_namespace, name)
        dgd = _load_final_dgd_from_configmap(cm)

        requested = _total_worker_gpus(dgd)
        assert (
            requested <= total_gpus_budget
        ), f"Generated DGD requests {requested} GPUs, exceeds budget {total_gpus_budget}"
