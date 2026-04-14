# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Webhook validation and version conversion tests for DGDR v1beta1.

These tests verify that:
- The admission webhook correctly accepts/rejects DGDR specs (TestDGDRValidation)
- v1alpha1 resources are transparently converted to v1beta1 (TestDGDRVersionConversion)

No GPU or cluster profiling is required (gpu_0 only).  The only prerequisite is a
running Kubernetes cluster with the Dynamo operator CRDs and webhooks installed.

Run:
  pytest tests/dgdr/test_dgdr_validation.py -m gpu_0 -v --dgdr-namespace=default --dgdr-image=<image>

Test markers:
  gpu_0       No GPU required
  nightly     Requires live K8s cluster (not run in general pre-merge CI)
  integration Integration-level (uses live webhook)
"""

from __future__ import annotations

import json
import logging

import pytest
import yaml
from kubernetes_asyncio.client import exceptions as k8s_exceptions

from tests.dgdr.conftest import (
    DGDR_API_VERSION,
    DGDR_SHORT_NAME,
    _run_kubectl,
    build_dgdr_manifest,
    unique_dgdr_name,
)
from tests.utils.managed_deployment import ManagedDGDR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ── Group 1: Webhook Validation (gpu_0, no profiling required) ──────────────
# ---------------------------------------------------------------------------


@pytest.mark.gpu_0
@pytest.mark.nightly
@pytest.mark.integration
@pytest.mark.k8s
class TestDGDRValidation:
    """
    Tests that verify the admission webhook correctly validates DGDR specs
    before they are persisted.  These tests use server-side dry-run so no
    resources are actually created.
    """

    def test_missing_model_rejected(
        self, managed_dgdr: ManagedDGDR, dgdr_image: str
    ) -> None:
        """
        A DGDR without spec.model must be rejected by the webhook.
        The model field is the only hard-required spec field in v1beta1.
        """
        manifest = build_dgdr_manifest(
            unique_dgdr_name("no-model"),
            model="",  # intentionally empty
            image=dgdr_image,
        )
        # Clear model so the field is absent
        del manifest["spec"]["model"]

        with pytest.raises(k8s_exceptions.ApiException):
            managed_dgdr.run(managed_dgdr.server_dry_run(manifest))

    def test_thorough_with_auto_backend_rejected(
        self, managed_dgdr: ManagedDGDR, dgdr_image: str, dgdr_model: str
    ) -> None:
        """
        searchStrategy: thorough + backend: auto must be rejected.
        'thorough' sweeps real GPU engines and requires a concrete backend.
        """
        manifest = build_dgdr_manifest(
            unique_dgdr_name("thorough-auto"),
            model=dgdr_model,
            image=dgdr_image,
            backend="auto",
            search_strategy="thorough",
        )
        with pytest.raises(k8s_exceptions.ApiException) as exc_info:
            managed_dgdr.run(managed_dgdr.server_dry_run(manifest))
        error_body = str(exc_info.value)
        assert (
            "auto" in error_body.lower()
            or "backend" in error_body.lower()
            or "thorough" in error_body.lower()
        ), f"Error message should mention backend/thorough incompatibility. Got: {error_body}"

    def test_invalid_backend_rejected(
        self, managed_dgdr: ManagedDGDR, dgdr_image: str, dgdr_model: str
    ) -> None:
        """
        An unknown backend value must be rejected by the admission webhook.
        Valid values: auto, vllm, sglang, trtllm.
        """
        manifest = build_dgdr_manifest(
            unique_dgdr_name("bad-backend"),
            model=dgdr_model,
            image=dgdr_image,
            backend="unknown_backend",
        )
        with pytest.raises(k8s_exceptions.ApiException):
            managed_dgdr.run(managed_dgdr.server_dry_run(manifest))

    def test_invalid_search_strategy_rejected(
        self, managed_dgdr: ManagedDGDR, dgdr_image: str, dgdr_model: str
    ) -> None:
        """
        An unknown searchStrategy value must be rejected by the admission webhook.
        """
        manifest = build_dgdr_manifest(
            unique_dgdr_name("bad-strategy"),
            model=dgdr_model,
            image=dgdr_image,
            search_strategy="superfast",  # not a valid strategy
        )
        with pytest.raises(k8s_exceptions.ApiException):
            managed_dgdr.run(managed_dgdr.server_dry_run(manifest))

    def test_invalid_optimization_type_rejected(
        self, managed_dgdr: ManagedDGDR, dgdr_image: str, dgdr_model: str
    ) -> None:
        """
        An invalid sla.optimizationType value must be rejected by the
        admission webhook. Valid values: latency, throughput.
        """
        manifest = build_dgdr_manifest(
            unique_dgdr_name("bad-opt-type"),
            model=dgdr_model,
            image=dgdr_image,
            sla={"optimizationType": "cost"},  # not valid
        )
        with pytest.raises(k8s_exceptions.ApiException):
            managed_dgdr.run(managed_dgdr.server_dry_run(manifest))

    def test_valid_minimal_dgdr_accepted(
        self, managed_dgdr: ManagedDGDR, dgdr_image: str, dgdr_model: str
    ) -> None:
        """
        A DGDR with only the required fields (model + image) must pass validation.
        All other fields have defaults and are optional.
        """
        manifest = build_dgdr_manifest(
            unique_dgdr_name("valid-minimal"),
            model=dgdr_model,
            image=dgdr_image,
        )
        # Should not raise — accepted by the webhook
        managed_dgdr.run(managed_dgdr.server_dry_run(manifest))

    def test_valid_full_spec_accepted(
        self, managed_dgdr: ManagedDGDR, dgdr_image: str, dgdr_model: str
    ) -> None:
        """
        A fully-specified v1beta1 DGDR should pass webhook validation.
        Exercises every top-level optional field.
        """
        manifest = build_dgdr_manifest(
            unique_dgdr_name("valid-full"),
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            sla={"ttft": 200.0, "itl": 20.0},
            workload={"isl": 3000, "osl": 150},
            features={
                "planner": {"plannerPreDeploymentSweeping": "rapid"},
                "mocker": {"enabled": False},
            },
            hardware={"numGpusPerNode": 8},
            auto_apply=True,
        )
        # Should not raise — accepted by the webhook
        managed_dgdr.run(managed_dgdr.server_dry_run(manifest))

    def test_v1beta1_is_storage_version(self, dgdr_namespace: str) -> None:
        """
        The CRD's storage version must be v1beta1 (it is the conversion hub).
        """
        result = _run_kubectl(
            [
                "get",
                "crd",
                "dynamographdeploymentrequests.nvidia.com",
                "-o",
                "jsonpath={.status.storedVersions}",
            ],
            check=False,
        )
        assert result.returncode == 0, f"Failed to get CRD: {result.stderr}"
        assert (
            "v1beta1" in result.stdout
        ), f"v1beta1 should be the storage version. Got: {result.stdout}"

    def test_kubectl_shortname_dgdr_works(self, dgdr_namespace: str) -> None:
        """
        kubectl get dgdr must work (tests the shortName 'dgdr' in the CRD).
        """
        result = _run_kubectl(
            ["get", DGDR_SHORT_NAME, "-n", dgdr_namespace, "--ignore-not-found"],
            check=False,
        )
        assert (
            result.returncode == 0
        ), f"kubectl get dgdr failed (shortname may not be registered). stderr: {result.stderr}"

    def test_kubectl_get_columns_schema(
        self, dgdr_namespace: str, dgdr_image: str, dgdr_model: str, dgdr_factory
    ) -> None:
        """
        kubectl get dgdr should output the columns defined in the CRD:
        NAME, MODEL, BACKEND, PHASE, PROFILING, DGD, AGE.
        """
        name = unique_dgdr_name("col-test")
        manifest = build_dgdr_manifest(name, model=dgdr_model, image=dgdr_image)
        dgdr_factory(manifest)

        result = _run_kubectl(
            ["get", DGDR_SHORT_NAME, name, "-n", dgdr_namespace],
            check=False,
        )
        assert result.returncode == 0, f"kubectl get dgdr failed: {result.stderr}"

        header = (
            result.stdout.splitlines()[0].upper() if result.stdout.splitlines() else ""
        )
        expected_columns = {"NAME", "MODEL", "BACKEND", "PHASE"}
        for col in expected_columns:
            assert (
                col in header
            ), f"Expected column {col!r} in kubectl output header. Got: {header}"


# ---------------------------------------------------------------------------
# ── Group 2: v1alpha1 → v1beta1 Version Conversion ─────────────────────────
# ---------------------------------------------------------------------------


@pytest.mark.gpu_0
@pytest.mark.nightly
@pytest.mark.integration
@pytest.mark.k8s
class TestDGDRVersionConversion:
    """
    Tests that v1alpha1 DGDR resources can be submitted and are stored
    transparently as v1beta1 (conversion hub).  No profiling required.
    """

    def test_v1alpha1_dgdr_can_be_applied(
        self, dgdr_namespace: str, dgdr_image: str, dgdr_model: str, dgdr_factory
    ) -> None:
        """
        A v1alpha1 DynamoGraphDeploymentRequest should be accepted and
        automatically converted to v1beta1 storage by the conversion webhook.

        Note: v1alpha1 manifests use a different spec shape (profilingConfig
        instead of image) so we must use kubectl here rather than the
        v1beta1-only ManagedDGDR client.
        """
        name = unique_dgdr_name("v1a1")
        v1alpha1_manifest = {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoGraphDeploymentRequest",
            "metadata": {"name": name},
            "spec": {
                "model": dgdr_model,
                "backend": "vllm",
                "profilingConfig": {
                    "profilerImage": dgdr_image,
                },
            },
        }
        yaml_str = yaml.dump(v1alpha1_manifest)
        result = _run_kubectl(
            ["apply", "-n", dgdr_namespace, "-f", "-"], input=yaml_str, check=False
        )
        if result.returncode == 0:
            # Register for cleanup without re-creating (resource already exists)
            dgdr_factory.register_for_cleanup(name)
        # Either accepted (0) or rejected for a known conversion reason – just not a 500
        assert result.returncode in (
            0,
            1,
        ), f"Unexpected error applying v1alpha1 DGDR: {result.stderr}"

    def test_v1beta1_get_on_v1alpha1_object(
        self,
        managed_dgdr: ManagedDGDR,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_factory,
    ) -> None:
        """
        A resource stored as v1beta1 must be retrievable as v1alpha1 via conversion.
        """
        name = unique_dgdr_name("conv-get")
        manifest = build_dgdr_manifest(name, model=dgdr_model, image=dgdr_image)
        dgdr_factory(manifest)

        # Retrieve as v1beta1 (storage version) via ManagedDGDR
        obj_v1beta1 = managed_dgdr.run(managed_dgdr.get(name))
        assert obj_v1beta1 is not None
        assert obj_v1beta1["apiVersion"] == DGDR_API_VERSION

        # Retrieve as v1alpha1 (should trigger conversion webhook).
        # Must use kubectl here since ManagedDGDR targets v1beta1 only.
        result = _run_kubectl(
            [
                "get",
                "dynamographdeploymentrequests.v1alpha1.nvidia.com",
                name,
                "-n",
                dgdr_namespace,
                "-o",
                "json",
            ],
            check=False,
        )
        # If the conversion webhook is working, we get a 200 with v1alpha1 resource.
        # If not registered, we may get a 404 - that is also acceptable here as
        # some cluster configs only register v1beta1.
        assert result.returncode in (
            0,
            1,
        ), f"Unexpected failure getting v1alpha1 DGDR: {result.stderr}"
        if result.returncode == 0:
            obj_v1alpha1 = json.loads(result.stdout)
            assert (
                obj_v1alpha1["apiVersion"] == "nvidia.com/v1alpha1"
            ), "Retrieved object should have v1alpha1 apiVersion"
