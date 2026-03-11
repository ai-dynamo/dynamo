# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profiler config_modifiers/protocol helpers."""

import pytest

from dynamo.profiler.utils.config_modifiers.protocol import apply_dgd_overrides

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
]


def test_apply_dgd_overrides_strips_envelope() -> None:
    """Envelope fields are stripped; nested payload keys are deep-merged."""
    dgd_config = {
        "apiVersion": "dynamo.ai/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "my-deployment", "namespace": "default"},
        "spec": {
            "services": {
                "Frontend": {"replicas": 1},
            }
        },
    }
    overrides = {
        # Envelope fields — must be stripped entirely.
        "apiVersion": "dynamo.ai/v1beta1",
        "kind": "SomethingElse",
        # metadata identity keys must be stripped; labels/annotations kept.
        "metadata": {
            "name": "injected-name",
            "namespace": "injected-ns",
            "uid": "abc-123",
            "resourceVersion": "999",
            "labels": {"team": "infra"},
            "annotations": {"note": "perf-run"},
        },
        # Regular payload key — must be deep-merged.
        "spec": {
            "services": {
                "Frontend": {"replicas": 3},
            }
        },
    }

    result = apply_dgd_overrides(dgd_config, overrides)

    # apiVersion and kind must not be changed.
    assert result["apiVersion"] == "dynamo.ai/v1alpha1"
    assert result["kind"] == "DynamoGraphDeployment"

    # Identity metadata keys must not be overwritten.
    assert result["metadata"]["name"] == "my-deployment"
    assert result["metadata"]["namespace"] == "default"
    assert "uid" not in result["metadata"]
    assert "resourceVersion" not in result["metadata"]

    # Safe metadata keys must be merged in.
    assert result["metadata"]["labels"] == {"team": "infra"}
    assert result["metadata"]["annotations"] == {"note": "perf-run"}

    # Regular spec overrides must be applied.
    assert result["spec"]["services"]["Frontend"]["replicas"] == 3

    # Original dicts must not be mutated.
    assert dgd_config["apiVersion"] == "dynamo.ai/v1alpha1"
    assert dgd_config["spec"]["services"]["Frontend"]["replicas"] == 1


def test_apply_dgd_overrides_no_metadata_in_overrides() -> None:
    """When overrides contain no metadata key, existing metadata is untouched."""
    dgd_config = {
        "apiVersion": "dynamo.ai/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "svc", "namespace": "ns"},
        "spec": {"services": {"Backend": {"replicas": 2}}},
    }
    overrides = {"spec": {"services": {"Backend": {"replicas": 5}}}}

    result = apply_dgd_overrides(dgd_config, overrides)

    assert result["metadata"] == {"name": "svc", "namespace": "ns"}
    assert result["spec"]["services"]["Backend"]["replicas"] == 5


def test_apply_dgd_overrides_metadata_only_identity_keys_dropped_entirely() -> None:
    """If metadata override contains only identity keys, nothing is merged into metadata."""
    dgd_config = {
        "apiVersion": "dynamo.ai/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "svc"},
        "spec": {},
    }
    overrides = {
        "metadata": {"name": "other", "namespace": "other-ns", "uid": "x"},
    }

    result = apply_dgd_overrides(dgd_config, overrides)

    # Only original metadata should remain — no extra keys added.
    assert result["metadata"] == {"name": "svc"}


def test_apply_dgd_overrides_extrapodspec_tolerations() -> None:
    """extraPodSpec.tolerations from overrides are merged into existing services.

    Regression test for TC-5.2a: interpolation DGDs were deployed without
    tolerations because apply_dgd_overrides was called after run_interpolation.
    This test verifies the merge logic itself is correct.
    """
    toleration = {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
    dgd_config = {
        "spec": {
            "services": {
                "VllmDecodeWorker": {
                    "componentType": "worker",
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "my-image",
                            "args": ["--model", "Qwen3-32B"],
                        }
                    },
                    "replicas": 1,
                },
                "Frontend": {
                    "extraPodSpec": {},
                },
            }
        }
    }
    overrides = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "placeholder"},
        "spec": {
            "services": {
                "VllmDecodeWorker": {"extraPodSpec": {"tolerations": [toleration]}},
                "Frontend": {"extraPodSpec": {"tolerations": [toleration]}},
            }
        },
    }

    result = apply_dgd_overrides(dgd_config, overrides)

    # Tolerations must be present on both services.
    decode_eps = result["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"]
    assert decode_eps["tolerations"] == [toleration]
    # mainContainer must be preserved (not overwritten).
    assert decode_eps["mainContainer"]["image"] == "my-image"

    frontend_eps = result["spec"]["services"]["Frontend"]["extraPodSpec"]
    assert frontend_eps["tolerations"] == [toleration]

    # Original must not be mutated.
    assert (
        "tolerations"
        not in dgd_config["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"]
    )


def test_apply_dgd_overrides_missing_service_skipped_with_warning() -> None:
    """Overrides for services absent from the DGD are skipped (warns, no crash)."""
    dgd_config = {
        "spec": {
            "services": {
                "Frontend": {"replicas": 1},
            }
        }
    }
    overrides = {
        "spec": {
            "services": {
                "Frontend": {"replicas": 2},
                "NonExistentWorker": {
                    "extraPodSpec": {"tolerations": [{"key": "foo"}]}
                },
            }
        }
    }

    result = apply_dgd_overrides(dgd_config, overrides)

    assert result["spec"]["services"]["Frontend"]["replicas"] == 2
    assert "NonExistentWorker" not in result["spec"]["services"]
