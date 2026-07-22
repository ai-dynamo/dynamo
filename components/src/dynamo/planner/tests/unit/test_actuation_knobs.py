# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Power actuation is DGD-owned: read/resolve, never mutate.

The planner reads per-GPU caps from the DGD worker podTemplate annotations and
projects a power budget. It does NOT patch Pods. These tests pin both halves of
that contract:

* the read/resolve path — ``KubernetesConnector.get_component_power_configs``
  resolves per-role ``ComponentPowerConfig`` from a DGD dict (disagg + agg) and
  propagates the typed parser errors; and
* the no-mutation guarantee — the connector, the Kubernetes API client, and the
  planner base expose no Pod-write / power-sweep surface at all.
"""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.connectors.clients.kubernetes_api import KubernetesAPI
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.errors import (
    PowerAnnotationInvalidError,
    PowerAnnotationMissingError,
    SubComponentNotFoundError,
)
from dynamo.planner.monitoring.dgd_services import POWER_ANNOTATION_KEY

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture
def mock_kube_api():
    api = Mock()
    api.get_graph_deployment = Mock()
    api.wait_for_graph_deployment_ready = AsyncMock()
    return api


@pytest.fixture
def connector(mock_kube_api, monkeypatch):
    """A KubernetesConnector with all K8s calls mocked out."""
    monkeypatch.setattr(
        "dynamo.planner.connectors.kubernetes.KubernetesAPI",
        Mock(return_value=mock_kube_api),
    )
    with patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "test-dgd"}):
        return KubernetesConnector("test-ns")


def _worker(name, comp_type, watts, gpus):
    component = {
        "name": name,
        "type": comp_type,
        "podTemplate": {
            "spec": {
                "containers": [
                    {"name": "main", "resources": {"limits": {"nvidia.com/gpu": gpus}}}
                ]
            }
        },
    }
    if watts is not None:
        component["podTemplate"]["metadata"] = {
            "annotations": {POWER_ANNOTATION_KEY: watts}
        }
    return component


def _dgd(*components):
    return {"spec": {"components": list(components)}}


# ---------------------------------------------------------------------------
# Read / resolve path
# ---------------------------------------------------------------------------


class TestGetComponentPowerConfigs:
    def test_resolves_disagg_from_annotations(self, connector, mock_kube_api):
        mock_kube_api.get_graph_deployment.return_value = _dgd(
            _worker("VllmPrefillWorker", "prefill", "350", "2"),
            _worker("VllmDecodeWorker", "decode", "300", "4"),
        )

        prefill, decode = connector.get_component_power_configs(
            require_prefill=True, require_decode=True
        )

        assert prefill.watts_per_replica == 700  # 2 GPU × 350 W
        assert decode.watts_per_replica == 1200  # 4 GPU × 300 W

    def test_resolves_agg_decode_only(self, connector, mock_kube_api):
        mock_kube_api.get_graph_deployment.return_value = _dgd(
            _worker("VllmWorker", "worker", "300", "4"),
        )

        prefill, decode = connector.get_component_power_configs(
            require_prefill=False, require_decode=True
        )

        assert prefill is None
        assert decode.watts_per_replica == 1200

    def test_missing_annotation_propagates(self, connector, mock_kube_api):
        mock_kube_api.get_graph_deployment.return_value = _dgd(
            _worker("VllmDecodeWorker", "decode", None, "4"),
        )
        with pytest.raises(PowerAnnotationMissingError):
            connector.get_component_power_configs(
                require_prefill=False, require_decode=True
            )

    def test_malformed_annotation_propagates(self, connector, mock_kube_api):
        mock_kube_api.get_graph_deployment.return_value = _dgd(
            _worker("VllmDecodeWorker", "decode", "0", "4"),
        )
        with pytest.raises(PowerAnnotationInvalidError):
            connector.get_component_power_configs(
                require_prefill=False, require_decode=True
            )

    def test_missing_role_propagates(self, connector, mock_kube_api):
        mock_kube_api.get_graph_deployment.return_value = _dgd(
            _worker("VllmDecodeWorker", "decode", "300", "4"),
        )
        with pytest.raises(SubComponentNotFoundError):
            connector.get_component_power_configs(
                require_prefill=True, require_decode=True
            )


# ---------------------------------------------------------------------------
# No-mutation contract — the write surface must not exist anywhere.
# ---------------------------------------------------------------------------


class TestNoPodMutationSurface:
    @pytest.mark.parametrize(
        "attr",
        [
            "patch_pod_annotation",
            "remove_pod_annotation",
            "list_pods_by_label",
        ],
    )
    def test_kubernetes_api_has_no_pod_write_methods(self, attr):
        assert not hasattr(KubernetesAPI, attr), (
            f"KubernetesAPI.{attr} must not exist: the planner never writes Pod "
            "annotations under the DGD-owned power model."
        )

    @pytest.mark.parametrize(
        "attr",
        [
            "get_component_pods",
            "list_frontend_pods",
            "post_busy_threshold",
            "get_replica_gpu_counts_for_power_projection",
        ],
    )
    def test_connector_has_no_pod_listing_or_patch_methods(self, attr):
        assert not hasattr(KubernetesConnector, attr), (
            f"KubernetesConnector.{attr} must not exist: superseded Pod-sweep / "
            "admission scaffolding was removed with the DGD-owned power rework."
        )

    @pytest.mark.parametrize(
        "attr",
        [
            "_apply_power_annotations",
            "_run_power_annotation_sweep",
            "_run_power_annotation_removal",
            "_should_sweep_power_annotations",
            "_resolve_power_projection_gpu_counts",
        ],
    )
    def test_planner_base_has_no_power_sweep_methods(self, attr):
        assert not hasattr(NativePlannerBase, attr), (
            f"NativePlannerBase.{attr} must not exist: the planner reads caps "
            "from the DGD and never sweeps/patches Pods."
        )

    def test_connector_still_exposes_the_read_method(self, connector):
        # The one supported power surface is read-only.
        assert callable(connector.get_component_power_configs)
        assert SubComponentType.PREFILL  # sanity import touch
