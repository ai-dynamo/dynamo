# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for all three planner actuation knobs:

Knob 1  Worker replica counts  — KubernetesConnector.set_component_replicas
        (replica scaling itself is covered in test_kubernetes_connector.py;
         this file covers the _apply_scaling_targets advisory guard.)

Knob 2  Per-GPU power cap (TGP)  — NativePlannerBase._apply_power_annotations
        → KubernetesAPI.patch_pod_annotation on each worker pod.

Knob 3  Admission-control thresholds  — KubernetesConnector.post_busy_threshold
        → HTTP POST /busy_threshold to each frontend pod with three sub-fields:
          active_decode_blocks_threshold  (θ_decode)
          active_prefill_tokens_threshold_frac  (θ_prefill_frac)
          active_prefill_tokens_threshold  (θ_prefill_abs, absolute defense-in-depth)
"""

import logging
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from kubernetes.client import ApiException

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.budget import POWER_ANNOTATION_KEY
from dynamo.planner.core.types import PlannerEffects, ScalingDecision
from dynamo.planner.errors import (
    DynamoGraphDeploymentNotFoundError,
    ModelNameNotFoundError,
    PlannerError,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_kube_api():
    api = Mock()
    api.get_graph_deployment = Mock()
    api.update_graph_replicas = Mock()
    api.is_deployment_ready = Mock(return_value=True)
    api.list_pods_by_label = Mock(return_value=[])
    api.patch_pod_annotation = Mock()
    api.remove_pod_annotation = Mock()
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


def _mock_pod(name: str, annotation_value=None) -> Mock:
    """Build a minimal mock Pod with controllable annotations."""
    pod = MagicMock()
    pod.metadata.name = name
    pod.metadata.annotations = (
        {POWER_ANNOTATION_KEY: annotation_value} if annotation_value is not None else {}
    )
    pod.status.pod_ip = f"10.0.0.{hash(name) % 200 + 1}"
    return pod


def _bare_planner(connector, config, require_prefill=True, require_decode=True):
    """Return a NativePlannerBase instance without calling __init__."""
    planner = object.__new__(NativePlannerBase)
    environment = Mock()
    environment.controller = connector
    environment.apply_scaling = AsyncMock()
    planner.environment = environment
    planner.config = config
    planner.require_prefill = require_prefill
    planner.require_decode = require_decode
    return planner


def _power_config(
    enable=True,
    prefill_cap=300,
    decode_cap=250,
    advisory=False,
    mode=None,
    backend=None,
):
    cfg = Mock()
    cfg.enable_power_awareness = enable
    cfg.prefill_engine_gpu_power_limit = prefill_cap
    cfg.decode_engine_gpu_power_limit = decode_cap
    cfg.advisory = advisory
    # mode/backend drive the sweep's component-name fallback. Default None so
    # WORKER_COMPONENT_NAMES.get(None) is None (component_name=None, resolution
    # by role only) — the behavior existing disagg tests rely on.
    cfg.mode = mode
    cfg.backend = backend
    return cfg


# ===========================================================================
# Knob 1 — Worker replicas (_apply_scaling_targets advisory guard)
# ===========================================================================


class TestApplyScalingTargets:
    """_apply_scaling_targets must be a no-op in advisory mode regardless of targets."""

    @pytest.mark.asyncio
    async def test_advisory_mode_skips_connector_call(self, connector):
        cfg = Mock()
        cfg.advisory = True
        planner = _bare_planner(connector, cfg)

        targets = [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                desired_replicas=4,
                component_name="VllmDecodeWorker",
            )
        ]
        await planner._apply_scaling_targets(targets, blocking=False)

        planner.environment.apply_scaling.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_advisory_mode_calls_connector(self, connector):
        cfg = Mock()
        cfg.advisory = False
        planner = _bare_planner(connector, cfg)

        targets = [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                desired_replicas=4,
                component_name="VllmDecodeWorker",
            )
        ]
        await planner._apply_scaling_targets(targets, blocking=False)

        planner.environment.apply_scaling.assert_awaited_once_with(
            targets, blocking=False
        )

    @pytest.mark.asyncio
    async def test_empty_targets_skips_connector_call(self, connector):
        cfg = Mock()
        cfg.advisory = False
        planner = _bare_planner(connector, cfg)

        await planner._apply_scaling_targets([], blocking=False)

        planner.environment.apply_scaling.assert_not_called()


# ===========================================================================
# Knob 2 — Per-GPU power cap (TGP) — pod annotation patching
# ===========================================================================


class TestPatchPodAnnotation:
    """KubernetesAPI.patch_pod_annotation must issue the correct core_api PATCH."""

    def test_patch_calls_core_api_with_correct_body(self, mock_kube_api):
        """Verify the PATCH body carries the annotation key/value pair."""
        mock_kube_api.current_namespace = "test-ns"

        # Call the real method body by directly invoking it on a bare API object.
        # core_api is already a Mock on mock_kube_api; just verify the call shape.
        from dynamo.planner.connectors.clients.kubernetes_api import KubernetesAPI

        api = object.__new__(KubernetesAPI)
        api.core_api = Mock()
        api.current_namespace = "test-ns"

        api.patch_pod_annotation("my-pod", POWER_ANNOTATION_KEY, "300")

        api.core_api.patch_namespaced_pod.assert_called_once_with(
            name="my-pod",
            namespace="test-ns",
            body={"metadata": {"annotations": {POWER_ANNOTATION_KEY: "300"}}},
            _content_type="application/merge-patch+json",
        )


class TestApplyPowerAnnotations:
    """NativePlannerBase._apply_power_annotations — the TGP patching knob."""

    @pytest.mark.asyncio
    async def test_patches_pod_when_annotation_absent(self, connector, mock_kube_api):
        """When a pod has no annotation, the planner must PATCH it."""
        pod = _mock_pod("worker-0")  # no annotation
        connector.get_component_pods = Mock(return_value=[pod])

        planner = _bare_planner(
            connector,
            _power_config(prefill_cap=300),
            require_prefill=True,
            require_decode=False,
        )
        await planner._apply_power_annotations()

        mock_kube_api.patch_pod_annotation.assert_called_once_with(
            "worker-0", POWER_ANNOTATION_KEY, "300"
        )

    @pytest.mark.asyncio
    async def test_patches_pod_when_annotation_is_stale(self, connector, mock_kube_api):
        """When annotation exists but has the wrong value, PATCH must update it."""
        pod = _mock_pod("worker-0", annotation_value="200")  # stale cap
        connector.get_component_pods = Mock(return_value=[pod])

        planner = _bare_planner(
            connector,
            _power_config(prefill_cap=350),
            require_prefill=True,
            require_decode=False,
        )
        await planner._apply_power_annotations()

        mock_kube_api.patch_pod_annotation.assert_called_once_with(
            "worker-0", POWER_ANNOTATION_KEY, "350"
        )

    @pytest.mark.asyncio
    async def test_skips_patch_when_annotation_already_correct(
        self, connector, mock_kube_api
    ):
        """No PATCH must be issued when the current annotation already matches."""
        pod = _mock_pod("worker-0", annotation_value="300")  # already correct
        connector.get_component_pods = Mock(return_value=[pod])

        planner = _bare_planner(
            connector,
            _power_config(prefill_cap=300),
            require_prefill=True,
            require_decode=False,
        )
        await planner._apply_power_annotations()

        mock_kube_api.patch_pod_annotation.assert_not_called()

    @pytest.mark.asyncio
    async def test_patches_prefill_and_decode_with_separate_limits(
        self, connector, mock_kube_api
    ):
        """Prefill and decode pods must receive their own per-component caps."""
        pod_p = _mock_pod("prefill-0")  # no annotation
        pod_d = _mock_pod("decode-0")  # no annotation

        def get_pods(sub_type, deployment=None, component_name=None):
            if sub_type == SubComponentType.PREFILL:
                return [pod_p]
            return [pod_d]

        connector.get_component_pods = Mock(side_effect=get_pods)

        planner = _bare_planner(
            connector,
            _power_config(prefill_cap=320, decode_cap=280),
            require_prefill=True,
            require_decode=True,
        )
        await planner._apply_power_annotations()

        calls = {
            c.args[0]: c.args[2]
            for c in mock_kube_api.patch_pod_annotation.call_args_list
        }
        assert calls["prefill-0"] == "320"
        assert calls["decode-0"] == "280"

    @pytest.mark.asyncio
    async def test_dual_role_sweep_reads_deployment_once(
        self, connector, mock_kube_api
    ):
        """A prefill+decode sweep must issue exactly one DGD GET (not one per role).

        Uses the real KubernetesConnector.get_component_pods so the shared-
        deployment plumbing is exercised end to end: 1 get_graph_deployment +
        2 list_pods_by_label, instead of the pre-optimization 2 + 2.
        """
        deployment = {
            "metadata": {"name": "test-dgd"},
            "spec": {
                "components": [
                    {"name": "VllmPrefillWorker", "type": "prefill"},
                    {"name": "VllmDecodeWorker", "type": "decode"},
                ]
            },
        }
        mock_kube_api.get_graph_deployment.return_value = deployment
        mock_kube_api.list_pods_by_label.return_value = []

        planner = _bare_planner(
            connector,
            _power_config(),
            require_prefill=True,
            require_decode=True,
        )
        await planner._apply_power_annotations()

        assert mock_kube_api.get_graph_deployment.call_count == 1
        assert mock_kube_api.list_pods_by_label.call_count == 2

    @pytest.mark.asyncio
    async def test_removes_stale_annotation_when_power_awareness_disabled(
        self, connector, mock_kube_api
    ):
        """enable_power_awareness=False must remove the annotation from pods that have it."""
        pod = _mock_pod("worker-0", annotation_value="300")
        connector.get_component_pods = Mock(return_value=[pod])

        planner = _bare_planner(
            connector,
            _power_config(enable=False),
            require_prefill=False,
            require_decode=True,
        )
        await planner._apply_power_annotations()

        connector.get_component_pods.assert_called_once()
        mock_kube_api.patch_pod_annotation.assert_not_called()
        mock_kube_api.remove_pod_annotation.assert_called_once_with(
            "worker-0", POWER_ANNOTATION_KEY
        )

    @pytest.mark.asyncio
    async def test_skips_removal_when_no_annotation_and_disabled(
        self, connector, mock_kube_api
    ):
        """Disabled cleanup must not issue unnecessary PATCH when pod has no annotation."""
        pod = _mock_pod("worker-0")  # no annotation
        connector.get_component_pods = Mock(return_value=[pod])

        planner = _bare_planner(
            connector,
            _power_config(enable=False),
            require_prefill=False,
            require_decode=True,
        )
        await planner._apply_power_annotations()

        connector.get_component_pods.assert_called_once()
        mock_kube_api.patch_pod_annotation.assert_not_called()
        mock_kube_api.remove_pod_annotation.assert_not_called()
        assert planner._disabled_power_removal_latched is True

    @pytest.mark.asyncio
    async def test_resolution_failure_does_not_latch_cleanup(
        self, connector, mock_kube_api
    ):
        """A component-resolution failure must not be mistaken for a clean sweep.

        get_component_pods(strict=True) raises on resolution failure so the
        removal path returns without latching; otherwise a renamed/invalid
        component would permanently disable cleanup and strand stale caps.
        """
        connector.get_component_pods = Mock(
            side_effect=PlannerError("could not resolve decode component")
        )

        planner = _bare_planner(
            connector,
            _power_config(enable=False),
            require_prefill=False,
            require_decode=True,
        )
        planner._disabled_power_removal_latched = False

        await planner._apply_power_annotations()

        # strict=True was requested so the failure is not masked as an empty list.
        assert connector.get_component_pods.call_args.kwargs["strict"] is True
        mock_kube_api.remove_pod_annotation.assert_not_called()
        assert planner._disabled_power_removal_latched is False

    @pytest.mark.asyncio
    async def test_disabled_latches_off_after_clean_removal_sweep(
        self, connector, mock_kube_api
    ):
        """After a clean disabled sweep, steady state must not re-list pods."""
        pod = _mock_pod("worker-0")
        connector.get_component_pods = Mock(return_value=[pod])

        planner = _bare_planner(
            connector,
            _power_config(enable=False),
            require_prefill=False,
            require_decode=True,
        )
        planner._last_power_annotation_sweep_s = 0.0
        planner._disabled_power_removal_latched = False
        planner.config.power_annotation_interval_seconds = 60.0

        await planner._apply_power_annotations()
        assert planner._disabled_power_removal_latched is True

        connector.get_component_pods.reset_mock()
        assert not planner._should_sweep_power_annotations(5000.0, PlannerEffects())
        connector.get_component_pods.assert_not_called()

    @pytest.mark.asyncio
    async def test_advisory_mode_skips_patch(self, connector, mock_kube_api):
        """advisory=True must produce zero PATCH calls even when annotations are stale.

        Pod annotations are a cluster-visible side effect on customer-owned
        Pod objects; advisory mode (PR #9683 review) must mirror the
        contract that ``_apply_scaling_targets`` already follows — log
        intent, mutate nothing.
        """
        pod = _mock_pod("worker-0")  # no annotation → would normally PATCH
        connector.get_component_pods = Mock(return_value=[pod])

        planner = _bare_planner(
            connector,
            _power_config(prefill_cap=300, advisory=True),
            require_prefill=True,
            require_decode=False,
        )
        await planner._apply_power_annotations()

        connector.get_component_pods.assert_not_called()
        mock_kube_api.patch_pod_annotation.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_entirely_for_non_kubernetes_connector(self, mock_kube_api):
        """Non-K8s connectors don't have patch_pod_annotation; method must bail early."""
        non_k8s_connector = Mock()  # not a KubernetesConnector instance

        planner = _bare_planner(
            non_k8s_connector,
            _power_config(),
            require_prefill=True,
            require_decode=False,
        )
        # Must not raise even though non_k8s_connector lacks kube_api
        await planner._apply_power_annotations()

        non_k8s_connector.get_component_pods.assert_not_called()

    @pytest.mark.asyncio
    async def test_patch_apierror_is_caught_not_raised(
        self, connector, mock_kube_api, caplog
    ):
        """A Kubernetes PATCH failure must log a warning and let the tick continue."""
        pod = _mock_pod("worker-0")  # no annotation
        connector.get_component_pods = Mock(return_value=[pod])
        # Production narrows the catch to ApiException (core/base.py), so the
        # injected failure must be the real Kubernetes patch error type.
        mock_kube_api.patch_pod_annotation.side_effect = ApiException(
            status=503, reason="k8s unavailable"
        )

        planner = _bare_planner(
            connector,
            _power_config(prefill_cap=300),
            require_prefill=True,
            require_decode=False,
        )

        with caplog.at_level(logging.WARNING):
            # Must not raise
            await planner._apply_power_annotations()

        assert any("worker-0" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_unexpected_patch_exception_propagates(
        self, connector, mock_kube_api
    ):
        """A non-ApiException must NOT be swallowed — the catch is intentionally
        narrow so unexpected reconciliation bugs surface instead of silently
        no-op'ing the sweep."""
        pod = _mock_pod("worker-0")  # no annotation
        connector.get_component_pods = Mock(return_value=[pod])
        mock_kube_api.patch_pod_annotation.side_effect = RuntimeError("unexpected bug")

        planner = _bare_planner(
            connector,
            _power_config(prefill_cap=300),
            require_prefill=True,
            require_decode=False,
        )

        with pytest.raises(RuntimeError, match="unexpected bug"):
            await planner._apply_power_annotations()

    @pytest.mark.asyncio
    async def test_sweep_skipped_when_dgd_read_fails(
        self, connector, mock_kube_api, caplog
    ):
        """A transient DGD-read ApiException must be swallowed, not propagated.

        The sweep runs inside run()'s try/finally loop; an escaping exception
        would shut the engine down and exit the planner. A transient apiserver
        error (5xx/timeout) on the DGD GET must instead log and skip so the
        next sweep can retry.
        """
        mock_kube_api.get_graph_deployment.side_effect = ApiException(
            status=503, reason="apiserver unavailable"
        )
        connector.get_component_pods = Mock()

        planner = _bare_planner(
            connector,
            _power_config(prefill_cap=300),
            require_prefill=True,
            require_decode=True,
        )

        with caplog.at_level(logging.WARNING):
            # Must not raise
            await planner._apply_power_annotations()

        # Bailed before touching pods; nothing patched.
        connector.get_component_pods.assert_not_called()
        mock_kube_api.patch_pod_annotation.assert_not_called()
        assert any("sweep skipped" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_sweep_skipped_when_dgd_not_found(
        self, connector, mock_kube_api, caplog
    ):
        """A DynamoGraphDeploymentNotFoundError (PlannerError, e.g. 404 blip)
        on the DGD read must also be swallowed, not propagated."""
        mock_kube_api.get_graph_deployment.side_effect = (
            DynamoGraphDeploymentNotFoundError(
                deployment_name="test-dgd", namespace="test-ns"
            )
        )

        planner = _bare_planner(
            connector,
            _power_config(prefill_cap=300),
            require_prefill=True,
            require_decode=False,
        )

        with caplog.at_level(logging.WARNING):
            await planner._apply_power_annotations()

        mock_kube_api.patch_pod_annotation.assert_not_called()
        assert any("sweep skipped" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_sweep_skipped_when_pod_listing_fails(
        self, connector, mock_kube_api, caplog
    ):
        """A transient pod-listing ApiException must be swallowed, not propagated.

        Exercises the real KubernetesConnector.get_component_pods path: the DGD
        read succeeds, but list_pods_by_label raises. The sweep must log and
        skip rather than tear down the planner.
        """
        mock_kube_api.get_graph_deployment.return_value = {
            "metadata": {"name": "test-dgd"},
            "spec": {
                "components": [
                    {"name": "VllmPrefillWorker", "type": "prefill"},
                ]
            },
        }
        mock_kube_api.list_pods_by_label.side_effect = ApiException(
            status=503, reason="apiserver unavailable"
        )

        planner = _bare_planner(
            connector,
            _power_config(prefill_cap=300),
            require_prefill=True,
            require_decode=False,
        )

        with caplog.at_level(logging.WARNING):
            # Must not raise
            await planner._apply_power_annotations()

        mock_kube_api.patch_pod_annotation.assert_not_called()
        assert any("sweep skipped" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_unexpected_read_exception_propagates(self, connector, mock_kube_api):
        """A read failure outside (ApiException, DynamoGraphDeploymentNotFound)
        must NOT be swallowed — the catch is intentionally narrow so unexpected
        reconciliation bugs surface instead of silently no-op'ing every
        sweep."""
        mock_kube_api.get_graph_deployment.side_effect = RuntimeError(
            "unexpected read bug"
        )

        planner = _bare_planner(
            connector,
            _power_config(prefill_cap=300),
            require_prefill=True,
            require_decode=False,
        )

        with pytest.raises(RuntimeError, match="unexpected read bug"):
            await planner._apply_power_annotations()

    @pytest.mark.asyncio
    async def test_unexpected_planner_error_propagates(self, connector, mock_kube_api):
        """A non-not-found PlannerError subclass must NOT be swallowed.

        The catch is narrowed to DynamoGraphDeploymentNotFoundError, not the
        PlannerError base — so a validation/model/component error (a real bug,
        not a transient read failure) surfaces instead of being masked as a
        skipped sweep.
        """
        mock_kube_api.get_graph_deployment.side_effect = ModelNameNotFoundError()

        planner = _bare_planner(
            connector,
            _power_config(prefill_cap=300),
            require_prefill=True,
            require_decode=False,
        )

        with pytest.raises(ModelNameNotFoundError):
            await planner._apply_power_annotations()

    @pytest.mark.asyncio
    async def test_multiple_pods_same_component_each_patched(
        self, connector, mock_kube_api
    ):
        """All worker pods for a component get their annotation patched."""
        pods = [_mock_pod(f"worker-{i}") for i in range(3)]
        connector.get_component_pods = Mock(return_value=pods)

        planner = _bare_planner(
            connector,
            _power_config(prefill_cap=300),
            require_prefill=True,
            require_decode=False,
        )
        await planner._apply_power_annotations()

        assert mock_kube_api.patch_pod_annotation.call_count == 3
        patched_names = {
            c.args[0] for c in mock_kube_api.patch_pod_annotation.call_args_list
        }
        assert patched_names == {"worker-0", "worker-1", "worker-2"}


class TestGetComponentPodsLabelSelector:
    """KubernetesConnector.get_component_pods must build the right label selector."""

    def test_prefill_label_selector(self, connector, mock_kube_api):
        """Prefill pods are filtered by nvidia.com/dynamo-component=<service-key>.

        Label names match the operator's KubeLabelDynamoGraphDeploymentName /
        KubeLabelDynamoComponent constants in deploy/operator/internal/consts/consts.go.
        """
        deployment = {
            "metadata": {"name": "test-dgd"},
            "spec": {
                "components": [
                    {"name": "VllmPrefillWorker", "replicas": 1, "type": "prefill"},
                ]
            },
        }
        mock_kube_api.get_graph_deployment.return_value = deployment
        mock_kube_api.list_pods_by_label.return_value = []

        connector.get_component_pods(SubComponentType.PREFILL)

        call_args = mock_kube_api.list_pods_by_label.call_args[0][0]
        assert "nvidia.com/dynamo-graph-deployment-name=test-dgd" in call_args
        assert "nvidia.com/dynamo-component=VllmPrefillWorker" in call_args

    def test_decode_label_selector(self, connector, mock_kube_api):
        """Decode pods get their own component-key label, not the prefill one."""
        deployment = {
            "metadata": {"name": "test-dgd"},
            "spec": {
                "components": [
                    {"name": "VllmDecodeWorker", "replicas": 1, "type": "decode"},
                ]
            },
        }
        mock_kube_api.get_graph_deployment.return_value = deployment
        mock_kube_api.list_pods_by_label.return_value = []

        connector.get_component_pods(SubComponentType.DECODE)

        call_args = mock_kube_api.list_pods_by_label.call_args[0][0]
        assert "nvidia.com/dynamo-component=VllmDecodeWorker" in call_args

    def test_missing_component_returns_empty_list(
        self, connector, mock_kube_api, caplog
    ):
        """If the DGD has no matching component, return [] AND log a warning.

        The [] return keeps the sweep non-fatal, but the warning is what makes
        an unexpected "0 pods annotated" diagnosable rather than silent.
        """
        deployment = {
            "metadata": {"name": "test-dgd"},
            "spec": {"components": []},
        }
        mock_kube_api.get_graph_deployment.return_value = deployment
        with caplog.at_level(logging.WARNING):
            result = connector.get_component_pods(SubComponentType.PREFILL)
        assert result == []
        assert any("could not resolve" in r.message.lower() for r in caplog.records)

    def test_strict_mode_raises_on_resolution_failure(self, connector, mock_kube_api):
        """strict=True must re-raise resolution failures instead of returning []."""
        deployment = {
            "metadata": {"name": "test-dgd"},
            "spec": {"components": []},
        }
        mock_kube_api.get_graph_deployment.return_value = deployment

        with pytest.raises(PlannerError):
            connector.get_component_pods(SubComponentType.PREFILL, strict=True)

        mock_kube_api.list_pods_by_label.assert_not_called()

    def test_reuses_passed_deployment_without_refetch(self, connector, mock_kube_api):
        """When given a deployment, get_component_pods must not GET it again."""
        deployment = {
            "metadata": {"name": "test-dgd"},
            "spec": {
                "components": [
                    {"name": "VllmPrefillWorker", "replicas": 1, "type": "prefill"},
                ]
            },
        }
        mock_kube_api.list_pods_by_label.return_value = []

        connector.get_component_pods(SubComponentType.PREFILL, deployment=deployment)

        mock_kube_api.get_graph_deployment.assert_not_called()
        call_args = mock_kube_api.list_pods_by_label.call_args[0][0]
        assert "nvidia.com/dynamo-component=VllmPrefillWorker" in call_args

    def test_agg_worker_resolves_unique_generic_worker(self, connector, mock_kube_api):
        """Aggregated (mode=agg) DGDs label their single worker `type: worker`,
        which does not map to the DECODE role. When it is the unique generic
        worker, the resolver should carry the actual DGD component name instead
        of depending on a backend hard-coded agg name.
        """
        deployment = {
            "metadata": {"name": "test-dgd"},
            "spec": {
                "components": [
                    # Real vLLM agg examples use both VllmDecodeWorker and
                    # VllmWorker for the single generic worker component.
                    {"name": "VllmDecodeWorker", "replicas": 1, "type": "worker"},
                ]
            },
        }
        mock_kube_api.list_pods_by_label.return_value = []

        connector.get_component_pods(SubComponentType.DECODE, deployment=deployment)
        call_args = mock_kube_api.list_pods_by_label.call_args[0][0]
        assert "nvidia.com/dynamo-component=VllmDecodeWorker" in call_args

        mock_kube_api.list_pods_by_label.reset_mock()

        # Even a stale/wrong fallback name must not override the DGD source of
        # truth when there is exactly one generic worker.
        connector.get_component_pods(
            SubComponentType.DECODE,
            deployment=deployment,
            component_name="VllmWorker",
        )
        call_args = mock_kube_api.list_pods_by_label.call_args[0][0]
        assert "nvidia.com/dynamo-component=VllmDecodeWorker" in call_args

    @pytest.mark.parametrize("worker_name", ["VllmDecodeWorker", "VllmWorker"])
    @pytest.mark.asyncio
    async def test_agg_mode_sweep_resolves_actual_generic_worker(
        self, connector, mock_kube_api, worker_name
    ):
        """End-to-end sweep for mode=agg must use the actual DGD worker name.

        vLLM agg manifests in this repo use both VllmDecodeWorker and
        VllmWorker for the single generic worker component.
        """
        deployment = {
            "metadata": {"name": "test-dgd"},
            "spec": {"components": [{"name": worker_name, "type": "worker"}]},
        }
        mock_kube_api.get_graph_deployment.return_value = deployment
        mock_kube_api.list_pods_by_label.return_value = []

        planner = _bare_planner(
            connector,
            _power_config(mode="agg", backend="vllm"),
            require_prefill=False,
            require_decode=True,
        )
        await planner._apply_power_annotations()

        call_args = mock_kube_api.list_pods_by_label.call_args[0][0]
        assert f"nvidia.com/dynamo-component={worker_name}" in call_args

    @pytest.mark.asyncio
    async def test_trtllm_agg_mode_sweep_resolves_trtllm_worker(
        self, connector, mock_kube_api
    ):
        """mode=agg + backend=trtllm must select the TRTLLMWorker component.

        The fallback ``decode_worker_k8s_name="decode"`` would miss it, so the
        resolver must select the unique generic worker from the DGD.
        """
        deployment = {
            "metadata": {"name": "test-dgd"},
            "spec": {"components": [{"name": "TRTLLMWorker", "type": "worker"}]},
        }
        mock_kube_api.get_graph_deployment.return_value = deployment
        mock_kube_api.list_pods_by_label.return_value = []

        planner = _bare_planner(
            connector,
            _power_config(mode="agg", backend="trtllm"),
            require_prefill=False,
            require_decode=True,
        )
        await planner._apply_power_annotations()

        mock_kube_api.list_pods_by_label.assert_called_once()
        call_args = mock_kube_api.list_pods_by_label.call_args[0][0]
        assert "nvidia.com/dynamo-component=TRTLLMWorker" in call_args

    @pytest.mark.asyncio
    async def test_sglang_agg_mode_sweep_resolves_decode_worker(
        self, connector, mock_kube_api
    ):
        """mode=agg + backend=sglang must select the decode component.

        SGLang v1beta1 agg DGDs name the single worker ``decode``, which
        matches the backend decode fallback, so this covers the explicit-name
        branch. The vLLM/TRT-LLM tests cover unique generic-worker resolution.
        """
        deployment = {
            "metadata": {"name": "test-dgd"},
            "spec": {"components": [{"name": "decode", "type": "worker"}]},
        }
        mock_kube_api.get_graph_deployment.return_value = deployment
        mock_kube_api.list_pods_by_label.return_value = []

        planner = _bare_planner(
            connector,
            _power_config(mode="agg", backend="sglang"),
            require_prefill=False,
            require_decode=True,
        )
        await planner._apply_power_annotations()

        mock_kube_api.list_pods_by_label.assert_called_once()
        call_args = mock_kube_api.list_pods_by_label.call_args[0][0]
        assert "nvidia.com/dynamo-component=decode" in call_args


# ===========================================================================
# Knob 3 — Admission-control thresholds (post_busy_threshold)
# ===========================================================================


class TestPostBusyThreshold:
    """KubernetesConnector.post_busy_threshold — the admission-control knob."""

    def _frontend_pod(self, pod_ip="10.0.0.5"):
        pod = MagicMock()
        pod.metadata.name = "frontend-pod-0"
        pod.status.pod_ip = pod_ip
        return pod

    @pytest.mark.asyncio
    async def test_all_three_fields_in_body(self, connector):
        """When all thresholds are set, all three must appear in the POST body."""
        pod = self._frontend_pod()

        mock_response = MagicMock()  # sync: raise_for_status() is not awaited
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await connector.post_busy_threshold(
                pod,
                model="Qwen/Qwen3-0.6B",
                port=8000,
                active_decode_blocks_threshold=0.8,
                active_prefill_tokens_threshold=512,
                active_prefill_tokens_threshold_frac=0.6,
            )

        body = mock_client.post.call_args.kwargs.get(
            "json"
        ) or mock_client.post.call_args[1].get("json")
        assert body["active_decode_blocks_threshold"] == pytest.approx(0.8)
        assert body["active_prefill_tokens_threshold"] == 512
        assert body["active_prefill_tokens_threshold_frac"] == pytest.approx(0.6)
        assert body["model"] == "Qwen/Qwen3-0.6B"

    @pytest.mark.asyncio
    async def test_url_includes_pod_ip_and_port(self, connector):
        """URL must be http://<pod_ip>:<port>/busy_threshold."""
        pod = self._frontend_pod(pod_ip="192.168.1.42")

        mock_response = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await connector.post_busy_threshold(
                pod,
                model="mymodel",
                port=9000,
                active_decode_blocks_threshold=0.7,
                active_prefill_tokens_threshold=None,
                active_prefill_tokens_threshold_frac=None,
            )

        url = mock_client.post.call_args[0][0]
        assert url == "http://192.168.1.42:9000/busy_threshold"

    @pytest.mark.asyncio
    async def test_url_brackets_ipv6_pod_ip(self, connector):
        """An IPv6 pod IP must be bracketed in the URL authority (RFC 3986)."""
        pod = self._frontend_pod(pod_ip="fd00::1234")

        mock_response = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await connector.post_busy_threshold(
                pod,
                model="mymodel",
                port=8000,
                active_decode_blocks_threshold=0.7,
                active_prefill_tokens_threshold=None,
                active_prefill_tokens_threshold_frac=None,
            )

        url = mock_client.post.call_args[0][0]
        assert url == "http://[fd00::1234]:8000/busy_threshold"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "field,value",
        [
            ("active_decode_blocks_threshold", None),
            ("active_prefill_tokens_threshold", None),
            ("active_prefill_tokens_threshold_frac", None),
        ],
    )
    async def test_none_field_omitted_from_body(self, connector, field, value):
        """None threshold fields must be omitted from the POST body entirely."""
        pod = self._frontend_pod()

        kwargs = dict(
            active_decode_blocks_threshold=0.8,
            active_prefill_tokens_threshold=512,
            active_prefill_tokens_threshold_frac=0.6,
        )
        kwargs[field] = None

        mock_response = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await connector.post_busy_threshold(pod, model="m", port=8000, **kwargs)

        body = mock_client.post.call_args.kwargs.get(
            "json"
        ) or mock_client.post.call_args[1].get("json")
        assert field not in body, (
            f"{field}=None must not appear in the POST body — "
            "the frontend treats missing fields as 'no change'"
        )

    @pytest.mark.asyncio
    async def test_pod_with_no_ip_raises_value_error(self, connector):
        """A pod with pod_ip=None must raise ValueError before any HTTP call."""
        pod = self._frontend_pod(pod_ip=None)
        pod.status.pod_ip = None

        with pytest.raises(ValueError, match="no pod IP"):
            await connector.post_busy_threshold(
                pod,
                model="m",
                port=8000,
                active_decode_blocks_threshold=0.8,
                active_prefill_tokens_threshold=None,
                active_prefill_tokens_threshold_frac=None,
            )

    @pytest.mark.asyncio
    async def test_http_error_propagates(self, connector):
        """An HTTP 4xx/5xx from the frontend must raise so the caller can count failures.

        raise_for_status() is a synchronous call on the response object, so the
        response mock must be a plain MagicMock (not AsyncMock) so that the
        side_effect fires on the synchronous call rather than being deferred to
        an awaitable coroutine.
        """
        import httpx

        pod = self._frontend_pod()

        # sync mock: raise_for_status() is called without await in post_busy_threshold
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503 Service Unavailable",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                await connector.post_busy_threshold(
                    pod,
                    model="m",
                    port=8000,
                    active_decode_blocks_threshold=0.7,
                    active_prefill_tokens_threshold=None,
                    active_prefill_tokens_threshold_frac=None,
                )

    @pytest.mark.asyncio
    async def test_model_always_in_body(self, connector):
        """The model field is unconditionally required in the POST body."""
        pod = self._frontend_pod()

        mock_response = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await connector.post_busy_threshold(
                pod,
                model="Llama-3-8B",
                port=8000,
                active_decode_blocks_threshold=None,
                active_prefill_tokens_threshold=None,
                active_prefill_tokens_threshold_frac=None,
            )

        body = mock_client.post.call_args.kwargs.get(
            "json"
        ) or mock_client.post.call_args[1].get("json")
        assert body.get("model") == "Llama-3-8B"


class TestListFrontendPods:
    """KubernetesConnector.list_frontend_pods must filter by the frontend component-type label."""

    def test_uses_frontend_component_type_label(self, connector, mock_kube_api):
        mock_kube_api.list_pods_by_label.return_value = []
        connector.list_frontend_pods()

        call_args = mock_kube_api.list_pods_by_label.call_args[0][0]
        assert "nvidia.com/dynamo-component-type=frontend" in call_args
        assert "nvidia.com/dynamo-graph-deployment-name=test-dgd" in call_args

    def test_returns_empty_when_no_frontend_pods(self, connector, mock_kube_api):
        mock_kube_api.list_pods_by_label.return_value = []
        result = connector.list_frontend_pods()
        assert result == []


# ===========================================================================
# Knob 2 — Per-GPU power cap (TGP) — sweep throttle / force-after-scale-up
# ===========================================================================


def _throttle_config(interval=60.0, enable=True, advisory=False):
    cfg = Mock()
    cfg.enable_power_awareness = enable
    cfg.advisory = advisory
    cfg.power_annotation_interval_seconds = interval
    return cfg


def _throttle_planner(config, p_workers=2, d_workers=2):
    """A bare planner wired only with the state _should_sweep... touches."""
    planner = object.__new__(NativePlannerBase)
    planner.config = config
    planner._last_power_annotation_sweep_s = 0.0
    planner._force_power_annotations_until_s = 0.0
    planner._disabled_power_removal_latched = False
    # Orchestrator-only path: current worker counts come from the cached tick
    # input (``_last_worker_counts``), the single source ``_current_worker_counts``
    # reads. The legacy PSM fallback was removed, so ``_state_machine`` is left
    # unset (None); tests that assert it is never consulted populate it locally.
    planner._last_worker_counts = Mock(
        ready_num_prefill=p_workers, ready_num_decode=d_workers
    )
    planner._state_machine = None
    return planner


class TestPowerAnnotationSweepThrottle:
    """NativePlannerBase._should_sweep_power_annotations — apiserver-load throttle.

    A sweep costs a DGD read + a pod list per managed component, so steady-state
    sweeps are throttled to power_annotation_interval_seconds while scale-ups
    force a prompt pass so new pods are annotated without waiting out the
    throttle.
    """

    def test_first_tick_sweeps(self):
        planner = _throttle_planner(_throttle_config(interval=60.0))
        assert planner._should_sweep_power_annotations(1000.0, PlannerEffects())
        assert planner._last_power_annotation_sweep_s == 1000.0

    def test_throttled_within_interval(self):
        planner = _throttle_planner(_throttle_config(interval=60.0))
        assert planner._should_sweep_power_annotations(1000.0, PlannerEffects())
        # 30s later, no scaling — must be throttled, bookkeeping unchanged.
        assert not planner._should_sweep_power_annotations(1030.0, PlannerEffects())
        assert planner._last_power_annotation_sweep_s == 1000.0

    def test_sweeps_again_after_interval(self):
        planner = _throttle_planner(_throttle_config(interval=60.0))
        assert planner._should_sweep_power_annotations(1000.0, PlannerEffects())
        assert planner._should_sweep_power_annotations(1061.0, PlannerEffects())
        assert planner._last_power_annotation_sweep_s == 1061.0

    def test_scale_up_forces_sweep_inside_throttle(self):
        planner = _throttle_planner(_throttle_config(interval=60.0), p_workers=2)
        # Establish a recent steady-state sweep.
        assert planner._should_sweep_power_annotations(1000.0, PlannerEffects())
        scale_up = PlannerEffects(scale_to=ScalingDecision(num_prefill=4))
        # 5s later (well inside the 60s throttle) a scale-up still forces a sweep.
        assert planner._should_sweep_power_annotations(1005.0, scale_up)

    def test_force_window_persists_after_scale_up(self):
        planner = _throttle_planner(_throttle_config(interval=60.0), d_workers=1)
        planner._should_sweep_power_annotations(1000.0, PlannerEffects())
        scale_up = PlannerEffects(scale_to=ScalingDecision(num_decode=3))
        assert planner._should_sweep_power_annotations(1005.0, scale_up)
        # Subsequent ticks with no new decision still sweep until the window
        # (opened at 1005 + 60 = 1065) closes — catches pods created post-scale.
        assert planner._should_sweep_power_annotations(1020.0, PlannerEffects())
        assert planner._should_sweep_power_annotations(1064.0, PlannerEffects())
        assert not planner._should_sweep_power_annotations(1066.0, PlannerEffects())

    def test_scale_down_does_not_force(self):
        planner = _throttle_planner(_throttle_config(interval=60.0), p_workers=4)
        planner._should_sweep_power_annotations(1000.0, PlannerEffects())
        scale_down = PlannerEffects(scale_to=ScalingDecision(num_prefill=1))
        # No new pods on scale-down — throttle still applies inside the interval.
        assert not planner._should_sweep_power_annotations(1010.0, scale_down)

    def test_hold_decision_does_not_force(self):
        planner = _throttle_planner(_throttle_config(interval=60.0), p_workers=2)
        planner._should_sweep_power_annotations(1000.0, PlannerEffects())
        hold = PlannerEffects(scale_to=ScalingDecision(num_prefill=2))
        assert not planner._should_sweep_power_annotations(1010.0, hold)

    def test_advisory_never_sweeps(self):
        planner = _throttle_planner(_throttle_config(interval=60.0, advisory=True))
        # Interval elapsed and a scale-up present, yet advisory mutates nothing.
        scale_up = PlannerEffects(scale_to=ScalingDecision(num_prefill=4))
        assert not planner._should_sweep_power_annotations(5000.0, scale_up)

    def test_disabled_sweeps_on_throttle_until_latched(self):
        """Disabled mode sweeps on throttle until a clean removal pass latches off."""
        planner = _throttle_planner(_throttle_config(interval=60.0, enable=False))
        assert planner._should_sweep_power_annotations(5000.0, PlannerEffects())
        assert not planner._should_sweep_power_annotations(5030.0, PlannerEffects())
        planner._disabled_power_removal_latched = True
        assert not planner._should_sweep_power_annotations(5061.0, PlannerEffects())

    def test_disabled_scale_up_does_not_open_force_window(self):
        """A scale-up decision must not extend the force window when disabled."""
        planner = _throttle_planner(_throttle_config(interval=60.0, enable=False))
        planner._should_sweep_power_annotations(1000.0, PlannerEffects())
        scale_up = PlannerEffects(scale_to=ScalingDecision(num_prefill=4))
        # Inside throttle but with a scale-up — disabled, so no force window opened.
        assert not planner._should_sweep_power_annotations(1010.0, scale_up)
        assert planner._force_power_annotations_until_s == 0.0

    def test_scale_up_uses_cached_counts_not_psm(self):
        # Cached worker_counts present -> _scaling_up must read them, NOT the
        # PSM internals. Stale PSM counts (99) would mask a real scale-up.
        planner = _throttle_planner(_throttle_config(interval=60.0))
        planner._last_worker_counts = Mock(ready_num_prefill=2, ready_num_decode=2)
        planner._state_machine = Mock(_num_p_workers=99, _num_d_workers=99)
        assert planner._should_sweep_power_annotations(1000.0, PlannerEffects())
        scale_up = PlannerEffects(scale_to=ScalingDecision(num_prefill=4))
        # 4 > cached 2 -> force; had it read PSM's 99, 4 > 99 would be False.
        assert planner._should_sweep_power_annotations(1005.0, scale_up)

    def test_orchestrator_path_uses_cached_counts_without_building_psm(self):
        # Orchestrator path has no PSM. _scaling_up must rely on the cached
        # counts and must NOT touch the state_machine property (which would
        # lazily construct a PSM + warm predictors).
        planner = _throttle_planner(_throttle_config(interval=60.0))
        planner._state_machine = None
        planner._last_worker_counts = Mock(ready_num_prefill=1, ready_num_decode=1)
        planner._should_sweep_power_annotations(1000.0, PlannerEffects())
        scale_up = PlannerEffects(scale_to=ScalingDecision(num_decode=3))
        assert planner._should_sweep_power_annotations(1005.0, scale_up)
        # No PSM was constructed as a side effect.
        assert planner._state_machine is None

    def test_cached_counts_above_decision_do_not_force(self):
        # Cached counts already at/above the decision -> not a scale-up.
        planner = _throttle_planner(_throttle_config(interval=60.0))
        planner._last_worker_counts = Mock(ready_num_prefill=4, ready_num_decode=4)
        planner._state_machine = None
        planner._should_sweep_power_annotations(1000.0, PlannerEffects())
        not_up = PlannerEffects(scale_to=ScalingDecision(num_prefill=4, num_decode=2))
        assert not planner._should_sweep_power_annotations(1010.0, not_up)
