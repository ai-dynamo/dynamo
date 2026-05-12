# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.budget import POWER_ANNOTATION_KEY

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
    planner.connector = connector
    planner.config = config
    planner.require_prefill = require_prefill
    planner.require_decode = require_decode
    return planner


def _power_config(
    enable=True,
    prefill_cap=300,
    decode_cap=250,
    advisory=False,
):
    cfg = Mock()
    cfg.enable_power_awareness = enable
    cfg.prefill_engine_gpu_power_limit = prefill_cap
    cfg.decode_engine_gpu_power_limit = decode_cap
    cfg.advisory = advisory
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
        connector.set_component_replicas = AsyncMock()

        targets = [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                desired_replicas=4,
                component_name="VllmDecodeWorker",
            )
        ]
        await planner._apply_scaling_targets(targets, blocking=False)

        connector.set_component_replicas.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_advisory_mode_calls_connector(self, connector):
        cfg = Mock()
        cfg.advisory = False
        planner = _bare_planner(connector, cfg)
        connector.set_component_replicas = AsyncMock()

        targets = [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                desired_replicas=4,
                component_name="VllmDecodeWorker",
            )
        ]
        await planner._apply_scaling_targets(targets, blocking=False)

        connector.set_component_replicas.assert_called_once_with(
            targets, blocking=False
        )

    @pytest.mark.asyncio
    async def test_empty_targets_skips_connector_call(self, connector):
        cfg = Mock()
        cfg.advisory = False
        planner = _bare_planner(connector, cfg)
        connector.set_component_replicas = AsyncMock()

        await planner._apply_scaling_targets([], blocking=False)

        connector.set_component_replicas.assert_not_called()


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
        from dynamo.planner.connectors.kubernetes_api import KubernetesAPI

        api = object.__new__(KubernetesAPI)
        api.core_api = Mock()
        api.current_namespace = "test-ns"

        api.patch_pod_annotation("my-pod", POWER_ANNOTATION_KEY, "300")

        api.core_api.patch_namespaced_pod.assert_called_once_with(
            name="my-pod",
            namespace="test-ns",
            body={"metadata": {"annotations": {POWER_ANNOTATION_KEY: "300"}}},
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

        def get_pods(sub_type):
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
    async def test_skips_entirely_when_power_awareness_disabled(
        self, connector, mock_kube_api
    ):
        """enable_power_awareness=False must be a hard no-op."""
        pod = _mock_pod("worker-0")
        connector.get_component_pods = Mock(return_value=[pod])

        planner = _bare_planner(
            connector,
            _power_config(enable=False),
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
    async def test_patch_exception_is_caught_not_raised(
        self, connector, mock_kube_api, caplog
    ):
        """A PATCH failure must log a warning and let the tick continue."""
        import logging

        pod = _mock_pod("worker-0")  # no annotation
        connector.get_component_pods = Mock(return_value=[pod])
        mock_kube_api.patch_pod_annotation.side_effect = RuntimeError("k8s unavailable")

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
                "services": {
                    "VllmPrefillWorker": {"replicas": 1, "subComponentType": "prefill"},
                }
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
                "services": {
                    "VllmDecodeWorker": {"replicas": 1, "subComponentType": "decode"},
                }
            },
        }
        mock_kube_api.get_graph_deployment.return_value = deployment
        mock_kube_api.list_pods_by_label.return_value = []

        connector.get_component_pods(SubComponentType.DECODE)

        call_args = mock_kube_api.list_pods_by_label.call_args[0][0]
        assert "nvidia.com/dynamo-component=VllmDecodeWorker" in call_args

    def test_missing_component_returns_empty_list(self, connector, mock_kube_api):
        """If the DGD has no matching component, return [] instead of raising."""
        deployment = {
            "metadata": {"name": "test-dgd"},
            "spec": {"services": {}},
        }
        mock_kube_api.get_graph_deployment.return_value = deployment

        result = connector.get_component_pods(SubComponentType.PREFILL)

        assert result == []


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
