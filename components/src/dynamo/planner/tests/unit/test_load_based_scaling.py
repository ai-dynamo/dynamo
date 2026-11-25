# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker info refresh tests of the planner environment.

FPM-driven scaling integration tests live in the plugin/orchestrator test suite.
"""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.types import TrafficObservation
from dynamo.planner.environment.base import PlannerEnvironmentImpl
from dynamo.planner.monitoring.dgd_services import ComponentGPUShape
from dynamo.planner.monitoring.worker_info import WorkerInfo

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ── Connector-driven refresh tests ──────────────────────────────────


class TestRefreshWorkerInfoFromConnector:
    """Tests for PlannerEnvironmentImpl worker info refresh.

    The tick-loop refresh delegates to the connector's ``get_worker_info``,
    which is where each connector implements its own MDC source (K8s CRDs
    for KubernetesConnector, discovery watch for VirtualConnector).  These
    tests exercise the shared refresh plumbing with a mock connector.
    """

    def _make_planner(self, require_prefill=False, require_decode=True):
        """Build a minimal NativePlannerBase with no_operation=True."""
        # Bypass Prometheus registration (Gauge+Enum double-register across
        # tests). KubernetesConnector.__init__ loads ~/.kube/config and reads
        # DYN_PARENT_DGD_K8S_NAME; stub both so this runs in plain pytest envs.
        with (
            patch("dynamo.planner.core.base.PlannerPrometheusMetrics") as mock_metrics,
            patch("dynamo.planner.connectors.kubernetes.KubernetesAPI"),
            patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "test-graph"}),
        ):
            mock_metrics.return_value = Mock()
            config = PlannerConfig.model_construct(
                throughput_adjustment_interval_seconds=60,
                prefill_engine_num_gpu=1,
                decode_engine_num_gpu=1,
                min_endpoint=1,
                max_gpu_budget=-1,
                ttft_ms=500.0,
                itl_ms=50.0,
                backend="vllm",
                no_operation=True,
                metric_pulling_prometheus_endpoint="http://localhost:9090",
                metric_reporting_prometheus_port=0,
                load_predictor="constant",
                environment="kubernetes",
                namespace="test-namespace",
                mode="agg",
                enable_load_scaling=True,
                enable_throughput_scaling=True,
                load_adjustment_interval_seconds=5,
                max_num_fpm_samples=50,
                fpm_sample_bucket_size=16,
                load_scaling_down_sensitivity=80,
                load_min_observations=5,
            )
            controller = Mock()
            controller.get_worker_info.return_value = WorkerInfo()
            controller.get_gpu_shapes.return_value = (
                ComponentGPUShape(1, 1),
                ComponentGPUShape(1, 1),
            )
            controller.get_actual_worker_counts = AsyncMock(return_value=(0, 0, True))
            controller.get_model_name.return_value = "test-model"
            environment = PlannerEnvironmentImpl(
                config=config,
                controller=controller,
                require_prefill=require_prefill,
                require_decode=require_decode,
            )
            environment.deployment_state().prefill.info = WorkerInfo()
            environment.deployment_state().decode.info = WorkerInfo()
            planner = NativePlannerBase(None, config, environment)
        planner.require_prefill = require_prefill
        planner.require_decode = require_decode
        return planner

    def _install_mock_connector(self, planner, **fresh_info_kwargs):
        """Replace environment's controller with a Mock returning fresh WorkerInfo."""
        fresh = WorkerInfo(**fresh_info_kwargs)
        mock_connector = Mock()
        mock_connector.get_worker_info.return_value = fresh
        mock_connector.get_gpu_shapes.return_value = (
            ComponentGPUShape(1, 1),
            ComponentGPUShape(1, 1),
        )
        mock_connector.get_actual_worker_counts = AsyncMock(return_value=(0, 0, True))
        mock_connector.get_model_name.return_value = "test-model"
        planner.environment.controller = mock_connector
        return mock_connector

    def _refresh_worker_info(self, planner):
        planner.environment._refresh_worker_info()

    def _decode_info(self, planner):
        return planner.environment.deployment_state().decode.info

    def _prefill_info(self, planner):
        return planner.environment.deployment_state().prefill.info

    def test_refresh_populates_missing_fields(self):
        """Connector returns a populated WorkerInfo; missing fields backfill."""
        planner = self._make_planner()
        assert self._decode_info(planner).max_num_batched_tokens is None

        self._install_mock_connector(
            planner,
            max_num_batched_tokens=8192,
            total_kv_blocks=1024,
            max_num_seqs=256,
            kv_cache_block_size=16,
            context_length=4096,
        )

        self._refresh_worker_info(planner)
        assert self._decode_info(planner).max_num_batched_tokens == 8192
        assert self._decode_info(planner).total_kv_blocks == 1024
        assert self._decode_info(planner).max_num_seqs == 256
        assert self._decode_info(planner).kv_cache_block_size == 16
        assert self._decode_info(planner).context_length == 4096

    def test_noop_when_already_set(self):
        """Does not re-query once max_num_batched_tokens is populated."""
        planner = self._make_planner()
        planner.environment.deployment_state().decode.info = WorkerInfo(
            max_num_batched_tokens=2048
        )

        mock_connector = self._install_mock_connector(
            planner, max_num_batched_tokens=8192
        )

        self._refresh_worker_info(planner)
        assert self._decode_info(planner).max_num_batched_tokens == 2048
        mock_connector.get_worker_info.assert_not_called()

    def test_noop_when_connector_lacks_get_worker_info(self):
        """Silently does nothing if the connector does not implement get_worker_info."""
        planner = self._make_planner()

        class _StubConnector:
            pass

        planner.environment.controller = _StubConnector()
        self._refresh_worker_info(planner)
        assert self._decode_info(planner).max_num_batched_tokens is None

    def test_noop_when_connector_returns_none_fields(self):
        """Fresh WorkerInfo with None everywhere does not overwrite anything."""
        planner = self._make_planner()
        self._install_mock_connector(planner)  # All Nones
        self._refresh_worker_info(planner)
        assert self._decode_info(planner).max_num_batched_tokens is None

    def test_exception_does_not_propagate(self):
        """If connector.get_worker_info throws, refresh is a no-op."""
        planner = self._make_planner()
        mock_connector = Mock()
        mock_connector.get_worker_info.side_effect = RuntimeError("boom")
        planner.environment.controller = mock_connector

        self._refresh_worker_info(planner)  # must not raise
        assert self._decode_info(planner).max_num_batched_tokens is None

    async def test_updates_engine_capabilities(self):
        """Builtin engine capabilities are refreshed after worker-info discovery."""
        planner = self._make_planner()
        engine = planner._ensure_engine()

        self._install_mock_connector(planner, max_num_batched_tokens=4096)

        await planner._refresh_and_update_capabilities()
        assert self._decode_info(planner).max_num_batched_tokens == 4096
        assert engine._scaling_state._capabilities.decode.max_num_batched_tokens == 4096

    def test_refresh_skips_unneeded_sub_component(self):
        """Only sub-components with require_* True are refreshed."""
        planner = self._make_planner(require_prefill=False, require_decode=True)

        def _side_effect(sub_type, backend):
            # Should only be called for DECODE.
            assert sub_type.value == "decode"
            return WorkerInfo(max_num_batched_tokens=4096)

        mock_connector = Mock()
        mock_connector.get_worker_info.side_effect = _side_effect
        planner.environment.controller = mock_connector

        self._refresh_worker_info(planner)
        assert self._prefill_info(planner).max_num_batched_tokens is None
        assert self._decode_info(planner).max_num_batched_tokens == 4096

    async def test_load_only_accept_length_missing_returns_observation(self):
        planner = self._make_planner(require_prefill=False, require_decode=True)
        planner.environment.collect_kv_hit_rate_observation = AsyncMock(
            return_value=TrafficObservation(
                duration_s=5,
                num_req=0,
                isl=0,
                osl=0,
                kv_hit_rate=None,
                accept_length=None,
            )
        )

        obs = await planner._collect_kv_hit_rate_observation(duration_s=5)

        assert obs is not None
        assert obs.kv_hit_rate is None
        assert obs.accept_length is None
        planner.environment.collect_kv_hit_rate_observation.assert_awaited_once_with(5)
