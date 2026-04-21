# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.encode import EncodePlanner
from dynamo.planner.core.state import PlannerSharedState
from dynamo.planner.monitoring.traffic_metrics import Metrics

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture(autouse=True)
def mock_prometheus_metrics():
    with patch("dynamo.planner.monitoring.planner_metrics.Gauge") as mock_gauge:
        mock_gauge.return_value = Mock()
        yield


def _build_config(**overrides) -> PlannerConfig:
    defaults = dict(
        throughput_adjustment_interval=60,
        min_endpoint=1,
        max_gpu_budget=-1,
        backend="vllm",
        no_operation=True,
        no_correction=True,
        metric_pulling_prometheus_endpoint="http://localhost:9090",
        metric_reporting_prometheus_port=0,
        load_predictor="constant",
        load_predictor_warmup_trace=None,
        load_predictor_log1p=False,
        profile_results_dir=os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "profiling_results",
            "H200_encode_TP1",
        ),
        environment="kubernetes",
        namespace="test-namespace",
        mode="encode",
        model_name="test-model",
        encode_engine_num_gpu=2,
        enable_throughput_scaling=True,
        enable_load_scaling=False,
    )
    defaults.update(overrides)
    return PlannerConfig.model_construct(**defaults)


def _build_planner(**config_overrides) -> EncodePlanner:
    config = _build_config(**config_overrides)
    prometheus_client = Mock()
    planner = EncodePlanner(
        None,
        config,
        shared_state=PlannerSharedState(),
        prometheus_traffic_client=prometheus_client,
        start_prometheus_server=False,
    )
    planner.encode_interpolator = Mock()
    planner.num_req_predictor = Mock()
    planner.isl_predictor = Mock()
    planner.osl_predictor = Mock()
    planner.prometheus_traffic_client = prometheus_client
    planner.model_name = "test-model"
    return planner


def test_encode_planner_constructs_in_encode_mode():
    planner = _build_planner()
    assert planner.component_type == SubComponentType.ENCODE


def test_encode_planner_replica_calculation_basic():
    planner = _build_planner()
    planner.encode_interpolator.interpolate_thpt_per_gpu.return_value = 1.5

    desired = planner._compute_replica_requirements(360, 0, 0)

    assert desired == 2


def test_encode_planner_respects_min_endpoint():
    planner = _build_planner(min_endpoint=3)
    planner.encode_interpolator.interpolate_thpt_per_gpu.return_value = 10.0

    desired = planner._compute_replica_requirements(60, 0, 0)

    assert desired == 3


def test_encode_planner_zero_throughput_falls_back_to_min_endpoint():
    planner = _build_planner(min_endpoint=2)
    planner.encode_interpolator.interpolate_thpt_per_gpu.return_value = 0.0

    desired = planner._compute_replica_requirements(120, 0, 0)

    assert desired == 2


def test_encode_planner_updates_only_request_predictor():
    planner = _build_planner()

    planner.update_predictors_from_metrics(Metrics(num_req=120, isl=99.0, osl=77.0))

    planner.num_req_predictor.add_data_point.assert_called_once_with(120)
    planner.isl_predictor.add_data_point.assert_not_called()
    planner.osl_predictor.add_data_point.assert_not_called()


def test_encode_planner_applies_component_budget():
    planner = _build_planner(encode_engine_num_gpu=2, max_gpu_budget=4)
    planner.encode_interpolator.interpolate_thpt_per_gpu.return_value = 1.0

    desired = planner._compute_replica_requirements(600, 0, 0)
    budgeted = planner.apply_component_budget(desired)

    assert desired == 5
    assert budgeted == 2


def test_encode_planner_missing_engine_gpu_raises_clear_error():
    planner = _build_planner(encode_engine_num_gpu=None)

    with pytest.raises(ValueError, match="Missing encode_engine_num_gpu"):
        planner._engine_num_gpu()


def test_encode_planner_rejects_pre_swept_results_with_clear_error():
    config = _build_config(
        profile_results_dir="use-pre-swept-results:H200:model:vllm:1.0:1:1:1:16:1"
    )

    with pytest.raises(
        ValueError,
        match="Encode mode does not support .*use-pre-swept-results",
    ):
        EncodePlanner(
            None,
            config,
            shared_state=PlannerSharedState(),
            prometheus_traffic_client=Mock(),
            start_prometheus_server=False,
        )


def test_encode_observe_traffic_updates_only_encode_worker_state():
    planner = _build_planner()
    planner.shared_state.num_p_workers = 7
    planner.shared_state.num_d_workers = 5
    planner.get_single_component_workers = AsyncMock(return_value=(2, True))
    planner.prometheus_traffic_client.get_avg_time_to_first_token.return_value = 0.1
    planner.prometheus_traffic_client.get_avg_inter_token_latency.return_value = 0.01
    planner.prometheus_traffic_client.get_avg_request_count.return_value = 120
    planner.prometheus_traffic_client.get_avg_request_duration.return_value = 3.0
    planner.prometheus_traffic_client.get_avg_input_sequence_tokens.return_value = 0.0
    planner.prometheus_traffic_client.get_avg_output_sequence_tokens.return_value = 0.0

    asyncio.run(
        planner.observe_traffic_stats(
            require_prefill=False,
            require_decode=False,
            require_encode=True,
        )
    )

    assert planner.shared_state.num_e_workers == 2
    assert planner.shared_state.num_p_workers == 7
    assert planner.shared_state.num_d_workers == 5
    assert planner.last_metrics == Metrics(
        ttft=100.0,
        itl=10.0,
        num_req=120,
        isl=0.0,
        osl=0.0,
        request_duration=3.0,
        p_load=None,
        d_load=None,
    )


def test_encode_plan_adjustment_accepts_request_only_metrics():
    planner = _build_planner()
    planner.last_metrics = Metrics(num_req=120)
    planner.num_req_predictor.predict_next.return_value = 120.0
    planner.encode_interpolator.interpolate_thpt_per_gpu.return_value = 1.0

    desired = planner.plan_adjustment()

    assert desired == 1
    planner.num_req_predictor.predict_next.assert_called_once_with()
    planner.isl_predictor.predict_next.assert_not_called()
    planner.osl_predictor.predict_next.assert_not_called()


def test_encode_async_init_uses_generic_validation_for_global_planner():
    config = _build_config(
        environment="global-planner",
        global_planner_namespace="global-ns",
        no_operation=False,
    )
    planner = EncodePlanner(
        Mock(),
        config,
        shared_state=PlannerSharedState(),
        prometheus_traffic_client=Mock(),
        start_prometheus_server=False,
    )

    planner.connector._async_init = AsyncMock()
    planner.connector.validate_deployment = AsyncMock()
    planner.connector.wait_for_deployment_ready = AsyncMock()

    asyncio.run(planner._async_init())

    planner.connector.validate_deployment.assert_awaited_once_with(
        prefill_component_name=None,
        decode_component_name=None,
        require_prefill=False,
        require_decode=False,
    )
    planner.connector.wait_for_deployment_ready.assert_awaited_once_with(
        include_planner=False
    )
    assert planner.model_name == "test-model"
