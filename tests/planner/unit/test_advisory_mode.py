# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Planner Advisory Mode (ScalingMode, config fields, advisory engine)."""

import math
import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import ValidationError

from dynamo.planner.defaults import ScalingMode
from dynamo.planner.utils.planner_config import PlannerConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# ScalingMode enum
# ---------------------------------------------------------------------------


def test_scaling_mode_values():
    assert ScalingMode.ACTIVE == "active"
    assert ScalingMode.ADVISORY == "advisory"
    assert ScalingMode.NOOP == "noop"


# ---------------------------------------------------------------------------
# PlannerConfig: new fields and defaults
# ---------------------------------------------------------------------------


def test_scaling_mode_default():
    config = PlannerConfig(namespace="test-ns")
    assert config.scaling_mode == ScalingMode.ACTIVE


def test_scaling_mode_advisory():
    config = PlannerConfig(
        namespace="test-ns",
        scaling_mode="advisory",
        enable_throughput_scaling=False,
        enable_load_scaling=True,
        environment="kubernetes",
        load_router_metrics_url="http://localhost:9090",
    )
    assert config.scaling_mode == ScalingMode.ADVISORY
    assert config.effective_scaling_mode == ScalingMode.ADVISORY


def test_scaling_mode_noop():
    config = PlannerConfig(namespace="test-ns", scaling_mode="noop")
    assert config.scaling_mode == ScalingMode.NOOP
    assert config.effective_scaling_mode == ScalingMode.NOOP


def test_advisory_defaults():
    config = PlannerConfig(namespace="test-ns")
    assert config.advisory_max_step_size == 1
    assert config.advisory_anomaly_threshold == 10
    assert config.advisory_file_output is False


def test_advisory_max_step_size_invalid():
    with pytest.raises(ValidationError, match="advisory_max_step_size must be >= 1"):
        PlannerConfig(namespace="test-ns", advisory_max_step_size=0)


def test_advisory_file_output_requires_log_dir():
    with pytest.raises(
        ValidationError, match="advisory_file_output=True requires log_dir"
    ):
        PlannerConfig(
            namespace="test-ns",
            advisory_file_output=True,
            log_dir=None,
        )


def test_advisory_file_output_with_log_dir():
    config = PlannerConfig(
        namespace="test-ns",
        advisory_file_output=True,
        log_dir="/tmp/advisory",
    )
    assert config.advisory_file_output is True
    assert config.log_dir == "/tmp/advisory"


# ---------------------------------------------------------------------------
# Backward compatibility: no_operation -> noop
# ---------------------------------------------------------------------------


def test_no_operation_true_maps_to_noop(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        config = PlannerConfig(namespace="test-ns", no_operation=True)
    assert config.scaling_mode == ScalingMode.NOOP
    assert "DEPRECATION" in caplog.text


def test_no_operation_false_keeps_active():
    config = PlannerConfig(namespace="test-ns", no_operation=False)
    assert config.scaling_mode == ScalingMode.ACTIVE


def test_explicit_scaling_mode_overrides_no_operation():
    """Explicit scaling_mode takes precedence over no_operation."""
    config = PlannerConfig(
        namespace="test-ns",
        scaling_mode="noop",
        no_operation=False,  # no_operation=False, but scaling_mode=noop
    )
    assert config.scaling_mode == ScalingMode.NOOP


def test_effective_scaling_mode_backward_compat_model_construct():
    """model_construct bypasses validator; effective_scaling_mode must still respect no_operation."""
    config = PlannerConfig.model_construct(
        no_operation=True,
        scaling_mode=ScalingMode.ACTIVE,
    )
    # effective_scaling_mode should return NOOP because no_operation=True
    assert config.effective_scaling_mode == ScalingMode.NOOP


def test_effective_scaling_mode_noop_explicit():
    config = PlannerConfig.model_construct(
        no_operation=False,
        scaling_mode=ScalingMode.NOOP,
    )
    assert config.effective_scaling_mode == ScalingMode.NOOP


def test_effective_scaling_mode_advisory():
    config = PlannerConfig.model_construct(
        no_operation=False,
        scaling_mode=ScalingMode.ADVISORY,
    )
    assert config.effective_scaling_mode == ScalingMode.ADVISORY


# ---------------------------------------------------------------------------
# BasePlanner advisory engine methods
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_prometheus():
    """Mock Gauge and Counter to avoid registry conflicts."""
    with patch("dynamo.planner.utils.planner_core.Gauge") as mock_g, patch(
        "dynamo.planner.utils.planner_core.Counter"
    ) as mock_c:
        mock_g.return_value = Mock()
        mock_c.return_value = Mock()
        yield


def _make_base_planner(scaling_mode=ScalingMode.ADVISORY, enable_throughput=True):
    """Build a minimal BasePlanner in dryrun mode for testing advisory methods."""
    from dynamo.planner.utils.planner_config import PlannerConfig
    from dynamo.planner.utils.planner_core import (
        BasePlanner,
        PlannerPrometheusMetrics,
        PlannerSharedState,
        SubComponentType,
    )

    config = PlannerConfig.model_construct(
        no_operation=False,
        scaling_mode=scaling_mode,
        advisory_max_step_size=1,
        advisory_anomaly_threshold=10,
        advisory_file_output=False,
        log_dir=None,
        ttft=500.0,
        itl=50.0,
        throughput_adjustment_interval=60,
        prefill_engine_num_gpu=1,
        decode_engine_num_gpu=1,
        min_endpoint=1,
        max_gpu_budget=-1,
        backend="vllm",
        no_correction=True,
        metric_pulling_prometheus_endpoint="http://localhost:9090",
        metric_reporting_prometheus_port=8080,
        load_predictor="constant",
        load_predictor_warmup_trace=None,
        load_predictor_log1p=False,
        profile_results_dir=os.path.join(
            os.path.dirname(__file__),
            "..",
            "profiling_results",
            "H200_TP1P_TP1D",
        ),
        environment="kubernetes",
        namespace="test-ns",
        mode="disagg",
        enable_throughput_scaling=enable_throughput,
        enable_load_scaling=not enable_throughput,
        load_router_metrics_url="http://localhost:9090" if not enable_throughput else None,
        load_adjustment_interval=5,
        load_learning_window=50,
        load_scaling_down_sensitivity=80,
        load_metric_samples=10,
        load_min_observations=5,
    )

    shared_state = PlannerSharedState()
    shared_state.num_p_workers = 1
    shared_state.num_d_workers = 1

    pm = PlannerPrometheusMetrics()

    planner = BasePlanner.__new__(BasePlanner)
    planner.config = config
    planner.shared_state = shared_state
    planner.prometheus_metrics = pm
    planner.prometheus_port = 8080
    planner.enable_throughput = enable_throughput
    planner.enable_load = not enable_throughput
    planner.dryrun = True
    planner.no_correction = True
    planner.component_type = SubComponentType.PREFILL

    return planner


def _set_valid_metrics(planner):
    from dynamo.planner.utils.prometheus import Metrics

    planner.shared_state.last_metrics = Metrics(
        num_req=100.0,
        isl=3000.0,
        osl=150.0,
        ttft=80.0,
        itl=10.0,
        request_duration=1.5,
    )


# --- _safe_gauge_set ---


def test_safe_gauge_set_normal_value():
    from dynamo.planner.utils.planner_core import BasePlanner

    gauge = Mock()
    BasePlanner._safe_gauge_set(gauge, 5.0)
    gauge.set.assert_called_once_with(5.0)


def test_safe_gauge_set_none():
    from dynamo.planner.utils.planner_core import BasePlanner

    gauge = Mock()
    BasePlanner._safe_gauge_set(gauge, None)
    gauge.set.assert_called_once_with(float("nan"))


def test_safe_gauge_set_nan():
    from dynamo.planner.utils.planner_core import BasePlanner

    gauge = Mock()
    BasePlanner._safe_gauge_set(gauge, float("nan"))
    val = gauge.set.call_args[0][0]
    assert math.isnan(val)


# --- _emit_advisory_metrics: metrics validity gate ---


def test_emit_advisory_skips_when_metrics_invalid():
    planner = _make_base_planner()
    # last_metrics is empty (invalid)
    planner._emit_advisory_metrics(3, 2, "throughput")
    # No gauge set calls since metrics are invalid
    planner.prometheus_metrics.advisory_recommended_p.set.assert_not_called()


def test_emit_advisory_emits_when_metrics_valid():
    planner = _make_base_planner()
    _set_valid_metrics(planner)
    planner._emit_advisory_metrics(3, 2, "throughput")
    planner.prometheus_metrics.advisory_recommended_p.set.assert_called_once_with(3)
    planner.prometheus_metrics.advisory_recommended_d.set.assert_called_once_with(2)


# --- _emit_advisory_metrics: non-negative guard ---


def test_emit_advisory_non_negative_guard():
    planner = _make_base_planner()
    _set_valid_metrics(planner)
    planner._emit_advisory_metrics(-5, -2, "throughput")
    # Should clamp to 0
    planner.prometheus_metrics.advisory_recommended_p.set.assert_called_once_with(0)
    planner.prometheus_metrics.advisory_recommended_d.set.assert_called_once_with(0)


# --- _emit_advisory_metrics: anomaly detection ---


def test_emit_advisory_anomaly_warning(caplog):
    import logging

    planner = _make_base_planner()
    _set_valid_metrics(planner)
    planner.shared_state.num_p_workers = 1
    planner.shared_state.num_d_workers = 1

    with caplog.at_level(logging.WARNING):
        planner._emit_advisory_metrics(50, 1, "throughput")  # delta_p=49 >> threshold=10

    assert "Unusually large delta" in caplog.text


# --- _emit_advisory_metrics: action and reason codes ---


def test_emit_advisory_scale_up_action():
    planner = _make_base_planner()
    _set_valid_metrics(planner)
    planner.shared_state.num_p_workers = 1
    planner.shared_state.num_d_workers = 1

    planner._emit_advisory_metrics(3, 2, "throughput")
    planner.prometheus_metrics.advisory_scaling_action.set.assert_called_once_with(1)
    planner.prometheus_metrics.advisory_action_reason.set.assert_called_once_with(1)
    planner.prometheus_metrics.advisory_scaleup_total.inc.assert_called_once()


def test_emit_advisory_scale_down_action():
    planner = _make_base_planner()
    _set_valid_metrics(planner)
    planner.shared_state.num_p_workers = 3
    planner.shared_state.num_d_workers = 2

    planner._emit_advisory_metrics(1, 1, "throughput")
    planner.prometheus_metrics.advisory_scaling_action.set.assert_called_once_with(-1)
    planner.prometheus_metrics.advisory_action_reason.set.assert_called_once_with(2)
    planner.prometheus_metrics.advisory_scaledown_total.inc.assert_called_once()


def test_emit_advisory_hold_action():
    planner = _make_base_planner()
    _set_valid_metrics(planner)
    planner.shared_state.num_p_workers = 2
    planner.shared_state.num_d_workers = 1

    planner._emit_advisory_metrics(2, 1, "throughput")
    planner.prometheus_metrics.advisory_scaling_action.set.assert_called_once_with(0)
    planner.prometheus_metrics.advisory_action_reason.set.assert_called_once_with(6)
    planner.prometheus_metrics.advisory_hold_total.inc.assert_called_once()


def test_emit_advisory_load_source_reason_codes():
    planner = _make_base_planner()
    _set_valid_metrics(planner)
    planner.shared_state.num_p_workers = 1
    planner.shared_state.num_d_workers = 1

    planner._emit_advisory_metrics(3, 2, "load")
    # Scale up from load source -> reason code 3
    planner.prometheus_metrics.advisory_action_reason.set.assert_called_once_with(3)


# --- _estimate_sla_with_replicas: load-only mode ---


def test_estimate_sla_unavailable_in_load_only_mode(caplog):
    import logging

    planner = _make_base_planner(enable_throughput=False)
    _set_valid_metrics(planner)

    with caplog.at_level(logging.INFO):
        result = planner._estimate_sla_with_replicas(2, 1)

    assert result["est_ttft"] is None
    assert result["est_itl"] is None
    assert "SLA estimation unavailable" in caplog.text


# --- _build_path_recommendation ---


def test_build_path_no_change():
    planner = _make_base_planner()
    path = planner._build_path_recommendation(2, 1, 2, 1)
    assert path == ["2P1D"]


def test_build_path_scale_up_p():
    planner = _make_base_planner()
    path = planner._build_path_recommendation(1, 1, 3, 1)
    # With max_step_size=1: 1P1D -> 2P1D (observe) -> 3P1D
    assert path[0] == "1P1D"
    assert path[-1] == "3P1D"
    assert len(path) == 3


def test_build_path_scale_down():
    planner = _make_base_planner()
    path = planner._build_path_recommendation(3, 2, 1, 1)
    assert path[0] == "3P2D"
    assert path[-1] == "1P1D"


def test_build_path_max_step_size_2():
    planner = _make_base_planner()
    planner.config = PlannerConfig.model_construct(
        **{**planner.config.model_dump(), "advisory_max_step_size": 2}
    )
    path = planner._build_path_recommendation(1, 1, 5, 1)
    # Steps: 1->3->5 with D staying at 1
    assert path[0] == "1P1D"
    assert path[-1] == "5P1D"


# --- _write_advisory_jsonl ---


def test_write_advisory_jsonl(tmp_path):
    planner = _make_base_planner()
    planner.config = PlannerConfig.model_construct(
        **{**planner.config.model_dump(), "log_dir": str(tmp_path), "advisory_file_output": True}
    )
    planner._write_advisory_jsonl({"ts": 1234, "action": "scale_up"})
    import json

    filepath = tmp_path / "advisory_history.jsonl"
    assert filepath.exists()
    line = json.loads(filepath.read_text().strip())
    assert line["ts"] == 1234
    assert line["action"] == "scale_up"


def test_write_advisory_jsonl_no_log_dir():
    """No file should be written if log_dir is None."""
    planner = _make_base_planner()
    # Should not raise even when log_dir is None
    planner._write_advisory_jsonl({"ts": 1234})


# --- Integration: ACTIVE mode also emits advisory metrics ---


def test_active_mode_emits_advisory_metrics():
    planner = _make_base_planner(scaling_mode=ScalingMode.ACTIVE)
    _set_valid_metrics(planner)
    planner._emit_advisory_metrics(3, 2, "throughput")
    # In ACTIVE mode, gauges should still be set
    planner.prometheus_metrics.advisory_recommended_p.set.assert_called_once_with(3)


# --- Integration: NOOP mode does not reach advisory methods ---


def test_effective_scaling_mode_noop_with_no_operation():
    """Ensure model_construct + no_operation=True works with effective_scaling_mode."""
    config = PlannerConfig.model_construct(
        no_operation=True,
        scaling_mode=ScalingMode.ACTIVE,
        advisory_max_step_size=1,
        advisory_anomaly_threshold=10,
        advisory_file_output=False,
        log_dir=None,
    )
    assert config.effective_scaling_mode == ScalingMode.NOOP
    assert config.effective_scaling_mode != ScalingMode.ACTIVE
    assert config.effective_scaling_mode != ScalingMode.ADVISORY
