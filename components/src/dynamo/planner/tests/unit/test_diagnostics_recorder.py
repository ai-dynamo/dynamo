# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DiagnosticsRecorder and HTML report generation."""

import os
import tempfile

import pytest

try:
    import plotly  # noqa: F401
except ImportError:
    pytest.skip("plotly required for report tests", allow_module_level=True)

try:
    import msgspec  # noqa: F401
except ImportError:
    pytest.skip("msgspec required for FPM data", allow_module_level=True)

try:
    from dynamo.common.forward_pass_metrics import (
        ForwardPassMetrics,
        QueuedRequestMetrics,
        ScheduledRequestMetrics,
    )
except ImportError:
    pytest.skip("forward_pass_metrics not available", allow_module_level=True)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    FpmObservations,
    PlannerEffects,
    ScalingDecision,
    TickDiagnostics,
    TickInput,
    WorkerCounts,
)
from dynamo.planner.monitoring.diagnostics_recorder import DiagnosticsRecorder
from dynamo.planner.monitoring.traffic_metrics import Metrics

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _make_config(tmp_dir: str, **overrides) -> PlannerConfig:
    defaults = dict(
        mode="disagg",
        ttft=500.0,
        itl=50.0,
        min_endpoint=1,
        max_gpu_budget=-1,
        throughput_adjustment_interval=60,
        load_adjustment_interval=5,
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        load_predictor="constant",
        backend="vllm",
        metric_pulling_prometheus_endpoint="http://localhost:9090",
        metric_reporting_prometheus_port=0,
        report_interval_hours=0.5,
        report_output_dir=tmp_dir,
        live_dashboard_port=0,
    )
    defaults.update(overrides)
    return PlannerConfig.model_construct(**defaults)


def _synthetic_ticks(
    num_ticks: int = 40,
    start_s: float = 1000.0,
    interval_s: float = 60.0,
) -> list[tuple[TickInput, PlannerEffects, Metrics, float]]:
    """Generate a realistic multi-phase scaling scenario."""
    data = []
    gpu_hours = 0.0
    num_p, num_d = 1, 1

    for i in range(num_ticks):
        t = start_s + i * interval_s
        phase = i / num_ticks

        # Ramp up traffic, then stabilize, then drop
        if phase < 0.3:
            rps = 2.0 + 8.0 * (phase / 0.3)
            isl, osl = 800.0 + 200 * phase, 120.0 + 30 * phase
        elif phase < 0.7:
            rps = 10.0
            isl, osl = 1000.0, 150.0
        else:
            rps = 10.0 - 8.0 * ((phase - 0.7) / 0.3)
            isl, osl = 1000.0 - 200 * (phase - 0.7), 150.0 - 30 * (phase - 0.7)

        observed_ttft = 200.0 + rps * 30
        observed_itl = 20.0 + rps * 3

        # Decisions based on phase
        if phase < 0.15:
            load_reason = "insufficient_data"
            tp_reason = "model_not_ready"
            est_ttft, est_itl = None, None
            scale_p, scale_d = None, None
        elif phase < 0.3:
            load_reason = "scale_up"
            tp_reason = "set_lower_bound"
            est_ttft = observed_ttft * 1.2
            est_itl = observed_itl * 1.1
            num_p = min(num_p + 1, 5)
            num_d = min(num_d + 1, 5)
            scale_p, scale_d = num_p, num_d
        elif phase < 0.7:
            load_reason = "no_change"
            tp_reason = "set_lower_bound"
            est_ttft = observed_ttft * 0.8
            est_itl = observed_itl * 0.9
            scale_p, scale_d = None, None
        elif phase < 0.85:
            load_reason = "scale_down_capped_by_throughput"
            tp_reason = "set_lower_bound"
            est_ttft = observed_ttft * 0.5
            est_itl = observed_itl * 0.5
            scale_p, scale_d = None, None
        else:
            load_reason = "scale_down"
            tp_reason = "set_lower_bound"
            est_ttft = observed_ttft * 0.3
            est_itl = observed_itl * 0.3
            num_p = max(num_p - 1, 1)
            num_d = max(num_d - 1, 1)
            scale_p, scale_d = num_p, num_d

        gpu_hours += (num_p + num_d) * interval_s / 3600.0

        prefill_fpm = {
            (f"pw{j}", 0): ForwardPassMetrics(
                worker_id=f"pw{j}",
                dp_rank=0,
                wall_time=0.01,
                scheduled_requests=ScheduledRequestMetrics(
                    sum_prefill_tokens=int(500 + rps * 50),
                    num_prefill_requests=max(1, int(rps)),
                    sum_decode_kv_tokens=0,
                    num_decode_requests=0,
                ),
                queued_requests=QueuedRequestMetrics(
                    sum_prefill_tokens=int(200 * rps + j * 100),
                    sum_decode_kv_tokens=0,
                ),
            )
            for j in range(num_p)
        }
        decode_fpm = {
            (f"dw{j}", 0): ForwardPassMetrics(
                worker_id=f"dw{j}",
                dp_rank=0,
                wall_time=0.01,
                scheduled_requests=ScheduledRequestMetrics(
                    sum_prefill_tokens=0,
                    num_prefill_requests=0,
                    sum_decode_kv_tokens=int(3000 + rps * 200 + j * 500),
                    num_decode_requests=max(1, int(rps * 2)),
                ),
                queued_requests=QueuedRequestMetrics(
                    sum_prefill_tokens=0,
                    sum_decode_kv_tokens=int(1000 * rps + j * 300),
                ),
            )
            for j in range(num_d)
        }

        tick_input = TickInput(
            now_s=t,
            worker_counts=WorkerCounts(ready_num_prefill=num_p, ready_num_decode=num_d),
            fpm_observations=FpmObservations(prefill=prefill_fpm, decode=decode_fpm),
        )

        effects = PlannerEffects(
            scale_to=(
                ScalingDecision(num_prefill=scale_p, num_decode=scale_d)
                if scale_p is not None or scale_d is not None
                else None
            ),
            diagnostics=TickDiagnostics(
                estimated_ttft_ms=est_ttft,
                estimated_itl_ms=est_itl,
                predicted_num_req=rps * interval_s,
                predicted_isl=isl,
                predicted_osl=osl,
                engine_rps_prefill=5.0 if phase > 0.15 else None,
                engine_rps_decode=8.0 if phase > 0.15 else None,
                load_decision_reason=load_reason,
                throughput_decision_reason=tp_reason,
            ),
        )

        observed = Metrics(
            ttft=observed_ttft,
            itl=observed_itl,
            num_req=rps * interval_s,
            isl=isl,
            osl=osl,
            request_duration=2.5,
        )

        data.append((tick_input, effects, observed, gpu_hours))

    return data


class TestDiagnosticsRecorder:
    def test_disabled_when_no_interval(self, tmp_path):
        cfg = _make_config(str(tmp_path), report_interval_hours=None)
        recorder = DiagnosticsRecorder(config=cfg)
        assert not recorder.enabled

    def test_enabled_when_interval_set(self, tmp_path):
        cfg = _make_config(str(tmp_path), report_interval_hours=1.0)
        recorder = DiagnosticsRecorder(config=cfg)
        assert recorder.enabled

    def test_record_accumulates_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = _make_config(tmp_dir)
            recorder = DiagnosticsRecorder(config=cfg)

            data = _synthetic_ticks(num_ticks=5)
            for ti, eff, obs, gpu in data:
                recorder.record(ti, eff, obs, gpu)

            assert len(recorder._snapshots) == 5

    def test_should_generate_report_after_interval(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = _make_config(tmp_dir, report_interval_hours=0.5)
            recorder = DiagnosticsRecorder(config=cfg)

            data = _synthetic_ticks(num_ticks=40, interval_s=60.0)
            for ti, eff, obs, gpu in data:
                recorder.record(ti, eff, obs, gpu)

            last_t = data[-1][0].now_s
            assert recorder.should_generate_report(last_t)

    def test_generate_report_creates_html(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = _make_config(tmp_dir)
            recorder = DiagnosticsRecorder(config=cfg)

            data = _synthetic_ticks(num_ticks=40)
            for ti, eff, obs, gpu in data:
                recorder.record(ti, eff, obs, gpu)

            filepath = recorder.generate_report()
            assert filepath is not None
            assert os.path.exists(filepath)
            assert filepath.endswith(".html")

            with open(filepath) as f:
                content = f.read()
            assert len(content) > 1000
            assert "plotly" in content.lower()
            assert "Replica Counts" in content
            assert "Observed TTFT vs SLA" in content
            assert "Observed ITL vs SLA" in content
            assert "Estimated TTFT vs SLA" in content
            assert "Estimated ITL vs SLA" in content
            assert "Prefill Engine Load" in content
            assert "Decode Engine Load" in content
            assert "Request Rate" in content
            assert "Engine Capacity" in content
            assert "Load Scaling Decisions" in content
            assert "Throughput Scaling Decisions" in content
            assert "Planner Diagnostics Report" in content

    def test_generate_report_clears_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = _make_config(tmp_dir)
            recorder = DiagnosticsRecorder(config=cfg)

            data = _synthetic_ticks(num_ticks=10)
            for ti, eff, obs, gpu in data:
                recorder.record(ti, eff, obs, gpu)

            recorder.generate_report()
            assert len(recorder._snapshots) == 0

    def test_finalize_generates_final_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = _make_config(tmp_dir)
            recorder = DiagnosticsRecorder(config=cfg)

            data = _synthetic_ticks(num_ticks=5)
            for ti, eff, obs, gpu in data:
                recorder.record(ti, eff, obs, gpu)

            filepath = recorder.finalize()
            assert filepath is not None
            assert os.path.exists(filepath)

    def test_finalize_noop_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = _make_config(tmp_dir)
            recorder = DiagnosticsRecorder(config=cfg)
            assert recorder.finalize() is None

    def test_record_without_fpm_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = _make_config(tmp_dir)
            recorder = DiagnosticsRecorder(config=cfg)

            tick_input = TickInput(
                now_s=1000.0,
                worker_counts=WorkerCounts(ready_num_prefill=2, ready_num_decode=3),
                fpm_observations=None,
            )
            effects = PlannerEffects(
                diagnostics=TickDiagnostics(load_decision_reason="no_fpm_data"),
            )
            observed = Metrics(ttft=100.0, itl=10.0, num_req=50, isl=800, osl=120)
            recorder.record(tick_input, effects, observed, 1.0)

            assert len(recorder._snapshots) == 1
            snap = recorder._snapshots[0]
            assert snap.prefill_engines == []
            assert snap.decode_engines == []

            filepath = recorder.generate_report()
            assert filepath is not None
            assert os.path.exists(filepath)
