# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic HTML report generation for planner diagnostics.

Accumulates per-tick snapshots and generates self-contained HTML reports
with interactive Plotly charts.  Uses ``TickInput.now_s`` for timestamps
so reports work identically in live mode (wall clock) and replay
(simulated clock).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import PlannerEffects, TickDiagnostics, TickInput
from dynamo.planner.monitoring.traffic_metrics import Metrics

logger = logging.getLogger(__name__)


@dataclass
class PerEngineFpm:
    """FPM queue depths for a single engine at a single tick."""

    worker_id: str = ""
    dp_rank: int = 0
    queued_prefill_tokens: int = 0
    queued_decode_kv_tokens: int = 0
    inflight_decode_kv_tokens: int = 0


@dataclass
class TickSnapshot:
    """All metrics captured at a single tick, for report generation."""

    timestamp_s: float = 0.0

    # Replica counts
    num_prefill_replicas: Optional[int] = None
    num_decode_replicas: Optional[int] = None

    # Observed traffic (adapter-level, from Metrics)
    observed_ttft_ms: Optional[float] = None
    observed_itl_ms: Optional[float] = None
    observed_requests_per_second: Optional[float] = None
    observed_request_duration_seconds: Optional[float] = None
    observed_input_sequence_tokens: Optional[float] = None
    observed_output_sequence_tokens: Optional[float] = None

    # Diagnostics from state machine
    estimated_ttft_ms: Optional[float] = None
    estimated_itl_ms: Optional[float] = None
    predicted_requests_per_second: Optional[float] = None
    predicted_input_sequence_tokens: Optional[float] = None
    predicted_output_sequence_tokens: Optional[float] = None
    engine_rps_prefill: Optional[float] = None
    engine_rps_decode: Optional[float] = None
    load_decision_reason: Optional[str] = None
    throughput_decision_reason: Optional[str] = None

    # Per-engine FPM queue depths
    prefill_engines: list[PerEngineFpm] = field(default_factory=list)
    decode_engines: list[PerEngineFpm] = field(default_factory=list)

    # Scaling decision
    scale_to_prefill: Optional[int] = None
    scale_to_decode: Optional[int] = None

    # GPU usage
    gpu_hours: float = 0.0


@dataclass
class DiagnosticsRecorder:
    """Accumulates per-tick snapshots and generates periodic HTML reports.

    Usable from both the live adapter (``base.py``) and standalone
    replay harnesses.
    """

    config: PlannerConfig
    _snapshots: list[TickSnapshot] = field(default_factory=list)
    _last_report_s: float = 0.0
    _report_count: int = 0
    _interval_s: float = 0.0
    _max_snapshots: int = 50000

    def __post_init__(self) -> None:
        if self.config.report_interval_hours is not None:
            self._interval_s = self.config.report_interval_hours * 3600.0

    @property
    def enabled(self) -> bool:
        return self._interval_s > 0 or self.config.live_dashboard_port != 0

    def record(
        self,
        tick_input: TickInput,
        effects: PlannerEffects,
        observed: Metrics,
        gpu_hours: float,
    ) -> None:
        if not self.enabled:
            return

        diag = effects.diagnostics or TickDiagnostics()
        interval = self.config.throughput_adjustment_interval

        prefill_engines: list[PerEngineFpm] = []
        decode_engines: list[PerEngineFpm] = []
        fpm_obs = tick_input.fpm_observations
        if fpm_obs is not None:
            if fpm_obs.prefill:
                for (wid, dp), fpm in fpm_obs.prefill.items():
                    prefill_engines.append(
                        PerEngineFpm(
                            worker_id=wid,
                            dp_rank=dp,
                            queued_prefill_tokens=fpm.queued_requests.sum_prefill_tokens,
                            queued_decode_kv_tokens=fpm.queued_requests.sum_decode_kv_tokens,
                            inflight_decode_kv_tokens=fpm.scheduled_requests.sum_decode_kv_tokens,
                        )
                    )
            if fpm_obs.decode:
                for (wid, dp), fpm in fpm_obs.decode.items():
                    decode_engines.append(
                        PerEngineFpm(
                            worker_id=wid,
                            dp_rank=dp,
                            queued_prefill_tokens=fpm.queued_requests.sum_prefill_tokens,
                            queued_decode_kv_tokens=fpm.queued_requests.sum_decode_kv_tokens,
                            inflight_decode_kv_tokens=fpm.scheduled_requests.sum_decode_kv_tokens,
                        )
                    )

        snap = TickSnapshot(
            timestamp_s=tick_input.now_s,
            num_prefill_replicas=(
                tick_input.worker_counts.ready_num_prefill
                if tick_input.worker_counts
                else None
            ),
            num_decode_replicas=(
                tick_input.worker_counts.ready_num_decode
                if tick_input.worker_counts
                else None
            ),
            observed_ttft_ms=observed.ttft,
            observed_itl_ms=observed.itl,
            observed_requests_per_second=(
                observed.num_req / interval
                if observed.num_req is not None and interval > 0
                else None
            ),
            observed_request_duration_seconds=observed.request_duration,
            observed_input_sequence_tokens=observed.isl,
            observed_output_sequence_tokens=observed.osl,
            estimated_ttft_ms=diag.estimated_ttft_ms,
            estimated_itl_ms=diag.estimated_itl_ms,
            predicted_requests_per_second=(
                diag.predicted_num_req / interval
                if diag.predicted_num_req is not None and interval > 0
                else None
            ),
            predicted_input_sequence_tokens=diag.predicted_isl,
            predicted_output_sequence_tokens=diag.predicted_osl,
            engine_rps_prefill=diag.engine_rps_prefill,
            engine_rps_decode=diag.engine_rps_decode,
            load_decision_reason=diag.load_decision_reason,
            throughput_decision_reason=diag.throughput_decision_reason,
            prefill_engines=prefill_engines,
            decode_engines=decode_engines,
            scale_to_prefill=(
                effects.scale_to.num_prefill if effects.scale_to else None
            ),
            scale_to_decode=(effects.scale_to.num_decode if effects.scale_to else None),
            gpu_hours=gpu_hours,
        )
        self._snapshots.append(snap)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots :]

    def should_generate_report(self, now_s: float) -> bool:
        if not self.enabled or not self._snapshots:
            return False
        if self._last_report_s == 0.0:
            self._last_report_s = self._snapshots[0].timestamp_s
        return now_s - self._last_report_s >= self._interval_s

    def _build_report_html(self, snaps: list[TickSnapshot]) -> Optional[str]:
        """Build the HTML report string from the given snapshots.

        Returns the HTML string, or None if plotly is not available.
        This method has no side effects (no file I/O, no snapshot clearing).
        """
        try:
            import plotly.graph_objects as go  # type: ignore[import-untyped]
            from plotly.subplots import make_subplots  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "plotly is not installed -- cannot generate HTML report. "
                "Install with: pip install plotly"
            )
            return None

        ts = [s.timestamp_s for s in snaps]
        labels = [
            datetime.fromtimestamp(t, tz=timezone.utc).strftime("%H:%M:%S") for t in ts
        ]

        fig = make_subplots(
            rows=6,
            cols=2,
            subplot_titles=(
                "Replica Counts",
                "Request Rate (Observed vs Predicted)",
                "Observed TTFT vs SLA",
                "Observed ITL vs SLA",
                "Estimated TTFT vs SLA",
                "Estimated ITL vs SLA",
                "Prefill Engine Load (queued prefill tokens)",
                "Decode Engine Load (queued + inflight decode KV tokens)",
                "Engine Capacity (req/s)",
                "Sequence Lengths (Observed vs Predicted)",
                "Load Scaling Decisions",
                "Throughput Scaling Decisions",
            ),
            vertical_spacing=0.055,
            horizontal_spacing=0.08,
        )

        def _vals(attr: str) -> list:
            return [getattr(s, attr) for s in snaps]

        # -- Row 1 --------------------------------------------------------

        # 1a. Worker counts
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("num_prefill_replicas"),
                name="Prefill Replicas",
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("num_decode_replicas"),
                name="Decode Replicas",
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )

        # 1b. Request rate
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("observed_requests_per_second"),
                name="Observed RPS",
                mode="lines",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("predicted_requests_per_second"),
                name="Predicted RPS",
                mode="lines",
                line=dict(dash="dot"),
            ),
            row=1,
            col=2,
        )

        # -- Row 2: Observed TTFT and ITL in separate plots ---------------

        # 2a. Observed TTFT
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("observed_ttft_ms"),
                name="Observed TTFT",
                mode="lines",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=self.config.ttft,
            line_dash="dash",
            line_color="red",
            annotation_text=f"SLA ({self.config.ttft:.0f}ms)",
            row=2,
            col=1,
        )

        # 2b. Observed ITL
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("observed_itl_ms"),
                name="Observed ITL",
                mode="lines",
                line=dict(color="orange"),
            ),
            row=2,
            col=2,
        )
        fig.add_hline(
            y=self.config.itl,
            line_dash="dash",
            line_color="red",
            annotation_text=f"SLA ({self.config.itl:.0f}ms)",
            row=2,
            col=2,
        )

        # -- Row 3: Estimated TTFT and ITL in separate plots --------------

        # 3a. Estimated TTFT
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("estimated_ttft_ms"),
                name="Estimated TTFT",
                mode="lines+markers",
            ),
            row=3,
            col=1,
        )
        fig.add_hline(
            y=self.config.ttft,
            line_dash="dash",
            line_color="red",
            annotation_text=f"SLA ({self.config.ttft:.0f}ms)",
            row=3,
            col=1,
        )

        # 3b. Estimated ITL
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("estimated_itl_ms"),
                name="Estimated ITL",
                mode="lines+markers",
                line=dict(color="orange"),
            ),
            row=3,
            col=2,
        )
        fig.add_hline(
            y=self.config.itl,
            line_dash="dash",
            line_color="red",
            annotation_text=f"SLA ({self.config.itl:.0f}ms)",
            row=3,
            col=2,
        )

        # -- Row 4: Per-engine FPM load -----------------------------------

        # Collect all engine IDs seen across all ticks
        prefill_engine_ids: set[str] = set()
        decode_engine_ids: set[str] = set()
        for s in snaps:
            for e in s.prefill_engines:
                prefill_engine_ids.add(f"{e.worker_id}:dp{e.dp_rank}")
            for e in s.decode_engines:
                decode_engine_ids.add(f"{e.worker_id}:dp{e.dp_rank}")

        # 4a. Prefill engine load (one line per engine)
        for eid in sorted(prefill_engine_ids):
            y = []
            for s in snaps:
                val = None
                for e in s.prefill_engines:
                    if f"{e.worker_id}:dp{e.dp_rank}" == eid:
                        val = e.queued_prefill_tokens
                        break
                y.append(val)
            fig.add_trace(
                go.Scatter(x=labels, y=y, name=f"P {eid} queued", mode="lines+markers"),
                row=4,
                col=1,
            )

        # 4b. Decode engine load (queued + inflight, one line each per engine)
        for eid in sorted(decode_engine_ids):
            y_queued = []
            y_inflight = []
            for s in snaps:
                q, f_ = None, None
                for e in s.decode_engines:
                    if f"{e.worker_id}:dp{e.dp_rank}" == eid:
                        q = e.queued_decode_kv_tokens
                        f_ = e.inflight_decode_kv_tokens
                        break
                y_queued.append(q)
                y_inflight.append(f_)
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=y_queued,
                    name=f"D {eid} queued",
                    mode="lines+markers",
                ),
                row=4,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=y_inflight,
                    name=f"D {eid} inflight",
                    mode="lines",
                    line=dict(dash="dot"),
                ),
                row=4,
                col=2,
            )

        # -- Row 5 --------------------------------------------------------

        # 5a. Engine capacity
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("engine_rps_prefill"),
                name="Prefill Engine RPS",
                mode="lines+markers",
            ),
            row=5,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("engine_rps_decode"),
                name="Decode Engine RPS",
                mode="lines+markers",
            ),
            row=5,
            col=1,
        )

        # 5b. Sequence lengths
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("observed_input_sequence_tokens"),
                name="Observed ISL",
                mode="lines",
            ),
            row=5,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("predicted_input_sequence_tokens"),
                name="Predicted ISL",
                mode="lines",
                line=dict(dash="dot"),
            ),
            row=5,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("observed_output_sequence_tokens"),
                name="Observed OSL",
                mode="lines",
            ),
            row=5,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=_vals("predicted_output_sequence_tokens"),
                name="Predicted OSL",
                mode="lines",
                line=dict(dash="dot"),
            ),
            row=5,
            col=2,
        )

        # -- Row 6: Decision timelines -----------------------------------

        load_reasons = _vals("load_decision_reason")
        _LOAD_COLORS = {
            "scale_up": "green",
            "scale_down": "blue",
            "scale_down_capped_by_throughput": "purple",
            "no_change": "gray",
            "disabled": "lightgray",
            "no_fpm_data": "yellow",
            "scaling_in_progress": "orange",
            "worker_count_mismatch": "red",
            "insufficient_data": "pink",
        }
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=[1] * len(labels),
                mode="markers",
                marker=dict(
                    color=[_LOAD_COLORS.get(r or "", "gray") for r in load_reasons],
                    size=10,
                    symbol="square",
                ),
                text=load_reasons,
                name="Load Decision",
                hoverinfo="text+x",
            ),
            row=6,
            col=1,
        )

        tp_reasons = _vals("throughput_decision_reason")
        _TP_COLORS = {
            "scale": "green",
            "set_lower_bound": "blue",
            "disabled": "lightgray",
            "no_traffic_data": "yellow",
            "predict_failed": "red",
            "model_not_ready": "orange",
        }
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=[1] * len(labels),
                mode="markers",
                marker=dict(
                    color=[_TP_COLORS.get(r or "", "gray") for r in tp_reasons],
                    size=10,
                    symbol="diamond",
                ),
                text=tp_reasons,
                name="Throughput Decision",
                hoverinfo="text+x",
            ),
            row=6,
            col=2,
        )

        # -- Layout -------------------------------------------------------

        num_scaling_events = sum(
            1
            for s in snaps
            if s.scale_to_prefill is not None or s.scale_to_decode is not None
        )
        t0 = datetime.fromtimestamp(ts[0], tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        t1 = datetime.fromtimestamp(ts[-1], tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        summary = (
            f"<b>Planner Diagnostics Report</b><br>"
            f"Time range: {t0} — {t1} ({len(snaps)} ticks)<br>"
            f"Scaling events: {num_scaling_events} | "
            f"GPU hours: {snaps[-1].gpu_hours:.2f}<br>"
            f"SLA targets: TTFT={self.config.ttft:.0f}ms, ITL={self.config.itl:.0f}ms"
        )
        fig.update_layout(
            title=dict(text=summary, font=dict(size=14), y=0.99, yanchor="top"),
            height=2000,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.03),
            template="plotly_white",
            margin=dict(t=100),
        )

        return fig.to_html(include_plotlyjs=True, full_html=True)

    def generate_report(self) -> Optional[str]:
        """Generate a periodic report, write it to disk, and clear snapshots."""
        if not self._snapshots:
            return None

        snaps = list(self._snapshots)
        html = self._build_report_html(snaps)
        if html is None:
            return None

        ts = [s.timestamp_s for s in snaps]

        output_dir = self.config.report_output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._report_count += 1
        ts_label = datetime.fromtimestamp(ts[-1], tz=timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        )
        filename = f"planner_report_{ts_label}_{self._report_count:03d}.html"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            f.write(html)
        logger.info(f"Planner diagnostics report written to {filepath}")

        self._last_report_s = ts[-1]
        self._snapshots.clear()
        return filepath

    def render_live_html(self) -> Optional[str]:
        """Render the current accumulated snapshots as HTML without side effects.

        Unlike ``generate_report()``, this does NOT clear snapshots or write
        to disk.  Intended for the live dashboard HTTP endpoint.
        """
        if not self._snapshots:
            return None
        return self._build_report_html(list(self._snapshots))

    def finalize(self) -> Optional[str]:
        if self._snapshots:
            return self.generate_report()
        return None
