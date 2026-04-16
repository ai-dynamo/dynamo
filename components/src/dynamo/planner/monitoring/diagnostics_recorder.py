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

import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

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
    load_decision_reason_prefill: Optional[str] = None
    load_decision_reason_decode: Optional[str] = None
    throughput_decision_reason_prefill: Optional[str] = None
    throughput_decision_reason_decode: Optional[str] = None

    # Per-engine FPM queue depths
    prefill_engines: list[PerEngineFpm] = field(default_factory=list)
    decode_engines: list[PerEngineFpm] = field(default_factory=list)

    # Throughput lower bound
    throughput_lower_bound_prefill: Optional[int] = None
    throughput_lower_bound_decode: Optional[int] = None

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
    max_kv_tokens: Optional[int] = None
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
            load_decision_reason_prefill=diag.load_decision_reason_prefill,
            load_decision_reason_decode=diag.load_decision_reason_decode,
            throughput_decision_reason_prefill=diag.throughput_decision_reason_prefill,
            throughput_decision_reason_decode=diag.throughput_decision_reason_decode,
            throughput_lower_bound_prefill=diag.throughput_lower_bound_prefill,
            throughput_lower_bound_decode=diag.throughput_lower_bound_decode,
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

    def _build_report_html(self, snaps: list[TickSnapshot]) -> str:
        """Build the HTML report string from the given snapshots.

        This method has no side effects (no file I/O, no snapshot clearing).
        """
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
        tp_lower_p = _vals("throughput_lower_bound_prefill")
        if any(v is not None and v > 1 for v in tp_lower_p):
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=tp_lower_p,
                    name="Prefill TP Lower Bound",
                    mode="lines",
                    line=dict(dash="dash", color="darkblue"),
                ),
                row=1,
                col=1,
            )
        tp_lower_d = _vals("throughput_lower_bound_decode")
        if any(v is not None and v > 1 for v in tp_lower_d):
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=tp_lower_d,
                    name="Decode TP Lower Bound",
                    mode="lines",
                    line=dict(dash="dash", color="red"),
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
                connectgaps=True,
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
                go.Scatter(
                    x=labels,
                    y=y,
                    name=f"P {eid} queued",
                    mode="lines+markers",
                    showlegend=False,
                ),
                row=4,
                col=1,
            )

        # 4b. Decode engine load (queued + inflight combined per engine)
        for eid in sorted(decode_engine_ids):
            y_total = []
            for s in snaps:
                val = None
                for e in s.decode_engines:
                    if f"{e.worker_id}:dp{e.dp_rank}" == eid:
                        val = e.queued_decode_kv_tokens + e.inflight_decode_kv_tokens
                        break
                y_total.append(val)
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=y_total,
                    name=f"D {eid} total KV",
                    mode="lines+markers",
                    showlegend=False,
                ),
                row=4,
                col=2,
            )

        # KV capacity line (set by adapter if available)
        if self.max_kv_tokens is not None and self.max_kv_tokens > 0:
            fig.add_hline(
                y=self.max_kv_tokens,
                line_dash="dash",
                line_color="red",
                annotation_text=f"KV Capacity ({self.max_kv_tokens:,})",
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
                connectgaps=True,
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
                connectgaps=True,
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
                connectgaps=True,
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
                connectgaps=True,
            ),
            row=5,
            col=2,
        )

        # -- Row 6: Decision timelines -----------------------------------
        #
        # Layout adapts to scaling mode:
        #   - Disagg (per-component reasons populated): two tracks per
        #     subplot, prefill on y=2, decode on y=1, with "prefill" /
        #     "decode" y-axis labels.
        #   - Agg / easy mode (only aggregate reason populated): single
        #     track at y=1 with "Load Decision" / "Throughput Decision"
        #     labels.
        # We detect the mode by whether any snapshot has a per-component
        # reason set; switching mode mid-run would produce a mixed chart,
        # but that doesn't happen because mode is fixed at planner init.

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
        _TP_COLORS = {
            "scale": "green",
            "set_lower_bound": "blue",
            "disabled": "lightgray",
            "no_traffic_data": "yellow",
            "predict_failed": "red",
            "model_not_ready": "orange",
        }

        # Detect disagg mode: if any per-component reason is populated,
        # plot two horizontal tracks (prefill at y=2, decode at y=1);
        # otherwise plot a single aggregate track at y=1.
        has_per_component_load = any(
            s.load_decision_reason_prefill is not None
            or s.load_decision_reason_decode is not None
            for s in snaps
        )
        has_per_component_tp = any(
            s.throughput_decision_reason_prefill is not None
            or s.throughput_decision_reason_decode is not None
            for s in snaps
        )

        def _add_decision_track(
            field_name: str,
            y_value: int,
            label: str,
            colors: dict,
            symbol: str,
            row: int,
            col: int,
        ) -> None:
            reasons = _vals(field_name)
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=[y_value] * len(labels),
                    mode="markers",
                    marker=dict(
                        color=[colors.get(r or "", "gray") for r in reasons],
                        size=10,
                        symbol=symbol,
                    ),
                    text=reasons,
                    name=label,
                    hoverinfo="text+x",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        if has_per_component_load:
            _add_decision_track(
                "load_decision_reason_prefill",
                2,
                "Load (prefill)",
                _LOAD_COLORS,
                "square",
                6,
                1,
            )
            _add_decision_track(
                "load_decision_reason_decode",
                1,
                "Load (decode)",
                _LOAD_COLORS,
                "square",
                6,
                1,
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=[1, 2],
                ticktext=["decode", "prefill"],
                range=[0.5, 2.5],
                row=6,
                col=1,
            )
        else:
            _add_decision_track(
                "load_decision_reason",
                1,
                "Load Decision",
                _LOAD_COLORS,
                "square",
                6,
                1,
            )

        if has_per_component_tp:
            _add_decision_track(
                "throughput_decision_reason_prefill",
                2,
                "TP (prefill)",
                _TP_COLORS,
                "diamond",
                6,
                2,
            )
            _add_decision_track(
                "throughput_decision_reason_decode",
                1,
                "TP (decode)",
                _TP_COLORS,
                "diamond",
                6,
                2,
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=[1, 2],
                ticktext=["decode", "prefill"],
                range=[0.5, 2.5],
                row=6,
                col=2,
            )
        else:
            _add_decision_track(
                "throughput_decision_reason",
                1,
                "Throughput Decision",
                _TP_COLORS,
                "diamond",
                6,
                2,
            )

        # -- Layout -------------------------------------------------------

        # Count actual replica transitions, not just ticks where a decision
        # was recorded: two consecutive ticks with scale_to=5 aren't two
        # scaling events.
        num_scaling_events = 0
        prev_p: Optional[int] = None
        prev_d: Optional[int] = None
        for s in snaps:
            cur_p = s.num_prefill_replicas
            cur_d = s.num_decode_replicas
            if prev_p is not None and cur_p is not None and cur_p != prev_p:
                num_scaling_events += 1
            if prev_d is not None and cur_d is not None and cur_d != prev_d:
                num_scaling_events += 1
            if cur_p is not None:
                prev_p = cur_p
            if cur_d is not None:
                prev_d = cur_d

        t0 = datetime.fromtimestamp(ts[0], tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        t1 = datetime.fromtimestamp(ts[-1], tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        summary = (
            f"<b>Planner Diagnostics Report</b><br>"
            f"Time range: {t0} — {t1} ({len(snaps)} ticks)<br>"
            f"Replica transitions: {num_scaling_events} | "
            f"GPU hours: {snaps[-1].gpu_hours:.2f}<br>"
            f"SLA targets: TTFT={self.config.ttft:.0f}ms, ITL={self.config.itl:.0f}ms"
        )
        fig.update_layout(
            title=dict(text=summary, font=dict(size=14), y=0.99, yanchor="top"),
            height=2000,
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.05),
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
        ts = [s.timestamp_s for s in snaps]

        output_dir = self.config.report_output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._report_count += 1
        if self.config.report_filename:
            filename = self.config.report_filename
        else:
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
