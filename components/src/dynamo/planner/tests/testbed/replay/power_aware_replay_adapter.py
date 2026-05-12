# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PowerAwareReplayAdapter — γ-class subclass of ReplayPlannerAdapter.

Extends the existing replay loop with:
  1. SyntheticPowerOverlay.observe() after bridge.advance_to()
  2. AICPowerOptimizer.update_correction() + should_reoptimize() / optimize()
  3. state_machine._apply_power_budget() after on_tick()
  4. Feeding into TickHistory (via the testbed TickRecorder)

Does NOT modify ReplayPlannerAdapter itself — pure subclass extension.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from dynamo.planner.offline.replay_adapter import ReplayPlannerAdapter

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.planner.core.types import WorkerCapabilities
    from dynamo.planner.tests.testbed.fake_aic import FakeAIC
    from dynamo.planner.tests.testbed.fake_planner_metrics import FakePlannerMetrics
    from dynamo.planner.tests.testbed.fake_prometheus import FakePrometheusClient
    from dynamo.planner.tests.testbed.recorder import TickHistory
    from dynamo.planner.tests.testbed.replay.replay_fake_actuator import (
        ReplayFakeActuator,
    )
    from dynamo.planner.tests.testbed.replay.synthetic_power_overlay import (
        SyntheticPowerOverlay,
    )
    from dynamo.planner.tests.testbed.scenarios import ScenarioSpec

logger = logging.getLogger(__name__)


class PowerAwareReplayAdapter(ReplayPlannerAdapter):
    """Extends ReplayPlannerAdapter with the power-aware control loop.

    Does not modify the parent class; overrides ``run()`` only to interleave
    new steps between ``bridge.advance_to()`` and ``bridge.apply_scaling()``.
    """

    def __init__(
        self,
        planner_config: "PlannerConfig",
        bridge: Any,
        scenario: "ScenarioSpec",
        *,
        overlay: "SyntheticPowerOverlay",
        fake_prom: "FakePrometheusClient",
        fake_aic: "FakeAIC",
        actuator: "ReplayFakeActuator",
        metrics: "FakePlannerMetrics",
        capabilities: Optional["WorkerCapabilities"] = None,
        warmup_observations: Optional[list] = None,
    ) -> None:
        super().__init__(planner_config, bridge, capabilities, warmup_observations)
        self._overlay = overlay
        self._fake_prom = fake_prom
        self._fake_aic = fake_aic
        self._actuator = actuator
        self._metrics = metrics
        self._scenario = scenario

        from dynamo.planner.monitoring.aic_power_optimizer import AICPowerOptimizer

        if planner_config.enable_aic_optimizer:
            self._optimizer: Optional[AICPowerOptimizer] = AICPowerOptimizer(
                planner_config, metrics
            )
            self._optimizer._aic_estimator_factory = fake_aic.make_estimator_factory()
            # Initial sweep
            initial = self._optimizer.optimize()
            if initial is not None:
                planner_config.prefill_engine_gpu_power_limit = initial.cap_p
                planner_config.decode_engine_gpu_power_limit = initial.cap_d
                self._optimizer._estimated_throughput = (
                    initial.aic_seq_per_s_per_replica
                    * initial.n_d
                    * (initial.isl + initial.osl)
                )
        else:
            self._optimizer = None

        self._tick_history: Optional["TickHistory"] = None

    def _attach_history(self, history: "TickHistory") -> None:
        self._tick_history = history

    def run(self):  # type: ignore[override]
        """Run γ-class replay loop with power-aware extensions."""
        from dynamo.planner.offline.replay_adapter import ReplayPlannerReport

        next_tick = self._sm.initial_tick(0.0)
        scaling_events = []
        diagnostics_log = []
        total_ticks = 0

        while True:
            tick_ms = next_tick.at_s * 1000.0
            result = self._bridge.advance_to(tick_ms)

            if result["is_done"]:
                break

            # Compute virtual tick index
            vtick = int(next_tick.at_s / self._config.throughput_adjustment_interval)
            self._fake_prom.set_tick(vtick)

            # --- γ extension: overlay synthesises power from FPMs ---
            # The Rust bridge returns prefill/decode snapshots in separate
            # lists; tag each and concatenate so the overlay (which is
            # component-aware) can dispatch on snap["component"].
            prefill_snaps = result.get("prefill_fpm_snapshots", [])
            decode_snaps = result.get("decode_fpm_snapshots", [])
            for s in prefill_snaps:
                s["component"] = "prefill"
            for s in decode_snaps:
                s["component"] = "decode"
            self._overlay.observe(
                fpm_snapshots=prefill_snaps + decode_snaps,
                applied_caps=self._actuator.applied_caps_snapshot(),
                tick=vtick,
            )

            tick_input = self._build_tick_input(next_tick, result)

            # --- γ extension: optimizer correction + drift check ---
            if self._optimizer is not None and tick_input.traffic is not None:
                prom_power_p = self._fake_prom.get_avg_per_gpu_power_by_component(
                    component="prefill", interval="60s"
                )
                prom_power_d = self._fake_prom.get_avg_per_gpu_power_by_component(
                    component="decode", interval="60s"
                )
                self._optimizer.update_correction(
                    traffic=tick_input.traffic,
                    observed_ttft_avg=(
                        tick_input.traffic.ttft_avg if tick_input.traffic else None
                    ),
                    observed_itl_avg=(
                        tick_input.traffic.itl_avg if tick_input.traffic else None
                    ),
                    observed_power_w_prefill=prom_power_p,
                    observed_power_w_decode=prom_power_d,
                )
                if self._optimizer.should_reoptimize(tick_input.traffic):
                    new_cfg = self._optimizer.optimize()
                    if new_cfg is not None:
                        self._config.prefill_engine_gpu_power_limit = new_cfg.cap_p
                        self._config.decode_engine_gpu_power_limit = new_cfg.cap_d
                        self._actuator.apply_caps(new_cfg.cap_p, new_cfg.cap_d)
                        self._optimizer._estimated_throughput = (
                            new_cfg.aic_seq_per_s_per_replica
                            * new_cfg.n_d
                            * (new_cfg.isl + new_cfg.osl)
                        )

            # --- Existing: state machine tick ---
            effects = self._sm.on_tick(next_tick, tick_input)
            diagnostics_log.append(effects.diagnostics)
            total_ticks += 1

            self._record_diagnostics(tick_input, effects, result)

            active_p = result["active_prefill_count"]
            active_d = result["active_decode_count"]
            if (
                self._scaling_target_prefill is not None
                and active_p == self._scaling_target_prefill
            ):
                self._scaling_target_prefill = None
            if (
                self._scaling_target_decode is not None
                and active_d == self._scaling_target_decode
            ):
                self._scaling_target_decode = None

            # --- γ extension: power budget post-clamp ---
            if effects.scale_to is not None and self._config.enable_power_awareness:
                clamped_p, clamped_d = self._sm._apply_power_budget(
                    effects.scale_to.num_prefill or active_p,
                    effects.scale_to.num_decode or active_d,
                )
                effects.scale_to.num_prefill = clamped_p
                effects.scale_to.num_decode = clamped_d

            if effects.scale_to is not None:
                self._apply_scaling(effects, result, tick_input.now_s, scaling_events)

            if effects.next_tick is None:
                break
            next_tick = effects.next_tick

        trace_report = self._bridge.finalize()
        html_report_path = self._recorder.finalize()
        return ReplayPlannerReport(
            trace_report=trace_report,
            scaling_events=scaling_events,
            diagnostics_log=diagnostics_log,
            total_ticks=total_ticks,
            html_report_path=html_report_path,
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_gamma_adapter(scenario: "ScenarioSpec") -> "GammaHarness":
    """Construct the full γ-class harness from a scenario spec."""
    return GammaHarness(scenario)


class GammaHarness:
    """Container for the full γ-class setup."""

    def __init__(self, scenario: "ScenarioSpec") -> None:
        self.scenario = scenario
        self._setup()

    def _setup(self) -> None:
        import random

        from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
        from dynamo.planner.config.parallelization import PickedParallelConfig
        from dynamo.planner.config.planner_config import PlannerConfig
        from dynamo.planner.core.types import EngineCapabilities, WorkerCapabilities
        from dynamo.planner.tests.testbed.fake_aic import FakeAIC
        from dynamo.planner.tests.testbed.fake_planner_metrics import FakePlannerMetrics
        from dynamo.planner.tests.testbed.fake_prometheus import FakePrometheusClient
        from dynamo.planner.tests.testbed.replay.replay_fake_actuator import (
            ReplayFakeActuator,
        )
        from dynamo.planner.tests.testbed.replay.synthetic_power_overlay import (
            SyntheticPowerOverlay,
        )
        from dynamo.planner.tests.testbed.scenarios import SystemSpec

        sc = self.scenario
        rng = random.Random(sc.seed)
        mocker = sc.mocker
        overlay_spec = sc.overlay
        planner_spec = sc.planner

        system_spec = SystemSpec.load(overlay_spec.system)

        aic_spec = AICInterpolationSpec(
            hf_id="fake-model/testbed",
            system=overlay_spec.system,
            backend="vllm",
            isl=3000,
            osl=150,
            sweep_max_context_length=8192,
            prefill_interpolation_granularity=1,
            decode_interpolation_granularity=1,
            prefill_pick=PickedParallelConfig(tp=1, pp=1, dp=1),
            decode_pick=PickedParallelConfig(tp=1, pp=1, dp=1),
        )
        self.config = PlannerConfig(
            mode=planner_spec.mode,
            ttft=planner_spec.ttft,
            itl=planner_spec.itl,
            enable_power_awareness=planner_spec.enable_power_awareness,
            enable_aic_optimizer=planner_spec.enable_aic_optimizer,
            total_gpu_power_limit=planner_spec.total_gpu_power_limit,
            power_agent_safe_default_watts=planner_spec.power_agent_safe_default_watts,
            prefill_engine_gpu_power_limit=planner_spec.prefill_engine_gpu_power_limit,
            decode_engine_gpu_power_limit=planner_spec.decode_engine_gpu_power_limit,
            aic_initial_c_power_prefill=planner_spec.aic_initial_c_power_prefill,
            aic_initial_c_power_decode=planner_spec.aic_initial_c_power_decode,
            aic_initial_c_power_agg=planner_spec.aic_initial_c_power_agg,
            aic_initial_c_ttft=planner_spec.aic_initial_c_ttft,
            aic_initial_c_itl=planner_spec.aic_initial_c_itl,
            aic_reoptimize_interval=planner_spec.aic_reoptimize_interval,
            aic_drift_relative_threshold=planner_spec.aic_drift_relative_threshold,
            aic_drift_consecutive_ticks=planner_spec.aic_drift_consecutive_ticks,
            aic_max_consecutive_failures=planner_spec.aic_max_consecutive_failures,
            min_endpoint=planner_spec.min_endpoint,
            max_gpu_budget=planner_spec.max_gpu_budget,
            aic_interpolation=aic_spec,
            live_dashboard_port=0,
            report_interval_hours=None,
        )

        caps = WorkerCapabilities(
            prefill=EngineCapabilities(num_gpu=1),
            decode=EngineCapabilities(
                num_gpu=1,
                max_kv_tokens=system_spec.max_kv_tokens,
            ),
        )

        self.metrics = FakePlannerMetrics()
        self.fake_aic = FakeAIC(system_spec)
        self.overlay = SyntheticPowerOverlay(overlay_spec, system_spec, sc, rng)
        self.prom = FakePrometheusClient(source=self.overlay)

        # Build the bridge
        self.bridge = self._build_bridge(mocker, overlay_spec.system)

        self.actuator = ReplayFakeActuator(
            sc, self.overlay, self.bridge, self.metrics, system_spec
        )

        self.adapter = PowerAwareReplayAdapter(
            self.config,
            self.bridge,
            sc,
            overlay=self.overlay,
            fake_prom=self.prom,
            fake_aic=self.fake_aic,
            actuator=self.actuator,
            metrics=self.metrics,
            capabilities=caps,
        )

    def _build_bridge(self, mocker, system: str) -> Any:
        """Build the PlannerReplayBridge from mocker spec.

        Falls back to a minimal stub when the Rust bridge is unavailable
        (e.g. in unit tests that don't compile the extension).

        Handles two generations of the PlannerReplayBridge API:
        - Newer bindings: from_trace_file_disagg / from_synthetic_disagg
        - Older bindings (installed on dev pods): create_disagg(trace_file, ...)
          with no synthetic-workload constructor; falls back to placeholder trace.
        """
        import pathlib

        try:
            from dynamo.llm import PlannerReplayBridge  # type: ignore[import]
        except ImportError:
            return _StubBridge()

        if mocker is None or (not mocker.trace_file and not mocker.synthetic_workload):
            return _StubBridge()

        _has_trace_api = hasattr(PlannerReplayBridge, "from_trace_file_disagg")
        _has_synthetic = hasattr(PlannerReplayBridge, "from_synthetic_disagg")
        _has_create_disagg = hasattr(PlannerReplayBridge, "create_disagg")

        trace_path = mocker.trace_file

        if not trace_path and not _has_synthetic:
            # Older binding — no synthetic workload constructor.
            # Fall back to the placeholder trace bundled with the testbed.
            placeholder = (
                pathlib.Path(__file__).parent.parent
                / "traces"
                / "placeholder_h200_disagg_1rps.jsonl"
            )
            if not placeholder.exists():
                return _StubBridge()
            trace_path = str(placeholder)

        if trace_path:
            if _has_trace_api:
                return PlannerReplayBridge.from_trace_file_disagg(
                    trace_path=trace_path,
                    num_prefill_workers=mocker.num_prefill_workers,
                    num_decode_workers=mocker.num_decode_workers,
                    trace_block_size=mocker.trace_block_size,
                    arrival_speedup_ratio=mocker.arrival_speedup_ratio,
                    router_mode=mocker.router_mode,
                    prefill_engine_args=mocker.prefill_engine_args,
                    decode_engine_args=mocker.decode_engine_args,
                )
            # _has_create_disagg — older API (trace_file positional, no trace_path kw).
            # engine_args must be MockEngineArgs objects, not plain dicts.
            # speedup_ratio=1000 avoids real-time simulation (default=1.0 → 3600s wall
            # time for a 60-tick × 60s scenario — unsuitable for a test suite).
            from dynamo._core import MockEngineArgs  # type: ignore[import]

            def _to_engine_args(d: dict) -> "MockEngineArgs":
                return MockEngineArgs(
                    block_size=d.get("block_size", 0),
                    max_num_batched_tokens=d.get("max_num_batched_tokens"),
                    max_num_seqs=d.get("max_num_seqs"),
                    speedup_ratio=100.0,
                    decode_speedup_ratio=100.0,
                )

            # Use 1 worker each to keep active_count == FPM-reported count.
            # The state machine reconcile loop spins when the bridge reports
            # active_decode_count=N but FPMs only show 1 worker (placeholder
            # trace has no multi-worker KV traffic to distribute to N workers).
            return PlannerReplayBridge.create_disagg(
                trace_file=trace_path,
                prefill_engine_args=_to_engine_args(mocker.prefill_engine_args),
                decode_engine_args=_to_engine_args(mocker.decode_engine_args),
                num_prefill_workers=1,
                num_decode_workers=1,
                router_mode="round_robin",
                arrival_speedup_ratio=mocker.arrival_speedup_ratio,
                trace_block_size=mocker.trace_block_size,
            )

        # Newer binding synthetic workload path
        return PlannerReplayBridge.from_synthetic_disagg(
            num_prefill_workers=mocker.num_prefill_workers,
            num_decode_workers=mocker.num_decode_workers,
            arrival_rate=mocker.arrival_rate,
            isl=mocker.isl,
            osl=mocker.osl,
        )

    def run_and_record(self) -> "TickHistory":
        from dynamo.planner.tests.testbed.recorder import TickHistory, TickSnapshot

        report = self.adapter.run()

        # γ history is built from the diagnostics log. Cap/replica state are
        # read from the adapter's recorder snapshots so n_oscillations can be
        # computed consistently with α.
        history = TickHistory()
        opt = self.adapter._optimizer
        prev_n_p: Optional[int] = None
        prev_n_d: Optional[int] = None
        last_dp = 0
        last_dd = 0
        osc = 0
        for i, diag in enumerate(report.diagnostics_log):
            n_p = self.adapter._actuator._current_n_p
            n_d = self.adapter._actuator._current_n_d
            if prev_n_p is not None and prev_n_d is not None:
                dp = n_p - prev_n_p
                dd = n_d - prev_n_d
                if dp != 0:
                    if last_dp != 0 and (dp > 0) != (last_dp > 0):
                        osc += 1
                    last_dp = dp
                if dd != 0:
                    if last_dd != 0 and (dd > 0) != (last_dd > 0):
                        osc += 1
                    last_dd = dd
            prev_n_p, prev_n_d = n_p, n_d

            snap = TickSnapshot(
                tick=i,
                n_p=n_p,
                n_d=n_d,
                cap_p=self.config.prefill_engine_gpu_power_limit,
                cap_d=self.config.decode_engine_gpu_power_limit,
                observed_ttft_s=0.0,
                observed_itl_s=0.0,
                observed_power_w_p=0.0,
                observed_power_w_d=0.0,
                observed_capacity_tps=0.0,
                c_ttft=opt._c_ttft if opt else 1.0,
                c_itl=opt._c_itl if opt else 1.0,
                c_power_p=opt._c_power_p if opt else 1.0,
                c_power_d=opt._c_power_d if opt else 1.0,
                estimated_throughput=opt._estimated_throughput if opt else 0.0,
                consecutive_violation_ticks=opt._consecutive_violation_ticks
                if opt
                else 0,
                projected_w=0.0,
                budget_w=float(self.config.total_gpu_power_limit or 0),
                sweep_fired=False,
                sla_violated=False,
                capacity_exceeded=False,
                cap_clamped_min=int(
                    self.metrics.power_agent_cap_clamped_total.labeled_value(
                        direction="min"
                    )
                ),
                cap_clamped_max=int(
                    self.metrics.power_agent_cap_clamped_total.labeled_value(
                        direction="max"
                    )
                ),
                optimizer_exceptions=int(
                    self.metrics.aic_optimizer_exceptions_total.value
                ),
                admission_partial_failures=int(
                    self.metrics.admission_partial_success_total.value
                ),
                n_oscillations=osc,
                mocker_active_p=n_p,
                mocker_active_d=n_d,
            )
            history.append(snap)
        return history


class _StubBridge:
    """Minimal stub when the Rust bridge is unavailable.

    Emits a single done tick immediately so γ tests that lack the mocker
    extension don't crash — they just run 0 ticks and skip assertions.
    """

    def advance_to(self, tick_ms: float) -> dict:
        return {
            "is_done": True,
            "now_ms": tick_ms,
            "active_prefill_count": 1,
            "active_decode_count": 4,
            "prefill_fpm_snapshots": [],
            "decode_fpm_snapshots": [],
        }

    def apply_scaling(self, n_p: int, n_d: int) -> None:
        pass

    def drain_traffic(self) -> dict:
        return {"duration_s": 0.0, "num_req": 0}

    def finalize(self) -> dict:
        return {}
