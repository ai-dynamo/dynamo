# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ScenarioRunner — dispatches α and γ scenarios, runs the closed-loop tick.

α-class per-tick data flow (§2.1 of testbed design):
  1. Apply scenario events for this tick
  2. Get offered load from load profile
  3. SyntheticFleet.step() → Observation
  4. FakePrometheus updates (via source)
  5. optimizer.update_correction(...)
  6. if optimizer.should_reoptimize(): optimize() → apply caps
  7. state_machine._apply_power_budget(desired_p, desired_d) → clamped
  8. FakeActuator.apply_replicas(...)
  9. TickRecorder.record(...)

γ-class delegates to PowerAwareReplayAdapter.run() after setup.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dynamo.planner.tests.testbed.recorder import TickHistory
    from dynamo.planner.tests.testbed.scenarios import ScenarioSpec

logger = logging.getLogger(__name__)


class ScenarioRunner:
    """Main harness for a single scenario.

    Usage::

        spec = load_scenario("scenarios/A1_power_under_estimate_decode.yaml")
        runner = ScenarioRunner(spec)
        history = runner.run()
    """

    def __init__(self, scenario: "ScenarioSpec") -> None:
        self.scenario = scenario

    # ------------------------------------------------------------------
    # α-class setup
    # ------------------------------------------------------------------

    def _setup_alpha(self):
        """Build all α-class fakes and inject seams into production code."""
        from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
        from dynamo.planner.config.parallelization import PickedParallelConfig
        from dynamo.planner.config.planner_config import PlannerConfig
        from dynamo.planner.core.state_machine import PlannerStateMachine
        from dynamo.planner.core.types import EngineCapabilities, WorkerCapabilities
        from dynamo.planner.monitoring.aic_power_optimizer import AICPowerOptimizer
        from dynamo.planner.tests.testbed.clock import Clock
        from dynamo.planner.tests.testbed.fake_actuator import FakeActuator
        from dynamo.planner.tests.testbed.fake_aic import FakeAIC
        from dynamo.planner.tests.testbed.fake_planner_metrics import FakePlannerMetrics
        from dynamo.planner.tests.testbed.fake_prometheus import FakePrometheusClient
        from dynamo.planner.tests.testbed.recorder import TickHistory
        from dynamo.planner.tests.testbed.scenarios import SystemSpec
        from dynamo.planner.tests.testbed.synthetic_fleet import SyntheticFleet

        sc = self.scenario
        rng = random.Random(sc.seed)

        system_spec = SystemSpec.load(sc.fleet.system)

        self.fleet = SyntheticFleet(sc.fleet, system_spec, sc, rng)
        self.metrics = FakePlannerMetrics()
        self.actuator = FakeActuator(sc, self.fleet, self.metrics, system_spec)
        self.prom = FakePrometheusClient(source=self.fleet)
        self.fake_aic = FakeAIC(system_spec)
        self.clock = Clock(interval_s=sc.interval_s)

        # Build PlannerConfig from scenario planner spec
        planner_spec = sc.planner
        aic_spec = AICInterpolationSpec(
            hf_id="fake-model/testbed",
            system=sc.fleet.system,
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
            # disable report generation in testbed
            live_dashboard_port=0,
            report_interval_hours=None,
        )

        # Build WorkerCapabilities for state machine
        caps = WorkerCapabilities(
            prefill=EngineCapabilities(num_gpu=sc.fleet.gpus_per_prefill_engine),
            decode=EngineCapabilities(num_gpu=sc.fleet.gpus_per_decode_engine),
        )
        self.state_machine = PlannerStateMachine(self.config, caps)

        # Parse events once at setup — Pydantic model construction is the
        # bottleneck for long scenarios (200 ticks × N events × 2 reparse).
        self._events = sc.parsed_events()

        # ------------------------------------------------------------------
        # Step 1 — process environmental events scheduled at tick 0 BEFORE
        # the cold-start sweep. These represent cluster state present at
        # planner startup (e.g. an NVML clamp fault is already in effect when
        # the planner boots). Without this, ``apply_caps()`` in setup never
        # sees the fault and B7/B8-style clamp scenarios silently no-op.
        # We re-fire these in ``_tick_alpha`` (set_actuation_fault etc. are
        # idempotent) so existing event-driven assertions still see them.
        # ------------------------------------------------------------------
        from dynamo.planner.tests.testbed.scenarios import (
            ActuationFaultEvent,
            AicFailureEvent,
            BudgetChangeEvent,
            FrontendPostFaultEvent,
        )

        for event in self._events:
            if getattr(event, "at_tick", None) != 0:
                continue
            if isinstance(event, ActuationFaultEvent):
                self.actuator.set_actuation_fault(event.mode)
            elif isinstance(event, AicFailureEvent):
                self.fake_aic.set_fault_mode(event.mode)
                self._aic_fault_reset_tick = event.n_consecutive
            elif isinstance(event, BudgetChangeEvent):
                self.config.total_gpu_power_limit = event.new_total_w
            elif isinstance(event, FrontendPostFaultEvent):
                # Install the frontend POST fault on the fleet so the
                # cold-start /busy_threshold fan-out can hit it.
                self.fleet.apply_event(event, 0)

        # ------------------------------------------------------------------
        # Step 2 — apply the *configured* per-engine caps first. Production
        # planner does this via ``_apply_power_annotations`` on the first
        # tick (before the AIC optimizer's first sweep returns). Doing it
        # here lets B7/B8 actually exercise NVML clamping when the
        # configured value lies outside [sku_min, sku_max].
        # ------------------------------------------------------------------
        self.actuator.apply_caps(
            int(planner_spec.prefill_engine_gpu_power_limit),
            int(planner_spec.decode_engine_gpu_power_limit),
        )

        if planner_spec.enable_aic_optimizer:
            # Patch time.monotonic with the virtual clock BEFORE constructing
            # the optimizer so its initial _time_of_last_optimize reads 0.0 in
            # virtual time, not wall-clock time.
            import unittest.mock as mock

            self._clock_patch = mock.patch("time.monotonic", side_effect=self.clock.now)
            self._clock_patch.start()

            self.optimizer = AICPowerOptimizer(self.config, self.metrics)
            self.optimizer._aic_estimator_factory = (
                self.fake_aic.make_estimator_factory()
            )

            # Run the cold-start sweep — mirrors production base.py setup_async()
            # and γ-adapter PowerAwareReplayAdapter.__init__. Without this,
            # _last_optimal_config stays None and update_correction() is a
            # permanent no-op (every α scenario silently fails to drive c_*).
            initial = self.optimizer.optimize()
            if initial is not None:
                self.config.prefill_engine_gpu_power_limit = initial.cap_p
                self.config.decode_engine_gpu_power_limit = initial.cap_d
                self.actuator.apply_caps(initial.cap_p, initial.cap_d)
                # Seed truth-side replica counts AFTER the power-budget clamp
                # so subsequent ticks start at a feasible point. Using the
                # raw optimizer pick (initial.n_p/n_d) overruns budget — the
                # state machine clamps to (final_p, final_d). Apply that
                # clamp now so the truth state matches production behavior.
                final_p, final_d = self.state_machine._apply_power_budget(
                    initial.n_p, initial.n_d
                )
                # Set ``_estimated_throughput`` from the **post-budget-clamp**
                # replica count, not the optimizer's raw pick. Production has
                # the same shape but the discrepancy is hidden by other layers;
                # here, the synthetic fleet's truth capacity is tied directly
                # to ``n_d_truth``, so without this fix the drift detector
                # compares observed traffic against a fantasy capacity and
                # the capacity_exceeded trigger never fires (F26, C13).
                self.optimizer._estimated_throughput = (
                    initial.aic_seq_per_s_per_replica
                    * final_d
                    * (initial.isl + initial.osl)
                )
                self.fleet.state.n_p_truth = final_p
                self.fleet.state.n_d_truth = final_d
                self.actuator.apply_replicas(final_p, final_d)
                if sc.planner.admission_mode == "autoset":
                    self._fanout_busy_threshold_posts()
        else:
            self.optimizer = None
            self._clock_patch = None

        self.history = TickHistory()

    # ------------------------------------------------------------------
    # α-class tick loop
    # ------------------------------------------------------------------

    def _tick_alpha(self, tick: int) -> None:
        from dynamo.planner.tests.testbed.recorder import TickSnapshot
        from dynamo.planner.tests.testbed.scenarios import (
            ActuationFaultEvent,
            AicFailureEvent,
            BudgetChangeEvent,
        )

        sc = self.scenario
        self.clock.advance(tick)
        self.prom.set_tick(tick)

        # Apply events scheduled at this tick
        for event in self._events:
            event_at = getattr(event, "at_tick", None)
            if event_at != tick:
                continue
            if isinstance(event, AicFailureEvent):
                if self.fake_aic:
                    self.fake_aic.set_fault_mode(event.mode)
                    self._aic_fault_reset_tick = tick + event.n_consecutive
            elif isinstance(event, BudgetChangeEvent):
                self.config.total_gpu_power_limit = event.new_total_w
            elif isinstance(event, ActuationFaultEvent):
                self.actuator.set_actuation_fault(event.mode)
            else:
                self.fleet.apply_event(event, tick)

        # Clear expired AIC fault
        if (
            hasattr(self, "_aic_fault_reset_tick")
            and tick >= self._aic_fault_reset_tick
        ):
            if self.fake_aic:
                self.fake_aic.reset_fault()

        # Clear expired fleet events
        self.fleet.clear_expired_events(tick)

        # Clear expired actuation fault
        for event in self._events:
            if not isinstance(event, ActuationFaultEvent):
                continue
            if event.at_tick + event.duration_ticks <= tick:
                self.actuator.set_actuation_fault(None)

        # Step the fleet
        offered_load = sc.offered_load_at(tick)
        obs = self.fleet.step(tick, offered_load)

        # Capture only pre-tick pegged baselines (we want per-tick *event*
        # semantics for ``correction_pegged`` — "did the coefficient clamp
        # *this tick*"). Other counter snapshot fields below use cumulative
        # values (natural Prometheus counter semantics) so assertions like
        # ``cap_clamped_min > 0 at tick 10`` read "has clamping happened by
        # tick 10", which matches the scenarios' intent.
        before_pegged = dict(self._read_pegged_counters())

        # AIC optimizer update + potential re-sweep
        sweep_fired = False
        # Default desired = current truth state; sweep can override below.
        desired_p = self.fleet.state.n_p_truth
        desired_d = self.fleet.state.n_d_truth

        if self.optimizer is not None and obs.traffic.num_req is not None:
            self.optimizer.update_correction(
                traffic=obs.traffic,
                observed_ttft_avg=obs.ttft_avg_s,
                observed_itl_avg=obs.itl_avg_s,
                observed_power_w_prefill=obs.power_w_prefill,
                observed_power_w_decode=obs.power_w_decode,
            )
            # When admission_mode == "autoset", production's
            # ``_apply_aic_config`` fans out a /busy_threshold POST to every
            # frontend pod after every sweep AND on cold-start, then the
            # planner relies on those pods enforcing the threshold. The
            # testbed mirrors that fan-out (3 synthetic pods) only on
            # sweeps so the B11 frontend-POST-fault counter is reachable.
            if self.optimizer.should_reoptimize(obs.traffic):
                new_cfg = self.optimizer.optimize()
                if new_cfg is not None:
                    self.config.prefill_engine_gpu_power_limit = new_cfg.cap_p
                    self.config.decode_engine_gpu_power_limit = new_cfg.cap_d
                    self.actuator.apply_caps(new_cfg.cap_p, new_cfg.cap_d)
                    # Honor the optimizer's replica recommendation — this is
                    # the path that exercises _apply_power_budget's clamp +
                    # min_endpoint enforcement (scenarios A6, E22, E25).
                    desired_p = new_cfg.n_p
                    desired_d = new_cfg.n_d
                    # Estimated throughput uses the *post-budget-clamp* n_d
                    # so drift detection compares against achievable capacity.
                    # See note in ``_setup_alpha`` for the production-divergence
                    # rationale (testbed scenarios assume post-clamp semantics).
                    (
                        sweep_final_p,
                        sweep_final_d,
                    ) = self.state_machine._apply_power_budget(new_cfg.n_p, new_cfg.n_d)
                    self.optimizer._estimated_throughput = (
                        new_cfg.aic_seq_per_s_per_replica
                        * sweep_final_d
                        * (new_cfg.isl + new_cfg.osl)
                    )
                    sweep_fired = True

                    # Fan out /busy_threshold POSTs for autoset admission.
                    # Pure synthetic — no real network — but exercises the
                    # frontend-POST fault path (B11) and respects asyncio
                    # boundary by gathering results.
                    if sc.planner.admission_mode == "autoset":
                        self._fanout_busy_threshold_posts()

        # Power budget clamp
        final_p, final_d = self.state_machine._apply_power_budget(desired_p, desired_d)

        # Apply replicas (may fault)
        try:
            self.actuator.apply_replicas(final_p, final_d)
        except RuntimeError:
            pass  # RBAC fault; state unchanged

        # Compute projected power
        cap_p = self.fleet.state.applied_cap_p
        cap_d = self.fleet.state.applied_cap_d
        gpus_p = sc.fleet.gpus_per_prefill_engine
        gpus_d = sc.fleet.gpus_per_decode_engine
        projected_w = (
            self.fleet.state.n_p_truth * cap_p * gpus_p
            + self.fleet.state.n_d_truth * cap_d * gpus_d
        )

        # Read optimizer state
        c_ttft = c_itl = c_power_p = c_power_d = 1.0
        estimated_throughput = 0.0
        consecutive_violations = 0
        if self.optimizer:
            c_ttft = self.optimizer._c_ttft
            c_itl = self.optimizer._c_itl
            c_power_p = self.optimizer._c_power_p
            c_power_d = self.optimizer._c_power_d
            estimated_throughput = self.optimizer._estimated_throughput
            consecutive_violations = self.optimizer._consecutive_violation_ticks

        after_pegged = self._read_pegged_counters()
        pegged_delta = {
            k: after_pegged.get(k, 0) - before_pegged.get(k, 0) for k in after_pegged
        }

        n_oscillations = self._update_oscillation_count(
            self.fleet.state.n_p_truth, self.fleet.state.n_d_truth
        )

        snap = TickSnapshot(
            tick=tick,
            n_p=self.fleet.state.n_p_truth,
            n_d=self.fleet.state.n_d_truth,
            cap_p=self.fleet.state.applied_cap_p,
            cap_d=self.fleet.state.applied_cap_d,
            observed_ttft_s=obs.ttft_avg_s,
            observed_itl_s=obs.itl_avg_s,
            observed_power_w_p=obs.power_w_prefill,
            observed_power_w_d=obs.power_w_decode,
            observed_capacity_tps=obs.total_tokens_per_sec,
            c_ttft=c_ttft,
            c_itl=c_itl,
            c_power_p=c_power_p,
            c_power_d=c_power_d,
            estimated_throughput=estimated_throughput,
            consecutive_violation_ticks=consecutive_violations,
            projected_w=projected_w,
            budget_w=float(self.config.total_gpu_power_limit or 0),
            sweep_fired=sweep_fired,
            sla_violated=(
                obs.ttft_avg_s > (sc.planner.ttft / 1000.0)
                or obs.itl_avg_s > (sc.planner.itl / 1000.0)
            ),
            capacity_exceeded=obs.total_tokens_per_sec > (estimated_throughput * 1.15)
            if estimated_throughput > 0
            else False,
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
            optimizer_exceptions=int(self.metrics.aic_optimizer_exceptions_total.value),
            correction_pegged={k: v for k, v in pegged_delta.items() if v > 0},
            admission_partial_failures=int(
                self.metrics.admission_partial_success_total.value
            ),
            n_oscillations=n_oscillations,
        )
        self.history.append(snap)

    # Synthesised count of frontend pods receiving /busy_threshold POSTs.
    # 3 is enough to make the per-call Bernoulli fault model converge to
    # the configured failing_fraction over a 10-tick window.
    _FRONTEND_POD_COUNT = 3

    def _fanout_busy_threshold_posts(self) -> None:
        """Synthetic equivalent of ``base.py::_apply_aic_config``'s POST fan-out.

        Calls ``actuator.post_busy_threshold`` for each synthetic frontend
        pod. Exceptions (synthetic 503s under FrontendPostFaultEvent) are
        swallowed here — they're already accounted for in the
        admission_partial_success_total counter inside the actuator.
        """
        import asyncio

        async def _gather():
            await asyncio.gather(
                *(
                    self.actuator.post_busy_threshold(
                        pod=f"frontend-{i}",
                        model="fake-model/testbed",
                        port=8080,
                        active_decode_blocks_threshold=0.97,
                        active_prefill_tokens_threshold=4096,
                        active_prefill_tokens_threshold_frac=1.0,
                    )
                    for i in range(self._FRONTEND_POD_COUNT)
                ),
                return_exceptions=True,
            )

        try:
            asyncio.run(_gather())
        except RuntimeError:
            # Caller already running event loop (γ adapter); fall through
            # via run-until-complete on the loop.
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_gather())
            finally:
                loop.close()

    def _update_oscillation_count(self, n_p: int, n_d: int) -> int:
        """Maintain a cumulative replica direction-flip counter.

        Increments when the sign of (n_p_now − n_p_prev) flips relative to the
        previous non-zero delta — same for n_d. Used by G2's "at most 3
        flip-flops" assertion.
        """
        prev_snap = self.history.snapshots[-1] if self.history.snapshots else None
        if prev_snap is None:
            self._osc_count = 0
            self._osc_last_dp = 0
            self._osc_last_dd = 0
            self._osc_prev_n_p = n_p
            self._osc_prev_n_d = n_d
            return 0

        dp = n_p - self._osc_prev_n_p
        dd = n_d - self._osc_prev_n_d

        if dp != 0:
            if self._osc_last_dp != 0 and (dp > 0) != (self._osc_last_dp > 0):
                self._osc_count += 1
            self._osc_last_dp = dp
        if dd != 0:
            if self._osc_last_dd != 0 and (dd > 0) != (self._osc_last_dd > 0):
                self._osc_count += 1
            self._osc_last_dd = dd

        self._osc_prev_n_p = n_p
        self._osc_prev_n_d = n_d
        return self._osc_count

    def _read_pegged_counters(self) -> dict[str, float]:
        c = self.metrics.aic_correction_pegged_total
        result = {}
        for key, labeled in c._labels.items():
            coeff = dict(key).get("coefficient", "unknown")
            result[coeff] = labeled.value
        return result

    # ------------------------------------------------------------------
    # γ-class setup + run
    # ------------------------------------------------------------------

    def _setup_and_run_gamma(self) -> "TickHistory":
        from dynamo.planner.tests.testbed.replay.power_aware_replay_adapter import (
            build_gamma_adapter,
        )

        return build_gamma_adapter(self.scenario).run_and_record()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        csv_path: Optional[Path] = None,
        prom_path: Optional[Path] = None,
        plot_path: Optional[Path] = None,
    ) -> "TickHistory":
        sc = self.scenario
        t0 = time.perf_counter()

        if sc.class_name == "alpha":
            self._setup_alpha()
            for tick in range(sc.ticks):
                self._tick_alpha(tick)
            if self._clock_patch:
                self._clock_patch.stop()
            history = self.history
        else:
            history = self._setup_and_run_gamma()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Scenario %s completed %d ticks in %.2fs", sc.name, len(history), elapsed
        )

        if csv_path:
            history.to_csv(Path(csv_path))
        if prom_path:
            history.to_prom_textfile(Path(prom_path), sc.name)
        if plot_path:
            history.plot(Path(plot_path), sc.name)

        return history


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Power Planner stress testbed runner",
    )
    parser.add_argument("--scenario", help="Scenario name (e.g. A1) or full YAML path")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument(
        "--class-filter", choices=["alpha", "gamma", "all"], default="all"
    )
    parser.add_argument("--csv", help="Output CSV path (single scenario)")
    parser.add_argument("--csv-dir", help="Output CSV directory (--all mode)")
    parser.add_argument("--plot", help="Output PNG path (single scenario)")
    parser.add_argument("--prom-textfile", help="Prometheus textfile output path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    from dynamo.planner.tests.testbed.scenarios import load_all_scenarios, load_scenario

    scenarios_dir = Path(__file__).parent / "scenarios"

    if args.all:
        results = []
        all_specs = load_all_scenarios()
        for name, spec in all_specs:
            if args.class_filter != "all" and spec.class_name != args.class_filter:
                continue
            csv_path = Path(args.csv_dir) / f"{name}.csv" if args.csv_dir else None
            runner = ScenarioRunner(spec)
            try:
                runner.run(csv_path=csv_path)
                results.append((name, "PASS"))
            except Exception as e:
                results.append((name, f"FAIL: {e}"))
        for name, status in results:
            print(f"  {status:8s} {name}")
        failed = [r for r in results if not r[1].startswith("PASS")]
        if failed:
            raise SystemExit(f"{len(failed)} scenario(s) FAILED")
    else:
        if not args.scenario:
            parser.error("Either --scenario or --all is required")
        # Resolve scenario
        if Path(args.scenario).exists():
            path = Path(args.scenario)
        else:
            # Search by prefix
            candidates = list(scenarios_dir.glob(f"{args.scenario}*.yaml"))
            if not candidates:
                raise SystemExit(
                    f"No scenario matching {args.scenario!r} in {scenarios_dir}"
                )
            path = candidates[0]
        spec = load_scenario(path)
        runner = ScenarioRunner(spec)
        runner.run(
            csv_path=args.csv,
            prom_path=args.prom_textfile,
            plot_path=args.plot,
        )
        print(f"PASS {spec.name}")


if __name__ == "__main__":
    main()
