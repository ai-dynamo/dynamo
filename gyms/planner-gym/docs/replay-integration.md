<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Arena replay integration with Dynamo telemetry v1

Planner Gym uses the native replay runtime and Planner preparation helpers
from the same Dynamo checkout. Rebuild the binding after updating Rust or
AISimulate dependencies; Python-only installs can leave an incompatible native
runtime behind. The setup script checks the scaling-policy and telemetry APIs.

## Integration boundary

`run_arena_replay()` owns the policy selection and reporting additions:

1. It lowers engine arguments with `dynamo.replay.config.load_engine_args()`
   and reuses the capability/FPM helpers from Dynamo's current
   `dynamo.replay.planner` preparation module. This avoids the removed replay
   CLI module without copying logic that could drift from Dynamo.
2. It calls the selected Arena policy factory with the shared `PlannerConfig`
   and capabilities. The built-in Planner factory constructs Dynamo's native
   `OrchestratorEngineAdapter` with a `VirtualClock`; static, reactive, and KEDA
   factories implement the same `EngineProtocol`.
3. It passes the resulting engine to `ReplayPlannerAdapter` and enters the
   adapter as a context manager. For the builtin Planner, Arena loads configured
   predictor-warmup observations and mirrors current Dynamo selection of
   runner-neutral performance-model metadata, role-specific or shared AISimulate
   sessions, bootstrap FPMs, and bootstrap provenance. Rival engines explicitly
   opt out of that Planner-only preparation.
4. It calls `dynamo._core.run_mocker_trace_replay()` with the adapter as
   `scaling_policy`, `capture_telemetry=False`, and a distinct
   `telemetry_jsonl_path`. The Rust runtime owns replay progress, invokes the
   policy at controller ticks, applies returned scaling targets, and streams
   policy-neutral telemetry snapshots directly to that file.
5. While the adapter is active, it finalizes Planner details from the lifecycle
   operations and combines the native summary, coverage, and per-request rows
   into Dynamo's canonical `ReplayReport`. Arena then parses the completed
   telemetry JSONL and projects those rows plus exact Planner decision markers
   into its report timeline.

Conceptually, the call path is:

```python
engine = policy_factory(planner_config, capabilities)
adapter = ReplayPlannerAdapter(
    planner_config=planner_config,
    engine=engine,
    capabilities=capabilities,
)

with adapter:
    native = run_mocker_trace_replay(
        trace_paths,
        scaling_policy=adapter,
        capture_per_request=True,
        capture_telemetry=False,
        telemetry_sample_interval_ms=5_000.0,
        telemetry_jsonl_path=run_dir / "telemetry.jsonl",
        # shared engine, router, replay, and SLA arguments
    )
    replay_report = ReplayReport(
        summary=native.summary,
        per_request=native.per_request,
        coverage=native.coverage,
        planner=adapter.finalize(native.lifecycle_operations),
    )

telemetry_samples = read_telemetry_jsonl(run_dir / "telemetry.jsonl")
timeline = build_timeline(telemetry_samples, replay_report.planner)
```

The telemetry stream is separate from controller ticks and from the Planner
callback contract. Provisioned counts remain active + starting + draining for
cost and capacity reporting. Exact decision markers come from finalized Planner
details; queue, traffic, cache, and fleet observations come only from JSONL.
Malformed JSONL fails the run rather than silently producing a partial chart.

Arena identifies this persisted boundary as
`dynamo.replay.telemetry.v1`. It validates and projects one UTF-8 JSON object at
a time, retaining report rows rather than a second list of raw snapshots. V1
requires contiguous ordinals starting at zero, a first `baseline`, optional
`periodic` samples, and an optional last `final`; sample timestamps are
monotonically non-decreasing and each interval start is no later than its
sample timestamp. The traffic object, both interval-counter objects, both
scheduler-row arrays, pending counts, and all six fleet-ID arrays are required
and type-checked. Unknown fields remain legal for forward-compatible Dynamo
additions. Match results save the contract name, sample count, and SHA-256 of
the exact JSONL bytes. A telemetry destination that aliases a trace input or
the trace-report destination is rejected before replay can overwrite it.

## Planner preparation parity

Arena cannot invoke `dynamo.replay.planner.planner_replay_adapter()` as one
opaque operation: that factory always creates the builtin Planner, while the
leaderboard must wrap Planner, KEDA, reactive, and static engines behind the
same replay-policy seam. Instead, Arena keeps a deliberately narrow parity
helper for the builtin Planner path:

- configured `load_predictor_warmup_trace` data is converted with Dynamo's
  `extract_traffic_observations_from_trace()` and passed into
  `ReplayPlannerAdapter`;
- Match Config preserves role-specific identities through Dynamo's canonical AISimulate configuration; omitted AISimulate versions are backfilled per
  role from Dynamo's lowered engine arguments;
- Dynamo's current performance-model selection, session-argument, FPM
  generation, and digest helpers remain authoritative;
- one AISimulate session is shared only when the resolved prefill and decode session
  arguments are equal; and
- bootstrap status metadata and load-only failure fallback match
  `prepare_planner_replay()`.

This claim is limited to those preparation behaviors. Arena still owns the
generic engine factory and low-level `scaling_policy` composition needed for
cross-autoscaler comparison. Focused tests cover warmup routing, metadata
selection, shared versus role-specific sessions, and rival opt-out behavior.

## Arena-owned reporting

Dynamo's canonical report remains the source of truth. Arena-only additions are
composed in `ArenaReplayResult`:

- `replay_report`: Dynamo's canonical `ReplayReport` containing the summary,
  coverage, optional per-request records, and Planner details. Its optional
  in-memory telemetry field remains unset.
- `timeline`: backend-neutral traffic, fleet, and decision samples.
- `per_request`: a convenience view of `replay_report.per_request`.
- `gpu_hours`: a convenience view of the authoritative
  `replay_report.summary["gpu_hours"]` value.

With `capture_per_request=True`, the low-level runtime returns deterministic
terminal request records directly in memory. The Arena uses those rows for
post-hoc SLO scoring. Rejected, canceled, and failed terminal records remain
visible and do not count as good requests. With capture disabled,
`per_request` is `None`; callers may use the runtime's single-SLA `goodput_*`
summary for the SLA supplied to replay.

GPU-hour accounting comes directly from `ReplayReport.summary`. The Rust
runtime integrates provisioned GPU allocation over replay time; the Arena does
not maintain a competing accumulator.

## Repository ownership

The Dynamo/Arena boundary is intentionally narrow:

- Dynamo owns the replay clock, request simulation, fleet lifecycle,
  `scaling_policy` callbacks, terminal request capture, summary metrics,
  telemetry JSONL emission, and Planner adapter lifecycle.
- The Arena owns policy factories, rival policy implementations, FPM
  aggregation for those rivals, AISimulate bootstrap selection, strict telemetry
  JSONL ingestion and timeline projection, and multi-profile scoring.

Rival engines expose a no-op `install_regressions_from_fpms()` method to match
the replay adapter's extended engine surface. Their
`supports_ais_bootstrap = False` marker skips AISimulate benchmark generation before
that no-op; only the Planner engine consumes fitted regression data.

This keeps the upstream adapter unchanged while allowing every policy to run
against the same substrate and callback contract.

## Build requirement

Rebuild the Python/Rust binding after changing Dynamo dependencies or Rust
sources. For a clean build from `gyms/planner-gym`, move any existing `.venv` aside, then let the setup script create
the Python environment, build the native runtime before the editable Dynamo
package, install Arena, and check the replay contract:

```bash
bash scripts/setup_simulation_env.sh
source .venv/bin/activate
```

Building first matters when the source checkout requires an exact
`ai-dynamo-runtime` version that has not been published as a wheel.

The script verifies both entry points. The public wrapper deliberately groups
file and callback sinks under `telemetry_options`; only the low-level binding
exposes `telemetry_jsonl_path` and `telemetry_sample_interval_ms` as direct
arguments.

AISimulate dependency pins and performance-data availability are separate substrate
concerns; they do not change the unified replay-policy boundary described here.
