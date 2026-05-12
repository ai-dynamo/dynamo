# Plugin Proto v1

Plugin contract for **DEP-XXXX Dynamo Planner Plugin Architecture** (v11).

This directory contains:

| File | Purpose |
|---|---|
| `plugin.proto` | Single-source-of-truth proto3 schema |
| `plugin_pb2.py` | Generated protobuf Python stubs (check-in; CI verifies regen drift) |
| `plugin_pb2_grpc.py` | Generated gRPC client/server stubs |
| `plugin_pb2.pyi` | Generated type stubs for IDE / mypy |
| `__init__.py` | Module marker |

## Schema overview

Total: **6 services / 33 messages / 3 enums**

### Services

| Service | RPCs | Owner |
|---|---|---|
| `PluginRegistry` | `Register` / `Heartbeat` / `Unregister` / `ListPlugins` | Orchestrator-side; plugins call to register / report liveness |
| `PluginLifecycle` | `Bootstrap` / `Reset` | Plugin-side; orchestrator calls to prime / clear plugin state |
| `PredictPlugin` | `Predict` | Plugin-side; chain-augment partial-merge per PREDICT spec |
| `ProposePlugin` | `Propose` | Plugin-side; type-aware merge per PROPOSE spec |
| `ReconcilePlugin` | `Reconcile` | Plugin-side; type-aware merge per RECONCILE spec |
| `ConstrainPlugin` | `Constrain` | Plugin-side; type-aware merge (set_allowed=False) per CONSTRAIN spec |

### Enums

| Enum | Values | Notes |
|---|---|---|
| `HoldPolicy` | `ACCEPT_WHEN_IDLE` (0) / `HOLD_LAST` (1) | Default 0 = no opinion between invocations |
| `OverrideType` | `SET` (0) / `AT_LEAST` (1) / `AT_MOST` (2) | Used in `ComponentTarget.type` |
| `CircuitState` | `CLOSED` (0) / `OPEN` (1) / `HALF_OPEN` (2) | Used in `PluginInfo.circuit_state` |

### Messages — by category

- **PluginRegistry**: `RegisterRequest` / `RegisterResponse` / `HeartbeatRequest` / `HeartbeatResponse` / `UnregisterRequest` / `UnregisterResponse` / `ListPluginsRequest` / `ListPluginsResponse` / `PluginInfo`
- **PipelineContext + observation**: `PipelineContext` / `ObservationData` / `TrafficMetrics` / `FpmData` / `WorkerState` / `PredictionData` / `ScalingProposal` / `ComponentTarget` / `OverrideResult` / `AcceptResult` / `RejectResult`
- **Stage request/response**: `PredictStageRequest` / `PredictStageResponse` / `ProposeStageRequest` / `ProposeStageResponse` / `ProposeResult` / `ReconcileStageRequest` / `ReconcileStageResponse` / `ConstrainStageRequest` / `ConstrainStageResponse`
- **PluginLifecycle**: `BootstrapRequest` / `BootstrapResponse` / `ResetRequest` / `ResetResponse`

## Generation

```bash
# Regenerate all stubs (plugin_pb2.py / plugin_pb2_grpc.py / plugin_pb2.pyi)
tools/build/gen_planner_proto.sh

# CI mode: regenerate to temp dir + diff against committed; non-zero exit on drift
tools/build/gen_planner_proto.sh --check
```

CI integration: existing `planner-build` job in `.github/workflows/pr.yaml`
runs `--check` to detect proto-vs-generated drift. Generated `*.py` files
are checked into git for two reasons:

1. **Import friendliness** — modules import `from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb` directly without build-step preconditions
2. **Drift catching** — `git diff --exit-code` after regen surfaces accidental
   "edited proto but forgot to regen" mistakes in PR review

## Schema evolution policy (proto3, must-follow)

1. **NEVER reuse a field tag** — always add `reserved` for any deleted tag
2. **NEVER change the type** of an existing field
3. **NEVER rename** an existing field (clients may key on field names via
   reflection / JSON transcoding)
4. **ALL new fields MUST be optional** or have safe-zero defaults
5. **Bumping `protocol_version`** (in `RegisterRequest.protocol_version`)
   is reserved for *additive* contract changes; *breaking* changes require
   a new package path (`v2/`)

These rules are enforced by reviewer judgment + the round-trip test suite
in `tests/plugins/proto/test_round_trip.py` (any new message added to the
proto must be added to the Pydantic mirror in `plugins/types.py`, otherwise
`test_class_coverage_proto_side` fails CI).

## Critical schema invariants (v11 review)

These are not mere conventions — they are required by downstream PR
algorithms; violating them silently breaks the architecture.

### `PredictionData` fields MUST be `optional float`

```proto
message PredictionData {
  optional float predicted_num_req = 1;  // unset → preserve prev in chain-augment
  optional float predicted_isl     = 2;
  optional float predicted_osl     = 3;
  string source                    = 4;
}
```

PR 4 chain-augment partial-merge uses `HasField()` to distinguish:

- `field set` → plugin actively asserts this value (even `0.0`)
- `field unset` → plugin has no opinion; preserve previous chain plugin's value

Without `optional`, proto3 default `0.0` makes "I assert 0" indistinguishable
from "I have no opinion", breaking the layered-predictor pattern documented
in DEP main doc (e.g. `user-llm-predictor` outputs `(num_req=1200)`, leaving
`isl` / `osl` from the upstream `builtin-load-predictor`).

### CONSTRAIN `SET` is silently dropped at runtime (NOT register-time rejected)

```proto
message ConstrainStageResponse {
  oneof result { ... }
  bool final = 4;  // SILENTLY IGNORED
}
```

v11 decision: `ConstrainStageResponse.override` carrying `OverrideType.SET`
is silently dropped at runtime; `final=true` is silently ignored.
Register-time static rejection is infeasible because proto3 has no
plugin-self-declared output-type metadata.

If your CONSTRAIN plugin needs to "win", tighten the bound:
- larger `AT_LEAST` (raises floor)
- smaller `AT_MOST` (lowers ceiling)

`max` / `min` monotonicity guarantees your bound always participates.

### `result` oneof empty = plugin contract violation

For `Propose` / `Reconcile` / `Constrain` stage responses, `WhichOneof("result")`
returning `None` (plugin forgot to set `accept` / `override` / `reject`) is
**NOT** treated as ACCEPT — the orchestrator (PR 5) raises
`PluginSerializationError`, triggers circuit breaker, and surfaces the
plugin bug in audit log + Prometheus.

### `final=true` semantics differ between PREDICT and PROPOSE/RECONCILE

| Stage | `final=true` rule |
|---|---|
| `PROPOSE` / `RECONCILE` | priority number smallest (= highest priority) wins |
| `PREDICT` (chain-augment) | first `final=true` in chain wins (chain ordered priority-descending → first-encountered = lowest priority) |

**Strong contract for PREDICT**: `final=true` in PREDICT plugin MUST be
configured with priority = lowest number (highest priority). Otherwise
the chain breaks BEFORE higher-priority plugins run. PR 4 4-5 implements
runtime detection + `WARNING` log + Prometheus
`predict_chain_final_at_non_lowest_priority_total{plugin_id}`.

### `final=true` does NOT skip CONSTRAIN

Even when a `PROPOSE` / `RECONCILE` plugin sets `final=true`, the CONSTRAIN
stage runs normally. `builtin-budget-constrain` always provides
`AT_LEAST(min_endpoint)` + `AT_MOST(max_gpu_budget)` as the safety net; no
`final` can bypass it.

### REJECT > final priority

If any plugin returns `RejectResult` in the same stage, the entire stage
short-circuits — even when `final=true` plugins are also present. This
matches K8s admission controller `deny > allow` semantics: safety override
is higher priority than authority override.

## Adding a new stage / RPC / message

1. Edit `plugin.proto` following the schema evolution policy above
2. Add corresponding Pydantic mirror class in `plugins/types.py`
3. Register `(Pydantic, proto)` pair in `_PYD_TO_PROTO` dict in
   `plugins/_proto_bridge.py`
4. Add a round-trip test case in `tests/plugins/proto/test_round_trip.py`
5. Run `tools/build/gen_planner_proto.sh` to regenerate stubs
6. Run `pytest dynamo/planner/tests/plugins/proto/` — both
   `test_class_coverage_*` tests catch missing mirror / converter; all
   round-trip cases must still pass
7. Commit `plugin.proto` + generated `*.py` + Pydantic mirror + test case
   in the same PR

## FPM `bytes` field encoding

`FpmData.prefill_engines` / `decode_engines` are `map<string, bytes>`. The
proto **does not constrain bytes content** — interpretation is dispatched
by `RegisterRequest.fpm_encoding`:

| `fpm_encoding` | Bytes interpretation | Use case |
|---|---|---|
| `"msgspec"` (default) | Native msgspec encoding of `ForwardPassMetrics` (see `forward_pass_metrics.py`) | In-process plugins, Python sidecar plugins; zero overhead |
| `"proto"` | A separate `FpmEnginePayload` message with explicit fields (NOT defined in this proto; lives in a future companion proto file) | Cross-language plugins (Go / Rust / ...) |
| `"json"` | JSON encoding of the same logical fields | Development, dashboards |

This dispatch decouples the wire format from the schema — adding a new
encoding is a registry-side concern, not a proto change.

## References

- `dynamo/planner/plugins/types.py` — Pydantic v2 mirror
- `dynamo/planner/plugins/_proto_bridge.py` — bidirectional converter
- `tests/plugins/proto/test_round_trip.py` — equivalence + lock-step tests
