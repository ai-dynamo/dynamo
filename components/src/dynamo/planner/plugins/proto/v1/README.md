# Plugin Proto v1

Plugin RPC contract for the Dynamo Planner plugin framework.

This directory contains:

| File | Purpose | Status |
|---|---|---|
| `plugin.proto` | Single-source-of-truth proto3 schema | tracked |
| `plugin_pb2.py` | Generated protobuf Python stubs | tracked (regenerate locally when editing `plugin.proto` — see "Generation" below) |
| `plugin_pb2_grpc.py` | Generated gRPC client/server stubs | tracked (same as `plugin_pb2.py`) |
| `plugin_pb2.pyi` | Generated type stubs for IDE / mypy | tracked |
| `__init__.py` | Module marker | tracked |

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

Generated stubs (`plugin_pb2.py`, `plugin_pb2_grpc.py`, `plugin_pb2.pyi`)
are **checked into git** so that test/build environments don't need
`grpcio-tools` installed just to import the module. The repo-wide
`.gitignore` excludes `*_pb2.py` / `*_pb2.pyi` for other consumers; the
planner stubs are explicitly negated with `!components/src/dynamo/planner/plugins/proto/v1/*`.

Regenerate locally when you edit `plugin.proto`:

```bash
# Regenerate all three stubs (run from components/src/)
cd components/src
python -m grpc_tools.protoc \
    --python_out=. --grpc_python_out=. --pyi_out=. --proto_path=. \
    dynamo/planner/plugins/proto/v1/plugin.proto

# protoc strips the SPDX header — re-prepend it on generated Python stubs.
SPDX=$'# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n'
for f in dynamo/planner/plugins/proto/v1/plugin_pb2.py \
         dynamo/planner/plugins/proto/v1/plugin_pb2_grpc.py \
         dynamo/planner/plugins/proto/v1/plugin_pb2.pyi; do
  printf '%s%s' "$SPDX" "$(cat "$f")" > "$f"
done
```

**Workflow status**: PR #1 does NOT ship a wrapper script or CI
drift-catching step. The proposed `tools/build/gen_planner_proto.sh`
(and a `planner-build --check` job that diffs regenerated stubs against
committed ones) is deferred to a follow-up build infra PR.

Until then, developers who edit `plugin.proto` are responsible for
running the protoc command above, committing the regenerated stubs,
and (separately) updating the Pydantic mirror in `plugins/types.py`.
The two `test_class_coverage_*` round-trip tests catch missing Pydantic
mirrors at CI time; they do NOT catch a stale `plugin_pb2.py` against
an updated `plugin.proto` (the drift-catching CI step is deferred).

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

### `PredictionData` numeric fields MUST be `optional double`

```proto
message PredictionData {
  optional double predicted_num_req        = 1;  // unset -> preserve prev
  optional double predicted_isl            = 2;
  optional double predicted_osl            = 3;
  string source                            = 4;
  optional double predicted_kv_hit_rate    = 5;
  optional double predicted_accept_length  = 6;
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

### `result` oneof empty = silent ACCEPT (graceful degradation)

For `Propose` / `Reconcile` / `Constrain` stage responses,
`WhichOneof("result")` returning `None` is treated as **silent ACCEPT**:
the plugin's response is dropped from the merge, a WARNING is logged, and
`plugin_evaluations_total{result="error"}` is incremented. The circuit
breaker is **not** tripped (only transport errors / timeouts trip it).

This aligns with the DEP main-doc invariant that plugins missing required
input data MUST return ACCEPT to enable graceful degradation rather than
escalating into a failure.

Plugin authors who want to explicitly abstain should set
`accept=AcceptResult()` — but proto3 cannot distinguish "explicit empty
`AcceptResult`" from "no oneof field set" on the wire (both produce zero
field tags), so the orchestrator treats the two identically. The
`{result="error"}` counter is the signal a plugin author should watch
when investigating whether their plugin is correctly setting the oneof.

### `final=true` semantics differ between PREDICT and PROPOSE/RECONCILE

| Stage | `final=true` rule |
|---|---|
| `PROPOSE` / `RECONCILE` | priority number smallest (= highest priority) wins |
| `PREDICT` (chain-augment) | first `final=true` in chain wins (chain ordered priority-ascending → smallest priority number runs first; partial-merge is first-writer-wins, so smallest priority is effectively most authoritative) |

**Convention for PREDICT**: `final=true` is most commonly used as "my
answer is enough; skip all remaining plugins". With ascending priority
sort the authoritative plugin always runs first, so the cleanest way
to express that intent is to set `final=true` on the smallest-priority
plugin.

When `final=true` comes from a non-lowest-priority plugin, the chain
still breaks at that plugin: the smallest-priority plugin has already
weighed in (its values are protected by first-writer-wins regardless),
but larger-priority-number plugins after the final-setter are skipped.
They lose the chance to populate fields earlier plugins left as `None`.
This may be **intentional** (e.g. a policy plugin saying "skip the
expensive fallback for this scenario") or a **configuration mistake**;
`chain_augment` cannot tell which from the response alone. To surface
the event for operator audit, `chain_augment` logs a `WARNING` and
records the message on `ChainAugmentOutcome.chain_break_warnings`
(surfaced via `PipelineOutcome.audit_events`). A Prometheus counter
for this signal is deferred to a follow-up observability PR.

### `final=true` does NOT skip CONSTRAIN

Even when a `PROPOSE` / `RECONCILE` plugin sets `final=true`, the CONSTRAIN
stage runs normally. After CONSTRAIN, the planner adapter also applies the
local planner's final `min_endpoint` / GPU-budget invariants before any
scaling target reaches the connector, so no `final` proposal can bypass those
planner safety checks.

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
5. Regenerate stubs locally with the protoc command in "Generation"
   above (don't forget the SPDX re-prepend step) — the generated
   `plugin_pb2.py` / `plugin_pb2_grpc.py` / `plugin_pb2.pyi` are
   **checked into git**, so this also produces the diff you need to commit
6. Run `pytest dynamo/planner/tests/plugins/proto/` — both
   `test_class_coverage_*` tests catch missing mirror / converter; all
   round-trip cases must still pass
7. Commit `plugin.proto` + regenerated stubs + Pydantic mirror + test
   case in the same PR

## FPM `bytes` field encoding

`FpmData.prefill_engines` / `decode_engines` are `map<string, bytes>`. Each
value is a **msgspec/msgpack-encoded** `ForwardPassMetrics` record (see
`dynamo.common.forward_pass_metrics`). Wire format is standard msgpack, so
cross-language plugins decode with any msgpack library (Go's
vmihailenco/msgpack, Rust's rmp-serde, JS @msgpack/msgpack, etc.) plus
knowledge of the `ForwardPassMetrics` struct layout.

The planner populates this field on ticks that collect FPM observations.
Plugins that declare `needs=["observations.fpm"]` receive the per-engine
maps when available; absence of the field means no FPM data was collected
for that tick.

## References

- `dynamo/planner/plugins/types.py` — Pydantic v2 mirror
- `dynamo/planner/plugins/_proto_bridge.py` — bidirectional converter
- `tests/plugins/proto/test_round_trip.py` — equivalence + lock-step tests
