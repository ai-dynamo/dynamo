<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Engine Group admission — discussion draft

This is the shared-worker registration part of [DEP #13121](https://github.com/ai-dynamo/dynamo/issues/13121),
based on [#16752](https://github.com/ai-dynamo/dynamo/pull/16752). It does not enable an engine deployment.

## Boundary

The existing `Worker` remains the only serving/discovery owner. An optional `WorkerAdmission`
adds controller permission, independent of engine health, KV consistency and operator pause:

```text
Backend observer: exact engine lifetime + committed topology
Controller:       permission after commit + serving verification
Worker:           engine/KV ready + operator not paused + permission matches
                                      |
                           register existing endpoint
                                      |
                           acknowledge exact target
```

Neither a healthy engine nor a successful resume grants permission. Withdrawing permission is
not an engine failure: it must not restart the engine or its KV publishers.

## Contract

- `WorkerAdmission::new(group_id, native_member_id)` starts closed and creates a fresh worker
  lifetime ID. This ID is separate from the Pod UID, discovery ID and native engine ID.
  `group_id` identifies the logical Engine Group lifetime, not a Grove object incarnation.
- The trusted backend observer calls `bind` with the actual engine lifetime and committed
  topology generation. Controller requests cannot establish that binding. Engine-native epochs
  and controller topology generations need an explicit, authoritative mapping; they are not
  assumed to be interchangeable.
- `Worker::with_admission` opts in. Unmanaged workers keep their existing behavior. The first
  shape has one independently addressable native member per worker and one serving endpoint;
  auxiliary RL discovery is rejected rather than left outside the gate.
- `/engine/admission/target` accepts an absolute revision, worker lifetime, group/member identity,
  exact binding and `admitted` boolean. A grant represents the controller's commit/verification
  decision; the worker additionally requires the matching backend observation and engine/KV
  readiness. Targets use the existing trusted administrative plane, not public inference traffic.
- An exact replay is idempotent and wakes level reconciliation. A stale revision, changed payload
  at the same revision, wrong worker lifetime, or mismatched grant is rejected without discarding
  the last accepted target.
- A grant awaiting its matching backend observation is retryable, not a permanent target rejection.
  Identity, revision and payload violations are invalid requests. Neither outcome mutates engine membership.
- `/engine/admission/status` separates `accepted` intent from the exact `observed` target whose
  registration or withdrawal completed. Clearing a registration acknowledgement is not a positive
  withdrawal acknowledgement. Both routes
  remain responsive before discovery publication and during recovery or slow discovery work.
- A new target, identity change or recovery closes local request admission immediately. In-flight
  requests are not forcibly cancelled by the admission gate. New requests carrying stale routing
  decisions are rejected before engine dispatch.
- The worker rechecks the exact target after asynchronous registration. If permission or binding
  changed, it withdraws the endpoint without acknowledging the obsolete grant.
- Worker restart starts closed and rejects requests for the previous worker lifetime. The controller
  retains durable intent and must revalidate/reapply it; the worker adds no transaction journal.

## Lifecycle integration

The request listener can start without discovery registration. This lets the existing worker
observe engine health and recovery while waiting for permission, without a second lifecycle loop
or an initial register-then-unregister window.

Startup, recovery completion and resume all feed the same registration decision. Recovery clears
the backend binding; a fresh authoritative observation is required before permission can match.
Pause/resume intent stays separate from controller permission, and shutdown closes admission
before waiting for ongoing administrative work.

Backend health integration must distinguish expected reconfiguration pauses from actual engine
failure. Withdrawing admission alone must not invoke the worker's engine-recovery path.

Registration/withdrawal evidence is not drain completion. This draft does not implement a complete
traffic adapter, prove engine collectives make progress, or authorize capacity release. Those
require the engine and controller contracts in the DEP.

## Not wired yet

There is no deployment flag, Go traffic client, or backend-specific binding publisher in this
draft. The current Python `dynamo.vllm` wrapper has its own lifecycle and would need explicit
integration. The sidecar uses the Rust worker, but still needs native identity/topology correlation
and the remaining serving/drain evidence; HTTP and gRPC engine IDs are not assumed equal.

The next review with the worker/sidecar maintainers should settle the backend binding integration
and administrative route placement before either deployment path is enabled. The existing Go
membership adapter ([#16861](https://github.com/ai-dynamo/dynamo/pull/16861)) and launch/Grove work
([#16929](https://github.com/ai-dynamo/dynamo/pull/16929)) remain separate.
