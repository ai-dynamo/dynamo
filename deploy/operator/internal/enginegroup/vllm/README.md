<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM Engine Group integration

The runtime image must include the native correlated controls in
[vLLM #62082](https://github.com/vllm-project/vllm/pull/62082) and native automatic
bootstrap in [#62973](https://github.com/vllm-project/vllm/pull/62973).
The managed deployment uses the existing `python3 -m dynamo.vllm` worker.

## Startup and growth

Engine Groups reuse the standard operator Pod renderer for environment,
discovery, service accounts, shared memory, security and image-pull configuration.
The Engine Group path adds no command wrapper or engine-argument rewriting.
The template explicitly declares
`--elastic-ep-auto-bootstrap`, the immutable initial DP size, and references to
`$(DYNAMO_RANK)` and `$(DYNAMO_LEADER_ADDRESS)`. It omits `--data-parallel-size`: vLLM learns the complete
growth target through its native handshake, not from the joining rank or a
controller-published command.

```text
DGD declaration → one Grove template → native vLLM automatic bootstrap
                                         │
                               complete initial world

Engine Group target → Grove /scale → joiner waits in native bootstrap
                  └→ correlated resize on every old rank → membership commit
```

There is no Dynamo bootstrap process, command annotation, receipt file or receipt
probe. Pod allocation/readiness and native inference readiness remain independent;
an API-unready joiner must not block submission of the resize that initializes it.

The current launch profile has one local DP member per non-restarting Pod, with
its fixed TP subgroup filling the GPU allocation. Packed launching, multi-node TP,
fault-tolerant masking and arbitrary selected retirement are not enabled. Initial
DP must exceed one, and the native MP/external-DP profile reserves an explicit maximum.

See [`poc.yaml`](poc.yaml) for the worker template. It declares the
Downward API engine identity, native mode environment and retention readiness gate;
standard infrastructure settings do not need repeating. The `DYNAMO_ENGINE_GROUP=1`
entry enables the worker integration; it is not an Engine Group object identity.
Parent `spec.env` entries use the standard component-over-parent merge.
The operator supplies `DYNAMO_RANK` as the allocation index and
`DYNAMO_LEADER_ADDRESS` as this world's allocation-zero hostname. The mapping to
Grove stays inside the renderer; user commands need no `GROVE_*` references.
Both aliases precede user environment entries and cannot be overridden, including
through parent `spec.env`. In this profile one allocation owns one native DP member.

Engine Group main containers do not inherit the generated inference probes;
explicit caller probes are preserved. Choose allocation/process health appropriate to the
engine: a joiner need not be inference-ready before resize submission, and an
expected inference pause should not accidentally trigger Grove gang termination.
Normal Kubernetes defaults and Grove scheduling/identity injection still apply;
the operator retains its ownership, discovery and revision labels.

## Identity and ownership

| Owner | Responsibility |
|---|---|
| Engine Group controller | Persist targets, correlate evidence, order operations and authorize exact release. |
| Grove | Place and create Pods; expose stable slots and Pod UIDs. |
| vLLM | Bootstrap processes, serialize collective changes, commit membership and report native identity/state. |
| Dynamo worker | Verify native serving progress, register eligible endpoints and honor withdrawal/drain intent. |

The Engine Group retains a clique-UID/slot/Pod-UID catalog across Pod disappearance.
A new occupant or missing catalog requires recovery; it cannot inherit an old
process's evidence. This catalog records physical identity, not launch commands
or native transaction phases. Process restart counts and native API instance IDs
provide additional fences.

The worker does not need the Engine Group CR's UID to start. Its status identifies
the exact Pod UID and native member, while the operator proves their ownership
chain and correlates the native API instance. Registration never waits for a
second controller permission or status write.

## Membership and release

The adapter fans the same operation ID and absolute target to every existing
rank, including retirees, with each server's expected instance ID. Native status
is authoritative for pending/unknown/terminal outcomes, effective topology and epoch.
The journal retains exact target and native identity correlation; it does not
run a second collective state machine.

Native protocol support covers contiguous append, healthy suffix retirement and
suffix restoration. The composed planner enables healthy growth and suffix shrink
down to the configured initial size; reusing a released slot remains unsupported.
Group deletion is not a supported terminal-retirement operation.

Healthy shrink requires native request draining, durable incarnation-bound drain
evidence, committed smaller membership, and exact Pod-UID/topology release evidence.
`VLLM_ELASTIC_EP_DRAIN_REQUESTS=1` must be declared on every worker. Serving verification is
topology-bound and established by the runtime independently of controller availability.

Grove scale-in steering is isolated behind its capacity adapter. The retention
readiness gate marks the engine-retired victim NotReady before lowering the
clique count. This remains steering, not atomic deletion authorization; the adapter
checks the exact victim UID and fails closed on mismatch.

## Validation and scope

Go tests cover native request/status codecs, all-rank fan-out, preflight, ambiguous
transport outcomes, restart recovery, exact process bindings, serving/drain evidence,
native-template rendering and real API-server Scale/Pod identity behavior.
Worker tests cover registration from native readiness without controller permission.

GPU validation requires images built with both native prerequisites and this
worker integration. Existing functional-test image pins do not qualify the native
bootstrap path; rebuild and rerun the manual fixtures before claiming GPU validation.

Sidecar integration follows [DEP #14657](https://github.com/ai-dynamo/dynamo/issues/14657)
and [DEP #14897](https://github.com/ai-dynamo/dynamo/issues/14897). This HTTP/worker
integration does not establish native gRPC or sidecar conformance.

## Bounded native submission

The native prerequisites must expose bounded `submit_elastic_ep` acceptance.
The immutable request includes operation ID, expected API instance, original
opaque topology token, absolute target and drain parameters. Native execution
owns the accepted resize independently of the HTTP connection. Dynamo observes
later progress without holding reconciliation open for the collective. Existing
synchronous callers and Ray retain their APIs.

Acceptance proves only this client's participation and complete local range, not
every client's receipt of the shared operation. Retained scoped receipts and
identical reassertion repair partial delivery. Transport failures and request-level
409/422 responses are not authoritative whole-operation rejection. Pending or
uncertain native work cannot be superseded. Results remain retained until a
safe successor, with base fencing preventing delayed old requests becoming new
work. Commit alone does not prove serving readiness or terminal drain.
