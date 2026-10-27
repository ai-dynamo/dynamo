<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM Engine Group integration

The runtime image must include the native correlated controls in
[vLLM #62082](https://github.com/vllm-project/vllm/pull/62082) and native automatic
range bootstrap in [#62973](https://github.com/vllm-project/vllm/pull/62973),
including allocation-index startup, and
[native local-range resizing](https://github.com/julienmancuso/vllm/pull/1).
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
                  └→ one correlated resize per old client → membership commit
```

There is no Dynamo bootstrap process, command annotation, receipt file or receipt
probe. Native preparation waits for the main container to be running, not image
pulling. Pod allocation/readiness and native inference readiness remain independent;
an API-unready joiner must not block submission of the resize that initializes it.

Each non-restarting Pod runs one Dynamo worker with one native client. The client
owns `--data-parallel-size-local` engines, each using a fixed TP subgroup; together
they fill the Pod's GPU allocation. vLLM owns their launch, GPU assignment, local
routing, cancellation and resize barriers. Dynamo does not supervise child processes.
Multi-node TP, fault-tolerant masking and arbitrary selected retirement are not enabled.
Initial DP must exceed one, but it may fit in a single packed allocation.

See [`poc.yaml`](poc.yaml) for the worker template. It declares the
Downward API engine identity, native mode environment and retention readiness gate;
standard infrastructure settings do not need repeating. The `DYNAMO_ENGINE_GROUP=1`
entry enables the worker integration; it is not an Engine Group object identity.
Parent `spec.env` entries use the standard component-over-parent merge.
The operator supplies `DYNAMO_RANK` as the allocation index and
`DYNAMO_LEADER_ADDRESS` as this world's allocation-zero hostname. The mapping to
Grove stays inside the renderer; user commands need no `GROVE_*` references.
Both aliases precede user environment entries and cannot be overridden, including
through parent `spec.env`. Packed templates pass the index explicitly as
`--elastic-ep-allocation-index=$(DYNAMO_RANK)`; native vLLM computes the first global
rank as index times local DP size. Single-member templates may instead use the native
external rank flag. The operator does not rewrite either command.

For the four-GPU smoke test, [`packed-test.yaml`](packed-test.yaml) declares
one DGD with two GPUs per Pod and TP1:

```text
Engine Group replicas:    1          2          1
Pods / Dynamo workers:   1    ->    2    ->    1
Native DP / EP:          2          4          2

Pod 0: worker -> native client -> engines dp-0, dp-1 (retained)
Pod 1: worker -> native client -> engines dp-2, dp-3 (created, then retired)
```

Each worker verifies all its local engines before registering its serving endpoint.
The status reports the complete owned member list and the shared native client ID.
Every member remains individually correlated with its Pod UID/rank and with native
retirement evidence. Child restarts and partial survival require recovery support
and are not enabled by this healthy-resize profile.

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
the exact Pod UID and native members, while the operator proves their ownership
chain and correlates the native API instance. Registration never waits for a
second controller permission or status write.

## Membership and release

The adapter fans the same operation ID and absolute target to every existing
client, including retirees, with each server's expected instance ID. One request
covers all the client's local ranks; targets cannot bisect its range. Native status
is authoritative for pending/unknown/terminal outcomes, effective topology and epoch.
The journal retains exact target and native identity correlation; it does not
run a second collective state machine.

The native prerequisites must expose bounded `submit_elastic_ep` acceptance.
The request includes the original opaque topology token and complete effect
parameters. Native execution owns the accepted resize independently of the HTTP
connection; Dynamo returns from reconciliation without waiting for the collective.
The existing synchronous API and Ray path are not replaced.

Acceptance is per client and its complete local range, not the shared collective
record. The adapter retains those receipts and reasserts an identical request to
repair missing participation. A transport failure or request-level 409/422 is not
proof that the whole operation was rejected. Pending or uncertain native work
cannot be superseded. Terminal evidence remains retained until a safely accepted
successor, and a late old request cannot bypass the original topology fence.

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

Go tests cover native request/status codecs, all-client fan-out, preflight, ambiguous
transport outcomes, restart recovery, exact process bindings, serving/drain evidence,
native-template rendering and real API-server Scale/Pod identity behavior.
Worker tests cover registration from native readiness without controller permission.

GPU validation requires images built with the native prerequisites and this
worker integration. Existing functional-test image pins do not qualify the native
bootstrap path; rebuild and rerun the manual fixtures before claiming GPU validation.

Sidecar integration follows [DEP #14657](https://github.com/ai-dynamo/dynamo/issues/14657)
and [DEP #14897](https://github.com/ai-dynamo/dynamo/issues/14897). This HTTP/worker
integration does not establish native gRPC or sidecar conformance.
