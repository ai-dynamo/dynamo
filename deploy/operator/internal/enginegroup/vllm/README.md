<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM Engine Group membership adapter

This package implements the Go membership adapter for vLLM's correlated external
Elastic EP API in [vLLM #62082](https://github.com/vllm-project/vllm/pull/62082).
It owns neither process launch, discovery registration nor physical release.

## Membership contract

The adapter calls every existing rank, including retirees, with the same operation
ID and absolute DP target plus each rank's frozen API instance ID. Native vLLM
status is authoritative for transaction phase, effective topology, process identity
and operation epoch.

Dynamo persists target/profile/process bindings and terminal correlation receipts.
It observes before replaying an ambiguous request and resubmits only the identical
target. It does not implement a second native collective state machine.

Supported shapes are contiguous append, high-rank-suffix retirement and re-append
of a fully retired suffix. ReplicaLayout maps each allocation to its complete
native-member range, including packed protocol layouts. Sparse survivor masking,
arbitrary selected retirement, native remapping and complete-world retirement
are unsupported. Membership support does
not imply safe physical shrink.

Direct native HTTP and existing Dynamo worker control routes share the same
request/status codec and transaction authority.

## Ownership

| Owner | Responsibility |
|---|---|
| Engine Group controller | Persist targets, order operations, correlate evidence and authorize exact release. |
| Workload manager | Allocate capacity, expose slots/Pod UIDs and enforce the supported release contract. |
| vLLM bootstrap | Start the native engine from the shared template and resolve its allocation role. |
| vLLM | Commit membership and report native topology, lifetime and serving state. |
| Dynamo serving runtime | Verify serving progress and register eligible endpoints from native evidence and runtime/KV prerequisites. Honor outstanding withdrawal intent. |

The serving runtime does not wait for a second controller permission,
acknowledgment or status write. Preparation may leave existing members serving;
native readiness determines whether inference is actually available.

Engine-managed drain requires an integration-supplied ManagedDrainValidator.
It validates every frozen native process identity before effects and submission.
The native adapter does not query worker-specific endpoints; without a validator,
it rejects EngineManagedDrain.

## Integration boundaries

[#16929](https://github.com/ai-dynamo/dynamo/pull/16929) composes native automatic
bootstrap, Grove capacity, DGD rendering and native runtime resolution.
[#17018](https://github.com/ai-dynamo/dynamo/pull/17018) enables the existing
Python worker's readiness-driven serving and the narrow managed-growth pathway.

Allocated capacity is not inference readiness. A joiner's API can depend on
membership preparation, so preparing the resize cannot wait for that API to
become ready.

Retirement needs durable withdrawal and exact-incarnation terminal drain, not
discovery absence or a wrapper-local request count. Physical release additionally
requires the coordinator's exact Pod-UID/topology authorization.

Sidecar integration follows [DEP #14657](https://github.com/ai-dynamo/dynamo/issues/14657)
and [DEP #14897](https://github.com/ai-dynamo/dynamo/issues/14897). Remote native
controls, identity correlation, serving evidence and durable withdrawal must compose
with the serving runtime's health and KV lifecycle. This adapter's HTTP protocol
does not establish native gRPC or sidecar conformance.

## Validation

Tests cover native protocol validation, exact all-rank fan-out, plan-specific
preflight, ambiguous transport outcomes and restart recovery of durable bindings.

## Grove deployment

The image must also include native automatic bootstrap from
[vLLM #62973](https://github.com/vllm-project/vllm/pull/62973). The template declares
the native command, `--elastic-ep-auto-bootstrap`, the immutable initial DP size,
and `$(DYNAMO_RANK)` / `$(DYNAMO_LEADER_ADDRESS)` references. It omits
`--data-parallel-size`: the joiner learns the complete target through vLLM's
native handshake. The operator neither rewrites that command nor distributes
an executable launch assignment.

The standard Pod renderer supplies infrastructure defaults. The Engine Group
renderer maps the portable topology references to Grove identity and preserves
explicit user probes instead of adding inference probes to waiting joiners.
Each Pod owns one native DP member and its complete fixed TP subgroup.

The capacity adapter retains the clique UID and each slot's exact Pod UID across
Pod disappearance. A replacement occupant requires recovery; it cannot inherit
an old process's evidence. This is a physical identity catalog, not a launch
command or native transaction journal. Container restarts, packed launches,
replacement recovery and complete-world retirement are unsupported.
