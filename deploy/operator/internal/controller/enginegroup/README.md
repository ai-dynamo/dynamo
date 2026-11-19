<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Engine Group state ownership

The controller coordinates the workflow without interpreting a workload provider's
native resources. A `WorkloadProvider` owns physical bindings, lifecycle fences and
fenced replacement. Its `workload.Capacity` reports physical allocations and applies
allocation targets. Engine integrations bind those allocations to native processes
and compose membership, traffic and serving verification. Setup connects a compatible
workload provider and engine; provider-native watches stay in that composition layer.

A physical lifetime is opaque to engine integration and checkpoint handling. Changing
its identity requires proof that the predecessor stopped, followed by fresh engine
formation and serving verification. Physical lifecycle reconciliation remains active
even when engine resolution fails. Neither Pod readiness nor a same-named replacement
can authorize reuse of old membership, slot bindings or release evidence.

Reconciliation reads desired spec, private history and fresh adapter observations.
It computes and saves the next checkpoint before projecting public status. The
coordinator applies only targets already present in its input checkpoint. A
successful status write is not a prerequisite for native serving or evidence of
external acceptance.

| Record | Authority and writer | Identity / lifetime | Crash boundary and regression |
|---|---|---|---|
| Engine Group spec | Declared scaling owner writes the absolute replica target and policy. DGD seeds creation, not live size. | Engine Group UID; spec generation. | New target queues behind the immutable operation. `TestEngineGroupCheckpointSurvivesPublicStatusWriteFailure`. |
| Private checkpoint | Engine Group controller owns desired/accepted levels, canonical identities, transition and historical assignment/stability. ConfigMap CAS fences concurrent writes. | Engine Group UID, immutable profile fingerprint, physical world lifetime; operation ID/revision/digest. | Save before effect; load before replay. `checkpoint_test.go` covers failed checkpoint/status writes, tampered projections, missing/corrupt authority and forged release. |
| Public status | Controller-owned projection. Counts, conditions and operation presentation are not recovery authority. | Observation generation plus exact topology and process incarnations. | Failed observations publish unknown health. `status_test.go`, `controller_polling_test.go`. |
| Grove world fence | Concrete Grove lifecycle reconciler records exact Pod UID/slot and kubelet-confirmed process stop before removing Pod finalizers. A successor records its exact predecessor before rebinding. | Engine Group UID and physical clique UID. | Old process proof precedes new binding and checkpoint reset. `grove_restart_test.go`, `grove_capacity_envtest_test.go`. |
| Grove capacity journal | Capacity adapter retains accepted revision/digest, in-progress release and exact release receipts. Grove remains authority for count and physical Pods. | Engine Group UID, clique UID; release operation/topology generation and Pod UID. | Acceptance precedes retention-gate/count writes; replay repairs drift. `grovecapacity/release_test.go`, `capacity_grove_release_envtest_test.go`. |
| Native membership correlation journal | Backend adapter records the exact immutable request, native runtime/epoch association and consumed result. The engine owns collective acceptance, execution and outcome. | Group UID/physical world lifetime, native instance IDs, operation ID, control revision, canonical digest and frozen processes. | Save target before RPC; observe after timeout; replay identical payload. `vllm/membership_test.go`, `sglang/membership_test.go`. |
| Traffic journal | Backend traffic adapter records accepted withdrawal intent and terminal all-member drain evidence. It is not a positive registration grant. | Group UID/physical world lifetime; operation/topology, member ID and process incarnation. | Persist proof before release; retain it across Pod disappearance. `vllm/worker_test.go`, `sglang/retirement_test.go`. |
| Engine operation / serving state | Native engine owns membership, operation correlation and readiness. Worker follows readiness and verifies serving progress independently of the controller. | Native process instance plus collective epoch; restarting the engine invalidates old authority. | Controller timeout/restart must not cancel or duplicate an accepted collective. Worker tests cover topology change during verification and registration retry. |

Adapter records contain evidence their external systems do not durably expose.
They are scoped to different lifetimes and effect boundaries; putting them in one
ConfigMap would not make Kubernetes writes and native collectives atomic. Remove
an adapter journal only when its external authority can retain and report the
same evidence across the required restart boundary.

`RecoveryRequired` pauses unsafe effects without rejecting or canceling the native
operation. Persistent faults are reported by fresh observations, and adapters
revalidate safety before every effect. Invocation errors remain retryable without
overwriting native results or blocking their interpretation. Fresh serving evidence is independent
of that diagnosis; failed observations cannot refresh health or stability history.
Known pending state uses bounded polling, while unexpected errors use controller
backoff. Capacity observations, not engine flags, report the release mechanism.

Grove readiness-steered shrink is an explicit experimental delivery choice.
`ExactReleaseSupported=False`, reason `ReadinessSteeredRelease`, makes that limitation
visible without changing serving availability. All-member terminal drain, frozen
UID checks and post-deletion verification remain required. Victim ordering is not
atomic UID-authorized release and does not satisfy DEP R13.
