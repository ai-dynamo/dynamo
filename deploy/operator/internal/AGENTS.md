<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DGD workload orchestration

These rules apply to `DynamoGraphDeployment` workload selection, rendering,
reconciliation, rollout, restart, readiness, status, watches, and provider integration.

## Controller and program boundaries

- The outer DGD controller owns fetching, finalization, selection of exactly one
  workload program, persistence of the returned DGD status, post-persistence event
  emission, manager setup, and watch wiring. It must not own provider control flow.
- A workload program is one complete reconciler with one `Reconcile` method. It owns
  operation order, early returns, `ctrl.Result`, rollout, scaling interaction,
  readiness, provider diagnostics, and compatibility cleanup.
- Keep sequencing visible in the program's control flow. Do not replace it with a
  generic sequence runner, lifecycle callback pipeline, or shared mutable chain state.
- Keep each workload pathway as a separate complete program. Programs share only the
  small request/result contract and concrete operations with genuinely identical
  invariants. Do not introduce a common provider lifecycle or make one complete program
  invoke another.
- The DCD CR and controller remain the child retry, ownership, status, and observability
  boundary. A workload mechanism selected within the DCD controller does not become a
  graph-level program.
- A composite pathway may reuse narrow rendering or reconciliation units for child
  workloads. It must not construct another controller or invoke another complete
  program as a subroutine.
- Express composition through concrete nested reconcilers with explicit dependencies.
  Programs must not retain or call an all-capable `DynamoGraphDeploymentReconciler`.
- Keep shared stable-resource reconciliation inside each complete program and preserve
  explicit dependency ordering. Do not pre-run it in the outer controller to populate
  generic program inputs, and do not add a generic common-tail callback.

## Request, result, and data flow

- The program request contains only the explicitly mutable DGD. Clients, recorders,
  configuration, and other dependencies live on the concrete program or collaborator
  that uses them.
- The program result contains `ctrl.Result`, the complete desired DGD status, and queued
  status-transition events.
- Do not add speculative `Facts`, `Inputs`, `ComputedState`, `WorkloadModel`, render
  request, provider request, or universal `ReconcileState` bags.
- Earlier operations return focused typed values; later operations consume them through
  local variables and direct arguments.
- Persist intermediate state only for a deliberate retry, ownership, or observability
  boundary, not merely to connect in-process steps.
- Keep sequencing dependencies explicit: resolve an input before its consumer, and
  apply readiness or startup gates before downstream mutations they are meant to
  suppress.

## Status, primary writes, and events

- The outer DGD controller is the only DGD status-subresource writer. Programs and
  nested reconcilers never call `Status().Update` for a DGD and do not use
  `request.DGD.Status` as their output accumulator.
- Initialize every program result from the current DGD status. Return a complete,
  authoritative, meaningful status on success and error, preserving prior fields and
  partial progress unless an operation deliberately replaces them.
- The selected program owns the complete status and final `Ready` condition. The outer
  controller persists the returned status unchanged; it does not merge or override
  program-owned fields.
- Advance status-level `ObservedGeneration` only after error-free program
  reconciliation. A condition's `ObservedGeneration` identifies the generation that
  produced that condition, including a failure condition.
- Perform required non-status DGD mutations directly on `request.DGD` and persist them
  at the state-machine boundary that requires them. Preserve the resulting resource
  version for the final status write. Do not defer these mutations through result
  callbacks, patch collections, or mutation bags.
- Queue transitions represented by returned status in the program result and emit them
  only after status persistence succeeds.
- Emit ordinary resource-mutation events directly only after a semantic create, update,
  patch, or delete succeeds. Do not emit them for no-ops, failed operations, ignored
  `AlreadyExists`, or ignored `NotFound`.

## Reconciler and renderer boundaries

- Extract a reconciler when it owns a cohesive resource family, a distinct dependency
  set, or an independently testable fixture contract. Keep ordinary calculations as
  functions; do not wrap a trivial `Get` merely to manufacture another abstraction.
- Use `Reconcile` for external convergence, `Resolve` for read-only observation or
  derivation, and `Render` for desired-object construction without persistence.
- Give read-only code `client.Reader`, write-only code `client.Writer`, and a complete
  `client.Client` only when a cohesive unit genuinely requires both. Do not introduce
  object-specific one-method reader interfaces.
- A renderer constructs desired resources. It does not persist resources, write status,
  register watches, or own retries. A narrow `client.Reader` is acceptable when
  Kubernetes-backed retrieval is part of the renderer's cohesive responsibility.
- Pass actual domain inputs to renderers. Do not manufacture workload models or render
  request bags to make rendering appear pure.
- Do not construct one controller to reuse another controller's helpers. Extract a
  controller-independent renderer or reconciliation unit with explicit ownership and
  error semantics.
- Keep provider-native API objects at provider boundaries. Provider-neutral contracts
  and base renderers must not expose them.

## Provider behavior and runtime bindings

- Keep rollout algorithms provider-owned. Capabilities may validate or select user
  intent, but common code must not derive provider rollout steps from normalized
  replicas or readiness.
- Workload-path switching, provider failover, and zero-downtime cross-path migration
  are not common invariants. Preserve provider-specific compatibility cleanup without
  creating a universal cutover protocol.
- Provider code owns its native runtime identity and launch conventions.
  Provider-neutral backend and launcher code should consume typed runtime bindings and
  must not require external orchestrators to publish Dynamo-specific values.
- Structural refactors preserve resource names, labels, annotations, selectors, owner
  references, hashes, adoption and deletion behavior, generated commands, and
  feature-gate selection unless the change explicitly targets that behavior.

## Watches and package movement

- Register watches statically at the manager composition root. Provider helpers may own
  kinds, indexes, predicates, and mapping functions, but watch registration is not part
  of the runtime program interface.
- Watch every dependency transition that can unblock reconciliation. Preserve
  significant-change filtering and avoid duplicate watches or reconciliations.
- Move a program into a focused package only after its inputs, outputs, dependencies,
  and tests form a cohesive boundary. Do not create a speculative target package tree.
- Program packages must not import `internal/controller`. Break cycles by extracting
  narrow provider-neutral libraries, not by moving provider behavior into common code.
- Promote the private program contract only after its shape is proven. Exporting it
  must not add clients, provider objects, scratch state, or lifecycle methods.
- Treat stable-resource ownership changes separately, one resource family at a time.
  Do not hide owner-reference migration inside control-flow, renderer, or package
  refactors.

## Tests

- Characterize behavior before moving it across a receiver or package boundary. Cover
  success, pending, failure, idempotence, and meaningful partial status on error.
- Prove the outer controller remains the sole DGD status writer and programs do not
  mutate `request.DGD.Status` as output.
- Prove status-transition events are suppressed when status persistence fails and are
  emitted after successful persistence.
- Preserve coverage for pathway selection, rollout and restart transitions, readiness,
  significant watches, ownership, cleanup, and generated-resource parity.
- Keep provider-specific tests with their owning state machine. Shared conformance tests
  may enforce mechanical invariants but must not impose a universal provider rollout or
  readiness algorithm.
