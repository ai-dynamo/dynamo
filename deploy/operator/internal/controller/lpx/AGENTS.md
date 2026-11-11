<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scope

- Review against supported failure scenarios, including ownership errors, missing
  wakeups, unsafe deletion, and invalid API writes.
- Ask before changing documented trade-offs. Update these instructions when an
  intentional contract changes.
- Support DGD input edits and external scaling.
- Recovery from direct edits, force-deletion, or orphaning of managed
  resources are out of scope unless explicitly requested.
- Do not add migration or recovery machinery solely for unsupported cases.
- Minimize or exclude self-evident comments.

# Reconciliation

- Keep reconciliation linear: observe, derive desired state, apply changes, and
  return `ctrl.Result` and an error. Keep ordering and early returns visible.
- Pass native API objects and focused values directly. Avoid custom request or
  result bags, classification structs, and callback pipelines. Calculations and
  renderers may return values or native resources.
- Validate ownership and input freshness at the observation boundary, then trust
  those objects downstream. Document meaningful nil cases; do not repeat checks
  that callers have already established.
- Admission validates static user intent. Reconciliation validates the input
  handoff, observed resource structure, and compiler-dependent constraints.
  Placement, topology discovery, and GPU-shape calculation belong to the
  compiler/rendering and scheduler layers.
- Name functions after the resource or condition they handle. Prefer
  `pipelineRequest`, `schedulingFailed`, and `dgd` to ambiguous terms such as
  "attempt", "source", or "LPX deadline".

# Ownership and Identity

- The DynamoGraphDeployment (DGD) owns the LPXGraphDeployment (LPXGD). The LPXGD
  owns its PodCliqueSet (PCS), runtime ConfigMaps, and Services. The PCS owns the
  LPUPipelineRequests (LPRs) and Grove hierarchy. Each independent workload has one
  PodCliqueScalingGroup (PCSG). A shared draft/target runtime remains one workload.
- LPRs have a controlling PCS owner reference with `blockOwnerDeletion: true`.
  Foreground PCS deletion keeps the owner present until blocking LPRs are gone.
  This is the supported lifecycle, not a guarantee against external orphaning.
- Admission prevents removing LPX components or changing them to non-LPX.
  Deleting the DGD removes its LPXGD through owner garbage collection.
  A missing/deleting DGD or incomplete input handoff stops publication; it does
  not authorize cleanup. A DGD UID mismatch is an error. Disabled integration
  also leaves existing workloads intact.
- Index LPRs by controlling PCS UID, validating the owner's API version and kind
  in the index. Do not add a PCS-name index or namespace-wide request scan.
- PCS identity derives from the LPXGD UID and stays stable across input edits.
  LPR names identify the LPXGD namespace, name, UID, workload, model, and PCSG
  ordinal.
- Keep identity and change detection separate: `InputRevision` synchronizes the
  parent-to-child handoff; workload digests detect immutable workload changes;
  `CompilerSnapshotDigestAnnotation` records compiler provenance and is part of
  immutable LPR intent. Neither input revision nor generation defines a batch.
- Accept restart completion only when the child's input revision matches the
  current DGD. A matching restart token and an old Ready condition are insufficient.

# Observation and Retries

- Read workload dependencies through the cached client and validate them once
  per reconciliation; do not add uncached reads or pre-write revalidation.
- Cyborg GPU resolution is a validation-only exception to dependency watches.
  Do not add ResourceClaim, ResourceClaimTemplate, or DeviceClass watches for it:
  the resolved count does not affect the rendered workload. Lookup failures use
  ordinary error retries, and DGD edits use the existing watch.
- Per-fabric-pod Cyborg placement is the only reader of Pods and Nodes through
  watches. Its Pod watch selects only Cyborg Pods that carry the placement
  label, and its Node watch is metadata-only. Do not widen either watch or
  add a complete Node cache.
- Observations are not an atomic snapshot. Publishing previously observed
  intent during a concurrent edit is accepted; watches drive convergence.
  UID/resource-version preconditions and write conflicts protect mutations.
- Use watches for cache visibility, publication, finalization, and input or
  scheduler changes. A watch-driven wait returns a zero result, not a polling
  interval. Keep progress gates explicit rather than treating a timer as a
  "stop this pass" signal.
- Reserve `RequeueAfter` for scheduling deadlines, bounded error retries while
  a deadline is active, and external download checks. The other deliberate use
  is the follow-up after recording `SchedulingFailed`, because status-only
  LPXGD updates are filtered. Ordinary errors use controller-runtime backoff.
- Synchronize the desired PCS before any replica write, including scale-down and
  deadline cleanup. After a PCS write, wait for its watched cache observation. Use
  the shared scaling predicate to defer Coherent replica writes until Grove's
  observed generation matches the current PCS generation and no rollout is active.
  Completed progress from an older generation does not acknowledge a new spec.
  Apply this generation wait only to Coherent: observed generation can lag
  throughout RollingRecreate without blocking its scaling. Initial Coherent
  configuration acknowledgement does not require scheduler requests or Ready pods.
  Continue observing readiness and component status while scaling is deferred.
- After writing a PCS, wait for its watched observation before publishing LPRs.
  An LPR `AlreadyExists` response means wait for observation, not adopt an
  unverified object. Never adopt a foreign resource with the expected name.

# Publication

- Retain matching LPRs and their status across input edits. Collect missing
  requests in the same ordered pass that constructs desired requests. An
  immutable mismatch waits for its Grove engine to roll and LPX to release the
  old requests; incompatible layouts require PCS replacement.
- Publish missing requests independently in workload/PCSG ordinal/model order.
  Partial publication and idempotent retries are supported; exactly-once
  batches are not required. Creation order does not guarantee scheduler order or prevent
  interior scheduling failures.
- Wait for terminating requests to disappear before reusing their names.
  Do not introduce batch finalizers or cleanup protocols.

# Capacity and Deletion

- Keep the top-level PCS replica count at one; workload capacity belongs to the
  PCSG. Correct top-level replica drift through normal synchronization, not
  PCS replacement.
- Seed the PCS's scaling-group template from immutable `MinAvailable` (default
  one), never from live or desired capacity. Apply explicit desired capacity
  through the PCSG `/scale` subresource before publishing requests. The seed stays
  positive and unchanged even when live capacity is zero.
- Explicit DGD replicas give this controller capacity ownership. Omitted
  replicas leave capacity externally managed: derive request ordinals from the
  observed PCSG and never change its capacity, including during deadline cleanup.
- External capacity is not persisted across PCS or PCSG replacement. A new
  group starts from the immutable template seed; its external scaler must
  reapply the desired count. Explicit DGD replicas are reapplied by this controller.
- Index PCSGs by controlling PCS, validating the owner's API version and kind
  in the index. Do not add a namespace-wide request scan.
- Per-fabric-pod placement owns the capacity of each Cyborg PodClique that has
  the placement label, also when the conductor role omits replicas. The other
  Cyborg capacity rules, including external scaling, apply only to PodCliques
  without that label.
- Initial PCS creation needs no PCSG. Once the PCS exists, `Reconcile` waits
  until every configured group has an owned, non-deleting PCSG before proceeding.
  The PCSG watch resumes reconciliation. Downstream capacity, request, deadline,
  and readiness operations can rely on all configured groups being present.
  This also defers PCS updates and replacement until all groups are observed.
- Render a complete valid replacement before deleting an existing workload.
  A replacement error preserves the existing workload, even if its digest differs.
- For incompatible layouts, foreground-delete the PCS and recreate it after
  garbage collection. Grove rolls compatible layouts, including build changes.
- Admission rejects OnDelete for LPX DGDs: Grove would leave running Pods on
  stale templates, which the LPX scheduler treats as dependency loss. Do not
  add OnDelete-specific rollout handling.
- Grove makes PCS clique composition and scaling-group `CliqueNames` immutable.
  Model removal therefore requires PCS replacement. Do not trim group membership
  while retaining templates; those templates can become standalone cliques.
- For replica-only scale-in, lower PCSG capacity before deleting removed LPRs.
  Preserve the PCS and clique templates.
- For replica-only scale-out, wait for old request names to disappear, update
  capacity, then publish missing LPRs.
- Ordinary retirement waits for `status.committed` to clear after Grove cleanup.
  For build changes, all retiring requests in an engine must release before any
  are deleted. Keep UID/resource-version delete preconditions. Deadline cancellation
  remains separate.
- Ordinary retirement leaves Pod and PodClique cleanup to Grove and owner GC.
  The Dynamo operator's LPX controller may directly delete Pods only for an observed,
  owned, non-deleting NodeLocal LPR in `Releasing` for `requestChanged`,
  `dependencyChanged`, or `bindingFailed`: remove its journaled current and target
  Agent Pod UIDs, including plan-replacement predecessors, after verifying their
  PodClique -> PCSG -> PCS owner chain, including terminating PodCliques. Keep namespace
  checks and UID/resource-version delete preconditions. Retry missing observations
  without blocking cleanup of the remaining journaled Pods. Preserve Grove capacity
  and LPUPipelineRequest status, finalizers, and annotations.
- The only other direct Pod deletion is the removal of surplus Cyborg Pods by
  per-fabric-pod placement; see Cyborg Fabric-Pod Placement.
- Attempt release cleanup after DGD identity and input-handoff checks and owned
  Grove-resource observations, including every configured PCSG. It bypasses model
  downloads and desired workload resolution, not these earlier gates. Use observed
  LPRs even when they no longer match desired intent. Accepted tradeoff: before the
  desired PCS template is synchronized, Grove may recreate deleted Agents from the
  old template, requiring another release cycle.
- Cleanup errors do not stop downloads or workload reconciliation. Combine them
  with the final error before status persistence,
  retain the previous ObservedGeneration, and retry. Cap any later scheduled wakeup
  at the existing bounded error-retry interval; otherwise use ordinary error backoff.
- A terminating LPR is pending cleanup, not a reason to delete its PCS or Pods.
  Wait for scheduler finalization through watches, including when external
  capacity is unchanged.

# Cyborg Fabric-Pod Placement

- The annotation `lpx.nvidia.com/cyborg-placement: per-fabric-pod` on a hybrid
  conductor role's Pod template selects it. It applies only to a workload with a
  Cyborg and at least one LPU Agent. An unknown value is an error. Without the
  annotation nothing changes, so an operator upgrade does not roll a workload.
- Where the placement applies, `renderPodCliqueSet` sets the label
  `lpx.nvidia.com/cyborg-placement` on the Cyborg template, and so on its
  PodCliques and Pods. Only this controller sets the label; it removes an
  authored one, and an authored `lpx.nvidia.com/fabric-pod`. Pod admission, the
  Pod watch and the placement select by the label, so the label must imply that
  this controller pins the Pod. Do not set it where the placement does not apply.
- Derive each placement from observed objects only: the labelled PodClique, its
  node selector and required node affinity, the replica count of its template in
  the observed PodCliqueSet, and the observed LPRs of its PCSG replica. A
  PodClique keeps its placement, and its capacity, until Grove replaces it
  without the label. Do not derive a placement from the DGD: admission gates by
  the observed label, so the two must not disagree.
- Pin from these placements directly after the DGD identity check and the Grove
  observations: before the input handoff check, release cleanup, downloads and
  desired workload resolution. The pin is the only workload write that does not
  wait for the handoff. Remove Pods and write capacity only after PCS
  synchronization, as for every replica write. Join placement errors into the
  final error, as for release cleanup: a PodClique that selects no Node must
  not stop the PCS write that corrects it, or the placement of other PodCliques.
- The count for each fabric pod follows the template of the observed
  PodCliqueSet at once, also for a PodClique that Grove has not replaced yet.
  With omitted conductor replicas, a build with a different minimum count thus
  changes the capacity of an old PodClique before Grove replaces it.
- Reject the annotation with the Coherent update strategy when rendering. Grove
  refuses replica changes during a Coherent update, each replaced replica
  starts with the template count, and the capacity difference would also defer
  request publication, so the update could not end.
- The role's replica count applies to each fabric pod, the value of Node label
  `network.topology.nvidia.com/lpu-fabric-pod`. It must form complete client
  groups, as a count without placement must.
- Provision, then prune, for each PCSG replica independently. Until every LPR of
  the replica is `Bound` or `Degraded` on a committed NodeLocal plan, the targets
  are all fabric pods that the Cyborg template can select. Afterwards they are
  the `fabricId` values of the committed rows. A replica without a settled plan
  provisions again; a `samePlacement` repair keeps its fabric pods and does not.
- A fabric pod is selectable when one of its Nodes matches the node selector
  and required node affinity of the Cyborg PodClique. Taints and free capacity
  are not evaluated: a Pod pinned to a fabric pod without capacity stays Pending
  and gives the LPX scheduler no evidence. No selectable Node is an error.
- Do not change the Cyborg template when a placement changes. Grove replaces
  template scheduling gates, rejects `topologySpreadConstraints`, and rolls every
  Pod on a template change, and the LPX scheduler fences the Cyborg template
  hash. The placement label is static: it follows only the DGD and its build.
- Pod admission adds the scheduling gate `lpx.nvidia.com/fabric-pod-pending` to
  each Pod with the placement label
  (`internal/webhook/mutation/pod_cyborg_placement_handler.go`). Kubernetes
  accepts a new gate only at Pod creation. Grove sets and removes only its own
  gate, so this gate holds the Pod until the pin. The chart registers the
  webhook only with `lpx.enabled` in cluster-wide mode, with an object selector
  on the label and failure policy `Fail`. The handler gates only a Cyborg Pod
  that names its LPX deployment and has a Grove PodClique as controller.
- With `Fail`, Grove cannot create a labelled Pod while the operator webhook is
  unavailable. The LPX scheduler tolerates a pending replacement Pod, but a
  refused create leaves no Pod, so it can release a bound plan whose last Cyborg
  Pod in a fabric pod is lost in that time.
- Pin each Pod with one strategic merge patch that carries the observed
  resource version. It sets the `nodeSelector`, records the fabric pod in Pod
  label `lpx.nvidia.com/fabric-pod`, and removes the placement gate by name.
  The operator Pod cache holds the scheduling gates but no node selector, so
  the patch names its changes. Never remove the gate in a separate write: the
  Pod would be schedulable without its pin. A conflict or a missing Pod is not
  an error; a later Pod version wakes the controller. Every other refusal is an
  error, because no later Pod version removes it. One cause is the Grove
  authorizer, which refuses a Pod update from an account that is not exempt.
- Pin a Pod that has the placement gate only after Grove removed its own gate.
  Grove removes its gate with a merge patch that lists the gates that stay.
  Kubernetes refuses that patch when a listed gate is gone, and Grove then
  records an error on the PodClique. The placement gate therefore leaves last.
- The gate leaves a Pod only with its pin. A gated Pod waits while a pass
  stops before the pin: the LPXGD or its PCS is in deletion, the DGD identity
  check fails, or a configured scaling group is not observed. It also waits while
  its PodClique has no placement, for example with no selectable Node. The LPX
  scheduler tolerates a pending replacement Pod of a bound plan only for the
  PCS `terminationDelay`.
- A Pod without the placement gate can exist: the webhook is not registered,
  its failure policy is `Ignore` during an outage, or the Pod is older than the
  webhook. Pin such a Pod while Grove's gate holds it. When Grove releases it
  first, it has no gate, Kubernetes accepts no selector, and it binds without
  one. Adopt it when its Node's fabric pod has an open position, and delete it
  otherwise.
- The gate needs Grove commit `160c0216` (API tag `v0.1.0-alpha.12-rc1`) or
  later; the chart requires `v0.2.0-beta1`. An earlier Grove clears every gate
  with its own, and each Pod then binds without a pin.
- An unbound Pod with Grove index `i` takes fabric pod number `i / replicas` in
  name order when that fabric pod has an open position, and otherwise the first
  fabric pod that has one. A provision into an empty PodClique therefore gives
  fabric pod `k` the index block `[k * replicas, (k + 1) * replicas)`, in
  whatever order the Pods appear. A Pod keeps its index for its life, so the
  Pods that stay after a prune can have indexes at or above the new total.
- Renumber when the capacity is correct: remove a placed Pod whose index is at
  or above the placement total. Grove gives the replacement the lowest free
  index of the PodClique, the placement gate holds it, and the pin puts it in
  the position that the removal opened. Cyborg pipeline-parallel mode needs
  exactly the indexes `0..TOTAL_REPLICAS-1` in one PodClique. With one fabric
  pod in use, this step gives it those indexes. With two or more, that mode
  needs a Cyborg change.
- Renumber only while every LPR of the PCSG replica is `Bound`. Only for a
  `Bound` plan does the LPX scheduler wait, for the PCS `terminationDelay`, for
  the replacement of the last bound Cyborg Pod of a fabric pod. For a
  `Degraded` or `Planned` plan it releases the plan at once.
- Remove one Pod at a time, and only when the PodClique has exactly the
  placement total of live Pods and each of them is placed and bound. Grove then
  owes exactly one replacement, and a fabric pod with two or more positions
  always keeps a bound Cyborg Pod. With one position, the fabric pod has no
  bound Cyborg Pod until the replacement binds.
- Never remove a Ready Pod to renumber it: it serves with the index that it
  has. The Pods that stay after a prune are not Ready, because a Cyborg starts
  after its Agents, and a pipeline-parallel Cyborg with an index out of range
  does not become Ready. A Ready Pod can keep an index at or above the total,
  for example after a scale-in with two or more fabric pods. A later
  replacement in its fabric pod then gets a lower index, and two Pods of that
  fabric pod can have the same `index % ioFpgaCount`.
- Take the first fabric pod in name order that has such a Pod, lowest index
  first. One fabric pod is then complete before the next one starts and gets
  consecutive free indexes: one index block again when the provision gave
  blocks. The blocks of two or more fabric pods are then not in name order.
- Keep bound Pods before unbound, ready before not ready, and then the lower
  Grove index. A fabric pod that holds LPUs therefore keeps a bound Cyborg Pod,
  and with one fabric pod in use, the Pods that stay after a scale-in need no
  replacement.
- Delete surplus placed Pods before lowering the PodClique capacity. Grove
  excludes terminating Pods from its scale-in selection; with the lower capacity
  first, Grove selects its own victims and can remove the last Cyborg Pod of a
  selected fabric pod.
- While a lower capacity is not yet written, never delete an unplaced Pod. Grove
  replaces each removed Pod until the capacity is lower, and its scale-in then
  removes unbound and newer Pods first. Deleting the replacements starts a
  create and delete loop.
- Write the placement capacity without a resource-version precondition. The
  placement is its only owner, and a conflict only delays the write that stops
  the replacements.
- Capacity writes, and the removals that a lower capacity needs, follow the
  shared scaling predicate. Pinning does not.
- Placement does not delay request publication: the LPX scheduler waits for
  bound Cyborg Pods by itself.
- Placement needs a scheduler that places each pinned Pod independently, such as
  the default scheduler. KAI schedules the PodClique as one gang and does not
  continue past a member that cannot run.
- Namespace-restricted mode cannot read Nodes. Reject the annotation there when
  rendering, before a PodCliqueSet can create a Pod with the placement label.

# Scheduling Cycles

- `attemptDeadlineSeconds` bounds how long an individual LPR remains pending,
  including waiting for the scheduler to start. It is not a solver budget or a
  deadline for all deployment Pending states, runtime readiness, or asynchronous cleanup.
- Use a nonzero `LPR.status.schedulingStartedAt` when present; otherwise use
  `LPR.metadata.creationTimestamp`. The scheduler supplies a new start when a
  surviving request begins another cycle, including Bound-to-Pending transitions;
  that timestamp takes precedence over the request's original creation time.
- Current-generation `Bound`, `NoFit`, and `Unsupported` states are exempt.
  `Degraded`, `Releasing`, and `Released` are exempt only with a current-generation
  status and a committed execution accepted for that generation.
- Exemption is not readiness: `NoFit` remains pending, and `Bound` still requires
  Grove runtime readiness.
- The deadline is best effort: PCSG observation, download checks, compiler-registry
  resolution and desired-state validation precede expiry evaluation. Their failures can delay
  failure reporting and deadline cleanup indefinitely, including after expiry.

# Scheduling Failure and Cleanup

- Persist `SchedulingFailed` and observe it in a later reconciliation before
  cleanup. Its timestamp must cover the expired scheduling cycles; a newer cycle
  needs a renewed failure record.
- Re-evaluate expiry from current request status on every pass. Do not delete
  a request solely because it expired in an earlier pass. The condition records
  failure, not a deletion queue; do not persist an expired-request UID list.
- Evaluate deadline cleanup independently for each workload. One expired model
  request makes its entire workload replica eligible, including sibling model
  requests. Cleanup may remove only a trailing suffix.
  Any interior expired ordinal blocks that workload's deadline scale-down and
  request deletion, even if another expired suffix exists in the same workload.
  For four replicas, expiry at `{2,3}` permits scale to two; `{0}` or `{0,3}`
  permits no cleanup.
- Continue cleanup beyond an already-lowered group count on subsequent passes.
  Delete non-expired siblings before expired requests so a partial failure
  leaves expiry evidence for the next reconciliation.
- A scheduling failure blocks publication, not normal request cleanup after
  explicit DGD scale-in or external scale-in, even when an interior failure
  blocks deadline cleanup. Lower explicitly managed PCSG capacity before
  deleting removed requests; never write externally managed capacity.
- Preserve healthy higher ordinals even when interior failures leave the
  deployment failed or pending indefinitely.

# Input Edits and Retries

- Failure remains in effect for the recorded child generation, preventing
  automatic scale-up or recreation of expired requests.
- A new input revision bypasses previous-revision failure. Resolve the new
  desired state first; if it errors, report that error instead of applying
  deadline cleanup to the old revision.
- Retained requests keep their scheduling timestamps. An already-expired request
  covered by a durable failure uses that record for cleanup even after an input
  edit; preserve its original failure generation so the edit still permits retry.
  A cycle started after the failure needs a new record and can fail the new
  revision. An edit does not reset a retained cycle's deadline. Requests removed
  by the edit follow normal scale-in or replacement cleanup.

# Status and Runtime Artifacts

- `Ready` is the only general outcome condition, with reason `Ready`, `Pending`,
  or `Failed`. Keep `SchedulingFailed` as independent failure and retry evidence,
  not duplicate timestamp/generation fields. An error may replace the Ready
  diagnostic without clearing scheduling failure.
- `Ready.LastTransitionTime` changes only when its boolean status changes.
  Renewing `SchedulingFailed` deliberately gives it a new timestamp even when
  it remains true.
- Derive each component's readiness from its own workload's scheduler receipts and
  Grove roles. The graph is Ready only when every workload is ready; aggregate
  readiness must not overwrite independently observed component readiness.
- Persist child status once at the reconciliation boundary, including early
  errors. Record errors before converting them into bounded deadline retries.
  Advance `ObservedGeneration` after error-free reconciliation, including pending
  or disabled outcomes; this does not imply readiness or freshness of every field.
- Convert workload errors into deadline retries only after deciding generation
  acknowledgement and persisting status. Status-write failures remain errors.
  `reconcileWorkload` preserves its error and puts any bounded deadline retry in
  `ctrl.Result`; only the outer `Reconcile` converts that pair to a successful retry.
- Check downloads before acquiring compiler metadata or deriving deadlines.
  Reuse fresh successful observations by build URL across revisions, without
  renewing `LastCheckedAt`. Check new or expired builds. Only a Ready deployment
  whose generation has been observed may ignore a failed periodic refresh.
- Runtime ConfigMaps are immutable, content-addressed, LPXGD-owned, and labeled
  with its UID. Find obsolete maps by label, verify ownership before deletion,
  and retain desired names. Delete obsolete maps only after readiness, not during
  PCS replacement, so existing pods keep their configuration.
- A workload whose builds contain `lpuUnits` gets no runtime ConfigMap; its
  runtimes read placement from the build. The builds of one workload all contain
  `lpuUnits` or none do. Independent workloads in one graph may differ.

# Tests

- Pair test files with production files and test behavior in the package that
  owns it. Exercise controller ordering through `Reconcile`; do not retest shared
  synchronization, conversion, or fake-client behavior here.
- Use data-only tables for variations of one function. Keep complex setup and
  multi-reconcile scenarios as separate tests, with their steps visible in the
  test body rather than table callbacks.
- Share constructors, fixture loaders, and the test-client harness, not mutable
  fixtures or a test-only implementation of controller decisions. Use minimal
  native inputs; add compiler fixtures and Grove hierarchies only when relevant.
- Load reusable workloads from `testdata/` and derive child ownership and input
  revisions from them. Construct compiler artifacts programmatically when testing
  malformed or variant binary metadata.
- Use golden files for substantial deterministic output, not scalar assertions.
  Review regenerated diffs; never update expectations merely to silence failures.
- Fake clients do not enforce Grove admission, garbage collection, or RBAC.
  Check API mutability and permissions when changing writes.
- For code changes, run `go test ./...` and `make lint` from `deploy/operator`,
  using the repository's envtest assets. Run LPX race tests for lifecycle changes.
- Regenerate the condition golden with
  `go test ./internal/controller/lpx -run TestPipelineRequestReadyConditionGolden -args -update`.
- Do not write tautological tests.
