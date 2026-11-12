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
  authored one, and an authored `lpx.nvidia.com/fabric-pod` or
  `lpx.nvidia.com/served`. Pod admission, the
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
  desired workload resolution. The pin and the served record are the only
  workload writes that do not wait for the handoff. Remove Pods and write
  capacity only after PCS synchronization, as for every replica write. Join
  placement errors into the
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
- Renumber a settled placement: give the Pods that stay the Grove indexes below
  the placement total. Cyborg pipeline-parallel mode needs exactly the indexes
  `0..TOTAL_REPLICAS-1` in one PodClique. With one fabric pod in use, this step
  gives it those indexes. With two or more, that mode needs a Cyborg change.
- Add first, then remove. Give the fabric pod of a Pod above the range one more
  position, so that the capacity is the total plus one. Grove creates one Pod
  with the lowest free index of the PodClique, the placement gate holds it, and
  the pin puts it in the added position. When the fabric pod then has its count
  of other Pods that the kubelet started, and every request of the workload
  replica is `Bound`, the Pod above the range gives up its position, and the
  prune steps remove it.
- Never remove a Pod above the range first. The fabric pod would have one bound
  Cyborg Pod less, and its GPU can go to a different Pending Pod. When it was
  the last bound Cyborg Pod of the fabric pod, the LPX scheduler can release
  the plan: it waits only while the removed Pod or its replacement exists, and
  since Kubernetes 1.33 Grove can create the replacement only after the removed
  Pod is gone (the deletion mark changes the Pod generation, and Grove ignores
  an update with a changed generation).
- End a step only when the kubelet started the new Pod, not when the Pod is
  bound. The kubelet can refuse a bound Pod, which then has phase Failed, and
  the LPX scheduler does not accept such a Pod as the Cyborg Pod of its fabric
  pod. The kubelet started a Pod when one of its containers or init containers
  has no waiting state; the operator Pod cache keeps the waiting and the
  terminated state. A Cyborg Pod that waits for its Agents runs its init
  containers, so the kubelet started it, and those containers keep that state
  while its other containers wait for a restart.
- The new Pod needs one free GPU in the fabric pod. Without one it stays
  Pending, the capacity stays at the total plus one, and no Pod goes. When the
  Pod above the range becomes Ready in that time, the position goes away and
  the new Pod is removed. A pipeline-parallel Cyborg with an index out of
  range never becomes Ready, so that workload does not start until a GPU is
  free. There is no step that removes the Pod first in that state.
- The step is for the first fabric pod in name order whose bound Pod with the
  lowest index above the range is not Ready and has no served record, and for
  that Pod. One fabric pod is then complete before the next one starts and
  gets consecutive free indexes: one index block again when the provision gave
  blocks. The blocks of two or more fabric pods are then not in name order.
- The Pod that gives up its position is the Pod that the step is for, not the
  Pod that the retention order selects. Cyborg takes its FPGA route from
  `index % ioFpgaCount`. After a provision into index blocks, the lowest free
  index has the route of the lowest index above the range in the fabric pod,
  so the routes of the fabric pod are the same after each step. The retention
  order removes the highest index, and two Pods then have the same route
  until the last step. During a step, the new Pod and the Pod that goes have
  the same route.
- Never skip a Pod above the range that keeps its index. Its fabric pod then
  takes no more part: the next new Pod would get the route of that Pod. When
  that Pod served while a step added a Pod for it, the fabric pod has one
  bound Pod more than its count, and two Pods with one route. The added Pod is
  the Pod with the highest index below the range. It gives up its position
  when it is not Ready and has no served record. When it served too, the
  retention order selects the Pod that goes, as in a scale-in: the Pod with
  the lower index stays, which is the added Pod. Do not find the added Pod by
  its age: a Pod that Grove created for a failed Pod is also newer, and the
  rule would then remove a Ready Pod in a scale-in.
- Add a position only to a complete placement that Grove has synchronized: the
  capacity is the total, `status.observedGeneration` equals the PodClique
  generation, `status.replicas` equals the total, and the PodClique has exactly
  the total of live Pods, each placed and bound. Between the surplus deletion
  and the lower capacity of a prune, Grove creates Pods for the removed ones,
  and it removes them when it observes the capacity. More capacity at that time
  lets one of them stay with an index that is not the lowest one. A position
  that the capacity already has stays while a Pod above the range needs one,
  so the capacity is the total plus one from the first step to the last. When
  a removed Pod is gone, Grove creates the Pod of the next step.
- Decide the step in the placement step, from the Pods that this step lists,
  and use that decision for its pins, its removals and its capacity. With an
  older decision, the pass can pin the new Pod for a Pod that served in
  between. The pin step at the start of a pass uses the placement without the
  step. It thus does not pin the new Pod in a pass that stops before the
  placement step, which also does the removal that ends the step.
- The scaling state needs no Pod list (`capacityPending`): a capacity other
  than the total is a change that waits, but one Pod more than the total of a
  settled placement is the added position. So a step that waits to start, and
  a scale-in by one Pod of a settled placement, are not in the scaling state.
- The added position does not wait for `Bound` requests, and the removal of
  the Pod above the range does. The new Pod can go away between its
  observation and that removal. The fabric pod then has one Cyborg Pod less.
  For a `Bound` request the LPX scheduler waits for the next Pod, and for a
  `Degraded` request it releases the plan at once when that was the last one.
- A PodClique observation that is older than the write of the added position
  shows neither the start nor the stay condition. Such a pass removes the new
  Pod while the kubelet has not started it, and Grove creates the next one.
  This repeats only while the PodClique cache is behind the Pod cache, and no
  Pod that the kubelet started goes.
- Lower the capacity only when Grove has observed each removal: in a pass
  that removes no Pod and that finds as many live Pods of the PodClique as
  its capacity. Grove reads Pods and PodCliques through different watches, and
  the deletion mark starts no Grove reconcile. When Grove observes the lower
  capacity before the mark, it removes a Pod that it selects by itself: a Pod
  without a Node, then a Pod that is not Running, then a Pod that is not
  Ready. That can be the new Pod of a step, and the fabric pod then has no
  Cyborg Pod. Grove creates a Pod for a removed one only when its Pod cache
  shows the removal, so the full count is the proof. Do not wait for the end
  of the removed Pods in its place: a Pod without a Node is gone at once. The
  Pods that Grove created have no position and no Node, and Grove's scale-in
  removes them first. A higher capacity does not wait.
- The Pod watch passes these changes of a placed Pod: its fabric pod label,
  its Node, its deletion mark, whether the kubelet started it, and whether its
  phase is terminal. Each one can change a step or a position. It also passes
  each version of a Ready Pod that has no served record; the pass that writes
  the record ends these events. Do not depend on the PodClique status for
  that record: its ready count can stay the same when a Pod becomes Ready.
- Remove a surplus Pod with its UID as the only precondition. A pass can read
  a Pod before the cache has its deletion mark, and a second removal must
  change nothing. A removal can thus use an old Ready condition: a Pod that
  became Ready in the cache delay can go. In a renumber, its fabric pod then
  already has the Pod that takes its place.
- A Pod in phase Failed or Succeeded holds no position: remove it. It cannot
  run again, the LPX scheduler does not accept it as the Cyborg Pod of its
  fabric pod, and Grove creates no Pod in its place while it is there. Never
  count it as a bound Pod: a new Pod that fails after its bind must not end
  the added position.
- Never replace a Pod that serves, or that served: it uses the index that it
  has. A Pod serves when it is Ready. Record each Ready Pod in Pod label
  `lpx.nvidia.com/served`, in the pin step, and keep a Pod with that label
  also while it is not Ready. Only this controller sets the label; remove an
  authored one when rendering. A Pod that stays after a prune is not
  Ready while it waits for its Agents, and a pipeline-parallel Cyborg with an
  index out of range does not become Ready, so those get new Pods. A Pod above
  the total that becomes Ready before then keeps its index. So does a Pod that
  stays after a scale-in: with two or more fabric pods in use, or when a Pod
  with a lower index had a worse rank. A later replacement in its fabric pod
  then gets a lower index, and two Pods of that fabric pod can have the same
  `index % ioFpgaCount`.
- Keep bound Pods before unbound; then a Ready Pod, a Pod with the served
  record, a Pod that the kubelet started, and the other Pods; then the lower
  Grove index. A fabric pod that holds LPUs therefore keeps a bound Cyborg
  Pod, and a Pod that runs before a Pod that the kubelet can still refuse.
  With one fabric pod in use, the Pods that stay after a scale-in have the
  lowest indexes when no Pod with a lower index has a worse rank.
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
  shared scaling predicate. Pinning does not. The added position of a renumber
  is a capacity write.
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
