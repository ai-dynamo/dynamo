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

- Read through the cached client. Observe dependencies and validate them once
  per reconciliation; do not add uncached reads or pre-write revalidation.
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
- After writing a PCS, wait for its watched observation before publishing LPRs.
  An LPR `AlreadyExists` response means wait for observation, not adopt an
  unverified object. Never adopt a foreign resource with the expected name.

# Publication

- Retain matching LPRs and their status across input edits. Collect missing
  requests in the same ordered pass that constructs desired requests. An
  immutable mismatch invalidates that result and requires PCS replacement.
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
- Initial PCS creation needs no PCSG. Once the PCS exists, `Reconcile` waits
  until every configured group has an owned, non-deleting PCSG before proceeding.
  The PCSG watch resumes reconciliation. Downstream capacity, request, deadline,
  and readiness operations can rely on all configured groups being present.
  This also defers PCS updates and replacement until all groups are observed.
- Render a complete valid replacement before deleting an existing workload.
  A replacement error preserves the existing workload, even if its digest differs.
- For immutable workload or request changes, foreground-delete the PCS and
  recreate it after garbage collection.
- Grove makes PCS clique composition and scaling-group `CliqueNames` immutable.
  Model removal therefore requires PCS replacement. Do not trim group membership
  while retaining templates; those templates can become standalone cliques.
- For replica-only scale-in, lower PCSG capacity before deleting removed LPRs.
  Preserve the PCS and clique templates.
- For replica-only scale-out, wait for old request names to disappear, update
  capacity, then publish missing LPRs.
- Write scale-down before deleting LPRs, but do not wait for Grove or pod
  deletion. Asynchronous pod cleanup allows scheduler finalizers to complete.
- Never directly delete Pods or PodCliques from graph reconciliation or LPR
  lifecycle helpers. Grove scaling and owner garbage collection own workload
  cleanup; scheduler Pod references and release journals do not authorize it.
- A terminating LPR is pending cleanup, not a reason to delete its PCS or Pods.
  Wait for scheduler finalization through watches, including when external
  capacity is unchanged.

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
