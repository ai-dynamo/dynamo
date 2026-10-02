# Elastic EP Engine Group implementation plan

Status: working draft, updated 2026-10-02

This plan tracks Dynamo work for [DEP #13121](https://github.com/ai-dynamo/dynamo/issues/13121).
It separates the dependency-independent orchestration foundation from production engine and
workload-manager integrations. Mocks can prove ordering and restart safety, but they do not make a
production backend conformant.

## Objectives

- Reconcile physical capacity, engine membership, traffic, and serving verification as distinct
  observed systems.
- Make every operation absolute, idempotent, restart-observable, and safe after an ambiguous call.
- Preserve stable correlation among logical replicas, workload slots, Pod UIDs, runtime
  incarnations, and engine-native members.
- Fail closed when membership, traffic withdrawal, serving progress, or release safety is unknown.
- Continuously observe engine state even when Kubernetes objects and the desired target are unchanged.
- Validate the capacity model against real DGD configurations before freezing the Engine Group CRD.
- Keep production Grove, vLLM, and SGLang details behind transport-neutral contracts.

## Current implementation status

| Slice | Pull request | Status |
|---|---|---|
| Reconciliation foundation | [#14816](https://github.com/ai-dynamo/dynamo/pull/14816) | Implemented and reviewed; remains a draft while DEP #13121 is proposed. |
| Profile geometry resolver | [#14819](https://github.com/ai-dynamo/dynamo/pull/14819) | Implemented as a stacked draft. The merged SGLang growth profile resolves; vLLM remains explicitly unsupported, without a speculative parser; under review. |
| Engine Group API and `/scale` | [#14896](https://github.com/ai-dynamo/dynamo/pull/14896) | Implemented as a stacked draft with v1beta1 CRD, logical-replica Scale surface, identity status, generated artifacts, and real API-server coverage. |
| Kubernetes controller | [#14939](https://github.com/ai-dynamo/dynamo/pull/14939) | Implemented as a stacked draft with durable restart journal, per-member status projection, and periodic observation covering runtime-only failure, drift, and incarnation changes. |
| SGLang growth integration | [#15545](https://github.com/ai-dynamo/dynamo/pull/15545) | EP1 → EP2 adapters, snapshot-fenced ConfigMap journals, Grove PodClique capacity, template-invariant bootstrap, representative labels, serving verification, and a standalone Scale fixture are implemented. Live GPU validation remains pending; engine-side correlation and admission limitations are isolated in the SGLang integration. |
| DGD lifecycle | Not opened | Creation/retirement of child groups and representative-label integration with production workload managers remain separate slices. |
| Mocker-backed process integration | Not opened | Follows the controller skeleton. |

The API now uses `engineGroup.initialSize` and `policy.minSize/maxSize` on the DGD
creation template, preserving them through v1alpha1 round-trips. DGD-driven creation remains
explicitly rejected until its lifecycle slice exists. Engine Group status separates desired and
active native members from allocated replicas, and projects per-member membership/traffic plus
current/candidate allocation fields. Packed partial-survival status is tested, but executing packed
recovery, candidate promotion, and member-level retirement remains backend-gated work; the SGLang
growth PoC still supports one native member per allocation. Desired native-member assignment is
owned by reconciliation and correlated with the spec generation, not inferred by status projection.
Runtime-resolution and observation failures invalidate current health claims while retaining
historical evidence. The coordinator is organized by capacity, membership, traffic, and verification;
the legacy SGLang growth adapter and traffic projection remain explicitly non-production bridges.

## Track 1: Reconciliation foundation

Implemented in #14816.

### Reconciliation model

The coordinator is level-triggered. Each subsystem has a typed desired/observed contract because
the evidence needed for capacity, membership, traffic, and verification is different. There is no
generic `Adapter[T]` abstraction.

```text
persist desired level
        -> apply the same absolute target
        -> observe authoritative state
        -> repeat until converged, definitively rejected, or blocked unknown
```

The coordinator owns cross-subsystem ordering. Each adapter owns the durable mechanics needed to
converge its subsystem. In particular, the membership adapter—not the coordinator—owns the
serialized engine transaction beneath its level-based interface.

### Capacity contract

The capacity adapter:

- observes stable replica and slot identities, exact Pod UIDs, runtime incarnations,
  availability, and durable release fences;
- applies a revisioned absolute allocation set;
- creates new capacity only when the target carries explicit bootstrap intent;
- treats an exact existing incarnation as an identity assertion, not permission to recreate it;
- deletes only Pod UIDs named by an operation- and topology-bound release fence; and
- reasserts an accepted target when observed capacity drifts.

Capacity status separates:

```text
Desired   newest persisted target, including one that may be rejected
Accepted  last exact target revision durably acknowledged by the adapter
Observed  authoritative allocation and release-fence state
```

`Desired` is promoted to `Accepted` only after the matching applied revision is observed. If a
newer target is rejected, the coordinator retains and continuously enforces the previous accepted
target. This prevents a blocked transition from losing its last safe capacity level.

An applied release means every authorized Pod UID is absent and every corresponding stable slot is
durably fenced against autonomous recreation. Name matching is insufficient. Fence observation
must also prevent a creation that raced the fence from publishing delayed usable capacity.

### Membership contract

The membership adapter exposes:

```go
ValidatePlan(ctx, groupID, request) (PreflightResult, error)
ValidateTarget(ctx, groupID, target) (PreflightResult, error)
Observe(ctx, groupID, transitionID) (MembershipObservation, error)
Apply(ctx, groupID, target) error
```

The desired target contains:

- a controller revision and transition ID;
- a shared-code canonical target digest;
- durable validation evidence;
- one exact base topology and generation;
- a serializable, profile-resolved change; and
- exact joining runtime identities where applicable.

`Observe` is scoped to the requested transition ID. A nil transition authoritatively means that
the adapter has no record of that exact ID. An inconclusive read returns an error or `Unknown`; it
must not masquerade as absence.

The adapter exposes these correlated transition states:

```text
nil     -> Pending | Rejected | Unknown
Pending -> Committed | Rejected | Unknown
Unknown -> Pending | Committed | Rejected
```

`Committed` and `Rejected` are immutable terminal results and remain observable until the
controller acknowledges them or a newer control revision is durably accepted. `Pending` and
`Unknown` cannot be superseded.

`Apply` is an atomic compare-and-apply against the complete base topology. Replaying the same ID
and payload is idempotent. A call error is ambiguous: reconciliation observes first and may replay
only the identical target. A definitive rejection is authoritative only when its transition ID,
revision, and digest match and the general committed topology still equals the target's base.

The engine's current `CommittedTopology` and a transition's immutable `ResultTopology` are
separate evidence. Commit is accepted only when they match. An unrelated later topology fails
closed and is handled as an external transition, not as success for the requested operation.

### Resolved operation shapes

The coordinator uses a serializable tagged union rather than executable plan implementations:

- `Grow`: add named fresh logical replicas;
- `Retire`: gracefully drain and remove selected healthy replicas;
- `ReduceToSurvivors`: remove identities already known to be failed or inactive;
- `Restore`: recreate excluded stable identities and native-member slots; and
- `Remap`: change the complete native-member mapping without changing logical cardinality.

Traffic and verification requirements are resolved into the immutable plan. Adapters validate the
complete shape; broad booleans such as `Grow` or `Shrink` are not sufficient capability claims.

Every cardinal operation preserves retained logical-to-native mappings. Only an explicit `Remap`
plan may change them.

### Two-stage preflight

`ValidatePlan` runs before capacity or traffic prework. It receives the coordinator-computed
canonical plan digest and returns evidence bound to:

- that plan digest;
- the immutable profile fingerprint; and
- the adapter capability generation.

After joining incarnations and the exact membership target are known, `ValidateTarget` runs again.
It adds the canonical target digest and must return the same profile and capability generations.
Both preflight results are durable and correlated to the transition.

For a successful method call, exactly one of validation evidence or a definitive rejection is
present. For a method error, neither result is authoritative. A rejection guarantees that the
validated subject cannot later commit.

### Traffic contract

The traffic adapter applies and observes absolute admitted and draining identity sets. It supports:

- graceful drain for planned retirement; and
- confirmation that an already failed identity is inactive for survivor reduction.

The plan kind determines the required evidence. A healthy `Retire` cannot use failure-only
withdrawal evidence. Engine membership changes and discovery events never admit traffic
implicitly; admission is a separate explicit target after commit and any required serving proof.

Traffic status also separates `Desired`, `Accepted`, and `Observed`. A rejected newer admission
target must not erase the previously accepted fail-closed target. While a transition is blocked,
completed, or rolled back, reconciliation continues repairing drift toward the accepted target.

Two traffic-safety profiles are supported:

- `KeepServing`: retained replicas may continue serving while nominated retirees are withdrawn and
  drained;
- `QuiesceGroup`: the complete base topology is withdrawn and drained before membership mutation,
  then explicitly readmitted after commit and verification.

Drain evidence is tied to exact member incarnations and remains durable until a later explicit
admission.

### Serving verification

Membership commit proves topology, not inference progress. The serving verifier is therefore a
separate, safely repeatable probe against one exact committed topology.

It returns either a topology-bound positive proof, a conclusive failure, or an inconclusive method
error. A stale proof cannot authorize admission after the topology changes. A failed check leaves
membership committed but traffic fenced and the transition blocked. Recovery requires a distinct
plan or intervention.

The initial verifier is deliberately not an asynchronous operation journal. Re-running the same
probe is safe, and the positive proof is persisted by the coordinator.

### Durable state

The coordinator persists:

- a group-global monotonic control revision;
- the logical replica registry and stable slot bindings;
- immutable topology history;
- desired, accepted, and observed capacity state;
- desired and observed membership transition state;
- desired, accepted, and observed traffic state; and
- one transition record containing its immutable spec, both preflight results, verification
  result, outcome, timestamps, and structured failure.

The transition outcome summarizes the cross-subsystem workflow:

| Outcome | Meaning |
|---|---|
| `Progressing` | Safe work remains. |
| `Reverting` | A definitively rejected, provably uncommitted target is restoring the base level. |
| `RolledBack` | Base capacity and traffic are restored after definitive rejection. |
| `Blocked` | Membership, verification, or safety evidence cannot progress automatically. Accepted fail-closed targets remain enforced. |
| `Completed` | Capacity, membership, verification, and traffic reached the resolved plan. Accepted terminal targets remain enforced against drift. |

These are coordinator outcomes, not a duplicate of backend transaction phases.

### Workflow ordering

For growth or restoration:

```text
observe exact base topology
    -> validate and persist the resolved plan
    -> allocate and observe complete capacity
    -> freeze exact runtime identities
    -> validate and persist the exact membership target
    -> apply/observe membership until authoritative commit
    -> pin committed capacity incarnations
    -> verify serving when required
    -> explicitly admit committed membership
```

For planned retirement:

```text
observe exact base topology
    -> validate and persist selected victims
    -> withdraw and gracefully drain exact victim incarnations
    -> apply/observe the smaller membership topology
    -> verify/readmit retained topology when required
    -> authorize exact Pod UIDs at the committed topology generation
    -> apply/observe physical release and durable fences
```

For failure recovery:

```text
observe authoritative survivor topology and traffic exclusion
    -> validate/adopt survivor reduction without rewriting desired fleet size
    -> release failed incarnations exactly
    -> allocate replacements for the excluded stable identities
    -> restore their native-member slots through a separate transition
    -> verify and explicitly readmit the restored topology
```

At every step the coordinator re-observes all subsystems. Unknown membership, stale topology,
missing drain evidence, failed serving verification, or stale Pod UID authorization prevents
admission and release.

### Foundation test contract

#14816 uses deterministic in-memory adapters to cover:

- restart before and after every external effect;
- ambiguous membership apply followed by absent, pending, committed, rejected, or unknown state;
- stale base topology and superseded spec revisions;
- canonical plan and target digest correlation;
- definitive preflight and apply rejection;
- growth, retirement, survivor reduction, restoration, and native-member remapping;
- required group quiescence and graceful selected-member drain;
- topology-bound serving verification before admission;
- Pod-name reuse with another UID and stale release authorization;
- exact release fences and raced replacement capacity;
- drift repair for completed and blocked traffic/capacity targets;
- preservation of the previous accepted target after a newer target is rejected; and
- adoption of supported external survivor transitions without pretending Dynamo initiated them.

## Track 2: Profile geometry and DEP validation

Implemented in stacked draft #14819 and rebased onto the reviewed #14816 foundation. The current
review pass rejects vLLM layouts whose process ownership cannot satisfy the narrow profile and
adds SGLang's merged width-one growth profile.

### Resolver boundary

The resolver consumes only declarative configuration available before workload creation and
returns either a complete immutable geometry result or a typed unsupported reason. It must never
guess from opaque commands, external configuration, or incomplete engine arguments.

The current slice resolves:

- one rank-owning workload role;
- the physical GPU requirement of one logical replica;
- `podsPerReplica` for a dedicated, exactly divisible layout;
- a canonical geometry fingerprint including workload revision; and
- explicit unsupported reasons for packed, distributed, ambiguous, or environment-dependent
  configurations;
- explicit rejection of current vLLM Elastic EP ownership modes that cannot prove one
  independently releasable replica per Pod; and
- the merged SGLang width-one, growth-only Elastic EP profile.

The first accepted shape is deliberately narrow: one logical replica per dedicated GPU Pod,
`podsPerReplica: 1`. This preserves independent allocation and release at the Kubernetes resource
boundary.

This slice does not claim to resolve bootstrap behavior, native-member placement, safe live bounds,
topology requirements, traffic policy, serving verification, or a complete backend capability
profile. Those require explicit backend/runtime contracts.

### Compatibility findings

| In-tree shape | Current result | Reason |
|---|---|---|
| vLLM single-Pod Elastic EP demo | Unsupported | Multiple DP replicas share one Pod and cannot release GPUs independently. |
| vLLM multi-node Elastic EP demo | Unsupported | Ray actors are packed into preallocated Pods; `nodeCount` is not a logical replica-allocation unit. |
| Static vLLM multi-node TP/PP | Unsupported by the narrow resolver | One logical replica spans multiple Pods and needs a complete multi-Pod allocation contract. |
| SGLang merged scale-up profile | Supported by the narrow resolver when one DP/EP rank has one dedicated one-GPU Pod | Growth uses the merged absolute EP target and externally launched joiners; shrink and recovery remain separate capabilities. |
| SGLang single-Pod warm-standby demo | Unsupported | Multiple independently scalable ranks and reserved GPUs share one Pod, so Kubernetes cannot allocate or release them independently. |
| TensorRT-LLM | Not implemented | The backend needs its own declarative source resolver and runtime capability contract. |

These results are DEP evidence, not incidental implementation limitations. If a common target
profile cannot be represented without inference, revise the DEP applicability boundary or API
before freezing the CRD.

### Stable identities

The implementation and eventual API must keep these identities distinct:

```text
logical replica ID       stable across replacement
capacity-slot identity   stable workload-manager position backing that replica
Pod UID                  one concrete Kubernetes Pod incarnation
runtime incarnation      one concrete engine process incarnation
native member identity   engine rank or rank set correlated with the replica
```

Pod names are not durable identities. Exact release authorization is bound to Pod UID, transition,
and committed topology generation.

## Track 3: Engine Group API

Implemented initially in stacked draft #14896. Keep the feature gated while DEP #13121 remains
proposed and treat the v1beta1 shape as provisional until one realistic controller and adapter
exercise it.

Implement in two reviewable steps. The first API-only step contains:

- `DynamoGraphDeploymentEngineGroup` spec and status;
- `/status` and `/scale` subresources;
- `spec.replicas` as the absolute logical-replica target;
- `status.replicas` as allocated complete replica allocations;
- a selector that matches exactly one representative Pod per allocated replica;
- allocated, available, engine-active, traffic-admitted, draining, and drained identity sets;
- the immutable resolved profile and fingerprint;
- the stable logical/slot/Pod/runtime/native identity correlation;
- exact topology-bound release authorization;
- target-validation status, conditions, and printer columns; and
- generated CRDs, deep-copy code, scheme registration, and API tests.

The API-only PR description must state that Kubernetes conversion is unaffected: this is a new
v1beta1-only kind with no v1alpha1 counterpart.

The controller step then adds only the restart journal it actually consumes:

- desired, accepted, and observed capacity and traffic projections;
- desired/observed correlated membership transition state;
- preflight evidence, serving proof, transition outcome, and structured failure;
- RBAC and controller integration tests.

Add the DGD component's initial Engine Group replica count and optional policy bounds. Admission
enforces only statically known constraints. Resolver-derived bounds are enforced by reconciliation
and reported in status.

Do not expose the old coordinator-owned `Pending -> Submitting -> Accepted -> Committing` phase
machine. The public status should represent the desired/observed contracts and coarse transition
outcome established by #14816.

## Track 4: Controller and DGD lifecycle

Wire the foundation to Kubernetes resources while retaining fake adapters first.

The DGD controller should:

- create one stably named Engine Group per independent engine world;
- initialize it from the component's declared initial Engine Group replica count;
- establish ownership, finalizers, and labels without treating Pod names as identity; and
- initially require `components[].replicas == 1` while outer-world fan-out is unresolved.

The Engine Group controller should:

- persist status before invoking adapters;
- recover all desired/accepted/observed state after restart;
- periodically observe resolved runtimes after convergence and while blocked, without depending on Pod changes;
- apply one representative label per complete allocation for `podsPerReplica: 1`;
- remove the representative label before release;
- derive `status.replicas` from complete allocation state, not label count; and
- reject unsupported production adapters and profiles explicitly.

Outer-world retirement remains deferred. A DGD rollout or scale-down must not delete a group that
still has active membership or unreleased capacity.

### Polling first; notifications are optional acceleration

The controller schedules another observation even when a group is healthy or its transition is
completed, rolled back, or blocked. A software failure inside a packed Pod may leave its Kubernetes
status unchanged. Pod and journal watches can accelerate reconciliation, but cannot replace
periodic engine observation.

Use a ten-second observation interval after progress stops, retaining the existing one-second
requeue while the coordinator requests progress. Errors use controller-runtime's retry backoff.
These intervals determine when observation is attempted, not a guarantee of detection or recovery
latency. Production adapters need bounded observation deadlines and must be checked against the
backend's fault-detection and recovery windows before recovery is enabled.

Every reconcile reads authoritative adapter state. A timer or notification neither identifies a
failed member nor authorizes removal. Inconclusive observations publish unknown health while
preserving the durable journal; a blocked group continues maintaining its accepted capacity and
traffic targets. An unexpected runtime incarnation is not silently adopted or admitted.

Regression tests cover runtime availability changes without Pod updates, each observation
authority becoming unreachable, routing drift after completion or blocking, and a changed process
incarnation in the same physical slot.

Keep the initial transport simple: the controller invokes the existing typed adapters. Optional
notifications may later enqueue the group for an earlier observation; they remain hints, not
health or membership authority. No Pod-status publisher, notification CR, or pull-based operation
executor is required for this slice.

Polling supplies detection opportunities, not a recovery implementation. The SGLang proof remains
growth-only. A future recovery adapter must interpret backend fault evidence, persist and observe
correlated outcomes, and verify the exact committed survivor topology before traffic admission.
In particular, vLLM #46370's `request_id` coordinates a round but does not suppress duplicate
submissions; the adapter still owns the restart-safe transaction contract. Proactive prepare/pause
remains a backend dependency where idle engines cannot enter recovery safely; synthetic inference
requests are not the general production mechanism.

## Track 5: Mocker-backed integration

After the CRD and controller skeleton exist, extend Mocker with a process-level implementation of
the membership contract:

- accept absolute targets identified by transition ID, revision, and digest;
- expose authoritative committed topology independently from the correlated transition result;
- durably retain pending and terminal transition observations across reconnects;
- enforce one compare-and-apply transaction at a time;
- simulate delayed, rejected, ambiguous, and committed outcomes;
- expose two-stage validation evidence and structured capability generation;
- preserve retained native-member mappings unless an explicit remap is requested; and
- support topology-bound serving verification and deterministic fault injection.

Pair it with fake capacity and traffic endpoints that retain accepted absolute targets, repair
drift, and expose exact drain/release evidence. This proves a complete controller restart through
real process boundaries without claiming vLLM, SGLang, or Grove conformance.

## Initial workload rendering

Build a dedicated Engine Group rendering path whose initial physical capacity is:

```text
initial logical replicas x podsPerReplica
```

Control-only roles remain outside that capacity. The live target is owned by the Engine Group and
does not mutate `multinode.nodeCount`.

The initial renderer may target fakes or render an initial Grove shape. It must reject production
profiles that still depend on immutable peer lists or launch-time world-size flags for later live
membership changes.

## Separate `minAvailable` workstream

Current multi-node Grove rendering uses one value for both atomic initial admission and the runtime
gang-termination threshold. Lowering it blindly could admit an incomplete initial world; leaving it
at the initial width can terminate a world that the engine could safely run in degraded form.

Work that is independent now:

- retain regression tests for current rendering and failure behavior;
- resolve `minSafeServingReplicas` only when the backend can prove it;
- isolate initial-admission policy from runtime survivor policy; and
- avoid changing generated thresholds until the workload realization can express both semantics.

This should remain a separate small change, not be folded into the Engine Group API slice.

## External dependency boundary

| Dependency | Production behavior it blocks |
|---|---|
| [Grove GREP #823](https://github.com/ai-dynamo/grove/pull/823) | Stable capacity slots and exact UID-authorized release. |
| [Grove #793](https://github.com/ai-dynamo/grove/issues/793) | Dynamic complete-allocation placement, topology-local growth, survivor-preserving failure policy, and separate initial/runtime gang thresholds. |
| [vLLM PR #43202](https://github.com/vllm-project/vllm/pull/43202) | Planned Elastic EP growth and shrink plus a conforming restart-observable orchestration surface. |
| [vLLM PR #46370](https://github.com/vllm-project/vllm/pull/46370) | Proposed survivor reduction. Proactive pause, restoration, and integration with Elastic EP remain required follow-up capabilities, not guarantees of this PR. |
| [SGLang RFC #35376](https://github.com/sgl-project/sglang/issues/35376) and [PR #33111](https://github.com/sgl-project/sglang/pull/33111) | Planned shrink, selected retirement, and recovery alignment. Growth is available through merged [PR #30164](https://github.com/sgl-project/sglang/pull/30164). |
| R16-compatible backend bootstrap | Replacing static peer lists and world-size launch contracts with identity-aware joining and recovery. |
| Runtime traffic and verification integration | Durable admission, withdrawal, drain, and topology-bound serving proof. |

Real adapters advertise exact supported shapes. A mock-supported operation is never enabled for a
production backend by implication.

## Pull-request sequence

1. **#14816 — reconciliation foundation:** implemented and reviewed.
2. **#14819 — profile geometry resolver:** implemented as a stacked draft; the merged SGLang
   width-one growth profile resolves and current vLLM Elastic EP ownership is rejected explicitly;
   complete review and merge.
3. **Engine Group API:** CRD, status, `/scale`, generated artifacts, and envtest coverage.
4. **Controller skeleton:** fake-backed reconciliation, status persistence, and finalization.
5. **DGD lifecycle and representative labels:** one world initially; no outer retirement.
6. **Mocker integration:** restart-safe process-boundary membership and serving verification.
7. **Initial workload rendering:** narrow supported geometry and explicit rejection of static
   bootstrap profiles.

The dependency-independent foundation is complete when a mock-backed Kubernetes Engine Group can
prove restart-safe growth, planned retirement, survivor reduction, replacement restoration, exact
release, serving verification, and traffic admission without claiming production engine or
workload-manager support.

## Deferred production phases

Proceed as each dependency becomes conformant:

1. Grove stable identity, complete-allocation observation, placement, and exact release.
2. First production engine membership adapter and topology-bound serving verifier.
3. Topology-local growth.
4. Planned shrink with graceful drain and exact release.
5. Survivor reduction, replacement bootstrap, and restoration.
6. Multi-Pod replica allocations.
7. HPA and KEDA integration after enforcing a single target writer and supported metric path.
8. Outer-world fleet reshaping and terminal Engine Group retirement.

## Validation strategy

- Pure table tests for geometry resolution, plan normalization, canonical digesting, and every
  supported operation shape.
- Race-enabled unit tests for arbitrary reconcile interruptions, ambiguous adapter calls, accepted
  target retention, and drift repair.
- Property or fuzz tests for idempotency, identity correlation, and stale revisions.
- Envtest coverage for CRD validation, `/scale`, status conflicts, owner references, finalizers,
  and representative labels.
- Process-level Mocker tests for controller and adapter restart, timeout, retained terminal
  observations, authoritative commit, verification, and exact release.
- Shared conformance scenarios for every future production capacity, membership, traffic, and
  serving-verification adapter.

## Risks and checkpoints

- **Profile mismatch:** revise the DEP if common configurations cannot resolve without guessing.
- **Stack drift:** keep #14819 and later stacked branches rebased on the reviewed foundation before
  evaluating their diffs or test results.
- **Duplicate effects:** persist desired state before `Apply`; replay only the identical absolute
  target after ambiguity.
- **Lost safe level:** retain accepted capacity and traffic payloads independently from a newer
  rejected desired target.
- **Membership ambiguity:** never supersede pending or unknown transitions; terminal evidence is
  correlated and retained durably.
- **Identity drift:** keep logical replica, slot, Pod UID, runtime incarnation, and native member
  distinct through the full workflow.
- **Unsafe release:** require exact UID- and topology-bound authorization plus durable fencing.
- **Stale proof:** bind serving verification to the exact committed topology and revalidate before
  admission.
- **Static bootstrap leakage:** reject profiles whose survivors require relaunch for membership
  changes.
- **Premature workload policy:** do not claim Grove resizing or alter `minAvailable` until initial
  admission and runtime survivor semantics are both representable.

## Rough remaining effort

After #14816 and #14819, the dependency-independent Kubernetes foundation is approximately two to
four engineer-weeks:

- Engine Group CRD, Scale surface, and generated/API tests: 0.5–1 week;
- controller and DGD lifecycle with fake adapters: 0.75–1.5 weeks;
- Mocker process integration and restart scenarios: 0.75–1.5 weeks; and
- initial narrow workload rendering and compatibility validation: 0.25–0.5 week.

Production Grove and engine adapters, topology-aware placement, traffic integration, recovery, and
fleet reshaping are excluded from this estimate.
