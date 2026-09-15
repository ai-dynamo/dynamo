/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package enginegroup

import "context"

// AuthorizedReplica binds one logical replica and stable slot to every concrete Pod authorized for removal.
// Empty CapacityRefs request only a durable slot-bound fence for already-absent capacity.
type AuthorizedReplica struct {
	ReplicaID    ReplicaID
	SlotID       CapacitySlotID
	CapacityRefs []CapacityRef
}

// ReleaseAuthorization binds an absolute capacity target and exact removal identities to one membership transition
// or aborted-operation cleanup. An empty replica list is a target barrier that cancels older allocation work without
// authorizing physical removal.
type ReleaseAuthorization struct {
	ID                 string
	OperationID        string
	TopologyGeneration int64
	TargetReplicas     int32
	Replicas           []AuthorizedReplica
}

// CapacityReleasePhase is the workload manager's observable state for one authorized release.
type CapacityReleasePhase string

const (
	// CapacityReleasePhaseAbsent is a durable observation that no release with this identity was accepted and no
	// previously completed Release call for it can later mutate capacity.
	CapacityReleasePhaseAbsent CapacityReleasePhase = "Absent"
	// CapacityReleasePhaseApplying means removal of the exact authorized capacity is still converging.
	CapacityReleasePhaseApplying CapacityReleasePhase = "Applying"
	// CapacityReleasePhaseApplied means the absolute target barrier is durable, every authorized Pod UID has been
	// removed, and each released logical slot is fenced until a later operation explicitly requires that replica ID.
	CapacityReleasePhaseApplied CapacityReleasePhase = "Applied"
	// CapacityReleasePhaseRefused means the authorization was rejected without changing target, capacity, or fences.
	CapacityReleasePhaseRefused CapacityReleasePhase = "Refused"
	// CapacityReleasePhaseFailed means an accepted release stopped before completion. ObserveCapacity reflects every
	// partial removal and fence, and this release identity can never mutate capacity again.
	CapacityReleasePhaseFailed CapacityReleasePhase = "Failed"
)

// CapacityReleaseObservation is the restart-observable outcome of one authorized release.
type CapacityReleaseObservation struct {
	ReleaseID string
	Phase     CapacityReleasePhase
	Failure   *OperationFailure
}

// RequiredReplicaAllocation binds required logical capacity to its stable workload-manager slot. Physical and runtime
// incarnations may be repaired or replaced within that slot.
type RequiredReplicaAllocation struct {
	ReplicaID ReplicaID
	SlotID    CapacitySlotID
}

// CapacityRequest asks for an operation-correlated absolute allocation count while preserving frozen logical-slot
// bindings.
type CapacityRequest struct {
	OperationID        string
	TopologyGeneration int64
	TargetReplicas     int32
	RequiredReplicas   []RequiredReplicaAllocation
}

// CapacityAdapter owns physical replica allocation and exact authorized release.
type CapacityAdapter interface {
	// ObserveCapacity returns stable allocation identities, concrete Pod UIDs, and durable released-slot fences.
	// A fence may temporarily coexist with capacity created concurrently before the fence was installed; such an
	// allocation remains ineligible for engine membership and must be released or explicitly reopened. Fence
	// observation is linearizable: every fence retains the exact logical-replica-to-slot binding, and once a fence is
	// observed without an allocation, no request predating that fence may later publish one for the fenced identity.
	ObserveCapacity(ctx context.Context, groupID GroupID) (CapacitySnapshot, error)
	// EnsureCapacity idempotently ensures the absolute target and every named allocation identity for the membership
	// operation and observed topology generation. It is additive and must never remove an existing allocation when
	// the target is below current capacity. It serializes concurrent provisioning and never creates a fresh logical
	// allocation once the unfenced allocation count reaches the target; replacement repairs the required logical
	// identity rather than minting an extra one. Target-only growth allocates fresh logical identities; only a later
	// operation's RequiredReplicas may reopen a release-fenced logical slot and remove its observable fence atomically
	// before creating capacity. A required replica whose requested slot differs from its durable fence is rejected.
	// It converges every required replica in its exact named slot and enough unfenced
	// allocations for the target toward Available, repairing or replacing unavailable physical capacity or returning
	// an actionable error. It must reject an existing logical replica in a different slot rather than silently remap it.
	EnsureCapacity(ctx context.Context, groupID GroupID, request CapacityRequest) error
	// ObserveRelease recovers one exact release outcome after timeout or restart. Absent is a durable negative
	// observation: every Release call for the same ID that completed before it is ordered before the observation and
	// cannot later be accepted or mutate capacity. An adapter that cannot prove that ordering returns an error so the
	// caller retains the authorization. Applying eventually advances to Applied or Failed; it is not a terminal state.
	// Refused and Failed are immutable terminal decisions.
	ObserveRelease(ctx context.Context, groupID GroupID, releaseID string) (CapacityReleaseObservation, error)
	// Release atomically establishes TargetReplicas as the group's absolute capacity target at TopologyGeneration.
	// It supersedes all capacity work ordered before that barrier, across operation IDs, and removes only the exact
	// Pod UIDs authorized by the named operation and topology generation. It then durably fences their logical slots
	// as defined by CapacityReleasePhaseApplied. Once Applied is observable, older work may not publish allocation
	// above the target. EnsureCapacity requests from an older topology generation are refused; a closed operation may
	// repair required survivors at or below its barrier but cannot raise the target or reopen a fenced identity. A
	// later operation may change the target explicitly. The adapter durably rejects a Release older than its highest
	// applied topology-generation barrier without changing either target or capacity. Repeating an already-applied
	// release ID returns its original result without reapplying an older target. A
	// conflicting payload or a replica slot now occupied by a different UID must be refused rather than redirected
	// to replacement capacity. A fence-only entry whose matching logical fence already exists is a successful no-op
	// so one absolute request can finish partially completed cleanup. An empty replica list establishes only the
	// absolute target barrier and authorizes no physical removal.
	// If an accepted multi-allocation release cannot finish, the adapter reports Failed only after it has durably
	// stopped all work for that release ID. ObserveCapacity then exposes every partial removal and logical-slot fence.
	// A non-nil error leaves acceptance ambiguous. Once this call returns, a later Absent observation for the same
	// release ID carries the durable ordering guarantee documented by ObserveRelease.
	Release(ctx context.Context, groupID GroupID, authorization ReleaseAuthorization) error
}

// BackendOperationPhase is the membership backend's observable operation state.
type BackendOperationPhase string

const (
	// BackendOperationPhaseAbsent means the backend has no operation with this identity.
	BackendOperationPhaseAbsent BackendOperationPhase = "Absent"
	// BackendOperationPhaseAccepted means the backend acknowledged the operation.
	BackendOperationPhaseAccepted BackendOperationPhase = "Accepted"
	// BackendOperationPhaseCommitting means the backend is changing membership.
	BackendOperationPhaseCommitting BackendOperationPhase = "Committing"
	// BackendOperationPhaseCommitted means the exact correlated request produced the reported topology generation.
	BackendOperationPhaseCommitted BackendOperationPhase = "Committed"
	// BackendOperationPhaseFailed means the backend explicitly failed the operation.
	BackendOperationPhaseFailed BackendOperationPhase = "Failed"
	// BackendOperationPhaseUnknown means the backend cannot correlate the operation identity.
	BackendOperationPhaseUnknown BackendOperationPhase = "Unknown"
)

// BackendOperation is the backend's authoritative observation of one membership operation.
type BackendOperation struct {
	ID             string
	Attempt        int32
	BackendID      string
	TargetReplicas int32
	Phase          BackendOperationPhase
	// CommittedTopology is the exact topology produced by a Committed result. It is nil in every other phase.
	CommittedTopology *MembershipTopology
	Failure           *OperationFailure
}

// MembershipRequest is an absolute, idempotent engine-membership request.
type MembershipRequest struct {
	ID                 string
	Attempt            int32
	Intent             OperationIntent
	Capability         ResolvedOperationCapability
	BaseTopology       MembershipTopology
	TargetReplicas     int32
	JoiningReplicas    []ReplicaIncarnation
	RestoredMembership []ReplicaNativeMembership
	NominatedReplicas  []ReplicaID
	// TargetMembership is the exact desired engine-native mapping for a cardinally stable remap.
	TargetMembership []ReplicaMembership
}

// ObservedMembershipTransition describes an engine-committed topology change that was not submitted through the
// current controller operation. It carries enough immutable context for an adapter to decide whether the exact
// transition may be adopted without replaying membership mutation.
type ObservedMembershipTransition struct {
	PreviousTopology MembershipTopology
	ObservedTopology MembershipTopology
	Plan             OperationPlan
}

// MembershipAdapter observes committed topology and applies idempotent membership operations.
type MembershipAdapter interface {
	// ObserveCapabilities reports complete semantic operation shapes supported for this group. Capabilities describe
	// engine membership and reconfiguration safety, not transient readiness or process-lifecycle ownership.
	ObserveCapabilities(ctx context.Context, groupID GroupID) (MembershipCapabilities, error)
	// ValidatePlan resolves and validates one semantic plan against the authoritative base topology before the caller
	// persists Pending or performs capacity and traffic prework. A successful result freezes support for that immutable
	// plan and resolved capability until its durable operation reaches Committed, terminal Failed, Aborted, or an
	// Unknown state requiring intervention; adapter or engine upgrades that remove support serialize after resolution.
	// A definitive rejection that cannot succeed for the same plan must wrap ErrMembershipOperationUnsupported; every
	// other error is treated as transient and may be retried unchanged.
	ValidatePlan(
		ctx context.Context,
		groupID GroupID,
		baseTopology MembershipTopology,
		plan OperationPlan,
	) (ResolvedOperationCapability, error)
	// ValidateObservedTransition resolves and validates one exact, already-committed external transition. The adapter
	// must reject a transition it cannot correlate with the previous durable serving topology and logical identities.
	// The returned traffic requirement describes the safety contract under which that transition was allowed; callers
	// must not adopt a QuiesceGroup transition unless historical whole-group quiescence is independently provable.
	ValidateObservedTransition(
		ctx context.Context,
		groupID GroupID,
		transition ObservedMembershipTransition,
	) (ResolvedOperationCapability, error)
	// ObserveTopology returns authoritative committed membership rather than process or Pod readiness. Generations
	// increase monotonically, so an operation bound to an older generation can never become valid again after a
	// different generation is observed. Any process or capacity incarnation that may serve must advance the
	// generation before becoming eligible for traffic, even when fixed-slot recovery preserves every logical and native member
	// ID. An adapter must not advance the reported generation until every excluded logical replica's previous native
	// members are quiesced. That topology-level attestation applies to submitted operations and externally recovered
	// topologies;
	// after separate traffic drain, it permits exact release of capacity bound to an excluded logical replica.
	// Between controller-submitted operations, later external generations may remove logical replicas or remap native
	// members of retained replicas. They may also restore a previously excluded stable logical identity through a
	// backend-native replacement path after all release and cleanup ownership for that identity is complete. Such a
	// restoration remains fenced until an explicit Recover plan validates and adopts its exact slot, incarnation, and
	// native-member mapping. An external generation must never invent a fresh logical identity, and an excluded
	// identity cannot be reintroduced while its release may still be outstanding.
	ObserveTopology(ctx context.Context, groupID GroupID) (MembershipTopology, error)
	// ObserveOperation recovers the outcome of one exact operation identity after timeout or restart. A Committed
	// result attests to the complete immutable request previously stored for that ID and attempt, including its base,
	// joining, and nominated replica identities, and returns the exact committed logical-to-native member mapping.
	// Failed is terminal for that exact attempt and guarantees it cannot later commit; an adapter unable to provide
	// that guarantee reports Unknown instead. Absent is also a durable negative observation: after it is returned,
	// every SubmitOperation call for the same ID and attempt that completed before this observation is ordered before
	// it and cannot later be accepted or mutate membership. An adapter that cannot exclude delayed acceptance reports
	// Unknown instead. Operation and topology observations are linearizable with each other and with completed
	// SubmitOperation calls:
	// after a caller observes topology and then observes an operation, that operation result cannot omit a commit
	// already represented by the earlier topology observation. Absent is represented by an otherwise zero-valued
	// BackendOperation, so unrelated identity data can never authorize replay of the queried request.
	ObserveOperation(ctx context.Context, groupID GroupID, operationID string, attempt int32) (BackendOperation, error)
	// ValidateRequest performs a side-effect-free preflight of the exact request after all logical identities are
	// frozen. Success does not authorize the caller to assume a later submission is valid; it only guarantees that
	// support for this durable request remains stable through resolution as described by ValidatePlan. A definitive
	// rejection that cannot succeed for the same request must wrap ErrMembershipOperationUnsupported; every other
	// error is treated as transient and may be retried unchanged.
	ValidateRequest(ctx context.Context, groupID GroupID, request MembershipRequest) error
	// SubmitOperation is the final authority. It is idempotent by request ID and attempt, rejects a conflicting payload
	// for either identity, and atomically validates the complete request, resolved capability, current committed base
	// generation, logical identities, and native-member mapping before applying it. Prior ValidatePlan or
	// ValidateRequest success must not be treated as authorization to skip these checks. A definitive rejection that
	// guarantees no membership mutation can later occur is returned as a correlated terminal
	// BackendOperationPhaseFailed observation. A non-nil error means the submission outcome is ambiguous; callers must
	// retain Submitting and recover it through ObserveOperation. Once this call returns, a later Absent observation for
	// the same ID and attempt carries the durable ordering guarantee documented by ObserveOperation.
	SubmitOperation(ctx context.Context, groupID GroupID, request MembershipRequest) (BackendOperation, error)
}

// TrafficAction identifies one ordered runtime traffic mutation.
type TrafficAction string

const (
	// TrafficActionAdmit makes exact replica incarnations routable.
	TrafficActionAdmit TrafficAction = "Admit"
	// TrafficActionWithdraw makes exact replica incarnations non-routable and drains them.
	TrafficActionWithdraw TrafficAction = "Withdraw"
)

// TrafficCommandPhase is the durable decision for one ordered traffic command.
type TrafficCommandPhase string

const (
	// TrafficCommandPhaseAccepted means the exact command is durable and its asynchronous effect may be in progress.
	TrafficCommandPhaseAccepted TrafficCommandPhase = "Accepted"
	// TrafficCommandPhaseRefused means the exact command is durably rejected and can never mutate traffic state.
	TrafficCommandPhaseRefused TrafficCommandPhase = "Refused"
	// TrafficCommandPhaseFailed means an accepted command stopped before reaching its requested state. Its partial
	// effects, if any, are reflected in the exact traffic snapshot and it can never mutate state again.
	TrafficCommandPhaseFailed TrafficCommandPhase = "Failed"
)

// TrafficCommandObservation is the latest group-ordered traffic command decided by the runtime adapter.
type TrafficCommandObservation struct {
	Command TrafficCommand
	Phase   TrafficCommandPhase
	Failure *OperationFailure
}

// TrafficSnapshot is the runtime's exact current admission and drain state. LatestCommand orders asynchronous
// mutations and records whether the most recent revision was accepted, refused before mutation, or failed after
// acceptance; Admitted and Drained remain authoritative across later non-conflicting commands. Admitted contains at
// most one exact incarnation per logical replica. Drained may retain multiple historical incarnations so replacement
// capacity cannot inherit prior traffic state.
type TrafficSnapshot struct {
	LatestCommand *TrafficCommandObservation
	Admitted      []ReplicaIncarnation
	Drained       []ReplicaIncarnation
}

// TrafficRequest binds one ordered admission or withdrawal to an authoritative topology generation.
type TrafficRequest struct {
	Revision           int64
	OperationID        string
	TopologyGeneration int64
	Replicas           []ReplicaIncarnation
}

// TrafficCommand is one exact, group-ordered traffic mutation persisted before dispatch.
type TrafficCommand struct {
	Action  TrafficAction
	Request TrafficRequest
}

// TrafficAdapter owns runtime traffic admission, withdrawal, and drain observation.
type TrafficAdapter interface {
	// ObserveTraffic returns exact admission and drain state for logical replica incarnations.
	// A replica excluded by an externally recovered authoritative topology must disappear from Admitted without
	// requiring a controller-issued withdrawal for the prior operation; this keeps survivor observations coherent.
	// Engine membership changes and discovery registration never add an identity to Admitted implicitly: a successful
	// Admit call is the only transition that can make a replica routable. Admission is bound internally to the exact
	// topology generation supplied to Admit; an incomplete effect is canceled when that generation ceases to be
	// authoritative, and a later process or capacity incarnation cannot inherit it even when the replacement preserves
	// the same logical and native member IDs.
	// LatestCommand advances as soon as a request is durably accepted or definitively refused, before asynchronous
	// admission or drain completion. Its revision never regresses. An equal revision always identifies the same action
	// and exact request. Accepted may advance only to Failed; Refused and Failed are terminal. Refused guarantees no
	// effect occurred. Failed guarantees no further effect can occur and exposes any partial result through the exact
	// traffic sets. Lower revisions are rejected even if their delayed delivery follows a newer command. Publishing a
	// decision for revision N also guarantees that no command below N can later mutate state. Adapter state ahead of the
	// controller's durable revision, or an equal revision with a different payload, is an invariant violation and fails
	// closed.
	// External topology invalidation may remove Admitted entries without advancing the revision, but cannot add them.
	ObserveTraffic(ctx context.Context, groupID GroupID) (TrafficSnapshot, error)
	// Admit is the only API that makes replicas routable. It idempotently admits only the named replicas for the
	// operation. The adapter atomically refuses
	// a stale topology generation or any replica absent from that exact committed topology.
	// Admitting a replica explicitly clears only the matching exact incarnation's durable drain and preserves
	// historical drain tombstones for prior incarnations of the same logical replica.
	// Repeating the same revision and exact request is idempotent. A conflicting equal revision and every lower
	// revision are rejected. A higher revision supersedes older in-flight work without allowing its later delivery to
	// overwrite the new state. A definitive rejection, including stale topology, is recorded as Refused before the
	// method returns; any other error leaves acceptance ambiguous and callers observe the same revision. After an exact
	// drain is observably complete, a later operation may take ownership and admit only its exact current-generation
	// topology.
	Admit(ctx context.Context, groupID GroupID, request TrafficRequest) error
	// Withdraw idempotently makes the named replicas non-routable and starts or resumes their drain. The adapter
	// atomically refuses a stale topology generation before changing traffic state.
	// One request may name both the prior and current exact incarnations of a logical replica. The adapter preserves
	// each as a distinct drain tombstone even when their stable logical replica and slot identities are equal.
	// Repeating the same revision and exact request is idempotent. A conflicting equal revision and every lower
	// revision are rejected. A higher revision supersedes older in-flight work. A definitive rejection, including stale
	// topology, is recorded as Refused before the method returns; any other error leaves acceptance ambiguous. A later
	// operation may take ownership of an observably complete drain only by preserving or expanding the exact drained
	// set; that handoff cannot route any replica. Once drain is observed, those replicas remain non-routable across
	// process or controller restart until a later explicit Admit operation.
	// After a compensating Admit has observably completed, the same operation may issue a distinct Withdraw for a
	// former member discovered during exact abort cleanup.
	Withdraw(ctx context.Context, groupID GroupID, request TrafficRequest) error
}
