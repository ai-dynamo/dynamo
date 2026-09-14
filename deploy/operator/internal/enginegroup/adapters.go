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

// AuthorizedReplica binds one logical replica to every concrete Pod authorized for removal.
// Empty SlotID and CapacityRefs request only a durable logical fence for already-absent capacity.
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
	// CapacityReleasePhaseAbsent means no release with this operation identity has been attempted.
	CapacityReleasePhaseAbsent CapacityReleasePhase = "Absent"
	// CapacityReleasePhaseApplying means removal of the exact authorized capacity is still converging.
	CapacityReleasePhaseApplying CapacityReleasePhase = "Applying"
	// CapacityReleasePhaseApplied means the absolute target barrier is durable, every authorized Pod UID has been
	// removed, and each released logical slot is fenced until a later operation explicitly requires that replica ID.
	CapacityReleasePhaseApplied CapacityReleasePhase = "Applied"
	// CapacityReleasePhaseRefused means the authorization was rejected without changing target, capacity, or fences.
	CapacityReleasePhaseRefused CapacityReleasePhase = "Refused"
)

// CapacityReleaseObservation is the restart-observable outcome of one authorized release.
type CapacityReleaseObservation struct {
	ReleaseID string
	Phase     CapacityReleasePhase
	Failure   *OperationFailure
}

// CapacityRequest asks for an operation-correlated absolute allocation count while preserving frozen identities.
type CapacityRequest struct {
	OperationID        string
	TopologyGeneration int64
	TargetReplicas     int32
	RequiredReplicas   []ReplicaID
}

// CapacityAdapter owns physical replica allocation and exact authorized release.
type CapacityAdapter interface {
	// ObserveCapacity returns stable allocation identities, concrete Pod UIDs, and durable released-slot fences.
	// A fence may temporarily coexist with capacity created concurrently before the fence was installed; such an
	// allocation remains ineligible for engine membership and must be released or explicitly reopened. Fence
	// observation is linearizable: once a fence is observed without an allocation, no request predating that fence
	// may later publish one for the fenced identity.
	ObserveCapacity(ctx context.Context, groupID GroupID) (CapacitySnapshot, error)
	// EnsureCapacity idempotently ensures the absolute target and every named allocation identity for the membership
	// operation and observed topology generation. It is additive and must never remove an existing allocation when
	// the target is below current capacity. It serializes concurrent provisioning and never creates a fresh logical
	// allocation once the unfenced allocation count reaches the target; replacement repairs the required logical
	// identity rather than minting an extra one. Target-only growth allocates fresh logical identities; only a later
	// operation's RequiredReplicas may reopen a release-fenced logical slot and remove its observable fence atomically
	// before creating capacity. It converges every required replica and enough unfenced allocations for the target
	// toward Available, repairing or replacing unavailable physical capacity or returning an actionable error.
	EnsureCapacity(ctx context.Context, groupID GroupID, request CapacityRequest) error
	// ObserveRelease recovers one exact release outcome after timeout or restart.
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
	ID                          string
	Attempt                     int32
	BackendID                   string
	TargetReplicas              int32
	Phase                       BackendOperationPhase
	CommittedTopologyGeneration int64
	Failure                     *OperationFailure
}

// MembershipRequest is an absolute, idempotent engine-membership request.
type MembershipRequest struct {
	ID                     string
	Attempt                int32
	Intent                 OperationIntent
	BaseTopologyGeneration int64
	BaseReplicas           []ReplicaID
	TargetReplicas         int32
	JoiningReplicas        []ReplicaID
	NominatedReplicas      []ReplicaID
}

// MembershipAdapter observes committed topology and applies idempotent membership operations.
type MembershipAdapter interface {
	// ObserveCapabilities reports complete semantic operations supported for this group. Recover support includes
	// any backend-specific pause or quiescence required to commit or attest a survivor topology safely. Capabilities
	// describe semantics, not transient readiness. Once an intent is advertised and a durable Pending operation is
	// created, that intent remains supported until the operation reaches Committed, terminal Failed, Aborted, or an
	// Unknown state requiring intervention; adapter or engine upgrades that remove support serialize after resolution.
	ObserveCapabilities(ctx context.Context, groupID GroupID) (MembershipCapabilities, error)
	// ObserveTopology returns authoritative committed membership rather than process or Pod readiness. Generations
	// increase monotonically, so an operation bound to an older generation can never become valid again after a
	// different generation is observed. An adapter must not advance the reported generation until every excluded
	// logical replica's previous native members are
	// quiesced. That topology-level attestation applies to submitted operations and externally recovered topologies;
	// after separate traffic drain, it permits exact release of capacity bound to an excluded logical replica.
	// Between controller-submitted operations, later external generations may only remove members: an excluded
	// logical replica cannot be reintroduced autonomously while its release may be outstanding.
	ObserveTopology(ctx context.Context, groupID GroupID) (MembershipTopology, error)
	// ObserveOperation recovers the outcome of one exact operation identity after timeout or restart. A Committed
	// result attests to the complete immutable request previously stored for that ID and attempt, including its base,
	// joining, and nominated replica identities. Failed is terminal for that exact attempt and guarantees it cannot
	// later commit; an adapter unable to provide that guarantee reports Unknown instead. Operation and topology
	// observations are linearizable with each other: after a caller observes topology and then observes an operation,
	// that operation result cannot omit a commit already represented by the earlier topology observation.
	ObserveOperation(ctx context.Context, groupID GroupID, operationID string, attempt int32) (BackendOperation, error)
	// SubmitOperation is idempotent by request ID and attempt and rejects a conflicting payload for either identity.
	// It atomically compares BaseTopologyGeneration and BaseReplicas with current committed membership before
	// applying the request, refusing a stale base instead of mutating a different topology.
	SubmitOperation(ctx context.Context, groupID GroupID, request MembershipRequest) (BackendOperation, error)
}

// TrafficSnapshot is the runtime's operation-scoped admission and drain state.
type TrafficSnapshot struct {
	OperationID string
	Admitted    []ReplicaID
	Drained     []ReplicaID
}

// TrafficRequest binds one admission or withdrawal to an authoritative topology generation.
type TrafficRequest struct {
	OperationID        string
	TopologyGeneration int64
	Replicas           []ReplicaID
}

// TrafficAdapter owns runtime traffic admission, withdrawal, and drain observation.
type TrafficAdapter interface {
	// ObserveTraffic returns operation-scoped admission and drain state for exact logical replicas.
	// A replica excluded by an externally recovered authoritative topology must disappear from Admitted without
	// requiring a controller-issued withdrawal for the prior operation; this keeps survivor observations coherent.
	ObserveTraffic(ctx context.Context, groupID GroupID) (TrafficSnapshot, error)
	// Admit idempotently makes only the named replicas routable for the operation. The adapter atomically refuses
	// a stale topology generation or any replica absent from that exact committed topology.
	// Admitting a replica explicitly clears any durable drain left by the same aborted operation.
	// Repeating the same admission request is idempotent; a conflicting admission payload or different operation
	// must be rejected while a prior traffic mutation remains unresolved. A compensating Admit after Withdraw is
	// an explicit transition of the same operation, not a conflicting admission replay.
	Admit(ctx context.Context, groupID GroupID, request TrafficRequest) error
	// Withdraw idempotently makes the named replicas non-routable and starts or resumes their drain. The adapter
	// atomically refuses a stale topology generation before changing traffic state.
	// Repeating the same withdrawal request is idempotent; a conflicting withdrawal payload or different operation
	// must be rejected while a prior traffic mutation remains unresolved. Once drain is observed, those replicas
	// remain non-routable across process or controller restart until a later explicit Admit operation.
	// After a compensating Admit has observably completed, the same operation may issue a distinct Withdraw for a
	// former member discovered during exact abort cleanup.
	Withdraw(ctx context.Context, groupID GroupID, request TrafficRequest) error
}
