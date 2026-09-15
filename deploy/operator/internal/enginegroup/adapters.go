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

// CapacityReplicaTarget describes one logical allocation required by an absolute capacity target.
type CapacityReplicaTarget struct {
	ReplicaID   ReplicaID
	SlotID      CapacitySlotID
	Incarnation *ReplicaIncarnation
	Bootstrap   BootstrapMode
}

// ReleaseFence authorizes removal of only the named concrete Pod incarnations from one stable slot.
type ReleaseFence struct {
	ReplicaID    ReplicaID
	SlotID       CapacitySlotID
	CapacityRefs []CapacityRef
}

// CapacityTarget is one durable, group-ordered absolute physical-capacity projection.
type CapacityTarget struct {
	ControlRevision       int64
	TransitionID          string
	ProfileFingerprint    string
	ProcessLifecycleOwner ProcessLifecycleOwner
	Replicas              []CapacityReplicaTarget
	ReleaseFences         []ReleaseFence
}

// CapacityAllocation is one workload-manager allocation and its complete availability state.
type CapacityAllocation struct {
	Incarnation ReplicaIncarnation
	Available   bool
}

// CapacityObservation is the workload manager's current allocation state.
type CapacityObservation struct {
	AppliedRevision int64
	Allocations     []CapacityAllocation
	ReleaseFences   []ReleaseFence
}

// TrafficTarget is one durable, group-ordered absolute routing projection.
type TrafficTarget struct {
	ControlRevision    int64
	TransitionID       string
	TopologyGeneration int64
	Admitted           []ReplicaMembership
	Drain              []ReplicaMembership
}

// TrafficObservation is the runtime's exact routing and drain state.
type TrafficObservation struct {
	AppliedRevision int64
	Admitted        []ReplicaMembership
	Draining        []ReplicaMembership
	Drained         []ReplicaMembership
}

// MembershipBackendPhase is the engine's observable state for one idempotent membership request.
type MembershipBackendPhase string

const (
	// MembershipBackendPhaseAbsent means the engine has not accepted the operation identity.
	MembershipBackendPhaseAbsent MembershipBackendPhase = "Absent"
	// MembershipBackendPhaseRunning means the engine accepted the operation and is changing membership.
	MembershipBackendPhaseRunning MembershipBackendPhase = "Running"
	// MembershipBackendPhaseCommitted means the request produced the reported topology.
	MembershipBackendPhaseCommitted MembershipBackendPhase = "Committed"
	// MembershipBackendPhaseRejected means the request definitively cannot mutate membership.
	MembershipBackendPhaseRejected MembershipBackendPhase = "Rejected"
	// MembershipBackendPhaseUnknown means the engine cannot establish whether the request mutated membership.
	MembershipBackendPhaseUnknown MembershipBackendPhase = "Unknown"
)

// MembershipRequest is the exact immutable engine request derived from a durable transition.
type MembershipRequest struct {
	ID              string
	BaseTopology    MembershipTopology
	Plan            ResolvedPlan
	JoiningReplicas []JoiningReplica
}

// MembershipOperationObservation is the engine's current result for one request identity.
type MembershipOperationObservation struct {
	ID                string
	Phase             MembershipBackendPhase
	CommittedTopology *MembershipTopology
	Failure           *Failure
}

// ApplyResult reports a definitive, replay-stable rejection of an adapter request.
// For capacity and traffic, rejection guarantees that no further work for the rejected revision can occur; any partial
// effect is returned by the next level-based observation. For membership submission, rejection guarantees no mutation.
// A nil Rejection means the adapter accepted the request or had already applied it.
type ApplyResult struct {
	Rejection *Failure
}

// VerificationResult is one safely rerunnable serving check against an immutable topology.
type VerificationResult struct {
	Proof   *ServingProof
	Failure *Failure
}

// CapacityAdapter converges physical capacity to revisioned absolute targets.
type CapacityAdapter interface {
	// Observe returns stable allocation identities, exact Pod UIDs, availability, and durable release fences.
	Observe(ctx context.Context, groupID GroupID) (CapacityObservation, error)
	// Apply converges to target without deleting capacity absent from an exact UID-bound release fence.
	// Revisions are group-global and monotonic. Repeating an equal revision and payload is idempotent; an equal revision
	// with another payload or a lower revision is definitively rejected. Deletion uses Pod UID preconditions, and an
	// applied fence remains observable until a later target explicitly reopens the stable replica slot.
	Apply(ctx context.Context, groupID GroupID, target CapacityTarget) (ApplyResult, error)
}

// MembershipAdapter observes committed engine topology and owns the one necessarily transactional external operation.
type MembershipAdapter interface {
	// ObserveTopology returns the complete authoritative committed topology. Generation advances whenever any serving
	// process incarnation or logical-to-native membership changes.
	ObserveTopology(ctx context.Context, groupID GroupID) (MembershipTopology, error)
	// ObserveOperation returns the current result for the exact request identity.
	ObserveOperation(
		ctx context.Context,
		groupID GroupID,
		operationID string,
	) (MembershipOperationObservation, error)
	// Submit atomically validates the exact base topology and resolved plan before mutation. It is idempotent by request
	// ID and exact payload. A non-nil error leaves acceptance ambiguous, so callers retry the same identity and payload.
	// A definitive rejection is returned as ApplyResult.Rejection and remains stable when the request is replayed.
	Submit(ctx context.Context, groupID GroupID, request MembershipRequest) (ApplyResult, error)
}

// TrafficAdapter converges runtime discovery and routing to revisioned absolute identity sets.
type TrafficAdapter interface {
	// Observe returns exact admitted, draining, and durably drained replica incarnations.
	Observe(ctx context.Context, groupID GroupID) (TrafficObservation, error)
	// Apply converges toward the exact admitted set and preserves drain tombstones for the target's Drain set. Revisions
	// are group-global and monotonic. A lower revision or conflicting equal revision is definitively rejected. No engine
	// membership or discovery event may implicitly admit an incarnation absent from the latest target.
	Apply(ctx context.Context, groupID GroupID, target TrafficTarget) (ApplyResult, error)
}

// ServingVerifier performs a safely repeatable serving-progress check against one exact committed topology.
type ServingVerifier interface {
	// Verify returns a positive proof or a conclusive failure. A non-nil error is inconclusive and may be retried against
	// the same snapshot. Repeating a successful check is safe; a crash before proof persistence therefore needs no
	// separate verification-operation journal.
	Verify(ctx context.Context, groupID GroupID, topology MembershipTopology) (VerificationResult, error)
}
