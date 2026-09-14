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

import "time"

// GroupID identifies one independently resizable engine world.
type GroupID string

// ReplicaID identifies one logical replica throughout capacity replacement.
type ReplicaID string

// CapacitySlotID identifies one stable workload-manager position backing a replica.
type CapacitySlotID string

// NativeMemberID identifies one engine-native member or rank.
type NativeMemberID string

// PodUID identifies one concrete Kubernetes Pod instance.
type PodUID string

// ReplicaAvailability is the profile-derived availability of one complete replica allocation.
type ReplicaAvailability string

const (
	// ReplicaAvailabilityAvailable means every capacity Pod and required runtime check is ready.
	ReplicaAvailabilityAvailable ReplicaAvailability = "Available"
	// ReplicaAvailabilityUnavailable means at least one required capacity or runtime check has failed.
	ReplicaAvailabilityUnavailable ReplicaAvailability = "Unavailable"
	// ReplicaAvailabilityUnknown means complete allocation availability cannot be established.
	ReplicaAvailabilityUnknown ReplicaAvailability = "Unknown"
)

// CapacityRef identifies one concrete capacity Pod without coupling the contract to a Kubernetes API type.
type CapacityRef struct {
	Namespace string
	Name      string
	UID       PodUID
}

// ReplicaAllocation is one physically disjoint allocation backing a logical replica.
type ReplicaAllocation struct {
	ID           ReplicaID
	SlotID       CapacitySlotID
	CapacityRefs []CapacityRef
	Availability ReplicaAvailability
}

// CapacitySnapshot is the workload manager's observed allocation state for an Engine Group.
type CapacitySnapshot struct {
	Allocations    []ReplicaAllocation
	FencedReplicas []ReplicaID
}

// ReplicaMembership correlates one logical replica with its engine-native members.
// NativeMembers is authoritative for its topology generation and may change in a later generation.
type ReplicaMembership struct {
	ReplicaID     ReplicaID
	NativeMembers []NativeMemberID
}

// MembershipTopology is the inference engine's authoritative committed membership.
type MembershipTopology struct {
	Generation int64
	Replicas   []ReplicaMembership
}

// ReplicaCount returns the number of committed logical replicas.
func (t MembershipTopology) ReplicaCount() int32 {
	return int32(len(t.Replicas))
}

// OperationIntent identifies why engine membership is changing.
type OperationIntent string

const (
	// OperationIntentGrow adds replicas to a running engine world.
	OperationIntentGrow OperationIntent = "Grow"
	// OperationIntentShrink removes replicas from a running engine world.
	OperationIntentShrink OperationIntent = "Shrink"
	// OperationIntentRecover restores or remaps membership after unexpected loss.
	OperationIntentRecover OperationIntent = "Recover"
	// OperationIntentRetire removes all membership before deleting an engine world.
	OperationIntentRetire OperationIntent = "Retire"
)

// OperationPhase is the durable controller-facing phase of one membership operation.
type OperationPhase string

const (
	// OperationPhasePending means the operation is durable but no membership call may have occurred.
	// Capacity allocation or traffic withdrawal may already be converging for this immutable operation.
	OperationPhasePending OperationPhase = "Pending"
	// OperationPhaseSubmitting means the call may have occurred but acknowledgement is unknown.
	OperationPhaseSubmitting OperationPhase = "Submitting"
	// OperationPhaseAccepted means the backend acknowledged the operation identity and target.
	OperationPhaseAccepted OperationPhase = "Accepted"
	// OperationPhaseCommitting means the backend is changing membership.
	OperationPhaseCommitting OperationPhase = "Committing"
	// OperationPhaseCommitted means the expected committed topology was authoritatively observed.
	OperationPhaseCommitted OperationPhase = "Committed"
	// OperationPhaseFailed means the backend explicitly failed the operation.
	OperationPhaseFailed OperationPhase = "Failed"
	// OperationPhaseUnknown means the operation cannot be correlated with authoritative backend state.
	OperationPhaseUnknown OperationPhase = "Unknown"
	// OperationPhaseAborting means a provably uncommitted operation is durably restoring current serving state.
	OperationPhaseAborting OperationPhase = "Aborting"
	// OperationPhaseAborted means current authoritative members were restored after an operation could not commit.
	OperationPhaseAborted OperationPhase = "Aborted"
)

// FailureClassification states whether the same operation may be retried.
type FailureClassification string

const (
	// FailureClassificationRetryable permits retrying the same operation identity and target.
	FailureClassificationRetryable FailureClassification = "Retryable"
	// FailureClassificationTerminal forbids retrying the failed operation unchanged.
	FailureClassificationTerminal FailureClassification = "Terminal"
)

// OperationFailure is a structured backend failure.
type OperationFailure struct {
	Classification FailureClassification
	Reason         string
	Message        string
}

// OperationPlan describes a caller-owned durable, absolute, identity-aware membership request.
// Reusing its ID with a different payload is invalid.
type OperationPlan struct {
	ID                string
	Intent            OperationIntent
	TargetReplicas    int32
	NominatedReplicas []ReplicaID
}

// MembershipCapabilities lists complete semantic operations an engine adapter can safely perform.
type MembershipCapabilities struct {
	Intents []OperationIntent
}

// Operation is the durable record of one membership transaction.
type Operation struct {
	ID                          string
	Attempt                     int32
	PlanID                      string
	Intent                      OperationIntent
	SpecGeneration              int64
	BaseTopologyGeneration      int64
	BaseReplicas                []ReplicaID
	TargetReplicas              int32
	JoiningReplicas             []ReplicaID
	NominatedReplicas           []ReplicaID
	CleanupReplicas             []ReplicaID
	CapacityTargetReplicas      int32
	CapacityTopologyGeneration  int64
	CapacityTargetApplied       bool
	QueuedTargetReplicas        *int32
	Phase                       OperationPhase
	BackendOperationID          string
	CommittedTopologyGeneration int64
	PostCommitComplete          bool
	Adopted                     bool
	StartedAt                   time.Time
	LastTransitionTime          time.Time
	Failure                     *OperationFailure
}

// OperationInput is the desired and durable state consumed by one coordinator step.
type OperationInput struct {
	GroupID         GroupID
	SpecGeneration  int64
	DesiredReplicas int32
	// Plan carries identity-aware semantics that cannot be derived from desired and active counts.
	// Its owner must retain the same non-empty ID until the resulting operation records that PlanID.
	// A nil Plan permits automatic cardinal growth; reductions require exact nominated identities.
	Plan      *OperationPlan
	Operation *Operation
}

// OperationResult is the complete operation state and topology observed by one coordinator step.
type OperationResult struct {
	Operation        *Operation
	Topology         MembershipTopology
	OperationChanged bool
	SubmissionNeeded bool
}
