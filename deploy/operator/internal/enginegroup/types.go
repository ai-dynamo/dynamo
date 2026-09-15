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

// RuntimeIncarnationID identifies one complete logical-replica runtime incarnation. It changes when any constituent
// engine process restarts, including a restart inside the same Pod, and is unique within an Engine Group.
type RuntimeIncarnationID string

// ReplicaSlotBinding associates one stable logical replica with its workload-manager slot.
type ReplicaSlotBinding struct {
	ReplicaID ReplicaID
	SlotID    CapacitySlotID
}

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

// ReplicaIncarnation identifies the exact physical and runtime instance behind one stable logical replica.
// SlotID remains stable across replacement; CapacityRefs changes with a Pod and RuntimeID with an engine process.
type ReplicaIncarnation struct {
	ReplicaID    ReplicaID
	SlotID       CapacitySlotID
	CapacityRefs []CapacityRef
	RuntimeID    RuntimeIncarnationID
}

// ReplicaAllocation is one physically disjoint allocation backing a logical replica.
type ReplicaAllocation struct {
	Incarnation  ReplicaIncarnation
	Availability ReplicaAvailability
}

// CapacitySnapshot is the workload manager's observed allocation state for an Engine Group.
type CapacitySnapshot struct {
	Allocations        []ReplicaAllocation
	FencedReplicaSlots []ReplicaSlotBinding
}

// ReplicaMembership correlates one exact physical incarnation with its engine-native members.
// The complete mapping is authoritative for its topology generation and remains durable after capacity loss.
type ReplicaMembership struct {
	Incarnation   ReplicaIncarnation
	NativeMembers []NativeMemberID
}

// ReplicaNativeMembership binds one stable logical replica and capacity slot to the native members a recovery must
// restore. A replacement may use new physical and runtime incarnations, but it must return to this exact slot.
type ReplicaNativeMembership struct {
	ReplicaID     ReplicaID
	SlotID        CapacitySlotID
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
	// OperationPhaseFailed means the backend definitively failed the membership request or a required post-commit
	// serving step failed terminally.
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

// OperationFailure is a structured membership or serving-workflow failure.
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
	// RestoredMembership names the absent logical identities, stable slots, and exact native members restored by
	// recovery expansion.
	RestoredMembership []ReplicaNativeMembership
	// TargetMembership is the exact desired engine-native mapping for a cardinally stable remap.
	TargetMembership []ReplicaMembership
}

// OperationShape identifies the exact membership transition an engine adapter can safely perform.
type OperationShape string

const (
	// OperationShapeFreshGrowth adds new logical replicas to a healthy topology.
	OperationShapeFreshGrowth OperationShape = "FreshGrowth"
	// OperationShapePlannedHighRankSuffixShrink removes a high-rank suffix of healthy logical replicas.
	OperationShapePlannedHighRankSuffixShrink OperationShape = "PlannedHighRankSuffixShrink"
	// OperationShapePlannedSelectedRetirement removes an arbitrary caller-selected set of healthy logical replicas.
	OperationShapePlannedSelectedRetirement OperationShape = "PlannedSelectedRetirement"
	// OperationShapeSurvivorReduction adopts or commits a smaller topology after unexpected member loss.
	OperationShapeSurvivorReduction OperationShape = "SurvivorReduction"
	// OperationShapeReplacementRestoration restores missing logical replicas to a survivor topology.
	OperationShapeReplacementRestoration OperationShape = "ReplacementRestoration"
	// OperationShapeFixedSlotReplacement replaces physical or runtime incarnations while preserving logical and native IDs.
	OperationShapeFixedSlotReplacement OperationShape = "FixedSlotReplacement"
	// OperationShapeNativeMemberRemapping changes engine-native membership without changing logical cardinality.
	OperationShapeNativeMemberRemapping OperationShape = "NativeMemberRemapping"
	// OperationShapeFullRetirement removes all membership before deleting an engine world.
	OperationShapeFullRetirement OperationShape = "FullRetirement"
)

// ReconfigurationTrafficRequirement states how serving traffic must be handled while an operation commits.
type ReconfigurationTrafficRequirement string

const (
	// ReconfigurationTrafficKeepServing means existing members may safely keep serving during the transition.
	ReconfigurationTrafficKeepServing ReconfigurationTrafficRequirement = "KeepServing"
	// ReconfigurationTrafficQuiesceGroup means the complete group must be withdrawn and drained before commit.
	ReconfigurationTrafficQuiesceGroup ReconfigurationTrafficRequirement = "QuiesceGroup"
)

// ServingVerificationRequirement states whether a committed topology needs a separate serving proof before admission.
type ServingVerificationRequirement string

const (
	// ServingVerificationNotRequired means the resolved profile does not require an additional post-commit check.
	ServingVerificationNotRequired ServingVerificationRequirement = "NotRequired"
	// ServingVerificationRequired means the exact committed topology must pass serving verification before admission.
	ServingVerificationRequired ServingVerificationRequirement = "Required"
)

// CapacityRecoveryPhase is the durable state of re-establishing serving after physical availability regresses
// without an engine topology change.
type CapacityRecoveryPhase string

const (
	// CapacityRecoveryPhaseNone means no same-topology capacity recovery is in progress.
	CapacityRecoveryPhaseNone CapacityRecoveryPhase = ""
	// CapacityRecoveryPhaseRepairing means traffic is fenced and exact physical capacity is being restored.
	CapacityRecoveryPhaseRepairing CapacityRecoveryPhase = "Repairing"
	// CapacityRecoveryPhaseVerifying means capacity is available and a post-repair serving check is required.
	CapacityRecoveryPhaseVerifying CapacityRecoveryPhase = "Verifying"
)

// ResolvedOperationCapability is the engine safety contract for one concrete durable membership operation.
type ResolvedOperationCapability struct {
	Shape                   OperationShape
	TrafficRequirement      ReconfigurationTrafficRequirement
	VerificationRequirement ServingVerificationRequirement
}

// MembershipCapabilities lists the exact operation shapes an engine adapter can safely perform.
// Process launch and teardown ownership is a separate profile and cross-adapter concern.
type MembershipCapabilities struct {
	OperationShapes []OperationShape
}

// Operation is the durable record of one membership transaction.
type Operation struct {
	ID                         string
	Attempt                    int32
	PlanID                     string
	Intent                     OperationIntent
	Capability                 ResolvedOperationCapability
	SpecGeneration             int64
	BaseTopology               MembershipTopology
	TargetReplicas             int32
	RestoredMembership         []ReplicaNativeMembership
	JoiningReplicas            []ReplicaIncarnation
	NominatedReplicas          []ReplicaID
	TargetMembership           []ReplicaMembership
	CleanupReplicaSlots        []ReplicaSlotBinding
	CapacityTargetReplicas     int32
	CapacityTopologyGeneration int64
	CapacityTargetApplied      bool
	QueuedTargetReplicas       *int32
	Phase                      OperationPhase
	BackendOperationID         string
	CommittedTopology          *MembershipTopology
	// CompensationTopology is the exact authoritative topology restored after a provably uncommitted operation.
	CompensationTopology       *MembershipTopology
	ServingVerificationAttempt int32
	ServingVerificationTarget  *MembershipTopology
	ServingVerificationProof   *ServingVerificationProof
	// TerminalAdmissionFailure preserves a non-retryable traffic-admission outcome after later traffic commands
	// supersede it in the adapter's single latest-command register.
	TerminalAdmissionFailure *TrafficAdmissionFailure
	// CapacityRecoveryPhase prevents a restart or repeated reconcile from reusing serving evidence that predates an
	// availability regression in an otherwise unchanged topology.
	CapacityRecoveryPhase CapacityRecoveryPhase
	PostCommitComplete    bool
	Adopted               bool
	StartedAt             time.Time
	LastTransitionTime    time.Time
	Failure               *OperationFailure
}

// TrafficAdmissionFailure binds one terminal admission failure to its exact refused or partially applied command.
type TrafficAdmissionFailure struct {
	Command TrafficCommand
	Failure OperationFailure
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
	TopologyObserved bool
	OperationChanged bool
	SubmissionNeeded bool
}
