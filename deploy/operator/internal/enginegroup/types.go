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

// ReplicaID identifies one logical replica throughout physical replacement.
type ReplicaID string

// CapacitySlotID identifies one stable workload-manager position backing a replica.
type CapacitySlotID string

// NativeMemberID identifies one engine-native member, such as a DP rank.
type NativeMemberID string

// PodUID identifies one concrete Kubernetes Pod incarnation.
type PodUID string

// RuntimeIncarnationID identifies one engine process incarnation.
type RuntimeIncarnationID string

// CapacityRef identifies one concrete Pod allocated to a logical replica.
type CapacityRef struct {
	Name string
	UID  PodUID
}

// ReplicaIncarnation binds one logical replica and stable slot to concrete physical and runtime capacity.
type ReplicaIncarnation struct {
	ReplicaID          ReplicaID
	SlotID             CapacitySlotID
	RuntimeIncarnation RuntimeIncarnationID
	CapacityRefs       []CapacityRef
}

// ReplicaMembership is the engine-owned logical, runtime, and native-member identity of one active replica.
type ReplicaMembership struct {
	ReplicaID          ReplicaID
	RuntimeIncarnation RuntimeIncarnationID
	NativeMembers      []NativeMemberID
}

// JoiningReplica identifies one concrete engine process that may join membership.
type JoiningReplica struct {
	ReplicaID          ReplicaID
	RuntimeIncarnation RuntimeIncarnationID
}

// ReplicaNativeMembership describes an exact stable logical-to-native membership mapping.
type ReplicaNativeMembership struct {
	ReplicaID     ReplicaID
	SlotID        CapacitySlotID
	NativeMembers []NativeMemberID
}

// MembershipTopology is one immutable engine-authoritative committed topology snapshot.
type MembershipTopology struct {
	Generation int64
	Replicas   []ReplicaMembership
}

// ReplicaCount returns the number of committed logical replicas.
func (t MembershipTopology) ReplicaCount() int32 {
	return int32(len(t.Replicas))
}

// ProcessLifecycleOwner identifies the system responsible for starting and stopping engine processes.
type ProcessLifecycleOwner string

const (
	// ProcessLifecycleOwnerEngine means the engine starts and stops its own members.
	ProcessLifecycleOwnerEngine ProcessLifecycleOwner = "Engine"
	// ProcessLifecycleOwnerOrchestrator means the orchestrator starts and stops engine members.
	ProcessLifecycleOwnerOrchestrator ProcessLifecycleOwner = "Orchestrator"
)

// BootstrapMode describes how one new physical incarnation joins the engine world.
type BootstrapMode string

const (
	// BootstrapModeJoin creates a previously unknown logical member.
	BootstrapModeJoin BootstrapMode = "Join"
	// BootstrapModeRestoreFixedSlot restores a known logical and native-member identity.
	BootstrapModeRestoreFixedSlot BootstrapMode = "RestoreFixedSlot"
)

// TrafficRequirement describes the traffic boundary required around membership mutation.
type TrafficRequirement string

const (
	// TrafficRequirementKeepServing permits retained replicas to serve throughout the mutation.
	TrafficRequirementKeepServing TrafficRequirement = "KeepServing"
	// TrafficRequirementQuiesceGroup requires the complete base topology to drain before mutation.
	TrafficRequirementQuiesceGroup TrafficRequirement = "QuiesceGroup"
)

// VerificationRequirement describes whether a committed topology must prove serving progress before admission.
type VerificationRequirement string

const (
	// VerificationRequirementNone permits admission immediately after a valid membership commit.
	VerificationRequirementNone VerificationRequirement = "None"
	// VerificationRequirementRequired requires a matching serving proof before admission.
	VerificationRequirementRequired VerificationRequirement = "Required"
)

// PlanKind identifies the concrete membership-change semantics of a resolved plan.
type PlanKind string

const (
	// PlanKindGrow adds previously unknown logical replicas.
	PlanKindGrow PlanKind = "Grow"
	// PlanKindRetire removes explicitly selected healthy replicas.
	PlanKindRetire PlanKind = "Retire"
	// PlanKindReduceToSurvivors requests an exact survivor set after failure.
	PlanKindReduceToSurvivors PlanKind = "ReduceToSurvivors"
	// PlanKindRestore restores excluded stable replicas in their original slots.
	PlanKindRestore PlanKind = "Restore"
	// PlanKindRemap changes native membership without changing logical cardinality.
	PlanKindRemap PlanKind = "Remap"
)

// ReplicaTarget is the resolved physical identity and bootstrap intent for one joining replica.
type ReplicaTarget struct {
	ReplicaID     ReplicaID
	SlotID        CapacitySlotID
	Bootstrap     BootstrapMode
	NativeMembers []NativeMemberID
}

// RestorationTarget describes one stable logical identity and native membership to restore.
type RestorationTarget struct {
	ReplicaTarget
}

// GrowChange adds the named new logical replicas.
type GrowChange struct {
	Replicas []ReplicaTarget
}

// RetireChange removes the named healthy logical replicas after drain.
type RetireChange struct {
	Replicas []ReplicaID
}

// ReduceToSurvivorsChange requests removal of every base member not present in Survivors.
type ReduceToSurvivorsChange struct {
	Survivors []ReplicaID
}

// RestoreChange restores the named stable logical and native-member identities.
type RestoreChange struct {
	Replicas []RestorationTarget
}

// RemapChange replaces the complete logical-to-native member mapping at the same cardinality.
type RemapChange struct {
	Membership []ReplicaNativeMembership
}

// ResolvedChange is the serializable tagged union of membership changes understood by the coordinator.
// Exactly one variant must be present and must match Kind.
type ResolvedChange struct {
	Kind              PlanKind                 `json:"kind"`
	Grow              *GrowChange              `json:"grow,omitempty"`
	Retire            *RetireChange            `json:"retire,omitempty"`
	ReduceToSurvivors *ReduceToSurvivorsChange `json:"reduceToSurvivors,omitempty"`
	Restore           *RestoreChange           `json:"restore,omitempty"`
	Remap             *RemapChange             `json:"remap,omitempty"`
}

// ResolvedPlan is an immutable, profile-resolved membership transition.
type ResolvedPlan struct {
	ID                      string
	ProfileFingerprint      string
	ProcessLifecycleOwner   ProcessLifecycleOwner
	TrafficRequirement      TrafficRequirement
	VerificationRequirement VerificationRequirement
	Change                  ResolvedChange
}

// ReplicaHistoryEntry retains one excluded incarnation and its exact engine-native membership.
type ReplicaHistoryEntry struct {
	TopologyGeneration int64
	Incarnation        ReplicaIncarnation
	NativeMembers      []NativeMemberID
}

// ReplicaRecord is the canonical durable record for one logical replica and stable slot.
type ReplicaRecord struct {
	ReplicaID ReplicaID
	SlotID    CapacitySlotID
	Current   *ReplicaIncarnation
	History   []ReplicaHistoryEntry
}

// ReplicaRegistry is the single authoritative home for logical-replica-to-slot bindings.
type ReplicaRegistry struct {
	Replicas []ReplicaRecord
}

// TopologyHistory retains immutable snapshots referenced by the current transition and serving proofs.
type TopologyHistory struct {
	CurrentGeneration int64
	Snapshots         []MembershipTopology
}

// TransitionSpec is the immutable durable input to one membership transition.
type TransitionSpec struct {
	ID                     string
	BaseTopologyGeneration int64
	Plan                   ResolvedPlan
}

// FailureClassification states whether reconciliation may retry the same external intent.
type FailureClassification string

const (
	// FailureClassificationRetryable means the same intent may be tried again.
	FailureClassificationRetryable FailureClassification = "Retryable"
	// FailureClassificationTerminal means the same intent cannot safely make progress.
	FailureClassificationTerminal FailureClassification = "Terminal"
)

// Failure is one structured subsystem or transition failure.
type Failure struct {
	Classification FailureClassification
	Reason         string
	Message        string
}

// VerificationPhase is the independent state of serving verification.
type VerificationPhase string

const (
	// VerificationPhasePending means a proof is required for the committed topology.
	VerificationPhasePending VerificationPhase = "Pending"
	// VerificationPhasePassed means the exact topology produced serving progress.
	VerificationPhasePassed VerificationPhase = "Passed"
	// VerificationPhaseFailed means a conclusive check failed.
	VerificationPhaseFailed VerificationPhase = "Failed"
)

// ServingProof binds successful serving verification to one immutable topology snapshot.
type ServingProof struct {
	TopologyGeneration int64
	RuntimeDigest      string
	ObservedAt         time.Time
}

// VerificationStatus records serving health without rewriting membership history.
type VerificationStatus struct {
	Phase   VerificationPhase
	Proof   *ServingProof
	Failure *Failure
}

// PreflightStatus durably records the exact digest and authoritative outcome of one side-effect-free validation.
type PreflightStatus struct {
	TransitionID    string
	ControlRevision int64
	SubjectDigest   string
	Evidence        *ValidationEvidence
	Rejection       *Failure
}

// TransitionOutcome summarizes the complete cross-subsystem transition.
type TransitionOutcome string

const (
	// TransitionOutcomeProgressing means reconciliation still has safe work to perform.
	TransitionOutcomeProgressing TransitionOutcome = "Progressing"
	// TransitionOutcomeReverting means a provably uncommitted transition is restoring its canonical base state.
	TransitionOutcomeReverting TransitionOutcome = "Reverting"
	// TransitionOutcomeRolledBack means preparatory traffic and capacity were restored after definitive rejection.
	TransitionOutcomeRolledBack TransitionOutcome = "RolledBack"
	// TransitionOutcomeBlocked means membership or a post-commit serving step cannot proceed automatically.
	TransitionOutcomeBlocked TransitionOutcome = "Blocked"
	// TransitionOutcomeCompleted means capacity, membership, traffic, and verification reached the resolved plan.
	TransitionOutcomeCompleted TransitionOutcome = "Completed"
)

// TransitionStatus owns one immutable transition and its independent verification progress.
type TransitionStatus struct {
	Spec            TransitionSpec
	PlanPreflight   PreflightStatus
	TargetPreflight PreflightStatus
	Verification    VerificationStatus
	Outcome         TransitionOutcome
	Failure         *Failure
	StartedAt       time.Time
	UpdatedAt       time.Time
}

// CapacityStatus records the latest requested and durably accepted absolute capacity projections.
type CapacityStatus struct {
	// Desired is the newest persisted target, including a target that may later be definitively rejected.
	Desired *CapacityTarget
	// Accepted is the last Desired payload whose exact revision the adapter durably acknowledged.
	Accepted *CapacityTarget
	Observed CapacityObservation
}

// TrafficStatus records the latest requested and durably accepted absolute traffic projections.
type TrafficStatus struct {
	// Desired is the newest persisted target, including a target that may later be definitively rejected.
	Desired *TrafficTarget
	// Accepted is the last Desired payload whose exact revision the adapter durably acknowledged.
	Accepted *TrafficTarget
	Observed TrafficObservation
}

// MembershipStatus records one desired topology transition and the adapter's latest correlated observation.
type MembershipStatus struct {
	Desired  *MembershipTarget
	Observed MembershipObservation
}

// GroupStatus is the complete durable state owned by one Engine Group reconciler.
type GroupStatus struct {
	ControlRevision int64
	Registry        ReplicaRegistry
	Topologies      TopologyHistory
	Capacity        CapacityStatus
	Traffic         TrafficStatus
	Membership      MembershipStatus
	Transition      *TransitionStatus
}

// ReconcileResult contains the complete desired status and whether prompt requeueing is useful.
type ReconcileResult struct {
	Status  GroupStatus
	Requeue bool
}
