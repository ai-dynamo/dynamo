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

package v1beta1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// EngineGroupReconciliationStatus is the complete restart journal consumed by the Engine Group controller.
type EngineGroupReconciliationStatus struct {
	// controlRevision orders every capacity, traffic, and membership target within this group.
	// +kubebuilder:validation:Minimum=0
	ControlRevision int64 `json:"controlRevision"`

	// registry preserves stable logical-replica and capacity-slot identities across replacement.
	// +optional
	// +listType=map
	// +listMapKey=replicaID
	Registry []EngineGroupReplicaRecordStatus `json:"registry,omitempty"`

	// topologyHistory retains the current topology and snapshots referenced by durable evidence.
	TopologyHistory EngineGroupTopologyHistoryStatus `json:"topologyHistory"`

	// capacity records the newest desired target, the last accepted target, and physical observation.
	Capacity EngineGroupCapacityReconciliationStatus `json:"capacity"`

	// traffic records the newest desired target, the last accepted target, and routing observation.
	Traffic EngineGroupTrafficReconciliationStatus `json:"traffic"`

	// membership records the exact desired transition and adapter-owned correlated observation.
	Membership EngineGroupMembershipReconciliationStatus `json:"membership"`

	// transition owns the immutable plan and cross-subsystem progress for one active or terminal change.
	// +optional
	Transition *EngineGroupTransitionStatus `json:"transition,omitempty"`
}

// EngineGroupReplicaRecordStatus is the canonical durable record for one logical replica and stable slot.
type EngineGroupReplicaRecordStatus struct {
	// replicaID is stable across physical and runtime replacement.
	// +kubebuilder:validation:MinLength=1
	ReplicaID string `json:"replicaID"`

	// slotID identifies the stable workload-manager position backing this replica.
	// +kubebuilder:validation:MinLength=1
	SlotID string `json:"slotID"`

	// current is the concrete physical and runtime incarnation currently assigned to the slot.
	// +optional
	Current *EngineGroupControlIncarnationStatus `json:"current,omitempty"`

	// history retains excluded incarnations and their exact native membership.
	// +optional
	History []EngineGroupReplicaHistoryStatus `json:"history,omitempty"`
}

// EngineGroupReplicaHistoryStatus retains one excluded incarnation and its exact native membership.
type EngineGroupReplicaHistoryStatus struct {
	// topologyGeneration identifies the topology that excluded this incarnation.
	// +kubebuilder:validation:Minimum=1
	TopologyGeneration int64 `json:"topologyGeneration"`

	// incarnation is the concrete allocation and runtime identity that was excluded.
	Incarnation EngineGroupControlIncarnationStatus `json:"incarnation"`

	// nativeMembers are the backend-native identities formerly correlated with this replica.
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:items:MinLength=1
	NativeMembers []string `json:"nativeMembers"`
}

// EngineGroupControlIncarnationStatus binds stable logical and slot identities to concrete capacity.
type EngineGroupControlIncarnationStatus struct {
	// replicaID is the stable logical identity.
	// +kubebuilder:validation:MinLength=1
	ReplicaID string `json:"replicaID"`

	// slotID is the stable workload-manager position.
	// +kubebuilder:validation:MinLength=1
	SlotID string `json:"slotID"`

	// runtimeIncarnation identifies the concrete engine process incarnation.
	// +kubebuilder:validation:MinLength=1
	RuntimeIncarnation string `json:"runtimeIncarnation"`

	// capacityRefs contains every concrete Pod in this replica allocation.
	// +kubebuilder:validation:MinItems=1
	CapacityRefs []EngineGroupCapacityRef `json:"capacityRefs"`
}

// EngineGroupTopologyHistoryStatus retains immutable topology snapshots needed by durable evidence.
type EngineGroupTopologyHistoryStatus struct {
	// currentGeneration identifies the engine's current authoritative topology snapshot.
	// +kubebuilder:validation:Minimum=1
	CurrentGeneration int64 `json:"currentGeneration"`

	// snapshots contains the current topology and any snapshot still referenced by controller state.
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=generation
	Snapshots []EngineGroupTopologyStatus `json:"snapshots"`
}

// EngineGroupCapacityReconciliationStatus records desired, accepted, and observed capacity levels.
type EngineGroupCapacityReconciliationStatus struct {
	// desired is the newest persisted absolute target, including one that may later be rejected.
	// +optional
	Desired *EngineGroupCapacityTargetStatus `json:"desired,omitempty"`

	// accepted is the last exact target durably acknowledged by the capacity adapter.
	// +optional
	Accepted *EngineGroupCapacityTargetStatus `json:"accepted,omitempty"`

	// observed is the workload manager's authoritative allocation and release-fence state.
	Observed EngineGroupCapacityObservationStatus `json:"observed"`
}

// EngineGroupCapacityTargetStatus is one group-ordered absolute physical-capacity projection.
type EngineGroupCapacityTargetStatus struct {
	// controlRevision orders this target within the Engine Group.
	// +kubebuilder:validation:Minimum=1
	ControlRevision int64 `json:"controlRevision"`

	// transitionID correlates this target with one membership transition.
	// +kubebuilder:validation:MinLength=1
	TransitionID string `json:"transitionID"`

	// profileFingerprint binds the target to immutable engine and workload geometry.
	// +kubebuilder:validation:MinLength=1
	ProfileFingerprint string `json:"profileFingerprint"`

	// processLifecycleOwner identifies who starts and stops engine processes.
	ProcessLifecycleOwner EngineGroupProcessLifecycleOwner `json:"processLifecycleOwner"`

	// replicas is the complete desired set of physical replica allocations.
	// +optional
	// +listType=map
	// +listMapKey=replicaID
	Replicas []EngineGroupCapacityReplicaTargetStatus `json:"replicas,omitempty"`

	// releaseFences authorize removal of exact Pod UIDs from exact stable slots.
	// +optional
	ReleaseFences []EngineGroupReleaseAuthorization `json:"releaseFences,omitempty"`
}

// EngineGroupCapacityReplicaTargetStatus describes one allocation required by an absolute target.
type EngineGroupCapacityReplicaTargetStatus struct {
	// replicaID is the stable logical identity.
	// +kubebuilder:validation:MinLength=1
	ReplicaID string `json:"replicaID"`

	// slotID identifies the stable workload-manager position.
	// +kubebuilder:validation:MinLength=1
	SlotID string `json:"slotID"`

	// incarnation asserts the exact existing capacity that must remain assigned.
	// +optional
	Incarnation *EngineGroupControlIncarnationStatus `json:"incarnation,omitempty"`

	// bootstrap authorizes creation of a new orchestrator-owned incarnation.
	// +optional
	Bootstrap *EngineGroupCapacityBootstrapStatus `json:"bootstrap,omitempty"`
}

// EngineGroupCapacityBootstrapStatus is the profile-resolved creation intent for one allocation.
type EngineGroupCapacityBootstrapStatus struct {
	// mode distinguishes fresh joining capacity from fixed-slot restoration.
	Mode EngineGroupBootstrapMode `json:"mode"`

	// baseTopologyGeneration binds bootstrap to the topology from which the change starts.
	// +kubebuilder:validation:Minimum=1
	BaseTopologyGeneration int64 `json:"baseTopologyGeneration"`

	// nativeMembers names the fixed backend identities when the bootstrap mode requires them.
	// +optional
	// +kubebuilder:validation:items:MinLength=1
	NativeMembers []string `json:"nativeMembers,omitempty"`
}

// EngineGroupCapacityObservationStatus is the workload manager's authoritative capacity state.
type EngineGroupCapacityObservationStatus struct {
	// appliedRevision is the last capacity target revision durably accepted by the adapter.
	// +kubebuilder:validation:Minimum=0
	AppliedRevision int64 `json:"appliedRevision"`

	// allocations contains every concrete logical-replica allocation currently owned by the group.
	// +optional
	// +listType=map
	// +listMapKey=replicaID
	Allocations []EngineGroupCapacityAllocationStatus `json:"allocations,omitempty"`

	// releaseFences are the exact release authorizations durably enforced by the adapter.
	// +optional
	ReleaseFences []EngineGroupReleaseAuthorization `json:"releaseFences,omitempty"`
}

// EngineGroupCapacityAllocationStatus is one complete physical allocation and its availability.
type EngineGroupCapacityAllocationStatus struct {
	// replicaID duplicates incarnation.replicaID as the list-map key.
	// +kubebuilder:validation:MinLength=1
	ReplicaID string `json:"replicaID"`

	// incarnation binds the allocation to stable and concrete identities.
	Incarnation EngineGroupControlIncarnationStatus `json:"incarnation"`

	// available reports whether the complete allocation satisfies profile-required readiness.
	Available bool `json:"available"`
}

// EngineGroupTrafficReconciliationStatus records desired, accepted, and observed routing levels.
type EngineGroupTrafficReconciliationStatus struct {
	// desired is the newest persisted absolute routing and drain target.
	// +optional
	Desired *EngineGroupTrafficTargetStatus `json:"desired,omitempty"`

	// accepted is the last exact target durably acknowledged by the traffic adapter.
	// +optional
	Accepted *EngineGroupTrafficTargetStatus `json:"accepted,omitempty"`

	// observed is the runtime's authoritative routing and drain state.
	Observed EngineGroupTrafficObservationStatus `json:"observed"`
}

// EngineGroupTrafficTargetStatus is one group-ordered absolute routing projection.
type EngineGroupTrafficTargetStatus struct {
	// controlRevision orders this target within the Engine Group.
	// +kubebuilder:validation:Minimum=1
	ControlRevision int64 `json:"controlRevision"`

	// transitionID correlates this target with one membership transition.
	// +kubebuilder:validation:MinLength=1
	TransitionID string `json:"transitionID"`

	// topologyGeneration binds the target to one authoritative membership snapshot.
	// +kubebuilder:validation:Minimum=1
	TopologyGeneration int64 `json:"topologyGeneration"`

	// admitted is the exact set of member incarnations eligible for new work.
	// +optional
	Admitted []EngineGroupMemberStatus `json:"admitted,omitempty"`

	// drain contains the exact terminal non-serving evidence requested for each member.
	// +optional
	Drain []EngineGroupTrafficDrainTargetStatus `json:"drain,omitempty"`
}

// EngineGroupTrafficDrainTargetStatus requests terminal non-serving evidence for one member.
type EngineGroupTrafficDrainTargetStatus struct {
	// membership is the exact member incarnation to withdraw and drain.
	Membership EngineGroupMemberStatus `json:"membership"`

	// mode distinguishes graceful drain from confirmation that a failed member is inactive.
	Mode EngineGroupTrafficDrainMode `json:"mode"`
}

// EngineGroupTrafficObservationStatus is the runtime's authoritative routing and drain state.
type EngineGroupTrafficObservationStatus struct {
	// appliedRevision is the last traffic target revision durably accepted by the adapter.
	// +kubebuilder:validation:Minimum=0
	AppliedRevision int64 `json:"appliedRevision"`

	// admitted is the exact set of member incarnations eligible for new work.
	// +optional
	Admitted []EngineGroupMemberStatus `json:"admitted,omitempty"`

	// draining contains member incarnations whose graceful drain has not completed.
	// +optional
	Draining []EngineGroupMemberStatus `json:"draining,omitempty"`

	// drained is durable terminal non-serving evidence for exact member incarnations.
	// +optional
	Drained []EngineGroupMemberStatus `json:"drained,omitempty"`
}

// EngineGroupMembershipReconciliationStatus records one desired transition and adapter observation.
type EngineGroupMembershipReconciliationStatus struct {
	// desired is one exact immutable membership compare-and-apply target.
	// +optional
	Desired *EngineGroupMembershipTargetStatus `json:"desired,omitempty"`

	// observed contains authoritative committed topology and the correlated transition result.
	Observed EngineGroupMembershipObservationStatus `json:"observed"`
}

// EngineGroupMembershipTargetStatus is one exact immutable membership transition target.
type EngineGroupMembershipTargetStatus struct {
	// controlRevision orders this target within the Engine Group.
	// +kubebuilder:validation:Minimum=1
	ControlRevision int64 `json:"controlRevision"`

	// transitionID uniquely identifies this target and its durable adapter transaction.
	// +kubebuilder:validation:MinLength=1
	TransitionID string `json:"transitionID"`

	// targetDigest is the shared-code canonical digest of this exact normalized target.
	// +kubebuilder:validation:MinLength=1
	TargetDigest string `json:"targetDigest"`

	// validation is the durable adapter evidence for this exact target.
	Validation EngineGroupValidationEvidenceStatus `json:"validation"`

	// baseTopology is the complete topology against which the target is atomically compared.
	BaseTopology EngineGroupTopologyStatus `json:"baseTopology"`

	// plan is the immutable profile-resolved semantic change.
	Plan EngineGroupResolvedPlanStatus `json:"plan"`

	// joining freezes exact runtime identities supplied to the membership transaction.
	// +optional
	// +listType=map
	// +listMapKey=replicaID
	Joining []EngineGroupJoiningReplicaStatus `json:"joining,omitempty"`
}

// EngineGroupJoiningReplicaStatus identifies one concrete process joining engine membership.
type EngineGroupJoiningReplicaStatus struct {
	// replicaID is the stable logical identity.
	// +kubebuilder:validation:MinLength=1
	ReplicaID string `json:"replicaID"`

	// runtimeIncarnation identifies the concrete engine process.
	// +kubebuilder:validation:MinLength=1
	RuntimeIncarnation string `json:"runtimeIncarnation"`
}

// EngineGroupMembershipObservationStatus separates current topology from one correlated transaction result.
type EngineGroupMembershipObservationStatus struct {
	// committedTopology is the engine's current authoritative complete topology.
	CommittedTopology EngineGroupTopologyStatus `json:"committedTopology"`

	// requestedTransitionID records the exact identity requested from the adapter observer.
	// +optional
	RequestedTransitionID string `json:"requestedTransitionID,omitempty"`

	// transition is the adapter's durable state for the requested transition; absence is authoritative.
	// +optional
	Transition *EngineGroupMembershipTransitionObservationStatus `json:"transition,omitempty"`
}

// EngineGroupMembershipTransitionObservationStatus is the durable result for one exact target.
type EngineGroupMembershipTransitionObservationStatus struct {
	// transitionID echoes the requested transition identity.
	// +kubebuilder:validation:MinLength=1
	TransitionID string `json:"transitionID"`

	// controlRevision echoes the requested controller revision.
	// +kubebuilder:validation:Minimum=1
	ControlRevision int64 `json:"controlRevision"`

	// targetDigest echoes the requested canonical target digest.
	// +kubebuilder:validation:MinLength=1
	TargetDigest string `json:"targetDigest"`

	// phase is the adapter-owned durable transaction state.
	Phase EngineGroupMembershipTransitionPhase `json:"phase"`

	// resultTopology is the immutable topology produced by a committed transition.
	// +optional
	ResultTopology *EngineGroupTopologyStatus `json:"resultTopology,omitempty"`

	// failure describes an authoritative rejection or unknown outcome.
	// +optional
	Failure *EngineGroupFailureStatus `json:"failure,omitempty"`
}

// EngineGroupTransitionStatus owns one immutable semantic transition across all subsystems.
type EngineGroupTransitionStatus struct {
	// spec is the immutable transition identity, base topology, and resolved plan.
	Spec EngineGroupTransitionSpecStatus `json:"spec"`

	// planPreflight is durable adapter evidence captured before capacity or traffic prework.
	// +optional
	PlanPreflight *EngineGroupPreflightStatus `json:"planPreflight,omitempty"`

	// targetPreflight is durable adapter evidence for the exact frozen membership target.
	// +optional
	TargetPreflight *EngineGroupPreflightStatus `json:"targetPreflight,omitempty"`

	// verification records the topology-bound serving proof or conclusive failure.
	// +optional
	Verification *EngineGroupVerificationStatus `json:"verification,omitempty"`

	// outcome summarizes cross-subsystem workflow progress without duplicating adapter phases.
	Outcome EngineGroupTransitionOutcome `json:"outcome"`

	// failure describes the reason an otherwise durable transition is blocked or reverting.
	// +optional
	Failure *EngineGroupFailureStatus `json:"failure,omitempty"`

	// startedAt is when the controller first persisted the immutable transition.
	StartedAt metav1.Time `json:"startedAt"`

	// updatedAt is when the controller last changed durable transition state.
	UpdatedAt metav1.Time `json:"updatedAt"`
}

// EngineGroupTransitionSpecStatus is the immutable input to one membership transition.
type EngineGroupTransitionSpecStatus struct {
	// id uniquely identifies this transition.
	// +kubebuilder:validation:MinLength=1
	ID string `json:"id"`

	// baseTopologyGeneration is the exact topology from which the change starts.
	// +kubebuilder:validation:Minimum=1
	BaseTopologyGeneration int64 `json:"baseTopologyGeneration"`

	// plan is the complete normalized semantic operation.
	Plan EngineGroupResolvedPlanStatus `json:"plan"`
}

// EngineGroupResolvedPlanStatus is one immutable, profile-resolved membership transition.
type EngineGroupResolvedPlanStatus struct {
	// id is the caller-selected stable plan identity.
	// +kubebuilder:validation:MinLength=1
	ID string `json:"id"`

	// profileFingerprint binds the plan to immutable engine and workload geometry.
	// +kubebuilder:validation:MinLength=1
	ProfileFingerprint string `json:"profileFingerprint"`

	// processLifecycleOwner identifies who starts and stops joining engine processes.
	ProcessLifecycleOwner EngineGroupProcessLifecycleOwner `json:"processLifecycleOwner"`

	// trafficRequirement declares whether retained members may serve during the mutation.
	TrafficRequirement EngineGroupTrafficRequirement `json:"trafficRequirement"`

	// verificationRequirement declares whether serving progress must be proven before admission.
	VerificationRequirement EngineGroupVerificationRequirement `json:"verificationRequirement"`

	// change is the serializable tagged union of exact membership semantics.
	Change EngineGroupResolvedChangeStatus `json:"change"`
}

// EngineGroupResolvedChangeStatus is the tagged union of membership changes understood by the coordinator.
// +kubebuilder:validation:XValidation:rule="(self.kind == 'Grow' && has(self.grow) && !has(self.retire) && !has(self.reduceToSurvivors) && !has(self.restore) && !has(self.remap)) || (self.kind == 'Retire' && !has(self.grow) && has(self.retire) && !has(self.reduceToSurvivors) && !has(self.restore) && !has(self.remap)) || (self.kind == 'ReduceToSurvivors' && !has(self.grow) && !has(self.retire) && has(self.reduceToSurvivors) && !has(self.restore) && !has(self.remap)) || (self.kind == 'Restore' && !has(self.grow) && !has(self.retire) && !has(self.reduceToSurvivors) && has(self.restore) && !has(self.remap)) || (self.kind == 'Remap' && !has(self.grow) && !has(self.retire) && !has(self.reduceToSurvivors) && !has(self.restore) && has(self.remap))",message="exactly one change variant must match kind"
type EngineGroupResolvedChangeStatus struct {
	// kind selects the populated semantic variant.
	Kind EngineGroupPlanKind `json:"kind"`

	// grow adds previously unknown logical replicas.
	// +optional
	Grow *EngineGroupGrowChangeStatus `json:"grow,omitempty"`

	// retire gracefully removes selected healthy replicas.
	// +optional
	Retire *EngineGroupRetireChangeStatus `json:"retire,omitempty"`

	// reduceToSurvivors removes base members absent from an authoritative survivor set.
	// +optional
	ReduceToSurvivors *EngineGroupReduceToSurvivorsChangeStatus `json:"reduceToSurvivors,omitempty"`

	// restore recreates excluded stable identities in their original native slots.
	// +optional
	Restore *EngineGroupRestoreChangeStatus `json:"restore,omitempty"`

	// remap changes complete native membership without changing logical cardinality.
	// +optional
	Remap *EngineGroupRemapChangeStatus `json:"remap,omitempty"`
}

// EngineGroupGrowChangeStatus adds named fresh logical replicas.
type EngineGroupGrowChangeStatus struct {
	// replicas contains every new logical, slot, bootstrap, and native identity.
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=replicaID
	Replicas []EngineGroupReplicaTargetStatus `json:"replicas"`
}

// EngineGroupRetireChangeStatus removes selected healthy logical replicas after drain.
type EngineGroupRetireChangeStatus struct {
	// replicas contains the exact stable logical identities selected for retirement.
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:items:MinLength=1
	// +listType=set
	Replicas []string `json:"replicas"`
}

// EngineGroupReduceToSurvivorsChangeStatus removes every base member absent from survivors.
type EngineGroupReduceToSurvivorsChangeStatus struct {
	// survivors is the exact authoritative survivor set.
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:items:MinLength=1
	// +listType=set
	Survivors []string `json:"survivors"`
}

// EngineGroupRestoreChangeStatus restores stable logical and native-member identities.
type EngineGroupRestoreChangeStatus struct {
	// replicas contains every exact stable identity to restore.
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=replicaID
	Replicas []EngineGroupReplicaTargetStatus `json:"replicas"`
}

// EngineGroupRemapChangeStatus replaces complete logical-to-native membership at one cardinality.
type EngineGroupRemapChangeStatus struct {
	// membership is the complete target logical, slot, and native-member mapping.
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=replicaID
	Membership []EngineGroupNativeMembershipStatus `json:"membership"`
}

// EngineGroupReplicaTargetStatus is the resolved physical and native identity for one joining replica.
type EngineGroupReplicaTargetStatus struct {
	// replicaID is the stable logical identity.
	// +kubebuilder:validation:MinLength=1
	ReplicaID string `json:"replicaID"`

	// slotID is the stable workload-manager position.
	// +kubebuilder:validation:MinLength=1
	SlotID string `json:"slotID"`

	// bootstrap distinguishes a new logical member from restoration of a fixed slot.
	Bootstrap EngineGroupBootstrapMode `json:"bootstrap"`

	// nativeMembers contains backend-native identities required by the resolved plan.
	// +optional
	// +kubebuilder:validation:items:MinLength=1
	NativeMembers []string `json:"nativeMembers,omitempty"`
}

// EngineGroupNativeMembershipStatus is one stable logical-to-native membership mapping.
type EngineGroupNativeMembershipStatus struct {
	// replicaID is the stable logical identity.
	// +kubebuilder:validation:MinLength=1
	ReplicaID string `json:"replicaID"`

	// slotID is the stable workload-manager position.
	// +kubebuilder:validation:MinLength=1
	SlotID string `json:"slotID"`

	// nativeMembers is the complete backend-native identity set for the replica.
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:items:MinLength=1
	NativeMembers []string `json:"nativeMembers"`
}

// EngineGroupPreflightStatus durably records one side-effect-free validation result.
type EngineGroupPreflightStatus struct {
	// transitionID correlates this validation with the immutable transition.
	// +kubebuilder:validation:MinLength=1
	TransitionID string `json:"transitionID"`

	// controlRevision is the controller revision at which validation completed.
	// Plan preflight runs before the first external target and therefore uses revision zero.
	// +kubebuilder:validation:Minimum=0
	ControlRevision int64 `json:"controlRevision"`

	// subjectDigest identifies the exact normalized plan or target that was validated.
	// +kubebuilder:validation:MinLength=1
	SubjectDigest string `json:"subjectDigest"`

	// evidence is authoritative approval for the validated subject.
	// +optional
	Evidence *EngineGroupValidationEvidenceStatus `json:"evidence,omitempty"`

	// rejection is an authoritative terminal rejection of the validated subject.
	// +optional
	Rejection *EngineGroupFailureStatus `json:"rejection,omitempty"`
}

// EngineGroupValidationEvidenceStatus binds approval to exact plan, target, profile, and capabilities.
type EngineGroupValidationEvidenceStatus struct {
	// planDigest is the canonical digest of the normalized resolved plan.
	// +kubebuilder:validation:MinLength=1
	PlanDigest string `json:"planDigest"`

	// targetDigest is the canonical digest of the exact membership target, when known.
	// +optional
	TargetDigest string `json:"targetDigest,omitempty"`

	// profileFingerprint binds validation to immutable engine and workload geometry.
	// +kubebuilder:validation:MinLength=1
	ProfileFingerprint string `json:"profileFingerprint"`

	// capabilityGeneration identifies the adapter capability snapshot used for validation.
	// +kubebuilder:validation:MinLength=1
	CapabilityGeneration string `json:"capabilityGeneration"`
}

// EngineGroupVerificationStatus records serving progress independently from membership commit.
type EngineGroupVerificationStatus struct {
	// phase is the durable serving-verification state.
	Phase EngineGroupVerificationPhase `json:"phase"`

	// proof is a positive result bound to one immutable topology.
	// +optional
	Proof *EngineGroupServingProofStatus `json:"proof,omitempty"`

	// failure is a conclusive verification failure.
	// +optional
	Failure *EngineGroupFailureStatus `json:"failure,omitempty"`
}

// EngineGroupServingProofStatus binds successful verification to an immutable topology.
type EngineGroupServingProofStatus struct {
	// topologyGeneration identifies the verified topology.
	// +kubebuilder:validation:Minimum=1
	TopologyGeneration int64 `json:"topologyGeneration"`

	// runtimeDigest identifies the runtime-observed membership and serving path.
	// +kubebuilder:validation:MinLength=1
	RuntimeDigest string `json:"runtimeDigest"`

	// observedAt records when serving progress was proven.
	ObservedAt metav1.Time `json:"observedAt"`
}

// EngineGroupFailureStatus is one structured subsystem or transition failure.
type EngineGroupFailureStatus struct {
	// classification states whether the same external intent may be retried.
	Classification EngineGroupFailureClassification `json:"classification"`

	// reason is a stable machine-readable failure reason.
	// +kubebuilder:validation:MinLength=1
	Reason string `json:"reason"`

	// message explains the failure for a human reader.
	// +optional
	Message string `json:"message,omitempty"`
}

// EngineGroupProcessLifecycleOwner identifies who starts and stops engine processes.
// +kubebuilder:validation:Enum=Engine;Orchestrator
type EngineGroupProcessLifecycleOwner string

const (
	// EngineGroupProcessLifecycleOwnerEngine means the inference engine owns process lifecycle.
	EngineGroupProcessLifecycleOwnerEngine EngineGroupProcessLifecycleOwner = "Engine"
	// EngineGroupProcessLifecycleOwnerOrchestrator means Kubernetes-side orchestration owns process lifecycle.
	EngineGroupProcessLifecycleOwnerOrchestrator EngineGroupProcessLifecycleOwner = "Orchestrator"
)

// EngineGroupBootstrapMode describes how one new incarnation joins the engine world.
// +kubebuilder:validation:Enum=Join;RestoreFixedSlot
type EngineGroupBootstrapMode string

const (
	// EngineGroupBootstrapModeJoin creates a previously unknown logical member.
	EngineGroupBootstrapModeJoin EngineGroupBootstrapMode = "Join"
	// EngineGroupBootstrapModeRestoreFixedSlot restores a known logical and native-member identity.
	EngineGroupBootstrapModeRestoreFixedSlot EngineGroupBootstrapMode = "RestoreFixedSlot"
)

// EngineGroupTrafficRequirement describes the traffic boundary around membership mutation.
// +kubebuilder:validation:Enum=KeepServing;QuiesceGroup
type EngineGroupTrafficRequirement string

const (
	// EngineGroupTrafficRequirementKeepServing permits retained replicas to continue serving.
	EngineGroupTrafficRequirementKeepServing EngineGroupTrafficRequirement = "KeepServing"
	// EngineGroupTrafficRequirementQuiesceGroup drains the complete base topology before mutation.
	EngineGroupTrafficRequirementQuiesceGroup EngineGroupTrafficRequirement = "QuiesceGroup"
)

// EngineGroupVerificationRequirement states whether commit needs a serving-progress proof.
// +kubebuilder:validation:Enum=None;Required
type EngineGroupVerificationRequirement string

const (
	// EngineGroupVerificationRequirementNone permits admission after authoritative membership commit.
	EngineGroupVerificationRequirementNone EngineGroupVerificationRequirement = "None"
	// EngineGroupVerificationRequirementRequired requires a matching serving proof before admission.
	EngineGroupVerificationRequirementRequired EngineGroupVerificationRequirement = "Required"
)

// EngineGroupPlanKind identifies exact membership-change semantics.
// +kubebuilder:validation:Enum=Grow;Retire;ReduceToSurvivors;Restore;Remap
type EngineGroupPlanKind string

const (
	// EngineGroupPlanKindGrow adds previously unknown logical replicas.
	EngineGroupPlanKindGrow EngineGroupPlanKind = "Grow"
	// EngineGroupPlanKindRetire gracefully removes selected healthy replicas.
	EngineGroupPlanKindRetire EngineGroupPlanKind = "Retire"
	// EngineGroupPlanKindReduceToSurvivors removes identities absent from an authoritative survivor set.
	EngineGroupPlanKindReduceToSurvivors EngineGroupPlanKind = "ReduceToSurvivors"
	// EngineGroupPlanKindRestore recreates excluded stable identities.
	EngineGroupPlanKindRestore EngineGroupPlanKind = "Restore"
	// EngineGroupPlanKindRemap changes native membership without changing logical cardinality.
	EngineGroupPlanKindRemap EngineGroupPlanKind = "Remap"
)

// EngineGroupTrafficDrainMode distinguishes planned drain from failed-member withdrawal evidence.
// +kubebuilder:validation:Enum=Graceful;ConfirmInactive
type EngineGroupTrafficDrainMode string

const (
	// EngineGroupTrafficDrainModeGraceful waits for a reachable member's in-flight work to finish.
	EngineGroupTrafficDrainModeGraceful EngineGroupTrafficDrainMode = "Graceful"
	// EngineGroupTrafficDrainModeConfirmInactive proves a failed member is no longer routable.
	EngineGroupTrafficDrainModeConfirmInactive EngineGroupTrafficDrainMode = "ConfirmInactive"
)

// EngineGroupMembershipTransitionPhase is adapter-owned durable state for one exact target.
// +kubebuilder:validation:Enum=Pending;Committed;Rejected;Unknown
type EngineGroupMembershipTransitionPhase string

const (
	// EngineGroupMembershipTransitionPhasePending means the adapter accepted and is applying the target.
	EngineGroupMembershipTransitionPhasePending EngineGroupMembershipTransitionPhase = "Pending"
	// EngineGroupMembershipTransitionPhaseCommitted means the target produced its result topology.
	EngineGroupMembershipTransitionPhaseCommitted EngineGroupMembershipTransitionPhase = "Committed"
	// EngineGroupMembershipTransitionPhaseRejected means the target definitively cannot mutate membership.
	EngineGroupMembershipTransitionPhaseRejected EngineGroupMembershipTransitionPhase = "Rejected"
	// EngineGroupMembershipTransitionPhaseUnknown means the adapter cannot establish the target's outcome.
	EngineGroupMembershipTransitionPhaseUnknown EngineGroupMembershipTransitionPhase = "Unknown"
)

// EngineGroupFailureClassification states whether reconciliation may retry the same external intent.
// +kubebuilder:validation:Enum=Retryable;Terminal
type EngineGroupFailureClassification string

const (
	// EngineGroupFailureClassificationRetryable permits retrying the same intent.
	EngineGroupFailureClassificationRetryable EngineGroupFailureClassification = "Retryable"
	// EngineGroupFailureClassificationTerminal means the same intent cannot safely make progress.
	EngineGroupFailureClassificationTerminal EngineGroupFailureClassification = "Terminal"
)

// EngineGroupVerificationPhase is the independent serving-verification state.
// +kubebuilder:validation:Enum=Pending;Passed;Failed
type EngineGroupVerificationPhase string

const (
	// EngineGroupVerificationPhasePending means a proof is required for the committed topology.
	EngineGroupVerificationPhasePending EngineGroupVerificationPhase = "Pending"
	// EngineGroupVerificationPhasePassed means the exact topology produced serving progress.
	EngineGroupVerificationPhasePassed EngineGroupVerificationPhase = "Passed"
	// EngineGroupVerificationPhaseFailed means a conclusive serving check failed.
	EngineGroupVerificationPhaseFailed EngineGroupVerificationPhase = "Failed"
)

// EngineGroupTransitionOutcome summarizes the cross-subsystem workflow.
// +kubebuilder:validation:Enum=Progressing;Reverting;RolledBack;Blocked;Completed
type EngineGroupTransitionOutcome string

const (
	// EngineGroupTransitionOutcomeProgressing means safe work remains.
	EngineGroupTransitionOutcomeProgressing EngineGroupTransitionOutcome = "Progressing"
	// EngineGroupTransitionOutcomeReverting means a provably uncommitted target is restoring base state.
	EngineGroupTransitionOutcomeReverting EngineGroupTransitionOutcome = "Reverting"
	// EngineGroupTransitionOutcomeRolledBack means base capacity and traffic were restored.
	EngineGroupTransitionOutcomeRolledBack EngineGroupTransitionOutcome = "RolledBack"
	// EngineGroupTransitionOutcomeBlocked means safety evidence cannot progress automatically.
	EngineGroupTransitionOutcomeBlocked EngineGroupTransitionOutcome = "Blocked"
	// EngineGroupTransitionOutcomeCompleted means all subsystems reached the resolved plan.
	EngineGroupTransitionOutcomeCompleted EngineGroupTransitionOutcome = "Completed"
)
