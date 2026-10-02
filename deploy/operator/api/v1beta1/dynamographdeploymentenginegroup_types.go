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

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// DynamoGraphDeploymentEngineGroupSpec defines the desired capacity of one independently
// resizable distributed engine world.
// +kubebuilder:validation:XValidation:rule="!has(self.policy) || !has(self.policy.minReplicas) || self.replicas >= self.policy.minReplicas",message="replicas must be greater than or equal to policy.minReplicas"
// +kubebuilder:validation:XValidation:rule="!has(self.policy) || !has(self.policy.maxReplicas) || self.replicas <= self.policy.maxReplicas",message="replicas must be less than or equal to policy.maxReplicas"
type DynamoGraphDeploymentEngineGroupSpec struct {
	// replicas is the absolute desired number of independently scalable replica allocations.
	// The profile maps each allocation to whole Pods and one or more native engine members.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=1
	Replicas int32 `json:"replicas"`

	// policy constrains user- or autoscaler-selected replica targets independently from hard
	// engine capability bounds reported in status.profile.
	// +optional
	Policy *EngineGroupScalingPolicy `json:"policy,omitempty"`
}

// EngineGroupScalingPolicy defines operator-selected scaling bounds for one Engine Group.
// +kubebuilder:validation:XValidation:rule="!has(self.minReplicas) || !has(self.maxReplicas) || self.minReplicas <= self.maxReplicas",message="minReplicas must be less than or equal to maxReplicas"
type EngineGroupScalingPolicy struct {
	// minReplicas is the minimum ordinary scaling target. Terminal group retirement is driven by
	// deletion and may retire membership below this bound without writing an out-of-policy target.
	// +optional
	// +kubebuilder:validation:Minimum=1
	MinReplicas *int32 `json:"minReplicas,omitempty"`

	// maxReplicas is the maximum target selected by policy. It cannot exceed the resolved hard
	// engine capability bound.
	// +optional
	// +kubebuilder:validation:Minimum=1
	MaxReplicas *int32 `json:"maxReplicas,omitempty"`
}

// DynamoGraphDeploymentEngineGroupStatus defines the observed state of one Engine Group.
type DynamoGraphDeploymentEngineGroupStatus struct {
	// observedGeneration is the most recent object generation observed by the controller.
	// +optional
	// +kubebuilder:validation:Minimum=0
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// replicas is the number of logical replicas with complete physical allocations. It is the
	// current replica count exposed through the scale subresource.
	// +kubebuilder:validation:Minimum=0
	Replicas int32 `json:"replicas"`

	// availableReplicas counts allocations whose complete steady-state member set is active
	// and admitted with usable backing capacity. Partially serving allocations do not count.
	// +optional
	// +kubebuilder:validation:Minimum=0
	AvailableReplicas int32 `json:"availableReplicas,omitempty"`

	// desiredNativeMembers is the canonical durable assignment of exact identities to the desired
	// replica slots, not a projection of observed membership. Survivor recovery does not rewrite it.
	// +optional
	// +listType=set
	// +kubebuilder:validation:items:MinLength=1
	DesiredNativeMembers []string `json:"desiredNativeMembers,omitempty"`

	// desiredAssignmentGeneration correlates the resolved assignment with a spec generation.
	// A superseded or unresolved target cannot be reported as reached from an older assignment.
	// +optional
	// +kubebuilder:validation:Minimum=0
	DesiredAssignmentGeneration int64 `json:"desiredAssignmentGeneration,omitempty"`

	// desiredNativeMemberCount is the cardinality of desiredNativeMembers.
	// +optional
	// +kubebuilder:validation:Minimum=0
	DesiredNativeMemberCount int32 `json:"desiredNativeMemberCount,omitempty"`

	// activeNativeMemberCount counts exact members in the authoritative committed topology.
	// A partially serving allocation contributes only its active members, not one active replica.
	// +optional
	// +kubebuilder:validation:Minimum=0
	ActiveNativeMemberCount int32 `json:"activeNativeMemberCount,omitempty"`

	// selector matches exactly one representative Pod for every allocated logical replica.
	// It represents allocation, not availability or engine admission.
	// +optional
	Selector string `json:"selector,omitempty"`

	// scaleUnit names the logical unit counted by spec.replicas and status.replicas.
	// +optional
	ScaleUnit EngineGroupScaleUnit `json:"scaleUnit,omitempty"`

	// profile is the immutable resolved mapping between logical replicas and physical capacity.
	// +optional
	Profile *EngineGroupProfileStatus `json:"profile,omitempty"`

	// topology is the engine's current authoritative committed topology.
	// +optional
	Topology *EngineGroupTopologyStatus `json:"topology,omitempty"`

	// lastStableReplicas is the most recent allocation count that reached its desired target
	// without degradation.
	// +optional
	// +kubebuilder:validation:Minimum=0
	LastStableReplicas int32 `json:"lastStableReplicas,omitempty"`

	// lastStableTopologyGeneration identifies the exact last fully restored serving membership.
	// It distinguishes unexpected member loss from a planned change in allocation target.
	// +optional
	// +kubebuilder:validation:Minimum=1
	LastStableTopologyGeneration int64 `json:"lastStableTopologyGeneration,omitempty"`

	// replicaStates contains the stable identity and independently observed physical and engine
	// state of each known logical replica.
	// +optional
	// +listType=map
	// +listMapKey=replicaID
	ReplicaStates []EngineGroupReplicaStatus `json:"replicaStates,omitempty"`

	// traffic is the runtime's authoritative routing and drain observation.
	// +optional
	Traffic *EngineGroupTrafficStatus `json:"traffic,omitempty"`

	// releaseAuthorizations names the exact Pod UIDs that may be removed from stable replica slots.
	// +optional
	ReleaseAuthorizations []EngineGroupReleaseAuthorization `json:"releaseAuthorizations,omitempty"`

	// targetValidation describes a desired replica target rejected from reconciliation-time
	// profile or capability information.
	// +optional
	TargetValidation *EngineGroupTargetValidationStatus `json:"targetValidation,omitempty"`

	// conditions contains the latest observations of group availability, progress, degradation,
	// target convergence, target validity, and topology authority.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// EngineGroupProfileStatus records immutable geometry and hard capability bounds resolved for a group.
// +kubebuilder:validation:XValidation:rule="self.minSupportedReplicas <= self.maxSupportedReplicas",message="minSupportedReplicas must be less than or equal to maxSupportedReplicas"
type EngineGroupProfileStatus struct {
	// backend is the inference engine that owns native membership.
	// +kubebuilder:validation:Enum=sglang;vllm;trtllm
	Backend string `json:"backend"`

	// fingerprint identifies the immutable engine and workload geometry used by this group.
	// +kubebuilder:validation:MinLength=1
	Fingerprint string `json:"fingerprint"`

	// gpusPerReplica is the accelerator requirement of one logical replica.
	// +kubebuilder:validation:Minimum=1
	GPUsPerReplica int64 `json:"gpusPerReplica"`

	// podsPerReplica is the number of physically disjoint capacity Pods allocated and released
	// together for one logical replica.
	// +kubebuilder:validation:Minimum=1
	PodsPerReplica int32 `json:"podsPerReplica"`

	// nativeMembersPerReplica is the fixed steady-state member count per allocation.
	// Partial survival and advertised surge may temporarily change the active count.
	// +kubebuilder:validation:Minimum=1
	NativeMembersPerReplica int32 `json:"nativeMembersPerReplica"`

	// minSafeServingNativeMembers is the lowest committed native-member count at which this profile may
	// continue serving while degraded or recovering.
	// +kubebuilder:validation:Minimum=1
	MinSafeServingNativeMembers int32 `json:"minSafeServingNativeMembers"`

	// minSupportedReplicas is the hard lower bound for live membership operations other than terminal
	// group retirement.
	// +kubebuilder:validation:Minimum=1
	MinSupportedReplicas int32 `json:"minSupportedReplicas"`

	// maxSupportedReplicas is the hard upper bound for live membership operations.
	// +kubebuilder:validation:Minimum=1
	MaxSupportedReplicas int32 `json:"maxSupportedReplicas"`
}

// EngineGroupTopologyStatus is one immutable engine-authoritative committed topology snapshot.
type EngineGroupTopologyStatus struct {
	// generation is the engine's topology generation.
	// +kubebuilder:validation:Minimum=1
	Generation int64 `json:"generation"`

	// replicas is the complete logical-to-native membership mapping at this generation.
	// +optional
	// +listType=map
	// +listMapKey=replicaID
	Replicas []EngineGroupMemberStatus `json:"replicas,omitempty"`
}

// EngineGroupMemberStatus is one engine-owned logical, runtime, and native-member mapping.
type EngineGroupMemberStatus struct {
	// replicaID is the stable logical identity.
	// +kubebuilder:validation:MinLength=1
	ReplicaID string `json:"replicaID"`

	// runtimeIncarnation identifies the concrete engine process in this topology.
	// +kubebuilder:validation:MinLength=1
	RuntimeIncarnation string `json:"runtimeIncarnation"`

	// nativeMembers are the backend-specific ranks or member identities committed for the replica.
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:items:MinLength=1
	NativeMembers []string `json:"nativeMembers"`
}

// EngineGroupReplicaStatus preserves one stable logical identity across physical replacement.
type EngineGroupReplicaStatus struct {
	// replicaID is stable across physical and runtime replacement.
	// +kubebuilder:validation:MinLength=1
	ReplicaID string `json:"replicaID"`

	// slotID identifies the stable workload-manager position backing this replica.
	// +kubebuilder:validation:MinLength=1
	SlotID string `json:"slotID"`

	// representativeRef identifies the Pod selected by status.selector for this allocation.
	// +optional
	RepresentativeRef *EngineGroupCapacityRef `json:"representativeRef,omitempty"`

	// currentAllocation is the allocation currently backing this stable replica slot.
	// +optional
	CurrentAllocation *EngineGroupReplicaAllocationStatus `json:"currentAllocation,omitempty"`

	// candidateAllocation is the sole replacement being prepared for this replica.
	// A dormant candidate has no native identities until reuse is safe. A backend
	// advertising surge may temporarily assign it distinct members before promotion.
	// +optional
	CandidateAllocation *EngineGroupReplicaAllocationStatus `json:"candidateAllocation,omitempty"`

	// nativeMembers reports membership and traffic independently for every correlated member.
	// +optional
	// +listType=map
	// +listMapKey=id
	NativeMembers []EngineGroupNativeMemberStatus `json:"nativeMembers,omitempty"`
}

// EngineGroupReplicaAllocationStatus binds concrete capacity to independently observed health.
type EngineGroupReplicaAllocationStatus struct {
	// runtimeIncarnation identifies one concrete engine process incarnation.
	// A dormant candidate may not yet have a runtime incarnation.
	// +optional
	// +kubebuilder:validation:MinLength=1
	RuntimeIncarnation string `json:"runtimeIncarnation,omitempty"`

	// capacityRefs contains every concrete Pod incarnation in this replica allocation.
	// +kubebuilder:validation:MinItems=1
	CapacityRefs []EngineGroupCapacityRef `json:"capacityRefs"`

	// availability reports usable backing capacity for the members this allocation still serves.
	// Pod Ready is only an input; a degraded allocation may keep serving surviving members.
	Availability EngineGroupReplicaAvailability `json:"availability"`

	// health records physical and runtime health independently of committed membership.
	Health EngineGroupAllocationHealth `json:"health"`
}

// EngineGroupNativeMemberStatus records one engine-authoritative native member and its traffic evidence.
type EngineGroupNativeMemberStatus struct {
	// id is a stable backend-native member identity.
	// +kubebuilder:validation:MinLength=1
	ID string `json:"id"`

	// membership distinguishes committed participation from masking or orchestration intent.
	Membership EngineGroupReplicaMembership `json:"membership"`

	// traffic is admission or terminal drain evidence from the runtime traffic authority.
	Traffic EngineGroupMemberTraffic `json:"traffic"`
}

// EngineGroupCapacityRef identifies one concrete Pod allocated to a logical replica.
// +kubebuilder:validation:XValidation:rule="size(self.uid) > 0",message="uid must not be empty"
type EngineGroupCapacityRef struct {
	// name is the Pod name.
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// uid is the concrete Pod incarnation and prevents name reuse from inheriting authority.
	UID types.UID `json:"uid"`
}

// EngineGroupTrafficStatus is the runtime's exact routing and terminal drain state.
type EngineGroupTrafficStatus struct {
	// operationID identifies the operation for which this observation was produced.
	// +optional
	OperationID string `json:"operationID,omitempty"`

	// topologyGeneration binds the observation to an exact committed topology.
	// +kubebuilder:validation:Minimum=1
	TopologyGeneration int64 `json:"topologyGeneration"`

	// admitted is the exact set of member incarnations eligible for new work.
	// +optional
	Admitted []EngineGroupMemberStatus `json:"admitted,omitempty"`

	// draining is the exact set of member incarnations whose in-flight work has not completed.
	// +optional
	Draining []EngineGroupMemberStatus `json:"draining,omitempty"`

	// drained is durable terminal non-serving evidence for exact member incarnations.
	// +optional
	Drained []EngineGroupMemberStatus `json:"drained,omitempty"`
}

// EngineGroupReleaseAuthorization permits removal of only named concrete Pod incarnations.
type EngineGroupReleaseAuthorization struct {
	// operationID identifies the membership result authorizing release.
	// +kubebuilder:validation:MinLength=1
	OperationID string `json:"operationID"`

	// topologyGeneration identifies the exact committed topology authorizing release.
	// +kubebuilder:validation:Minimum=1
	TopologyGeneration int64 `json:"topologyGeneration"`

	// replicaID is the stable logical replica being released.
	// +kubebuilder:validation:MinLength=1
	ReplicaID string `json:"replicaID"`

	// slotID is the stable workload-manager position being fenced.
	// +kubebuilder:validation:MinLength=1
	SlotID string `json:"slotID"`

	// capacityRefs is the complete set of concrete Pod UIDs authorized for deletion.
	// +kubebuilder:validation:MinItems=1
	CapacityRefs []EngineGroupCapacityRef `json:"capacityRefs"`
}

// EngineGroupTargetValidationStatus records a reconcile-time target rejection without silently
// changing the requested target.
type EngineGroupTargetValidationStatus struct {
	// requestedReplicas is the rejected desired target.
	// +kubebuilder:validation:Minimum=0
	RequestedReplicas int32 `json:"requestedReplicas"`

	// effectiveReplicas is the last valid target still in force.
	// +kubebuilder:validation:Minimum=0
	EffectiveReplicas int32 `json:"effectiveReplicas"`

	// minReplicas is the effective lower bound used for validation.
	// +optional
	// +kubebuilder:validation:Minimum=1
	MinReplicas int32 `json:"minReplicas,omitempty"`

	// maxReplicas is the effective upper bound used for validation.
	// +optional
	// +kubebuilder:validation:Minimum=1
	MaxReplicas int32 `json:"maxReplicas,omitempty"`

	// reason is a stable machine-readable rejection reason.
	// +kubebuilder:validation:MinLength=1
	Reason string `json:"reason"`

	// message explains the rejection for a human reader.
	// +optional
	Message string `json:"message,omitempty"`
}

// EngineGroupReplicaAvailability reports physical and runtime availability independently from membership.
// +kubebuilder:validation:Enum=Available;Unavailable;Unknown
type EngineGroupReplicaAvailability string

// EngineGroupAllocationHealth records allocation health without collapsing it into membership.
// +kubebuilder:validation:Enum=Healthy;Degraded;Failed;Unknown
type EngineGroupAllocationHealth string

const (
	EngineGroupAllocationHealthHealthy  EngineGroupAllocationHealth = "Healthy"
	EngineGroupAllocationHealthDegraded EngineGroupAllocationHealth = "Degraded"
	EngineGroupAllocationHealthFailed   EngineGroupAllocationHealth = "Failed"
	EngineGroupAllocationHealthUnknown  EngineGroupAllocationHealth = "Unknown"
)

// EngineGroupMemberTraffic records per-member traffic evidence, independently of Pod readiness.
// +kubebuilder:validation:Enum=Admitted;Draining;Drained;Withdrawn;Unknown
type EngineGroupMemberTraffic string

const (
	EngineGroupMemberTrafficAdmitted  EngineGroupMemberTraffic = "Admitted"
	EngineGroupMemberTrafficDraining  EngineGroupMemberTraffic = "Draining"
	EngineGroupMemberTrafficDrained   EngineGroupMemberTraffic = "Drained"
	EngineGroupMemberTrafficWithdrawn EngineGroupMemberTraffic = "Withdrawn"
	EngineGroupMemberTrafficUnknown   EngineGroupMemberTraffic = "Unknown"
)

const (
	// EngineGroupReplicaAvailabilityAvailable means capacity is usable for the members it still backs.
	EngineGroupReplicaAvailabilityAvailable EngineGroupReplicaAvailability = "Available"
	// EngineGroupReplicaAvailabilityUnavailable means at least one required capacity or runtime check failed.
	EngineGroupReplicaAvailabilityUnavailable EngineGroupReplicaAvailability = "Unavailable"
	// EngineGroupReplicaAvailabilityUnknown means availability cannot currently be established.
	EngineGroupReplicaAvailabilityUnknown EngineGroupReplicaAvailability = "Unknown"
)

// EngineGroupReplicaMembership reports committed engine state or current orchestration intent.
// +kubebuilder:validation:Enum=Active;Masked;Joining;Retiring;Unknown
type EngineGroupReplicaMembership string

const (
	// EngineGroupReplicaMembershipActive means the engine has committed this native member.
	EngineGroupReplicaMembershipActive EngineGroupReplicaMembership = "Active"
	// EngineGroupReplicaMembershipMasked means the engine committed a topology excluding this member.
	EngineGroupReplicaMembershipMasked EngineGroupReplicaMembership = "Masked"
	// EngineGroupReplicaMembershipJoining means orchestration intends this replica to join.
	EngineGroupReplicaMembershipJoining EngineGroupReplicaMembership = "Joining"
	// EngineGroupReplicaMembershipRetiring means orchestration intends this replica to leave.
	EngineGroupReplicaMembershipRetiring EngineGroupReplicaMembership = "Retiring"
	// EngineGroupReplicaMembershipUnknown means authoritative engine membership is unavailable.
	EngineGroupReplicaMembershipUnknown EngineGroupReplicaMembership = "Unknown"
)

// EngineGroupScaleUnit names the logical unit exposed through the Scale subresource.
// +kubebuilder:validation:Enum=replicas
type EngineGroupScaleUnit string

const (
	// EngineGroupScaleUnitReplicas means Scale counts logical engine replicas within one world.
	EngineGroupScaleUnitReplicas EngineGroupScaleUnit = "replicas"
)

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.replicas,selectorpath=.status.selector
// +kubebuilder:printcolumn:name="DESIRED",type="integer",JSONPath=".spec.replicas",description="Desired logical replicas"
// +kubebuilder:printcolumn:name="ALLOCATED",type="integer",JSONPath=".status.replicas",description="Logically complete physical allocations"
// +kubebuilder:printcolumn:name="AVAILABLE",type="integer",JSONPath=".status.availableReplicas",description="Available logical replicas"
// +kubebuilder:printcolumn:name="ACTIVE MEMBERS",type="integer",JSONPath=".status.activeNativeMemberCount",description="Engine-committed native members"
// +kubebuilder:printcolumn:name="UNITS",type="string",JSONPath=".status.scaleUnit",description="Unit counted by desired and observed replica columns"
// +kubebuilder:printcolumn:name="PODS/REPLICA",type="integer",JSONPath=".status.profile.podsPerReplica",description="Physical Pods allocated per logical replica"
// +kubebuilder:printcolumn:name="AGE",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:resource:shortName={dgdeg}

// DynamoGraphDeploymentEngineGroup represents one independently resizable distributed engine world.
// The scale subresource counts logical replicas, while status keeps physical allocation, engine
// membership, and traffic admission separately observable.
type DynamoGraphDeploymentEngineGroup struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoGraphDeploymentEngineGroupSpec   `json:"spec,omitempty"`
	Status DynamoGraphDeploymentEngineGroupStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoGraphDeploymentEngineGroupList contains a list of DynamoGraphDeploymentEngineGroup.
type DynamoGraphDeploymentEngineGroupList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeploymentEngineGroup `json:"items"`
}
