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
	// replicas is the absolute desired number of logical engine replicas in this group.
	// For Elastic EP, one replica maps to one data-parallel replica through the resolved profile.
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

	// availableReplicas is the number of allocated replicas whose complete physical allocation
	// and profile-required runtime checks are available.
	// +optional
	// +kubebuilder:validation:Minimum=0
	AvailableReplicas int32 `json:"availableReplicas,omitempty"`

	// activeReplicas is the number of replicas in the engine's authoritative committed topology.
	// +optional
	// +kubebuilder:validation:Minimum=0
	ActiveReplicas int32 `json:"activeReplicas,omitempty"`

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

	// lastStableReplicas is the most recent membership count that reached its desired target
	// without degradation.
	// +optional
	// +kubebuilder:validation:Minimum=0
	LastStableReplicas int32 `json:"lastStableReplicas,omitempty"`

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

	// reconciliation is the controller's durable desired, accepted, and observed journal.
	// It is persisted before external effects so the same transition can resume after restart.
	// +optional
	Reconciliation *EngineGroupReconciliationStatus `json:"reconciliation,omitempty"`

	// conditions contains the latest observations of group availability, progress, degradation,
	// target convergence, target validity, and topology authority.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// EngineGroupProfileStatus records immutable geometry and hard capability bounds resolved for a group.
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

	// minSafeServingReplicas is the lowest committed replica count at which this profile may
	// continue serving while degraded or recovering.
	// +kubebuilder:validation:Minimum=1
	MinSafeServingReplicas int32 `json:"minSafeServingReplicas"`

	// minReplicas is the hard lower bound for live membership operations other than terminal
	// group retirement.
	// +kubebuilder:validation:Minimum=1
	MinReplicas int32 `json:"minReplicas"`

	// maxReplicas is the hard upper bound for live membership operations.
	// +kubebuilder:validation:Minimum=1
	MaxReplicas int32 `json:"maxReplicas"`
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

	// current is the currently allocated physical and runtime incarnation.
	// +optional
	Current *EngineGroupReplicaIncarnation `json:"current,omitempty"`

	// previousIncarnations retain identities required for recovery or exact release.
	// +optional
	PreviousIncarnations []EngineGroupReplicaIncarnation `json:"previousIncarnations,omitempty"`

	// nativeMembers are the engine identities currently correlated with this logical replica.
	// +optional
	// +kubebuilder:validation:items:MinLength=1
	NativeMembers []string `json:"nativeMembers,omitempty"`

	// availability reports complete physical and profile-required runtime readiness.
	Availability EngineGroupReplicaAvailability `json:"availability"`

	// membership reports committed engine state or current orchestration intent.
	Membership EngineGroupReplicaMembership `json:"membership"`
}

// EngineGroupReplicaIncarnation binds one logical replica and stable slot to concrete capacity.
type EngineGroupReplicaIncarnation struct {
	// runtimeIncarnation identifies one concrete engine process incarnation.
	// +kubebuilder:validation:MinLength=1
	RuntimeIncarnation string `json:"runtimeIncarnation"`

	// capacityRefs contains every concrete Pod incarnation in this replica allocation.
	// +kubebuilder:validation:MinItems=1
	CapacityRefs []EngineGroupCapacityRef `json:"capacityRefs"`
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

const (
	// EngineGroupReplicaAvailabilityAvailable means every capacity Pod and required runtime check is ready.
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
	// EngineGroupReplicaMembershipActive means the engine has committed this replica.
	EngineGroupReplicaMembershipActive EngineGroupReplicaMembership = "Active"
	// EngineGroupReplicaMembershipMasked means the engine committed a survivor topology excluding this replica.
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
// +kubebuilder:printcolumn:name="ACTIVE",type="integer",JSONPath=".status.activeReplicas",description="Engine-committed logical replicas"
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
