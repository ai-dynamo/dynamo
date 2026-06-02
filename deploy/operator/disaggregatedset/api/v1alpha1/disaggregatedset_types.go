/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	leaderworkerset "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// DisaggregatedRoleSpec defines the configuration for a disaggregated role.
// This structure embeds LeaderWorkerSetSpec from sigs.k8s.io/lws, with validation
// to reject unsupported fields (RolloutStrategy.Type must be RollingUpdate,
// RolloutStrategy.RollingUpdateConfiguration.Partition must not be set).
type DisaggregatedRoleSpec struct {
	// Name is the unique identifier for this role.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Pattern=`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`
	// +required
	Name string `json:"name"`

	// LeaderWorkerSetSpec is embedded inline to inherit all LWS configuration fields.
	// Note: RolloutStrategy.Type must be RollingUpdate (or empty) and
	// RolloutStrategy.RollingUpdateConfiguration.Partition must not be set.
	// DisaggregatedSet handles rollouts across roles and does not propagate
	// RolloutStrategy to the underlying LWS resources.
	leaderworkerset.LeaderWorkerSetSpec `json:",inline"`

	// Metadata allows setting labels and annotations on the LWS CR's ObjectMeta.
	// This is useful for integrations like Kueue (queue assignment via
	// kueue.x-k8s.io/queue-name label) and LWS exclusive-topology scheduling
	// (leaderworkerset.sigs.k8s.io/exclusive-topology label).
	// Only labels and annotations are used; other ObjectMeta fields are ignored.
	// System labels (disaggregatedset.x-k8s.io/*, app) take precedence over user labels.
	// +optional
	Metadata *metav1.ObjectMeta `json:"metadata,omitempty"`
}

// DisaggregatedSetSpec defines the desired state of DisaggregatedSet
// +kubebuilder:validation:XValidation:rule="self.roles.all(r, self.roles.filter(s, s.name == r.name).size() == 1)",message="role names must be unique"
// +kubebuilder:validation:XValidation:rule="self.roles.all(r, r.replicas == 0) || self.roles.all(r, r.replicas > 0)",message="replicas must be zero for all roles or non-zero for all roles"
type DisaggregatedSetSpec struct {
	// Roles defines the list of roles (at least 2 required).
	// Each role has a unique name and its own configuration.
	// +kubebuilder:validation:MinItems=2
	// +kubebuilder:validation:MaxItems=10
	// +required
	Roles []DisaggregatedRoleSpec `json:"roles"`
}

// RoleStatus defines the observed state of a single role.
type RoleStatus struct {
	// Name is the name of the role (matches spec.roles[].name).
	// +required
	Name string `json:"name"`

	// Replicas is the total number of replicas for this role.
	// +optional
	Replicas int32 `json:"replicas,omitempty"`

	// ReadyReplicas is the number of ready replicas for this role.
	// +optional
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	// UpdatedReplicas is the number of replicas updated to the latest revision.
	// +optional
	UpdatedReplicas int32 `json:"updatedReplicas,omitempty"`
}

// DisaggregatedSetStatus defines the observed state of DisaggregatedSet.
type DisaggregatedSetStatus struct {
	// For Kubernetes API conventions, see:
	// https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#typical-status-properties

	// RoleStatuses contains the status for each role.
	// The order matches spec.roles.
	// +listType=map
	// +listMapKey=name
	// +optional
	RoleStatuses []RoleStatus `json:"roleStatuses,omitempty"`

	// conditions represent the current state of the DisaggregatedSet resource.
	// Each condition has a unique type and reflects the status of a specific aspect of the resource.
	//
	// Standard condition types include:
	// - "Available": the resource is fully functional
	// - "Progressing": the resource is being created or updated
	// - "Degraded": the resource failed to reach or maintain its desired state
	//
	// The status of each condition is one of True, False, or Unknown.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// DisaggregatedSet is the Schema for the disaggregatedsets API
type DisaggregatedSet struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is a standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitzero"`

	// spec defines the desired state of DisaggregatedSet
	// +required
	Spec DisaggregatedSetSpec `json:"spec"`

	// status defines the observed state of DisaggregatedSet
	// +optional
	Status DisaggregatedSetStatus `json:"status,omitzero"`
}

// +kubebuilder:object:root=true

// DisaggregatedSetList contains a list of DisaggregatedSet
type DisaggregatedSetList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitzero"`
	Items           []DisaggregatedSet `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DisaggregatedSet{}, &DisaggregatedSetList{})
}
