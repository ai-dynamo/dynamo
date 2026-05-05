/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	corev1 "k8s.io/api/core/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// DynamoGraphDeploymentSpec defines the desired state of a DynamoGraphDeployment.
type DynamoGraphDeploymentSpec struct {
	// annotations to propagate to all child resources (PCS, DCD, Deployments,
	// and pod templates). Component-level (`podTemplate`) values take precedence
	// on conflict.
	// +optional
	Annotations map[string]string `json:"annotations,omitempty"`

	// labels to propagate to all child resources. Same precedence rules as `annotations`.
	// +optional
	Labels map[string]string `json:"labels,omitempty"`

	// components are the components deployed as part of this graph. Each entry
	// carries its own stable logical `name`, and names must be unique within
	// the list. Component types are generally repeatable, except `type: epp`
	// which may appear at most once.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=25
	// +kubebuilder:validation:XValidation:rule="self.filter(c, has(c.type) && c.type == 'epp').size() <= 1",message="at most one component may have type epp"
	Components []DynamoComponentDeploymentSharedSpec `json:"components,omitempty"`

	// env is prepended to every component's environment. Component-specific
	// env entries with the same name take precedence and may reference values
	// from this list.
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`

	// backendFramework specifies the backend framework (e.g. "sglang", "vllm", "trtllm").
	// +kubebuilder:validation:Enum=sglang;vllm;trtllm
	BackendFramework string `json:"backendFramework,omitempty"`

	// restart specifies the restart policy for the graph deployment.
	// +optional
	Restart *Restart `json:"restart,omitempty"`

	// topologyConstraint is the deployment-level topology constraint. When
	// set, `spec.topologyConstraint.clusterTopologyName` names the ClusterTopology
	// CR to use. `spec.topologyConstraint.packDomain` is optional at this
	// level and can be omitted when only components carry constraints.
	// Components without their own `topologyConstraint` inherit from this value.
	// +optional
	TopologyConstraint *SpecTopologyConstraint `json:"topologyConstraint,omitempty"`
}

// DynamoGraphDeploymentStatus defines the observed state of a DynamoGraphDeployment.
// Unchanged between v1alpha1 and v1beta1.
type DynamoGraphDeploymentStatus struct {
	// observedGeneration is the most recent generation observed by the controller.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// state is a high-level textual status of the graph deployment lifecycle.
	// +kubebuilder:default=initializing
	State DGDState `json:"state"`

	// conditions contains the latest observed conditions of the graph deployment.
	// Merged by type on patch updates.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// components contains per-component replica status information, keyed by component name.
	// +optional
	Components map[string]ComponentReplicaStatus `json:"components,omitempty"`

	// restart contains the status of a graph-level restart.
	// +optional
	Restart *RestartStatus `json:"restart,omitempty"`

	// checkpoints contains per-component checkpoint status, keyed by component name.
	// +optional
	Checkpoints map[string]ComponentCheckpointStatus `json:"checkpoints,omitempty"`

	// rollingUpdate tracks the progress of operator-managed rolling updates.
	// Currently only supported for single-node, non-Grove deployments (DCD/Deployment).
	// +optional
	RollingUpdate *RollingUpdateStatus `json:"rollingUpdate,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dgd
// +kubebuilder:printcolumn:name="Ready",type="string",JSONPath=`.status.conditions[?(@.type=="Ready")].status`,description="Ready status of the graph deployment"
// +kubebuilder:printcolumn:name="Backend",type="string",JSONPath=`.spec.backendFramework`,description="Backend framework (sglang, vllm, trtllm)"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// DynamoGraphDeployment is the Schema for the dynamographdeployments API.
//
// v1beta1 is a served version: the API server accepts reads and writes
// against it, and transparently converts to/from v1alpha1 (still the
// storage version until a later MR flips it). Conversion goes through the
// operator's conversion webhook; see api/v1alpha1/*_conversion.go.
type DynamoGraphDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec defines the desired state for this graph deployment.
	Spec DynamoGraphDeploymentSpec `json:"spec,omitempty"`
	// status reflects the current observed state of this graph deployment.
	Status DynamoGraphDeploymentStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoGraphDeploymentList contains a list of DynamoGraphDeployment.
type DynamoGraphDeploymentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeployment `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoGraphDeployment{}, &DynamoGraphDeploymentList{})
}

// SetState updates the high-level lifecycle state.
func (s *DynamoGraphDeployment) SetState(state DGDState) {
	s.Status.State = state
}

// GetState returns the current lifecycle state as a string.
func (s *DynamoGraphDeployment) GetState() string {
	return string(s.Status.State)
}

// GetSpec returns the spec as an interface, used by generic resource helpers.
func (s *DynamoGraphDeployment) GetSpec() any {
	return s.Spec
}

// SetSpec assigns the spec from an interface value.
func (s *DynamoGraphDeployment) SetSpec(spec any) {
	s.Spec = spec.(DynamoGraphDeploymentSpec)
}

// AddStatusCondition adds or updates the condition slice by type.
func (s *DynamoGraphDeployment) AddStatusCondition(condition metav1.Condition) {
	apimeta.SetStatusCondition(&s.Status.Conditions, condition)
}

// GetComponentByName returns the component entry with the given name,
// or nil if not found. Helper for the v1beta1 list-based `components` field.
func (s *DynamoGraphDeployment) GetComponentByName(name string) *DynamoComponentDeploymentSharedSpec {
	for i := range s.Spec.Components {
		if s.Spec.Components[i].ComponentName == name {
			return &s.Spec.Components[i]
		}
	}
	return nil
}

// HasAnyTopologyConstraint reports whether any topology constraint is set at any level.
func (s *DynamoGraphDeployment) HasAnyTopologyConstraint() bool {
	if s.Spec.TopologyConstraint != nil {
		return true
	}
	for i := range s.Spec.Components {
		if s.Spec.Components[i].TopologyConstraint != nil {
			return true
		}
	}
	return false
}

// HasAnyMultinodeComponent reports whether any component is configured with more than one node.
func (s *DynamoGraphDeployment) HasAnyMultinodeComponent() bool {
	for i := range s.Spec.Components {
		if s.Spec.Components[i].GetNumberOfNodes() > 1 {
			return true
		}
	}
	return false
}

// HasEPPComponent returns true if any component in the DGD has the EPP component type.
func (s *DynamoGraphDeployment) HasEPPComponent() bool {
	for i := range s.Spec.Components {
		if s.Spec.Components[i].ComponentType == ComponentTypeEPP {
			return true
		}
	}
	return false
}

// GetEPPComponent returns the EPP component's name and shared spec if present.
// The API allows at most one component with `type: epp`.
func (s *DynamoGraphDeployment) GetEPPComponent() (string, *DynamoComponentDeploymentSharedSpec, bool) {
	for i := range s.Spec.Components {
		comp := &s.Spec.Components[i]
		if comp.ComponentType == ComponentTypeEPP {
			return comp.ComponentName, comp, true
		}
	}
	return "", nil, false
}

// GetDynamoNamespaceForComponent returns the Dynamo namespace for a given component.
func (s *DynamoGraphDeployment) GetDynamoNamespaceForComponent(comp *DynamoComponentDeploymentSharedSpec) string {
	return ComputeDynamoNamespace(comp.GlobalDynamoNamespace, s.GetNamespace(), s.GetName())
}
