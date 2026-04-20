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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// DynamoComponentDeploymentService pairs a service name with its shared spec.
// In v1beta1 the DGD `services` field is a list of these rather than a
// name-keyed map, so that object ordering is stable in YAML and GitOps diffs
// and so that the name lives as a first-class field alongside the spec.
type DynamoComponentDeploymentService struct {
	// Name identifies the service within the graph (e.g. "vllm-worker", "frontend").
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// DynamoComponentDeploymentSharedSpec embeds the component's spec inline.
	DynamoComponentDeploymentSharedSpec `json:",inline"`
}

// DynamoGraphDeploymentSpec defines the desired state of a DynamoGraphDeployment.
type DynamoGraphDeploymentSpec struct {
	// Annotations to propagate to all child resources (PCS, DCD, Deployments,
	// and pod templates). Service-level (podTemplate) values take precedence
	// on conflict.
	// +optional
	Annotations map[string]string `json:"annotations,omitempty"`

	// Labels to propagate to all child resources. Same precedence rules as Annotations.
	// +optional
	Labels map[string]string `json:"labels,omitempty"`

	// Services are the services deployed as part of this graph. Names must be
	// unique within the list; the API server enforces this via the
	// listType=map + listMapKey=name markers below (no CEL rule needed, which
	// keeps the CRD within its x-kubernetes-validations cost budget).
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=25
	Services []DynamoComponentDeploymentService `json:"services,omitempty"`

	// Env is the list of environment variables applied to all services in the
	// deployment unless overridden by service-specific configuration. Renamed
	// from v1alpha1's `envs` to match the Kubernetes convention
	// (`pod.spec.containers[].env`).
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`

	// BackendFramework specifies the backend framework (e.g. "sglang", "vllm", "trtllm").
	// +kubebuilder:validation:Enum=sglang;vllm;trtllm
	BackendFramework string `json:"backendFramework,omitempty"`

	// Restart specifies the restart policy for the graph deployment.
	// +optional
	Restart *Restart `json:"restart,omitempty"`

	// TopologyConstraint is the deployment-level topology constraint. When set,
	// `topologyProfile` names the ClusterTopology CR to use. `packDomain` is
	// optional at this level and can be omitted when only services carry constraints.
	// Services without their own topologyConstraint inherit from this value.
	// +optional
	TopologyConstraint *SpecTopologyConstraint `json:"topologyConstraint,omitempty"`
}

// DynamoGraphDeploymentStatus defines the observed state of a DynamoGraphDeployment.
// Unchanged between v1alpha1 and v1beta1.
type DynamoGraphDeploymentStatus struct {
	// ObservedGeneration is the most recent generation observed by the controller.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// State is a high-level textual status of the graph deployment lifecycle.
	// +kubebuilder:default=initializing
	State DGDState `json:"state"`

	// Conditions contains the latest observed conditions of the graph deployment.
	// Merged by type on patch updates.
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

	// Services contains per-service replica status information, keyed by service name.
	// +optional
	Services map[string]ServiceReplicaStatus `json:"services,omitempty"`

	// Restart contains the status of a graph-level restart.
	// +optional
	Restart *RestartStatus `json:"restart,omitempty"`

	// Checkpoints contains per-service checkpoint status, keyed by service name.
	// +optional
	Checkpoints map[string]ServiceCheckpointStatus `json:"checkpoints,omitempty"`

	// RollingUpdate tracks the progress of operator-managed rolling updates.
	// Currently only supported for single-node, non-Grove deployments (DCD/Deployment).
	// +optional
	RollingUpdate *RollingUpdateStatus `json:"rollingUpdate,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:unservedversion
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dgd
// +kubebuilder:printcolumn:name="Ready",type="string",JSONPath=`.status.conditions[?(@.type=="Ready")].status`,description="Ready status of the graph deployment"
// +kubebuilder:printcolumn:name="Backend",type="string",JSONPath=`.spec.backendFramework`,description="Backend framework (sglang, vllm, trtllm)"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// DynamoGraphDeployment is the Schema for the dynamographdeployments API.
//
// v1beta1 is currently an UNSERVED version: it is defined so that conversion
// scaffolding and type generation can land ahead of the full multi-version
// wiring. Callers must continue to use v1alpha1 until v1beta1 is promoted to
// served in a subsequent MR.
type DynamoGraphDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired state for this graph deployment.
	Spec DynamoGraphDeploymentSpec `json:"spec,omitempty"`
	// Status reflects the current observed state of this graph deployment.
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
	if s.Status.Conditions == nil {
		s.Status.Conditions = []metav1.Condition{}
	}
	for i, existing := range s.Status.Conditions {
		if existing.Type == condition.Type {
			s.Status.Conditions[i] = condition
			return
		}
	}
	s.Status.Conditions = append(s.Status.Conditions, condition)
}

// GetServiceByName returns the service entry with the given name, or nil if not found.
// Helper for the v1beta1 list-based services field.
func (s *DynamoGraphDeployment) GetServiceByName(name string) *DynamoComponentDeploymentService {
	for i := range s.Spec.Services {
		if s.Spec.Services[i].Name == name {
			return &s.Spec.Services[i]
		}
	}
	return nil
}

// HasAnyTopologyConstraint reports whether any topology constraint is set at any level.
func (s *DynamoGraphDeployment) HasAnyTopologyConstraint() bool {
	if s.Spec.TopologyConstraint != nil {
		return true
	}
	for i := range s.Spec.Services {
		if s.Spec.Services[i].TopologyConstraint != nil {
			return true
		}
	}
	return false
}

// HasAnyMultinodeService reports whether any service is configured with more than one node.
func (s *DynamoGraphDeployment) HasAnyMultinodeService() bool {
	for i := range s.Spec.Services {
		if s.Spec.Services[i].GetNumberOfNodes() > 1 {
			return true
		}
	}
	return false
}

// HasEPPService returns true if any service in the DGD has the EPP component type.
func (s *DynamoGraphDeployment) HasEPPService() bool {
	for i := range s.Spec.Services {
		if s.Spec.Services[i].ComponentType == ComponentTypeEPP {
			return true
		}
	}
	return false
}

// GetEPPService returns the EPP service's name and spec if present.
func (s *DynamoGraphDeployment) GetEPPService() (string, *DynamoComponentDeploymentSharedSpec, bool) {
	for i := range s.Spec.Services {
		svc := &s.Spec.Services[i]
		if svc.ComponentType == ComponentTypeEPP {
			return svc.Name, &svc.DynamoComponentDeploymentSharedSpec, true
		}
	}
	return "", nil, false
}

// GetDynamoNamespaceForService returns the Dynamo namespace for a given service.
func (s *DynamoGraphDeployment) GetDynamoNamespaceForService(svc *DynamoComponentDeploymentSharedSpec) string {
	return ComputeDynamoNamespace(svc.GlobalDynamoNamespace, s.GetNamespace(), s.GetName())
}
