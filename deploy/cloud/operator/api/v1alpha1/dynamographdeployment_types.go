/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 Atalaya Tech. Inc
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
 */

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// ComponentKind represents the type of underlying Kubernetes resource.
// +kubebuilder:validation:Enum=PodClique;PodCliqueScalingGroup;Deployment;LeaderWorkerSet
type ComponentKind string

const (
	// ComponentKindPodClique represents a PodClique resource.
	ComponentKindPodClique ComponentKind = "PodClique"
	// ComponentKindPodCliqueScalingGroup represents a PodCliqueScalingGroup resource.
	ComponentKindPodCliqueScalingGroup ComponentKind = "PodCliqueScalingGroup"
	// ComponentKindDeployment represents a Deployment resource.
	ComponentKindDeployment ComponentKind = "Deployment"
	// ComponentKindLeaderWorkerSet represents a LeaderWorkerSet resource.
	ComponentKindLeaderWorkerSet ComponentKind = "LeaderWorkerSet"
)

// DynamoGraphDeploymentSpec defines the desired state of DynamoGraphDeployment.
type DynamoGraphDeploymentSpec struct {
	// PVCs defines a list of persistent volume claims that can be referenced by components.
	// Each PVC must have a unique name that can be referenced in component specifications.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:MaxItems=100
	PVCs []PVC `json:"pvcs,omitempty"`
	// Services are the services to deploy as part of this deployment.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:MaxProperties=25
	Services map[string]*DynamoComponentDeploymentSharedSpec `json:"services,omitempty"`
	// Envs are environment variables applied to all services in the deployment unless
	// overridden by service-specific configuration.
	// +kubebuilder:validation:Optional
	Envs []corev1.EnvVar `json:"envs,omitempty"`
	// BackendFramework specifies the backend framework (e.g., "sglang", "vllm", "trtllm").
	// +kubebuilder:validation:Enum=sglang;vllm;trtllm
	BackendFramework string `json:"backendFramework,omitempty"`
}

// DynamoGraphDeploymentStatus defines the observed state of DynamoGraphDeployment.
type DynamoGraphDeploymentStatus struct {
	// State is a high-level textual status of the graph deployment lifecycle.
	State string `json:"state,omitempty"`
	// Conditions contains the latest observed conditions of the graph deployment.
	// The slice is merged by type on patch updates.
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
	// Services contains per-service replica status information.
	// The map key is the service name from spec.services.
	// +optional
	Services map[string]ServiceReplicaStatus `json:"services,omitempty"`
}

// ServiceReplicaStatus contains replica information for a single service.
type ServiceReplicaStatus struct {
	// ComponentKind is the underlying resource kind (e.g., "PodClique", "PodCliqueScalingGroup", "Deployment", "LeaderWorkerSet").
	ComponentKind ComponentKind `json:"componentKind"`
	// ComponentName is the name of the underlying resource.
	ComponentName string `json:"componentName"`

	// Replicas is the total number of non-terminated replicas.
	// Required for all component kinds.
	// +kubebuilder:validation:Minimum=0
	Replicas int32 `json:"replicas"`

	// UpdatedReplicas is the number of replicas at the current/desired revision.
	// Required for all component kinds.
	// +kubebuilder:validation:Minimum=0
	UpdatedReplicas int32 `json:"updatedReplicas"`

	// ReadyReplicas is the number of ready replicas.
	// Populated for PodClique, Deployment, and LeaderWorkerSet.
	// Not available for PodCliqueScalingGroup.
	// When nil, the field is omitted from the API response.
	// +optional
	// +kubebuilder:validation:Minimum=0
	ReadyReplicas *int32 `json:"readyReplicas,omitempty"`

	// AvailableReplicas is the number of available replicas.
	// For Deployment: replicas ready for >= minReadySeconds.
	// For PodCliqueScalingGroup: replicas where all constituent PodCliques have >= MinAvailable ready pods.
	// Not available for PodClique or LeaderWorkerSet.
	// When nil, the field is omitted from the API response.
	// +optional
	// +kubebuilder:validation:Minimum=0
	AvailableReplicas *int32 `json:"availableReplicas,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dgd
// DynamoGraphDeployment is the Schema for the dynamographdeployments API.
type DynamoGraphDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired state for this graph deployment.
	Spec DynamoGraphDeploymentSpec `json:"spec,omitempty"`
	// Status reflects the current observed state of this graph deployment.
	Status DynamoGraphDeploymentStatus `json:"status,omitempty"`
}

func (s *DynamoGraphDeployment) SetState(state string) {
	s.Status.State = state
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

func (s *DynamoGraphDeployment) GetSpec() any {
	return s.Spec
}

func (s *DynamoGraphDeployment) SetSpec(spec any) {
	s.Spec = spec.(DynamoGraphDeploymentSpec)
}

func (s *DynamoGraphDeployment) AddStatusCondition(condition metav1.Condition) {
	if s.Status.Conditions == nil {
		s.Status.Conditions = []metav1.Condition{}
	}
	// Check if condition with same type already exists
	for i, existingCondition := range s.Status.Conditions {
		if existingCondition.Type == condition.Type {
			// Replace the existing condition
			s.Status.Conditions[i] = condition
			return
		}
	}
	// If no matching condition found, append the new one
	s.Status.Conditions = append(s.Status.Conditions, condition)
}

// HasAnyMultinodeService reports whether any service in the graph is configured with more than one node.
func (s *DynamoGraphDeployment) HasAnyMultinodeService() bool {
	for _, svc := range s.Spec.Services {
		if svc != nil && svc.GetNumberOfNodes() > 1 {
			return true
		}
	}
	return false
}
