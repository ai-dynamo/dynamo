/*
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
 */

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// SLASpec defines the Service Level Agreement profiling targets.
type SLASpec struct {
	// ITL is the target Inter-Token Latency in milliseconds.
	// +kubebuilder:validation:Required
	ITL int `json:"itl"`
	// TTFT is the target Time To First Token in milliseconds.
	// +kubebuilder:validation:Required
	TTFT int `json:"ttft"`
}

// GPUSpec defines optional GPU type specification.
type GPUSpec struct {
	// Type specifies the GPU type (e.g., "h200", "h100", "a100").
	// +kubebuilder:validation:Optional
	Type string `json:"type,omitempty"`
}

// ConfigMapKeySelector selects a key from a ConfigMap.
type ConfigMapKeySelector struct {
	// Name of the ConfigMap.
	// +kubebuilder:validation:Required
	Name string `json:"name"`
	// Key in the ConfigMap to select.
	// +kubebuilder:default=disagg.yaml
	Key string `json:"key,omitempty"`
}

// ProfilingConfigSpec defines the profiling configuration.
type ProfilingConfigSpec struct {
	// ConfigMapRef is a reference to a ConfigMap containing the profiling configuration.
	// The ConfigMap should contain a key (default: "disagg.yaml") with the configuration file.
	// Can be used for both online and offline (AIC) profiling.
	// +kubebuilder:validation:Optional
	ConfigMapRef *ConfigMapKeySelector `json:"configMapRef,omitempty"`
}

// DeploymentOverridesSpec defines metadata overrides for the auto-created DGD.
type DeploymentOverridesSpec struct {
	// Name is the name for the created DynamoGraphDeployment.
	// If not specified, defaults to the DGDR name.
	// +kubebuilder:validation:Optional
	Name string `json:"name,omitempty"`

	// Namespace is the namespace for the created DynamoGraphDeployment.
	// If not specified, defaults to the DGDR namespace.
	// +kubebuilder:validation:Optional
	Namespace string `json:"namespace,omitempty"`

	// Labels are additional labels to add to the DynamoGraphDeployment.
	// These are merged with auto-generated labels.
	// +kubebuilder:validation:Optional
	Labels map[string]string `json:"labels,omitempty"`

	// Annotations are additional annotations to add to the DynamoGraphDeployment.
	// +kubebuilder:validation:Optional
	Annotations map[string]string `json:"annotations,omitempty"`
}

// DynamoGraphDeploymentRequestSpec defines the desired state of DynamoGraphDeploymentRequest.
// This CRD serves as the primary interface for users to request model deployments
// with specific performance and resource constraints for SLA-driven deployments.
type DynamoGraphDeploymentRequestSpec struct {
	// ModelName specifies the model to deploy (e.g., "meta/llama3-70b").
	// +kubebuilder:validation:Required
	ModelName string `json:"modelName"`

	// Backend specifies the backend framework to use.
	// +kubebuilder:validation:Enum=vllm;sglang;trtllm
	// +kubebuilder:default=trtllm
	Backend string `json:"backend,omitempty"`

	// SLA defines the Service Level Agreement profiling targets.
	// +kubebuilder:validation:Required
	SLA SLASpec `json:"sla"`

	// GPU defines optional GPU type specification.
	// +kubebuilder:validation:Optional
	GPU *GPUSpec `json:"gpu,omitempty"`

	// Online indicates whether to use online profiler (true) or AI Configurator (false).
	// When true, uses real deployment for profiling (2-4 hours).
	// When false, uses AI Configurator for fast profiling (20-30 seconds).
	// +kubebuilder:default=false
	Online bool `json:"online,omitempty"`

	// AutoApply indicates whether to automatically create a DynamoGraphDeployment
	// after profiling completes. If false, only the spec is generated in status.
	// +kubebuilder:default=false
	AutoApply bool `json:"autoApply,omitempty"`

	// DeploymentOverrides allows overriding metadata for the auto-created DGD.
	// Only used when AutoApply is true.
	// +kubebuilder:validation:Optional
	DeploymentOverrides *DeploymentOverridesSpec `json:"deploymentOverrides,omitempty"`

	// ProfilingConfig provides configuration for the profiling job.
	// Can be used for both online and offline (AIC) profiling.
	// +kubebuilder:validation:Optional
	ProfilingConfig *ProfilingConfigSpec `json:"profilingConfig,omitempty"`
}

// DeploymentStatus tracks the auto-created DGD status.
type DeploymentStatus struct {
	// Name is the name of the created DynamoGraphDeployment.
	Name string `json:"name,omitempty"`

	// Namespace is the namespace of the created DynamoGraphDeployment.
	Namespace string `json:"namespace,omitempty"`

	// State is the current state of the DynamoGraphDeployment.
	// This is mirrored from the DGD's status.state field.
	State string `json:"state,omitempty"`

	// Created indicates whether the DGD has been created.
	// Used to prevent recreation if DGD is deleted by user.
	Created bool `json:"created,omitempty"`
}

// DynamoGraphDeploymentRequestStatus defines the observed state of DynamoGraphDeploymentRequest.
type DynamoGraphDeploymentRequestStatus struct {
	// State is a high-level textual status of the deployment request lifecycle.
	// Possible values: "Pending", "Profiling", "Deploying", "Ready", "DeploymentDeleted", "Failed"
	State string `json:"state,omitempty"`

	// ObservedGeneration reflects the generation of the most recently observed spec.
	// Used to detect spec changes and enforce immutability.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Conditions contains the latest observed conditions of the deployment request.
	// The slice is merged by type on patch updates.
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

	// ProfilingResults contains references to the profiling data and results.
	// +kubebuilder:validation:Optional
	ProfilingResults string `json:"profilingResults,omitempty"`

	// GeneratedDeployment contains the full generated DynamoGraphDeployment (including metadata)
	// based on profiling results. This can be used to create a DynamoGraphDeployment resource.
	// Stored as RawExtension to preserve all fields including metadata.
	// +kubebuilder:validation:Optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:EmbeddedResource
	GeneratedDeployment *runtime.RawExtension `json:"generatedDeployment,omitempty"`

	// Deployment tracks the auto-created DGD if AutoApply is true.
	// +kubebuilder:validation:Optional
	Deployment *DeploymentStatus `json:"deployment,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dgdr
// +kubebuilder:printcolumn:name="Model",type=string,JSONPath=`.spec.modelName`
// +kubebuilder:printcolumn:name="Backend",type=string,JSONPath=`.spec.backend`
// +kubebuilder:printcolumn:name="State",type=string,JSONPath=`.status.state`
// +kubebuilder:printcolumn:name="DGD-State",type=string,JSONPath=`.status.deployment.state`
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// DynamoGraphDeploymentRequest is the Schema for the dynamographdeploymentrequests API.
// It serves as the primary interface for users to request model deployments with
// specific performance and resource constraints, enabling SLA-driven deployments.
type DynamoGraphDeploymentRequest struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired state for this deployment request.
	Spec DynamoGraphDeploymentRequestSpec `json:"spec,omitempty"`
	// Status reflects the current observed state of this deployment request.
	Status DynamoGraphDeploymentRequestStatus `json:"status,omitempty"`
}

func (s *DynamoGraphDeploymentRequest) SetState(state string) {
	s.Status.State = state
}

// +kubebuilder:object:root=true

// DynamoGraphDeploymentRequestList contains a list of DynamoGraphDeploymentRequest.
type DynamoGraphDeploymentRequestList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeploymentRequest `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoGraphDeploymentRequest{}, &DynamoGraphDeploymentRequestList{})
}

func (s *DynamoGraphDeploymentRequest) GetSpec() any {
	return s.Spec
}

func (s *DynamoGraphDeploymentRequest) SetSpec(spec any) {
	s.Spec = spec.(DynamoGraphDeploymentRequestSpec)
}

func (s *DynamoGraphDeploymentRequest) AddStatusCondition(condition metav1.Condition) {
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
