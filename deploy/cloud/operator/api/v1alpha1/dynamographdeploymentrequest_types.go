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

/*
Package v1alpha1 contains API Schema definitions for the nvidia.com v1alpha1 API group.

This package defines the DynamoGraphDeploymentRequest (DGDR) custom resource, which provides
a high-level, SLA-driven interface for deploying machine learning models on Dynamo.
*/
package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// SLASpec defines Service Level Agreement targets for model profiling and deployment.
// These targets guide the profiling process to find optimal deployment configurations
// that meet the specified performance requirements.
type SLASpec struct {
	// ITL is the target Inter-Token Latency in milliseconds.
	// This represents the maximum time allowed between consecutive tokens in the output.
	// +kubebuilder:default=10
	// +optional
	ITL int `json:"itl,omitempty"`

	// TTFT is the target Time To First Token in milliseconds.
	// This represents the maximum time allowed from request submission to receiving the first token.
	// +kubebuilder:default=50
	// +optional
	TTFT int `json:"ttft,omitempty"`

	// ISL is the Input Sequence Length for profiling.
	// Defines the length of input sequences to use during profiling tests.
	// +kubebuilder:default=3000
	// +kubebuilder:validation:Minimum=1
	// +optional
	ISL int `json:"isl,omitempty"`

	// OSL is the Output Sequence Length for profiling.
	// Defines the expected length of output sequences to generate during profiling tests.
	// +kubebuilder:default=500
	// +kubebuilder:validation:Minimum=1
	// +optional
	OSL int `json:"osl,omitempty"`
}

// GPUSpec defines optional GPU type and resource specifications for profiling and deployment.
// These constraints help narrow down the search space during profiling to find configurations
// that fit within specified hardware bounds.
type GPUSpec struct {
	// Type specifies the GPU type to target (e.g., "h200", "h100", "a100").
	// If specified, profiling will focus on configurations optimized for this GPU type.
	// +kubebuilder:validation:Optional
	Type string `json:"type,omitempty"`

	// MinNumGPUsPerEngine specifies the minimum number of GPUs per engine for profiling.
	// The profiler will not consider configurations with fewer GPUs than this value.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	MinNumGPUsPerEngine int `json:"minNumGPUsPerEngine,omitempty"`

	// MaxNumGPUsPerEngine specifies the maximum number of GPUs per engine for profiling.
	// The profiler will not consider configurations with more GPUs than this value.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=8
	MaxNumGPUsPerEngine int `json:"maxNumGPUsPerEngine,omitempty"`
}

// ConfigMapKeySelector selects a specific key from a ConfigMap.
// Used to reference external configuration data stored in ConfigMaps.
type ConfigMapKeySelector struct {
	// Name of the ConfigMap containing the desired data.
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// Key in the ConfigMap to select. If not specified, defaults to "disagg.yaml".
	// +kubebuilder:default=disagg.yaml
	Key string `json:"key,omitempty"`
}

// ProfilingConfigSpec defines configuration for the profiling process.
// Allows users to provide custom profiling parameters via ConfigMap references.
type ProfilingConfigSpec struct {
	// ConfigMapRef is a reference to a ConfigMap containing profiling configuration.
	// The ConfigMap should contain a key (default: "disagg.yaml") with the configuration file.
	// This configuration is used by both online and offline (AIC) profiling modes.
	// +kubebuilder:validation:Optional
	ConfigMapRef *ConfigMapKeySelector `json:"configMapRef,omitempty"`
}

// DeploymentOverridesSpec allows users to customize metadata for auto-created DynamoGraphDeployments.
// When autoApply is enabled, these overrides are applied to the generated DGD resource.
type DeploymentOverridesSpec struct {
	// Name is the desired name for the created DynamoGraphDeployment.
	// If not specified, defaults to the DGDR name.
	// +kubebuilder:validation:Optional
	Name string `json:"name,omitempty"`

	// Namespace is the desired namespace for the created DynamoGraphDeployment.
	// If not specified, defaults to the DGDR namespace.
	// +kubebuilder:validation:Optional
	Namespace string `json:"namespace,omitempty"`

	// Labels are additional labels to add to the DynamoGraphDeployment metadata.
	// These are merged with auto-generated labels from the profiling process.
	// +kubebuilder:validation:Optional
	Labels map[string]string `json:"labels,omitempty"`

	// Annotations are additional annotations to add to the DynamoGraphDeployment metadata.
	// +kubebuilder:validation:Optional
	Annotations map[string]string `json:"annotations,omitempty"`
}

// DynamoGraphDeploymentRequestSpec defines the desired state of a DynamoGraphDeploymentRequest.
// This CRD serves as the primary interface for users to request model deployments with
// specific performance constraints and resource requirements, enabling SLA-driven deployments.
type DynamoGraphDeploymentRequestSpec struct {
	// ModelName specifies the model to deploy (e.g., "meta/llama3-70b").
	// This should be a valid model identifier that the profiler can resolve.
	// +kubebuilder:validation:Required
	ModelName string `json:"modelName"`

	// Backend specifies the inference backend framework to use.
	// Supported values are: "vllm", "sglang", "trtllm".
	// +kubebuilder:validation:Enum=vllm;sglang;trtllm
	// +kubebuilder:default=trtllm
	Backend string `json:"backend,omitempty"`

	// SLA defines the Service Level Agreement profiling targets.
	// The profiler uses these targets to find an optimal deployment configuration.
	// +kubebuilder:validation:Required
	SLA SLASpec `json:"sla"`

	// GPU defines optional GPU type and resource specifications.
	// These constraints guide the profiler to find configurations within specified bounds.
	// +kubebuilder:validation:Optional
	GPU *GPUSpec `json:"gpu,omitempty"`

	// Online indicates whether to use online profiler (true) or AI Configurator (false).
	// Online profiling uses real deployments for accurate measurements (2-4 hours).
	// Offline profiling uses AI Configurator for fast simulation-based profiling (20-30 seconds).
	// +kubebuilder:default=false
	Online bool `json:"online,omitempty"`

	// AutoApply indicates whether to automatically create a DynamoGraphDeployment
	// after profiling completes. If false, only the spec is generated and stored in status.
	// Users can then manually create a DGD using the generated spec.
	// +kubebuilder:default=false
	AutoApply bool `json:"autoApply,omitempty"`

	// DeploymentOverrides allows customizing metadata for the auto-created DGD.
	// Only applicable when AutoApply is true.
	// +kubebuilder:validation:Optional
	DeploymentOverrides *DeploymentOverridesSpec `json:"deploymentOverrides,omitempty"`

	// ProfilingConfig provides custom configuration for the profiling job.
	// Applicable to both online and offline (AIC) profiling modes.
	// +kubebuilder:validation:Optional
	ProfilingConfig *ProfilingConfigSpec `json:"profilingConfig,omitempty"`
}

// DeploymentStatus tracks the state of an auto-created DynamoGraphDeployment.
// This status is populated when autoApply is enabled and a DGD is created.
type DeploymentStatus struct {
	// Name is the name of the created DynamoGraphDeployment.
	Name string `json:"name,omitempty"`

	// Namespace is the namespace of the created DynamoGraphDeployment.
	Namespace string `json:"namespace,omitempty"`

	// State is the current state of the DynamoGraphDeployment.
	// This value is mirrored from the DGD's status.state field.
	State string `json:"state,omitempty"`

	// Created indicates whether the DGD has been successfully created.
	// Used to prevent recreation if the DGD is manually deleted by users.
	Created bool `json:"created,omitempty"`
}

// DynamoGraphDeploymentRequestStatus represents the observed state of a DynamoGraphDeploymentRequest.
// The controller updates this status as the DGDR progresses through its lifecycle.
type DynamoGraphDeploymentRequestStatus struct {
	// State is a high-level textual status of the deployment request lifecycle.
	// Possible values: "", "Pending", "Profiling", "Deploying", "Ready", "DeploymentDeleted", "Failed"
	// Empty string ("") represents the initial state before initialization.
	State string `json:"state,omitempty"`

	// ObservedGeneration reflects the generation of the most recently observed spec.
	// Used to detect spec changes and enforce immutability after profiling starts.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Conditions contains the latest observed conditions of the deployment request.
	// Standard condition types include: Validation, Profiling, SpecGenerated, DeploymentReady.
	// Conditions are merged by type on patch updates.
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

	// ProfilingResults contains a reference to the ConfigMap holding profiling data.
	// Format: "configmap/<name>"
	// +kubebuilder:validation:Optional
	ProfilingResults string `json:"profilingResults,omitempty"`

	// GeneratedDeployment contains the full generated DynamoGraphDeployment specification
	// including metadata, based on profiling results. Users can extract this to create
	// a DGD manually, or it's used automatically when autoApply is true.
	// Stored as RawExtension to preserve all fields including metadata.
	// +kubebuilder:validation:Optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:EmbeddedResource
	GeneratedDeployment *runtime.RawExtension `json:"generatedDeployment,omitempty"`

	// Deployment tracks the auto-created DGD when AutoApply is true.
	// Contains name, namespace, state, and creation status of the managed DGD.
	// +kubebuilder:validation:Optional
	Deployment *DeploymentStatus `json:"deployment,omitempty"`
}

// DynamoGraphDeploymentRequest is the Schema for the dynamographdeploymentrequests API.
// It serves as the primary interface for users to request model deployments with
// specific performance and resource constraints, enabling SLA-driven deployments.
//
// Lifecycle:
//  1. Initial → Pending: Validates spec and prepares for profiling
//  2. Pending → Profiling: Creates and runs profiling job (online or AIC)
//  3. Profiling → Ready/Deploying: Generates DGD spec after profiling completes
//  4. Deploying → Ready: When autoApply=true, monitors DGD until Ready
//  5. Ready: Terminal state when DGD is operational or spec is available
//  6. DeploymentDeleted: Terminal state when auto-created DGD is manually deleted
//
// The spec becomes immutable once profiling starts. Users must delete and recreate
// the DGDR to modify configuration after this point.
//
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dgdr
// +kubebuilder:printcolumn:name="Model",type=string,JSONPath=`.spec.modelName`
// +kubebuilder:printcolumn:name="Backend",type=string,JSONPath=`.spec.backend`
// +kubebuilder:printcolumn:name="State",type=string,JSONPath=`.status.state`
// +kubebuilder:printcolumn:name="DGD-State",type=string,JSONPath=`.status.deployment.state`
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
type DynamoGraphDeploymentRequest struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired state for this deployment request.
	Spec DynamoGraphDeploymentRequestSpec `json:"spec,omitempty"`

	// Status reflects the current observed state of this deployment request.
	Status DynamoGraphDeploymentRequestStatus `json:"status,omitempty"`
}

// SetState updates the State field in the DGDR status.
func (s *DynamoGraphDeploymentRequest) SetState(state string) {
	s.Status.State = state
}

// GetSpec returns the spec of this DGDR as a generic interface.
// Implements a common interface used by controller utilities.
func (s *DynamoGraphDeploymentRequest) GetSpec() any {
	return s.Spec
}

// SetSpec updates the spec of this DGDR from a generic interface value.
// Implements a common interface used by controller utilities.
func (s *DynamoGraphDeploymentRequest) SetSpec(spec any) {
	s.Spec = spec.(DynamoGraphDeploymentRequestSpec)
}

// AddStatusCondition adds or updates a condition in the status.
// If a condition with the same type already exists, it replaces it.
// Otherwise, it appends the new condition to the list.
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

// DynamoGraphDeploymentRequestList contains a list of DynamoGraphDeploymentRequest resources.
//
// +kubebuilder:object:root=true
type DynamoGraphDeploymentRequestList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeploymentRequest `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoGraphDeploymentRequest{}, &DynamoGraphDeploymentRequestList{})
}
