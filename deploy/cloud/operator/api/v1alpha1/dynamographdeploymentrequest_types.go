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
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

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
// This structure maps directly to the profile_sla.py config format.
// See benchmarks/profiler/utils/profiler_argparse.py for the complete schema.
type ProfilingConfigSpec struct {
	// Config is the profiling configuration as arbitrary JSON/YAML. This will be passed directly to the profiler.
	// The profiler will validate the configuration and report any errors.
	// +kubebuilder:validation:Optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	Config *apiextensionsv1.JSON `json:"config,omitempty"`

	// ConfigMapRef is an optional reference to a ConfigMap containing the DynamoGraphDeployment
	// base config file (disagg.yaml). This is separate from the profiling config above.
	// The path to this config will be set as engine.config in the profiling config.
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
	// ModelName specifies the model to deploy (e.g., "Qwen/Qwen3-0.6B", "meta-llama/Llama-3-70b").
	// This is a high-level identifier for easy reference in kubectl output and logs.
	// +kubebuilder:validation:Required
	ModelName string `json:"modelName"`

	// ProfilingConfig provides the complete configuration for the profiling job.
	// This configuration is passed directly to the profiler.
	// The structure matches the profile_sla config format exactly (see ProfilingConfigSpec for schema).
	// The profiler will validate the configuration and report any errors.
	// +kubebuilder:validation:Required
	ProfilingConfig ProfilingConfigSpec `json:"profilingConfig"`

	// AutoApply indicates whether to automatically create a DynamoGraphDeployment
	// after profiling completes. If false, only the spec is generated and stored in status.
	// Users can then manually create a DGD using the generated spec.
	// +kubebuilder:default=false
	AutoApply bool `json:"autoApply,omitempty"`

	// DeploymentOverrides allows customizing metadata for the auto-created DGD.
	// Only applicable when AutoApply is true.
	// +kubebuilder:validation:Optional
	DeploymentOverrides *DeploymentOverridesSpec `json:"deploymentOverrides,omitempty"`
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

	// Backend is extracted from profilingConfig.config.engine.backend for display purposes.
	// This field is populated by the controller and shown in kubectl output.
	// +kubebuilder:validation:Optional
	Backend string `json:"backend,omitempty"`

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
// +kubebuilder:printcolumn:name="Backend",type=string,JSONPath=`.status.backend`
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
