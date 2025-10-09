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
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// DynamoModelSpec defines the desired state of DynamoModel.
type DynamoModelSpec struct {
	// Name is the canonical model name (matches external model repo, e.g. HuggingFace, NGC).
	// Example: "meta-llama/Llama-3.3-70B-Instruct"
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// Version is a version pin (e.g., SHA or tag from source repository).
	// This solves version drift by pinning deployments and benchmarking jobs to the same model artifact.
	// +kubebuilder:validation:Optional
	Version string `json:"version,omitempty"`

	// SourceURL is the source location of model weights (can be HF, S3, NGC).
	// Ensures flexibility in downstream storage strategies; permits flexible source management and credential injection.
	// Examples: "hf://meta-llama/Llama-3.3-70B-Instruct", "s3://bucket/path/to/model", "ngc://nvidia/model"
	// +kubebuilder:validation:Required
	SourceURL string `json:"sourceURL"`

	// SecretRef is an optional reference to a secret needed for accessing the source URL
	// (private repo, S3 credentials, etc.)
	// +kubebuilder:validation:Optional
	SecretRef string `json:"secretRef,omitempty"`

	// DownloaderRef is an optional reference to a custom downloader or workflow
	// (e.g., MLFlow or internal tools). Provides extensibility for specialized workflows
	// (internal or third-party).
	// +kubebuilder:validation:Optional
	DownloaderRef string `json:"downloaderRef,omitempty"`

	// PVC defines the persistent volume claim configuration for storing the model.
	// +kubebuilder:validation:Required
	PVC PVCSpec `json:"pvc"`
}

// PVCSpec defines the PVC configuration for model storage.
type PVCSpec struct {
	// Create indicates whether to create a new PVC or use an existing one.
	// +kubebuilder:default=true
	Create *bool `json:"create,omitempty"`

	// Name is the name of the PVC. If not specified, defaults to the DynamoModel name.
	// +kubebuilder:validation:Optional
	Name string `json:"name,omitempty"`

	// StorageClass to be used for PVC creation. Required when create is true.
	// +kubebuilder:validation:Optional
	StorageClass string `json:"storageClass,omitempty"`

	// Size of the volume, used during PVC creation. Required when create is true.
	// +kubebuilder:validation:Optional
	Size resource.Quantity `json:"size,omitempty"`

	// VolumeAccessMode is the volume access mode of the PVC. Defaults to ReadWriteMany.
	// +kubebuilder:default=ReadWriteMany
	// +kubebuilder:validation:Optional
	VolumeAccessMode corev1.PersistentVolumeAccessMode `json:"volumeAccessMode,omitempty"`
}

// DynamoModelStatus defines the observed state of DynamoModel.
type DynamoModelStatus struct {
	// State is a high-level textual status of the model lifecycle.
	// Possible values: "Pending", "Downloading", "Ready", "Failed"
	State string `json:"state,omitempty"`

	// Conditions contains the latest observed conditions of the model.
	// +kubebuilder:validation:Optional
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

	// PVCName is the name of the PVC created or used for this model.
	// +kubebuilder:validation:Optional
	PVCName string `json:"pvcName,omitempty"`

	// DownloadJobName is the name of the Job created to download the model.
	// +kubebuilder:validation:Optional
	DownloadJobName string `json:"downloadJobName,omitempty"`

	// LastDownloadTime is the timestamp of the last successful download.
	// +kubebuilder:validation:Optional
	LastDownloadTime *metav1.Time `json:"lastDownloadTime,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dm
// +kubebuilder:printcolumn:name="State",type=string,JSONPath=`.status.state`
// +kubebuilder:printcolumn:name="Model",type=string,JSONPath=`.spec.name`
// +kubebuilder:printcolumn:name="Version",type=string,JSONPath=`.spec.version`
// +kubebuilder:printcolumn:name="PVC",type=string,JSONPath=`.status.pvcName`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
// DynamoModel is the Schema for the dynamomodels API.
// It provides a high-level abstraction for managing model artifacts cached in PVCs in the cluster.
// All jobs referencing the same DynamoModel are guaranteed to use the same artifact,
// preventing drift and simplifying maintenance.
type DynamoModel struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired state for this model.
	Spec DynamoModelSpec `json:"spec,omitempty"`
	// Status reflects the current observed state of this model.
	Status DynamoModelStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoModelList contains a list of DynamoModel.
type DynamoModelList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoModel `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoModel{}, &DynamoModelList{})
}

// SetState sets the state of the DynamoModel.
func (m *DynamoModel) SetState(state string) {
	m.Status.State = state
}

// GetSpec returns the spec of the DynamoModel.
func (m *DynamoModel) GetSpec() any {
	return m.Spec
}

// SetSpec sets the spec of the DynamoModel.
func (m *DynamoModel) SetSpec(spec any) {
	m.Spec = spec.(DynamoModelSpec)
}

// AddStatusCondition adds or updates a status condition.
func (m *DynamoModel) AddStatusCondition(condition metav1.Condition) {
	if m.Status.Conditions == nil {
		m.Status.Conditions = []metav1.Condition{}
	}
	// Check if condition with same type already exists
	for i, existingCondition := range m.Status.Conditions {
		if existingCondition.Type == condition.Type {
			// Replace the existing condition
			m.Status.Conditions[i] = condition
			return
		}
	}
	// If no matching condition found, append the new one
	m.Status.Conditions = append(m.Status.Conditions, condition)
}

// IsReady returns true if the model is in Ready state.
func (m *DynamoModel) IsReady() bool {
	return m.Status.State == "Ready"
}

// GetPVCName returns the PVC name for this model.
func (m *DynamoModel) GetPVCName() string {
	if m.Status.PVCName != "" {
		return m.Status.PVCName
	}
	if m.Spec.PVC.Name != "" {
		return m.Spec.PVC.Name
	}
	return m.Name
}
