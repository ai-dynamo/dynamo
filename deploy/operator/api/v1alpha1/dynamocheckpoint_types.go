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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// DynamoCheckpointPhase represents the current phase of the checkpoint lifecycle
// +kubebuilder:validation:Enum=Pending;Creating;Ready;Failed
type DynamoCheckpointPhase string

const (
	// DynamoCheckpointPhasePending indicates the checkpoint CR has been created but the Job has not started
	DynamoCheckpointPhasePending DynamoCheckpointPhase = "Pending"
	// DynamoCheckpointPhaseCreating indicates the checkpoint Job is running
	DynamoCheckpointPhaseCreating DynamoCheckpointPhase = "Creating"
	// DynamoCheckpointPhaseReady indicates the checkpoint tar file is available on the PVC
	DynamoCheckpointPhaseReady DynamoCheckpointPhase = "Ready"
	// DynamoCheckpointPhaseFailed indicates the checkpoint creation failed
	DynamoCheckpointPhaseFailed DynamoCheckpointPhase = "Failed"
)

// DynamoCheckpointStorageType defines the supported storage backends for checkpoints
// +kubebuilder:validation:Enum=pvc;s3;oci
type DynamoCheckpointStorageType string

// DynamoCheckpointIdentity defines the inputs that determine checkpoint equivalence
// Two checkpoints with the same identity hash are considered equivalent
type DynamoCheckpointIdentity struct {
	// Model is the model identifier (e.g., "meta-llama/Llama-3-70B")
	// +kubebuilder:validation:Required
	Model string `json:"model"`

	// BackendFramework is the runtime framework (vllm, sglang, trtllm)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Enum=vllm;sglang;trtllm
	BackendFramework string `json:"backendFramework"`

	// DynamoVersion is the Dynamo platform version (optional)
	// If not specified, version is not included in identity hash
	// This ensures checkpoint compatibility across Dynamo releases
	// +optional
	DynamoVersion string `json:"dynamoVersion,omitempty"`

	// TensorParallelSize is the tensor parallel configuration
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	TensorParallelSize int32 `json:"tensorParallelSize,omitempty"`

	// PipelineParallelSize is the pipeline parallel configuration
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	PipelineParallelSize int32 `json:"pipelineParallelSize,omitempty"`

	// Dtype is the data type (fp16, bf16, fp8, etc.)
	// +optional
	Dtype string `json:"dtype,omitempty"`

	// MaxModelLen is the maximum sequence length
	// +optional
	// +kubebuilder:validation:Minimum=1
	MaxModelLen int32 `json:"maxModelLen,omitempty"`

	// ExtraParameters are additional parameters that affect the checkpoint hash
	// Use for any framework-specific or custom parameters not covered above
	// +optional
	ExtraParameters map[string]string `json:"extraParameters,omitempty"`
}

// DynamoCheckpointJobConfig defines the configuration for the checkpoint creation Job
type DynamoCheckpointJobConfig struct {
	// PodTemplateSpec allows customizing the checkpoint Job pod
	// This should include the container that runs the workload to be checkpointed
	// +kubebuilder:validation:Required
	PodTemplateSpec corev1.PodTemplateSpec `json:"podTemplateSpec"`

	// ActiveDeadlineSeconds specifies the maximum time the Job can run
	// +optional
	// +kubebuilder:default=3600
	ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty"`

	// BackoffLimit specifies the number of retries before marking the Job failed
	// +optional
	// +kubebuilder:default=3
	BackoffLimit *int32 `json:"backoffLimit,omitempty"`

	// TTLSecondsAfterFinished specifies how long to keep the Job after completion
	// +optional
	// +kubebuilder:default=300
	TTLSecondsAfterFinished *int32 `json:"ttlSecondsAfterFinished,omitempty"`
}

// DynamoCheckpointSpec defines the desired state of DynamoCheckpoint
type DynamoCheckpointSpec struct {
	// Identity defines the inputs that determine checkpoint equivalence
	// +kubebuilder:validation:Required
	Identity DynamoCheckpointIdentity `json:"identity"`

	// Job defines the configuration for the checkpoint creation Job
	// +kubebuilder:validation:Required
	Job DynamoCheckpointJobConfig `json:"job"`
}

// DynamoCheckpointConditionType defines the types of conditions for DynamoCheckpoint
type DynamoCheckpointConditionType string

const (
	// DynamoCheckpointConditionJobCreated indicates whether the checkpoint Job has been created
	DynamoCheckpointConditionJobCreated DynamoCheckpointConditionType = "JobCreated"
	// DynamoCheckpointConditionJobCompleted indicates whether the checkpoint Job has completed
	DynamoCheckpointConditionJobCompleted DynamoCheckpointConditionType = "JobCompleted"
	// DynamoCheckpointConditionTarAvailable indicates whether the checkpoint tar file exists
	DynamoCheckpointConditionTarAvailable DynamoCheckpointConditionType = "TarAvailable"
)

// DynamoCheckpointStatus defines the observed state of DynamoCheckpoint
type DynamoCheckpointStatus struct {
	// Phase represents the current phase of the checkpoint lifecycle
	// +optional
	Phase DynamoCheckpointPhase `json:"phase,omitempty"`

	// IdentityHash is the computed hash of the checkpoint identity
	// This hash is used to identify equivalent checkpoints
	// +optional
	IdentityHash string `json:"identityHash,omitempty"`

	// Location is the full URI/path to the checkpoint in the storage backend
	// For PVC: same as TarPath (e.g., /checkpoints/{hash}.tar)
	// For S3: s3://bucket/prefix/{hash}.tar
	// For OCI: oci://registry/repo:{hash}
	// +optional
	Location string `json:"location,omitempty"`

	// StorageType indicates the storage backend type used for this checkpoint
	// +optional
	StorageType DynamoCheckpointStorageType `json:"storageType,omitempty"`

	// JobName is the name of the checkpoint creation Job
	// +optional
	JobName string `json:"jobName,omitempty"`

	// CreatedAt is the timestamp when the checkpoint tar was created
	// +optional
	CreatedAt *metav1.Time `json:"createdAt,omitempty"`

	// Message provides additional information about the current state
	// +optional
	Message string `json:"message,omitempty"`

	// Conditions represent the latest available observations of the checkpoint's state
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dckpt
// +kubebuilder:printcolumn:name="Model",type="string",JSONPath=".spec.identity.model",description="Model identifier"
// +kubebuilder:printcolumn:name="Backend",type="string",JSONPath=".spec.identity.backendFramework",description="Backend framework"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase",description="Current phase of the checkpoint"
// +kubebuilder:printcolumn:name="Hash",type="string",JSONPath=".status.identityHash",description="Identity hash of the checkpoint"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:validation:XValidation:rule="!has(oldSelf.spec.identity) || self.spec.identity == oldSelf.spec.identity",message="spec.identity is immutable after creation"

// DynamoCheckpoint is the Schema for the dynamocheckpoints API
// It represents a container checkpoint that can be used to restore pods to a warm state
type DynamoCheckpoint struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoCheckpointSpec   `json:"spec,omitempty"`
	Status DynamoCheckpointStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoCheckpointList contains a list of DynamoCheckpoint
type DynamoCheckpointList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoCheckpoint `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoCheckpoint{}, &DynamoCheckpointList{})
}
