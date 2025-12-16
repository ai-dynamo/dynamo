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

const (
	// DynamoCheckpointStorageTypePVC uses a PersistentVolumeClaim for storage
	DynamoCheckpointStorageTypePVC DynamoCheckpointStorageType = "pvc"
	// DynamoCheckpointStorageTypeS3 uses S3-compatible object storage
	DynamoCheckpointStorageTypeS3 DynamoCheckpointStorageType = "s3"
	// DynamoCheckpointStorageTypeOCI uses an OCI registry for storage
	DynamoCheckpointStorageTypeOCI DynamoCheckpointStorageType = "oci"
)

// DynamoCheckpointStorageConfig defines the storage backend configuration
type DynamoCheckpointStorageConfig struct {
	// Type is the storage backend type (pvc, s3, oci)
	// +kubebuilder:validation:Required
	// +kubebuilder:default=pvc
	Type DynamoCheckpointStorageType `json:"type"`

	// PVC configuration (used when type=pvc)
	// +optional
	PVC *DynamoCheckpointPVCConfig `json:"pvc,omitempty"`

	// S3 configuration (used when type=s3)
	// +optional
	S3 *DynamoCheckpointS3Config `json:"s3,omitempty"`

	// OCI configuration (used when type=oci)
	// +optional
	OCI *DynamoCheckpointOCIConfig `json:"oci,omitempty"`
}

// DynamoCheckpointPVCConfig defines PVC storage configuration
type DynamoCheckpointPVCConfig struct {
	// PVCName is the name of the PersistentVolumeClaim to use
	// +kubebuilder:validation:Required
	// +kubebuilder:default="checkpoint-storage"
	PVCName string `json:"pvcName"`

	// BasePath is the base directory within the PVC for storing checkpoints
	// +optional
	// +kubebuilder:default="/checkpoints"
	BasePath string `json:"basePath,omitempty"`
}

// DynamoCheckpointS3Config defines S3 storage configuration
type DynamoCheckpointS3Config struct {
	// URI is the S3 location in format: s3://[endpoint/]bucket/prefix
	// Examples:
	//   - s3://my-bucket/checkpoints (AWS S3)
	//   - s3://minio.example.com/my-bucket/checkpoints (MinIO/custom endpoint)
	// +kubebuilder:validation:Required
	URI string `json:"uri"`

	// CredentialsSecretRef is a reference to a secret containing S3 credentials
	// The secret should contain AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and optionally AWS_REGION
	// If not provided, uses IRSA/Workload Identity
	// +optional
	CredentialsSecretRef string `json:"credentialsSecretRef,omitempty"`
}

// DynamoCheckpointOCIConfig defines OCI registry storage configuration
type DynamoCheckpointOCIConfig struct {
	// URI is the OCI location in format: oci://registry/repository
	// Examples:
	//   - oci://myregistry.io/checkpoints
	//   - oci://ghcr.io/myorg/checkpoints
	// +kubebuilder:validation:Required
	URI string `json:"uri"`

	// CredentialsSecretRef is a reference to a docker config secret for registry auth
	// +optional
	CredentialsSecretRef string `json:"credentialsSecretRef,omitempty"`
}

// DynamoCheckpointIdentity defines the inputs that determine checkpoint equivalence
// Two checkpoints with the same identity hash are considered equivalent
type DynamoCheckpointIdentity struct {
	// Model is the model identifier (e.g., "meta-llama/Llama-3-70B")
	// +kubebuilder:validation:Required
	Model string `json:"model"`

	// Framework is the runtime framework (vllm, sglang, trtllm)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Enum=vllm;sglang;trtllm
	Framework string `json:"framework"`

	// FrameworkVersion is the version of the framework (optional)
	// If not specified, version is not included in identity hash
	// +optional
	FrameworkVersion string `json:"frameworkVersion,omitempty"`

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

	// TarPath is the local path to the checkpoint tar file
	// For PVC: the mount path (e.g., /checkpoints/{hash}.tar)
	// For S3/OCI: the local path after download (e.g., /tmp/{hash}.tar)
	// +optional
	TarPath string `json:"tarPath,omitempty"`

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
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase",description="Current phase of the checkpoint"
// +kubebuilder:printcolumn:name="Hash",type="string",JSONPath=".status.identityHash",description="Identity hash of the checkpoint"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

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
