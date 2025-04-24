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
	dynamoCommon "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/schemas"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	DynamoComponentRequestConditionTypeModelsSeeding            = "ModelsSeeding"
	DynamoComponentRequestConditionTypeImageBuilding            = "ImageBuilding"
	DynamoComponentRequestConditionTypeImageExists              = "ImageExists"
	DynamoComponentRequestConditionTypeImageExistsChecked       = "ImageExistsChecked"
	DynamoComponentRequestConditionTypeModelsExists             = "ModelsExists"
	DynamoComponentRequestConditionTypeDynamoComponentAvailable = "DynamoComponentAvailable"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// DynamoComponentRequestSpec defines the desired state of DynamoComponentRequest
type DynamoComponentRequestSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// +kubebuilder:validation:Required
	DynamoComponent string `json:"dynamoComponent"`
	DownloadURL     string `json:"downloadUrl,omitempty"`
	ServiceName     string `json:"serviceName,omitempty"`

	// +kubebuilder:validation:Optional
	Image string `json:"image,omitempty"`

	ImageBuildTimeout *schemas.Duration `json:"imageBuildTimeout,omitempty"`

	// +kubebuilder:validation:Optional
	BuildArgs []string `json:"buildArgs,omitempty"`

	// +kubebuilder:validation:Optional
	ImageBuilderExtraPodMetadata *dynamoCommon.ExtraPodMetadata `json:"imageBuilderExtraPodMetadata,omitempty"`
	// +kubebuilder:validation:Optional
	ImageBuilderExtraPodSpec *dynamoCommon.ExtraPodSpec `json:"imageBuilderExtraPodSpec,omitempty"`
	// +kubebuilder:validation:Optional
	ImageBuilderExtraContainerEnv []corev1.EnvVar `json:"imageBuilderExtraContainerEnv,omitempty"`
	// +kubebuilder:validation:Optional
	ImageBuilderContainerResources *corev1.ResourceRequirements `json:"imageBuilderContainerResources,omitempty"`

	// +kubebuilder:validation:Optional
	DockerConfigJSONSecretName string `json:"dockerConfigJsonSecretName,omitempty"`

	// +kubebuilder:validation:Optional
	OCIRegistryInsecure *bool `json:"ociRegistryInsecure,omitempty"`

	// +kubebuilder:validation:Optional
	DownloaderContainerEnvFrom []corev1.EnvFromSource `json:"downloaderContainerEnvFrom,omitempty"`
}

// DynamoComponentRequestStatus defines the observed state of DynamoComponentRequest
type DynamoComponentRequestStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Conditions []metav1.Condition `json:"conditions"`
}

//+genclient
//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
//+kubebuilder:printcolumn:name="DynamoComponent",type="string",JSONPath=".spec.dynamoComponent",description="Dynamo component"
//+kubebuilder:printcolumn:name="Download-Url",type="string",JSONPath=".spec.downloadUrl",description="Download URL"
//+kubebuilder:printcolumn:name="Image-Exists",type="string",JSONPath=".status.conditions[?(@.type=='ImageExists')].status",description="Image Exists"
//+kubebuilder:printcolumn:name="DynamoComponent-Available",type="string",JSONPath=".status.conditions[?(@.type=='DynamoComponentAvailable')].status",description="Dynamo Component Available"
//+kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// DynamoComponentRequest is the Schema for the dynamocomponentrequests API
type DynamoComponentRequest struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoComponentRequestSpec   `json:"spec,omitempty"`
	Status DynamoComponentRequestStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// DynamoComponentRequestList contains a list of DynamoComponentRequest
type DynamoComponentRequestList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoComponentRequest `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoComponentRequest{}, &DynamoComponentRequestList{})
}

func (s *DynamoComponentRequest) GetSpec() any {
	return s.Spec
}

func (s *DynamoComponentRequest) SetSpec(spec any) {
	s.Spec = spec.(DynamoComponentRequestSpec)
}
