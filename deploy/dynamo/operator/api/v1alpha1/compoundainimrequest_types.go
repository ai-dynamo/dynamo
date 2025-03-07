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
	dynamoCommon "github.com/dynemo-ai/dynemo/deploy/dynamo/operator/api/dynamo/common"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/operator/api/dynamo/modelschemas"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	CompoundAINimRequestConditionTypeModelsSeeding          = "ModelsSeeding"
	CompoundAINimRequestConditionTypeImageBuilding          = "ImageBuilding"
	CompoundAINimRequestConditionTypeImageExists            = "ImageExists"
	CompoundAINimRequestConditionTypeImageExistsChecked     = "ImageExistsChecked"
	CompoundAINimRequestConditionTypeModelsExists           = "ModelsExists"
	CompoundAINimRequestConditionTypeCompoundAINimAvailable = "CompoundAINimAvailable"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// CompoundAINimRequestSpec defines the desired state of CompoundAINimRequest
type CompoundAINimRequestSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// +kubebuilder:validation:Required
	BentoTag    string        `json:"bentoTag"`
	DownloadURL string        `json:"downloadUrl,omitempty"`
	ServiceName string        `json:"serviceName,omitempty"`
	Context     *BentoContext `json:"context,omitempty"`
	Models      []BentoModel  `json:"models,omitempty"`

	// +kubebuilder:validation:Optional
	Image string `json:"image,omitempty"`

	ImageBuildTimeout *modelschemas.Duration `json:"imageBuildTimeout,omitempty"`

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

// CompoundAINimRequestStatus defines the observed state of CompoundAINimRequest
type CompoundAINimRequestStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Conditions []metav1.Condition `json:"conditions"`
}

//+genclient
//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
//+kubebuilder:printcolumn:name="Bento-Tag",type="string",JSONPath=".spec.bentoTag",description="Bento Tag"
//+kubebuilder:printcolumn:name="Download-Url",type="string",JSONPath=".spec.downloadUrl",description="Download URL"
//+kubebuilder:printcolumn:name="Image",type="string",JSONPath=".spec.image",description="Image"
//+kubebuilder:printcolumn:name="Image-Exists",type="string",JSONPath=".status.conditions[?(@.type=='ImageExists')].status",description="Image Exists"
//+kubebuilder:printcolumn:name="Bento-Available",type="string",JSONPath=".status.conditions[?(@.type=='BentoAvailable')].status",description="Bento Available"
//+kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// CompoundAINimRequest is the Schema for the dynamonimrequests API
type CompoundAINimRequest struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   CompoundAINimRequestSpec   `json:"spec,omitempty"`
	Status CompoundAINimRequestStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// CompoundAINimRequestList contains a list of CompoundAINimRequest
type CompoundAINimRequestList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []CompoundAINimRequest `json:"items"`
}

func init() {
	SchemeBuilder.Register(&CompoundAINimRequest{}, &CompoundAINimRequestList{})
}
