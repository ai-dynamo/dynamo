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
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	DynamoComponentConditionTypeImageBuilding            = "ImageBuilding"
	DynamoComponentConditionTypeImageExists              = "ImageExists"
	DynamoComponentConditionTypeImageExistsChecked       = "ImageExistsChecked"
	DynamoComponentConditionTypeModelsExists             = "ModelsExists"
	DynamoComponentConditionTypeDynamoComponentAvailable = "DynamoComponentAvailable"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// DynamoComponentSpec defines the desired state of DynamoComponent
type DynamoComponentSpec struct {
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
	ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`

	// +kubebuilder:validation:Optional
	DockerConfigJSONSecretName string `json:"dockerConfigJsonSecretName,omitempty"`

	// +kubebuilder:validation:Optional
	DownloaderContainerEnvFrom []corev1.EnvFromSource `json:"downloaderContainerEnvFrom,omitempty"`
}

// DynamoComponentStatus defines the observed state of DynamoComponent
type DynamoComponentStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Conditions []metav1.Condition `json:"conditions"`
}

// +genclient
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="DynamoComponent",type="string",JSONPath=".spec.dynamoComponent",description="Dynamo component"
// +kubebuilder:printcolumn:name="Image-Exists",type="string",JSONPath=".status.conditions[?(@.type=='ImageExists')].status",description="Image Exists"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:resource:shortName=dc
// DynamoComponent is the Schema for the dynamocomponents API
type DynamoComponent struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoComponentSpec   `json:"spec,omitempty"`
	Status DynamoComponentStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// DynamoComponentList contains a list of DynamoComponent
type DynamoComponentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoComponent `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoComponent{}, &DynamoComponentList{})
}

func (s *DynamoComponent) GetSpec() any {
	return s.Spec
}

func (s *DynamoComponent) SetSpec(spec any) {
	s.Spec = spec.(DynamoComponentSpec)
}

func (s *DynamoComponent) IsReady() bool {
	return meta.IsStatusConditionTrue(s.Status.Conditions, DynamoComponentConditionTypeDynamoComponentAvailable)
}

// GetImage returns the docker image of the DynamoComponent
func (s *DynamoComponent) GetImage() string {
	if s.Spec.Image != "" {
		// if the image is specified in the spec, return it
		return s.Spec.Image
	}
	// if the image is not specified in the spec, the image is stored in the status condition ImageExists
	if meta.FindStatusCondition(s.Status.Conditions, DynamoComponentConditionTypeImageExists) != nil {
		return meta.FindStatusCondition(s.Status.Conditions, DynamoComponentConditionTypeImageExists).Message
	}
	return ""
}
