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
	"strings"

	dynamoCommon "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	DynamoGraphDeploymentConditionTypeAvailable            = "Available"
	DynamoGraphDeploymentConditionTypeDynamoComponentReady = "DynamoComponentReady"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// DynamoComponentDeploymentSpec defines the desired state of DynamoComponentDeployment
type DynamoComponentDeploymentSpec struct {
	DynamoComponent string `json:"dynamoComponent"`
	// contains the tag of the DynamoComponent: for example, "my_package:MyService"
	DynamoTag string `json:"dynamoTag"`

	DynamoComponentDeploymentSharedSpec `json:",inline"`
}

type DynamoComponentDeploymentOverridesSpec struct {
	DynamoComponentDeploymentSharedSpec `json:",inline"`
}

type DynamoComponentDeploymentSharedSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	Annotations map[string]string `json:"annotations,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`

	// contains the name of the service
	ServiceName string `json:"serviceName,omitempty"`

	// dynamo namespace of the service (allows to override the dynamo namespace of the service defined in annotations inside the dynamo archive)
	DynamoNamespace *string `json:"dynamoNamespace,omitempty"`

	Resources        *dynamoCommon.Resources    `json:"resources,omitempty"`
	Autoscaling      *Autoscaling               `json:"autoscaling,omitempty"`
	Envs             []corev1.EnvVar            `json:"envs,omitempty"`
	EnvFromSecret    *string                    `json:"envFromSecret,omitempty"`
	PVC              *PVC                       `json:"pvc,omitempty"`
	RunMode          *RunMode                   `json:"runMode,omitempty"`
	ExternalServices map[string]ExternalService `json:"externalServices,omitempty"`

	Ingress IngressSpec `json:"ingress,omitempty"`

	// +optional
	ExtraPodMetadata *dynamoCommon.ExtraPodMetadata `json:"extraPodMetadata,omitempty"`
	// +optional
	ExtraPodSpec *dynamoCommon.ExtraPodSpec `json:"extraPodSpec,omitempty"`

	LivenessProbe  *corev1.Probe `json:"livenessProbe,omitempty"`
	ReadinessProbe *corev1.Probe `json:"readinessProbe,omitempty"`
	Replicas       *int32        `json:"replicas,omitempty"`
}

type RunMode struct {
	Standalone *bool `json:"standalone,omitempty"`
}

type ExternalService struct {
	DeploymentSelectorKey   string `json:"deploymentSelectorKey,omitempty"`
	DeploymentSelectorValue string `json:"deploymentSelectorValue,omitempty"`
}

type IngressTLSSpec struct {
	SecretName string `json:"secretName,omitempty"`
}

type IngressSpec struct {
	Enabled                    bool              `json:"enabled,omitempty"`
	Host                       string            `json:"host,omitempty"`
	UseVirtualService          bool              `json:"useVirtualService,omitempty"`
	VirtualServiceGateway      *string           `json:"virtualServiceGateway,omitempty"`
	HostPrefix                 *string           `json:"hostPrefix,omitempty"`
	Annotations                map[string]string `json:"annotations,omitempty"`
	Labels                     map[string]string `json:"labels,omitempty"`
	TLS                        *IngressTLSSpec   `json:"tls,omitempty"`
	HostSuffix                 *string           `json:"hostSuffix,omitempty"`
	IngressControllerClassName *string           `json:"ingressControllerClassName,omitempty"`
}

// DynamoComponentDeploymentStatus defines the observed state of DynamoComponentDeployment
type DynamoComponentDeploymentStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Conditions []metav1.Condition `json:"conditions"`

	PodSelector map[string]string `json:"podSelector,omitempty"`
}

// +genclient
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:printcolumn:name="DynamoComponent",type="string",JSONPath=".spec.dynamoComponent",description="Dynamo component"
// +kubebuilder:printcolumn:name="Available",type="string",JSONPath=".status.conditions[?(@.type=='Available')].status",description="Available"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:resource:shortName=dcd
// DynamoComponentDeployment is the Schema for the dynamocomponentdeployments API
type DynamoComponentDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoComponentDeploymentSpec   `json:"spec,omitempty"`
	Status DynamoComponentDeploymentStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoComponentDeploymentList contains a list of DynamoComponentDeployment
type DynamoComponentDeploymentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoComponentDeployment `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoComponentDeployment{}, &DynamoComponentDeploymentList{})
}

func (s *DynamoComponentDeploymentStatus) IsReady() bool {
	for _, condition := range s.Conditions {
		if condition.Type == DynamoGraphDeploymentConditionTypeAvailable && condition.Status == metav1.ConditionTrue {
			return true
		}
	}
	return false
}

func (s *DynamoComponentDeployment) GetSpec() any {
	return s.Spec
}

func (s *DynamoComponentDeployment) SetSpec(spec any) {
	s.Spec = spec.(DynamoComponentDeploymentSpec)
}

func (s *DynamoComponentDeployment) IsMainComponent() bool {
	return strings.HasSuffix(s.Spec.DynamoTag, s.Spec.ServiceName)
}
