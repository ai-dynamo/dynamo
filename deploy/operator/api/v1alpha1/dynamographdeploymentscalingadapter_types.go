/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
)

// DynamoGraphDeploymentScalingAdapterSpec defines the desired state of DynamoGraphDeploymentScalingAdapter
type DynamoGraphDeploymentScalingAdapterSpec struct {
	// Replicas is the desired number of replicas for the target service.
	// This field is modified by external autoscalers (HPA/KEDA/Planner) or manually by users.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=0
	Replicas int32 `json:"replicas"`

	// DGDRef references the DynamoGraphDeployment and the specific service to scale.
	// +kubebuilder:validation:Required
	DGDRef DynamoGraphDeploymentServiceRef `json:"dgdRef"`
}

// DynamoGraphDeploymentServiceRef identifies a specific service within a DynamoGraphDeployment
type DynamoGraphDeploymentServiceRef struct {
	// Name of the DynamoGraphDeployment
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// ServiceName is the key name of the service within the DGD's spec.services map to scale
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	ServiceName string `json:"serviceName"`
}

// DynamoGraphDeploymentScalingAdapterStatus defines the observed state of DynamoGraphDeploymentScalingAdapter
type DynamoGraphDeploymentScalingAdapterStatus struct {
	// Replicas is the current number of replicas for the target service.
	// This is synced from the DGD's service replicas and is required for the scale subresource.
	// +optional
	Replicas int32 `json:"replicas,omitempty"`

	// Selector is a label selector string for the pods managed by this adapter.
	// Required for HPA compatibility via the scale subresource.
	// +optional
	Selector string `json:"selector,omitempty"`

	// LastScaleTime is the last time the adapter scaled the target service.
	// +optional
	LastScaleTime *metav1.Time `json:"lastScaleTime,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.replicas,selectorpath=.status.selector
// +kubebuilder:printcolumn:name="DGD",type="string",JSONPath=".spec.dgdRef.name",description="DynamoGraphDeployment name"
// +kubebuilder:printcolumn:name="SERVICE",type="string",JSONPath=".spec.dgdRef.serviceName",description="Service name"
// +kubebuilder:printcolumn:name="REPLICAS",type="integer",JSONPath=".status.replicas",description="Current replicas"
// +kubebuilder:printcolumn:name="AGE",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:resource:shortName={dgdsa}

// DynamoGraphDeploymentScalingAdapter provides a scaling interface for individual services
// within a DynamoGraphDeployment. It implements the Kubernetes scale
// subresource, enabling integration with HPA, KEDA, and custom autoscalers.
//
// The adapter acts as an intermediary between autoscalers and the DGD,
// ensuring that only the adapter controller modifies the DGD's service replicas.
// This prevents conflicts when multiple autoscaling mechanisms are in play.
type DynamoGraphDeploymentScalingAdapter struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoGraphDeploymentScalingAdapterSpec   `json:"spec,omitempty"`
	Status DynamoGraphDeploymentScalingAdapterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoGraphDeploymentScalingAdapterList contains a list of DynamoGraphDeploymentScalingAdapter
type DynamoGraphDeploymentScalingAdapterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeploymentScalingAdapter `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoGraphDeploymentScalingAdapter{}, &DynamoGraphDeploymentScalingAdapterList{})
}
