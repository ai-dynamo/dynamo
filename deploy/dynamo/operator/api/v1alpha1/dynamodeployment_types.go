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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// DynamoDeploymentSpec defines the desired state of DynamoDeployment.
type DynamoDeploymentSpec struct {
	// required
	DynamoNim string `json:"dynamoNim"`
	// optional
	// key is the name of the service defined in DynamoNim
	// value is the DynamoNimDeployment override for that service
	// if not set, the DynamoNimDeployment will be used as is
	// +kubebuilder:validation:Optional
	Services map[string]*DynamoNimDeployment `json:"services,omitempty"`
}

// DynamoDeploymentStatus defines the observed state of DynamoDeployment.
type DynamoDeploymentStatus struct {
	State      string             `json:"state,omitempty"`
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// DynamoDeployment is the Schema for the dynamodeployments API.
type DynamoDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoDeploymentSpec   `json:"spec,omitempty"`
	Status DynamoDeploymentStatus `json:"status,omitempty"`
}

func (s *DynamoDeployment) SetState(state string) {
	s.Status.State = state
}

// +kubebuilder:object:root=true

// DynamoDeploymentList contains a list of DynamoDeployment.
type DynamoDeploymentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoDeployment `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoDeployment{}, &DynamoDeploymentList{})
}

func (s *DynamoDeployment) GetSpec() any {
	return s.Spec
}

func (s *DynamoDeployment) SetSpec(spec any) {
	s.Spec = spec.(DynamoDeploymentSpec)
}
