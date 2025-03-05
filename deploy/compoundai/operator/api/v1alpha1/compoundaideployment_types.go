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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// CompoundAIDeploymentSpec defines the desired state of CompoundAIDeployment.
type CompoundAIDeploymentSpec struct {
	// required
	CompoundAINim string `json:"compoundAINim"`
	// optional
	// key is the name of the service defined in CompoundAINim
	// value is the CompoundAINimDeployment override for that service
	// if not set, the CompoundAINimDeployment will be used as is
	// +kubebuilder:validation:Optional
	Services map[string]*CompoundAINimDeployment `json:"services,omitempty"`
}

// CompoundAIDeploymentStatus defines the observed state of CompoundAIDeployment.
type CompoundAIDeploymentStatus struct {
	State      string             `json:"state,omitempty"`
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// CompoundAIDeployment is the Schema for the compoundaideployments API.
type CompoundAIDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   CompoundAIDeploymentSpec   `json:"spec,omitempty"`
	Status CompoundAIDeploymentStatus `json:"status,omitempty"`
}

func (s *CompoundAIDeployment) SetState(state string) {
	s.Status.State = state
}

// +kubebuilder:object:root=true

// CompoundAIDeploymentList contains a list of CompoundAIDeployment.
type CompoundAIDeploymentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []CompoundAIDeployment `json:"items"`
}

func init() {
	SchemeBuilder.Register(&CompoundAIDeployment{}, &CompoundAIDeploymentList{})
}
