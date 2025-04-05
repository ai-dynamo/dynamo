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
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// DynamoNimSpec defines the desired state of DynamoNim
type DynamoNimSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// +kubebuilder:validation:Required
	Tag string `json:"tag"`
	// +kubebuilder:validation:Required
	Image       string        `json:"image"`
	ServiceName string        `json:"serviceName,omitempty"`
	Context     *BentoContext `json:"context,omitempty"`
	Models      []BentoModel  `json:"models,omitempty"`

	ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`
}

type BentoContext struct {
	BentomlVersion string `json:"bentomlVersion,omitempty"`
}

type BentoModel struct {
	// +kubebuilder:validation:Required
	Tag         string             `json:"tag"`
	DownloadURL string             `json:"downloadUrl,omitempty"`
	Size        *resource.Quantity `json:"size,omitempty"`
}

// DynamoNimStatus defines the observed state of DynamoNim
type DynamoNimStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Ready bool `json:"ready"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// DynamoNim is the Schema for the dynamonims API
type DynamoNim struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoNimSpec   `json:"spec,omitempty"`
	Status DynamoNimStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoNimList contains a list of DynamoNim
type DynamoNimList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoNim `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoNim{}, &DynamoNimList{})
}
