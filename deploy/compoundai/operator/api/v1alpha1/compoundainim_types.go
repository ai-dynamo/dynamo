/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// CompoundAINimSpec defines the desired state of CompoundAINim
type CompoundAINimSpec struct {
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

// CompoundAINimStatus defines the observed state of CompoundAINim
type CompoundAINimStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Ready bool `json:"ready"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// CompoundAINim is the Schema for the compoundainims API
type CompoundAINim struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   CompoundAINimSpec   `json:"spec,omitempty"`
	Status CompoundAINimStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// CompoundAINimList contains a list of CompoundAINim
type CompoundAINimList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []CompoundAINim `json:"items"`
}

func init() {
	SchemeBuilder.Register(&CompoundAINim{}, &CompoundAINimList{})
}
