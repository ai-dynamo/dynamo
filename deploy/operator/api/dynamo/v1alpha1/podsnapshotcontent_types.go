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
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// IsPodSnapshotContentSucceeded reports whether the PodSnapshotContent's Ready condition is True.
func IsPodSnapshotContentSucceeded(c *PodSnapshotContent) bool {
	return meta.IsStatusConditionTrue(c.Status.Conditions, PodSnapshotConditionReady)
}

// IsPodSnapshotContentFailed reports whether the PodSnapshotContent's Failed condition is True.
func IsPodSnapshotContentFailed(c *PodSnapshotContent) bool {
	return meta.IsStatusConditionTrue(c.Status.Conditions, PodSnapshotConditionFailed)
}

// PodSnapshotContentSpec defines the desired state of PodSnapshotContent. It is
// populated by the PodSnapshotReconciler (operator) at creation time and is
// immutable thereafter.
type PodSnapshotContentSpec struct {
	// PodSnapshotRef is the back-pointer to the bound PodSnapshot. It may span
	// namespaces because PodSnapshotContent is cluster-scoped.
	// +kubebuilder:validation:Required
	PodSnapshotRef PodSnapshotReference `json:"snapshotRef"`

	// Source describes what to capture: the source pod and the node it runs on.
	// +kubebuilder:validation:Required
	Source PodSnapshotContentSource `json:"source"`
}

// PodSnapshotReference is a cross-namespace reference to a PodSnapshot.
type PodSnapshotReference struct {
	// Namespace of the referenced PodSnapshot.
	// +kubebuilder:validation:Required
	Namespace string `json:"namespace"`

	// Name of the referenced PodSnapshot.
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// UID of the referenced PodSnapshot, recorded at binding time to detect a
	// stale reference after a delete and recreate.
	// +optional
	UID types.UID `json:"uid,omitempty"`
}

// PodSnapshotContentSource is the immutable source descriptor: what to dump
// (PodRef) and where it runs (NodeName).
type PodSnapshotContentSource struct {
	// PodRef identifies the pod to dump. Its UID guards against dumping a
	// same-named recreation of the pod.
	// +kubebuilder:validation:Required
	PodRef PodReference `json:"podRef"`

	// NodeName is the node the source pod runs on, denormalized from the live
	// pod so it travels with PodRef as one immutable unit and selects the node
	// agent that performs the dump.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	NodeName string `json:"nodeName"`
}

// PodSnapshotContentStatus defines the observed state of PodSnapshotContent.
type PodSnapshotContentStatus struct {
	// Conditions reflect the latest observations of the PodSnapshotContent's state.
	// Standard types are Ready and Failed.
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster,shortName=podsnapcontent
// +kubebuilder:printcolumn:name="PodSnapshot",type="string",JSONPath=".spec.snapshotRef.name",description="Bound PodSnapshot"
// +kubebuilder:printcolumn:name="Namespace",type="string",JSONPath=".spec.snapshotRef.namespace",description="PodSnapshot namespace"
// +kubebuilder:printcolumn:name="Ready",type="string",JSONPath=".status.conditions[?(@.type=='Ready')].status",description="Ready condition"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:validation:XValidation:rule="!has(oldSelf.spec) || self.spec == oldSelf.spec",message="spec is immutable"

// PodSnapshotContent is the Schema for the snapshotcontents API. It is the
// cluster-scoped artifact-of-record for a captured container checkpoint.
//
// No conversion: this type exists only in v1alpha1 (no other API version), so it
// is not part of any conversion scheme.
type PodSnapshotContent struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   PodSnapshotContentSpec   `json:"spec,omitempty"`
	Status PodSnapshotContentStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// PodSnapshotContentList contains a list of PodSnapshotContent.
type PodSnapshotContentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []PodSnapshotContent `json:"items"`
}

func init() {
	SchemeBuilder.Register(&PodSnapshotContent{}, &PodSnapshotContentList{})
}
