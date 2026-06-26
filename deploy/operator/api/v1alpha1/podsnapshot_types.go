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

// PodSnapshot and PodSnapshotContent status condition types. Both objects share this
// vocabulary; the operator and node agent set them via meta.SetStatusCondition.
const (
	// PodSnapshotConditionReady is True when capture and binding completed and the
	// artifact is usable for restore.
	PodSnapshotConditionReady = "Ready"
	// PodSnapshotConditionFailed is True when capture or binding failed terminally.
	PodSnapshotConditionFailed = "Failed"
)

// IsPodSnapshotSucceeded reports whether the PodSnapshot's Ready condition is True.
func IsPodSnapshotSucceeded(s *PodSnapshot) bool {
	return meta.IsStatusConditionTrue(s.Status.Conditions, PodSnapshotConditionReady)
}

// IsPodSnapshotFailed reports whether the PodSnapshot's Failed condition is True.
func IsPodSnapshotFailed(s *PodSnapshot) bool {
	return meta.IsStatusConditionTrue(s.Status.Conditions, PodSnapshotConditionFailed)
}

// PodSnapshotSpec defines the desired state of PodSnapshot.
type PodSnapshotSpec struct {
	// Source identifies the captured workload. It is a struct (rather than an
	// inlined reference) so future source variants can be added additively.
	// +kubebuilder:validation:Required
	Source PodSnapshotSource `json:"source"`
}

// PodSnapshotSource identifies the workload captured by a PodSnapshot.
type PodSnapshotSource struct {
	// PodRef references the pod, in the PodSnapshot's namespace, that is captured.
	// The operator prepares the pod (control volume, target-container annotation,
	// checkpoint storage mount) before creating the PodSnapshot.
	// +kubebuilder:validation:Required
	PodRef PodReference `json:"podRef"`
}

// PodReference names a pod in the same namespace as the referencing PodSnapshot.
type PodReference struct {
	// Name of the source pod.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// UID of the source pod, recorded so the node agent dumps that specific
	// pod and not a same-named recreation.
	// +optional
	UID types.UID `json:"uid,omitempty"`
}

// PodSnapshotStatus defines the observed state of PodSnapshot.
type PodSnapshotStatus struct {
	// BoundPodSnapshotContentName is the name of the cluster-scoped PodSnapshotContent
	// this PodSnapshot is bound to. It is nil until the agent has created the
	// content and recorded the binding.
	// +optional
	BoundPodSnapshotContentName *string `json:"boundSnapshotContentName,omitempty"`

	// Conditions reflect the latest observations of the PodSnapshot's state.
	// Standard types are Ready and Failed.
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=podsnap
// +kubebuilder:printcolumn:name="Content",type="string",JSONPath=".status.boundSnapshotContentName",description="Bound PodSnapshotContent"
// +kubebuilder:printcolumn:name="Ready",type="string",JSONPath=".status.conditions[?(@.type=='Ready')].status",description="Ready condition"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:validation:XValidation:rule="!has(oldSelf.spec) || self.spec == oldSelf.spec",message="spec is immutable"

// PodSnapshot is the Schema for the snapshots API. It is the namespaced binding
// for a captured container checkpoint and is consumed by restore paths.
//
// No conversion: this type exists only in v1alpha1 (no other API version), so it
// is not part of any conversion scheme.
type PodSnapshot struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   PodSnapshotSpec   `json:"spec,omitempty"`
	Status PodSnapshotStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// PodSnapshotList contains a list of PodSnapshot.
type PodSnapshotList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []PodSnapshot `json:"items"`
}
