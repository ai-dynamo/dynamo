/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package v1beta2

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// DGDCConditionValid reports whether the pre-defaulting DGD passed strict server-side dry-run.
const DGDCConditionValid = "Valid"

// DynamoGraphDeploymentCandidateStatus describes evaluation, never deployment health.
type DynamoGraphDeploymentCandidateStatus struct {
	// Parameters are the original resolved evaluation inputs. Overrides do not change them.
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	Parameters *runtime.RawExtension `json:"parameters,omitempty"`

	// Metrics are the resolved evaluation metrics for this point.
	// They describe the original evaluated configuration, not a deployment changed by overrides.
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	Metrics *runtime.RawExtension `json:"metrics,omitempty"`

	// Conditions records the final Valid result. Valid=True means atomic override application
	// and strict DGD dry-run succeeded at checking time, not that the deployment is Ready.
	// Invalid Candidates remain visible and count toward recommendation.maxCandidates.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dgdc
// +kubebuilder:printcolumn:name="Backend",type=string,JSONPath=`.spec.backendFramework`
// +kubebuilder:printcolumn:name="Valid",type=string,JSONPath=`.status.conditions[?(@.type=="Valid")].status`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// DynamoGraphDeploymentCandidate is one bounded, user-visible search result.
type DynamoGraphDeploymentCandidate struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec is exactly the unpruned DGD spec, before dry-run defaulting or webhook mutation.
	// It is opaque so invalid field types and unknown fields can be retained with Valid=False.
	// A failed override leaves the generated spec unchanged. Immutability is enforced by admission
	// because CEL cannot inspect fields in an opaque object.
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	Spec runtime.RawExtension `json:"spec"`

	Status DynamoGraphDeploymentCandidateStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoGraphDeploymentCandidateList contains a list of DynamoGraphDeploymentCandidate resources.
type DynamoGraphDeploymentCandidateList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeploymentCandidate `json:"items"`
}
