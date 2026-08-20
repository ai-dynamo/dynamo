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
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// DynamoGraphDeploymentCandidateStatus describes simulation and materialization,
// never deployment health.
type DynamoGraphDeploymentCandidateStatus struct {
	// Rank is the one-based scalar ordering and is absent for Pareto searches.
	// +optional
	// +kubebuilder:validation:Minimum=1
	Rank *int32 `json:"rank,omitempty"`

	// Conditions describes evaluation and materialization.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// Experimental contains Sweeper-version-specific diagnostics and round-trips
	// without a nested CRD schema.
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	Experimental *runtime.RawExtension `json:"experimental,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dgdc
// +kubebuilder:validation:XValidation:rule="!has(oldSelf.spec) || self.spec == oldSelf.spec",message="spec is immutable"
// +kubebuilder:printcolumn:name="Rank",type=integer,JSONPath=`.status.rank`
// +kubebuilder:printcolumn:name="Evaluated",type=string,JSONPath=`.status.conditions[?(@.type=="Evaluated")].status`
// +kubebuilder:printcolumn:name="Backend",type=string,JSONPath=`.spec.backendFramework`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// DynamoGraphDeploymentCandidate is one bounded, user-visible search result.
// Its spec is exactly the v1beta1 DynamoGraphDeploymentSpec schema.
type DynamoGraphDeploymentCandidate struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   v1beta1.DynamoGraphDeploymentSpec    `json:"spec"`
	Status DynamoGraphDeploymentCandidateStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoGraphDeploymentCandidateList contains a list of DynamoGraphDeploymentCandidate resources.
type DynamoGraphDeploymentCandidateList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeploymentCandidate `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoGraphDeploymentCandidate{}, &DynamoGraphDeploymentCandidateList{})
}
