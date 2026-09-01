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

package v1beta1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// DynamoGraphDeploymentScalingAdapterSpec defines the desired state of a
// DynamoGraphDeploymentScalingAdapter.
type DynamoGraphDeploymentScalingAdapterSpec struct {
	// replicas is the desired number of replicas for the target component or
	// component group.
	// This field is modified by external autoscalers (HPA/KEDA/Planner) or
	// manually by users.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=0
	Replicas int32 `json:"replicas"`

	// dgdRef references the DynamoGraphDeployment and the specific scale target.
	// +kubebuilder:validation:Required
	DGDRef DynamoGraphDeploymentComponentRef `json:"dgdRef"`
}

// DynamoGraphDeploymentComponentRef identifies a specific scale target within
// a DynamoGraphDeployment. Use ComponentName for component-scoped scaling or
// ComponentGroupName for experimental multi-component group scaling.
// +kubebuilder:validation:XValidation:rule="has(self.componentName) != has(self.componentGroupName)",message="exactly one of componentName or componentGroupName must be set"
type DynamoGraphDeploymentComponentRef struct {
	// name is the `metadata.name` of the target DynamoGraphDeployment.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// componentName is the `componentName` of the entry within the target
	// DGD's `spec.components` list to scale.
	// +optional
	// +kubebuilder:validation:MinLength=1
	ComponentName string `json:"componentName,omitempty"`

	// componentGroupName is the name of an entry within the target DGD's
	// `spec.experimental.componentGroups` list. Scaling this target updates
	// that component group's `replicas` field.
	// +optional
	// +kubebuilder:validation:MinLength=1
	ComponentGroupName string `json:"componentGroupName,omitempty"`
}

// DynamoGraphDeploymentScalingAdapterStatus defines the observed state of a
// DynamoGraphDeploymentScalingAdapter.
type DynamoGraphDeploymentScalingAdapterStatus struct {
	// replicas is the current number of replicas for the target component or
	// component group. This is synced from the DGD target and is required for
	// the Scale subresource.
	Replicas int32 `json:"replicas"`

	// selector is a label selector string for the pods managed by this
	// adapter. Required for HPA compatibility via the scale subresource.
	// +optional
	Selector string `json:"selector,omitempty"`

	// lastScaleTime is the last time the adapter scaled the target component.
	// +optional
	LastScaleTime *metav1.Time `json:"lastScaleTime,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.replicas,selectorpath=.status.selector
// +kubebuilder:printcolumn:name="DGD",type="string",JSONPath=".spec.dgdRef.name",description="DynamoGraphDeployment name"
// +kubebuilder:printcolumn:name="COMPONENT",type="string",JSONPath=".spec.dgdRef['componentName','componentGroupName']",description="Component or component group name"
// +kubebuilder:printcolumn:name="REPLICAS",type="integer",JSONPath=".status.replicas",description="Current replicas"
// +kubebuilder:printcolumn:name="AGE",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:resource:shortName={dgdsa}

// DynamoGraphDeploymentScalingAdapter provides a scaling interface for
// individual components and component groups within a DynamoGraphDeployment.
// It implements the Kubernetes scale subresource, enabling integration with
// HPA, KEDA, and custom autoscalers.
//
// The adapter acts as an intermediary between autoscalers and the DGD,
// ensuring that only the adapter controller modifies the DGD's target replicas.
// This prevents conflicts when multiple autoscaling mechanisms are in play.
//
// v1beta1 is the storage version; conversion to and from the served v1alpha1
// version is handled by the operator's conversion webhook
// (see api/v1alpha1/dynamographdeploymentscalingadapter_conversion.go).
type DynamoGraphDeploymentScalingAdapter struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoGraphDeploymentScalingAdapterSpec   `json:"spec,omitempty"`
	Status DynamoGraphDeploymentScalingAdapterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoGraphDeploymentScalingAdapterList contains a list of DynamoGraphDeploymentScalingAdapter.
type DynamoGraphDeploymentScalingAdapterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeploymentScalingAdapter `json:"items"`
}

// IsReady returns true if the adapter has active replicas and a selector.
func (d *DynamoGraphDeploymentScalingAdapter) IsReady() (bool, string) {
	if d.Status.Selector == "" {
		return false, "Selector not set"
	}
	if d.Status.Replicas == 0 {
		return false, "No replicas"
	}
	return true, ""
}

// GetState returns "ready" or "not_ready".
func (d *DynamoGraphDeploymentScalingAdapter) GetState() string {
	ready, _ := d.IsReady()
	if ready {
		return consts.ResourceStateReady
	}
	return consts.ResourceStateNotReady
}
