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

// DGDScalingAdapterSpec defines the desired state of DGDScalingAdapter
type DGDScalingAdapterSpec struct {
	// Replicas is the desired number of replicas for the target service.
	// This field is modified by external autoscalers (HPA/KEDA/Planner) or manually by users.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=0
	Replicas int32 `json:"replicas"`

	// DGDRef references the DynamoGraphDeployment and the specific service to scale.
	// +kubebuilder:validation:Required
	DGDRef DGDServiceRef `json:"dgdRef"`

	// ScalingPolicy defines optional constraints for scaling behavior.
	// These constraints are enforced by the adapter controller, providing
	// an additional safety layer beyond HPA's own min/max settings.
	// +optional
	ScalingPolicy *ScalingPolicy `json:"scalingPolicy,omitempty"`
}

// DGDServiceRef identifies a specific service within a DynamoGraphDeployment
type DGDServiceRef struct {
	// Name of the DynamoGraphDeployment
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// Service is the key name of the service within the DGD's spec.services map to scale
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Service string `json:"service"`
}

// ScalingPolicy defines constraints and behavior for scaling operations
type ScalingPolicy struct {
	// MinReplicas is the lower bound for scaling.
	// The adapter will not scale below this value even if the autoscaler requests it.
	// +kubebuilder:validation:Minimum=0
	// +optional
	MinReplicas *int32 `json:"minReplicas,omitempty"`

	// MaxReplicas is the upper bound for scaling.
	// The adapter will not scale above this value even if the autoscaler requests it.
	// +kubebuilder:validation:Minimum=1
	// +optional
	MaxReplicas *int32 `json:"maxReplicas,omitempty"`

	// ScaleDownStabilizationSeconds is the time to wait before scaling down
	// after the last scale operation. This provides additional protection against
	// rapid scale oscillations beyond what HPA provides.
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:default=0
	// +optional
	ScaleDownStabilizationSeconds *int32 `json:"scaleDownStabilizationSeconds,omitempty"`
}

// DGDScalingAdapterStatus defines the observed state of DGDScalingAdapter
type DGDScalingAdapterStatus struct {
	// Replicas is the current number of replicas for the target service.
	// This is synced from the DGD's service replicas.
	// +optional
	Replicas int32 `json:"replicas,omitempty"`

	// Selector is a label selector string for the pods managed by this adapter.
	// Required for HPA compatibility via the scale subresource.
	// +optional
	Selector string `json:"selector,omitempty"`

	// LastScaleTime is the last time the adapter scaled the target service.
	// +optional
	LastScaleTime *metav1.Time `json:"lastScaleTime,omitempty"`

	// Conditions represent the latest available observations of the adapter's state.
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.replicas,selectorpath=.status.selector
// +kubebuilder:printcolumn:name="DGD",type="string",JSONPath=".spec.dgdRef.name",description="DynamoGraphDeployment name"
// +kubebuilder:printcolumn:name="SERVICE",type="string",JSONPath=".spec.dgdRef.service",description="Service name"
// +kubebuilder:printcolumn:name="DESIRED",type="integer",JSONPath=".spec.replicas",description="Desired replicas"
// +kubebuilder:printcolumn:name="CURRENT",type="integer",JSONPath=".status.replicas",description="Current replicas"
// +kubebuilder:printcolumn:name="READY",type="string",JSONPath=".status.conditions[?(@.type=='Ready')].status",description="Ready status"
// +kubebuilder:printcolumn:name="AGE",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:resource:shortName={dgdsa}

// DGDScalingAdapter provides a scaling interface for individual services
// within a DynamoGraphDeployment. It implements the Kubernetes scale
// subresource, enabling integration with HPA, KEDA, and custom autoscalers.
//
// The adapter acts as an intermediary between autoscalers and the DGD,
// ensuring that only the adapter controller modifies the DGD's service replicas.
// This prevents conflicts when multiple autoscaling mechanisms are in play.
type DGDScalingAdapter struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DGDScalingAdapterSpec   `json:"spec,omitempty"`
	Status DGDScalingAdapterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DGDScalingAdapterList contains a list of DGDScalingAdapter
type DGDScalingAdapterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DGDScalingAdapter `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DGDScalingAdapter{}, &DGDScalingAdapterList{})
}

// Condition types for DGDScalingAdapter
const (
	// ConditionTypeReady indicates the adapter is synced with DGD and functioning correctly
	ConditionTypeAdapterReady = "Ready"
)

// Condition reasons for DGDScalingAdapter
const (
	// ReasonDGDNotFound indicates the referenced DGD does not exist
	ReasonDGDNotFound = "DGDNotFound"
	// ReasonServiceNotFound indicates the referenced service does not exist in the DGD
	ReasonServiceNotFound = "ServiceNotFound"
	// ReasonSynced indicates the adapter is successfully synced with the DGD
	ReasonSynced = "Synced"
	// ReasonScalingPolicyViolation indicates a scaling request was blocked by policy
	ReasonScalingPolicyViolation = "ScalingPolicyViolation"
)

// SetCondition updates or adds a condition to the adapter's status
func (a *DGDScalingAdapter) SetCondition(condType string, status metav1.ConditionStatus, reason, message string) {
	now := metav1.Now()
	condition := metav1.Condition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: now,
		Reason:             reason,
		Message:            message,
		ObservedGeneration: a.Generation,
	}

	// Update existing condition or append new one
	for i, c := range a.Status.Conditions {
		if c.Type == condType {
			// Only update if status or reason changed
			if c.Status != status || c.Reason != reason || c.Message != message {
				a.Status.Conditions[i] = condition
			}
			return
		}
	}
	a.Status.Conditions = append(a.Status.Conditions, condition)
}

// GetCondition returns the condition with the given type, or nil if not found
func (a *DGDScalingAdapter) GetCondition(condType string) *metav1.Condition {
	for i := range a.Status.Conditions {
		if a.Status.Conditions[i].Type == condType {
			return &a.Status.Conditions[i]
		}
	}
	return nil
}

// IsReady returns true if the adapter is in Ready state
func (a *DGDScalingAdapter) IsReady() bool {
	cond := a.GetCondition(ConditionTypeAdapterReady)
	return cond != nil && cond.Status == metav1.ConditionTrue
}
