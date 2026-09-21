// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// LPXGraphDeploymentSpec is the operator's handoff for all LPX components in a DGD.
// The owning DGD remains the only authored workload spec. The child reads that
// DGD only after verifying the controller owner and effective LPX input revision.
type LPXGraphDeploymentSpec struct {
	// inputRevision hashes only effective LPX inputs, including its selected
	// restart token. Unrelated component edits do not create an LPX revision.
	// +kubebuilder:validation:Pattern="^sha256:[a-f0-9]{64}$"
	InputRevision string `json:"inputRevision"`
}

// LPXGraphDeploymentStatus is the LPX controller's durable lifecycle state.
// Dynamo consumes readiness only for the current generation and input revision.
type LPXGraphDeploymentStatus struct {
	// observedGeneration is the child generation processed successfully.
	// Every spec.inputRevision change advances the child generation.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
	// conditions report compilation, placement, and runtime readiness.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
	// retainedReplicas preserves the last selected complete-engine count across
	// PodCliqueSet replacement. Explicit component replicas and an observed native
	// scaling group take precedence; nil means no count has been retained yet.
	// +optional
	// +kubebuilder:validation:Minimum=0
	RetainedReplicas *int32 `json:"retainedReplicas,omitempty"`
	// expiredRequestUIDs records requests selected for deadline cleanup until they disappear,
	// even if their scheduler phase or the deployment input changes.
	// +optional
	// +listType=set
	ExpiredRequestUIDs []types.UID `json:"expiredRequestUIDs,omitempty"`
	// components reports logical replicas by authored component name, never Agent Pods.
	// +optional
	Components map[string]v1beta1.ComponentReplicaStatus `json:"components,omitempty"`
	// modelDownload retains the existing remote-build download progress.
	// +optional
	ModelDownload *v1beta1.ModelDownloadStatus `json:"modelDownload,omitempty"`
	// placement retains scheduler placement progress.
	// +optional
	Placement *v1beta1.PlacementStatus `json:"placement,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:resource:shortName=lpxgd
// +kubebuilder:printcolumn:name="Ready",type="string",JSONPath=`.status.conditions[?(@.type=="Ready")].status`
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// LPXGraphDeployment is generated and owned by the Dynamo operator, not authored
// by users. It owns one LPX PCS and its build/placement resources.
// It serves only v1alpha1 and has no conversion contract.
type LPXGraphDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              LPXGraphDeploymentSpec `json:"spec"`
	// +optional
	Status LPXGraphDeploymentStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// LPXGraphDeploymentList contains operator-generated LPX deployments.
type LPXGraphDeploymentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []LPXGraphDeployment `json:"items"`
}
