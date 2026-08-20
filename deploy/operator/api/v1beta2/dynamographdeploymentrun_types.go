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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// DynamoGraphDeploymentRunSpec is an immutable, exact copy of an accepted DGDR spec.
type DynamoGraphDeploymentRunSpec = DynamoGraphDeploymentRequestSpec

// DGDRRunPhase is a concise presentation state. Conditions remain authoritative.
// +kubebuilder:validation:Enum=Pending;Running;Succeeded;Failed
type DGDRRunPhase string

const (
	DGDRRunPhasePending   DGDRRunPhase = "Pending"
	DGDRRunPhaseRunning   DGDRRunPhase = "Running"
	DGDRRunPhaseSucceeded DGDRRunPhase = "Succeeded"
	DGDRRunPhaseFailed    DGDRRunPhase = "Failed"
)

// ResourceReference identifies a namespaced resource by name and UID.
type ResourceReference struct {
	// Name is the resource name.
	Name string `json:"name"`

	// UID is the resource UID.
	// +optional
	UID types.UID `json:"uid,omitempty"`
}

// RoundProgress reports completed and total optimizer rounds.
type RoundProgress struct {
	Completed int32 `json:"completed"`
	Total     int32 `json:"total"`
}

// BranchProgress reports progress for one resolved deployment-mode branch.
type BranchProgress struct {
	Name   string        `json:"name"`
	Rounds RoundProgress `json:"rounds"`
}

// EvaluationProgress reports unique candidate outcomes without creating one resource per trial.
type EvaluationProgress struct {
	Scheduled   int64 `json:"scheduled,omitempty"`
	Running     int64 `json:"running,omitempty"`
	Feasible    int64 `json:"feasible,omitempty"`
	Infeasible  int64 `json:"infeasible,omitempty"`
	Failed      int64 `json:"failed,omitempty"`
	Unsupported int64 `json:"unsupported,omitempty"`
	CacheHits   int64 `json:"cacheHits,omitempty"`
}

// CandidateProgress reports bounded Kubernetes publication, not all internal trials.
type CandidateProgress struct {
	ParetoFront int64 `json:"paretoFront,omitempty"`
	Published   int32 `json:"published,omitempty"`
}

// DGDRRunProgress contains branch-aware round limits and evaluation outcome counters.
type DGDRRunProgress struct {
	// Branches reports each resolved deployment-mode branch because MaxRounds applies per branch.
	// +optional
	// +listType=map
	// +listMapKey=name
	Branches []BranchProgress `json:"branches,omitempty"`

	// Evaluations reports unique candidate outcomes.
	// +optional
	Evaluations *EvaluationProgress `json:"evaluations,omitempty"`

	// Candidates reports bounded Kubernetes publication.
	// +optional
	Candidates *CandidateProgress `json:"candidates,omitempty"`
}

// DGDRRunProvenance records execution implementation versions.
type DGDRRunProvenance struct {
	SweeperVersion string `json:"sweeperVersion,omitempty"`
	ReplayVersion  string `json:"replayVersion,omitempty"`
}

// DynamoGraphDeploymentRunStatus represents one search execution's observed state.
type DynamoGraphDeploymentRunStatus struct {
	// Phase is a concise presentation state. Conditions remain authoritative.
	// +optional
	Phase DGDRRunPhase `json:"phase,omitempty"`

	// JobRef identifies the controller-owned Job for this run.
	// +optional
	JobRef *ResourceReference `json:"jobRef,omitempty"`

	// Progress contains branch-aware round limits and evaluation outcome counters.
	// +optional
	Progress *DGDRRunProgress `json:"progress,omitempty"`

	// CandidateRefs identifies the current bounded DGDC projection.
	// +optional
	CandidateRefs []corev1.LocalObjectReference `json:"candidateRefs,omitempty"`

	// Provenance records execution implementations.
	// +optional
	Provenance *DGDRRunProvenance `json:"provenance,omitempty"`

	// StartTime is when execution started.
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// CompletionTime is present only after a terminal outcome.
	// +optional
	CompletionTime *metav1.Time `json:"completionTime,omitempty"`

	// LastProgressTime is the latest progress update or heartbeat.
	// +optional
	LastProgressTime *metav1.Time `json:"lastProgressTime,omitempty"`

	// Conditions contains standard Kubernetes conditions merged by type.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dgdrrun
// +kubebuilder:validation:XValidation:rule="!has(oldSelf.spec) || self.spec == oldSelf.spec",message="spec is immutable"
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Completed",type=string,JSONPath=`.status.conditions[?(@.type=="Completed")].status`
// +kubebuilder:printcolumn:name="Candidates",type=integer,JSONPath=`.status.progress.candidates.published`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// DynamoGraphDeploymentRun is one immutable accepted DGDR request and its execution status.
type DynamoGraphDeploymentRun struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoGraphDeploymentRunSpec   `json:"spec"`
	Status DynamoGraphDeploymentRunStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoGraphDeploymentRunList contains a list of DynamoGraphDeploymentRun resources.
type DynamoGraphDeploymentRunList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeploymentRun `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoGraphDeploymentRun{}, &DynamoGraphDeploymentRunList{})
}
