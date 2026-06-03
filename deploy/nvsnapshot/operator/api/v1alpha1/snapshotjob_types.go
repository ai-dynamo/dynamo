// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SnapshotJob is a one-shot producer of a Snapshot + SnapshotContent.
// Analogous to batch/v1.Job. Does not own the produced artifacts.
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=snapjob
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Content",type=string,JSONPath=`.status.contentName`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
type SnapshotJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   SnapshotJobSpec   `json:"spec,omitempty"`
	Status SnapshotJobStatus `json:"status,omitempty"`
}

// SnapshotJobList is the standard list wrapper for SnapshotJob.
// +kubebuilder:object:root=true
type SnapshotJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []SnapshotJob `json:"items"`
}

// SnapshotJobSpec describes the desired snapshot production operation.
// SnapshotJob is one-shot: spec fields are immutable after creation.
// +kubebuilder:validation:XValidation:rule="has(self.podTemplate.spec) && size(self.podTemplate.spec.containers) > 0",message="podTemplate.spec.containers must be non-empty"
// +kubebuilder:validation:XValidation:rule="self.podTemplate == oldSelf.podTemplate",message="podTemplate is immutable"
// +kubebuilder:validation:XValidation:rule="self.storage == oldSelf.storage",message="storage is immutable"
// +kubebuilder:validation:XValidation:rule="self.quiesceProbe == oldSelf.quiesceProbe",message="quiesceProbe is immutable"
// +kubebuilder:validation:XValidation:rule="self.targetContainers == oldSelf.targetContainers",message="targetContainers is immutable"
type SnapshotJobSpec struct {
	// PodTemplate is the pod to launch. The application running in the
	// target containers must implement the quiesce contract — see
	// QuiesceProbe. Immutable after creation.
	// +kubebuilder:validation:Required
	PodTemplate corev1.PodTemplateSpec `json:"podTemplate"`

	// QuiesceProbe defines how to determine when the pod is safe to dump.
	// When omitted, the agent uses the pod's standard Ready condition as
	// the quiesce gate.
	// +optional
	QuiesceProbe *QuiesceProbe `json:"quiesceProbe,omitempty"`

	// TargetContainers narrows which containers in PodTemplate are
	// captured. If empty, all containers in the PodTemplate are captured.
	// +optional
	TargetContainers []string `json:"targetContainers,omitempty"`

	// Storage describes where the artifact will be written. v1alpha1: PVC only.
	// +kubebuilder:validation:Required
	Storage SnapshotStorage `json:"storage"`

	// ActiveDeadlineSeconds bounds the total lifetime of the dump
	// operation (pod scheduling + quiesce wait + dump execution).
	// +optional
	// +kubebuilder:default=3600
	// +kubebuilder:validation:Minimum=1
	ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty"`
}

// SnapshotJobStatus is the observed state of a SnapshotJob. Producer
// users read this object only — fields here are the minimum needed to
// drive restore consumption (one-pane-of-glass UX).
type SnapshotJobStatus struct {
	// Phase is the high-level lifecycle stage.
	// +optional
	// +kubebuilder:validation:Enum=Pending;Running;Succeeded;Failed
	Phase SnapshotJobPhase `json:"phase,omitempty"`

	// ContentName is the name of the produced SnapshotContent. Restore
	// paths reference the artifact by this name.
	// +optional
	ContentName string `json:"contentName,omitempty"`

	// StartedAt is the time at which the SnapshotJob entered the
	// Running phase.
	// +optional
	StartedAt *metav1.Time `json:"startedAt,omitempty"`

	// CompletedAt is the time at which the SnapshotJob reached a
	// terminal phase (Succeeded or Failed).
	// +optional
	CompletedAt *metav1.Time `json:"completedAt,omitempty"`

	// Conditions reflect the latest observations of the SnapshotJob.
	// Standard condition types (modeled on batch/v1.Job):
	//   Complete — True once dump finished and Snapshot+SnapshotContent
	//              were produced.
	//   Failed   — True if dump failed (deadline, pod failure, agent error);
	//              reason and message carry the detail.
	// +optional
	// +patchStrategy=merge
	// +patchMergeKey=type
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
}

// SnapshotJobPhase is the lifecycle stage of a SnapshotJob.
// The enum validation marker lives on SnapshotJobStatus.Phase (the
// consuming field) — controller-gen reads enum markers from fields,
// not from type aliases.
type SnapshotJobPhase string

const (
	// SnapshotJobPhasePending: SnapshotJob is created but pod has not started.
	SnapshotJobPhasePending SnapshotJobPhase = "Pending"

	// SnapshotJobPhaseRunning: pod is running; may be in quiesce or dump phase.
	SnapshotJobPhaseRunning SnapshotJobPhase = "Running"

	// SnapshotJobPhaseSucceeded: dump complete; SnapshotContent + Snapshot produced.
	SnapshotJobPhaseSucceeded SnapshotJobPhase = "Succeeded"

	// SnapshotJobPhaseFailed: dump failed (deadline, pod failure, agent error).
	SnapshotJobPhaseFailed SnapshotJobPhase = "Failed"
)
