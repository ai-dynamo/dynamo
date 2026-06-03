// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Snapshot is the user-facing binding for a container checkpoint.
// It identifies what was captured and is consumed by restore paths.
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=snap
// +kubebuilder:printcolumn:name="Bound-Content",type=string,JSONPath=`.status.boundSnapshotContentName`
// +kubebuilder:printcolumn:name="Created",type=date,JSONPath=`.status.creationTime`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
type Snapshot struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   SnapshotSpec   `json:"spec,omitempty"`
	Status SnapshotStatus `json:"status,omitempty"`
}

// SnapshotList is the standard list wrapper for Snapshot.
// +kubebuilder:object:root=true
type SnapshotList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Snapshot `json:"items"`
}

// SnapshotSpec describes what this Snapshot is OF and how it should be
// captured. Two creation flows:
//  1. SnapshotJob-produced: the SnapshotJob controller creates a
//     Snapshot, copying its QuiesceProbe and Storage into Snapshot.Spec.
//  2. Direct user creation (live pod): the user creates a Snapshot with
//     PodRef pointing at an existing pod that already satisfies the
//     snapshot contract (control volume, target-container labels, etc.).
//
// +kubebuilder:validation:XValidation:rule="self.source == oldSelf.source",message="source is immutable after binding"
type SnapshotSpec struct {
	// Source identifies the origin of this snapshot. Immutable after binding.
	// +kubebuilder:validation:Required
	Source SnapshotSource `json:"source"`

	// QuiesceProbe defines how the agent detects that the target
	// containers are safe to dump. When omitted, the agent uses the
	// pod's standard Ready condition as the quiesce gate.
	// +optional
	QuiesceProbe *QuiesceProbe `json:"quiesceProbe,omitempty"`

	// Storage describes where the artifact will be written. Required for
	// direct creation; for SnapshotJob-produced Snapshots, copied from
	// SnapshotJob.Spec.Storage at creation time.
	// +kubebuilder:validation:Required
	Storage SnapshotStorage `json:"storage"`
}

// SnapshotSource identifies the captured workload. Kept as a struct
// (rather than inlined PodRef) so future variants can be added additively.
type SnapshotSource struct {
	// PodRef references the pod whose containers were (or will be)
	// captured. For direct creation, the pod must exist at Snapshot
	// creation time and satisfy the snapshot contract.
	// +kubebuilder:validation:Required
	PodRef PodReference `json:"podRef"`
}

// SnapshotStatus is the observed state of a Snapshot.
type SnapshotStatus struct {
	// BoundSnapshotContentName is the name of the SnapshotContent object
	// this Snapshot is bound to. nil until the binding is established.
	//
	// Consumers MUST verify binding by checking that both Snapshot and
	// SnapshotContent point at each other before treating this object as
	// usable for restore.
	// +optional
	BoundSnapshotContentName *string `json:"boundSnapshotContentName,omitempty"`

	// CreationTime is the timestamp at which the artifact was created
	// (CRIU dump complete, manifest finalized). Mirrors the bound
	// SnapshotContent's CreationTime.
	// +optional
	CreationTime *metav1.Time `json:"creationTime,omitempty"`

	// Conditions reflect the latest observations of the Snapshot's state.
	// Standard condition types:
	//   Ready  — True when the bound SnapshotContent is Ready and the
	//            artifact is usable for restore.
	//   Failed — True if creation or binding failed terminally; reason
	//            and message carry the detail.
	// +optional
	// +patchStrategy=merge
	// +patchMergeKey=type
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
}
