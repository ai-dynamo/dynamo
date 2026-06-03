// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// SnapshotContent is the cluster-scoped artifact-of-record for a
// captured container checkpoint. Equivalent in role to VolumeSnapshotContent.
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster,shortName=snapcontent
// +kubebuilder:printcolumn:name="Snapshot",type=string,JSONPath=`.spec.snapshotRef.name`
// +kubebuilder:printcolumn:name="Namespace",type=string,JSONPath=`.spec.snapshotRef.namespace`
// +kubebuilder:printcolumn:name="Size",type=integer,JSONPath=`.status.restoreSize`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
type SnapshotContent struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   SnapshotContentSpec   `json:"spec,omitempty"`
	Status SnapshotContentStatus `json:"status,omitempty"`
}

// SnapshotContentList is the standard list wrapper for SnapshotContent.
// +kubebuilder:object:root=true
type SnapshotContentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []SnapshotContent `json:"items"`
}

// SnapshotContentSpec describes the artifact and its lifecycle policy.
// +kubebuilder:validation:XValidation:rule="self.snapshotRef == oldSelf.snapshotRef",message="snapshotRef is immutable after binding"
// +kubebuilder:validation:XValidation:rule="self.source == oldSelf.source",message="source is immutable after binding"
// +kubebuilder:validation:XValidation:rule="!has(oldSelf.runtimeInformation) || (has(self.runtimeInformation) && self.runtimeInformation == oldSelf.runtimeInformation)",message="runtimeInformation is immutable after binding"
type SnapshotContentSpec struct {
	// SnapshotRef is the back-pointer to the bound Snapshot. May span
	// namespaces since SnapshotContent is cluster-scoped. Immutable
	// after binding.
	// +kubebuilder:validation:Required
	SnapshotRef SnapshotReference `json:"snapshotRef"`

	// Source describes where the artifact is stored. v1alpha1: PVC only.
	// Immutable after binding.
	// +kubebuilder:validation:Required
	Source SnapshotContentSource `json:"source"`

	// RuntimeInformation is a free-form map describing the runtime
	// environment that produced the artifact — typically GPU model,
	// driver version, node arch, CRIU version, etc. The agent uses it
	// at restore time to verify the node is compatible.
	//
	// MaxProperties=32 bounds the CEL immutability rule's evaluation
	// cost; far above the ~4 conventional keys for the v1alpha1 driver.
	// +optional
	// +kubebuilder:validation:MaxProperties=32
	RuntimeInformation map[string]string `json:"runtimeInformation,omitempty"`
}

// SnapshotReference is a cross-namespace reference to a Snapshot.
type SnapshotReference struct {
	// Namespace of the referent.
	// +kubebuilder:validation:Required
	Namespace string `json:"namespace"`

	// Name of the referent.
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// UID of the referent. Populated at binding time to detect stale
	// references. Safe to live in spec because the controller sets it
	// once at SnapshotContent creation, not via subsequent updates.
	// +optional
	UID types.UID `json:"uid,omitempty"`
}

// SnapshotContentSource describes the artifact backend via an opaque,
// driver-specific handle. SnapshotContent is cluster-scoped and cannot
// directly reference namespaced resources (e.g., PVCs); the handle
// encodes any cross-namespace location information as a string.
//
// v1alpha1 supports the PVC backend; future backends (object storage,
// URI, etc.) use different handle formats but the same field shape.
type SnapshotContentSource struct {
	// SnapshotHandle is an opaque, driver-specific identifier for the
	// physical artifact.
	//
	// For the v1alpha1 PVC backend, the format is:
	//   pvc://<namespace>/<claimName>/<basePath>
	//
	// Set ONLY for pre-provisioned (imported) SnapshotContents. For
	// dynamic creation, leave nil — the controller writes the resolved
	// handle to Status.SnapshotHandle. The presence of this field is
	// the discriminator between import (set) and dynamic (nil). Keeping
	// the controller-assigned handle in status (not spec) avoids
	// fighting the spec.source immutability CEL.
	//
	// Immutable after binding (CEL on enclosing spec.source).
	// +optional
	// +kubebuilder:validation:MinLength=1
	SnapshotHandle *string `json:"snapshotHandle,omitempty"`
}

// SnapshotContentStatus is the observed state of a SnapshotContent.
type SnapshotContentStatus struct {
	// SnapshotHandle is the canonical, validated handle for the artifact.
	// Authoritative location:
	//   - Dynamic creation: controller resolves the handle post-dump and
	//     writes it here; Spec.Source.SnapshotHandle is nil.
	//   - Pre-provisioned import: controller accepts the user-supplied
	//     Spec.Source.SnapshotHandle, validates the artifact, and mirrors
	//     the validated value here.
	// Consumers should read this field, not Spec.Source.SnapshotHandle.
	// +optional
	SnapshotHandle *string `json:"snapshotHandle,omitempty"`

	// CreationTime is the timestamp at which the artifact was created
	// (CRIU dump complete, manifest finalized). For pre-provisioned
	// imports, this is the time recorded in the artifact manifest.
	// +optional
	CreationTime *metav1.Time `json:"creationTime,omitempty"`

	// RestoreSize is the on-PVC artifact size in bytes. Consumers may
	// use this to size restore-target volumes / nodes.
	// +optional
	// +kubebuilder:validation:Minimum=0
	RestoreSize *int64 `json:"restoreSize,omitempty"`

	// Conditions reflect the latest observations of the SnapshotContent.
	// Standard condition types:
	//   Ready  — True when the artifact is complete and usable for restore.
	//   Failed — True if artifact provisioning or validation failed
	//            terminally; reason and message carry the detail.
	// +optional
	// +patchStrategy=merge
	// +patchMergeKey=type
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
}
