// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
)

// PodReference identifies the pod that was (or will be) captured
// and optionally narrows which of its containers are in scope.
//
// The pod's UID is intentionally NOT carried here: it is runtime
// state, not desired state, and the user cannot know it at Snapshot
// creation time. If the controller wrote UID back into the spec, the
// `self.source == oldSelf.source` immutability CEL on SnapshotSpec
// would reject the controller's own update.
type PodReference struct {
	// Name of the source pod, in the same namespace as the Snapshot.
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// Containers narrows the snapshot scope to specific containers
	// within the pod. If empty, all containers in the pod are captured.
	// +optional
	Containers []string `json:"containers,omitempty"`
}

// SnapshotStorage describes the artifact destination. Shared between
// Snapshot.Spec (direct creation) and SnapshotJob.Spec (factory).
// v1alpha1 supports PVC only; the tagged-union shape is preserved for
// additive future variants.
// +kubebuilder:validation:XValidation:rule="has(self.pvc)",message="exactly one storage variant must be set; v1alpha1 supports only pvc"
type SnapshotStorage struct {
	// PVC names the PersistentVolumeClaim that will store the artifact.
	// The artifact directory inside the PVC is producer-assigned (derived
	// from the producing Snapshot or SnapshotJob UID), not user-settable.
	// +optional
	PVC *PVCStorage `json:"pvc,omitempty"`
}

// PVCStorage references a PersistentVolumeClaim in the same namespace
// as the producing Snapshot or SnapshotJob.
type PVCStorage struct {
	// ClaimName of the PVC. Must support the access mode required by
	// the deployment (ReadWriteMany for production; ReadWriteOnce
	// permitted for single-node test/dev with appropriate nodeAffinity).
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	ClaimName string `json:"claimName"`
}

// QuiesceProbe describes how the snapshot agent detects that the
// target containers are ready to be dumped. Modeled on corev1.Probe
// but evaluated by the snapshot agent — not kubelet — so it can be
// applied without affecting the pod's serving readiness.
type QuiesceProbe struct {
	// Action is the probe operation. Exactly one variant in Action must be set.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:XValidation:rule="[has(self.httpGet),has(self.exec),has(self.tcpSocket),has(self.grpc),has(self.file)].exists_one(x,x)",message="exactly one probe action must be set"
	Action QuiesceProbeAction `json:"action"`

	// PeriodSeconds is the interval between consecutive probe attempts.
	// +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	PeriodSeconds int32 `json:"periodSeconds,omitempty"`

	// TimeoutSeconds is the per-attempt timeout. A probe attempt that
	// does not complete within this many seconds counts as a failure.
	// +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	TimeoutSeconds int32 `json:"timeoutSeconds,omitempty"`

	// SuccessThreshold is the number of consecutive successes required
	// before the probe is considered satisfied and the dump begins.
	// +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	SuccessThreshold int32 `json:"successThreshold,omitempty"`

	// FailureThreshold is the number of consecutive failures after which
	// the agent abandons the probe. Bounded; SnapshotJob.ActiveDeadlineSeconds
	// is the wall-clock hard deadline.
	// +optional
	// +kubebuilder:default=7200
	// +kubebuilder:validation:Minimum=1
	FailureThreshold int32 `json:"failureThreshold,omitempty"`
}

// QuiesceProbeAction is a tagged union of probe actions. Exactly one
// variant must be set; enforcement lives on the consuming field
// (QuiesceProbe.Action) — controller-gen picks up XValidation markers
// from fields, not type aliases.
type QuiesceProbeAction struct {
	// HTTPGet probes an HTTP endpoint on the target container.
	// +optional
	HTTPGet *corev1.HTTPGetAction `json:"httpGet,omitempty"`

	// Exec runs a command inside the target container.
	// +optional
	Exec *corev1.ExecAction `json:"exec,omitempty"`

	// TCPSocket probes a TCP port on the target container.
	// +optional
	TCPSocket *corev1.TCPSocketAction `json:"tcpSocket,omitempty"`

	// GRPC probes via the standard gRPC health protocol.
	// +optional
	GRPC *corev1.GRPCAction `json:"grpc,omitempty"`

	// File probes for the existence of a sentinel file in the target
	// container's filesystem. NVSnapshot-specific (no corev1 equivalent).
	// +optional
	File *FileAction `json:"file,omitempty"`
}

// FileAction probes for the existence of a sentinel file inside the
// target container's filesystem.
type FileAction struct {
	// Path is the absolute path to the sentinel file inside the container.
	// +kubebuilder:validation:Required
	Path string `json:"path"`
}
