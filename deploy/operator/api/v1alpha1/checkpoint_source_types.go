/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

// CheckpointSource describes the environment a checkpoint was created from.
type CheckpointSource struct {
	// Hardware contains available source hardware facts.
	// +optional
	Hardware *CheckpointSourceHardware `json:"hardware,omitempty"`

	// Provenance identifies the source node.
	// +optional
	Provenance *CheckpointSourceProvenance `json:"provenance,omitempty"`

	// Mounts lists source pod mounts required by the checkpoint. Mounted volume
	// contents are not captured in the checkpoint.
	// +optional
	Mounts []CheckpointSourceMount `json:"mounts,omitempty"`

	// MountCount is the number of source pod mounts required by the checkpoint.
	// +optional
	// +kubebuilder:validation:Minimum=0
	MountCount *int32 `json:"mountCount,omitempty"`

	// RuntimeManagedMounts is the number of additional mounts recreated by the
	// container runtime or kubelet.
	// +optional
	// +kubebuilder:validation:Minimum=0
	RuntimeManagedMounts *int32 `json:"runtimeManagedMounts,omitempty"`
}

// CheckpointSourceHardware contains available source hardware facts.
type CheckpointSourceHardware struct {
	// GPUCount is the number of GPUs visible to the captured container.
	// +optional
	// +kubebuilder:validation:Minimum=0
	GPUCount *int32 `json:"gpuCount,omitempty"`

	// GPUs identifies the GPUs visible to the captured container.
	// +optional
	GPUs []CheckpointSourceGPU `json:"gpus,omitempty"`
}

// CheckpointSourceGPU identifies one source GPU.
type CheckpointSourceGPU struct {
	// UUID is the GPU UUID recorded at capture time.
	// +optional
	UUID string `json:"uuid,omitempty"`
}

// CheckpointSourceProvenance identifies the source node.
type CheckpointSourceProvenance struct {
	// Node is the source node name.
	// +optional
	Node string `json:"node,omitempty"`
}

// CheckpointSourceMount describes a source pod mount required at restore.
// Mounted volume contents are not captured in the checkpoint.
type CheckpointSourceMount struct {
	// Path is the mount destination inside the captured container.
	Path string `json:"path"`

	// Volume is the source pod volume name.
	Volume string `json:"volume"`

	// ProvidedBy identifies the Kubernetes volume source.
	ProvidedBy string `json:"providedBy"`
}
