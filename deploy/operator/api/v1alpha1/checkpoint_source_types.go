/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

// CheckpointSource describes source details to use when configuring a restore pod.
type CheckpointSource struct {
	// Hardware describes the source GPU resources used to choose compatible restore placement.
	// +optional
	Hardware *CheckpointSourceHardware `json:"hardware,omitempty"`

	// Node identifies the source node for comparing restore placement dependencies.
	// +optional
	Node string `json:"node,omitempty"`

	// Mounts lists the volume mounts to recreate on the restore container.
	// +optional
	Mounts []CheckpointSourceMount `json:"mounts,omitempty"`

	// MountCount is the number of volume mount entries expected on the restore container.
	// +optional
	// +kubebuilder:validation:Minimum=0
	MountCount *int32 `json:"mountCount,omitempty"`
}

// CheckpointSourceHardware describes GPU resources to make available to the restore container.
type CheckpointSourceHardware struct {
	// GPUCount is the number of GPUs to make available to the restore container.
	// +optional
	// +kubebuilder:validation:Minimum=0
	GPUCount *int32 `json:"gpuCount,omitempty"`

	// GPUs identifies the source GPUs for checking restore hardware compatibility.
	// +optional
	GPUs []CheckpointSourceGPU `json:"gpus,omitempty"`
}

// CheckpointSourceGPU identifies one source GPU for restore compatibility checks.
type CheckpointSourceGPU struct {
	// UUID is the source GPU UUID to compare with available restore hardware.
	// +optional
	UUID string `json:"uuid,omitempty"`
}

// CheckpointSourceMount describes a volume mount to recreate on the restore container.
type CheckpointSourceMount struct {
	// Path is the mountPath to use in the restore container.
	Path string `json:"path" yaml:"path"`

	// Volume is the source volume name to correlate with the restore pod volume definition,
	// for example "model-cache" for spec.volumes[name: model-cache].
	Volume string `json:"volume" yaml:"volume"`

	// VolumeSource identifies the Kubernetes volume source to reproduce as kind[/identifier].
	// Kinds are PersistentVolumeClaim, ConfigMap, Secret, HostPath, CSI, NFS,
	// EmptyDir, Projected, DownwardAPI, Ephemeral, or Volume.
	VolumeSource string `json:"volumeSource" yaml:"volumeSource"`
}
