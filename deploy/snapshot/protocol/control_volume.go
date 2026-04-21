// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	corev1 "k8s.io/api/core/v1"
)

const (
	// SnapshotControlVolumeName is the per-pod emptyDir used to carry
	// checkpoint/restore lifecycle sentinels written by the snapshot agent
	// and observed by the workload. It replaces the SIGUSR1/SIGCONT signals
	// that previously required the workload to run as PID 1.
	SnapshotControlVolumeName = "snapshot-control"

	// SnapshotControlMountPath is where the control volume is mounted inside
	// the workload container.
	SnapshotControlMountPath = "/snapshot-control"

	// SnapshotControlDirEnv is the environment variable exposing the control
	// mount path to the workload.
	SnapshotControlDirEnv = "DYN_SNAPSHOT_CONTROL_DIR"

	// SnapshotCompleteFile is written by the snapshot agent inside the
	// control volume when a checkpoint has completed successfully.
	SnapshotCompleteFile = "snapshot-complete"

	// RestoreCompleteFile is written by the snapshot agent inside the
	// control volume when a restore has completed and the workload may
	// resume.
	RestoreCompleteFile = "restore-complete"

	// ReadyForCheckpointFile is written by the workload inside the control
	// volume when the model is loaded and the workload is ready for a
	// checkpoint. Observed by the checkpoint job's kubelet readiness probe
	// on the worker container.
	ReadyForCheckpointFile = "ready-for-checkpoint"
)

// EnsureControlVolume adds the snapshot-control emptyDir to the pod spec,
// mounts it on the given container at SnapshotControlMountPath, and sets
// DYN_SNAPSHOT_CONTROL_DIR on the container's env. Idempotent — safe to call
// from multiple code paths (operator checkpoint job, restore pod shaping, etc.).
func EnsureControlVolume(podSpec *corev1.PodSpec, container *corev1.Container) {
	if podSpec == nil || container == nil {
		return
	}

	hasVolume := false
	for _, v := range podSpec.Volumes {
		if v.Name == SnapshotControlVolumeName {
			hasVolume = true
			break
		}
	}
	if !hasVolume {
		podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
			Name:         SnapshotControlVolumeName,
			VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
		})
	}

	hasMount := false
	for _, m := range container.VolumeMounts {
		if m.Name == SnapshotControlVolumeName {
			hasMount = true
			break
		}
	}
	if !hasMount {
		container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
			Name:      SnapshotControlVolumeName,
			MountPath: SnapshotControlMountPath,
		})
	}

	hasEnv := false
	for _, e := range container.Env {
		if e.Name == SnapshotControlDirEnv {
			hasEnv = true
			break
		}
	}
	if !hasEnv {
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  SnapshotControlDirEnv,
			Value: SnapshotControlMountPath,
		})
	}
}
