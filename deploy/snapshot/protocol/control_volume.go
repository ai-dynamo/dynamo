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
	//
	// When a pod targets multiple containers (e.g. failover engine-0 +
	// engine-1), each container mounts the emptyDir with
	// subPath=<containerName>, so sentinels are isolated per-container on
	// disk while each container still sees them at SnapshotControlMountPath.
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
// mounts it on the given container at SnapshotControlMountPath (using
// subPath=<containerName> so concurrent target containers in a failover pod
// each see an isolated view), and sets DYN_SNAPSHOT_CONTROL_DIR on the
// container's env. Idempotent — safe to call from multiple code paths
// (operator checkpoint job, restore pod shaping, etc.).
//
// Callers must pass the container's own name; the subPath makes the mount
// container-scoped on disk even though the in-container path is the same.
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

	// Per-container subPath so each target container has its own sentinel
	// directory on the emptyDir's backing disk. An empty container name
	// degrades to the volume root, which is the correct (and only safe)
	// behavior for single-container pods.
	subPath := container.Name

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
			SubPath:   subPath,
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
