// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"path/filepath"
	"strings"

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

	// NCCLCheckpointKVSPathEnv points the NCCL checkpoint shim at an
	// indirection file that the snapshot agent rewrites before restore-complete.
	NCCLCheckpointKVSPathEnv = "NCCL_CHECKPOINT_KVS_PATH"

	// NCCLCheckpointKVSFile is captured in restored process env and contains
	// the Redis endpoint and prefix to use for this restore attempt.
	NCCLCheckpointKVSFile = "nccl-kvs.txt"

	// NCCLCheckpointRedisContainerName is the implicit restore-only Redis
	// sidecar used by NCCL checkpoint_restore to exchange fresh unique IDs.
	NCCLCheckpointRedisContainerName = "nccl-checkpoint-redis"

	// NCCLCheckpointRedisImage is intentionally internal: checkpoint restore
	// always injects the same lightweight Redis sidecar on leader pods.
	NCCLCheckpointRedisImage = "redis:7-alpine"

	// NCCLCheckpointRedisPort avoids the common Ray Redis port while keeping
	// the service inside the leader pod.
	NCCLCheckpointRedisPort = 46379

	// VLLMCheckpointRestoreEnabledEnv enables vLLM's Dynamo/CRIU checkpoint
	// integration path.
	VLLMCheckpointRestoreEnabledEnv = "VLLM_ENABLE_CHECKPOINT_RESTORE"

	// VLLMCheckpointRestoreFileStorePathEnv points vLLM at the shared
	// FileStore rendezvous file for torch.distributed initialization.
	VLLMCheckpointRestoreFileStorePathEnv = "VLLM_CHECKPOINT_RESTORE_FILESTORE_PATH"

	// VLLMDistributedUseSplitGroupEnv is forced off for checkpoint restore so
	// CPU groups are pure Gloo and can be torn down without touching NCCL.
	VLLMDistributedUseSplitGroupEnv = "VLLM_DISTRIBUTED_USE_SPLIT_GROUP"

	// VLLMCheckpointRestoreFileStoreDir is created under the checkpoint root
	// on the shared PVC, outside the version artifact directory.
	VLLMCheckpointRestoreFileStoreDir = "vllm-filestore"

	// VLLMCheckpointRestoreFileStoreFile is the FileStore backing file.
	VLLMCheckpointRestoreFileStoreFile = "torch_pg"

	// SnapshotCompleteFile is written by the snapshot agent inside the
	// control volume when a checkpoint has completed successfully.
	SnapshotCompleteFile = "snapshot-complete"

	// RestoreCompleteFile is written by the snapshot agent inside the
	// control volume when a restore has completed and the workload may
	// resume.
	RestoreCompleteFile = "restore-complete"

	// ReadyForSnapshotFile is written by the workload inside the control
	// volume when the model is loaded and the workload is ready for a
	// checkpoint. Observed by the checkpoint job's kubelet readiness probe
	// on the worker container.
	ReadyForSnapshotFile = "ready-for-snapshot"
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

	hasControlEnv := false
	hasNCCLKvsEnv := false
	for _, e := range container.Env {
		if e.Name == SnapshotControlDirEnv {
			hasControlEnv = true
		}
		if e.Name == NCCLCheckpointKVSPathEnv {
			hasNCCLKvsEnv = true
		}
	}
	if !hasControlEnv {
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  SnapshotControlDirEnv,
			Value: SnapshotControlMountPath,
		})
	}
	if !hasNCCLKvsEnv {
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  NCCLCheckpointKVSPathEnv,
			Value: SnapshotControlMountPath + "/" + NCCLCheckpointKVSFile,
		})
	}
}

// VLLMCheckpointRestoreFileStorePathForTarget returns the shared FileStore path
// for a target container in a resolved checkpoint artifact location.
func VLLMCheckpointRestoreFileStorePathForTarget(location string, target string) string {
	location = strings.TrimRight(strings.TrimSpace(location), "/")
	if location == "" {
		return ""
	}
	var dir string
	versionDir := filepath.Dir(location)
	if filepath.Base(versionDir) == "versions" {
		dir = filepath.Join(
			filepath.Dir(versionDir),
			VLLMCheckpointRestoreFileStoreDir,
			filepath.Base(location),
		)
	} else {
		dir = filepath.Join(location, VLLMCheckpointRestoreFileStoreDir)
	}
	if target = strings.TrimSpace(target); target != "" {
		dir = filepath.Join(dir, target)
	}
	return filepath.Join(dir, VLLMCheckpointRestoreFileStoreFile)
}

// VLLMCheckpointRestoreFileStorePathForLocation returns the shared FileStore
// path for a resolved checkpoint artifact location.
func VLLMCheckpointRestoreFileStorePathForLocation(location string) string {
	return VLLMCheckpointRestoreFileStorePathForTarget(location, "")
}

// VLLMCheckpointRestoreFileStorePath returns the shared FileStore path for a
// resolved checkpoint artifact location.
func VLLMCheckpointRestoreFileStorePath(storage Storage) string {
	return VLLMCheckpointRestoreFileStorePathForTarget(storage.Location, "")
}

func EnsureVLLMCheckpointRestoreEnv(container *corev1.Container, storage Storage) {
	if container == nil {
		return
	}
	fileStorePath := VLLMCheckpointRestoreFileStorePathForTarget(
		storage.Location,
		container.Name,
	)
	if fileStorePath == "" {
		return
	}
	ensureEnv(container, VLLMCheckpointRestoreEnabledEnv, "1")
	ensureEnv(container, VLLMDistributedUseSplitGroupEnv, "0")
	ensureEnv(container, VLLMCheckpointRestoreFileStorePathEnv, fileStorePath)
}

func ensureEnv(container *corev1.Container, name string, value string) {
	for i := range container.Env {
		if container.Env[i].Name == name {
			container.Env[i].Value = value
			container.Env[i].ValueFrom = nil
			return
		}
	}
	container.Env = append(container.Env, corev1.EnvVar{Name: name, Value: value})
}
