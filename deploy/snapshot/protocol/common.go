// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"fmt"
	"strings"
)

const (
	CheckpointSourceLabel               = "nvidia.com/snapshot-is-checkpoint-source"
	CheckpointIDLabel                   = "nvidia.com/snapshot-checkpoint-id"
	RestoreTargetLabel                  = "nvidia.com/snapshot-is-restore-target"
	CheckpointArtifactVersionAnnotation = "nvidia.com/snapshot-artifact-version"

	// CheckpointContainersAnnotation names the ordered, comma-separated
	// list of workload container names to snapshot and restore for this
	// pod. The operator stamps it at pod-build time; the snapshot agent
	// reads it on both the checkpoint job's pod template and the restore
	// pod. Containers not in this list are left untouched.
	CheckpointContainersAnnotation = "nvidia.com/snapshot-containers"

	// CheckpointJobStatusAnnotation is set on the checkpoint Job (not the
	// pod) once all container checkpoints have reached a terminal state.
	CheckpointJobStatusAnnotation = "nvidia.com/snapshot-checkpoint-job-status"

	// Per-container status annotation prefixes. The snapshot agent appends
	// the container name to these keys so multiple containers can record
	// independent state on the same pod without clobbering each other.
	CheckpointStatusAnnotationPrefix   = "nvidia.com/snapshot-checkpoint-status."
	RestoreStatusAnnotationPrefix      = "nvidia.com/snapshot-restore-status."
	RestoreContainerIDAnnotationPrefix = "nvidia.com/snapshot-restore-container-id."

	CheckpointVolumeName             = "checkpoint-storage"
	DefaultCheckpointArtifactVersion = "1"
	DefaultCheckpointJobTTLSeconds   = int32(300)
	DefaultSeccompLocalhostProfile   = "profiles/block-iouring.json"
	StorageTypePVC                   = "pvc"

	CheckpointStatusCompleted = "completed"
	CheckpointStatusFailed    = "failed"
	RestoreStatusInProgress   = "in_progress"
	RestoreStatusCompleted    = "completed"
	RestoreStatusFailed       = "failed"
)

type Storage struct {
	Type     string
	Location string
	PVCName  string
	BasePath string
}

func ArtifactVersion(version string) string {
	version = strings.TrimSpace(version)
	if version == "" {
		return DefaultCheckpointArtifactVersion
	}
	return version
}

// ResolveCheckpointStorage resolves the pod-scoped checkpoint root on
// shared storage: <basePath>/<checkpointID>/versions/<version>. Per-container
// artifacts live under <podRoot>/containers/<containerName>/; callers derive
// those with ContainerCheckpointPath.
func ResolveCheckpointStorage(checkpointID string, version string, storage Storage) (Storage, error) {
	resolved, err := resolveStorageConfig(storage)
	if err != nil {
		return Storage{}, err
	}
	resolved.Location = strings.TrimRight(resolved.BasePath, "/") + "/" + checkpointID + "/versions/" + ArtifactVersion(version)
	return resolved, nil
}

func ResolveRestoreStorage(checkpointID string, version string, location string, storage Storage) (Storage, error) {
	resolved, err := resolveStorageConfig(storage)
	if err != nil {
		return Storage{}, err
	}
	location = strings.TrimSpace(location)
	if location == "" {
		return ResolveCheckpointStorage(checkpointID, version, storage)
	}
	resolved.Location = location
	return resolved, nil
}

// ContainerCheckpointPath returns the per-container checkpoint directory
// under the pod's checkpoint root: <podRoot>/containers/<name>.
func ContainerCheckpointPath(podRoot string, containerName string) string {
	return strings.TrimRight(podRoot, "/") + "/containers/" + containerName
}

// ParseCheckpointContainers parses the CheckpointContainersAnnotation into
// an ordered list of container names. Duplicates and empty entries are
// rejected; an empty/missing annotation is an error.
func ParseCheckpointContainers(annotations map[string]string) ([]string, error) {
	raw := strings.TrimSpace(annotations[CheckpointContainersAnnotation])
	if raw == "" {
		return nil, fmt.Errorf("missing %s annotation", CheckpointContainersAnnotation)
	}
	parts := strings.Split(raw, ",")
	names := make([]string, 0, len(parts))
	seen := make(map[string]struct{}, len(parts))
	for _, part := range parts {
		name := strings.TrimSpace(part)
		if name == "" {
			return nil, fmt.Errorf("%s has empty entry: %q", CheckpointContainersAnnotation, raw)
		}
		if _, dup := seen[name]; dup {
			return nil, fmt.Errorf("%s has duplicate entry %q", CheckpointContainersAnnotation, name)
		}
		seen[name] = struct{}{}
		names = append(names, name)
	}
	return names, nil
}

// FormatCheckpointContainers renders the CheckpointContainersAnnotation value
// for a list of container names.
func FormatCheckpointContainers(names []string) string {
	return strings.Join(names, ",")
}

func ApplyRestoreTargetMetadata(labels map[string]string, annotations map[string]string, enabled bool, checkpointID string, artifactVersion string) {
	delete(labels, CheckpointSourceLabel)
	delete(labels, RestoreTargetLabel)
	delete(labels, CheckpointIDLabel)
	delete(annotations, CheckpointArtifactVersionAnnotation)
	for key := range annotations {
		if strings.HasPrefix(key, CheckpointStatusAnnotationPrefix) ||
			strings.HasPrefix(key, RestoreStatusAnnotationPrefix) ||
			strings.HasPrefix(key, RestoreContainerIDAnnotationPrefix) {
			delete(annotations, key)
		}
	}

	if !enabled {
		return
	}

	labels[RestoreTargetLabel] = "true"
	if checkpointID != "" {
		labels[CheckpointIDLabel] = checkpointID
	}
	annotations[CheckpointArtifactVersionAnnotation] = ArtifactVersion(artifactVersion)
}

func applyCheckpointSourceMetadata(labels map[string]string, annotations map[string]string, checkpointID string, artifactVersion string) {
	delete(labels, RestoreTargetLabel)
	delete(labels, CheckpointIDLabel)
	delete(annotations, CheckpointArtifactVersionAnnotation)

	labels[CheckpointSourceLabel] = "true"
	if checkpointID != "" {
		labels[CheckpointIDLabel] = checkpointID
	}
	annotations[CheckpointArtifactVersionAnnotation] = ArtifactVersion(artifactVersion)
}

func resolveStorageConfig(storage Storage) (Storage, error) {
	storageType := strings.TrimSpace(storage.Type)
	if storageType == "" {
		storageType = StorageTypePVC
	}
	if storageType != StorageTypePVC {
		return Storage{}, fmt.Errorf("checkpoint storage type %q is not supported", storageType)
	}
	basePath := strings.TrimSpace(storage.BasePath)
	if basePath == "" {
		return Storage{}, fmt.Errorf("checkpoint base path is required")
	}
	return Storage{
		Type:     storageType,
		PVCName:  strings.TrimSpace(storage.PVCName),
		BasePath: strings.TrimRight(basePath, "/"),
	}, nil
}
