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
	CheckpointStatusAnnotation          = "nvidia.com/snapshot-checkpoint-status"
	RestoreStatusAnnotation             = "nvidia.com/snapshot-restore-status"
	RestoreContainerIDAnnotation        = "nvidia.com/snapshot-restore-container-id"
	CheckpointVolumeName                = "checkpoint-storage"
	DefaultCheckpointArtifactVersion    = "1"
	DefaultCheckpointJobTTLSeconds      = int32(300)
	DefaultSeccompLocalhostProfile      = "profiles/block-iouring.json"
	StorageTypePVC                      = "pvc"

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

func ApplyRestoreTargetMetadata(labels map[string]string, annotations map[string]string, enabled bool, checkpointID string, artifactVersion string) {
	delete(labels, CheckpointSourceLabel)
	delete(labels, RestoreTargetLabel)
	delete(labels, CheckpointIDLabel)
	delete(annotations, CheckpointArtifactVersionAnnotation)
	delete(annotations, CheckpointStatusAnnotation)
	delete(annotations, RestoreStatusAnnotation)
	delete(annotations, RestoreContainerIDAnnotation)

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
