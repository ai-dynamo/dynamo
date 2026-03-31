// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import "strings"

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

func NormalizeArtifactVersion(version string) string {
	version = strings.TrimSpace(version)
	if version == "" {
		return DefaultCheckpointArtifactVersion
	}
	return version
}

func ArtifactVersionFromAnnotations(annotations map[string]string) string {
	if annotations == nil {
		return DefaultCheckpointArtifactVersion
	}
	return NormalizeArtifactVersion(annotations[CheckpointArtifactVersionAnnotation])
}

func CheckpointJobName(checkpointID string, artifactVersion string) string {
	return "checkpoint-job-" + checkpointID + "-" + NormalizeArtifactVersion(artifactVersion)
}

func applyCheckpointSourceMetadata(labels map[string]string, annotations map[string]string, checkpointID string, artifactVersion string) {
	delete(labels, RestoreTargetLabel)
	delete(labels, CheckpointIDLabel)
	delete(annotations, CheckpointArtifactVersionAnnotation)

	labels[CheckpointSourceLabel] = "true"
	if checkpointID != "" {
		labels[CheckpointIDLabel] = checkpointID
	}
	annotations[CheckpointArtifactVersionAnnotation] = NormalizeArtifactVersion(artifactVersion)
}

func ApplyRestoreTargetMetadata(labels map[string]string, annotations map[string]string, enabled bool, checkpointID string, artifactVersion string) {
	delete(labels, CheckpointSourceLabel)
	delete(labels, RestoreTargetLabel)
	delete(labels, CheckpointIDLabel)
	delete(annotations, CheckpointArtifactVersionAnnotation)

	if !enabled {
		return
	}

	labels[RestoreTargetLabel] = "true"
	if checkpointID != "" {
		labels[CheckpointIDLabel] = checkpointID
	}
	annotations[CheckpointArtifactVersionAnnotation] = NormalizeArtifactVersion(artifactVersion)
}
