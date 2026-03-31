package workload

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
	DefaultSeccompLocalhostProfile      = "profiles/block-iouring.json"
	StorageTypePVC                      = "pvc"
)

func applyCheckpointSourceMetadata(labels map[string]string, annotations map[string]string, checkpointID string, artifactVersion string) {
	delete(labels, RestoreTargetLabel)
	delete(labels, CheckpointIDLabel)
	delete(annotations, CheckpointArtifactVersionAnnotation)

	labels[CheckpointSourceLabel] = "true"
	if checkpointID != "" {
		labels[CheckpointIDLabel] = checkpointID
	}
	artifactVersion = strings.TrimSpace(artifactVersion)
	if artifactVersion == "" {
		artifactVersion = DefaultCheckpointArtifactVersion
	}
	annotations[CheckpointArtifactVersionAnnotation] = artifactVersion
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
	artifactVersion = strings.TrimSpace(artifactVersion)
	if artifactVersion == "" {
		artifactVersion = DefaultCheckpointArtifactVersion
	}
	annotations[CheckpointArtifactVersionAnnotation] = artifactVersion
}
