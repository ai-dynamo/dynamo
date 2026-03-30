package workload

const (
	CheckpointSourceLabel          = "nvidia.com/snapshot-is-checkpoint-source"
	CheckpointIDLabel              = "nvidia.com/snapshot-checkpoint-id"
	RestoreTargetLabel             = "nvidia.com/snapshot-is-restore-target"
	CheckpointLocationAnnotation   = "nvidia.com/snapshot-checkpoint-location"
	CheckpointStorageAnnotation    = "nvidia.com/snapshot-checkpoint-storage-type"
	CheckpointStatusAnnotation     = "nvidia.com/snapshot-checkpoint-status"
	RestoreStatusAnnotation        = "nvidia.com/snapshot-restore-status"
	RestoreContainerIDAnnotation   = "nvidia.com/snapshot-restore-container-id"
	CheckpointVolumeName           = "checkpoint-storage"
	DefaultSeccompLocalhostProfile = "profiles/block-iouring.json"
	StorageTypePVC                 = "pvc"
)

func applyCheckpointSourceMetadata(labels map[string]string, annotations map[string]string, checkpointID string, location string, storageType string) {
	delete(labels, RestoreTargetLabel)
	delete(labels, CheckpointIDLabel)
	delete(annotations, CheckpointLocationAnnotation)
	delete(annotations, CheckpointStorageAnnotation)

	labels[CheckpointSourceLabel] = "true"
	if checkpointID != "" {
		labels[CheckpointIDLabel] = checkpointID
	}
	if location != "" {
		annotations[CheckpointLocationAnnotation] = location
	}
	if storageType != "" {
		annotations[CheckpointStorageAnnotation] = storageType
	}
}

func ApplyRestoreTargetMetadata(labels map[string]string, annotations map[string]string, enabled bool, checkpointID string, location string, storageType string) {
	delete(labels, CheckpointSourceLabel)
	delete(labels, RestoreTargetLabel)
	delete(labels, CheckpointIDLabel)
	delete(annotations, CheckpointLocationAnnotation)
	delete(annotations, CheckpointStorageAnnotation)

	if !enabled {
		return
	}

	labels[RestoreTargetLabel] = "true"
	if checkpointID != "" {
		labels[CheckpointIDLabel] = checkpointID
	}
	if location != "" {
		annotations[CheckpointLocationAnnotation] = location
	}
	if storageType != "" {
		annotations[CheckpointStorageAnnotation] = storageType
	}
}
