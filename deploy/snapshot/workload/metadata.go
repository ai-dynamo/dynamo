package workload

const (
	CheckpointSourceLabel          = "nvidia.com/snapshot-is-checkpoint-source"
	CheckpointHashLabel            = "nvidia.com/snapshot-checkpoint-hash"
	RestoreTargetLabel             = "nvidia.com/snapshot-is-restore-target"
	CheckpointLocationAnnotation   = "nvidia.com/snapshot-checkpoint-location"
	CheckpointStorageAnnotation    = "nvidia.com/snapshot-checkpoint-storage-type"
	CheckpointStatusAnnotation     = "nvidia.com/snapshot-checkpoint-status"
	RestoreStatusAnnotation        = "nvidia.com/snapshot-restore-status"
	RestoreContainerIDAnnotation   = "nvidia.com/snapshot-restore-container-id"
	CheckpointVolumeName           = "checkpoint-storage"
	RestoreTUNVolumeName           = "host-dev-net-tun"
	DefaultSeccompLocalhostProfile = "profiles/block-iouring.json"
	StorageTypePVC                 = "pvc"
)

func ApplyCheckpointSourceMetadata(labels map[string]string, annotations map[string]string, hash string, location string, storageType string) {
	delete(labels, RestoreTargetLabel)
	delete(labels, CheckpointHashLabel)
	delete(annotations, CheckpointLocationAnnotation)
	delete(annotations, CheckpointStorageAnnotation)

	labels[CheckpointSourceLabel] = "true"
	if hash != "" {
		labels[CheckpointHashLabel] = hash
	}
	if location != "" {
		annotations[CheckpointLocationAnnotation] = location
	}
	if storageType != "" {
		annotations[CheckpointStorageAnnotation] = storageType
	}
}

func ApplyRestoreTargetMetadata(labels map[string]string, annotations map[string]string, enabled bool, hash string, location string, storageType string) {
	delete(labels, CheckpointSourceLabel)
	delete(labels, RestoreTargetLabel)
	delete(labels, CheckpointHashLabel)
	delete(annotations, CheckpointLocationAnnotation)
	delete(annotations, CheckpointStorageAnnotation)

	if !enabled {
		return
	}

	labels[RestoreTargetLabel] = "true"
	if hash != "" {
		labels[CheckpointHashLabel] = hash
	}
	if location != "" {
		annotations[CheckpointLocationAnnotation] = location
	}
	if storageType != "" {
		annotations[CheckpointStorageAnnotation] = storageType
	}
}
