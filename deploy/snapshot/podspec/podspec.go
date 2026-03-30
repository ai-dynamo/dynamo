package podspec

import corev1 "k8s.io/api/core/v1"

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

func InjectCheckpointVolume(podSpec *corev1.PodSpec, pvcName string) {
	for _, volume := range podSpec.Volumes {
		if volume.Name == CheckpointVolumeName {
			return
		}
	}

	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: CheckpointVolumeName,
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvcName,
			},
		},
	})
}

func InjectCheckpointVolumeMount(container *corev1.Container, basePath string) {
	for _, mount := range container.VolumeMounts {
		if mount.Name == CheckpointVolumeName {
			return
		}
	}

	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
		Name:      CheckpointVolumeName,
		MountPath: basePath,
	})
}

func InjectLocalhostSeccompProfile(podSpec *corev1.PodSpec, profile string) {
	if podSpec.SecurityContext == nil {
		podSpec.SecurityContext = &corev1.PodSecurityContext{}
	}
	podSpec.SecurityContext.SeccompProfile = &corev1.SeccompProfile{
		Type:             corev1.SeccompProfileTypeLocalhost,
		LocalhostProfile: &profile,
	}
}

func WrapWithCudaCheckpointLaunchJob(command []string, args []string) ([]string, []string) {
	wrappedArgs := make([]string, 0, len(command)+len(args)+1)
	wrappedArgs = append(wrappedArgs, "--launch-job")
	wrappedArgs = append(wrappedArgs, command...)
	wrappedArgs = append(wrappedArgs, args...)
	return []string{"cuda-checkpoint"}, wrappedArgs
}

func SetRestorePlaceholderCommand(container *corev1.Container) {
	container.Command = []string{"sleep", "infinity"}
	container.Args = nil
}

func InjectRestoreTUN(podSpec *corev1.PodSpec, container *corev1.Container) {
	charDevice := corev1.HostPathCharDev

	hasVolume := false
	for _, volume := range podSpec.Volumes {
		if volume.Name == RestoreTUNVolumeName {
			hasVolume = true
			break
		}
	}
	if !hasVolume {
		podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
			Name: RestoreTUNVolumeName,
			VolumeSource: corev1.VolumeSource{
				HostPath: &corev1.HostPathVolumeSource{
					Path: "/dev/net/tun",
					Type: &charDevice,
				},
			},
		})
	}

	for _, mount := range container.VolumeMounts {
		if mount.Name == RestoreTUNVolumeName {
			return
		}
	}
	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
		Name:      RestoreTUNVolumeName,
		MountPath: "/dev/net/tun",
	})
}
