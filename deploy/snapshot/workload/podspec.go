package workload

import corev1 "k8s.io/api/core/v1"

func PrepareRestorePodSpec(
	podSpec *corev1.PodSpec,
	container *corev1.Container,
	pvcName string,
	basePath string,
	seccompProfile string,
	placeholder bool,
) {
	injectLocalhostSeccompProfile(podSpec, seccompProfile)
	if pvcName != "" {
		injectCheckpointVolume(podSpec, pvcName)
	}
	if basePath != "" {
		injectCheckpointVolumeMount(container, basePath)
	}
	injectRestoreTUN(podSpec, container)
	if placeholder {
		setRestorePlaceholderCommand(container)
	}
}

func injectCheckpointVolume(podSpec *corev1.PodSpec, pvcName string) {
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

func injectCheckpointVolumeMount(container *corev1.Container, basePath string) {
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

func injectLocalhostSeccompProfile(podSpec *corev1.PodSpec, profile string) {
	if podSpec.SecurityContext == nil {
		podSpec.SecurityContext = &corev1.PodSecurityContext{}
	}
	podSpec.SecurityContext.SeccompProfile = &corev1.SeccompProfile{
		Type:             corev1.SeccompProfileTypeLocalhost,
		LocalhostProfile: &profile,
	}
}

func wrapWithCudaCheckpointLaunchJob(command []string, args []string) ([]string, []string) {
	wrappedArgs := make([]string, 0, len(command)+len(args)+1)
	wrappedArgs = append(wrappedArgs, "--launch-job")
	wrappedArgs = append(wrappedArgs, command...)
	wrappedArgs = append(wrappedArgs, args...)
	return []string{"cuda-checkpoint"}, wrappedArgs
}

func setRestorePlaceholderCommand(container *corev1.Container) {
	container.Command = []string{"sleep", "infinity"}
	container.Args = nil
}

func injectRestoreTUN(podSpec *corev1.PodSpec, container *corev1.Container) {
	charDevice := corev1.HostPathCharDev

	for _, volume := range podSpec.Volumes {
		if volume.Name == RestoreTUNVolumeName {
			goto mount
		}
	}
	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: RestoreTUNVolumeName,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{
				Path: "/dev/net/tun",
				Type: &charDevice,
			},
		},
	})

mount:
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
