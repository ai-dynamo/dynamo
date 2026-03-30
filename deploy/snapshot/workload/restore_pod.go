package workload

import corev1 "k8s.io/api/core/v1"

type RestorePodOptions struct {
	Namespace      string
	SnapshotID     string
	Location       string
	StorageType    string
	CheckpointPVC  string
	CheckpointPath string
	SeccompProfile string
}

func NewRestorePod(pod *corev1.Pod, opts RestorePodOptions) *corev1.Pod {
	pod = pod.DeepCopy()
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	ApplyRestoreTargetMetadata(pod.Labels, pod.Annotations, true, opts.SnapshotID, opts.Location, opts.StorageType)
	PrepareRestorePodSpec(&pod.Spec, &pod.Spec.Containers[0], opts.CheckpointPVC, opts.CheckpointPath, opts.SeccompProfile, true)
	pod.Namespace = opts.Namespace
	pod.Spec.RestartPolicy = corev1.RestartPolicyNever
	return pod
}

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
	if placeholder {
		container.Command = []string{"sleep", "infinity"}
		container.Args = nil
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
