package workload

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

import corev1 "k8s.io/api/core/v1"

type PodOptions struct {
	Namespace       string
	CheckpointID    string
	ArtifactVersion string
	Storage         protocol.Storage
	SeccompProfile  string
}

func NewRestorePod(pod *corev1.Pod, opts PodOptions) *corev1.Pod {
	pod = pod.DeepCopy()
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	protocol.ApplyRestoreTargetMetadata(pod.Labels, pod.Annotations, true, opts.CheckpointID, opts.ArtifactVersion)
	PrepareRestorePodSpec(&pod.Spec, &pod.Spec.Containers[0], opts.Storage, opts.SeccompProfile, true)
	pod.Namespace = opts.Namespace
	pod.Spec.RestartPolicy = corev1.RestartPolicyNever
	return pod
}

func PrepareRestorePodSpec(
	podSpec *corev1.PodSpec,
	container *corev1.Container,
	storage protocol.Storage,
	seccompProfile string,
	placeholder bool,
) {
	protocol.EnsureLocalhostSeccompProfile(podSpec, seccompProfile)
	if storage.PVCName != "" {
		injectCheckpointVolume(podSpec, storage.PVCName)
	}
	if storage.BasePath != "" {
		injectCheckpointVolumeMount(container, storage.BasePath)
	}
	if placeholder {
		container.Command = []string{"sleep", "infinity"}
		container.Args = nil
	}
}

func ValidateRestorePodSpec(
	podSpec *corev1.PodSpec,
	storage protocol.Storage,
	seccompProfile string,
) error {
	if podSpec == nil {
		return fmt.Errorf("pod spec is nil")
	}
	container := restoreTargetContainer(podSpec)
	if container == nil {
		return fmt.Errorf("restore target container is missing")
	}
	if storage.PVCName != "" {
		hasVolume := false
		for _, volume := range podSpec.Volumes {
			if volume.Name == protocol.CheckpointVolumeName {
				hasVolume = true
				break
			}
		}
		if !hasVolume {
			return fmt.Errorf("missing %s volume", protocol.CheckpointVolumeName)
		}
	}
	if storage.BasePath != "" {
		hasMount := false
		for _, mount := range container.VolumeMounts {
			if mount.Name == protocol.CheckpointVolumeName && mount.MountPath == storage.BasePath {
				hasMount = true
				break
			}
		}
		if !hasMount {
			return fmt.Errorf("missing %s mount at %s", protocol.CheckpointVolumeName, storage.BasePath)
		}
	}
	if seccompProfile == "" {
		return nil
	}
	if podSpec.SecurityContext == nil || podSpec.SecurityContext.SeccompProfile == nil {
		return fmt.Errorf("missing localhost seccomp profile")
	}
	profile := podSpec.SecurityContext.SeccompProfile
	if profile.Type != corev1.SeccompProfileTypeLocalhost || profile.LocalhostProfile == nil || *profile.LocalhostProfile != seccompProfile {
		return fmt.Errorf("expected localhost seccomp profile %q", seccompProfile)
	}
	return nil
}

func restoreTargetContainer(podSpec *corev1.PodSpec) *corev1.Container {
	var fallback *corev1.Container
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == "main" {
			return &podSpec.Containers[i]
		}
		if fallback == nil {
			fallback = &podSpec.Containers[i]
		}
	}
	return fallback
}

func injectCheckpointVolume(podSpec *corev1.PodSpec, pvcName string) {
	for _, volume := range podSpec.Volumes {
		if volume.Name == protocol.CheckpointVolumeName {
			return
		}
	}

	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: protocol.CheckpointVolumeName,
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvcName,
			},
		},
	})
}

func injectCheckpointVolumeMount(container *corev1.Container, basePath string) {
	for _, mount := range container.VolumeMounts {
		if mount.Name == protocol.CheckpointVolumeName {
			return
		}
	}

	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
		Name:      protocol.CheckpointVolumeName,
		MountPath: basePath,
	})
}
