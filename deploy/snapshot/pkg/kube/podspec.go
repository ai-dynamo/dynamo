package kube

import (
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

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

func MainContainer(podSpec *corev1.PodSpec) (*corev1.Container, error) {
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == "main" {
			return &podSpec.Containers[i], nil
		}
	}
	if len(podSpec.Containers) == 0 {
		return nil, fmt.Errorf("pod spec has no containers")
	}
	return &podSpec.Containers[0], nil
}

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

func PrepareRestoreTargetPodSpec(podSpec *corev1.PodSpec, seccompProfile string) error {
	mainContainer, err := MainContainer(podSpec)
	if err != nil {
		return err
	}
	if strings.TrimSpace(seccompProfile) == "" {
		seccompProfile = DefaultSeccompLocalhostProfile
	}

	SetRestorePlaceholderCommand(mainContainer)
	InjectLocalhostSeccompProfile(podSpec, seccompProfile)
	InjectRestoreTUN(podSpec, mainContainer)
	return nil
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

func ValidateRestoreTargetPod(pod *corev1.Pod, storage SnapshotStorage, seccompProfile string) error {
	mainContainer, err := MainContainer(&pod.Spec)
	if err != nil {
		return err
	}
	if strings.TrimSpace(seccompProfile) == "" {
		seccompProfile = DefaultSeccompLocalhostProfile
	}

	if !hasCheckpointVolume(&pod.Spec, storage.PVCName) {
		return fmt.Errorf("missing checkpoint PVC volume %q", storage.PVCName)
	}
	if !hasVolumeMount(mainContainer, CheckpointVolumeName, storage.BasePath) {
		return fmt.Errorf("main container is missing checkpoint mount %q", storage.BasePath)
	}
	if !hasLocalhostSeccompProfile(&pod.Spec, seccompProfile) {
		return fmt.Errorf("pod is missing localhost seccomp profile %q", seccompProfile)
	}
	if !hasRestorePlaceholderCommand(mainContainer) {
		return fmt.Errorf("main container command must be sleep infinity before restore")
	}
	if !hasRestoreTUN(&pod.Spec, mainContainer) {
		return fmt.Errorf("pod is missing /dev/net/tun restore mount")
	}
	return nil
}

func PrepareCheckpointPodTemplate(
	podTemplate *corev1.PodTemplateSpec,
	snapshotID string,
	artifactVersion string,
	storage SnapshotStorage,
	disableCudaCheckpointJobFile bool,
	seccompProfile string,
) error {
	mainContainer, err := MainContainer(&podTemplate.Spec)
	if err != nil {
		return err
	}
	if strings.TrimSpace(seccompProfile) == "" {
		seccompProfile = DefaultSeccompLocalhostProfile
	}

	if !disableCudaCheckpointJobFile {
		if len(mainContainer.Command) == 0 {
			return fmt.Errorf("main container command is required when cuda-checkpoint --launch-job wrapping is enabled")
		}
		mainContainer.Command, mainContainer.Args = WrapWithCudaCheckpointLaunchJob(mainContainer.Command, mainContainer.Args)
	}

	InjectCheckpointVolume(&podTemplate.Spec, storage.PVCName)
	InjectCheckpointVolumeMount(mainContainer, storage.BasePath)
	InjectLocalhostSeccompProfile(&podTemplate.Spec, seccompProfile)

	if podTemplate.Labels == nil {
		podTemplate.Labels = map[string]string{}
	}
	if podTemplate.Annotations == nil {
		podTemplate.Annotations = map[string]string{}
	}
	ApplyCheckpointSourceMetadata(
		podTemplate.Labels,
		podTemplate.Annotations,
		snapshotID,
		storage.CheckpointLocation(snapshotID, artifactVersion),
		StorageTypePVC,
	)

	return nil
}

func PrepareStandaloneRestorePod(
	pod *corev1.Pod,
	snapshotID string,
	artifactVersion string,
	storage SnapshotStorage,
	seccompProfile string,
) error {
	mainContainer, err := MainContainer(&pod.Spec)
	if err != nil {
		return err
	}

	InjectCheckpointVolume(&pod.Spec, storage.PVCName)
	InjectCheckpointVolumeMount(mainContainer, storage.BasePath)
	if err := PrepareRestoreTargetPodSpec(&pod.Spec, seccompProfile); err != nil {
		return err
	}

	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	ApplyRestoreTargetMetadata(
		pod.Labels,
		pod.Annotations,
		true,
		snapshotID,
		storage.CheckpointLocation(snapshotID, artifactVersion),
		StorageTypePVC,
	)
	return nil
}

func hasCheckpointVolume(podSpec *corev1.PodSpec, pvcName string) bool {
	for _, volume := range podSpec.Volumes {
		if volume.Name != CheckpointVolumeName || volume.PersistentVolumeClaim == nil {
			continue
		}
		if volume.PersistentVolumeClaim.ClaimName == pvcName {
			return true
		}
	}
	return false
}

func hasVolumeMount(container *corev1.Container, volumeName string, mountPath string) bool {
	for _, mount := range container.VolumeMounts {
		if mount.Name == volumeName && mount.MountPath == mountPath {
			return true
		}
	}
	return false
}

func hasLocalhostSeccompProfile(podSpec *corev1.PodSpec, profile string) bool {
	if podSpec.SecurityContext == nil || podSpec.SecurityContext.SeccompProfile == nil {
		return false
	}
	seccomp := podSpec.SecurityContext.SeccompProfile
	return seccomp.Type == corev1.SeccompProfileTypeLocalhost &&
		seccomp.LocalhostProfile != nil &&
		*seccomp.LocalhostProfile == profile
}

func hasRestorePlaceholderCommand(container *corev1.Container) bool {
	return len(container.Command) == 2 &&
		container.Command[0] == "sleep" &&
		container.Command[1] == "infinity" &&
		len(container.Args) == 0
}

func hasRestoreTUN(podSpec *corev1.PodSpec, container *corev1.Container) bool {
	hasVolume := false
	for _, volume := range podSpec.Volumes {
		if volume.Name == RestoreTUNVolumeName && volume.HostPath != nil && volume.HostPath.Path == "/dev/net/tun" {
			hasVolume = true
			break
		}
	}
	return hasVolume && hasVolumeMount(container, RestoreTUNVolumeName, "/dev/net/tun")
}
