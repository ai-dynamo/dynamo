package workload

import (
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
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

type CheckpointJobOptions struct {
	Namespace             string
	Name                  string
	SnapshotID            string
	Location              string
	StorageType           string
	ActiveDeadlineSeconds *int64
	TTLSecondsAfterFinish *int32
}

func NewCheckpointJob(podTemplate *corev1.PodTemplateSpec, opts CheckpointJobOptions) *batchv1.Job {
	podTemplate = podTemplate.DeepCopy()
	if podTemplate.Labels == nil {
		podTemplate.Labels = map[string]string{}
	}
	if podTemplate.Annotations == nil {
		podTemplate.Annotations = map[string]string{}
	}
	ApplyCheckpointSourceMetadata(podTemplate.Labels, podTemplate.Annotations, opts.SnapshotID, opts.Location, opts.StorageType)
	podTemplate.Spec.RestartPolicy = corev1.RestartPolicyNever

	return &batchv1.Job{
		TypeMeta: metav1.TypeMeta{APIVersion: "batch/v1", Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      opts.Name,
			Namespace: opts.Namespace,
			Labels: map[string]string{
				CheckpointHashLabel: opts.SnapshotID,
			},
		},
		Spec: batchv1.JobSpec{
			ActiveDeadlineSeconds:   opts.ActiveDeadlineSeconds,
			BackoffLimit:            ptr.To[int32](0),
			TTLSecondsAfterFinished: opts.TTLSecondsAfterFinish,
			Template:                *podTemplate,
		},
	}
}

func NewRestorePod(pod *corev1.Pod, namespace string, snapshotID string, location string, storageType string) *corev1.Pod {
	pod = pod.DeepCopy()
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	ApplyRestoreTargetMetadata(pod.Labels, pod.Annotations, true, snapshotID, location, storageType)
	pod.Namespace = namespace
	pod.Spec.RestartPolicy = corev1.RestartPolicyNever
	return pod
}
