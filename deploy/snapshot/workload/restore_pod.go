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
