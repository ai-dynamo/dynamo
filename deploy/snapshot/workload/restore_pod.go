package workload

import corev1 "k8s.io/api/core/v1"

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
