package workload

import (
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

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
