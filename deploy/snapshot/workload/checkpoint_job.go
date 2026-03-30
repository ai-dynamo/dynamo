package workload

import (
	"fmt"

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
	SeccompProfile        string
	WrapLaunchJob         bool
}

func NewCheckpointJob(podTemplate *corev1.PodTemplateSpec, opts CheckpointJobOptions) (*batchv1.Job, error) {
	podTemplate = podTemplate.DeepCopy()
	if podTemplate.Labels == nil {
		podTemplate.Labels = map[string]string{}
	}
	if podTemplate.Annotations == nil {
		podTemplate.Annotations = map[string]string{}
	}
	applyCheckpointSourceMetadata(podTemplate.Labels, podTemplate.Annotations, opts.SnapshotID, opts.Location, opts.StorageType)
	podTemplate.Spec.RestartPolicy = corev1.RestartPolicyNever
	if opts.SeccompProfile != "" {
		injectLocalhostSeccompProfile(&podTemplate.Spec, opts.SeccompProfile)
	}
	if opts.WrapLaunchJob {
		if len(podTemplate.Spec.Containers) == 0 {
			return nil, fmt.Errorf("checkpoint job requires one worker container")
		}
		if len(podTemplate.Spec.Containers[0].Command) == 0 {
			return nil, fmt.Errorf("checkpoint job requires container.command when cuda-checkpoint launch-job wrapping is enabled")
		}
		podTemplate.Spec.Containers[0].Command, podTemplate.Spec.Containers[0].Args = wrapWithCudaCheckpointLaunchJob(
			podTemplate.Spec.Containers[0].Command,
			podTemplate.Spec.Containers[0].Args,
		)
	}

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
	}, nil
}
