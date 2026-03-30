package workload

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestNewCheckpointJob(t *testing.T) {
	job := NewCheckpointJob(&corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      map[string]string{"existing": "label"},
			Annotations: map[string]string{"existing": "annotation"},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyAlways,
			Containers: []corev1.Container{{
				Name:  "main",
				Image: "test:latest",
			}},
		},
	}, CheckpointJobOptions{
		Namespace:             "test-ns",
		Name:                  "test-job",
		SnapshotID:            "hash",
		Location:              "/checkpoints/hash",
		StorageType:           StorageTypePVC,
		ActiveDeadlineSeconds: ptr.To(int64(60)),
		TTLSecondsAfterFinish: ptr.To(int32(300)),
	})

	if job.Name != "test-job" || job.Namespace != "test-ns" {
		t.Fatalf("unexpected job identity: %#v", job.ObjectMeta)
	}
	if job.Labels[CheckpointHashLabel] != "hash" {
		t.Fatalf("expected checkpoint hash label on job: %#v", job.Labels)
	}
	if job.Spec.Template.Labels[CheckpointSourceLabel] != "true" {
		t.Fatalf("expected checkpoint source label on template: %#v", job.Spec.Template.Labels)
	}
	if job.Spec.Template.Annotations[CheckpointLocationAnnotation] != "/checkpoints/hash" {
		t.Fatalf("expected checkpoint location annotation on template: %#v", job.Spec.Template.Annotations)
	}
	if job.Spec.Template.Spec.RestartPolicy != corev1.RestartPolicyNever {
		t.Fatalf("expected restartPolicy Never, got %#v", job.Spec.Template.Spec.RestartPolicy)
	}
	if job.Spec.BackoffLimit == nil || *job.Spec.BackoffLimit != 0 {
		t.Fatalf("expected backoffLimit 0, got %#v", job.Spec.BackoffLimit)
	}
	if job.Spec.ActiveDeadlineSeconds == nil || *job.Spec.ActiveDeadlineSeconds != 60 {
		t.Fatalf("unexpected activeDeadlineSeconds: %#v", job.Spec.ActiveDeadlineSeconds)
	}
	if job.Spec.TTLSecondsAfterFinished == nil || *job.Spec.TTLSecondsAfterFinished != 300 {
		t.Fatalf("unexpected ttlSecondsAfterFinished: %#v", job.Spec.TTLSecondsAfterFinished)
	}
}
