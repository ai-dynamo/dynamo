package workload

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestNewRestorePod(t *testing.T) {
	restorePod := NewRestorePod(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "worker",
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
	}, "test-ns", "hash", "/checkpoints/hash", StorageTypePVC)

	if restorePod.Name != "worker" || restorePod.Namespace != "test-ns" {
		t.Fatalf("unexpected restore pod identity: %#v", restorePod.ObjectMeta)
	}
	if restorePod.Labels[RestoreTargetLabel] != "true" {
		t.Fatalf("expected restore target label: %#v", restorePod.Labels)
	}
	if restorePod.Labels[CheckpointHashLabel] != "hash" {
		t.Fatalf("expected checkpoint hash label: %#v", restorePod.Labels)
	}
	if restorePod.Annotations[CheckpointLocationAnnotation] != "/checkpoints/hash" {
		t.Fatalf("expected checkpoint location annotation: %#v", restorePod.Annotations)
	}
	if restorePod.Spec.RestartPolicy != corev1.RestartPolicyNever {
		t.Fatalf("expected restartPolicy Never, got %#v", restorePod.Spec.RestartPolicy)
	}
}
