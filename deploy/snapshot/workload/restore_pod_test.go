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
				Name:    "main",
				Image:   "test:latest",
				Command: []string{"python3", "-m", "dynamo.vllm"},
				Args:    []string{"--model", "Qwen"},
			}},
		},
	}, RestorePodOptions{
		Namespace:      "test-ns",
		SnapshotID:     "hash",
		Location:       "/checkpoints/hash",
		StorageType:    StorageTypePVC,
		CheckpointPVC:  "snapshot-pvc",
		CheckpointPath: "/checkpoints",
		SeccompProfile: DefaultSeccompLocalhostProfile,
	})

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
	if len(restorePod.Spec.Containers[0].Command) != 2 || restorePod.Spec.Containers[0].Command[0] != "sleep" || restorePod.Spec.Containers[0].Command[1] != "infinity" {
		t.Fatalf("expected placeholder command, got %#v", restorePod.Spec.Containers[0].Command)
	}
	if restorePod.Spec.Containers[0].Args != nil {
		t.Fatalf("expected restore args to be cleared: %#v", restorePod.Spec.Containers[0].Args)
	}
	if restorePod.Spec.SecurityContext == nil || restorePod.Spec.SecurityContext.SeccompProfile == nil {
		t.Fatalf("expected seccomp profile to be injected: %#v", restorePod.Spec.SecurityContext)
	}
	if len(restorePod.Spec.Volumes) != 2 {
		t.Fatalf("expected checkpoint and tun volumes, got %#v", restorePod.Spec.Volumes)
	}
	if len(restorePod.Spec.Containers[0].VolumeMounts) != 2 {
		t.Fatalf("expected checkpoint and tun mounts, got %#v", restorePod.Spec.Containers[0].VolumeMounts)
	}
}

func TestPrepareRestorePodSpec(t *testing.T) {
	podSpec := corev1.PodSpec{}
	container := corev1.Container{
		Command: []string{"python3", "-m", "dynamo.vllm"},
		Args:    []string{"--model", "Qwen"},
	}

	PrepareRestorePodSpec(&podSpec, &container, "snapshot-pvc", "/checkpoints", DefaultSeccompLocalhostProfile, true)
	PrepareRestorePodSpec(&podSpec, &container, "snapshot-pvc", "/checkpoints", DefaultSeccompLocalhostProfile, true)

	if podSpec.SecurityContext == nil || podSpec.SecurityContext.SeccompProfile == nil {
		t.Fatalf("expected seccomp profile to be injected: %#v", podSpec.SecurityContext)
	}
	if len(podSpec.Volumes) != 2 {
		t.Fatalf("expected checkpoint and tun volumes, got %#v", podSpec.Volumes)
	}
	if len(container.VolumeMounts) != 2 {
		t.Fatalf("expected checkpoint and tun mounts, got %#v", container.VolumeMounts)
	}
	if len(container.Command) != 2 || container.Command[0] != "sleep" || container.Command[1] != "infinity" {
		t.Fatalf("expected placeholder command, got %#v", container.Command)
	}
	if container.Args != nil {
		t.Fatalf("expected restore args to be cleared: %#v", container.Args)
	}
}
