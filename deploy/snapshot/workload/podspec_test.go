package workload

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
)

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
