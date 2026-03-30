package workload

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
)

func TestInjectCheckpointVolume(t *testing.T) {
	podSpec := corev1.PodSpec{}
	InjectCheckpointVolume(&podSpec, "snapshot-pvc")
	InjectCheckpointVolume(&podSpec, "snapshot-pvc")

	if len(podSpec.Volumes) != 1 {
		t.Fatalf("expected one volume, got %d", len(podSpec.Volumes))
	}
	if podSpec.Volumes[0].Name != CheckpointVolumeName {
		t.Fatalf("unexpected volume: %#v", podSpec.Volumes[0])
	}
	if podSpec.Volumes[0].PersistentVolumeClaim == nil || podSpec.Volumes[0].PersistentVolumeClaim.ClaimName != "snapshot-pvc" {
		t.Fatalf("unexpected volume source: %#v", podSpec.Volumes[0])
	}
}

func TestInjectCheckpointVolumeMount(t *testing.T) {
	container := corev1.Container{}
	InjectCheckpointVolumeMount(&container, "/checkpoints")
	InjectCheckpointVolumeMount(&container, "/checkpoints")

	if len(container.VolumeMounts) != 1 {
		t.Fatalf("expected one volume mount, got %d", len(container.VolumeMounts))
	}
	if container.VolumeMounts[0].Name != CheckpointVolumeName || container.VolumeMounts[0].MountPath != "/checkpoints" {
		t.Fatalf("unexpected volume mount: %#v", container.VolumeMounts[0])
	}
}

func TestInjectLocalhostSeccompProfile(t *testing.T) {
	podSpec := corev1.PodSpec{}
	InjectLocalhostSeccompProfile(&podSpec, DefaultSeccompLocalhostProfile)

	if podSpec.SecurityContext == nil || podSpec.SecurityContext.SeccompProfile == nil {
		t.Fatalf("expected seccomp profile to be injected: %#v", podSpec.SecurityContext)
	}
	if podSpec.SecurityContext.SeccompProfile.Type != corev1.SeccompProfileTypeLocalhost {
		t.Fatalf("unexpected seccomp type: %#v", podSpec.SecurityContext.SeccompProfile)
	}
	if podSpec.SecurityContext.SeccompProfile.LocalhostProfile == nil || *podSpec.SecurityContext.SeccompProfile.LocalhostProfile != DefaultSeccompLocalhostProfile {
		t.Fatalf("unexpected localhost profile: %#v", podSpec.SecurityContext.SeccompProfile)
	}
}

func TestWrapWithCudaCheckpointLaunchJob(t *testing.T) {
	command, args := WrapWithCudaCheckpointLaunchJob([]string{"python3", "-m", "dynamo.vllm"}, []string{"--model", "Qwen"})

	if len(command) != 1 || command[0] != "cuda-checkpoint" {
		t.Fatalf("unexpected command: %#v", command)
	}
	expected := []string{"--launch-job", "python3", "-m", "dynamo.vllm", "--model", "Qwen"}
	if len(args) != len(expected) {
		t.Fatalf("unexpected args: %#v", args)
	}
	for index := range expected {
		if args[index] != expected[index] {
			t.Fatalf("unexpected args: %#v", args)
		}
	}
}

func TestSetRestorePlaceholderCommand(t *testing.T) {
	container := corev1.Container{
		Command: []string{"python3"},
		Args:    []string{"-m", "dynamo.vllm"},
	}

	SetRestorePlaceholderCommand(&container)

	if len(container.Command) != 2 || container.Command[0] != "sleep" || container.Command[1] != "infinity" {
		t.Fatalf("unexpected command: %#v", container.Command)
	}
	if container.Args != nil {
		t.Fatalf("expected restore args to be cleared: %#v", container.Args)
	}
}

func TestInjectRestoreTUN(t *testing.T) {
	podSpec := corev1.PodSpec{}
	container := corev1.Container{}

	InjectRestoreTUN(&podSpec, &container)
	InjectRestoreTUN(&podSpec, &container)

	if len(podSpec.Volumes) != 1 {
		t.Fatalf("expected one restore TUN volume, got %d", len(podSpec.Volumes))
	}
	if podSpec.Volumes[0].Name != RestoreTUNVolumeName {
		t.Fatalf("unexpected restore TUN volume: %#v", podSpec.Volumes[0])
	}
	if podSpec.Volumes[0].HostPath == nil || podSpec.Volumes[0].HostPath.Path != "/dev/net/tun" {
		t.Fatalf("unexpected restore TUN source: %#v", podSpec.Volumes[0])
	}
	if len(container.VolumeMounts) != 1 {
		t.Fatalf("expected one restore TUN mount, got %d", len(container.VolumeMounts))
	}
	if container.VolumeMounts[0].Name != RestoreTUNVolumeName || container.VolumeMounts[0].MountPath != "/dev/net/tun" {
		t.Fatalf("unexpected restore TUN mount: %#v", container.VolumeMounts[0])
	}
}
