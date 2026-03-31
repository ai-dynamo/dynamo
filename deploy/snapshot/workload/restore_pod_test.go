package workload

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
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
	}, PodOptions{
		Namespace:       "test-ns",
		CheckpointID:    "hash",
		ArtifactVersion: "2",
		Storage: protocol.Storage{
			Type:     protocol.StorageTypePVC,
			PVCName:  "snapshot-pvc",
			BasePath: "/checkpoints",
		},
		SeccompProfile: protocol.DefaultSeccompLocalhostProfile,
	})

	if restorePod.Name != "worker" || restorePod.Namespace != "test-ns" {
		t.Fatalf("unexpected restore pod identity: %#v", restorePod.ObjectMeta)
	}
	if restorePod.Labels[protocol.RestoreTargetLabel] != "true" {
		t.Fatalf("expected restore target label: %#v", restorePod.Labels)
	}
	if restorePod.Labels[protocol.CheckpointIDLabel] != "hash" {
		t.Fatalf("expected checkpoint id label: %#v", restorePod.Labels)
	}
	if restorePod.Annotations[protocol.CheckpointArtifactVersionAnnotation] != "2" {
		t.Fatalf("expected checkpoint artifact version annotation: %#v", restorePod.Annotations)
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
	if len(restorePod.Spec.Volumes) != 1 {
		t.Fatalf("expected checkpoint volume, got %#v", restorePod.Spec.Volumes)
	}
	if len(restorePod.Spec.Containers[0].VolumeMounts) != 1 {
		t.Fatalf("expected checkpoint mount, got %#v", restorePod.Spec.Containers[0].VolumeMounts)
	}
}

func TestPrepareRestorePodSpec(t *testing.T) {
	podSpec := corev1.PodSpec{}
	container := corev1.Container{
		Command: []string{"python3", "-m", "dynamo.vllm"},
		Args:    []string{"--model", "Qwen"},
	}

	storage := protocol.Storage{
		Type:     protocol.StorageTypePVC,
		PVCName:  "snapshot-pvc",
		BasePath: "/checkpoints",
	}
	PrepareRestorePodSpec(&podSpec, &container, storage, protocol.DefaultSeccompLocalhostProfile, true)
	PrepareRestorePodSpec(&podSpec, &container, storage, protocol.DefaultSeccompLocalhostProfile, true)

	if podSpec.SecurityContext == nil || podSpec.SecurityContext.SeccompProfile == nil {
		t.Fatalf("expected seccomp profile to be injected: %#v", podSpec.SecurityContext)
	}
	if len(podSpec.Volumes) != 1 {
		t.Fatalf("expected checkpoint volume, got %#v", podSpec.Volumes)
	}
	if len(container.VolumeMounts) != 1 {
		t.Fatalf("expected checkpoint mount, got %#v", container.VolumeMounts)
	}
	if len(container.Command) != 2 || container.Command[0] != "sleep" || container.Command[1] != "infinity" {
		t.Fatalf("expected placeholder command, got %#v", container.Command)
	}
	if container.Args != nil {
		t.Fatalf("expected restore args to be cleared: %#v", container.Args)
	}
}

func TestValidateRestorePodSpec(t *testing.T) {
	profile := protocol.DefaultSeccompLocalhostProfile
	podSpec := &corev1.PodSpec{
		SecurityContext: &corev1.PodSecurityContext{
			SeccompProfile: &corev1.SeccompProfile{
				Type:             corev1.SeccompProfileTypeLocalhost,
				LocalhostProfile: &profile,
			},
		},
		Volumes: []corev1.Volume{{
			Name: protocol.CheckpointVolumeName,
		}},
	}
	container := &corev1.Container{
		Name: "main",
		VolumeMounts: []corev1.VolumeMount{{
			Name:      protocol.CheckpointVolumeName,
			MountPath: "/checkpoints",
		}},
	}
	storage := protocol.Storage{
		Type:     protocol.StorageTypePVC,
		PVCName:  "snapshot-pvc",
		BasePath: "/checkpoints",
	}

	if err := ValidateRestorePodSpec(podSpec, container, storage, protocol.DefaultSeccompLocalhostProfile); err != nil {
		t.Fatalf("expected restore pod spec to be valid, got %v", err)
	}

	badSpec := podSpec.DeepCopy()
	badSpec.Volumes = nil
	if err := ValidateRestorePodSpec(badSpec, container.DeepCopy(), storage, protocol.DefaultSeccompLocalhostProfile); err == nil || err.Error() != "missing checkpoint-storage volume" {
		t.Fatalf("expected missing volume error, got %v", err)
	}

	badContainer := container.DeepCopy()
	badContainer.VolumeMounts = nil
	if err := ValidateRestorePodSpec(podSpec.DeepCopy(), badContainer, storage, protocol.DefaultSeccompLocalhostProfile); err == nil || err.Error() != "missing checkpoint-storage mount at /checkpoints" {
		t.Fatalf("expected missing mount error, got %v", err)
	}

	badSpec = podSpec.DeepCopy()
	badSpec.SecurityContext = nil
	if err := ValidateRestorePodSpec(badSpec, container.DeepCopy(), storage, protocol.DefaultSeccompLocalhostProfile); err == nil || err.Error() != "missing localhost seccomp profile" {
		t.Fatalf("expected missing seccomp error, got %v", err)
	}
}
