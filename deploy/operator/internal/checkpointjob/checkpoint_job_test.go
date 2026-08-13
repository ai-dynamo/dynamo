// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package checkpointjob

import (
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func requireCheckpointContainer(t *testing.T, containers []corev1.Container, name string) *corev1.Container {
	t.Helper()
	for i := range containers {
		if containers[i].Name == name {
			return &containers[i]
		}
	}
	t.Fatalf("container %q not found", name)
	return nil
}

func requireStableLaunchJobWrapper(t *testing.T, container *corev1.Container, original []string) {
	t.Helper()
	if strings.Join(container.Command, "|") != "cuda-checkpoint" {
		t.Fatalf("expected cuda-checkpoint wrapper command, got %#v", container.Command)
	}
	if len(container.Args) < 6 {
		t.Fatalf("launch-job wrapper args too short: %#v", container.Args)
	}
	if container.Args[0] != "--launch-job" || container.Args[1] != "/bin/sh" || container.Args[2] != "-c" || container.Args[4] != "dynamo-cuda-checkpoint" {
		t.Fatalf("unexpected launch-job wrapper prefix: %#v", container.Args[:6])
	}
	if container.Args[5] != CUDAJobFilePath {
		t.Fatalf("stable job file = %q, want %q", container.Args[5], CUDAJobFilePath)
	}
	if got := container.Args[6:]; strings.Join(got, "|") != strings.Join(original, "|") {
		t.Fatalf("original command = %#v, want %#v", got, original)
	}
}

func TestNewCheckpointJob(t *testing.T) {
	job, err := NewCheckpointJob(&corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{"existing": "label"},
			Annotations: map[string]string{
				"existing": "annotation",
			},
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
	}, CheckpointJobOptions{
		Namespace:             "test-ns",
		TargetContainer:       "main",
		CheckpointID:          "hash",
		ArtifactVersion:       "2",
		SeccompProfile:        DefaultSeccompLocalhostProfile,
		Name:                  "test-job",
		ActiveDeadlineSeconds: ptr.To(int64(60)),
		TTLSecondsAfterFinish: ptr.To(int32(300)),
		WrapLaunchJob:         true,
	})
	if err != nil {
		t.Fatalf("expected checkpoint job, got error: %v", err)
	}

	if job.Name != "test-job" || job.Namespace != "test-ns" {
		t.Fatalf("unexpected job identity: %#v", job.ObjectMeta)
	}
	if job.Labels[CheckpointIDLabel] != "hash" {
		t.Fatalf("expected checkpoint hash label on job: %#v", job.Labels)
	}
	if job.Spec.Template.Labels[CheckpointSourceLabel] != "true" {
		t.Fatalf("expected checkpoint source label on template: %#v", job.Spec.Template.Labels)
	}
	if job.Spec.Template.Annotations[CheckpointArtifactVersionAnnotation] != "2" {
		t.Fatalf("expected checkpoint artifact version annotation on template: %#v", job.Spec.Template.Annotations)
	}
	// The capture Job must not stamp the target-containers annotation; the target flows via
	// CheckpointJobOptions.TargetContainer. A regression that re-introduces the capture stamp fails here.
	if _, ok := job.Spec.Template.Annotations[TargetContainersAnnotation]; ok {
		t.Fatalf("capture job must not stamp %s annotation: %#v", TargetContainersAnnotation, job.Spec.Template.Annotations)
	}
	if len(job.Spec.Template.Spec.Volumes) != 1 || job.Spec.Template.Spec.Volumes[0].Name != SnapshotControlVolumeName {
		t.Fatalf("expected only %s volume, got %#v", SnapshotControlVolumeName, job.Spec.Template.Spec.Volumes)
	}
	main := &job.Spec.Template.Spec.Containers[0]
	if len(main.VolumeMounts) != 1 || main.VolumeMounts[0].MountPath != SnapshotControlMountPath {
		t.Fatalf("expected only %s mount at %s, got %#v", SnapshotControlVolumeName, SnapshotControlMountPath, main.VolumeMounts)
	}
	if main.VolumeMounts[0].SubPath != "main" {
		t.Fatalf("expected control mount subPath=main, got %#v", main.VolumeMounts[0])
	}
	if main.ReadinessProbe == nil || main.ReadinessProbe.Exec == nil {
		t.Fatalf("expected ready-file readiness probe, got %#v", main.ReadinessProbe)
	}
	expectedProbe := []string{"cat", SnapshotControlMountPath + "/" + ReadyForSnapshotFile}
	if strings.Join(main.ReadinessProbe.Exec.Command, " ") != strings.Join(expectedProbe, " ") {
		t.Fatalf("expected readiness probe %#v, got %#v", expectedProbe, main.ReadinessProbe.Exec.Command)
	}
	if main.LivenessProbe != nil || main.StartupProbe != nil {
		t.Fatalf("expected liveness and startup probes cleared, got liveness=%#v startup=%#v", main.LivenessProbe, main.StartupProbe)
	}
	if job.Spec.Template.Spec.RestartPolicy != corev1.RestartPolicyNever {
		t.Fatalf("expected restartPolicy Never, got %#v", job.Spec.Template.Spec.RestartPolicy)
	}
	if job.Spec.Template.Spec.SecurityContext == nil || job.Spec.Template.Spec.SecurityContext.SeccompProfile == nil {
		t.Fatalf("expected seccomp profile to be injected: %#v", job.Spec.Template.Spec.SecurityContext)
	}
	requireStableLaunchJobWrapper(t, main, []string{"python3", "-m", "dynamo.vllm", "--model", "Qwen"})
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

func TestNewCheckpointJobWrapsTargetContainer(t *testing.T) {
	job, err := NewCheckpointJob(&corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "sidecar", Command: []string{"sleep"}, Args: []string{"infinity"}},
				{Name: "worker", Command: []string{"python3", "-m", "dynamo.vllm"}, Args: []string{"--model", "Qwen"}},
			},
		},
	}, CheckpointJobOptions{
		Namespace:             "test-ns",
		TargetContainer:       "worker",
		CheckpointID:          "hash",
		ArtifactVersion:       "2",
		Name:                  "test-job",
		TTLSecondsAfterFinish: ptr.To(int32(300)),
		WrapLaunchJob:         true,
	})
	if err != nil {
		t.Fatalf("expected checkpoint job, got error: %v", err)
	}

	worker := requireCheckpointContainer(t, job.Spec.Template.Spec.Containers, "worker")
	requireStableLaunchJobWrapper(t, worker, []string{"python3", "-m", "dynamo.vllm", "--model", "Qwen"})

	sidecar := requireCheckpointContainer(t, job.Spec.Template.Spec.Containers, "sidecar")
	if len(sidecar.Command) != 1 || sidecar.Command[0] != "sleep" {
		t.Fatalf("expected sidecar command to remain unchanged, got %#v", sidecar.Command)
	}
	if len(sidecar.Args) != 1 || sidecar.Args[0] != "infinity" {
		t.Fatalf("expected sidecar args to remain unchanged, got %#v", sidecar.Args)
	}
	// Sidecar does not get a control volume mount, snapshot env, or ready probe.
	for _, mount := range sidecar.VolumeMounts {
		if mount.Name == SnapshotControlVolumeName {
			t.Fatalf("sidecar should not have control volume mount: %#v", sidecar.VolumeMounts)
		}
	}
	for _, env := range sidecar.Env {
		if env.Name == SnapshotControlDirEnv {
			t.Fatalf("sidecar should not have control env: %#v", sidecar.Env)
		}
	}
	if sidecar.ReadinessProbe != nil {
		t.Fatalf("sidecar should not have a readiness probe forced on it")
	}
}

func TestNewCheckpointJobDisablesServiceMeshInjection(t *testing.T) {
	cases := []struct {
		name        string
		annotations map[string]string
	}{
		{
			name:        "no user annotations",
			annotations: nil,
		},
		{
			name: "pre-existing enabled injection overwritten",
			annotations: map[string]string{
				linkerdInjectAnnotation:      "enabled",
				istioSidecarInjectAnnotation: "true",
				"example.com/keep":           "keep-value",
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var originalAnnotations map[string]string
			if tc.annotations != nil {
				originalAnnotations = make(map[string]string, len(tc.annotations))
				for k, v := range tc.annotations {
					originalAnnotations[k] = v
				}
			}

			source := &corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: tc.annotations,
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:    "main",
						Command: []string{"python3"},
					}},
				},
			}

			job, err := NewCheckpointJob(source, CheckpointJobOptions{
				Namespace:       "test-ns",
				CheckpointID:    "hash",
				Name:            "test-job",
				TargetContainer: "main",
			})
			if err != nil {
				t.Fatalf("NewCheckpointJob() error = %v", err)
			}

			got := job.Spec.Template.Annotations
			if got[linkerdInjectAnnotation] != linkerdInjectDisabled {
				t.Errorf("linkerd annotation = %q, want %q", got[linkerdInjectAnnotation], linkerdInjectDisabled)
			}
			if got[istioSidecarInjectAnnotation] != istioSidecarInjectDisabled {
				t.Errorf("istio annotation = %q, want %q", got[istioSidecarInjectAnnotation], istioSidecarInjectDisabled)
			}
			if tc.annotations != nil {
				if v, ok := tc.annotations["example.com/keep"]; ok && got["example.com/keep"] != v {
					t.Errorf("user annotation lost: got %q, want %q", got["example.com/keep"], v)
				}
				// Source template must not be mutated.
				for k, origV := range originalAnnotations {
					if tc.annotations[k] != origV {
						t.Errorf("source template mutated: key %q changed from %q to %q", k, origV, tc.annotations[k])
					}
				}
			}
		})
	}
}

func TestDisableCheckpointJobSidecarInjectionNilMap(t *testing.T) {
	got := DisableCheckpointJobSidecarInjection(nil)
	if got == nil {
		t.Fatal("expected non-nil map, got nil")
	}
	if got[linkerdInjectAnnotation] != linkerdInjectDisabled {
		t.Errorf("linkerd annotation = %q, want %q", got[linkerdInjectAnnotation], linkerdInjectDisabled)
	}
	if got[istioSidecarInjectAnnotation] != istioSidecarInjectDisabled {
		t.Errorf("istio annotation = %q, want %q", got[istioSidecarInjectAnnotation], istioSidecarInjectDisabled)
	}
}

func TestNewCheckpointJobRequiresTarget(t *testing.T) {
	_, err := NewCheckpointJob(&corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "worker", Command: []string{"python3"}}},
		},
	}, CheckpointJobOptions{
		Namespace:    "test-ns",
		CheckpointID: "hash",
		Name:         "test-job",
	})
	if err == nil || !strings.Contains(err.Error(), "TargetContainer is required") {
		t.Fatalf("expected missing-target error, got %v", err)
	}
}

func TestNewCheckpointJobRejectsUnknownTarget(t *testing.T) {
	_, err := NewCheckpointJob(&corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "worker", Command: []string{"python3"}}},
		},
	}, CheckpointJobOptions{
		Namespace:       "test-ns",
		TargetContainer: "missing",
		CheckpointID:    "hash",
		Name:            "test-job",
	})
	if err == nil || !strings.Contains(err.Error(), `"missing"`) {
		t.Fatalf("expected unknown-target error, got %v", err)
	}
}

// TestNewCheckpointJobNoWrapByDefault verifies that the container command is
// preserved unchanged when WrapLaunchJob is false (the default). This guards
// against accidentally re-introducing cuda-checkpoint wrapping as the default,
// which would require cuda-checkpoint to be present in the placeholder image
// at the exact path CRIU checkpointed it from.
func TestNewCheckpointJobNoWrapByDefault(t *testing.T) {
	originalCmd := []string{"python3", "-m", "dynamo.vllm"}
	originalArgs := []string{"--model", "Qwen"}

	job, err := NewCheckpointJob(&corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:    "main",
				Image:   "test:latest",
				Command: originalCmd,
				Args:    originalArgs,
			}},
		},
	}, CheckpointJobOptions{
		Namespace:       "test-ns",
		TargetContainer: "main",
		CheckpointID:    "hash",
		ArtifactVersion: "2",
		Name:            "test-job",
		WrapLaunchJob:   false, // explicit: this is the default from snapshotctl
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	main := requireCheckpointContainer(t, job.Spec.Template.Spec.Containers, "main")
	if strings.Join(main.Command, " ") != strings.Join(originalCmd, " ") {
		t.Errorf("command must be unchanged without WrapLaunchJob: got %v, want %v", main.Command, originalCmd)
	}
	if strings.Join(main.Args, " ") != strings.Join(originalArgs, " ") {
		t.Errorf("args must be unchanged without WrapLaunchJob: got %v, want %v", main.Args, originalArgs)
	}
	if len(main.Command) > 0 && main.Command[0] == "cuda-checkpoint" {
		t.Errorf("command must not be wrapped with cuda-checkpoint by default")
	}
}

func TestGetCheckpointJobName(t *testing.T) {
	name := GetCheckpointJobName("abc123def4567890", "2")
	if name != "checkpoint-job-abc123def4567890-2" {
		t.Fatalf("unexpected checkpoint job name: %s", name)
	}

	defaultName := GetCheckpointJobName("abc123def4567890", "")
	if defaultName != "checkpoint-job-abc123def4567890-"+DefaultCheckpointArtifactVersion {
		t.Fatalf("unexpected default checkpoint job name: %s", defaultName)
	}
}
