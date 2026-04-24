// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
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

func TestNewCheckpointJob(t *testing.T) {
	job, err := NewCheckpointJob(&corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
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
	}, CheckpointJobOptions{
		Namespace:             "test-ns",
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
	if len(job.Spec.Template.Spec.Volumes) != 1 || job.Spec.Template.Spec.Volumes[0].Name != SnapshotControlVolumeName {
		t.Fatalf("expected only %s volume, got %#v", SnapshotControlVolumeName, job.Spec.Template.Spec.Volumes)
	}
	main := &job.Spec.Template.Spec.Containers[0]
	if len(main.VolumeMounts) != 1 || main.VolumeMounts[0].MountPath != SnapshotControlMountPath {
		t.Fatalf("expected only %s mount at %s, got %#v", SnapshotControlVolumeName, SnapshotControlMountPath, main.VolumeMounts)
	}
	if main.ReadinessProbe == nil || main.ReadinessProbe.Exec == nil {
		t.Fatalf("expected ready-file readiness probe, got %#v", main.ReadinessProbe)
	}
	expectedProbe := []string{"cat", SnapshotControlMountPath + "/" + ReadyForCheckpointFile}
	if len(main.ReadinessProbe.Exec.Command) != len(expectedProbe) {
		t.Fatalf("expected readiness probe %#v, got %#v", expectedProbe, main.ReadinessProbe.Exec.Command)
	}
	for i := range expectedProbe {
		if main.ReadinessProbe.Exec.Command[i] != expectedProbe[i] {
			t.Fatalf("expected readiness probe %#v, got %#v", expectedProbe, main.ReadinessProbe.Exec.Command)
		}
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
	if len(job.Spec.Template.Spec.Containers[0].Command) != 1 || job.Spec.Template.Spec.Containers[0].Command[0] != "cuda-checkpoint" {
		t.Fatalf("expected cuda-checkpoint wrapper command: %#v", job.Spec.Template.Spec.Containers[0].Command)
	}
	expectedArgs := []string{"--launch-job", "python3", "-m", "dynamo.vllm", "--model", "Qwen"}
	if len(job.Spec.Template.Spec.Containers[0].Args) != len(expectedArgs) {
		t.Fatalf("expected launch-job args %#v, got %#v", expectedArgs, job.Spec.Template.Spec.Containers[0].Args)
	}
	for i := range expectedArgs {
		if job.Spec.Template.Spec.Containers[0].Args[i] != expectedArgs[i] {
			t.Fatalf("expected launch-job args %#v, got %#v", expectedArgs, job.Spec.Template.Spec.Containers[0].Args)
		}
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

func TestNewCheckpointJobWrapsFirstContainer(t *testing.T) {
	job, err := NewCheckpointJob(&corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "worker", Command: []string{"python3", "-m", "dynamo.vllm"}, Args: []string{"--model", "Qwen"}},
				{Name: "sidecar", Command: []string{"sleep"}, Args: []string{"infinity"}},
			},
		},
	}, CheckpointJobOptions{
		Namespace:             "test-ns",
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
	if len(worker.Command) != 1 || worker.Command[0] != "cuda-checkpoint" {
		t.Fatalf("expected first container to be wrapped, got %#v", worker.Command)
	}
	expectedArgs := []string{"--launch-job", "python3", "-m", "dynamo.vllm", "--model", "Qwen"}
	if len(worker.Args) != len(expectedArgs) {
		t.Fatalf("expected launch-job args %#v, got %#v", expectedArgs, worker.Args)
	}
	for i := range expectedArgs {
		if worker.Args[i] != expectedArgs[i] {
			t.Fatalf("expected launch-job args %#v, got %#v", expectedArgs, worker.Args)
		}
	}

	sidecar := requireCheckpointContainer(t, job.Spec.Template.Spec.Containers, "sidecar")
	if len(sidecar.Command) != 1 || sidecar.Command[0] != "sleep" {
		t.Fatalf("expected sidecar command to remain unchanged, got %#v", sidecar.Command)
	}
	if len(sidecar.Args) != 1 || sidecar.Args[0] != "infinity" {
		t.Fatalf("expected sidecar args to remain unchanged, got %#v", sidecar.Args)
	}
}

func TestNewCheckpointJobAllowsSingleNonMainContainer(t *testing.T) {
	job, err := NewCheckpointJob(&corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:    "worker",
				Command: []string{"python3", "-m", "dynamo.vllm"},
				Args:    []string{"--model", "Qwen"},
			}},
		},
	}, CheckpointJobOptions{
		Namespace:             "test-ns",
		CheckpointID:          "hash",
		ArtifactVersion:       "2",
		Name:                  "test-job",
		TTLSecondsAfterFinish: ptr.To(int32(300)),
		WrapLaunchJob:         true,
	})
	if err != nil {
		t.Fatalf("expected checkpoint job, got error: %v", err)
	}

	container := &job.Spec.Template.Spec.Containers[0]
	if len(container.Command) != 1 || container.Command[0] != "cuda-checkpoint" {
		t.Fatalf("expected single container to be wrapped, got %#v", container.Command)
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
