// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

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
	if len(job.Spec.Template.Spec.Volumes) != 0 {
		t.Fatalf("expected no checkpoint volume, got %#v", job.Spec.Template.Spec.Volumes)
	}
	if len(job.Spec.Template.Spec.Containers[0].VolumeMounts) != 0 {
		t.Fatalf("expected no checkpoint volume mount, got %#v", job.Spec.Template.Spec.Containers[0].VolumeMounts)
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
