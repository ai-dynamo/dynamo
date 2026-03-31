// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package checkpointjob

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestBuildCheckpointJobPrefersMainContainer(t *testing.T) {
	ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			Identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "Qwen/Qwen3-0.6B",
				BackendFramework: "vllm",
			},
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{
								Name:    "sidecar",
								Image:   "sidecar:latest",
								Command: []string{"sleep", "3600"},
							},
							{
								Name:    consts.MainContainerName,
								Image:   "worker:latest",
								Command: []string{"python3", "-m", "dynamo.vllm"},
								Resources: corev1.ResourceRequirements{
									Limits: corev1.ResourceList{
										corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("2"),
									},
								},
							},
						},
					},
				},
			},
		},
	}

	job, err := BuildCheckpointJob(&configv1alpha1.OperatorConfiguration{
		Checkpoint: configv1alpha1.CheckpointConfiguration{
			ReadyForCheckpointFilePath: "/tmp/ready-for-checkpoint",
		},
	}, ckpt, "checkpoint-job")
	if err != nil {
		t.Fatalf("expected checkpoint job, got error: %v", err)
	}

	sidecar := job.Spec.Template.Spec.Containers[0]
	if sidecar.Command[0] != "sleep" {
		t.Fatalf("expected sidecar command to be preserved, got %#v", sidecar.Command)
	}

	main := job.Spec.Template.Spec.Containers[1]
	if len(main.Command) != 1 || main.Command[0] != "cuda-checkpoint" {
		t.Fatalf("expected main container to be wrapped, got %#v", main.Command)
	}
	if main.ReadinessProbe == nil || main.ReadinessProbe.Exec == nil {
		t.Fatalf("expected readiness probe on main container, got %#v", main.ReadinessProbe)
	}
	if main.ReadinessProbe.Exec.Command[0] != "cat" {
		t.Fatalf("expected readiness probe on main container, got %#v", main.ReadinessProbe.Exec.Command)
	}
	if sidecar.ReadinessProbe != nil {
		t.Fatalf("expected sidecar readiness probe to remain unset, got %#v", sidecar.ReadinessProbe)
	}
}
