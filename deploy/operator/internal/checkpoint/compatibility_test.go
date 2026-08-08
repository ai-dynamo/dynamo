/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

func supportedFailoverComponent(numShadows int32) *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	return &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name: nvidiacomv1beta1.MainContainerName,
				Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
					corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
				}},
			}},
		}},
		Experimental: &nvidiacomv1beta1.ExperimentalSpec{
			Checkpoint:       &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
			GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{Mode: nvidiacomv1beta1.GMSModeIntraPod},
			Failover: &nvidiacomv1beta1.FailoverSpec{
				Mode:       nvidiacomv1beta1.GMSModeIntraPod,
				NumShadows: numShadows,
			},
		},
	}
}

func TestValidateCheckpointCompatibility(t *testing.T) {
	tests := []struct {
		name    string
		base    *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
		mutate  func(*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec)
		backend string
		context CompatibilityContext
		want    []string
	}{
		{
			name: "checkpoint without failover",
			base: &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				Multinode: &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
				Experimental: &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				},
			},
		},
		{
			name:    "DGD source",
			backend: "vllm",
			context: CompatibilityContextDGDSource,
		},
		{
			name:    "DGD source with two shadows",
			base:    supportedFailoverComponent(2),
			backend: "vllm",
			context: CompatibilityContextDGDSource,
		},
		{
			name:    "generated DCD",
			backend: "vllm",
			context: CompatibilityContextGeneratedDCD,
			mutate: func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.Experimental.Checkpoint.CheckpointRef = ptr.To("checkpoint")
			},
		},
		{
			name:    "generated DCD without reference",
			backend: "vllm",
			context: CompatibilityContextGeneratedDCD,
			want: []string{
				checkpointFailoverCompatibilityMessage + ": checkpointRef must name the DGD-managed automatic checkpoint",
			},
		},
		{
			name:    "standalone DCD with forged reference",
			backend: "vllm",
			context: CompatibilityContextStandaloneDCD,
			mutate: func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.Experimental.Checkpoint.CheckpointRef = ptr.To("checkpoint")
			},
			want: []string{
				checkpointFailoverCompatibilityMessage + ": checkpoint failover is only supported for an operator-generated DCD",
			},
		},
		{
			name:    "inter-pod profile",
			backend: "vllm",
			context: CompatibilityContextDGDSource,
			mutate: func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.Experimental.GPUMemoryService.Mode = nvidiacomv1beta1.GMSModeInterPod
				c.Experimental.Failover.Mode = nvidiacomv1beta1.GMSModeInterPod
			},
			want: []string{
				checkpointInterPodCompatibilityMessage,
				checkpointFailoverCompatibilityMessage + ": gpuMemoryService.mode must be IntraPod\nfailover.mode must be IntraPod",
			},
		},
		{
			name:    "multinode profile",
			backend: "vllm",
			context: CompatibilityContextDGDSource,
			mutate: func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			},
			want: []string{
				checkpointFailoverCompatibilityMessage + ": multinode/model-parallel worker topology is unsupported",
			},
		},
		{
			name:    "non vllm",
			backend: "sglang",
			context: CompatibilityContextDGDSource,
			want: []string{
				checkpointFailoverCompatibilityMessage + ": backendFramework must be vllm",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			component := tt.base
			if component == nil {
				component = supportedFailoverComponent(1)
			} else {
				component = component.DeepCopy()
			}
			if tt.mutate != nil {
				tt.mutate(component)
			}
			violations := ValidateCheckpointCompatibility(component, tt.backend, tt.context)
			got := make([]string, len(violations))
			for i := range violations {
				got[i] = violations[i].Error()
			}
			if len(got) == 0 {
				got = nil
			}
			assert.Equal(t, tt.want, got)
		})
	}
}
