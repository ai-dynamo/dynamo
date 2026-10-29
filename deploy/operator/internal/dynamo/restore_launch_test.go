/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"context"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	k8sresource "k8s.io/apimachinery/pkg/api/resource"
)

func TestRenderMainContainerLaunches(t *testing.T) {
	gpuLimits := corev1.ResourceList{corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): k8sresource.MustParse("2")}
	worker := func(command, args []string, nodes int32) *v1beta1.DynamoComponentDeploymentSharedSpec {
		component := &v1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: "worker",
			ComponentType: v1beta1.ComponentTypeWorker,
			PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
				Name:      commonconsts.MainContainerName,
				Command:   command,
				Args:      args,
				Resources: corev1.ResourceRequirements{Limits: gpuLimits},
			}}}},
		}
		if nodes > 1 {
			component.Multinode = &v1beta1.MultinodeSpec{NodeCount: nodes}
		}
		return component
	}
	vllm := []string{"python3", "-m", "dynamo.vllm"}
	elasticEPRay := []string{"--model", "m", "--enable-elastic-ep", "--data-parallel-backend", "ray"}

	tests := []struct {
		name      string
		component *v1beta1.DynamoComponentDeploymentSharedSpec
		backend   string
		wantRoles []Role
		// wantDirect lists, per rendered role, whether the launch is still a direct python command.
		wantDirect []bool
		wantErr    bool
	}{
		{
			name:       "single-node vLLM keeps its direct python launch",
			component:  worker(vllm, []string{"--model", "m"}, 1),
			backend:    "vllm",
			wantRoles:  []Role{RoleMain},
			wantDirect: []bool{true},
		},
		{
			name:       "elastic expert parallelism on Ray rewrites the launch into a shell",
			component:  worker(vllm, elasticEPRay, 1),
			backend:    "vllm",
			wantRoles:  []Role{RoleMain},
			wantDirect: []bool{false},
		},
		{
			name:       "elastic expert parallelism without the Ray backend keeps the launch",
			component:  worker(vllm, []string{"--model", "m", "--enable-elastic-ep"}, 1),
			backend:    "vllm",
			wantRoles:  []Role{RoleMain},
			wantDirect: []bool{true},
		},
		{
			name:       "single-node SGLang keeps its direct python launch",
			component:  worker([]string{"python3", "-m", "dynamo.sglang"}, []string{"--model-path", "m"}, 1),
			backend:    "sglang",
			wantRoles:  []Role{RoleMain},
			wantDirect: []bool{true},
		},
		{
			name:       "multinode TRT-LLM renders a shell for the leader and the worker",
			component:  worker([]string{"python3", "-m", "dynamo.trtllm"}, nil, 2),
			backend:    "trtllm",
			wantRoles:  []Role{RoleLeader, RoleWorker},
			wantDirect: []bool{false, false},
		},
		{
			name: "multinode inter-pod GMS renders rank-aware roles that are not modeled",
			component: func() *v1beta1.DynamoComponentDeploymentSharedSpec {
				component := worker(vllm, []string{"--model", "m"}, 2)
				component.Experimental = &v1beta1.ExperimentalSpec{
					GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{Mode: v1beta1.GMSModeInterPod},
				}
				return component
			}(),
			backend: "vllm",
			wantErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Given an authored worker component and the backend the graph selects")
			original := test.component.DeepCopy()

			t.Log("When the main container launch is rendered for every role that serves it")
			launches, err := RenderMainContainerLaunches(
				context.Background(), test.component, test.backend, nil, commonconsts.MultinodeDeploymentTypeGrove,
			)

			t.Log("Then the authored component is untouched and each role carries its rewritten launch")
			assert.Equal(t, original, test.component)
			if test.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Len(t, launches, len(test.wantRoles))
			for i, launch := range launches {
				assert.Equal(t, test.wantRoles[i], launch.Role)
				direct := len(launch.Command) > 0 && launch.Command[0] == "python3"
				assert.Equal(t, test.wantDirect[i], direct, "command=%q args=%q", launch.Command, launch.Args)
			}
		})
	}
}
