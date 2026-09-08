/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

func TestResolveComponentRuntimeStatus(t *testing.T) {
	t.Log("Resolve runtime identity from a shell-packed rendered command")
	status := ResolveComponentRuntimeStatus(
		&corev1.PodSpec{Containers: []corev1.Container{{
			Name:    commonconsts.MainContainerName,
			Command: []string{"/bin/sh", "-c"},
			Args: []string{
				`exec python -m dynamo.vllm --model ignored --served-model-name "Qwen/Qwen3-8B" --endpoint=dyn://prod.custom-decode.generate`,
			},
		}}},
		map[string]string{commonconsts.KubeAnnotationGPUPowerLimit: " 300 "},
	)

	assert.Equal(t, "Qwen/Qwen3-8B", status.ServedModelName)
	assert.Equal(t, "custom-decode", status.RuntimeComponentName)
	require.NotNil(t, status.GPUPowerLimitWatts)
	assert.Equal(t, int64(300), *status.GPUPowerLimitWatts)
}

func TestResolveComponentRuntimeStatusUsesModelPathFallback(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{
			name: "separate argument",
			args: []string{"--model-path", "/models/qwen"},
			want: "/models/qwen",
		},
		{
			name: "equals argument",
			args: []string{"--model-path=/models/qwen"},
			want: "/models/qwen",
		},
		{
			name: "shell-packed argument",
			args: []string{`exec python -m dynamo.sglang --model-path "/models/qwen 8b"`},
			want: "/models/qwen 8b",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			status := ResolveComponentRuntimeStatus(
				&corev1.PodSpec{Containers: []corev1.Container{{
					Name: commonconsts.MainContainerName,
					Args: test.args,
				}}},
				nil,
			)

			assert.Equal(t, test.want, status.ServedModelName)
		})
	}
}

func TestResolveGroveComponentRuntimeStatusesUsesServingRole(t *testing.T) {
	t.Log("Select the semantic leader instead of a generated worker clique")
	dgd := &v1beta1.DynamoGraphDeployment{
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "decode",
				ComponentType: v1beta1.ComponentTypeDecode,
				Replicas:      ptr.To(int32(1)),
				Multinode:     &v1beta1.MultinodeSpec{NodeCount: 2},
			}},
		},
	}
	pcs := &grovev1alpha1.PodCliqueSet{
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{
						Name: "decode-wkr",
						Spec: grovev1alpha1.PodCliqueSpec{PodSpec: runtimeStatusTestPodSpec("worker-model")},
					},
					{
						Name: "decode-ldr",
						Spec: grovev1alpha1.PodCliqueSpec{PodSpec: runtimeStatusTestPodSpec("leader-model")},
					},
				},
			},
		},
	}

	statuses := ResolveGroveComponentRuntimeStatuses(dgd, pcs)
	assert.Equal(t, "leader-model", statuses["decode"].ServedModelName)
}

func runtimeStatusTestPodSpec(model string) corev1.PodSpec {
	return corev1.PodSpec{Containers: []corev1.Container{{
		Name: commonconsts.MainContainerName,
		Args: []string{"--model", model},
	}}}
}
