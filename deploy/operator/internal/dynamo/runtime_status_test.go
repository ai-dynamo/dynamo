/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
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

func TestResolveComponentRuntimeStatusFromRenderedTRTLLMLeader(t *testing.T) {
	t.Log("Render a TRT-LLM leader and resolve identity through its nested bash launch wrapper")
	container := &corev1.Container{
		Name:    commonconsts.MainContainerName,
		Command: []string{"python3"},
		Args: []string{
			"--model", "Qwen/Qwen3-8B",
			"--endpoint", "dyn://prod.custom-decode.generate",
		},
	}
	backend := &TRTLLMBackend{MpiRunSecretName: "mpi-run"}
	require.NoError(t, backend.UpdateContainer(
		container,
		2,
		RoleLeader,
		&v1beta1.DynamoComponentDeploymentSharedSpec{},
		"decode",
		&GroveMultinodeDeployer{},
		staticContainerGPUCount(1),
	))

	status := ResolveComponentRuntimeStatus(&corev1.PodSpec{Containers: []corev1.Container{*container}}, nil)
	assert.Equal(t, "Qwen/Qwen3-8B", status.ServedModelName)
	assert.Equal(t, "custom-decode", status.RuntimeComponentName)
}

func TestResolveGroveComponentRuntimeStatusesUsesServingRole(t *testing.T) {
	t.Log("Select the semantic leader instead of a generated worker clique")
	dgd := &v1beta1.DynamoGraphDeployment{
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "decode",
				ComponentType: v1beta1.ComponentTypeDecode,
				Replicas:      new(int32(1)),
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

// TestNativeSidecarStatusDoesNotPredictIdentity covers shell expansion and direct
// launches after the operator has injected its runtime environment.
func TestNativeSidecarStatusDoesNotPredictIdentity(t *testing.T) {
	cases := []struct {
		name          string
		command, args []string
	}{
		{name: "shell dollar", command: []string{"/bin/sh", "-c"}, args: []string{`exec dynamo-vllm-sidecar --component "$WORKER_NAME"`}},
		{name: "shell braces", command: []string{"/bin/sh", "-c"}, args: []string{`exec dynamo-vllm-sidecar --component "${WORKER_NAME}"`}},
		{name: "direct custom aggregated", command: []string{"dynamo-vllm-sidecar"}, args: []string{"--component", "custom-worker"}},
		{name: "direct decode", command: []string{"dynamo-vllm-sidecar"}, args: []string{"--disaggregation-mode", "decode", "--component", "ignored"}},
		{name: "image entrypoint"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Render a native sidecar with independent engine flags and injected identity")
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers:     []corev1.Container{{Name: "main", Image: "engine:latest", Command: []string{"vllm-rs"}, Args: []string{"serve", "model", "--served-model-name", "engine-model", "--endpoint", "ns.engine.generate"}}},
					InitContainers: []corev1.Container{{Name: "runtime", Image: "runtime:1.6.0", RestartPolicy: new(corev1.ContainerRestartPolicyAlways), Command: tc.command, Args: tc.args, Env: []corev1.EnvVar{{Name: "WORKER_NAME", Value: "backend"}}}},
				}},
			}
			original := component.DeepCopy()
			pod, err := GeneratePodSpecForComponent(component, BackendFrameworkVLLM, nil, &v1beta1.DynamoGraphDeployment{}, RoleMain, 1, &configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
			require.NoError(t, err)
			require.Equal(t, "worker", envVarsToMap(pod.InitContainers[0].Env)[commonconsts.DynamoComponentEnvVar])

			t.Log("Leave identity to runtime registration while retaining unrelated power status")
			status := ResolveComponentRuntimeStatus(pod, map[string]string{commonconsts.KubeAnnotationGPUPowerLimit: "300"})
			assert.Empty(t, status.RuntimeComponentName)
			assert.Empty(t, status.ServedModelName)
			require.NotNil(t, status.GPUPowerLimitWatts)
			assert.Equal(t, int64(300), *status.GPUPowerLimitWatts)
			assert.Equal(t, original, component)
		})
	}
}
