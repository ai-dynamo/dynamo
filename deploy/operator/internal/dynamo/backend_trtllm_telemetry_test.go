// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestTRTLLMTelemetryEnvironment(t *testing.T) {
	t.Log("Configure file telemetry for four co-located ranks without a shared exporter port")
	directory := t.TempDir()
	fileSink := []corev1.EnvVar{
		{Name: "NIXL_TELEMETRY_ENABLE", Value: "y"},
		{Name: "NIXL_TELEMETRY_EXPORTER", Value: ""},
		{Name: "NIXL_TELEMETRY_DIR", Value: directory},
	}
	cases := []struct {
		name         string
		graphEnv     []corev1.EnvVar
		workerEnv    []corev1.EnvVar
		wantEnable   string
		wantExporter string
		wantFiles    bool
	}{
		{name: "disabled by default", wantEnable: "n", wantExporter: "prometheus"},
		{name: "graph file sink", graphEnv: fileSink, wantEnable: "y", wantFiles: true},
		{name: "worker file sink", workerEnv: fileSink, wantEnable: "y", wantFiles: true},
		{
			name: "empty worker exporter overrides graph prometheus",
			graphEnv: []corev1.EnvVar{
				{Name: "NIXL_TELEMETRY_ENABLE", Value: "y"},
				{Name: "NIXL_TELEMETRY_EXPORTER", Value: "prometheus"},
			},
			workerEnv: fileSink, wantEnable: "y", wantFiles: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Build the graph and a four-GPU TRT-LLM worker with explicit telemetry settings")
			dgd := &v1beta1.DynamoGraphDeployment{
				Name: "telemetry", Namespace: "default",
				Spec: v1beta1.DynamoGraphDeploymentSpec{Env: tc.graphEnv},
			}
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "prefill",
				ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name: commonconsts.MainContainerName, Image: "trtllm:1.6.0",
						Command: []string{"python3", "-m", "dynamo.trtllm"},
						Args:    []string{"--disaggregation-mode", "prefill", "--tensor-parallel-size", "4"},
						Env:     tc.workerEnv,
						Resources: corev1.ResourceRequirements{
							Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("4")},
						},
					}},
				}},
			}
			originalGraph, originalComponent := dgd.DeepCopy(), component.DeepCopy()

			t.Log("Render through the production graph-to-pod path, including worker defaults")
			pod, err := GeneratePodSpecForComponent(
				component, BackendFrameworkTRTLLM, &mockSecretsRetriever{}, dgd,
				RoleMain, 1, &configv1alpha1.OperatorConfiguration{},
				commonconsts.MultinodeDeploymentTypeGrove, "prefill", nil, staticContainerGPUCount(4),
			)
			require.NoError(t, err)
			require.Equal(t, originalGraph, dgd)
			require.Equal(t, originalComponent, component)

			t.Log("Preserve explicit empty exporter values and the existing sink directory")
			main := GetMainContainerFromPodSpec(pod)
			require.NotNil(t, main)
			env := envVarsToMap(main.Env)
			require.Len(t, main.Env, len(env), "environment overrides must not leave duplicate entries")
			require.Contains(t, env, "NIXL_TELEMETRY_ENABLE")
			require.Equal(t, tc.wantEnable, env["NIXL_TELEMETRY_ENABLE"])
			require.Contains(t, env, "NIXL_TELEMETRY_EXPORTER", "empty must not become absent")
			require.Equal(t, tc.wantExporter, env["NIXL_TELEMETRY_EXPORTER"])
			if tc.wantFiles {
				require.Equal(t, directory, env["NIXL_TELEMETRY_DIR"])
			} else {
				require.NotContains(t, env, "NIXL_TELEMETRY_DIR")
			}
		})
	}
}
