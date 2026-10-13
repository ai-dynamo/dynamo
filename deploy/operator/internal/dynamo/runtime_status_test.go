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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

// TestResolveComponentRuntimeStatusUsesNativeSidecar preserves engine model flags
// while projecting the component from flags accepted by the native vLLM runtime.
func TestResolveComponentRuntimeStatusUsesNativeSidecar(t *testing.T) {
	cases := []struct {
		name          string
		engineArgs    []string
		runtimeArgs   []string
		runtimeEnv    []corev1.EnvVar
		wantModel     string
		wantComponent string
	}{
		{
			name:        "component flag overrides environment with engine served model",
			engineArgs:  []string{"serve", "model-source", "--served-model-name", "served-model"},
			runtimeArgs: []string{"--grpc-endpoint", "127.0.0.1:50051", "--component", "custom-decode", "--endpoint", "generate"},
			runtimeEnv:  []corev1.EnvVar{{Name: commonconsts.DynamoComponentEnvVar, Value: "decode"}},
			wantModel:   "served-model", wantComponent: "custom-decode",
		},
		{
			name:        "bare endpoint uses aggregated runtime environment and preserves engine model flag",
			engineArgs:  []string{"--model", "engine-model", "--endpoint", "dyn://prod.engine-component.generate"},
			runtimeArgs: []string{"--grpc-endpoint", "127.0.0.1:50051", "--endpoint", "generate"},
			runtimeEnv:  []corev1.EnvVar{{Name: commonconsts.DynamoComponentEnvVar, Value: "decode"}},
			wantModel:   "engine-model", wantComponent: "decode",
		},
		{
			name:          "positional engine model needs registered metadata",
			engineArgs:    []string{"serve", "model-source"},
			runtimeArgs:   []string{"--grpc-endpoint", "127.0.0.1:50051", "--component=custom-decode"},
			wantComponent: "custom-decode",
		},
		{
			name:        "engine endpoint does not supply runtime identity",
			engineArgs:  []string{"--model", "engine-model", "--endpoint", "dyn://prod.engine-component.generate"},
			runtimeArgs: []string{"--grpc-endpoint", "127.0.0.1:50051"},
			wantModel:   "engine-model",
		},
		{
			name:        "unresolved environment is not a component name",
			runtimeArgs: []string{"--grpc-endpoint", "127.0.0.1:50051"},
			runtimeEnv:  []corev1.EnvVar{{Name: commonconsts.DynamoComponentEnvVar, Value: "$(COMPONENT)"}},
		},
		{
			name:        "unresolved component flag is not a component name",
			runtimeArgs: []string{"--grpc-endpoint", "127.0.0.1:50051", "--component=$(COMPONENT)"},
		},
		{
			name:        "last environment entry determines component",
			runtimeArgs: []string{"--grpc-endpoint", "127.0.0.1:50051"},
			runtimeEnv: []corev1.EnvVar{
				{Name: commonconsts.DynamoComponentEnvVar, Value: "old-component"},
				{Name: commonconsts.DynamoComponentEnvVar, Value: "custom-worker"},
			},
			wantComponent: "custom-worker",
		},
		{
			name:        "valueFrom does not retain an earlier literal component",
			runtimeArgs: []string{"--grpc-endpoint", "127.0.0.1:50051"},
			runtimeEnv: []corev1.EnvVar{
				{Name: commonconsts.DynamoComponentEnvVar, Value: "old-component"},
				{Name: commonconsts.DynamoComponentEnvVar, ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Configure an independent engine and a native vLLM runtime using supported sidecar flags")
			pod := &corev1.PodSpec{
				Containers: []corev1.Container{{Name: commonconsts.MainContainerName, Args: tc.engineArgs}},
				InitContainers: []corev1.Container{
					{Name: "setup", Args: []string{"--model", "setup-model"}},
					{Name: "runtime", Command: []string{"dynamo-vllm-sidecar"}, RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways), Args: tc.runtimeArgs, Env: tc.runtimeEnv},
				},
			}
			original := pod.DeepCopy()

			t.Log("Resolve model fallback and runtime component identity without mutating the Pod")
			status := ResolveComponentRuntimeStatus(pod, nil)
			assert.Equal(t, tc.wantModel, status.ServedModelName)
			assert.Equal(t, tc.wantComponent, status.RuntimeComponentName)
			require.Equal(t, original, pod)
		})
	}
}

// TestResolveComponentRuntimeStatusFromRenderedNativeSidecar preserves aggregated
// registration names while disaggregated workers use Planner's discovery defaults.
func TestResolveComponentRuntimeStatusFromRenderedNativeSidecar(t *testing.T) {
	cases := []struct {
		name          string
		componentType v1beta1.ComponentType
		args          []string
		env           []corev1.EnvVar
		envFrom       []corev1.EnvFromSource
		wantComponent string
	}{
		{name: "default aggregated registers injected worker", componentType: v1beta1.ComponentTypeWorker, wantComponent: "worker"},
		{name: "disaggregated decode ignores injected name", componentType: v1beta1.ComponentTypeDecode, args: []string{"--disaggregation-mode", "decode"}},
		{name: "decode type without mode is aggregated", componentType: v1beta1.ComponentTypeDecode, wantComponent: "decode"},
		{name: "decode ignores custom component", componentType: v1beta1.ComponentTypeDecode, args: []string{"--disaggregation-mode=decode", "--component=custom-decode"}},
		{name: "prefill ignores custom component", componentType: v1beta1.ComponentTypePrefill, args: []string{"--disaggregation-mode", "prefill", "--component", "custom-prefill"}},
		{name: "encode ignores custom component", componentType: v1beta1.ComponentTypeWorker, args: []string{"--disaggregation-mode=encode", "--component=custom-encode"}},
		{name: "default aggregated honors component", componentType: v1beta1.ComponentTypeWorker, args: []string{"--component", "custom-worker"}, wantComponent: "custom-worker"},
		{name: "agg honors component", componentType: v1beta1.ComponentTypeWorker, args: []string{"--disaggregation-mode", "agg", "--component", "custom-worker"}, wantComponent: "custom-worker"},
		{name: "aggregated alias honors component", componentType: v1beta1.ComponentTypeWorker, args: []string{"--disaggregation-mode=aggregated", "--component=custom-worker"}, wantComponent: "custom-worker"},
		{name: "bare dotted endpoint is not a component override", componentType: v1beta1.ComponentTypeWorker, args: []string{"--endpoint=ns.custom.generate"}, wantComponent: "worker"},
		{name: "unknown mode does not project component", componentType: v1beta1.ComponentTypeWorker, args: []string{"--disaggregation-mode=$(MODE)", "--component=custom-worker"}},
		{name: "environment mode is not assumed aggregated", componentType: v1beta1.ComponentTypeDecode, args: []string{"--component=custom-decode"}, env: []corev1.EnvVar{{Name: "DYN_DISAGGREGATION_MODE", Value: "decode"}}},
		{name: "environment aggregated honors injected component", componentType: v1beta1.ComponentTypeWorker, env: []corev1.EnvVar{{Name: "DYN_DISAGGREGATION_MODE", Value: "agg"}}, wantComponent: "worker"},
		{name: "unresolved environment mode is not assumed aggregated", componentType: v1beta1.ComponentTypeWorker, args: []string{"--component=custom-worker"}, env: []corev1.EnvVar{{Name: "DYN_DISAGGREGATION_MODE", ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "runtime-env"}, Key: "mode"}}}}},
		{name: "external environment may supply mode", componentType: v1beta1.ComponentTypeWorker, args: []string{"--component=custom-worker"}, envFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "runtime-env"}}}}},
		{name: "explicit aggregated mode overrides environment", componentType: v1beta1.ComponentTypeWorker, args: []string{"--disaggregation-mode=agg", "--component=custom-worker"}, env: []corev1.EnvVar{{Name: "DYN_DISAGGREGATION_MODE", Value: "decode"}}, wantComponent: "custom-worker"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Render an engine and native runtime through the controller path")
			dcd := &v1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "test"},
				Spec: v1beta1.DynamoComponentDeploymentSpec{DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "test", ComponentType: tc.componentType,
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: commonconsts.MainContainerName, Image: "engine:latest", Command: []string{"vllm-rs"}, Args: []string{"serve", "model", "--served-model-name", "served-model"}}},
						InitContainers: []corev1.Container{{
							Name: commonconsts.RuntimeContainerName, Image: "runtime:1.6.0", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
							Command: []string{"dynamo-vllm-sidecar"}, Args: append([]string{"--grpc-endpoint", "127.0.0.1:50051"}, tc.args...), Env: tc.env, EnvFrom: tc.envFrom,
						}},
					}},
				}},
			}
			pod, err := GenerateBasePodSpecForController(dcd, nil, &configv1alpha1.OperatorConfiguration{}, RoleMain, commonconsts.MultinodeDeploymentTypeGrove, staticContainerGPUCount(0), GenerateBasePodSpecForControllerOptions{})
			require.NoError(t, err)
			runtime := GetDynamoContainerFromPodSpec(pod)
			require.NotNil(t, runtime)
			require.Equal(t, string(tc.componentType), envVarsToMap(runtime.Env)[commonconsts.DynamoComponentEnvVar])
			original := pod.DeepCopy()

			t.Log("Project only supported component overrides while preserving the engine model")
			status := ResolveComponentRuntimeStatus(pod, nil)
			assert.Equal(t, tc.wantComponent, status.RuntimeComponentName)
			assert.Equal(t, "served-model", status.ServedModelName)
			require.Equal(t, original, pod)
		})
	}
}

// TestResolveComponentRuntimeStatusPrefersRuntimeModel keeps explicit runtime
// model identity ahead of the engine's model source for Python sidecar launches.
func TestResolveComponentRuntimeStatusPrefersRuntimeModel(t *testing.T) {
	t.Log("Configure different engine and runtime model names")
	pod := &corev1.PodSpec{
		Containers:     []corev1.Container{{Name: commonconsts.MainContainerName, Args: []string{"--model", "engine-model"}}},
		InitContainers: []corev1.Container{{Name: "runtime", Command: []string{"python3", "-m", "dynamo.vllm"}, Args: []string{"--served-model-name", "served-model", "--endpoint", "dyn://prod.decode.generate"}}},
	}

	t.Log("Prefer the explicit runtime identity to the engine fallback")
	status := ResolveComponentRuntimeStatus(pod, nil)
	assert.Equal(t, "served-model", status.ServedModelName)
	assert.Equal(t, "decode", status.RuntimeComponentName)
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

// TestDetectBackendFrameworkDoesNotMutateCommand protects callers that keep
// additional command data in the same backing array.
func TestDetectBackendFrameworkDoesNotMutateCommand(t *testing.T) {
	t.Log("Keep a sentinel beyond the command slice's length")
	backing := []string{"dynamo-vllm-sidecar", "unchanged"}
	command := backing[:1]

	t.Log("Detect the native launch without overwriting the caller's backing array")
	backend, err := DetectBackendFrameworkFromArgs(command, []string{"--grpc-endpoint=127.0.0.1:50051"})
	require.NoError(t, err)
	assert.Equal(t, BackendFrameworkVLLM, backend)
	assert.Equal(t, []string{"dynamo-vllm-sidecar", "unchanged"}, backing)
}
