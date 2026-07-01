/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
)

func TestWorkerHealthCheckDefaultsByBackendMode(t *testing.T) {
	tests := []struct {
		name      string
		backend   Backend
		component *v1beta1.DynamoComponentDeploymentSharedSpec
		container corev1.Container
		want      string
	}{
		{
			name:      "vLLM worker enables canary",
			backend:   &VLLMBackend{},
			component: workerHealthComponent(v1beta1.ComponentTypeWorker),
			container: workerHealthBaseContainer(),
			want:      envValueTrue,
		},
		{
			name:      "vLLM native decode worker keeps canary disabled",
			backend:   &VLLMBackend{},
			component: workerHealthComponent(v1beta1.ComponentTypeDecode),
			container: workerHealthBaseContainer(),
			want:      envValueFalse,
		},
		{
			name:      "vLLM legacy decode flag keeps canary disabled",
			backend:   &VLLMBackend{},
			component: workerHealthComponent(v1beta1.ComponentTypeWorker),
			container: workerHealthContainerWithArgs("--is-decode-worker"),
			want:      envValueFalse,
		},
		{
			name:      "vLLM disaggregation decode mode keeps canary disabled",
			backend:   &VLLMBackend{},
			component: workerHealthComponent(v1beta1.ComponentTypeWorker),
			container: workerHealthContainerWithArgs("--disaggregation-mode=decode"),
			want:      envValueFalse,
		},
		{
			name:      "TRT-LLM worker enables canary",
			backend:   &TRTLLMBackend{},
			component: workerHealthComponent(v1beta1.ComponentTypeWorker),
			container: workerHealthBaseContainer(),
			want:      envValueTrue,
		},
		{
			name:      "TRT-LLM decode mode keeps canary disabled",
			backend:   &TRTLLMBackend{},
			component: workerHealthComponent(v1beta1.ComponentTypeWorker),
			container: workerHealthContainerWithArgs("--disaggregation-mode decode"),
			want:      envValueFalse,
		},
		{
			name:      "SGLang worker enables canary",
			backend:   &SGLangBackend{},
			component: workerHealthComponent(v1beta1.ComponentTypeWorker),
			container: workerHealthBaseContainer(),
			want:      envValueTrue,
		},
		{
			name:      "SGLang tokenizer flag keeps canary disabled",
			backend:   &SGLangBackend{},
			component: workerHealthComponent(v1beta1.ComponentTypeWorker),
			container: workerHealthContainerWithArgs("--use-sglang-tokenizer"),
			want:      envValueFalse,
		},
		{
			name:      "SGLang tokenizer env keeps canary disabled",
			backend:   &SGLangBackend{},
			component: workerHealthComponent(v1beta1.ComponentTypeWorker),
			container: workerHealthContainerWithEnv("DYN_SGL_USE_TOKENIZER", envValueTrue),
			want:      envValueFalse,
		},
		{
			name:    "user override is preserved",
			backend: &VLLMBackend{},
			component: &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name: commonconsts.MainContainerName,
							Env:  []corev1.EnvVar{{Name: dynHealthCheckEnabledEnv, Value: envValueFalse}},
						}},
					},
				},
			},
			container: workerHealthBaseContainer(),
			want:      envValueFalse,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.backend.UpdateContainer(&tt.container, 1, RoleWorker, tt.component, "test-service", nil)
			assert.Equal(t, tt.want, workerHealthEnvValue(tt.container, dynHealthCheckEnabledEnv))
		})
	}
}

func TestSetWorkerHealthCheckDefaultSkipsNonWorkers(t *testing.T) {
	container := workerHealthBaseContainer()

	setWorkerHealthCheckDefault(&container, workerHealthComponent(v1beta1.ComponentTypeFrontend), true)

	assert.Equal(t, envValueFalse, workerHealthEnvValue(container, dynHealthCheckEnabledEnv))
}

func workerHealthComponent(componentType v1beta1.ComponentType) *v1beta1.DynamoComponentDeploymentSharedSpec {
	return &v1beta1.DynamoComponentDeploymentSharedSpec{ComponentType: componentType}
}

func workerHealthBaseContainer() corev1.Container {
	return corev1.Container{Env: []corev1.EnvVar{{Name: dynHealthCheckEnabledEnv, Value: envValueFalse}}}
}

func workerHealthContainerWithArgs(args ...string) corev1.Container {
	container := workerHealthBaseContainer()
	container.Args = args
	return container
}

func workerHealthContainerWithEnv(name, value string) corev1.Container {
	container := workerHealthBaseContainer()
	container.Env = append(container.Env, corev1.EnvVar{Name: name, Value: value})
	return container
}

func workerHealthEnvValue(container corev1.Container, name string) string {
	for _, env := range container.Env {
		if env.Name == name {
			return env.Value
		}
	}
	return ""
}
