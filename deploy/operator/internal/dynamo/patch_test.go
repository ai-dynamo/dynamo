/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestGenerateBasePodSpec_StrategicMergeReplacesExistingUnionStyleEntries(t *testing.T) {
	secretsRetriever := &mockSecretsRetriever{}
	controllerConfig := &configv1alpha1.OperatorConfiguration{}
	controllerConfig.PodGeneration.DefaultExtraPodSpecMergeStrategy = v1alpha1.ExtraPodSpecMergeStrategyStrategic

	t.Run("container ports with same name", func(t *testing.T) {
		component := &v1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: "decode",
			ComponentType: v1beta1.ComponentTypeDecode,
			PodTemplate: &corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: commonconsts.MainContainerName,
							Ports: []corev1.ContainerPort{
								{
									Name:          commonconsts.DynamoSystemPortName,
									ContainerPort: 28090,
									Protocol:      corev1.ProtocolTCP,
								},
							},
						},
					},
				},
			},
		}

		podSpec, err := GenerateBasePodSpec(
			component,
			BackendFrameworkVLLM,
			secretsRetriever,
			"test-deployment",
			"default",
			RoleMain,
			1,
			controllerConfig,
			commonconsts.MultinodeDeploymentTypeGrove,
			"decode",
			nil,
			staticContainerGPUCount(0),
		)

		require.NoError(t, err)
		require.Len(t, podSpec.Containers, 1)

		systemPorts := make([]int32, 0)
		for _, port := range podSpec.Containers[0].Ports {
			if port.Name == commonconsts.DynamoSystemPortName {
				systemPorts = append(systemPorts, port.ContainerPort)
			}
		}
		assert.Equal(t, []int32{28090}, systemPorts)
		assert.NotContains(t, podSpec.Containers[0].Ports, corev1.ContainerPort{
			Name:          commonconsts.DynamoSystemPortName,
			ContainerPort: int32(commonconsts.DynamoSystemPort),
			Protocol:      corev1.ProtocolTCP,
		})
	})

	t.Run("env vars", func(t *testing.T) {
		component := &v1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: "frontend",
			ComponentType: v1beta1.ComponentTypeFrontend,
			PodTemplate: &corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: commonconsts.MainContainerName,
							Env: []corev1.EnvVar{
								{Name: "POD_NAME", Value: "static-pod-name"},
							},
						},
					},
				},
			},
		}

		podSpec, err := GenerateBasePodSpec(
			component,
			BackendFrameworkVLLM,
			secretsRetriever,
			"test-deployment",
			"default",
			RoleMain,
			1,
			controllerConfig,
			commonconsts.MultinodeDeploymentTypeGrove,
			"frontend",
			nil,
			staticContainerGPUCount(0),
		)

		require.NoError(t, err)
		require.Len(t, podSpec.Containers, 1)

		matches := 0
		var podNameEnv *corev1.EnvVar
		for i := range podSpec.Containers[0].Env {
			if podSpec.Containers[0].Env[i].Name == "POD_NAME" {
				matches++
				podNameEnv = &podSpec.Containers[0].Env[i]
			}
		}

		require.Equal(t, 1, matches)
		require.NotNil(t, podNameEnv)
		assert.Equal(t, "static-pod-name", podNameEnv.Value)
		assert.Nil(t, podNameEnv.ValueFrom)
	})
}

func TestGenerateDynamoComponentsDeployments_ExtraPodSpecMergeStrategyPropagation(t *testing.T) {
	tests := []struct {
		name         string
		strategy     v1beta1.ExtraPodSpecMergeStrategy
		wantStrategy v1beta1.ExtraPodSpecMergeStrategy
	}{
		{
			name:         "explicit strategy propagates to generated DCD spec",
			strategy:     v1beta1.ExtraPodSpecMergeStrategyStrategic,
			wantStrategy: v1beta1.ExtraPodSpecMergeStrategyStrategic,
		},
		{
			name:         "unset strategy remains unset on generated DCD spec",
			strategy:     "",
			wantStrategy: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					BackendFramework: "vllm",
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName:             "frontend",
							ComponentType:             v1beta1.ComponentTypeFrontend,
							Replicas:                  ptr.To(int32(1)),
							ExtraPodSpecMergeStrategy: tt.strategy,
						},
					},
				},
			}

			dcds, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
			require.NoError(t, err)

			dcd := dcds["frontend"]
			require.NotNil(t, dcd)
			assert.Equal(t, tt.wantStrategy, dcd.Spec.ExtraPodSpecMergeStrategy)
		})
	}
}

func TestGenerateBasePodSpec_InvalidExtraPodSpecMergeStrategy(t *testing.T) {
	secretsRetriever := &mockSecretsRetriever{}
	controllerConfig := &configv1alpha1.OperatorConfiguration{}
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName:             "frontend",
		ComponentType:             v1beta1.ComponentTypeFrontend,
		ExtraPodSpecMergeStrategy: v1beta1.ExtraPodSpecMergeStrategy("invalid"),
	}

	_, err := GenerateBasePodSpec(
		component,
		BackendFrameworkVLLM,
		secretsRetriever,
		"test-deployment",
		"default",
		RoleMain,
		1,
		controllerConfig,
		commonconsts.MultinodeDeploymentTypeGrove,
		"frontend",
		nil,
		staticContainerGPUCount(0),
	)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid extraPodSpec merge strategy")
}

func TestGenerateBasePodSpecForController_UsesOperatorConfigDefaultExtraPodSpecMergeStrategy(t *testing.T) {
	tests := []struct {
		name            string
		strategy        v1beta1.ExtraPodSpecMergeStrategy
		defaultStrategy v1alpha1.ExtraPodSpecMergeStrategy
		wantPortNames   []string
	}{
		{
			name:            "falls back to override when field is absent",
			defaultStrategy: v1alpha1.ExtraPodSpecMergeStrategyOverride,
			wantPortNames:   []string{"metrics"},
		},
		{
			name:            "falls back to runtime default when field is absent",
			strategy:        "",
			defaultStrategy: v1alpha1.ExtraPodSpecMergeStrategyStrategic,
			wantPortNames:   []string{commonconsts.DynamoContainerPortName, "metrics"},
		},
		{
			name:            "explicit field overrides runtime default",
			strategy:        v1beta1.ExtraPodSpecMergeStrategyOverride,
			defaultStrategy: v1alpha1.ExtraPodSpecMergeStrategyStrategic,
			wantPortNames:   []string{"metrics"},
		},
		{
			name:            "explicit strategic field overrides runtime override",
			strategy:        v1beta1.ExtraPodSpecMergeStrategyStrategic,
			defaultStrategy: v1alpha1.ExtraPodSpecMergeStrategyOverride,
			wantPortNames:   []string{commonconsts.DynamoContainerPortName, "metrics"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			controllerConfig := &configv1alpha1.OperatorConfiguration{}
			controllerConfig.PodGeneration.DefaultExtraPodSpecMergeStrategy = tt.defaultStrategy

			podSpec, err := GenerateBasePodSpecForController(
				&v1beta1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "frontend",
						Namespace: "default",
					},
					Spec: v1beta1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
							ComponentName:             "frontend",
							ComponentType:             v1beta1.ComponentTypeFrontend,
							ExtraPodSpecMergeStrategy: tt.strategy,
							PodTemplate: &corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{
										{
											Name: commonconsts.MainContainerName,
											Ports: []corev1.ContainerPort{
												{
													Name:          "metrics",
													ContainerPort: 9001,
													Protocol:      corev1.ProtocolTCP,
												},
											},
										},
									},
								},
							},
						},
					},
				},
				&mockSecretsRetriever{},
				controllerConfig,
				RoleMain,
				commonconsts.MultinodeDeploymentTypeGrove,
				staticContainerGPUCount(0),
				GenerateBasePodSpecForControllerOptions{},
			)
			require.NoError(t, err)
			require.Len(t, podSpec.Containers, 1)

			var gotPortNames []string
			for _, port := range podSpec.Containers[0].Ports {
				gotPortNames = append(gotPortNames, port.Name)
			}
			assert.ElementsMatch(t, tt.wantPortNames, gotPortNames)
		})
	}
}
