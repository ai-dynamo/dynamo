/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"encoding/json"
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestEnvironmentLayersThroughDCD(t *testing.T) {
	for _, origin := range []string{"1.7.0", "1.6.0", ""} {
		t.Run("origin="+origin, func(t *testing.T) {
			t.Log("A default references the component, while the graph references that default")
			component := v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"},
					Env: []corev1.EnvVar{{Name: "Z_COMPONENT", Value: "nats://custom:4222"}},
				}}}},
			}
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: origin},
				Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm",
					Env:        []corev1.EnvVar{{Name: "A_DERIVED", Value: "$(NATS_SERVER)/model"}},
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component},
				},
			}
			config := &configv1alpha1.OperatorConfiguration{Infrastructure: configv1alpha1.InfrastructureConfiguration{NATSAddress: "$(Z_COMPONENT)"}}
			original := dgd.DeepCopy()

			t.Log("Materialize the DCD without flattening away the source layers")
			children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
			require.NoError(t, err)
			child := children["worker"]
			child.OwnerReferences = []metav1.OwnerReference{{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeployment", Name: dgd.Name, Controller: new(true)}}
			if origin == "1.7.0" {
				require.Equal(t, dgd.Spec.Env, child.Spec.Env)
				require.Equal(t, component.PodTemplate.Spec.Containers[0].Env, child.Spec.PodTemplate.Spec.Containers[0].Env)
			} else {
				require.Empty(t, child.Spec.Env)
			}

			t.Log("Grove and the DCD/LWS entrypoint must produce the same effective environment")
			grovePod, err := GeneratePodSpecForComponent(&component, BackendFrameworkVLLM, nil, dgd, RoleMain, 1, config, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
			require.NoError(t, err)
			lwsPod, err := GenerateBasePodSpecForController(child, nil, config, RoleMain, commonconsts.MultinodeDeploymentTypeLWS, staticContainerGPUCount(0), GenerateBasePodSpecForControllerOptions{})
			require.NoError(t, err)
			require.Equal(t, grovePod.Containers[0].Env, lwsPod.Containers[0].Env)
			var names []string
			for _, env := range grovePod.Containers[0].Env {
				if env.Name == "Z_COMPONENT" || env.Name == "NATS_SERVER" || env.Name == "A_DERIVED" {
					names = append(names, env.Name)
				}
			}
			if origin == "1.7.0" {
				require.Equal(t, []string{"Z_COMPONENT", "NATS_SERVER", "A_DERIVED"}, names)
			} else {
				require.Equal(t, []string{"A_DERIVED", "NATS_SERVER", "Z_COMPONENT"}, names)
			}
			require.Equal(t, original, dgd)

			t.Log("The materialized child owns its snapshot of graph defaults")
			dgd.Spec.Env[0].Value = "changed"
			if origin == "1.7.0" {
				require.Equal(t, "$(NATS_SERVER)/model", child.Spec.Env[0].Value)
			}
		})
	}
}

func TestEnvironmentCyclesFailBeforeRenderingWorkloads(t *testing.T) {
	t.Log("Create a cross-layer cycle that neither provider may silently reorder")
	component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name: "main", Image: "engine:1.7.0", Env: []corev1.EnvVar{{Name: "B", Value: "$(A)"}},
		}}}},
	}
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
		Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.7.0"},
		Spec:        v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Env: []corev1.EnvVar{{Name: "A", Value: "$(B)"}}, Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component}},
	}
	config := &configv1alpha1.OperatorConfiguration{}
	pod, err := GeneratePodSpecForComponent(&component, BackendFrameworkVLLM, nil, dgd, RoleMain, 1, config, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
	require.Nil(t, pod)
	require.ErrorContains(t, err, "environment dependency cycle")
	require.ErrorContains(t, err, "spec.env[0]")
	require.ErrorContains(t, err, "podTemplate.spec.containers[name=main].env[0]")

	t.Log("The materialized child retains the cycle rather than flattening it away")
	children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
	require.NoError(t, err)
	pod, err = GenerateBasePodSpecForController(children["worker"], nil, config, RoleMain, commonconsts.MultinodeDeploymentTypeLWS, staticContainerGPUCount(0), GenerateBasePodSpecForControllerOptions{})
	require.Nil(t, pod)
	require.ErrorContains(t, err, "environment dependency cycle")
}

func TestEnvironmentLateGMSBindings(t *testing.T) {
	for _, mode := range []v1beta1.GPUMemoryServiceMode{v1beta1.GMSModeIntraPod, v1beta1.GMSModeInterPod} {
		t.Run(string(mode), func(t *testing.T) {
			t.Log("Reference a late GMS binding from graph defaults and try overriding an owned mode")
			component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				Experimental: &v1beta1.ExperimentalSpec{GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{Mode: mode}},
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"},
					Env:       []corev1.EnvVar{{Name: "NATS_SERVER", Value: "custom"}, {Name: "DYN_VLLM_GMS_SHADOW_MODE", Value: "false"}},
					Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}},
				}}}},
			}
			if mode == v1beta1.GMSModeIntraPod {
				component.Experimental.GPUMemoryService.ExtraClientContainers = []string{"main"}
			} else {
				component.Experimental.Failover = &v1beta1.FailoverSpec{Mode: mode}
			}
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.7.0"},
				Spec:        v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Env: []corev1.EnvVar{{Name: "WEIGHTS", Value: "$(GMS_SOCKET_DIR)/weights"}}},
			}
			pod, err := generatePodSpecForRole(ServiceRole{Name: "worker", Role: RoleMain}, &component, BackendFrameworkVLLM, nil, dgd, 1,
				&configv1alpha1.OperatorConfiguration{Infrastructure: configv1alpha1.InfrastructureConfiguration{NATSAddress: "default"}}, "worker", roleLaunchOwnedByOperator, false, nil, staticContainerGPUCount(1))
			require.NoError(t, err)
			env := pod.Containers[0].Env
			require.Len(t, envVarsToMap(env), len(env), "cross-layer overrides must not create duplicate names")
			require.Equal(t, "custom", envVarsToMap(env)["NATS_SERVER"])
			require.Less(t, envIndex(t, env, gms.EnvSocketDir), envIndex(t, env, "WEIGHTS"))
			if mode == v1beta1.GMSModeInterPod {
				require.Equal(t, "true", envVarsToMap(env)["DYN_VLLM_GMS_SHADOW_MODE"])
			}
		})
	}
}

func TestEnvironmentGraphDeploymentConfigReaders(t *testing.T) {
	for _, componentType := range []v1beta1.ComponentType{v1beta1.ComponentTypeWorker, v1beta1.ComponentTypeFrontend} {
		t.Run(string(componentType), func(t *testing.T) {
			t.Log("Keep graph-scoped deployment configuration visible without flattening env")
			component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: string(componentType), ComponentType: componentType,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "engine:1.7.0"}}}},
			}
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.7.0"},
				Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component},
					Env: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: `{"worker":{"ServiceArgs":{"workers":3,"resources":{"cpu":"2"}}},"frontend":{"port":1234}}`}},
				},
			}
			children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
			require.NoError(t, err)
			child := children[string(componentType)]
			if componentType == v1beta1.ComponentTypeWorker {
				require.EqualValues(t, 3, *child.Spec.Replicas)
				require.Equal(t, "2", child.Spec.PodTemplate.Spec.Containers[0].Resources.Requests.Cpu().String())
			} else {
				config := findEnvVar(child.Spec.PodTemplate.Spec.Containers[0].Env, commonconsts.DynamoDeploymentConfigEnvVar)
				require.NotNil(t, config)
				var parsed map[string]map[string]any
				require.NoError(t, json.Unmarshal([]byte(config.Value), &parsed))
				require.EqualValues(t, commonconsts.DynamoServicePort, parsed["frontend"]["port"])
			}
		})
	}
}

func TestSemanticTopologyBindingsAndMPIExports(t *testing.T) {
	for _, backend := range []BackendFramework{BackendFrameworkNoop, BackendFrameworkTRTLLM} {
		t.Run(string(backend), func(t *testing.T) {
			t.Log("Authored aliases cannot replace provider bindings used by a graph-level reference")
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.7.0"},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.trtllm"},
						Env: []corev1.EnvVar{{Name: commonconsts.DynamoRankEnvVar, Value: "99"}, {Name: commonconsts.DynamoLeaderAddressEnvVar, Value: "ignored"}},
					}}},
				},
			}
			pod, err := GenerateBasePodSpec(component, backend, nil, "graph", "ns", RoleLeader, 2, &configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(1),
				corev1.EnvVar{Name: "TOPOLOGY", Value: "$(DYNAMO_RANK)/$(DYNAMO_LEADER_ADDRESS)"})
			require.NoError(t, err)
			container := pod.Containers[0]
			require.Len(t, envVarsToMap(container.Env), len(container.Env))
			require.Equal(t, "$(GROVE_PCSG_POD_INDEX)", envVarsToMap(container.Env)[commonconsts.DynamoRankEnvVar])
			require.Less(t, envIndex(t, container.Env, commonconsts.DynamoRankEnvVar), envIndex(t, container.Env, "TOPOLOGY"))
			require.Less(t, envIndex(t, container.Env, commonconsts.DynamoLeaderAddressEnvVar), envIndex(t, container.Env, "TOPOLOGY"))
			if backend == BackendFrameworkTRTLLM {
				t.Log("Backend command construction must still see the bindings before final env composition")
				command := strings.Join(append(container.Command, container.Args...), " ")
				require.Contains(t, command, "-x DYNAMO_RANK")
				require.Contains(t, command, "-x DYNAMO_LEADER_ADDRESS")
			}
		})
	}
}

func TestEffectiveEnvironmentDoesNotValidateBeforeOwnedBindings(t *testing.T) {
	t.Log("An apparent graph/container cycle disappears when the operator binds rank")
	component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
		Multinode: &v1beta1.MultinodeSpec{NodeCount: 2},
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name: "main", Image: "engine:1.7.0", Env: []corev1.EnvVar{{Name: "USER", Value: "$(DYNAMO_RANK)"}},
		}}}},
	}
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
		Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.7.0"},
		Spec:        v1beta1.DynamoGraphDeploymentSpec{Env: []corev1.EnvVar{{Name: commonconsts.DynamoRankEnvVar, Value: "$(USER)"}}, Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component}},
	}
	_, err := EffectiveDGDComponentTemplates(dgd, &component)
	require.NoError(t, err, "input consumers cannot validate a partial environment")
	pod, err := GeneratePodSpecForComponent(&component, BackendFrameworkNoop, nil, dgd, RoleLeader, 2, &configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
	require.NoError(t, err)
	require.Equal(t, "$(GROVE_PCSG_POD_INDEX)", envVarsToMap(pod.Containers[0].Env)[commonconsts.DynamoRankEnvVar])
	require.Less(t, envIndex(t, pod.Containers[0].Env, commonconsts.DynamoRankEnvVar), envIndex(t, pod.Containers[0].Env, "USER"))
}
