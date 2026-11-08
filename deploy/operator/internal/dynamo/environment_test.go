/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestEnvironmentLayersThroughDCD(t *testing.T) {
	for _, origin := range []string{"1.8.0", "1.7.0", "1.6.0", ""} {
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
			if origin == "1.8.0" {
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
			if origin == "1.8.0" {
				require.Equal(t, []string{"Z_COMPONENT", "NATS_SERVER", "A_DERIVED"}, names)
			} else {
				require.Equal(t, []string{"A_DERIVED", "NATS_SERVER", "Z_COMPONENT"}, names)
			}
			require.Equal(t, original, dgd)

			t.Log("The materialized child owns its snapshot of graph defaults")
			dgd.Spec.Env[0].Value = "changed"
			if origin == "1.8.0" {
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
		Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
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

func TestKVPolicyKeepsComputeOnlyRoleEnvironment(t *testing.T) {
	t.Log("Only the leader hosts a Dynamo runtime; the worker engine retains graph defaults")
	policy := corev1.ContainerRestartPolicyAlways
	engine := corev1.Container{Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"}}
	component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
		Multinode: &v1beta1.MultinodeSpec{NodeCount: 2},
		Roles: []v1beta1.ComponentRoleSpec{
			{Name: "leader", PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{engine},
				InitContainers: []corev1.Container{{Name: "runtime", Image: "runtime:1.7.0", RestartPolicy: &policy}}}}},
			{Name: "worker", PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{*engine.DeepCopy()}}}},
		},
	}
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
		Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component},
			Env: []corev1.EnvVar{{Name: commonconsts.EnvTopologyMountPath, Value: "/engine-data"}},
			Experimental: &v1beta1.DynamoGraphDeploymentExperimentalSpec{KvTransferPolicy: &v1beta1.KvTransferPolicy{
				LabelKey: "topology.kubernetes.io/zone", Domain: "zone", Enforcement: v1beta1.KvTransferEnforcementRequired,
			}},
		},
	}
	config := &configv1alpha1.OperatorConfiguration{}
	pod, err := GeneratePodSpecForComponent(&component, BackendFrameworkVLLM, nil, dgd, RoleWorker, 2, config, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
	require.NoError(t, err)
	require.Empty(t, pod.InitContainers)
	require.Equal(t, "/engine-data", envVarsToMap(pod.Containers[0].Env)[commonconsts.EnvTopologyMountPath])

	t.Log("The graph-generated DCD keeps the same compute-only role semantics")
	children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
	require.NoError(t, err)
	child := children["worker"]
	child.OwnerReferences = []metav1.OwnerReference{{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeployment", Name: dgd.Name, Controller: new(true)}}
	pod, err = GenerateBasePodSpecForController(child, nil, config, RoleWorker, commonconsts.MultinodeDeploymentTypeLWS, staticContainerGPUCount(0), GenerateBasePodSpecForControllerOptions{})
	require.NoError(t, err)
	require.Empty(t, pod.InitContainers)
	require.Equal(t, "/engine-data", envVarsToMap(pod.Containers[0].Env)[commonconsts.EnvTopologyMountPath])
}

func TestStandaloneTopologySourceDoesNotOwnEnvironment(t *testing.T) {
	t.Log("A standalone DCD's topology-label projection is not a compiled graph policy")
	dcd := &v1beta1.DynamoComponentDeployment{Name: "worker", Namespace: "ns",
		Annotations: map[string]string{commonconsts.KubeAnnotationTopologyLabelKey: "topology.kubernetes.io/zone"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{BackendFramework: "vllm",
			Env: []corev1.EnvVar{{Name: commonconsts.EnvKvTransferPreferredWeight, Value: "0.75"}},
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"}}}},
				},
			},
		},
	}
	pod, err := GenerateBasePodSpecForController(dcd, nil, &configv1alpha1.OperatorConfiguration{}, RoleMain, commonconsts.MultinodeDeploymentTypeLWS, staticContainerGPUCount(0), GenerateBasePodSpecForControllerOptions{})
	require.NoError(t, err)
	require.Equal(t, "0.75", envVarsToMap(pod.Containers[0].Env)[commonconsts.EnvKvTransferPreferredWeight])
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
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
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
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
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
				config := findEnvVar(slices.Concat(child.Spec.Env, child.Spec.PodTemplate.Spec.Containers[0].Env), commonconsts.DynamoDeploymentConfigEnvVar)
				require.NotNil(t, config)
				var parsed map[string]map[string]any
				require.NoError(t, json.Unmarshal([]byte(config.Value), &parsed))
				require.EqualValues(t, commonconsts.DynamoServicePort, parsed["frontend"]["port"])
			}
		})
	}
}

func TestEnvironmentDeploymentConfigDuplicateDefinitions(t *testing.T) {
	for _, origin := range []string{"1.8.0", "1.7.99", ""} {
		t.Run("origin="+origin, func(t *testing.T) {
			t.Log("Convert an alpha frontend whose atomic env list contains repeated config names")
			alpha := &v1alpha1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: origin},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {ComponentType: "frontend", Envs: []corev1.EnvVar{
						{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: `{"frontend":{"port":1111}}`},
						{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: `{"frontend":{"port":2222}}`},
					}},
				}},
			}
			dgd := &v1beta1.DynamoGraphDeployment{}
			require.NoError(t, alpha.ConvertTo(dgd))
			children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
			require.NoError(t, err)
			env := children["frontend"].Spec.PodTemplate.Spec.Containers[0].Env
			require.Len(t, env, 2)
			t.Log("Only semantic origins rewrite the effective last definition; legacy output is unchanged")
			updated, untouched := 0, 1
			if origin == "1.8.0" {
				updated, untouched = 1, 0
			}
			var config map[string]map[string]int
			require.NoError(t, json.Unmarshal([]byte(env[updated].Value), &config))
			require.Equal(t, commonconsts.DynamoServicePort, config["frontend"]["port"])
			require.Equal(t, alpha.Spec.Services["frontend"].Envs[untouched], env[untouched])
		})
	}
}

func TestFrontendDeploymentConfigKeepsGeneralSource(t *testing.T) {
	for _, chained := range []bool{false, true} {
		t.Run(fmt.Sprintf("chained=%t", chained), func(t *testing.T) {
			t.Log("An authored frontend value depends on graph-sourced deployment configuration")
			general := []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: `{"frontend":{"port":1234}}`}}
			authored := []corev1.EnvVar{{Name: "CFG", Value: "$(DYN_DEPLOYMENT_CONFIG)"}}
			want := []string{commonconsts.DynamoDeploymentConfigEnvVar, "CFG"}
			if chained {
				general = append(general, corev1.EnvVar{Name: "DERIVED", Value: "$(DYN_DEPLOYMENT_CONFIG)"})
				authored = []corev1.EnvVar{{Name: "FRONTEND_VAR", Value: "$(DERIVED)"}}
				want = []string{commonconsts.DynamoDeploymentConfigEnvVar, "DERIVED", "FRONTEND_VAR"}
			}
			component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "frontend:1.7.0", Env: authored}}}},
			}
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
				Spec:        v1beta1.DynamoGraphDeploymentSpec{Env: general, Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component}},
			}
			original := dgd.DeepCopy()

			t.Log("Normalize without promoting the rewritten general binding into the authored list")
			normalized, err := NormalizeDynamoGraphDeploymentComponents(dgd, nil, nil, RollingUpdateContext{})
			require.NoError(t, err)
			require.Equal(t, authored, normalized["frontend"].PodTemplate.Spec.Containers[0].Env)
			children, err := GenerateDynamoComponentsDeploymentsFromNormalized(dgd, normalized, RollingUpdateContext{})
			require.NoError(t, err)
			child := children["frontend"]
			child.OwnerReferences = []metav1.OwnerReference{{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeployment", Name: dgd.Name, Controller: new(true)}}
			require.Equal(t, authored, child.Spec.PodTemplate.Spec.Containers[0].Env)
			require.Len(t, child.Spec.Env, len(general))
			require.JSONEq(t, `{"frontend":{"port":8000}}`, child.Spec.Env[0].Value)
			require.Equal(t, general[1:], child.Spec.Env[1:])

			t.Log("Direct Grove, normalized DS and stored DCD rendering keep the same valid dependency order")
			config := &configv1alpha1.OperatorConfiguration{}
			grove, err := GeneratePodSpecForComponent(&component, BackendFrameworkNoop, nil, dgd, RoleMain, 1, config, commonconsts.MultinodeDeploymentTypeGrove, "frontend", nil, staticContainerGPUCount(0))
			require.NoError(t, err)
			ds, err := GenerateBasePodSpec(normalized["frontend"], BackendFrameworkNoop, nil, dgd.Name, dgd.Namespace, RoleMain, 1, config, commonconsts.MultinodeDeploymentTypeLWS, "frontend", nil, staticContainerGPUCount(0), GenerateBasePodSpecOptions{GeneralEnvironment: dgd.Spec.Env})
			require.NoError(t, err)
			stored, err := GenerateBasePodSpecForController(child, nil, config, RoleMain, commonconsts.MultinodeDeploymentTypeLWS, staticContainerGPUCount(0), GenerateBasePodSpecForControllerOptions{})
			require.NoError(t, err)
			for _, pod := range []*corev1.PodSpec{grove, ds, stored} {
				var names []string
				for _, binding := range pod.Containers[0].Env {
					if slices.Contains(want, binding.Name) {
						names = append(names, binding.Name)
					}
				}
				require.Equal(t, want, names)
				require.JSONEq(t, `{"frontend":{"port":8000}}`, findEnvVar(pod.Containers[0].Env, commonconsts.DynamoDeploymentConfigEnvVar).Value)
			}
			require.Equal(t, original, dgd)
		})
	}
}

func TestFrontendDeploymentConfigIgnoresShadowedGeneralSource(t *testing.T) {
	t.Log("A valid authored config shadows an unreadable graph-level binding")
	authored := []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: `{"frontend":{"port":1234}}`}}
	component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "frontend:1.7.0", Env: authored}}}},
	}
	general := []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{Name: "config", Key: "value"}}}}
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
		Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
		Spec:        v1beta1.DynamoGraphDeploymentSpec{Env: general, Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component}},
	}
	original := dgd.DeepCopy()

	t.Log("Do not parse or rewrite the shadowed source when snapshotting a child")
	children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
	require.NoError(t, err)
	child := children["frontend"]
	require.Equal(t, general, child.Spec.Env)
	require.JSONEq(t, `{"frontend":{"port":8000}}`, child.Spec.PodTemplate.Spec.Containers[0].Env[0].Value)
	require.Equal(t, original, dgd)
}

func TestFrontendDeploymentConfigPreservesSourceDiagnostics(t *testing.T) {
	t.Log("Rewriting the first general value does not renumber a later genuine dependency cycle")
	component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "frontend:1.7.0", Env: []corev1.EnvVar{{Name: "B", Value: "$(A)"}}}}}},
	}
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
		Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component},
			Env: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: `{"frontend":{"port":1234}}`}, {Name: "A", Value: "$(B)"}},
		},
	}
	config := &configv1alpha1.OperatorConfiguration{}
	_, err := GeneratePodSpecForComponent(&component, BackendFrameworkNoop, nil, dgd, RoleMain, 1, config, commonconsts.MultinodeDeploymentTypeGrove, "frontend", nil, staticContainerGPUCount(0))
	require.ErrorContains(t, err, "spec.env[1]")
	require.ErrorContains(t, err, "podTemplate.spec.containers[name=main].env[0]")

	t.Log("Persisting the projected general source preserves the same diagnostic indexes")
	children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
	require.NoError(t, err)
	_, err = GenerateBasePodSpecForController(children["frontend"], nil, config, RoleMain, commonconsts.MultinodeDeploymentTypeLWS, staticContainerGPUCount(0), GenerateBasePodSpecForControllerOptions{})
	require.ErrorContains(t, err, "spec.env[1]")
	require.ErrorContains(t, err, "podTemplate.spec.containers[name=main].env[0]")
}

func TestFrontendDeploymentConfigKeepsLegacyOutput(t *testing.T) {
	for _, origin := range []string{"1.7.0", ""} {
		t.Run("origin="+origin, func(t *testing.T) {
			t.Log("Legacy normalization still flattens, sorts and rewrites into the main source")
			component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "frontend:1.7.0", Env: []corev1.EnvVar{{Name: "CFG", Value: "$(DYN_DEPLOYMENT_CONFIG)"}}}}}},
			}
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
				Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component},
					Env: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: `{"frontend":{"port":1234}}`}},
				},
			}
			if origin != "" {
				dgd.Annotations = map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: origin}
			}
			original := dgd.DeepCopy()
			children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
			require.NoError(t, err)
			child := children["frontend"]
			require.Empty(t, child.Spec.Env)
			require.Equal(t, []corev1.EnvVar{{Name: "CFG", Value: "$(DYN_DEPLOYMENT_CONFIG)"}, {Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: `{"frontend":{"port":8000}}`}}, child.Spec.PodTemplate.Spec.Containers[0].Env)
			require.Equal(t, original, dgd)
		})
	}
}

func TestRoleEnvironmentGraphDeploymentConfigResources(t *testing.T) {
	for _, origin := range []string{"1.8.0", "1.7.99", ""} {
		t.Run("origin="+origin, func(t *testing.T) {
			t.Log("Role templates inherit graph configuration with admission-defaulted replicas")
			template := corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
				Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"},
			}}}}
			component := v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker, Replicas: new(int32(1)),
				Multinode: &v1beta1.MultinodeSpec{NodeCount: 2},
				Roles: []v1beta1.ComponentRoleSpec{
					{Name: "leader", PodTemplate: template.DeepCopy()},
					{Name: "worker", PodTemplate: template.DeepCopy()},
				},
			}
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: origin},
				Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component},
					Env: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: `{"worker":{"ServiceArgs":{"workers":3,"resources":{"cpu":"2","memory":"8Gi","gpu":"1"}}}}`}},
				},
			}
			original := dgd.DeepCopy()

			t.Log("Both DS normalization and DCD persistence populate every role's resources")
			normalized, err := NormalizeDynamoGraphDeploymentComponents(dgd, nil, nil, RollingUpdateContext{})
			require.NoError(t, err)
			children, err := GenerateDynamoComponentsDeploymentsFromNormalized(dgd, normalized, RollingUpdateContext{})
			require.NoError(t, err)
			for _, actual := range []*v1beta1.DynamoComponentDeploymentSharedSpec{normalized["worker"], &children["worker"].Spec.DynamoComponentDeploymentSharedSpec} {
				require.Nil(t, actual.PodTemplate)
				require.EqualValues(t, 1, *actual.Replicas, "deployment configuration must not override the admitted replica target")
				for _, role := range actual.Roles {
					resources := role.PodTemplate.Spec.Containers[0].Resources
					for _, values := range []corev1.ResourceList{resources.Requests, resources.Limits} {
						require.Equal(t, "2", values.Cpu().String(), role.Name)
						require.Equal(t, "8Gi", values.Memory().String(), role.Name)
						require.Equal(t, resource.MustParse("1"), values["nvidia.com/gpu"], role.Name)
					}
				}
			}
			for _, child := range children {
				if origin == "1.8.0" {
					require.Equal(t, dgd.Spec.Env, child.Spec.Env)
				} else {
					require.Empty(t, child.Spec.Env)
				}
			}
			require.Equal(t, original, dgd)
		})
	}
}

func TestSemanticTopologyBindingsAndMPIExports(t *testing.T) {
	for _, backend := range []BackendFramework{BackendFrameworkNoop, BackendFrameworkTRTLLM} {
		t.Run(string(backend), func(t *testing.T) {
			t.Log("Authored aliases cannot replace provider bindings used by a graph-level reference")
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.trtllm"},
						Env: []corev1.EnvVar{{Name: commonconsts.DynamoRankEnvVar, Value: "99"}, {Name: commonconsts.DynamoLeaderAddressEnvVar, Value: "ignored"}},
					}}},
				},
			}
			pod, err := GenerateBasePodSpec(component, backend, nil, "graph", "ns", RoleLeader, 2, &configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(1),
				GenerateBasePodSpecOptions{GeneralEnvironment: []corev1.EnvVar{{Name: "TOPOLOGY", Value: "$(DYNAMO_RANK)/$(DYNAMO_LEADER_ADDRESS)"}}})
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
		Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
		Spec:        v1beta1.DynamoGraphDeploymentSpec{Env: []corev1.EnvVar{{Name: commonconsts.DynamoRankEnvVar, Value: "$(USER)"}}, Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component}},
	}
	_, err := EffectiveDGDComponentTemplates(dgd, &component)
	require.NoError(t, err, "input consumers cannot validate a partial environment")
	pod, err := GeneratePodSpecForComponent(&component, BackendFrameworkNoop, nil, dgd, RoleLeader, 2, &configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
	require.NoError(t, err)
	require.Equal(t, "$(GROVE_PCSG_POD_INDEX)", envVarsToMap(pod.Containers[0].Env)[commonconsts.DynamoRankEnvVar])
	require.Less(t, envIndex(t, pod.Containers[0].Env, commonconsts.DynamoRankEnvVar), envIndex(t, pod.Containers[0].Env, "USER"))
}

func TestEnvironmentKVPolicyOwnsAbsentGraphSettings(t *testing.T) {
	for _, origin := range []string{"1.8.0", "1.7.0", "1.6.0", ""} {
		for _, runtimeSidecar := range []bool{false, true} {
			t.Run(fmt.Sprintf("origin=%s/sidecar=%t", origin, runtimeSidecar), func(t *testing.T) {
				t.Log("Required KV-transfer policy removes a graph-provided preferred weight")
				component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
						Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"},
					}}}},
				}
				if runtimeSidecar {
					policy := corev1.ContainerRestartPolicyAlways
					component.PodTemplate.Spec.InitContainers = []corev1.Container{{Name: "runtime", Image: "runtime:1.7.0", RestartPolicy: &policy}}
					component.PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "MODEL_PATH", Value: "$(DYN_TOPOLOGY_MOUNT_PATH)/model"}}
				}
				dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
					Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: origin},
					Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component},
						Env: []corev1.EnvVar{{Name: commonconsts.EnvKvTransferPreferredWeight, Value: "heavy"}, {Name: commonconsts.EnvTopologyMountPath, Value: "/engine-data"}, {Name: "GLOBAL", Value: "retained"}},
						Experimental: &v1beta1.DynamoGraphDeploymentExperimentalSpec{KvTransferPolicy: &v1beta1.KvTransferPolicy{
							LabelKey: "topology.kubernetes.io/zone", Domain: "zone", Enforcement: v1beta1.KvTransferEnforcementRequired,
						}},
					},
				}
				original := dgd.DeepCopy()
				config := &configv1alpha1.OperatorConfiguration{}
				pod, err := GeneratePodSpecForComponent(&component, BackendFrameworkVLLM, nil, dgd, RoleMain, 1, config, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
				require.NoError(t, err)
				pods := []*corev1.PodSpec{pod}

				if origin == "1.8.0" {
					t.Log("DS normalization keeps authored env intact until the final policy binding")
					normalized, err := NormalizeDynamoGraphDeploymentComponents(dgd, nil, nil, RollingUpdateContext{})
					require.NoError(t, err)
					require.Equal(t, component.PodTemplate.Spec.Containers[0].Env, normalized["worker"].PodTemplate.Spec.Containers[0].Env)
					if runtimeSidecar {
						require.Equal(t, component.PodTemplate.Spec.InitContainers[0].Env, normalized["worker"].PodTemplate.Spec.InitContainers[0].Env)
					}
					pod, err = GenerateBasePodSpec(normalized["worker"], BackendFrameworkVLLM, nil, dgd.Name, dgd.Namespace, RoleMain, 1, config, commonconsts.MultinodeDeploymentTypeLWS, "worker", nil, staticContainerGPUCount(0),
						GenerateBasePodSpecOptions{GeneralEnvironment: dgd.Spec.Env, KVTransferPolicy: KVTransferPolicyEnvironment(dgd.Spec.Experimental.KvTransferPolicy)})
					require.NoError(t, err)
					pods = append(pods, pod)
				}

				t.Log("Materializing and rendering a DCD retains the same policy-owned absence")
				children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
				require.NoError(t, err)
				child := children["worker"]
				if origin == "1.8.0" {
					require.Equal(t, dgd.Spec.Env, child.Spec.Env, "the DCD must retain the unfiltered graph source")
				}
				child.OwnerReferences = []metav1.OwnerReference{{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeployment", Name: dgd.Name, Controller: new(true)}}
				pod, err = GenerateBasePodSpecForController(child, nil, config, RoleMain, commonconsts.MultinodeDeploymentTypeLWS, staticContainerGPUCount(0), GenerateBasePodSpecForControllerOptions{})
				require.NoError(t, err)
				pods = append(pods, pod)
				for _, rendered := range pods {
					runtime := &rendered.Containers[0]
					if runtimeSidecar {
						runtime = &rendered.InitContainers[0]
						engineEnv := envVarsToMap(rendered.Containers[0].Env)
						require.Equal(t, "/engine-data", engineEnv[commonconsts.EnvTopologyMountPath])
						require.Equal(t, "heavy", engineEnv[commonconsts.EnvKvTransferPreferredWeight])
					}
					env := envVarsToMap(runtime.Env)
					require.NotContains(t, env, commonconsts.EnvKvTransferPreferredWeight)
					require.Equal(t, "required", env[commonconsts.EnvKvTransferEnforcement])
					require.Equal(t, "retained", env["GLOBAL"])
				}
				require.Equal(t, original, dgd)

				if origin == "1.8.0" && !runtimeSidecar {
					t.Log("Removing an owned setting preserves the original graph indexes in diagnostics")
					dgd.Spec.Env = []corev1.EnvVar{{Name: commonconsts.EnvKvTransferPreferredWeight, Value: "heavy"}, {Name: "A", Value: "$(B)"}}
					component.PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "B", Value: "$(A)"}}
					_, err := GeneratePodSpecForComponent(&component, BackendFrameworkVLLM, nil, dgd, RoleMain, 1, config, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
					require.ErrorContains(t, err, "spec.env[1]")
					require.ErrorContains(t, err, "podTemplate.spec.containers[name=main].env[0]")

					t.Log("A persisted DCD diagnostic instead names its own stored normalized index")
					dgd.Spec.Components[0] = component
					children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
					require.NoError(t, err)
					child := children["worker"]
					child.OwnerReferences = []metav1.OwnerReference{{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeployment", Name: dgd.Name, Controller: new(true)}}
					index := len(child.Spec.PodTemplate.Spec.Containers[0].Env) - 1
					_, err = GenerateBasePodSpecForController(child, nil, config, RoleMain, commonconsts.MultinodeDeploymentTypeLWS, staticContainerGPUCount(0), GenerateBasePodSpecForControllerOptions{})
					require.ErrorContains(t, err, fmt.Sprintf("podTemplate.spec.containers[name=main].env[%d]", index))
				}
			})
		}
	}
}
