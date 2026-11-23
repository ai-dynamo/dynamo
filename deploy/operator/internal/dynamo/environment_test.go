/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// renderStoredEnvironment passes a persisted DCD's sources through the shared renderer.
// Controller tests cover adapter-specific origin propagation and workload selection.
func renderStoredEnvironment(dcd *v1beta1.DynamoComponentDeployment, config *configv1alpha1.OperatorConfiguration, role Role) (*corev1.PodSpec, error) {
	backend, err := GetBackendFrameworkFromDynamoComponent(dcd)
	if err != nil {
		return nil, err
	}
	component := &dcd.Spec.DynamoComponentDeploymentSharedSpec
	return GenerateBasePodSpec(component, backend, nil, dcd.GetParentGraphDeploymentName(), dcd.Namespace,
		role, component.GetNumberOfNodes(), config, commonconsts.MultinodeDeploymentTypeLWS, GetDCDComponentName(dcd), nil,
		staticContainerGPUCount(0), GenerateBasePodSpecOptions{GeneralEnvironment: dcd.Spec.Env,
			KVTransferPolicy: DCDKVTransferPolicyEnvironment(dcd, component)})
}

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
			lwsPod, err := renderStoredEnvironment(child, config, RoleMain)
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
	pod, err = renderStoredEnvironment(children["worker"], config, RoleMain)
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
	pod, err = renderStoredEnvironment(child, config, RoleWorker)
	require.NoError(t, err)
	require.Empty(t, pod.InitContainers)
	require.Equal(t, "/engine-data", envVarsToMap(pod.Containers[0].Env)[commonconsts.EnvTopologyMountPath])
}

func TestStandaloneTopologySourceDoesNotOwnEnvironment(t *testing.T) {
	for _, test := range []struct {
		name       string
		apiVersion string
		controller bool
		ownsPolicy bool
	}{
		{name: "standalone"},
		{name: "non-controlling DGD", apiVersion: "nvidia.com/v1beta1"},
		{name: "foreign DGD controller", apiVersion: "example.com/v1beta1", controller: true},
		{name: "Dynamo DGD controller", apiVersion: "nvidia.com/v1beta1", controller: true, ownsPolicy: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Only a controlling Dynamo DGD can own the persisted topology bindings")
			dcd := &v1beta1.DynamoComponentDeployment{Name: "worker", Namespace: "ns",
				Annotations: map[string]string{commonconsts.KubeAnnotationTopologyLabelKey: "topology.kubernetes.io/zone", commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
				Spec: v1beta1.DynamoComponentDeploymentSpec{BackendFramework: "vllm",
					Env: []corev1.EnvVar{{Name: commonconsts.EnvKvTransferPreferredWeight, Value: "0.75"}},
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{ComponentType: v1beta1.ComponentTypeWorker,
						PodTemplate: &corev1.PodTemplateSpec{Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
							Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"},
								Env: []corev1.EnvVar{{Name: commonconsts.EnvTopologyMountPath, Value: "/engine-data"}},
							}}},
						},
					},
				},
			}
			if test.apiVersion != "" {
				dcd.OwnerReferences = []metav1.OwnerReference{{APIVersion: test.apiVersion, Kind: "DynamoGraphDeployment", Name: "graph", Controller: new(test.controller)}}
			}
			original := dcd.DeepCopy()
			pod, err := renderStoredEnvironment(dcd, &configv1alpha1.OperatorConfiguration{}, RoleMain)
			require.NoError(t, err)
			values := envVarsToMap(pod.Containers[0].Env)
			if test.ownsPolicy {
				require.NotContains(t, values, commonconsts.EnvKvTransferPreferredWeight)
			} else {
				require.Equal(t, "0.75", values[commonconsts.EnvKvTransferPreferredWeight])
			}
			require.Equal(t, "/engine-data", values[commonconsts.EnvTopologyMountPath])
			require.Equal(t, original, dcd)
		})
	}
}

func TestEnvironmentLateGMSBindings(t *testing.T) {
	for _, test := range []struct {
		name string
		mode v1beta1.GPUMemoryServiceMode
		cold bool
	}{
		{name: "IntraPod", mode: v1beta1.GMSModeIntraPod},
		{name: "IntraPod-cold-failover", mode: v1beta1.GMSModeIntraPod, cold: true},
		{name: "InterPod", mode: v1beta1.GMSModeInterPod},
	} {
		t.Run(test.name, func(t *testing.T) {
			mode := test.mode
			t.Log("Reference a late GMS binding from graph defaults and try overriding an owned mode")
			component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				Experimental: &v1beta1.ExperimentalSpec{GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{Mode: mode}},
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"},
					Env:       []corev1.EnvVar{{Name: "NATS_SERVER", Value: "custom"}, {Name: "DYN_VLLM_GMS_SHADOW_MODE", Value: "false"}, {Name: gms.EnvSocketDir, Value: "/authored"}, {Name: "UNOWNED", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{Name: "settings", Key: "value"}}}},
					Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}},
				}}}},
			}
			if mode == v1beta1.GMSModeIntraPod {
				component.Experimental.GPUMemoryService.ExtraClientContainers = []string{"main"}
				if test.cold {
					component.Experimental.GPUMemoryService.ExtraClientContainers = nil
					component.Experimental.Failover = &v1beta1.FailoverSpec{Mode: mode}
				}
			} else {
				component.Experimental.Failover = &v1beta1.FailoverSpec{Mode: mode}
				component.PodTemplate.Spec.Containers[0].Env = append(component.PodTemplate.Spec.Containers[0].Env,
					corev1.EnvVar{Name: "ENGINE_ID", Value: "authored"}, corev1.EnvVar{Name: "FAILOVER_LOCK_PATH", Value: "authored"},
					corev1.EnvVar{Name: "DYN_SYSTEM_STARTING_HEALTH_STATUS", Value: "authored"}, corev1.EnvVar{Name: "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS", Value: "true"},
					corev1.EnvVar{Name: "USER_REFERENCE", Value: "$(ENGINE_ID)-$(GMS_SOCKET_DIR)"})
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
			require.Equal(t, findEnvVar(component.PodTemplate.Spec.Containers[0].Env, "UNOWNED"), findEnvVar(env, "UNOWNED"))
			require.Less(t, envIndex(t, env, gms.EnvSocketDir), envIndex(t, env, "WEIGHTS"))
			if mode == v1beta1.GMSModeInterPod {
				require.Equal(t, "true", envVarsToMap(env)["DYN_VLLM_GMS_SHADOW_MODE"])
				require.Equal(t, "/run/gms/shared", envVarsToMap(env)[gms.EnvSocketDir])
				require.Equal(t, "/run/gms/shared/failover.lock", envVarsToMap(env)["FAILOVER_LOCK_PATH"])
				require.Equal(t, "notready", envVarsToMap(env)["DYN_SYSTEM_STARTING_HEALTH_STATUS"])
				require.NotContains(t, envVarsToMap(env), "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS")
				require.NotEqual(t, "authored", envVarsToMap(env)["ENGINE_ID"])
				require.Less(t, envIndex(t, env, "ENGINE_ID"), envIndex(t, env, "USER_REFERENCE"))
				require.Less(t, envIndex(t, env, gms.EnvSocketDir), envIndex(t, env, "USER_REFERENCE"))
			} else {
				require.Equal(t, "/gms-intrapod-control", envVarsToMap(env)[gms.EnvSocketDir])
				if test.cold {
					require.Len(t, pod.Containers, 2)
					for i, engine := range pod.Containers {
						values := envVarsToMap(engine.Env)
						require.Equal(t, "/gms-intrapod-control", values[gms.EnvSocketDir])
						require.Equal(t, "true", values["DYN_VLLM_GMS_SHADOW_MODE"])
						require.Equal(t, fmt.Sprint(i), values["ENGINE_ID"])
						require.Equal(t, fmt.Sprint(5600+i), values["VLLM_NIXL_SIDE_CHANNEL_PORT"])
						require.Equal(t, fmt.Sprint(20080+i), values["DYN_VLLM_KV_EVENT_PORT"])
					}
				} else {
					require.Equal(t, "false", envVarsToMap(env)["DYN_VLLM_GMS_SHADOW_MODE"])
				}
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

func TestFrontendDeploymentConfigBindingsThroughGraphLowering(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		general, authored    []corev1.EnvVar
		wantValue, wantError string
		generalRewritten     bool
		wantOrder            []string
	}{
		{name: "general config retains its source", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "{\"frontend\":{\"port\":1234}}"}},
			authored: []corev1.EnvVar{{Name: "CFG", Value: "$(DYN_DEPLOYMENT_CONFIG)"}}, wantValue: "{\"frontend\":{\"port\":8000}}", generalRewritten: true, wantOrder: []string{commonconsts.DynamoDeploymentConfigEnvVar, "CFG"}},
		{name: "chained general config retains its source", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "{\"frontend\":{\"port\":1234}}"}, {Name: "DERIVED", Value: "$(DYN_DEPLOYMENT_CONFIG)"}},
			authored: []corev1.EnvVar{{Name: "FRONTEND_VAR", Value: "$(DERIVED)"}}, wantValue: "{\"frontend\":{\"port\":8000}}", generalRewritten: true, wantOrder: []string{commonconsts.DynamoDeploymentConfigEnvVar, "DERIVED", "FRONTEND_VAR"}},
		{name: "container overrides general", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "{\"frontend\":{\"port\":4321}}"}},
			authored: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "{\"frontend\":{\"port\":1234}}"}}, wantValue: "{\"frontend\":{\"port\":8000}}"},
		{name: "container shadows opaque general", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{Name: "config", Key: "value"}}}},
			authored: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "{\"frontend\":{\"port\":1234}}"}}, wantValue: "{\"frontend\":{\"port\":8000}}"},
		{name: "general alias to container", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "$(Z_CONFIG)"}},
			authored: []corev1.EnvVar{{Name: "Z_CONFIG", Value: "{\"frontend\":{\"port\":1234}}"}}, wantValue: "{\"frontend\":{\"port\":8000}}", generalRewritten: true},
		{name: "container alias to general", general: []corev1.EnvVar{{Name: "Z_CONFIG", Value: "{\"frontend\":{\"port\":1234}}"}},
			authored: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "$(Z_CONFIG)"}}, wantValue: "{\"frontend\":{\"port\":8000}}"},
		{name: "escaped JSON is not expanded twice", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "{\"frontend\":{\"port\":1234,\"description\":\"$$(ALIAS)\"}}"}},
			authored: []corev1.EnvVar{{Name: "ALIAS", Value: "must-not-be-used"}}, wantValue: "{\"frontend\":{\"port\":8000,\"description\":\"$$(ALIAS)\"}}", generalRewritten: true},
		{name: "opaque general", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Name: "config", Key: "json"}}}}},
		{name: "opaque container", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "invalid shadowed JSON"}},
			authored: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{Name: "config", Key: "json"}}}}},
		{name: "unknown alias", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "$(PROVIDER_CONFIG)"}}},
		{name: "null selected service", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "{\"gateway\":null}"}}, wantError: "service \"gateway\" must be an object"},
		{name: "null frontend fallback", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "{\"Frontend\":null}"}}, wantError: "service \"Frontend\" must be an object"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Prepare frontend source layers for direct graph rendering and child lowering")
			component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "gateway", ComponentType: v1beta1.ComponentTypeFrontend,
				PodTemplate: &corev1.PodTemplateSpec{Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "frontend:1.7.0", Env: tc.authored}}}},
			}
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
				Spec:        v1beta1.DynamoGraphDeploymentSpec{Env: tc.general, Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component}},
			}
			original := dgd.DeepCopy()
			config := &configv1alpha1.OperatorConfiguration{}

			t.Log("Unknown bindings stay intact; known winning frontend configs get the shared port")
			direct, directErr := GeneratePodSpecForComponent(&component, BackendFrameworkNoop, nil, dgd, RoleMain, 1, config, commonconsts.MultinodeDeploymentTypeGrove, "gateway", nil, staticContainerGPUCount(0))
			normalized, normalizeErr := NormalizeDynamoGraphDeploymentComponents(dgd, nil, nil, RollingUpdateContext{})
			if tc.wantError != "" {
				for _, err := range []error{directErr, normalizeErr} {
					require.ErrorContains(t, err, tc.wantError)
				}
			} else {
				require.NoError(t, directErr)
				require.NoError(t, normalizeErr)
				children, err := GenerateDynamoComponentsDeploymentsFromNormalized(dgd, normalized, RollingUpdateContext{})
				require.NoError(t, err)
				child := children["gateway"]
				child.OwnerReferences = []metav1.OwnerReference{{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeployment", Name: dgd.Name, Controller: new(true)}}
				childPod, err := renderStoredEnvironment(child, config, RoleMain)
				require.NoError(t, err)

				t.Log("Rewriting a general winner never promotes it into the authored source")
				if tc.generalRewritten {
					require.Equal(t, tc.authored, normalized["gateway"].PodTemplate.Spec.Containers[0].Env)
					require.Equal(t, tc.authored, child.Spec.PodTemplate.Spec.Containers[0].Env)
					require.Len(t, child.Spec.Env, len(tc.general))
					require.JSONEq(t, tc.wantValue, child.Spec.Env[0].Value)
					require.Equal(t, tc.general[1:], child.Spec.Env[1:])
				} else {
					require.Equal(t, tc.general, child.Spec.Env)
				}
				for _, pod := range []*corev1.PodSpec{direct, childPod} {
					binding := findEnvVar(pod.Containers[0].Env, commonconsts.DynamoDeploymentConfigEnvVar)
					require.NotNil(t, binding)
					if tc.wantValue != "" {
						require.JSONEq(t, tc.wantValue, binding.Value)
					} else {
						want := findEnvVar(tc.authored, commonconsts.DynamoDeploymentConfigEnvVar)
						if want == nil {
							want = findEnvVar(tc.general, commonconsts.DynamoDeploymentConfigEnvVar)
						}
						require.Equal(t, want, binding)
					}
					for i := 1; i < len(tc.wantOrder); i++ {
						require.Less(t, envIndex(t, pod.Containers[0].Env, tc.wantOrder[i-1]), envIndex(t, pod.Containers[0].Env, tc.wantOrder[i]))
					}
				}
			}
			require.Equal(t, original, dgd)
		})
	}
}

func TestGraphDeploymentConfigWithoutAuthoredMain(t *testing.T) {
	for _, tc := range []struct {
		name         string
		general      []corev1.EnvVar
		wantError    bool
		wantApply    bool
		authoredMain bool
	}{
		{name: "literal", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: `{"worker":{"ServiceArgs":{"workers":4,"resources":{"cpu":"2"}}}}`}}, wantApply: true},
		{name: "literal with authored main", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: `{"worker":{"ServiceArgs":{"workers":4,"resources":{"cpu":"2"}}}}`}}, wantApply: true, authoredMain: true},
		{name: "known alias", general: []corev1.EnvVar{{Name: "Z_CONFIG", Value: `{"worker":{"ServiceArgs":{"workers":4,"resources":{"cpu":"2"}}}}`}, {Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "$(Z_CONFIG)"}}, wantApply: true},
		{name: "unknown alias", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "$(PROVIDER_CONFIG)"}}},
		{name: "malformed JSON", general: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "not JSON"}}, wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Normalize graph defaults even though the renderer will create the main container later")
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
				Spec:        v1beta1.DynamoGraphDeploymentSpec{Env: tc.general, Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker}}},
			}
			if tc.authoredMain {
				dgd.Spec.Components[0].PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "engine:1.7.0"}}}}
			}
			original := dgd.DeepCopy()
			children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
			if tc.wantError {
				require.ErrorContains(t, err, "parse deployment config")
			} else {
				require.NoError(t, err)
				if tc.wantApply {
					require.EqualValues(t, 4, *children["worker"].Spec.Replicas)
					main := GetMainContainer(&children["worker"].Spec.DynamoComponentDeploymentSharedSpec)
					require.NotNil(t, main)
					require.Equal(t, resource.MustParse("2"), main.Resources.Requests[corev1.ResourceCPU])
				} else {
					require.Nil(t, children["worker"].Spec.Replicas)
				}
			}
			require.Equal(t, original, dgd)
		})
	}
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
	_, err = renderStoredEnvironment(children["frontend"], config, RoleMain)
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

func TestDeploymentConfigAliasSurvivesUnrelatedPendingBindings(t *testing.T) {
	for _, rankInGeneral := range []bool{false, true} {
		t.Run(fmt.Sprintf("rank-general=%t", rankInGeneral), func(t *testing.T) {
			t.Log("Known deployment config must survive an unrelated cycle removed by topology binding")
			component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				Multinode: &v1beta1.MultinodeSpec{NodeCount: 2},
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name: "main", Image: "engine:1.8.0", Command: []string{"python3", "-m", "dynamo.vllm"},
					Env: []corev1.EnvVar{{Name: "Z_CONFIG", Value: `{"worker":{"ServiceArgs":{"workers":4,"resources":{"cpu":"2"}}}}`},
						{Name: commonconsts.DynamoRankEnvVar, Value: "$(PENDING)"}},
				}}}},
			}
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
				Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component},
					Env: []corev1.EnvVar{{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: "$(Z_CONFIG)"}, {Name: "PENDING", Value: "$(DYNAMO_RANK)"}},
				},
			}
			if rankInGeneral {
				dgd.Spec.Env[1], component.PodTemplate.Spec.Containers[0].Env[1] =
					component.PodTemplate.Spec.Containers[0].Env[1], dgd.Spec.Env[1]
			}
			original := dgd.DeepCopy()
			t.Log("Input consumers cannot validate the partial environment before owned bindings")
			_, err := EffectiveDGDComponentTemplates(dgd, &component)
			require.NoError(t, err)
			children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
			require.NoError(t, err)
			require.EqualValues(t, 4, *children["worker"].Spec.Replicas)
			main := GetMainContainer(&children["worker"].Spec.DynamoComponentDeploymentSharedSpec)
			require.Equal(t, resource.MustParse("2"), main.Resources.Requests[corev1.ResourceCPU])

			t.Log("Final rendering applies owned topology bindings before validating the complete graph")
			pod, err := GeneratePodSpecForComponent(&component, BackendFrameworkVLLM, nil, dgd, RoleLeader, 2,
				&configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
			require.NoError(t, err)
			resolved := semanticenv.Replay(pod.Containers[0].Env)
			require.True(t, resolved[commonconsts.DynamoDeploymentConfigEnvVar].Known)
			require.Equal(t, "$(GROVE_PCSG_POD_INDEX)", envVarsToMap(pod.Containers[0].Env)[commonconsts.DynamoRankEnvVar])
			require.Less(t, envIndex(t, pod.Containers[0].Env, commonconsts.DynamoRankEnvVar), envIndex(t, pod.Containers[0].Env, "PENDING"))
			require.JSONEq(t, component.PodTemplate.Spec.Containers[0].Env[0].Value, resolved[commonconsts.DynamoDeploymentConfigEnvVar].Value)
			require.Equal(t, original, dgd)
		})
	}
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
				runtime := &component.PodTemplate.Spec.Containers[0]
				if runtimeSidecar {
					runtime = &component.PodTemplate.Spec.InitContainers[0]
				}
				runtime.Env = []corev1.EnvVar{{Name: commonconsts.EnvKvTransferPreferredWeight, Value: "authored"},
					{Name: commonconsts.EnvKvTransferDomain, Value: "authored"}, {Name: commonconsts.EnvKvTransferEnforcement, Value: "authored"},
					{Name: commonconsts.EnvTopologyEnabled, Value: "false"}, {Name: commonconsts.EnvTopologyMountPath, Value: "/authored"},
					{Name: "UNOWNED", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{Name: "settings", Key: "value"}}}}
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
				pod, err = renderStoredEnvironment(child, config, RoleMain)
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
					require.Equal(t, "zone", env[commonconsts.EnvKvTransferDomain])
					require.Equal(t, "required", env[commonconsts.EnvKvTransferEnforcement])
					require.Equal(t, "true", env[commonconsts.EnvTopologyEnabled])
					require.Equal(t, "/etc/dynamo/topology", env[commonconsts.EnvTopologyMountPath])
					require.Equal(t, "retained", env["GLOBAL"])
					require.NotNil(t, findEnvVar(runtime.Env, "UNOWNED").ValueFrom.SecretKeyRef)
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
					_, err = renderStoredEnvironment(child, config, RoleMain)
					require.ErrorContains(t, err, fmt.Sprintf("podTemplate.spec.containers[name=main].env[%d]", index))
				}
			})
		}
	}
}
