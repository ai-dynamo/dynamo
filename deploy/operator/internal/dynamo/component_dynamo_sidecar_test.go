// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"sort"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestNativeSidecarEnvironmentIgnoresOrigin(t *testing.T) {
	const currentOrigin = "1.6.0"
	const legacyOrigin = "1.5.0"

	for _, origin := range []string{currentOrigin, legacyOrigin, ""} {
		t.Run("origin="+origin, func(t *testing.T) {
			t.Log("Configure graph, engine, runtime, and frontend dependencies with duplicate overrides")
			graphEnv := []corev1.EnvVar{{Name: "Z_GRAPH", Value: "graph"}, {Name: "A_GRAPH", Value: "$(Z_GRAPH)"}}
			engineEnv := []corev1.EnvVar{
				{Name: "A_CACHE", Value: "$(VLLM_CACHE_ROOT)"},
				{Name: "VLLM_CACHE_ROOT", Value: "$(VLLM_CACHE_ROOT)/custom"},
			}
			runtimeEnv := []corev1.EnvVar{
				{Name: "A_SYSTEM", Value: "$(NATS_SERVER)/$(NATS_TLS_CA_CERT_PATH)"},
				{Name: "NATS_SERVER", Value: "nats://user:4222"},
				{Name: "Z_RUNTIME", Value: "runtime"},
				{Name: "A_RUNTIME", Value: "$(Z_RUNTIME)"},
				{Name: "A_POLICY", Value: "$(DYN_KV_TRANSFER_ENFORCEMENT)"},
				{Name: commonconsts.EnvKvTransferEnforcement, Value: "preferred"},
			}
			frontendEnv := []corev1.EnvVar{
				{Name: "A_SYSTEM", Value: "$(NATS_SERVER)/$(NATS_TLS_CA_CERT_PATH)"},
				{Name: "NATS_SERVER", Value: "nats://frontend:4222"},
			}
			dgd := &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "test"},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					BackendFramework: string(BackendFrameworkVLLM), Env: graphEnv,
					Experimental: &v1beta1.DynamoGraphDeploymentExperimentalSpec{KvTransferPolicy: &v1beta1.KvTransferPolicy{
						LabelKey: "topology.example/zone", Domain: "zone", Enforcement: "required",
					}},
				},
			}
			if origin != "" {
				dgd.Annotations = map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: origin}
			}
			component := v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker, FrontendSidecar: ptr.To("frontend"),
				CompilationCache: &v1beta1.CompilationCacheConfig{PVCName: "cache", MountPath: "/cache"},
				PodTemplate: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: currentOrigin}},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: commonconsts.MainContainerName, Image: "engine:latest", Env: engineEnv},
							{Name: "frontend", Image: "frontend:1.6.0", Env: frontendEnv},
						},
						InitContainers: []corev1.Container{{Name: "runtime", Image: "runtime:1.6.0", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways), Env: runtimeEnv}},
					},
				},
			}
			if origin == currentOrigin {
				component.PodTemplate.Annotations[commonconsts.KubeAnnotationDynamoOperatorOriginVersion] = legacyOrigin
			}
			dgd.Spec.Components = []v1beta1.DynamoComponentDeploymentSharedSpec{component}
			original := dgd.DeepCopy()
			config := &configv1alpha1.OperatorConfiguration{Infrastructure: configv1alpha1.InfrastructureConfiguration{
				NATSAddress: "nats://system:4222", NATSTLSCAPath: "/certs/ca.crt",
			}}

			t.Log("Render directly and through a materialized DCD across graph and component origins")
			pod, err := GeneratePodSpecForComponent(&component, BackendFrameworkVLLM, nil, dgd, RoleMain, 1, config, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
			require.NoError(t, err)
			children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
			require.NoError(t, err)
			require.Len(t, children, 1)
			pods := []*corev1.PodSpec{pod}
			for _, child := range children {
				childPod, err := GenerateBasePodSpec(&child.Spec.DynamoComponentDeploymentSharedSpec, BackendFrameworkVLLM, nil, dgd.Name, dgd.Namespace, RoleMain, 1, config, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
				require.NoError(t, err)
				pods = append(pods, childPod)
			}
			require.Equal(t, original, dgd)

			t.Log("Verify legacy environment merging, overrides, and cache isolation")
			for _, rendered := range pods {
				engine, runtime, frontend := rendered.Containers[0], rendered.InitContainers[0], rendered.Containers[1]
				require.Contains(t, engine.VolumeMounts, corev1.VolumeMount{Name: "cache", MountPath: "/cache"})
				require.NotContains(t, runtime.VolumeMounts, corev1.VolumeMount{Name: "cache", MountPath: "/cache"})
				require.NotContains(t, envVarsToMap(runtime.Env), "VLLM_CACHE_ROOT")
				require.Equal(t, "required", envVarsToMap(runtime.Env)[commonconsts.EnvKvTransferEnforcement])
				require.Equal(t, "nats://user:4222", envVarsToMap(runtime.Env)["NATS_SERVER"])
				require.Equal(t, "nats://frontend:4222", envVarsToMap(frontend.Env)["NATS_SERVER"])
				require.Equal(t, []corev1.EnvVar{
					{Name: "A_CACHE", Value: "$(VLLM_CACHE_ROOT)"},
					{Name: "A_GRAPH", Value: "$(Z_GRAPH)"},
					{Name: "VLLM_CACHE_ROOT", Value: "$(VLLM_CACHE_ROOT)/custom"},
					{Name: "Z_GRAPH", Value: "graph"},
					{Name: "VLLM_CACHE_ROOT", Value: "/cache"},
				}, engine.Env)
				for _, container := range []corev1.Container{runtime, frontend} {
					require.True(t, sort.SliceIsSorted(container.Env, func(i, j int) bool { return container.Env[i].Name < container.Env[j].Name }))
					require.Len(t, envVarsToMap(container.Env), len(container.Env), "legacy env must have unique names")
				}
			}
		})
	}
}

func TestNativeSidecarRendering(t *testing.T) {
	for _, componentType := range []v1beta1.ComponentType{v1beta1.ComponentTypeWorker, v1beta1.ComponentTypePrefill, v1beta1.ComponentTypeDecode} {
		t.Run(string(componentType), func(t *testing.T) {
			t.Log("Configure independent engine, native runtime, and regular frontend containers")
			engine := corev1.Container{
				Name: commonconsts.MainContainerName, Image: "vllm/vllm-openai:latest", Command: []string{"vllm-rs"}, Args: []string{"serve", "model"},
				Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}},
				Env:       []corev1.EnvVar{{Name: "GLOBAL", Value: "engine"}},
			}
			dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "test"}, Spec: v1beta1.DynamoGraphDeploymentSpec{
				Env:          []corev1.EnvVar{{Name: "GLOBAL", Value: "default"}, {Name: "SHARED", Value: "value"}},
				Experimental: &v1beta1.DynamoGraphDeploymentExperimentalSpec{KvTransferPolicy: &v1beta1.KvTransferPolicy{LabelKey: "topology.example/zone", Domain: "zone", Enforcement: "required"}},
			}}
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker", ComponentType: componentType, FrontendSidecar: ptr.To("frontend"),
				CompilationCache: &v1beta1.CompilationCacheConfig{PVCName: "cache", MountPath: "/cache"},
				PodTemplate: &corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: "container"}, Labels: map[string]string{commonconsts.KubeLabelDynamoWorkerHash: "abc123"}}, Spec: corev1.PodSpec{
					Containers: []corev1.Container{engine, {Name: "frontend", Image: "frontend:1.5.0", Env: []corev1.EnvVar{{Name: "ETCD_ENDPOINTS", Value: "frontend-etcd:2379"}}}},
					InitContainers: []corev1.Container{{Name: "setup", Image: "setup:latest"}, {
						Name: "runtime", Image: "runtime:1.5.0", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
						Env:          []corev1.EnvVar{{Name: "GLOBAL", Value: "runtime"}, {Name: "NATS_TLS_CA_CERT_PATH", Value: "/runtime/ca.crt"}, {Name: commonconsts.EnvKvTransferEnforcement, Value: "preferred"}},
						StartupProbe: &corev1.Probe{ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"true"}}}},
					}},
				}},
			}
			original := component.DeepCopy()
			config := &configv1alpha1.OperatorConfiguration{}
			config.Infrastructure.NATSAddress = "nats://nats:4222"
			config.Infrastructure.ETCDAddress = "http://etcd:2379"
			config.Infrastructure.TCPTLSCertPath = "/certs/tls.crt"
			config.Infrastructure.NATSTLSCAPath = "/certs/ca.crt"
			secrets := &nativeSidecarSecretsRetriever{}

			t.Log("Render with graph defaults and verify runtime ownership without mutating the source")
			pod, err := GeneratePodSpecForComponent(component, BackendFrameworkVLLM, secrets, dgd, RoleMain, 1, config, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(1))
			require.NoError(t, err)
			require.Equal(t, original, component)
			runtime := pod.InitContainers[1]
			require.Equal(t, original.PodTemplate.Spec.InitContainers[0], pod.InitContainers[0])
			require.Equal(t, "runtime", runtime.Name)
			require.Nil(t, runtime.Command)
			require.Nil(t, runtime.StartupProbe.HTTPGet)
			require.Equal(t, []string{"true"}, runtime.StartupProbe.Exec.Command)
			require.Equal(t, "/health", runtime.ReadinessProbe.HTTPGet.Path)
			require.Len(t, runtime.Ports, 1)
			require.Equal(t, "system", runtime.Ports[0].Name)
			env := envVarsToMap(runtime.Env)
			require.Equal(t, "runtime", env["CONTAINER_NAME"])
			require.Equal(t, "nats://nats:4222", env["NATS_SERVER"])
			require.Equal(t, "http://etcd:2379", env["ETCD_ENDPOINTS"])
			require.Equal(t, "/certs/tls.crt", env["DYN_TCP_TLS_CERT_PATH"])
			require.Equal(t, "/runtime/ca.crt", env["NATS_TLS_CA_CERT_PATH"])
			require.ElementsMatch(t, []string{"vllm/vllm-openai:latest", "frontend:1.5.0", "setup:latest", "runtime:1.5.0"}, secrets.images)
			require.Equal(t, []corev1.LocalObjectReference{{Name: "runtime-pull-secret"}}, pod.ImagePullSecrets)
			require.Equal(t, string(componentType), env[commonconsts.DynamoComponentEnvVar])
			require.Equal(t, "runtime", env["GLOBAL"])
			require.Equal(t, "value", env["SHARED"])
			require.Equal(t, "abc123", env[commonconsts.DynamoNamespaceWorkerSuffixEnvVar])
			require.Equal(t, "required", env[commonconsts.EnvKvTransferEnforcement])
			for _, key := range []string{"DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS", "DYN_HEALTH_CHECK_ENABLED", "NIXL_TELEMETRY_ENABLE", "DYN_FORWARDPASS_METRIC_PORT"} {
				require.NotContains(t, env, key)
			}
			require.Contains(t, runtime.VolumeMounts, TopologyLabelVolumeMount())
			require.Empty(t, runtime.Resources)

			t.Log("Engine resources and launch stay intact, with only global env and engine cache/shm mounts added")
			main := pod.Containers[0]
			require.Equal(t, engine.Command, main.Command)
			require.Equal(t, engine.Args, main.Args)
			require.Equal(t, engine.Resources, main.Resources)
			require.Nil(t, main.StartupProbe)
			require.Nil(t, main.LivenessProbe)
			require.Nil(t, main.ReadinessProbe)
			require.Empty(t, main.Ports)
			require.Equal(t, "engine", envVarsToMap(main.Env)["GLOBAL"])
			require.NotContains(t, envVarsToMap(main.Env), commonconsts.DynamoNamespaceEnvVar)
			for _, name := range []string{"NATS_SERVER", "ETCD_ENDPOINTS", "DYN_TCP_TLS_CERT_PATH", "NATS_TLS_CA_CERT_PATH"} {
				require.NotContains(t, envVarsToMap(main.Env), name)
			}
			require.NotContains(t, main.VolumeMounts, TopologyLabelVolumeMount())
			require.GreaterOrEqual(t, len(main.VolumeMounts), 2)
			require.Equal(t, "frontend", pod.Containers[1].Name)
			require.NotNil(t, pod.Containers[1].ReadinessProbe)
			require.Equal(t, "frontend", envVarsToMap(pod.Containers[1].Env)["CONTAINER_NAME"])
			frontendEnv := envVarsToMap(pod.Containers[1].Env)
			require.Equal(t, "nats://nats:4222", frontendEnv["NATS_SERVER"])
			require.Equal(t, "frontend-etcd:2379", frontendEnv["ETCD_ENDPOINTS"])
			require.Equal(t, "/certs/tls.crt", frontendEnv["DYN_TCP_TLS_CERT_PATH"])
			require.Equal(t, "/certs/ca.crt", frontendEnv["NATS_TLS_CA_CERT_PATH"])

			t.Log("Materialize DCDs and retain sidecar selection, global env, and topology")
			dgd.Spec.BackendFramework = string(BackendFrameworkVLLM)
			dgd.Spec.Components = []v1beta1.DynamoComponentDeploymentSharedSpec{*component}
			children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
			require.NoError(t, err)
			require.Len(t, children, 1)
			for _, child := range children {
				childRuntime := GetDynamoContainer(&child.Spec.DynamoComponentDeploymentSharedSpec)
				require.Equal(t, "value", envVarsToMap(childRuntime.Env)["SHARED"])
				require.Equal(t, "required", envVarsToMap(childRuntime.Env)[commonconsts.EnvKvTransferEnforcement])
				require.Contains(t, childRuntime.VolumeMounts, TopologyLabelVolumeMount())
			}

			t.Log("Runtime version resolution uses the sidecar image, independently of the engine tag")
			require.Equal(t, "1.5.0", resolvedRuntimeVersionForHash(component))
			component.PodTemplate.Spec.InitContainers[1].Image = "runtime:1.6.0"
			require.Equal(t, "1.6.0", resolvedRuntimeVersionForHash(component))
		})
	}
}

func TestGenerateBasePodSpecRejectsInvalidDynamoSidecar(t *testing.T) {
	cases := []struct {
		name          string
		componentType v1beta1.ComponentType
		restartPolicy *corev1.ContainerRestartPolicy
		multinode     *v1beta1.MultinodeSpec
		wantErrors    field.ErrorList
	}{
		{
			name: "frontend with runtime init container", componentType: v1beta1.ComponentTypeFrontend,
			restartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
			wantErrors:    field.ErrorList{field.Forbidden(field.NewPath("spec", "podTemplate", "spec", "initContainers").Index(0).Child("name"), "is supported only for worker, prefill, and decode components")},
		},
		{
			name: "runtime without restartPolicy Always", componentType: v1beta1.ComponentTypeWorker,
			wantErrors: field.ErrorList{field.Invalid(field.NewPath("spec", "podTemplate", "spec", "initContainers").Index(0).Child("restartPolicy"), corev1.ContainerRestartPolicy(""), `must be Always for component "worker" with a runtime init container`)},
		},
		{
			name: "multinode worker", componentType: v1beta1.ComponentTypeWorker,
			restartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways), multinode: &v1beta1.MultinodeSpec{NodeCount: 2},
			wantErrors: field.ErrorList{field.Forbidden(field.NewPath("spec", "multinode"), `is not currently supported for component "worker" with a runtime init container; support is planned for a future release`)},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Construct a stored component that has not passed current admission")
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker", ComponentType: tc.componentType, Multinode: tc.multinode,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers:     []corev1.Container{{Name: commonconsts.MainContainerName, Image: "engine:1.6.0"}},
					InitContainers: []corev1.Container{{Name: "runtime", Image: "runtime:1.6.0", RestartPolicy: tc.restartPolicy}},
				}},
			}
			original := component.DeepCopy()

			t.Log("Reject rendering with the same typed field errors as admission")
			pod, err := GenerateBasePodSpec(component, BackendFrameworkVLLM, nil, "test", "test", RoleMain, 1, &configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
			require.Equal(t, tc.wantErrors.ToAggregate(), err)
			require.Nil(t, pod)
			require.Equal(t, original, component)
		})
	}
}

func TestRuntimeContainerModeTransitions(t *testing.T) {
	for _, mode := range []string{"native", "renamed", "removed", "regular"} {
		t.Run(mode, func(t *testing.T) {
			t.Log("Build a native sidecar component with a versioned main image")
			dgd := betaDGDWithRuntimeVersion(t, "runtime:1.5.0", "")
			component := &dgd.Spec.Components[0]
			component.PodTemplate.Spec.InitContainers = []corev1.Container{{Name: "runtime", Image: "runtime:1.6.0", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways)}}
			nativeHash := mustComputeBetaDGDWorkersSpecHash(t, dgd)

			t.Log("Change the runtime location and derive mode from the resulting pod template")
			switch mode {
			case "renamed":
				component.PodTemplate.Spec.InitContainers[0].Name = "setup"
			case "removed":
				component.PodTemplate.Spec.InitContainers = nil
			case "regular":
				component.PodTemplate.Spec.InitContainers = nil
				component.PodTemplate.Spec.Containers = append(component.PodTemplate.Spec.Containers, corev1.Container{Name: "runtime", Image: "helper:latest"})
			}
			original := dgd.DeepCopy()
			pod, err := GeneratePodSpecForComponent(component, BackendFrameworkVLLM, nil, dgd, RoleMain, 1, &configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
			require.NoError(t, err)

			t.Log("Check injection target, resolved version, rollout hash, and input immutability")
			if mode == "native" {
				require.Equal(t, "runtime", GetDynamoContainer(component).Name)
				require.Equal(t, "1.6.0", resolvedRuntimeVersionForHash(component))
				require.Equal(t, "true", envVarsToMap(pod.InitContainers[0].Env)["DYN_SYSTEM_ENABLED"])
				require.NotContains(t, envVarsToMap(pod.Containers[0].Env), "DYN_SYSTEM_ENABLED")
				require.Equal(t, nativeHash, mustComputeBetaDGDWorkersSpecHash(t, dgd))
			} else {
				require.Nil(t, GetDynamoSidecar(component))
				require.Equal(t, commonconsts.MainContainerName, GetDynamoContainer(component).Name)
				require.Equal(t, "1.5.0", resolvedRuntimeVersionForHash(component))
				require.Equal(t, "true", envVarsToMap(pod.Containers[0].Env)["DYN_SYSTEM_ENABLED"])
				require.Equal(t, component.PodTemplate.Spec.InitContainers, pod.InitContainers)
				require.NotEqual(t, nativeHash, mustComputeBetaDGDWorkersSpecHash(t, dgd))
				if mode == "regular" {
					require.Equal(t, component.PodTemplate.Spec.Containers[1], pod.Containers[1])
				}
			}
			require.Equal(t, original, dgd)
		})
	}
}

// nativeSidecarSecretsRetriever records every image lookup and only grants the runtime image a secret.
type nativeSidecarSecretsRetriever struct{ images []string }

func (r *nativeSidecarSecretsRetriever) GetSecrets(namespace, image string) ([]string, error) {
	r.images = append(r.images, image)
	if image == "runtime:1.5.0" {
		return []string{"runtime-pull-secret"}, nil
	}
	return nil, nil
}

func TestCombinedWorkerCompilationCacheEnvironmentOrder(t *testing.T) {
	const currentOrigin = "1.6.0"

	for _, origin := range []string{currentOrigin, "1.5.0", ""} {
		t.Run("origin="+origin, func(t *testing.T) {
			t.Log("Configure a combined worker that references and overrides the cache default")
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				CompilationCache: &v1beta1.CompilationCacheConfig{PVCName: "cache", MountPath: "/cache"},
				PodTemplate: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: origin}},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{
						Name: commonconsts.MainContainerName, Image: "runtime:1.6.0",
						Env: []corev1.EnvVar{
							{Name: "A_CACHE", Value: "$(VLLM_CACHE_ROOT)"},
							{Name: "VLLM_CACHE_ROOT", Value: "$(VLLM_CACHE_ROOT)/custom"},
							{Name: "VLLM_USE_V1", Value: "1"},
						},
					}}},
				},
			}

			t.Log("Render the full backend path and check cache injection relative to user entries")
			pod, err := GenerateBasePodSpec(component, BackendFrameworkVLLM, nil, "test", "test", RoleMain, 1, &configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
			require.NoError(t, err)
			var cacheEnv []corev1.EnvVar
			for _, env := range pod.Containers[0].Env {
				if env.Name == "A_CACHE" || env.Name == "VLLM_CACHE_ROOT" || env.Name == "VLLM_USE_V1" {
					cacheEnv = append(cacheEnv, env)
				}
			}
			require.Equal(t, []corev1.EnvVar{
				{Name: "A_CACHE", Value: "$(VLLM_CACHE_ROOT)"},
				{Name: "VLLM_CACHE_ROOT", Value: "$(VLLM_CACHE_ROOT)/custom"},
				{Name: "VLLM_USE_V1", Value: "1"},
				{Name: "VLLM_CACHE_ROOT", Value: "/cache"},
			}, cacheEnv)
			require.Equal(t, corev1.EnvVar{Name: "VLLM_CACHE_ROOT", Value: "/cache"}, pod.Containers[0].Env[len(pod.Containers[0].Env)-1])
		})
	}
}
