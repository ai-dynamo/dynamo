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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestNativeSidecarRendering(t *testing.T) {
	for _, componentType := range []v1beta1.ComponentType{v1beta1.ComponentTypeWorker, v1beta1.ComponentTypePrefill, v1beta1.ComponentTypeDecode} {
		t.Run(string(componentType), func(t *testing.T) {
			t.Log("Configure independent engine, native runtime, and regular frontend containers")
			engine := corev1.Container{
				Name: "main", Image: "vllm/vllm-openai:latest", Command: []string{"vllm-rs"}, Args: []string{"serve", "model"},
				Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}},
				Env:       []corev1.EnvVar{{Name: "GLOBAL", Value: "engine"}},
			}
			dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "test"}, Spec: v1beta1.DynamoGraphDeploymentSpec{
				Env:          []corev1.EnvVar{{Name: "GLOBAL", Value: "default"}, {Name: "SHARED", Value: "value"}},
				Experimental: &v1beta1.DynamoGraphDeploymentExperimentalSpec{KvTransferPolicy: &v1beta1.KvTransferPolicy{LabelKey: "topology.example/zone", Domain: "zone", Enforcement: "required"}},
			}}
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker", ComponentType: componentType, DynamoSidecar: ptr.To("runtime"), FrontendSidecar: ptr.To("frontend"),
				CompilationCache: &v1beta1.CompilationCacheConfig{PVCName: "cache", MountPath: "/cache"},
				PodTemplate: &corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: "container"}, Labels: map[string]string{commonconsts.KubeLabelDynamoWorkerHash: "abc123"}}, Spec: corev1.PodSpec{
					Containers: []corev1.Container{engine, {Name: "frontend", Image: "frontend:1.5.0"}},
					InitContainers: []corev1.Container{{Name: "setup", Image: "setup:latest"}, {
						Name: "runtime", Image: "runtime:1.5.0", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
						Env:          []corev1.EnvVar{{Name: "GLOBAL", Value: "runtime"}, {Name: commonconsts.EnvKvTransferEnforcement, Value: "preferred"}},
						StartupProbe: &corev1.Probe{ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"true"}}}},
					}},
				}},
			}
			original := component.DeepCopy()
			config := &configv1alpha1.OperatorConfiguration{}
			config.Infrastructure.NATSAddress = "nats://nats:4222"
			config.Infrastructure.ETCDAddress = "http://etcd:2379"
			config.Infrastructure.TCPTLSCertPath = "/certs/tls.crt"
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
			require.NotContains(t, main.VolumeMounts, TopologyLabelVolumeMount())
			require.GreaterOrEqual(t, len(main.VolumeMounts), 2)
			require.Equal(t, "frontend", pod.Containers[1].Name)
			require.NotNil(t, pod.Containers[1].ReadinessProbe)
			require.Equal(t, "frontend", envVarsToMap(pod.Containers[1].Env)["CONTAINER_NAME"])

			t.Log("Materialize DCDs and retain sidecar selection, global env, and topology")
			dgd.Spec.BackendFramework = "vllm"
			dgd.Spec.Components = []v1beta1.DynamoComponentDeploymentSharedSpec{*component}
			children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
			require.NoError(t, err)
			require.Len(t, children, 1)
			for _, child := range children {
				require.Equal(t, component.DynamoSidecar, child.Spec.DynamoSidecar)
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

// nativeSidecarSecretsRetriever records every image lookup and only grants the runtime image a secret.
type nativeSidecarSecretsRetriever struct{ images []string }

func (r *nativeSidecarSecretsRetriever) GetSecrets(namespace, image string) ([]string, error) {
	r.images = append(r.images, image)
	if image == "runtime:1.5.0" {
		return []string{"runtime-pull-secret"}, nil
	}
	return nil, nil
}
