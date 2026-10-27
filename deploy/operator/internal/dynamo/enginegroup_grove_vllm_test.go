// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"testing"

	config "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestEngineGroupRenderingSelectsAnImplementedBackend(t *testing.T) {
	for _, scenario := range []string{string(BackendFrameworkVLLM), "unspecified", string(BackendFrameworkTRTLLM), "invalid-native-profile"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("select the backend before interpreting engine-specific geometry")
			component := newVLLMComponentEngineGroupForTest()
			dgd := &api.DynamoGraphDeployment{Name: "deployment", Namespace: "test"}
			dgd.Spec.BackendFramework = string(BackendFrameworkVLLM)
			switch scenario {
			case "unspecified":
				dgd.Spec.BackendFramework = ""
			case string(BackendFrameworkTRTLLM):
				dgd.Spec.BackendFramework = string(BackendFrameworkTRTLLM)
				component.PodTemplate.Spec.Containers[0].Command = []string{"python3", "-m", "dynamo.trtllm"}
			case "invalid-native-profile":
				component.PodTemplate.Spec.Containers[0].Args = withoutTestVLLMProfileOptions(component.PodTemplate.Spec.Containers[0].Args, "--enable-elastic-ep")
			}
			dgd.Spec.Components = []api.DynamoComponentDeploymentSharedSpec{*component}
			before := dgd.DeepCopy()
			renderer := groveEngineGroupRenderer{operatorConfig: &config.OperatorConfiguration{}, runtimeConfig: &controller_common.RuntimeConfig{}}

			t.Log("render both workload and child declaration through their backend-neutral entrypoints")
			clique, world, workloadErr := renderer.Render(t.Context(), dgd, component, "", nil, nil, nil)
			group, declarationErr := RenderComponentEngineGroup(dgd, component, 0)
			if scenario == string(BackendFrameworkVLLM) {
				require.NoError(t, workloadErr)
				require.NoError(t, declarationErr)
				assert.Equal(t, group.Spec.Replicas, clique.Spec.Replicas)
				assert.Equal(t, []string{clique.Name}, world.CliqueNames)
				assert.Equal(t, group.Name, clique.Labels[consts.KubeLabelDynamoEngineGroup])
			} else {
				require.Error(t, workloadErr)
				require.Error(t, declarationErr)
				assert.Nil(t, clique)
				assert.Nil(t, group)
				assert.Empty(t, world.CliqueNames, "unsupported engines never fall back to static workload rendering")
			}
			assert.Equal(t, before, dgd)
		})
	}
}

func TestVLLMGroveRenderingPreservesCommonPodInfrastructure(t *testing.T) {
	t.Log("declare standard metadata, scheduling, storage, environment and a CPU-only sidecar")
	component := newVLLMComponentEngineGroupForTest()
	component.PodTemplate.Labels = map[string]string{"example.com/label": "kept", consts.KubeLabelDynamoWorkerHash: "revision"}
	component.PodTemplate.Annotations = map[string]string{"example.com/annotation": "kept"}
	component.PodTemplate.Spec.NodeSelector = map[string]string{"accelerator": "b200"}
	component.PodTemplate.Spec.SchedulerName = "custom-scheduler"
	component.PodTemplate.Spec.ImagePullSecrets = []corev1.LocalObjectReference{{Name: "registry"}}
	component.PodTemplate.Spec.Volumes = []corev1.Volume{{Name: "model", EmptyDir: &corev1.EmptyDirVolumeSource{}}}
	main := &component.PodTemplate.Spec.Containers[0]
	main.Env = append(main.Env, corev1.EnvVar{Name: "MODEL_VARIANT", Value: "sample"})
	main.ReadinessProbe = &corev1.Probe{TCPSocket: &corev1.TCPSocketAction{Port: intstr.FromInt32(8000)}}
	main.VolumeMounts = []corev1.VolumeMount{{Name: "model", MountPath: "/models"}}
	sidecar := corev1.Container{Name: "metrics", Image: "metrics:local", ReadinessProbe: &corev1.Probe{
		HTTPGet: &corev1.HTTPGetAction{Path: "/ready", Port: intstr.FromInt32(8081)},
	}}
	component.PodTemplate.Spec.Containers = append(component.PodTemplate.Spec.Containers, sidecar)
	dgd := &api.DynamoGraphDeployment{Name: "deployment", Namespace: "test",
		Spec: api.DynamoGraphDeploymentSpec{BackendFramework: string(BackendFrameworkVLLM), Components: []api.DynamoComponentDeploymentSharedSpec{*component}}}
	before := dgd.DeepCopy()
	operatorConfig := &config.OperatorConfiguration{}
	operatorConfig.Discovery.Backend = "kubernetes"
	operatorConfig.Infrastructure.NATSAddress = "nats://nats.test:4222"
	operatorConfig.Infrastructure.ETCDAddress = "http://etcd.test:2379"
	renderer := groveEngineGroupRenderer{operatorConfig: operatorConfig, runtimeConfig: &controller_common.RuntimeConfig{}}
	expected, err := generatePodSpecForComponent(component, BackendFrameworkVLLM, nil, dgd, RoleMain, 1, operatorConfig,
		consts.MultinodeDeploymentTypeGrove, component.ComponentName, nil, roleLaunchOwnedByOperator, nil, func() (int64, error) { return 4, nil })
	require.NoError(t, err)
	expectedMain := &expected.Containers[0]
	expectedMain.LivenessProbe, expectedMain.ReadinessProbe, expectedMain.StartupProbe = main.LivenessProbe, main.ReadinessProbe, main.StartupProbe
	expected.Containers[0].Env = append([]corev1.EnvVar{
		{Name: consts.DynamoRankEnvVar, Value: "$(GROVE_PCLQ_POD_INDEX)"},
		{Name: consts.DynamoLeaderAddressEnvVar, Value: "$(GROVE_PCLQ_NAME)-0.$(GROVE_HEADLESS_SERVICE)"},
	}, expectedMain.Env...)

	t.Log("reuse shared Pod defaults, preserve native command and arguments, and add topology aliases")
	clique, world, err := renderer.Render(t.Context(), dgd, component, "", nil, nil, nil)
	require.NoError(t, err)
	assert.Equal(t, *expected, clique.Spec.PodSpec)
	assert.Contains(t, clique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: "NATS_SERVER", Value: operatorConfig.Infrastructure.NATSAddress})
	assert.Contains(t, clique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: "ETCD_ENDPOINTS", Value: operatorConfig.Infrastructure.ETCDAddress})
	assert.Equal(t, "deployment-k8s-service-discovery", clique.Spec.PodSpec.ServiceAccountName)
	assert.Equal(t, "kept", clique.Labels["example.com/label"])
	assert.Equal(t, "revision", clique.Labels[consts.KubeLabelDynamoWorkerHash])
	assert.Equal(t, "kept", clique.Annotations["example.com/annotation"])
	assert.Equal(t, component.PodTemplate.Spec.NodeSelector, clique.Spec.PodSpec.NodeSelector)
	assert.Equal(t, "custom-scheduler", clique.Spec.PodSpec.SchedulerName)
	assert.Equal(t, component.PodTemplate.Spec.ImagePullSecrets, clique.Spec.PodSpec.ImagePullSecrets)
	assert.Contains(t, clique.Spec.PodSpec.Volumes, component.PodTemplate.Spec.Volumes[0])
	assert.Equal(t, []string{clique.Name}, world.CliqueNames)
	assert.Equal(t, int32(8), *clique.Spec.MinAvailable)
	for _, container := range clique.Spec.PodSpec.Containers {
		if container.Name == consts.MainContainerName {
			assert.Contains(t, container.Env, main.Env[0])
			assert.Contains(t, container.VolumeMounts, main.VolumeMounts[0])
			assert.Equal(t, main.Resources.Limits, container.Resources.Limits)
			assert.Equal(t, main.Command, container.Command)
			assert.Equal(t, main.Args, container.Args)
			assert.Contains(t, container.Args, "--elastic-ep-auto-bootstrap")
			assert.Nil(t, container.LivenessProbe)
		} else if container.Name == sidecar.Name {
			assert.Equal(t, sidecar, container)
		}
	}
	assert.Equal(t, before, dgd)
}
