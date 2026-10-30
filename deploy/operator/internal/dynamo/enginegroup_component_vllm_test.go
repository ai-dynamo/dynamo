/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	config "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestVLLMEngineGroupDGDRendering(t *testing.T) {
	t.Log("declare one native DP8 world, each member backed by one four-GPU Pod")
	component := newVLLMComponentEngineGroupForTest()
	dgd := &api.DynamoGraphDeployment{
		Name: "deployment", Namespace: "test", UID: "dgd-uid",
		Spec: api.DynamoGraphDeploymentSpec{BackendFramework: string(BackendFrameworkVLLM), Components: []api.DynamoComponentDeploymentSharedSpec{*component}},
	}
	before := dgd.DeepCopy()
	source, _, err := ResolveVLLMComponentEngineGroup(component, nil)
	require.NoError(t, err)
	assert.Equal(t, int32(8), source.InitialReplicas)
	assert.Equal(t, int64(4), source.MainContainerGPUs)

	t.Log("render one PCSG world containing one elastic member clique, never a static leader/worker split")
	pcs, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, &config.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, nil, nil, nil, nil, false, nil)
	require.NoError(t, err)
	require.Len(t, pcs.Spec.Template.Cliques, 1)
	require.Len(t, pcs.Spec.Template.PodCliqueScalingGroupConfigs, 1)
	clique := pcs.Spec.Template.Cliques[0]
	assert.Equal(t, int32(8), clique.Spec.Replicas)
	assert.Equal(t, int32(8), *clique.Spec.MinAvailable)
	assert.Equal(t, "deployment-worker-0", clique.Labels[consts.KubeLabelDynamoEngineGroup])
	assert.Equal(t, consts.KubeLabelDynamoScaleRepresentativeYes, clique.Labels[consts.KubeLabelDynamoScaleRepresentative])
	world := pcs.Spec.Template.PodCliqueScalingGroupConfigs[0]
	assert.Equal(t, int32(1), *world.Replicas)
	assert.Equal(t, []string{"worker"}, world.CliqueNames)
	assert.Equal(t, grove.OnDeleteStrategy, pcs.Spec.UpdateStrategy.Type)
	assert.True(t, pcs.Spec.Template.HeadlessServiceConfig.PublishNotReadyAddresses)

	t.Log("launch all allocations from one bootstrap template with readiness distinct from native inference")
	assert.Equal(t, corev1.RestartPolicyNever, clique.Spec.PodSpec.RestartPolicy)
	main := clique.Spec.PodSpec.Containers[0]
	assert.Equal(t, component.PodTemplate.Spec.Containers[0].Command, main.Command)
	assert.Contains(t, main.Args, "--elastic-ep-auto-bootstrap")
	assert.Contains(t, main.Args, "--data-parallel-rank=$(DYNAMO_RANK)")
	assert.Nil(t, main.LivenessProbe)
	assert.Nil(t, main.ReadinessProbe)
	for _, volume := range clique.Spec.PodSpec.Volumes {
		assert.Nil(t, volume.DownwardAPI, "native bootstrap needs no projected command assignment")
	}
	assert.Equal(t, before, dgd)
}

func TestVLLMComponentEngineGroupIdentity(t *testing.T) {
	t.Log("resolve the original creation seed and immutable native Pod declaration")
	component := newVLLMComponentEngineGroupForTest()
	_, initial, err := ResolveVLLMComponentEngineGroup(component, nil)
	require.NoError(t, err)

	t.Log("operator metadata and a different startup size do not redefine fixed geometry")
	component.PodTemplate.Labels = map[string]string{consts.KubeLabelDynamoWorkerHash: "operator-label"}
	component.EngineGroup.InitialSize = 10
	component.PodTemplate.Spec.Containers[0].Args = append(withoutTestVLLMProfileOptions(component.PodTemplate.Spec.Containers[0].Args, "--elastic-ep-initial-dp-size"), "--elastic-ep-initial-dp-size=10")
	_, resized, err := ResolveVLLMComponentEngineGroup(component, nil)
	require.NoError(t, err)
	assert.Equal(t, initial.Geometry.Fingerprint, resized.Geometry.Fingerprint)
	assert.Equal(t, int32(10), resized.InitialReplicas)

	t.Log("a changed native image is a different workload, not a live resize")
	component.PodTemplate.Spec.Containers[0].Image = "vllm:another-build"
	_, changed, err := ResolveVLLMComponentEngineGroup(component, nil)
	require.NoError(t, err)
	assert.NotEqual(t, initial.Geometry.Fingerprint, changed.Geometry.Fingerprint)
}

func TestVLLMComponentEngineGroupRejectsUnsupportedDeclarations(t *testing.T) {
	for _, scenario := range []string{
		"multiple-worlds", "static-multinode", "legacy-scaler", "missing-main", "missing-command", "missing-image",
		"automatic-restart", "host-network", "opaque-environment", "sidecar-gpus", "hidden-gpus",
		"fractional-gpus", "request-mismatch", "mismatched-initial-size", "policy-below-initial", "policy-above-reservation",
		"missing-multiprocessing", "missing-drain-mode", "missing-frontend-mode", "missing-worker-identity", "missing-restart-policy",
	} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("introduce one unsupported property into a valid native declaration")
			component := newVLLMComponentEngineGroupForTest()
			main := &component.PodTemplate.Spec.Containers[0]
			switch scenario {
			case "multiple-worlds":
				component.Replicas = new(int32(2))
			case "static-multinode":
				component.Multinode = &api.MultinodeSpec{NodeCount: 2}
			case "legacy-scaler":
				component.ScalingAdapter = &api.ScalingAdapter{}
			case "missing-main":
				main.Name = "not-the-main-container"
			case "missing-command":
				main.Command = nil
			case "missing-image":
				main.Image = ""
			case "automatic-restart":
				component.PodTemplate.Spec.RestartPolicy = corev1.RestartPolicyAlways
			case "host-network":
				component.PodTemplate.Spec.HostNetwork = true
			case "opaque-environment":
				main.EnvFrom = []corev1.EnvFromSource{{Prefix: "unknown"}}
			case "sidecar-gpus":
				component.PodTemplate.Spec.Containers = append(component.PodTemplate.Spec.Containers, corev1.Container{Name: "sidecar", Resources: main.Resources})
			case "hidden-gpus":
				main.Env = []corev1.EnvVar{{Name: "CUDA_VISIBLE_DEVICES", Value: "0"}}
			case "fractional-gpus":
				main.Resources.Limits[consts.KubeResourceGPUNvidia] = resource.MustParse("0.5")
			case "request-mismatch":
				main.Resources.Requests = corev1.ResourceList{consts.KubeResourceGPUNvidia: resource.MustParse("2")}
			case "mismatched-initial-size":
				component.EngineGroup.InitialSize = 3
			case "policy-below-initial":
				component.EngineGroup.Policy = &api.ComponentEngineGroupPolicy{MinSize: new(int32(2))}
			case "policy-above-reservation":
				component.EngineGroup.Policy = &api.ComponentEngineGroupPolicy{MaxSize: new(int32(64))}
			case "missing-multiprocessing":
				main.Env = append(main.Env[:0], main.Env[1:]...)
			case "missing-frontend-mode":
				main.Env[1].Value = "1"
			case "missing-drain-mode":
				main.Env[2].Value = "0"
			case "missing-worker-identity":
				main.Command = []string{"python3", "-m", "dynamo.vllm"}
				main.Env[4] = corev1.EnvVar{Name: "DYNAMO_POD_UID", Value: "stale"}
			case "missing-restart-policy":
				component.PodTemplate.Spec.RestartPolicy = ""
			}
			before := component.DeepCopy()

			t.Log("reject before workload creation and preserve the caller's declaration")
			_, _, err := ResolveVLLMComponentEngineGroup(component, nil)
			require.Error(t, err)
			assert.Equal(t, before, component)
		})
	}
}

func TestVLLMComponentEngineGroupRejectsUnsupportedControlEndpoints(t *testing.T) {
	for _, scenario := range []struct {
		name string
		args []string
	}{
		{name: "loopback-api", args: []string{"--host=127.0.0.1"}},
		{name: "invalid-api-port", args: []string{"--port=70000"}},
		{name: "duplicate-api-port", args: []string{"--port=8000", "--port=8001"}},
		{name: "unix-socket", args: []string{"--uds=/tmp/engine.sock"}},
		{name: "native-tls", args: []string{"--ssl-certfile=/tmp/cert.pem"}},
		{name: "port-collision", args: []string{"--port=13445", "--data-parallel-rpc-port=13445"}},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Log("declare a native control endpoint incompatible with this integration")
			component := newVLLMComponentEngineGroupForTest()
			main := &component.PodTemplate.Spec.Containers[0]
			main.Args = append(main.Args, scenario.args...)
			before := component.DeepCopy()

			t.Log("reject before workload creation without changing the caller's endpoint")
			_, _, err := ResolveVLLMComponentEngineGroup(component, nil)
			require.Error(t, err)
			assert.Equal(t, before, component)
		})
	}
}

func TestVLLMNativeValidationDoesNotRequireGrove(t *testing.T) {
	t.Log("declare native geometry without the Grove retention mechanism")
	component := newVLLMComponentEngineGroupForTest()
	component.PodTemplate.Spec.ReadinessGates = nil
	before := component.DeepCopy()

	t.Log("resolve the native profile but refuse the same declaration for readiness-steered Grove release")
	_, _, err := ResolveVLLMComponentEngineGroup(component, nil)
	require.NoError(t, err)
	require.ErrorContains(t, RequireGroveEngineGroupRetentionGate(component.PodTemplate), "exactly one")
	assert.Equal(t, before, component)
}

func TestVLLMWorkerInheritedEnvironmentMatchesRendering(t *testing.T) {
	t.Log("resolve the same inherited environment before profiling and physical rendering")
	component := newVLLMComponentEngineGroupForTest()
	component.PodTemplate.Spec.Containers[0].Command = []string{"python3", "-m", "dynamo.vllm", "--model=model"}
	parentEnv := []corev1.EnvVar{{Name: "DYN_SYSTEM_PORT", Value: "19090"}}
	before := component.DeepCopy()
	source, profile, err := ResolveVLLMComponentEngineGroup(component, parentEnv)
	require.NoError(t, err)
	port, _, err := ResolveVLLMControlPorts(source)
	require.NoError(t, err)
	assert.Equal(t, int32(19090), port)
	assert.Equal(t, before, component)

	t.Log("render one world and retain the exact profile and actual control listener configuration")
	dgd := &api.DynamoGraphDeployment{Name: "deployment", Namespace: "test", UID: "dgd",
		Spec: api.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Env: parentEnv, Components: []api.DynamoComponentDeploymentSharedSpec{*component}}}
	pcs, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, &config.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, nil, nil, nil, nil, false, nil)
	require.NoError(t, err)
	assert.Contains(t, pcs.Spec.Template.Cliques[0].Spec.PodSpec.Containers[0].Env, parentEnv[0])
	group, err := RenderComponentEngineGroup(dgd, component, 0)
	require.NoError(t, err)
	assert.Equal(t, profile.Geometry.Fingerprint, group.Annotations[consts.KubeAnnotationDynamoEngineGroupProfile])

	t.Log("inherited GPU visibility cannot bypass allocation geometry validation")
	_, _, err = ResolveVLLMComponentEngineGroup(component, []corev1.EnvVar{{Name: "CUDA_VISIBLE_DEVICES", Value: "0"}})
	require.Error(t, err)

	t.Log("parent environment cannot replace the provider-neutral topology bindings")
	for _, name := range []string{consts.DynamoRankEnvVar, consts.DynamoLeaderAddressEnvVar} {
		_, _, err := ResolveVLLMComponentEngineGroup(component, []corev1.EnvVar{{Name: name, Value: "wrong"}})
		require.ErrorContains(t, err, "operator-owned topology")
	}
}

func newVLLMComponentEngineGroupForTest() *api.DynamoComponentDeploymentSharedSpec {
	source := newTestVLLMProfileGeometrySource()
	source.Args = append(withoutTestVLLMProfileOptions(source.Args, "--data-parallel-size", "--data-parallel-rank"),
		"--elastic-ep-auto-bootstrap", "--elastic-ep-initial-dp-size=8",
		"--data-parallel-rank="+engineGroupOrdinalRef, "--data-parallel-address="+engineGroupRendezvousRef)
	return &api.DynamoComponentDeploymentSharedSpec{
		ComponentName: "worker", ComponentType: api.ComponentTypeWorker, Replicas: new(int32(1)),
		RuntimeVersionOverride: "1.5.0",
		EngineGroup:            &api.ComponentEngineGroupSpec{InitialSize: 8},
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			RestartPolicy:  corev1.RestartPolicyNever,
			ReadinessGates: []corev1.PodReadinessGate{{ConditionType: corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate)}},
			Containers: []corev1.Container{{
				Name: consts.MainContainerName, Image: "vllm:native", Command: source.Command, Args: source.Args,
				Env: []corev1.EnvVar{
					{Name: "VLLM_ENABLE_V1_MULTIPROCESSING", Value: "1"},
					{Name: "VLLM_USE_RUST_FRONTEND", Value: "0"},
					{Name: "VLLM_ELASTIC_EP_DRAIN_REQUESTS", Value: "1"},
					{Name: "DYNAMO_ENGINE_GROUP", Value: "1"},
					{Name: "DYNAMO_POD_UID", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.uid"}}},
				},
				Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{consts.KubeResourceGPUNvidia: resource.MustParse("4")}},
			}}}},
	}
}
