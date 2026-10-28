/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"os"
	"testing"

	config "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/yaml"
)

func TestResolveSGLangComponentEngineGroup(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*api.DynamoComponentDeploymentSharedSpec)
		want   string
	}{
		{name: "supported one-world growth"},
		{name: "native rendezvous uses Pod networking", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) { c.PodTemplate.Spec.HostNetwork = true }, want: "Pod networking"},
		{name: "internal routing cannot advertise per-rank KV ownership", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Args = append(c.PodTemplate.Spec.Containers[0].Args, "--kv-events-config", `{"publisher":"zmq"}`)
		}, want: "per-rank KV routing"},
		{name: "disaggregation requires a rank-routing contract", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Args = append(c.PodTemplate.Spec.Containers[0].Args, "--disaggregation-mode", "decode")
		}, want: "aggregated inference"},
		{name: "a creation seed cannot derive from zero", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) { c.EngineGroup.InitialSize = 0 }, want: "initialSize must be positive"},
		{name: "fractional GPUs do not prove an allocation", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Resources.Limits["nvidia.com/gpu"] = resource.MustParse("900m")
		}, want: "whole-GPU"},
		{name: "independent world count is not capacity", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) { c.Replicas = new(int32(2)) }, want: "one independent worker world"},
		{name: "legacy multinode is not a capacity seed", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) { c.Multinode = &api.MultinodeSpec{NodeCount: 2} }, want: "does not support multinode"},
		{name: "packed width must divide the declared EP sizes", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Resources.Limits["nvidia.com/gpu"] = resource.MustParse("4")
		}, want: "multiples"},
		{name: "packed EP8 to EP12 uses two initial four-GPU allocations", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Resources.Limits["nvidia.com/gpu"] = resource.MustParse("4")
			c.PodTemplate.Spec.Containers[0].Args = setTestSGLangOption(c.PodTemplate.Spec.Containers[0].Args, sglangElasticInitialSizeOption, "8")
			c.PodTemplate.Spec.Containers[0].Args = setTestSGLangOption(c.PodTemplate.Spec.Containers[0].Args, sglangMaximumEPSizeOption, "12")
		}},
		{name: "engine initial geometry must match seed", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) { c.EngineGroup.InitialSize = 3 }, want: "initial"},
		{name: "unfenced restart is rejected", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.RestartPolicy = corev1.RestartPolicyAlways
		}, want: "restartPolicy must explicitly be Never"},
		{name: "sidecar GPUs are rejected", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers = append(c.PodTemplate.Spec.Containers, corev1.Container{Name: "other", Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}})
		}, want: "exclusively"},
		{name: "policy cannot enable unsupported shrink", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) { c.EngineGroup.Policy.MinSize = new(int32(1)) }, want: "policy.minSize"},
		{name: "policy cannot exceed engine maximum", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) { c.EngineGroup.Policy.MaxSize = new(int32(4)) }, want: "policy.maxSize"},
		{name: "missing allocation index is not injected", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Args = setTestSGLangOption(c.PodTemplate.Spec.Containers[0].Args, sglangAllocationIndexOption, "0")
		}, want: "explicitly declare"},
		{name: "caller cannot forge an allocation incarnation", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Env[0] = corev1.EnvVar{Name: "DYNAMO_ALLOCATION_UID", Value: "stale-uid"}
		}, want: "Downward API"},
		{name: "omitted control port uses the shared worker default", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Env = c.PodTemplate.Spec.Containers[0].Env[:1]
		}},
		{name: "restart default is not injected", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.RestartPolicy = ""
		}, want: "restartPolicy must explicitly be Never"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("declare one immutable, width-one SGLang world")
			component := testEngineGroupComponent()
			if tc.mutate != nil {
				tc.mutate(component)
			}
			before := component.DeepCopy()

			t.Log("resolve supported geometry without changing the declaration")
			profile, err := ResolveSGLangComponentEngineGroup(component, nil)
			assert.Equal(t, before, component)
			if tc.want != "" {
				require.ErrorContains(t, err, tc.want)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, int32(2), profile.InitialReplicas)
			assert.Equal(t, int32(3), profile.MaximumReplicas)
		})
	}
}

func TestGenerateGrovePodCliqueSetEngineGroup(t *testing.T) {
	t.Log("declare two initial allocations in one SGLang Engine Group")
	dgd := &api.DynamoGraphDeployment{
		Name: "graph", Namespace: "default",
		Spec: api.DynamoGraphDeploymentSpec{BackendFramework: "sglang", Components: []api.DynamoComponentDeploymentSharedSpec{*testEngineGroupComponent()}},
	}
	before := dgd.DeepCopy()

	t.Log("render one PCSG world with one template-invariant member clique")
	pcs, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, &config.OperatorConfiguration{}, &common.RuntimeConfig{}, nil, &mockSecretsRetriever{}, nil, nil, false, nil)
	require.NoError(t, err)
	assert.Equal(t, before, dgd)
	require.Len(t, pcs.Spec.Template.Cliques, 1)
	require.Len(t, pcs.Spec.Template.PodCliqueScalingGroupConfigs, 1)
	clique := pcs.Spec.Template.Cliques[0]
	world := pcs.Spec.Template.PodCliqueScalingGroupConfigs[0]
	assert.Equal(t, int32(2), clique.Spec.Replicas)
	assert.Equal(t, int32(2), *clique.Spec.MinAvailable)
	assert.Equal(t, int32(1), *world.Replicas)
	assert.Equal(t, []string{clique.Name}, world.CliqueNames)
	assert.Equal(t, "OnDelete", string(pcs.Spec.UpdateStrategy.Type))
	assert.Equal(t, EngineGroupNameForComponent("graph", "Worker", 0), clique.Labels[consts.KubeLabelDynamoEngineGroup])
	assert.Equal(t, consts.KubeLabelDynamoScaleRepresentativeYes, clique.Labels[consts.KubeLabelDynamoScaleRepresentative])

	t.Log("preserve native command and arguments while retaining shared worker defaults")
	main := clique.Spec.PodSpec.Containers[0]
	assert.Equal(t, dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Command, main.Command)
	assert.Equal(t, dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Args, main.Args)
	assert.Equal(t, []corev1.EnvVar{
		{Name: consts.DynamoRankEnvVar, Value: "$(GROVE_PCLQ_POD_INDEX)"},
		{Name: consts.DynamoLeaderAddressEnvVar, Value: "$(GROVE_PCLQ_NAME)-0.$(GROVE_HEADLESS_SERVICE)"},
	}, main.Env[:2])
	assert.Contains(t, main.Env, corev1.EnvVar{Name: "DYN_SYSTEM_PORT", Value: "9090"})
	assert.Contains(t, main.Env, corev1.EnvVar{Name: "DYN_COMPONENT", Value: "worker"})
	assert.Equal(t, corev1.RestartPolicyNever, clique.Spec.PodSpec.RestartPolicy)
	assert.Nil(t, main.ReadinessProbe)
	assert.Nil(t, main.LivenessProbe)
	assert.Nil(t, main.StartupProbe)

	t.Log("validate the generated shape against the pinned Grove schema and CEL")
	newGrovePodCliqueSetRequestValidator(t).validate(t, pcs, nil)
}

func TestSGLangEngineGroupPreservesExplicitPodSpec(t *testing.T) {
	t.Log("declare process probes, storage, service account and component environment overrides")
	component := testEngineGroupComponent()
	main := &component.PodTemplate.Spec.Containers[0]
	probe := &corev1.Probe{Exec: &corev1.ExecAction{Command: []string{"test", "-d", "/proc/1"}}}
	main.ReadinessProbe, main.LivenessProbe, main.StartupProbe = probe.DeepCopy(), probe.DeepCopy(), probe.DeepCopy()
	main.VolumeMounts = []corev1.VolumeMount{{Name: "shm", MountPath: "/dev/shm"}}
	main.Env = append(main.Env, corev1.EnvVar{Name: "DECLARED_SETTING", Value: "local"})
	component.PodTemplate.Spec.ServiceAccountName = "declared-discovery-account"
	component.PodTemplate.Spec.Volumes = []corev1.Volume{{Name: "shm", EmptyDir: &corev1.EmptyDirVolumeSource{Medium: corev1.StorageMediumMemory}}}
	component.PodTemplate.Spec.Containers = append(component.PodTemplate.Spec.Containers, corev1.Container{Name: "metrics", Image: "metrics:test"})
	parentEnv := []corev1.EnvVar{{Name: "PARENT_ONLY", Value: "kept"}, {Name: "DYN_SYSTEM_PORT", Value: "1"}}
	dgd := &api.DynamoGraphDeployment{Name: "graph", Namespace: "default",
		Spec: api.DynamoGraphDeploymentSpec{BackendFramework: "sglang", Env: parentEnv, Components: []api.DynamoComponentDeploymentSharedSpec{*component}}}
	before := dgd.DeepCopy()
	operatorConfig := &config.OperatorConfiguration{}
	operatorConfig.Discovery.Backend = "kubernetes"
	operatorConfig.Infrastructure.NATSAddress = "nats://nats.test:4222"
	operatorConfig.Infrastructure.ETCDAddress = "http://etcd.test:2379"
	expected, err := generatePodSpecForComponent(component, BackendFrameworkSGLang, nil, dgd, RoleMain, 1, operatorConfig,
		consts.MultinodeDeploymentTypeGrove, component.ComponentName, nil, roleLaunchOwnedByOperator, nil, func() (int64, error) { return 1, nil })
	require.NoError(t, err)
	expected.Containers[0].LivenessProbe = main.LivenessProbe
	expected.Containers[0].ReadinessProbe = main.ReadinessProbe
	expected.Containers[0].StartupProbe = main.StartupProbe
	expected.Containers[0].Env = append([]corev1.EnvVar{
		{Name: consts.DynamoRankEnvVar, Value: "$(GROVE_PCLQ_POD_INDEX)"},
		{Name: consts.DynamoLeaderAddressEnvVar, Value: "$(GROVE_PCLQ_NAME)-0.$(GROVE_HEADLESS_SERVICE)"},
	}, expected.Containers[0].Env...)

	t.Log("reuse shared defaults while preserving supplied probes and exact native launch arguments")
	pcs, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, operatorConfig, &common.RuntimeConfig{}, nil, nil, nil, nil, false, nil)
	require.NoError(t, err)
	assert.Equal(t, *expected, pcs.Spec.Template.Cliques[0].Spec.PodSpec)
	renderedMain := pcs.Spec.Template.Cliques[0].Spec.PodSpec.Containers[0]
	assert.Equal(t, main.Command, renderedMain.Command)
	assert.Equal(t, main.Args, renderedMain.Args)
	assert.Contains(t, renderedMain.Env, corev1.EnvVar{Name: "NATS_SERVER", Value: operatorConfig.Infrastructure.NATSAddress})
	assert.Contains(t, renderedMain.Env, corev1.EnvVar{Name: "ETCD_ENDPOINTS", Value: operatorConfig.Infrastructure.ETCDAddress})
	assert.Contains(t, renderedMain.Env, parentEnv[0])
	assert.Contains(t, renderedMain.Env, corev1.EnvVar{Name: "DYN_SYSTEM_PORT", Value: "9090"})
	assert.Equal(t, before, dgd)

	t.Log("parent environment cannot replace the provider-neutral topology bindings")
	for _, name := range []string{consts.DynamoRankEnvVar, consts.DynamoLeaderAddressEnvVar} {
		_, err := ResolveSGLangComponentEngineGroup(component, []corev1.EnvVar{{Name: name, Value: "wrong"}})
		require.ErrorContains(t, err, "operator-owned topology")
	}
}

func TestSGLangEngineGroupManualTemplates(t *testing.T) {
	for _, path := range []string{"../enginegroup/sglang/growth-test.yaml", "../../../../tests/fault_tolerance/deploy/templates/sglang/sglang_elastic_ep_engine_group_poc.yaml"} {
		t.Run(path, func(t *testing.T) {
			t.Log("load the user-facing DGD without adding launch defaults")
			payload, err := os.ReadFile(path)
			require.NoError(t, err)
			dgd := &api.DynamoGraphDeployment{}
			require.NoError(t, yaml.UnmarshalStrict(payload, dgd))
			before := dgd.DeepCopy()

			t.Log("validate every Engine Group against its explicitly declared bootstrap contract")
			for i := range dgd.Spec.Components {
				component := &dgd.Spec.Components[i]
				if component.EngineGroup != nil {
					_, err := RenderComponentEngineGroup(dgd, component, 0)
					require.NoError(t, err)
				}
			}
			assert.Equal(t, before, dgd)
		})
	}
}

func testEngineGroupComponent() *api.DynamoComponentDeploymentSharedSpec {
	return &api.DynamoComponentDeploymentSharedSpec{
		ComponentName: "Worker", ComponentType: api.ComponentTypeWorker, Replicas: new(int32(1)),
		EngineGroup: &api.ComponentEngineGroupSpec{InitialSize: 2, Policy: &api.ComponentEngineGroupPolicy{MinSize: new(int32(2)), MaxSize: new(int32(3))}},
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyNever, Containers: []corev1.Container{{
			Name: "main", Image: "runtime:1.1.0", Command: []string{"python3", "-m", "dynamo.sglang"},
			Args: []string{"--model-path", "model", "--elastic-ep-join-mode", "auto", "--enable-dp-lm-head", "--elastic-ep-backend", "mooncake", "--moe-a2a-backend", "nixl", "--elastic-ep-initial-size", "2", "--max-ep-size", "3", "--load-balance-method", "round_robin", "--disable-cuda-graph",
				"--elastic-ep-allocation-index=$(DYNAMO_RANK)", "--elastic-ep-allocation-id=$(DYNAMO_ALLOCATION_UID)",
				"--dist-init-addr=$(DYNAMO_LEADER_ADDRESS):29500"},
			Env: []corev1.EnvVar{
				{Name: "DYNAMO_ALLOCATION_UID", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.uid"}}},
				{Name: "DYN_SYSTEM_PORT", Value: "9090"},
			},
			Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}},
		}}}},
	}
}
