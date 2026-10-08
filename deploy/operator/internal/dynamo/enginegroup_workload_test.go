/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"strings"
	"testing"

	config "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestResolveComponentEngineGroupProfile(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*api.DynamoComponentDeploymentSharedSpec)
		want   string
	}{
		{name: "supported one-world growth"},
		{name: "a creation seed cannot derive from zero", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) { c.EngineGroup.InitialSize = 0 }, want: "initialSize must be positive"},
		{name: "fractional GPUs do not prove an allocation", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Resources.Limits["nvidia.com/gpu"] = resource.MustParse("900m")
		}, want: "whole-GPU"},
		{name: "independent world count is not capacity", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) { c.Replicas = ptr.To(int32(2)) }, want: "one independent worker world"},
		{name: "legacy multinode is not a capacity seed", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) { c.Multinode = &api.MultinodeSpec{NodeCount: 2} }, want: "does not support multinode"},
		{name: "packed allocations are not implemented", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Resources.Limits["nvidia.com/gpu"] = resource.MustParse("4")
		}, want: "packed-replicas"},
		{name: "engine initial geometry must match seed", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) { c.EngineGroup.InitialSize = 3 }, want: "initial"},
		{name: "unfenced restart is rejected", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.RestartPolicy = corev1.RestartPolicyAlways
		}, want: "restartPolicy Never"},
		{name: "sidecar GPUs are rejected", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers = append(c.PodTemplate.Spec.Containers, corev1.Container{Name: "other", Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}})
		}, want: "exclusively"},
		{name: "policy cannot enable unsupported shrink", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) { c.EngineGroup.Policy.MinSize = ptr.To(int32(1)) }, want: "policy.minSize"},
		{name: "policy cannot exceed engine maximum", mutate: func(c *api.DynamoComponentDeploymentSharedSpec) { c.EngineGroup.Policy.MaxSize = ptr.To(int32(4)) }, want: "policy.maxSize"},
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
			profile, err := ResolveComponentEngineGroupProfile(component)
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
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec:       api.DynamoGraphDeploymentSpec{BackendFramework: "sglang", Components: []api.DynamoComponentDeploymentSharedSpec{*testEngineGroupComponent()}},
	}
	before := dgd.DeepCopy()

	t.Log("render one PCSG world with one template-invariant member clique")
	pcs, err := GenerateGrovePodCliqueSet(t.Context(), dgd, &config.OperatorConfiguration{}, &common.RuntimeConfig{}, nil, &mockSecretsRetriever{}, nil, nil, nil)
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

	t.Log("preserve launch arguments and forbid allocation health probes that only slot zero exposes")
	main := clique.Spec.PodSpec.Containers[0]
	assert.Equal(t, dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Command, main.Command)
	assert.Equal(t, dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Args, main.Args)
	assert.Equal(t, corev1.RestartPolicyNever, clique.Spec.PodSpec.RestartPolicy)
	assert.Nil(t, main.ReadinessProbe)
	assert.Nil(t, main.LivenessProbe)
	assert.Nil(t, main.StartupProbe)

	t.Log("validate the generated shape against the pinned Grove schema and CEL")
	newGrovePodCliqueSetRequestValidator(t).validate(t, pcs, nil)
}

func TestEngineGroupNameForComponent(t *testing.T) {
	cases := []struct {
		name string
		dgd  string
	}{
		{name: "short name", dgd: "graph"},
		{name: "truncated name", dgd: strings.Repeat("a", 100)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("derive stable names from the component and explicit world index")
			name := EngineGroupNameForComponent(tc.dgd, "Worker", 0)
			assert.LessOrEqual(t, len(name), 63)
			assert.Equal(t, name, EngineGroupNameForComponent(tc.dgd, "Worker", 0))
			assert.NotEqual(t, name, EngineGroupNameForComponent(tc.dgd, "Decode", 0))
			assert.NotEqual(t, name, EngineGroupNameForComponent(tc.dgd, "Worker", 1))
			assert.NotEqual(t, name, EngineGroupNameForComponent(tc.dgd, "Worker", 12))
		})
	}

	t.Log("preserve the existing world-zero name without treating every world as zero")
	assert.Equal(t, "graph-worker-0", EngineGroupNameForComponent("graph", "Worker", 0))
	assert.Equal(t, "graph-worker-12", EngineGroupNameForComponent("graph", "Worker", 12))
}

func testEngineGroupComponent() *api.DynamoComponentDeploymentSharedSpec {
	return &api.DynamoComponentDeploymentSharedSpec{
		ComponentName: "Worker", ComponentType: api.ComponentTypeWorker, Replicas: ptr.To(int32(1)),
		EngineGroup: &api.ComponentEngineGroupSpec{InitialSize: 2, Policy: &api.ComponentEngineGroupPolicy{MinSize: ptr.To(int32(2)), MaxSize: ptr.To(int32(3))}},
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name: "main", Image: "runtime:1.1.0", Command: []string{"python3", "-m", SGLangElasticEPBootstrapModule},
			Args:      []string{"--model-path", "model", "--tp", "2", "--dp", "2", "--nnodes", "2", "--moe-dense-tp-size", "1", "--enable-dp-attention", "--enable-dp-lm-head", "--elastic-ep-backend", "mooncake", "--moe-a2a-backend", "nixl", "--elastic-ep-initial-size", "2", "--max-ep-size", "3", "--load-balance-method", "round_robin", "--disable-cuda-graph"},
			Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}},
		}}}},
	}
}
