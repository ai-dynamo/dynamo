/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"slices"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestManifestRuntimeSelection(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name   string
		local  *v1beta1.LPXLocalPartitions
		remote string
		agents int32
	}{
		{name: "omitted", remote: "7,8", agents: 4},
		{name: "leading local", local: &v1beta1.LPXLocalPartitions{IDs: []int64{7}}, remote: "8", agents: 2},
		{name: "all local", local: &v1beta1.LPXLocalPartitions{All: true}, remote: "", agents: 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve the operator-owned remote placement")
			snapshot := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
			snapshot.build.CompilationMode = BuildCompilationModeHybrid
			snapshot.build.SelectedPropSyncChains = nil
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineLPX, Models: []string{"default"}, RuntimeBuildRef: "model-build",
				BuildSnapshot: snapshot, LocalPartitions: test.local,
			})
			require.NoError(t, err)
			projections[0].stage = testRenderComponentName
			workload := &Workload{modelProjections: projections, digest: projections[0].Digest(), scalingGroupReplicas: 2}
			plan, err := workload.PlanNodeLocalMaterialization("example")
			require.NoError(t, err)
			plan, err = plan.WithGroup("second-workload")
			require.NoError(t, err)

			t.Log("Migrate both runtime templates and render without ConfigMaps")
			agent := renderTestPodSpec()
			agent.Containers[0].Env = append(agent.Containers[0].Env, corev1.EnvVar{Name: runtimeContractEnv, Value: manifestRuntimeContract})
			cyborg := renderTestPCS(true).Spec.Template.Cliques[0]
			cyborg.Name = plan.CyborgTemplate
			for _, spec := range []*corev1.PodSpec{&agent, &cyborg.Spec.PodSpec} {
				for i := range spec.Containers {
					spec.Containers[i].VolumeMounts = slices.DeleteFunc(spec.Containers[i].VolumeMounts, func(m corev1.VolumeMount) bool { return m.Name == lpuConfigVolumeName })
				}
			}
			cyborg.Spec.PodSpec.Containers[0].Env = append(cyborg.Spec.PodSpec.Containers[0].Env,
				corev1.EnvVar{Name: runtimeContractEnv, Value: manifestRuntimeContract},
				corev1.EnvVar{Name: "LPX_REMOTE_PARTITION_IDS", Value: "untrusted"})
			rendered, err := RenderNodeLocal(workload, plan, RenderInput{
				Stages: map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: agent}}, Cyborg: cyborg,
			})
			require.NoError(t, err)
			require.Empty(t, rendered.Resources)
			var agentCount int32
			for _, clique := range rendered.Cliques {
				require.NotContains(t, clique.Annotations, v1alpha1.AnnotationExtraResourcesHash)
				container := clique.Spec.PodSpec.Containers[0]
				var ids []corev1.EnvVar
				for _, variable := range container.Env {
					if variable.Name == "LPX_REMOTE_PARTITION_IDS" {
						ids = append(ids, variable)
					}
				}
				require.Equal(t, []corev1.EnvVar{{Name: "LPX_REMOTE_PARTITION_IDS", Value: test.remote}}, ids)
				if clique.Name != plan.CyborgTemplate {
					agentCount += clique.Spec.Replicas
				}
				for _, volume := range clique.Spec.PodSpec.Volumes {
					require.Nil(t, volume.ConfigMap)
				}
			}
			require.Equal(t, test.agents, agentCount)
			require.Contains(t, cyborg.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{
				Name: "LPX_AGENT_HOST_TEMPLATE", Value: plan.ScalingGroupTemplate + "-${GROVE_PCSG_INDEX}-" + plan.Agents[0].TemplateName + "-${LPX_LEADER_OFFSET}",
			})
		})
	}
}

func TestNovaSelectionsRemainPerModel(t *testing.T) {
	t.Parallel()
	t.Log("Keep sparse draft and target selections independent and ordered")
	draft := &ModelProjection{pipeline: PipelineSpecDecode, configuredBuild: Build{Partitions: []BuildPartition{{SourcePartitionID: 7}, {SourcePartitionID: 3}}}}
	target := &ModelProjection{configuredBuild: Build{Partitions: []BuildPartition{{SourcePartitionID: 11}}}}
	container := &corev1.Container{Env: []corev1.EnvVar{{Name: "NOVA_DRAFT_REMOTE_PARTITION_IDS", Value: "0"}}}
	applyNovaSelections(container, []*ModelProjection{draft, target})
	require.Equal(t, []corev1.EnvVar{
		{Name: "NOVA_DRAFT_REMOTE_PARTITION_IDS", Value: "7,3"},
		{Name: "NOVA_TARGET_REMOTE_PARTITION_IDS", Value: "11"},
	}, container.Env)
}

func TestManifestRuntimeMigrationValidation(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name     string
		env      []corev1.EnvVar
		mounts   []corev1.VolumeMount
		selected bool
		invalid  bool
	}{
		{name: "unchanged legacy"},
		{name: "explicit manifest", env: []corev1.EnvVar{{Name: runtimeContractEnv, Value: manifestRuntimeContract}}, selected: true},
		{name: "unknown contract", env: []corev1.EnvVar{{Name: runtimeContractEnv, Value: "future"}}, invalid: true},
		{name: "stale directory", env: []corev1.EnvVar{{Name: runtimeContractEnv, Value: manifestRuntimeContract}, {Name: "NOVA_RESOLVED_PARTITIONS_DIR", Value: "/configs"}}, invalid: true},
		{name: "stale mount", env: []corev1.EnvVar{{Name: runtimeContractEnv, Value: manifestRuntimeContract}}, mounts: []corev1.VolumeMount{{Name: lpuConfigVolumeName, MountPath: "/configs"}}, invalid: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Validate an explicit main-container runtime migration")
			selected, err := manifestRuntime(corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Env: test.env, VolumeMounts: test.mounts}}})
			if test.invalid {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.selected, selected)
		})
	}
}
