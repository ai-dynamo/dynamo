/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
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
		{name: "leading local", local: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{7}}, remote: "8", agents: 2},
		{name: "all local", local: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeAll}, remote: "", agents: 0},
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

			t.Log("Publish the selection alongside existing runtime metadata")
			agent := renderTestPodSpec()
			cyborg := renderTestPCS(true).Spec.Template.Cliques[0]
			cyborg.Name = plan.CyborgTemplate
			cyborg.Spec.PodSpec.Containers[0].Env = append(cyborg.Spec.PodSpec.Containers[0].Env,
				corev1.EnvVar{Name: "LPX_REMOTE_PARTITION_IDS", Value: "untrusted"})
			rendered, err := RenderNodeLocal(workload, plan, RenderInput{
				Stages: map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: agent}}, Cyborg: cyborg,
			})
			require.NoError(t, err)
			var agentCount int32
			for _, clique := range rendered.Cliques {
				container := clique.Spec.PodSpec.Containers[0]
				var ids []corev1.EnvVar
				for _, variable := range container.Env {
					if variable.Name == "LPX_REMOTE_PARTITION_IDS" {
						ids = append(ids, variable)
					}
				}
				if test.local == nil {
					require.Empty(t, ids, "ordinary placement uses manifest defaults")
				} else {
					require.Equal(t, []corev1.EnvVar{{Name: "LPX_REMOTE_PARTITION_IDS", Value: test.remote}}, ids)
				}
				if clique.Name != plan.CyborgTemplate {
					agentCount += clique.Spec.Replicas
				}

			}
			require.Equal(t, test.agents, agentCount)
			require.Contains(t, cyborg.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{
				Name: "LPX_AGENT_HOST_TEMPLATE", Value: "${GROVE_PCS_NAME}-${GROVE_PCS_INDEX}-" + plan.ScalingGroupTemplate + "-${GROVE_PCSG_INDEX}-" + plan.Agents[0].TemplateName + "-${LPX_LEADER_OFFSET}.${GROVE_HEADLESS_SERVICE}",
			})
		})
	}
}

func TestNovaSelectionsRemainPerModel(t *testing.T) {
	t.Parallel()
	t.Log("Keep sparse draft and target selections independent and ordered")
	draft := &ModelProjection{remoteSelectionRequired: true, pipeline: PipelineSpecDecode, configuredBuild: Build{Partitions: []BuildPartition{{SourcePartitionID: 7}, {SourcePartitionID: 3}}}}
	target := &ModelProjection{remoteSelectionRequired: true, configuredBuild: Build{Partitions: []BuildPartition{{SourcePartitionID: 11}}}}
	container := &corev1.Container{Env: []corev1.EnvVar{{Name: "NOVA_DRAFT_REMOTE_PARTITION_IDS", Value: "0"}}}
	applyNovaSelections(container, []*ModelProjection{draft, target})
	require.Equal(t, []corev1.EnvVar{
		{Name: "NOVA_DRAFT_REMOTE_PARTITION_IDS", Value: "7,3"},
		{Name: "NOVA_TARGET_REMOTE_PARTITION_IDS", Value: "11"},
	}, container.Env)
}

func TestRenderHXHybridRuntimeSelection(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		selection *v1beta1.LPXLocalPartitions
		wantIDs   string
	}{
		{
			name:    "every partition on LPUs addresses chain 2-3 through partition 2",
			wantIDs: "1,2,3,4",
		},
		{
			name:      "partition 1 on the GPU",
			selection: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{1}},
			wantIDs:   "2,3,4",
		},
		{
			name:      "chain 2-3 on the GPU",
			selection: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{2}},
			wantIDs:   "1,4",
		},
		{
			name:      "every partition on the GPU",
			selection: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeAll},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Create a hybrid HX build with partitions 1-4, a selected chain 2-3, and one CUDA artifact")
			fixture := newV3CompilerFixture()
			fixture.compilationMode = manifestcapnp.CompilationMode_lpx
			fixture.numLPUNodes = 4
			for id := uint32(2); id <= 4; id++ {
				partition := fixture.partitions[0]
				partition.id = id
				fixture.partitions = append(fixture.partitions, partition)
			}
			fixture.partitions = append(fixture.partitions, testV3CapnpPartition{id: 5, deviceType: manifestcapnp.DeviceType_cuda})
			fixture.selectedPropSyncChains = [][]uint32{{2, 3}}
			normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeCompilerFixture(t, fixture)))
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineLPX, Models: []string{"default"}, BuildSnapshot: normalized,
				RuntimeBuildRef: "model-build", LocalPartitions: test.selection,
			})
			require.NoError(t, err)
			require.Equal(t, BuildFamilyHX, projections[0].configuredBuild.Family)

			t.Log("Render the hybrid workload with the explicit selection")
			projections[0].stage = testRenderComponentName
			workload := &Workload{modelProjections: projections, scalingGroupReplicas: 1}
			workload.digest, err = workloadSetDigest(projections)
			require.NoError(t, err)
			plan, err := workload.PlanNodeLocalMaterialization("test-pcs")
			require.NoError(t, err)
			pcs := renderTestPCS(true)
			rendered, err := RenderNodeLocal(workload, plan, RenderInput{
				Stages: map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}},
				Cyborg: pcs.Spec.Template.Cliques[0],
			})
			require.NoError(t, err)

			t.Log("Publish the physical selection to Cyborg and every Agent")
			for _, clique := range rendered.Cliques {
				container := clique.Spec.PodSpec.Containers[0]
				if test.selection == nil {
					for _, variable := range container.Env {
						require.NotEqual(t, "LPX_REMOTE_PARTITION_IDS", variable.Name)
					}
				} else {
					require.Contains(t, container.Env, corev1.EnvVar{Name: "LPX_REMOTE_PARTITION_IDS", Value: test.wantIDs})
				}
			}
			if test.wantIDs == "" {
				require.Len(t, rendered.Cliques, 1)
			}
		})
	}
}
