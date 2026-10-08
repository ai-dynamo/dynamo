/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestManifestRuntimeSelection(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name     string
		local    *v1beta1.LPXLocalPartitions
		localIDs string
		agents   int32
	}{
		{name: "omitted", localIDs: "", agents: 4},
		{name: "leading local", local: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{7}}, localIDs: "7", agents: 2},
		{name: "all local", local: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeAll}, localIDs: "7,8", agents: 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve the operator-owned remote placement")
			fixture := newV2CompilerFixture()
			fixture.compilationMode = manifestcapnp.CompilationMode_lpx
			fixture.selectedPropSyncChains = nil
			snapshot := acquireTestSnapshot(t, writeCompilerFixture(t, fixture))
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineHybrid, Models: []string{"default"}, RuntimeBuildRef: "model-build",
				Build: snapshot, LocalPartitions: test.local,
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
			cyborg := renderTestPCS(true).Spec.Template.Cliques[0]
			cyborg.Name = plan.CyborgTemplate
			cyborg.Spec.PodSpec.Containers[0].Env = append(cyborg.Spec.PodSpec.Containers[0].Env,
				corev1.EnvVar{Name: localPartitionIDsEnv, Value: "untrusted"})
			rendered, err := RenderNodeLocal(workload, plan, RenderInput{
				Stages: map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: agent}}, Cyborg: cyborg,
			})
			require.NoError(t, err)
			var agentCount int32
			for _, clique := range rendered.Cliques {
				require.NotContains(t, clique.Annotations, v1alpha1.AnnotationExtraResourcesHash)
				container := clique.Spec.PodSpec.Containers[0]
				var ids []corev1.EnvVar
				for _, variable := range container.Env {
					if variable.Name == localPartitionIDsEnv {
						ids = append(ids, variable)
					}
				}
				require.Equal(t, []corev1.EnvVar{{Name: localPartitionIDsEnv, Value: test.localIDs}}, ids)
				for _, variable := range container.Env {
					require.NotEqual(t, "LPX_REMOTE_PARTITION_IDS", variable.Name)
				}
				if clique.Name != plan.CyborgTemplate {
					agentCount += clique.Spec.Replicas
				}
				for _, volume := range clique.Spec.PodSpec.Volumes {
					require.Nil(t, volume.ConfigMap)
				}
			}
			require.Equal(t, test.agents, agentCount)
			// Addressing uses this workload's scaling-group replica for every role.
			for _, clique := range rendered.Cliques {
				require.Contains(t, rendered.ScalingGroup.CliqueNames, clique.Name)
			}
			allocation := ""
			if test.agents > 0 {
				allocation = "second-workload-agt"
			}
			require.Contains(t, cyborg.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{
				Name: allocationEnvVar, Value: allocation,
			})
		})
	}
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
			wantIDs: "",
		},
		{
			name:      "partition 1 on the GPU",
			selection: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{1}},
			wantIDs:   "1",
		},
		{
			name:      "chain 2-3 on the GPU",
			selection: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{2}},
			wantIDs:   "2",
		},
		{
			name:      "every partition on the GPU",
			selection: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeAll},
			wantIDs:   "1,2,4",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Create a hybrid HX build with partitions 1-4, a selected chain 2-3, and one CUDA artifact")
			fixture := newV3CompilerFixture()
			fixture.compilationMode = manifestcapnp.CompilationMode_lpx
			for id := uint32(2); id <= 4; id++ {
				partition := fixture.partitions[0]
				partition.id = id
				fixture.partitions = append(fixture.partitions, partition)
			}
			fixture.partitions = append(fixture.partitions, testV3CapnpPartition{id: 5, deviceType: manifestcapnp.DeviceType_cuda})
			fixture.selectedPropSyncChains = [][]uint32{{2, 3}}
			normalized := acquireTestSnapshot(t, writeCompilerFixture(t, fixture))
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineHybrid, Models: []string{"default"}, Build: normalized,
				RuntimeBuildRef: "model-build", LocalPartitions: test.selection,
			})
			require.NoError(t, err)
			require.Equal(t, BuildFamilyHX, projections[0].configuredBuild.family)

			t.Log("Render the hybrid workload without runtime ConfigMaps")
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

			t.Log("Publish the same local runtime roots to Cyborg and every Agent")
			for _, clique := range rendered.Cliques {
				container := clique.Spec.PodSpec.Containers[0]
				require.Contains(t, container.Env, corev1.EnvVar{Name: localPartitionIDsEnv, Value: test.wantIDs})
				for _, variable := range container.Env {
					require.NotEqual(t, "LPX_REMOTE_PARTITION_IDS", variable.Name)
				}
			}
			if test.selection != nil && test.selection.Mode == v1beta1.LPXLocalPartitionsModeAll {
				require.Len(t, rendered.Cliques, 1)
			}
		})
	}
}
