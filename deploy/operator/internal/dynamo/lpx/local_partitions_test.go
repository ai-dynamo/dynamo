/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

const testPart11Path = "part-11"

func TestProjectModelV2LocalPartitions(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name            string
		pipeline        Pipeline
		selection       *dynamov1beta1.LPXLocalPartitions
		wantErr         string
		wantAgents      int
		wantCompilerIDs []int64
		wantConnectors  int
		wantRuntimeIDs  string
		wantLocalIDs    []int
	}{
		{
			name: "independent partition runs locally", pipeline: PipelineLPX,
			selection:  &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{11}},
			wantAgents: 9, wantCompilerIDs: []int64{7, 8}, wantConnectors: 1,
			wantRuntimeIDs: "7", wantLocalIDs: []int{11},
		},
		{
			name: "selected chain runs locally as one runtime partition", pipeline: PipelineLPX,
			selection:  &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{7}},
			wantAgents: 8, wantCompilerIDs: []int64{11}, wantConnectors: 0,
			wantRuntimeIDs: "11", wantLocalIDs: []int{7},
		},
		{
			name: "every partition runs locally", pipeline: PipelineLPX,
			selection:  &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeAll},
			wantAgents: 0, wantCompilerIDs: []int64{}, wantConnectors: 0,
			wantRuntimeIDs: "", wantLocalIDs: []int{7, 11},
		},
		{
			name: "chain member is not a runtime partition", pipeline: PipelineLPX,
			selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{8}},
			wantErr:   "localPartitions references partition 8 of the prop-sync chain that starts at partition 7; select 7",
		},
		{
			name: "unknown partition", pipeline: PipelineLPX,
			selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{99}},
			wantErr:   "localPartitions references partition 99, which the build does not contain",
		},
		{
			name: "unsupported mode", pipeline: PipelineLPX,
			selection: &dynamov1beta1.LPXLocalPartitions{Mode: "Roles"},
			wantErr:   `localPartitions has unsupported mode "Roles"`,
		},
		{
			name: "LPU-only pipeline has no Cyborg GPU", pipeline: PipelineSingle,
			selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{11}},
			wantErr:   "unsupported LPX runtime: localPartitions requires a hybrid build with a Cyborg conductor",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Create a hybrid build with chain 7-8 and an independent partition 11")
			normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
			build := normalized.build
			build.CompilationMode = BuildCompilationModeHybrid
			build.Partitions[0].Topology = Topology{Raw: "stage-a", ChipCount: 8}
			build.Partitions[1].Topology = Topology{Raw: "stage-b", ChipCount: 64}
			third := build.Partitions[1]
			third.SourcePartitionID = 11
			third.PartPath = testPart11Path
			build.Partitions = append(build.Partitions, third)
			build.SelectedPropSyncChains = [][]int{{7, 8}}

			t.Log("Project the build with the selected local partitions")
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: test.pipeline, Models: []string{"default"}, BuildSnapshot: normalized,
				LocalPartitions: test.selection,
			})
			if test.wantErr != "" {
				require.EqualError(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
			projection := projections[0]

			t.Log("Request LPU placement only for the remote physical partitions")
			spec := projection.RequestSpec(&MaterializationPlan{}, "agents")
			compilerIDs := make([]int64, 0, len(spec.Partitions))
			for ordinal, partition := range spec.Partitions {
				require.Equal(t, int64(ordinal), partition.Ordinal)
				compilerIDs = append(compilerIDs, partition.CompilerPartitionID)
			}
			require.Equal(t, test.wantCompilerIDs, compilerIDs)
			require.Len(t, spec.PropSyncConnectors, test.wantConnectors)
			require.Equal(t, test.wantAgents, projection.AgentReplicas())

			t.Log("Publish only remote runtime partitions to Agents and Cyborg")
			require.Equal(t, test.wantRuntimeIDs, resolvedPartitionData([]*ModelProjection{projection})["partition_ids"])
			require.Equal(t, test.wantLocalIDs, projection.localPartitionIDs)

			t.Log("Change the workload digest when partitions move to the GPU")
			baseline := projectTestBuild(t, normalized, PipelineLPX)
			require.NotEqual(t, baseline.Digest(), projection.Digest())
		})
	}
}

func TestApplyLocalPartitionIDs(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		localIDs []int
		env      []corev1.EnvVar
		envFrom  []corev1.EnvFromSource
		wantEnv  []corev1.EnvVar
	}{
		{
			name:    "no local partitions removes authored values",
			env:     []corev1.EnvVar{{Name: localPartitionIDsEnv, Value: "authored"}, {Name: "OTHER", Value: "kept"}},
			wantEnv: []corev1.EnvVar{{Name: "OTHER", Value: "kept"}},
		},
		{
			name:    "no local partitions shadows envFrom sources",
			envFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "authored"}}}},
			wantEnv: []corev1.EnvVar{{Name: localPartitionIDsEnv}},
		},
		{
			name:     "resolved selection replaces authored values",
			localIDs: []int{0, 2, 12},
			env:      []corev1.EnvVar{{Name: localPartitionIDsEnv, Value: "authored"}, {Name: "OTHER", Value: "kept"}},
			wantEnv:  []corev1.EnvVar{{Name: localPartitionIDsEnv, Value: "0,2,12"}, {Name: "OTHER", Value: "kept"}},
		},
		{
			name:     "resolved selection precedes dependent values",
			localIDs: []int{11},
			env: []corev1.EnvVar{
				{Name: "FIRST", Value: "kept"},
				{Name: "LOCAL_IDS", Value: "$(" + localPartitionIDsEnv + ")"},
				{Name: localPartitionIDsEnv, Value: "authored"},
				{Name: "LAST", Value: "kept"},
			},
			wantEnv: []corev1.EnvVar{
				{Name: localPartitionIDsEnv, Value: "11"},
				{Name: "FIRST", Value: "kept"},
				{Name: "LOCAL_IDS", Value: "$(" + localPartitionIDsEnv + ")"},
				{Name: "LAST", Value: "kept"},
			},
		},
		{
			name:    "envFrom shadow precedes dependent values",
			envFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "authored"}}}},
			env: []corev1.EnvVar{
				{Name: "FIRST", Value: "kept"},
				{Name: "LOCAL_IDS", Value: "$(" + localPartitionIDsEnv + ")"},
			},
			wantEnv: []corev1.EnvVar{
				{Name: localPartitionIDsEnv},
				{Name: "FIRST", Value: "kept"},
				{Name: "LOCAL_IDS", Value: "$(" + localPartitionIDsEnv + ")"},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Publish the resolved local partitions into the Cyborg container")
			container := &corev1.Container{Env: test.env, EnvFrom: test.envFrom}
			applyLocalPartitionIDs(container, &ModelProjection{localPartitionIDs: test.localIDs})
			require.Equal(t, test.wantEnv, container.Env)
		})
	}
}

func TestRenderHybridLocalPartitions(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name             string
		selection        *dynamov1beta1.LPXLocalPartitions
		wantAgents       int32
		wantStartsAfter  []string
		wantGroupMembers []string
		wantLocalEnv     string
	}{
		{
			name: "partial selection renders Agents for remote partitions", selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{11}},
			wantAgents: 9, wantStartsAfter: []string{"agt"}, wantGroupMembers: []string{"agt", "cond"}, wantLocalEnv: "11",
		},
		{
			name: "all-local selection renders only Cyborg", selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeAll},
			wantStartsAfter: []string{}, wantGroupMembers: []string{"cond"}, wantLocalEnv: "7,11",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Project a hybrid build with chain 7-8 and an independent partition 11")
			normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
			build := normalized.build
			build.CompilationMode = BuildCompilationModeHybrid
			build.Partitions[0].Topology = Topology{Raw: "stage-a", ChipCount: 8}
			build.Partitions[1].Topology = Topology{Raw: "stage-b", ChipCount: 64}
			third := build.Partitions[1]
			third.SourcePartitionID = 11
			third.PartPath = testPart11Path
			build.Partitions = append(build.Partitions, third)
			build.SelectedPropSyncChains = [][]int{{7, 8}}
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineLPX, Models: []string{"default"}, BuildSnapshot: normalized,
				RuntimeBuildRef: "model-build", LocalPartitions: test.selection,
			})
			require.NoError(t, err)

			t.Log("Render the hybrid workload")
			rendered, err := renderSelectedForTest(renderTestPCS(true), projections, RenderInput{
				Stages: map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}},
			})
			require.NoError(t, err)

			t.Log("Render Agents only for remote partitions and start Cyborg after them")
			var agentReplicas int32
			for _, clique := range rendered.Spec.Template.Cliques {
				if clique.Name == testAgentTemplateName {
					agentReplicas = clique.Spec.Replicas
				}
			}
			require.Equal(t, test.wantAgents, agentReplicas)
			cyborg := namedClique(t, rendered, "cond")
			require.Equal(t, test.wantStartsAfter, cyborg.Spec.StartsAfter)
			require.ElementsMatch(t, test.wantGroupMembers, rendered.Spec.Template.PodCliqueScalingGroupConfigs[0].CliqueNames)

			t.Log("Publish the resolved local partitions to Cyborg")
			localEnv := ""
			for _, variable := range cyborg.Spec.PodSpec.Containers[0].Env {
				if variable.Name == localPartitionIDsEnv {
					localEnv = variable.Value
				}
			}
			require.Equal(t, test.wantLocalEnv, localEnv)
		})
	}
}

func TestProjectModelV2LocalPartitionsKeepsPhysicalBuildBound(t *testing.T) {
	t.Parallel()

	t.Log("Create a hybrid build with one more physical partition than the LPX limit")
	normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	build := normalized.build
	build.CompilationMode = BuildCompilationModeHybrid
	build.SelectedPropSyncChains = nil
	template := build.Partitions[0]
	build.Partitions = make([]BuildPartition, 0, maxLPXPartitions+1)
	for id := range maxLPXPartitions + 1 {
		partition := template
		partition.SourcePartitionID = id
		build.Partitions = append(build.Partitions, partition)
	}

	t.Log("Reject the oversized build even when every partition runs locally")
	_, err := appendModelProjections(nil, ModelProjectionInput{
		Pipeline: PipelineLPX, Models: []string{"default"}, BuildSnapshot: normalized,
		LocalPartitions: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeAll},
	})
	require.EqualError(t, err, "LPX projection has 257 partitions, limit is 1..256")
}
