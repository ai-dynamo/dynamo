/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"testing"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
)

func TestProjectModelV2UsesManifestPropSync(t *testing.T) {
	t.Parallel()

	t.Log("Acquire a XT build with compiler-selected prop-sync metadata")
	normalized := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	build := normalized
	require.Equal(t, [][]int{{7, 8}}, build.selectedPropSyncChains)

	t.Log("Preserve both physical partitions with their adjacent selected connector")
	projection := projectTestBuild(t, normalized, PipelineSingle)
	spec := projection.RequestSpec(&MaterializationPlan{}, "agents")
	require.Len(t, spec.Partitions, 2)
	require.Len(t, spec.PropSyncConnectors, 1)
	require.Equal(t, int64(0), *spec.PropSyncConnectors[0].Requirement.MaxInterPartitionOffset)
	require.Equal(t, [][]int{{7, 8}}, build.selectedPropSyncChains)

	t.Log("Keep all physical partitions without synthesizing a chain absent from the manifest")
	build.selectedPropSyncChains = nil
	projection = projectTestBuild(t, normalized, PipelineSingle)
	spec = projection.RequestSpec(&MaterializationPlan{}, "agents")
	require.Len(t, spec.Partitions, 2)
	require.Empty(t, spec.PropSyncConnectors)
	require.Equal(t, 4, projection.agentReplicas)
}

func TestProjectModelV2SingleEmbeddingPlacementFromManifest(t *testing.T) {
	for _, test := range []struct {
		name                     string
		cpuSupported, standalone bool
		firstPartition           int64
	}{
		{name: "default CPU embeddings", cpuSupported: true, standalone: true, firstPartition: 1},
		{name: "CPU embeddings unsupported", standalone: true},
		{name: "partition zero is runnable", cpuSupported: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Prepare a selected XT chain with explicit embedding placement capabilities")
			normalized := acquireTestSnapshot(t, writeV2CompilerFixture(t))
			build := normalized
			build.partitions[0].sourcePartitionID = 0
			build.partitions[1].sourcePartitionID = 1
			build.standaloneTokenEmbeddings = test.standalone
			build.supportsCPUEmbeddings = test.cpuSupported
			build.selectedPropSyncChains = [][]int{{0, 1}}

			t.Log("Project retained source partitions into placement and runtime metadata")
			projection := projectTestBuild(t, normalized, PipelineSingle)
			spec := projection.RequestSpec(&MaterializationPlan{}, "agents")
			require.Equal(t, test.firstPartition, spec.Partitions[0].CompilerPartitionID)
			require.EqualValues(t, 2*(2-test.firstPartition), projection.agentReplicas)
			require.EqualValues(t, test.firstPartition, projection.configuredBuild.partitions[0].sourcePartitionID)
			require.Empty(t, projection.configuredBuild.selectedPropSyncChains)

			t.Log("Keep scheduler output independently mutable from the projection and compiler snapshot")
			spec.Partitions[0].CompilerPartitionID = -1
			require.EqualValues(t, test.firstPartition, projection.partitions[0].sourcePartitionID)
			require.EqualValues(t, test.firstPartition, projection.RequestSpec(&MaterializationPlan{}, "agents").Partitions[0].CompilerPartitionID)
			require.Equal(t, 0, build.partitions[0].sourcePartitionID)
			require.Equal(t, [][]int{{0, 1}}, build.selectedPropSyncChains)
		})
	}
}

func TestProjectModelV2StrictHybridPreservesPartitionZero(t *testing.T) {
	t.Log("Prepare a hybrid XT build with a standalone embedding partition")
	normalized := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	build := normalized
	build.partitions[0].sourcePartitionID = 0
	build.partitions[1].sourcePartitionID = 1
	build.standaloneTokenEmbeddings = true
	build.supportsCPUEmbeddings = true
	build.selectedPropSyncChains = nil

	t.Log("Keep Cyborg's partition zero independently of Nova's embedding placement")
	projection := projectTestBuild(t, normalized, PipelineHybrid)
	spec := projection.RequestSpec(&MaterializationPlan{}, "agents")
	require.Equal(t, lpxv1alpha1.WorkloadModeV2StrictHybrid, spec.WorkloadMode)
	require.Len(t, spec.Partitions, 2)
	require.Equal(t, int64(0), spec.Partitions[0].CompilerPartitionID)
	require.Equal(t, 4, projection.agentReplicas)
	data := resolvedPartitionData([]*ModelProjection{projection})
	require.Equal(t, "0\n1", data["partition_ids"])
	require.Equal(t, "0\n2", data["partition_node_offsets"])
}

func TestProjectModelV2UsesOnlyTheSourceSelectedAdjacentChain(t *testing.T) {
	t.Parallel()

	t.Log("Prepare a selected adjacent chain between unselected physical partitions")
	normalized := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	build := normalized
	prefix, suffix := build.partitions[0], build.partitions[1]
	prefix.sourcePartitionID, prefix.partPath = 3, "part-3"
	suffix.sourcePartitionID, suffix.partPath = 11, "part-11"
	build.partitions = append([]buildPartition{prefix}, append(build.partitions, suffix)...)
	build.selectedPropSyncChains = [][]int{{7, 8}}
	build.partitions[1].topology = " stage-a "
	build.partitions[2].topology = "stage-b"

	t.Log("Project the compiler-selected chain")
	projection := projectTestBuild(t, normalized, PipelineSingle)

	t.Log("Preserve only the source-selected connector and partition identities")
	spec := projection.RequestSpec(&MaterializationPlan{}, "agents")
	require.Len(t, spec.Partitions, 2)
	require.Len(t, spec.PropSyncConnectors, 1)
	require.Equal(t, int64(7), spec.Partitions[0].CompilerPartitionID)
	require.Equal(t, int64(8), spec.Partitions[1].CompilerPartitionID)
	require.Equal(t, spec.Partitions[0].ID, spec.PropSyncConnectors[0].FromPartitionID)
	require.Equal(t, spec.Partitions[1].ID, spec.PropSyncConnectors[0].ToPartitionID)

	t.Log("Forward the selected stages' opaque names unchanged")

	require.NotContains(t, resolvedPartitionData([]*ModelProjection{projection}), "topologies")
}

func TestProjectModelV2SeparatesPhysicalPartitionsFromRuntimeChain(t *testing.T) {
	t.Parallel()
	normalized := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	build := normalized

	t.Log("Use individually schedulable physical shapes whose collapsed runtime total is not a scheduler shape")
	build.partitions[0].topology = "stage-a"
	build.partitions[0].numChips = 8
	build.partitions[1].topology = "stage-b"
	build.partitions[1].numChips = 64
	third := build.partitions[1]
	third.sourcePartitionID = 11
	third.partPath = "part-11"
	build.partitions = append(build.partitions, third)
	build.selectedPropSyncChains = [][]int{{7, 8}}

	t.Log("Project the selected chain beside an independent third partition")
	projection := projectTestBuild(t, normalized, PipelineHybrid)

	t.Log("Project all physical scheduler partitions and only the selected connector")
	spec := projection.RequestSpec(&MaterializationPlan{}, "agents")
	require.Len(t, spec.Partitions, 3)
	require.Len(t, spec.PropSyncConnectors, 1)
	require.Equal(t, spec.Partitions[0].ID, spec.PropSyncConnectors[0].FromPartitionID)
	require.Equal(t, spec.Partitions[1].ID, spec.PropSyncConnectors[0].ToPartitionID)

	t.Log("Collapse only the Agent runtime projection of the selected chain")
	data := resolvedPartitionData([]*ModelProjection{projection})
	require.Equal(t, "7\n11", data["partition_ids"])
	require.Equal(t, "9\n8", data["nodes_per_partition"])
	require.Equal(t, "0\n9", data["partition_node_offsets"])
	require.Equal(t, "part-7\npart-11", data["partition_paths"])

	require.NotContains(t, data, "topologies")
	require.Empty(t, projection.configuredBuild.selectedPropSyncChains)
	require.Equal(t, 17, projection.agentReplicas)
	require.Equal(t, [][]int{{7, 8}}, build.selectedPropSyncChains)
	require.Equal(t, 8, build.partitions[0].numChips)

	t.Log("Preserve physical output order when selected chains are declared in reverse")
	fourth := build.partitions[2]
	fourth.sourcePartitionID = 13
	build.partitions = append(build.partitions, fourth)
	build.selectedPropSyncChains = [][]int{{11, 13}, {7, 8}}
	connectors, err := xtConnectors(build, build.partitions)
	require.NoError(t, err)
	require.Len(t, connectors, 2)
	require.Equal(t, "partition-000", connectors[0].FromPartitionID)
	require.Equal(t, "partition-001", connectors[0].ToPartitionID)
	require.Equal(t, "partition-002", connectors[1].FromPartitionID)
	require.Equal(t, "partition-003", connectors[1].ToPartitionID)
	require.NotSame(t, connectors[0].Requirement.MaxInterPartitionOffset, connectors[1].Requirement.MaxInterPartitionOffset)
}

func TestProjectModelV2CollapsesSelectedChainIncludingPartitionZero(t *testing.T) {
	t.Parallel()
	normalized := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	build := normalized

	t.Log("Model a selected chain starting at Cyborg's standalone embedding partition")
	partitions := make([]buildPartition, 3)
	for sourceID := range partitions {
		partitions[sourceID] = build.partitions[0]
		partitions[sourceID].sourcePartitionID = sourceID
		partitions[sourceID].partPath = fmt.Sprintf("part-%d", sourceID)
		partitions[sourceID].numChips = 8
	}
	build.partitions = partitions
	build.selectedPropSyncChains = [][]int{{0, 1, 2}}
	build.standaloneTokenEmbeddings = true
	build.supportsCPUEmbeddings = true

	t.Log("Project without applying Nova's CPU-embedding partition omission")
	projection := projectTestBuild(t, normalized, PipelineHybrid)

	t.Log("Keep all physical scheduler partitions while collapsing their Agent runtime projection")
	spec := projection.RequestSpec(&MaterializationPlan{}, "agents")
	require.Len(t, spec.Partitions, 3)
	for index, partition := range spec.Partitions {
		require.Equal(t, int64(index), partition.CompilerPartitionID)
	}
	require.Len(t, spec.PropSyncConnectors, 2)
	require.Equal(t, 3, projection.agentReplicas)
	data := resolvedPartitionData([]*ModelProjection{projection})
	require.Equal(t, "0", data["partition_ids"])
	require.Equal(t, "3", data["nodes_per_partition"])
	require.Equal(t, "part-0", data["partition_paths"])
	require.Equal(t, build.partitions[0].numChips, projection.configuredBuild.partitions[0].numChips)
	require.Empty(t, projection.configuredBuild.selectedPropSyncChains)
	require.Equal(t, [][]int{{0, 1, 2}}, build.selectedPropSyncChains)
}

func TestProjectModelV2PreservesAgentReplicasWhenCollapsingSubHostPartitions(t *testing.T) {
	t.Parallel()

	t.Log("Prepare two selected XT sub-host partitions")
	normalized := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	build := normalized

	for index := range build.partitions {
		build.partitions[index].numChips = 2
	}
	build.selectedPropSyncChains = [][]int{{7, 8}}

	t.Log("Project and collapse the selected runtime chain")
	projection := projectTestBuild(t, normalized, PipelineHybrid)

	t.Log("Preserve both physical scheduler partitions and Agent replicas")
	require.Len(t, projection.RequestSpec(&MaterializationPlan{}, "agents").Partitions, 2)
	require.Equal(t, 2, projection.agentReplicas)

	t.Log("Keep both physical Agent endpoints and the original root topology")
	data := resolvedPartitionData([]*ModelProjection{projection})
	require.Equal(t, "2", data["nodes_per_partition"])
	require.Equal(t, "0", data["partition_node_offsets"])
	require.Equal(t, 2, projection.configuredBuild.partitions[0].effectiveNodeCount())
	require.Equal(t, build.partitions[0].numChips, projection.configuredBuild.partitions[0].numChips)
}

func TestBuildRejectsInvalidRuntimeSelectedPropSyncChain(t *testing.T) {
	t.Parallel()

	t.Log("Define malformed or incompatible selected prop-sync chains")
	tests := []struct {
		name    string
		chains  [][]int
		wantErr string
	}{
		{
			name:    "multiple chains",
			chains:  [][]int{{0, 1}, {2, 3}},
			wantErr: "LPU-only runtime requires exactly one selected prop-sync chain, got 2",
		},
		{
			name:    "missing partition",
			chains:  [][]int{{3, 4}},
			wantErr: "references missing partition id 4",
		},
		{
			name:    "duplicate partition",
			chains:  [][]int{{1, 2, 1}},
			wantErr: "contains duplicate partition id 1",
		},
		{
			name:    "noncontiguous partitions",
			chains:  [][]int{{1, 3}},
			wantErr: "is not contiguous at partition id 3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Configure malformed or incompatible LPU-only selected-chain metadata")
			configured := Build{
				partitions: []buildPartition{
					{sourcePartitionID: 0}, {sourcePartitionID: 1}, {sourcePartitionID: 2}, {sourcePartitionID: 3},
				},
				selectedPropSyncChains: tt.chains,
			}

			t.Log("Reject the build without consuming its selected-chain marker")
			err := configured.consumeRuntimeSelectedPropSyncChain()
			require.ErrorContains(t, err, tt.wantErr)
			require.Equal(t, tt.chains, configured.selectedPropSyncChains)
		})
	}
}
