/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"fmt"
	"math"
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/capnp/gbuild_manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	"github.com/stretchr/testify/require"
)

func TestProjectModelV2SubHostPartitionUsesWholeHostShape(t *testing.T) {
	t.Log("Create an immutable V2 compiler build with one two-chip physical partition")
	topology := "URSA_V2__Q8__2C__G_96_25__KP_FEC__GHZ_1_0__NO_FPGA"
	fixture := newV2CompilerFixture()
	fixture.numLPUNodes = 1
	fixture.selectedPropSyncChains = nil
	fixture.partitions = []testV3CapnpPartition{{
		id: 1, deviceType: manifestcapnp.DeviceType_lpu, topology: topology, numChips: 2, devicesPerNode: 8,
	}}
	buildDir := writeCompilerFixture(t, fixture)
	snapshot := acquireTestSnapshot(t, buildDir)

	t.Log("Project the sub-host build through the selected single-pipeline LPX path")
	projectionBatch, err := appendModelProjections(nil, ModelProjectionInput{
		Pipeline:      PipelineSingle,
		Models:        []string{"default"},
		BuildSnapshot: normalizeTestSnapshot(t, snapshot),
	})
	require.NoError(t, err)
	projection := projectionBatch[0]

	t.Log("Verify the two-chip compiler partition reserves one C8 host and produces one Agent replica")
	spec := projection.RequestSpec("test", "agents", nil)
	require.Len(t, spec.Partitions, 1)
	require.Equal(t, lpxv1alpha1.Xt8888PartitionShapeC8, *spec.Partitions[0].XtShape)
	require.NotNil(t, spec.NodeLocal)
	require.Len(t, spec.NodeLocal.PartitionMappings, 1)
	require.Equal(t, 1, projection.agentReplicas)
}

func TestProjectModelV2ValidatesPhysicalPartitionsInOrder(t *testing.T) {
	t.Log("Create a V2 build whose first physical partition has an invalid shape")
	normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	build := normalized.build
	build.CompilationMode = BuildCompilationModeHybrid
	build.SelectedPropSyncChains = [][]int{{100, 101}}
	build.Partitions[0].Topology = Topology{ChipCount: 9}
	build.Partitions[0].SourcePartitionID = 1

	t.Log("Project the invalid physical partition")
	_, err := appendModelProjections(nil, ModelProjectionInput{
		Pipeline: PipelineLPX, Models: []string{"default"},
		BuildSnapshot: normalized,
	})

	t.Log("Reject the invalid shape at the first applicable validation boundary")
	require.EqualError(t, err, "V2 compiler partition 1: chip count 9 is not a registered XT8888 partition shape")
}

func TestXTShape(t *testing.T) {
	t.Log("Check every registered shape and every integer gap through the largest shape")
	registeredShapes := map[int]lpxv1alpha1.Xt8888PartitionShape{
		8: lpxv1alpha1.Xt8888PartitionShapeC8, 16: lpxv1alpha1.Xt8888PartitionShapeC16,
		24: lpxv1alpha1.Xt8888PartitionShapeC24, 32: lpxv1alpha1.Xt8888PartitionShapeC32,
		40: lpxv1alpha1.Xt8888PartitionShapeC40, 48: lpxv1alpha1.Xt8888PartitionShapeC48,
		56: lpxv1alpha1.Xt8888PartitionShapeC56, 64: lpxv1alpha1.Xt8888PartitionShapeC64,
		96: lpxv1alpha1.Xt8888PartitionShapeC96, 128: lpxv1alpha1.Xt8888PartitionShapeC128,
	}
	for chipCount := -1; chipCount <= 129; chipCount++ {
		roundedChipCount := max(chipCount, 8)
		wantShape, registered := registeredShapes[roundedChipCount]
		registered = registered && chipCount > 0
		shape, endpoints, err := xtShape(chipCount)
		if !registered {
			require.Error(t, err, chipCount)
			continue
		}
		require.NoError(t, err, chipCount)
		require.Equal(t, wantShape, shape, chipCount)
		require.Equal(t, int64(roundedChipCount/8), endpoints, chipCount)
	}

	t.Log("Reject integer extremes without indexing outside the registry")
	for _, chipCount := range []int{math.MinInt, math.MaxInt} {
		_, _, err := xtShape(chipCount)
		require.Error(t, err, chipCount)
	}
}

func TestProjectModelV2UsesEffectivePropSyncSettings(t *testing.T) {
	t.Parallel()

	t.Log("Acquire a V2 build with compiler-selected prop-sync metadata")
	normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	build := normalized.build
	require.Equal(t, [][]int{{7, 8}}, build.SelectedPropSyncChains)

	t.Log("Preserve both physical partitions and their selected connector without model settings")
	defaultedBatch, err := appendModelProjections(nil, ModelProjectionInput{
		Pipeline: PipelineSingle, Models: []string{"default"}, BuildSnapshot: normalized,
	})
	require.NoError(t, err)
	defaulted := defaultedBatch[0]
	defaultSpec := defaulted.RequestSpec("test", "agents", nil)
	require.Len(t, defaultSpec.Partitions, 2)
	require.Len(t, defaultSpec.PropSyncConnectors, 1)
	require.Equal(t, [][]int{{7, 8}}, build.SelectedPropSyncChains)

	t.Log("Enable prop sync and project its effective connector offset")
	projection := projectTestBuild(t, normalized, PipelineSingle, `{"prop_sync":true,"prop_sync_multiple_load_sets":true}`)
	spec := projection.RequestSpec("test", "agents", nil)
	require.Len(t, spec.PropSyncConnectors, 1)
	require.Equal(t, int64(3), *spec.PropSyncConnectors[0].Requirement.MaxInterPartitionOffset)

	t.Log("Preserve the compiler-selected chain when model settings disable implicit prop sync")
	intent := ModelProjectionInput{
		Pipeline:      PipelineSingle,
		Models:        []string{"default"},
		BuildSnapshot: normalized,
		ModelSettings: json.RawMessage(`{"prop_sync":false}`),
	}
	disabledBatch, err := appendModelProjections(nil, intent)
	require.NoError(t, err)
	disabled := disabledBatch[0]
	require.Len(t, disabled.RequestSpec("test", "agents", nil).PropSyncConnectors, 1)

	t.Log("Require manifest-selected evidence when revision 2 enables prop sync")
	build.SelectedPropSyncChains = nil
	_, err = appendModelProjections(nil, ModelProjectionInput{
		Pipeline: PipelineSingle, Models: []string{"default"},
		BuildSnapshot: normalized,
		ModelSettings: json.RawMessage(`{"prop_sync":true}`),
	})
	require.ErrorContains(t, err, "manifest v2 settings.prop_sync=true requires a selected prop-sync chain")
}

func TestProjectModelV2SingleEmbeddingPlacementFromModelSettings(t *testing.T) {
	t.Log("Prepare a V2 build with a standalone embedding partition in its selected chain")
	normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	build := normalized.build
	build.Partitions[0].SourcePartitionID = 0
	build.Partitions[1].SourcePartitionID = 1
	build.StandaloneTokenEmbeddings = true
	build.SupportsCPUEmbeddings = true
	build.SelectedPropSyncChains = [][]int{{0, 1}}

	t.Log("Omit source partition zero when CPU embeddings are enabled")
	offLPU := projectTestBuild(t, normalized, PipelineSingle, `{"cpu_embeddings":true}`)
	require.Equal(t, int64(1), offLPU.RequestSpec("test", "agents", nil).Partitions[0].CompilerPartitionID)

	t.Log("Retain source partition zero when CPU embeddings are disabled")
	onLPU := projectTestBuild(t, normalized, PipelineSingle, `{"cpu_embeddings":false}`)
	require.Equal(t, 4, onLPU.agentReplicas)
	require.Equal(t, int64(0), onLPU.RequestSpec("test", "agents", nil).Partitions[0].CompilerPartitionID)
}

func TestProjectModelV2StrictHybridForcesCPUEmbeddingsAfterValidation(t *testing.T) {
	t.Log("Prepare a hybrid V2 build with a standalone embedding partition")
	normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	build := normalized.build
	build.CompilationMode = BuildCompilationModeHybrid
	build.Partitions[0].SourcePartitionID = 0
	build.Partitions[1].SourcePartitionID = 1
	build.StandaloneTokenEmbeddings = true
	build.SupportsCPUEmbeddings = true
	build.SelectedPropSyncChains = nil

	t.Log("Force strict-hybrid CPU embeddings after accepting an explicit false setting")
	projectionBatch, err := appendModelProjections(nil, ModelProjectionInput{
		Pipeline:      PipelineLPX,
		Models:        []string{"default"},
		BuildSnapshot: normalized,
		ModelSettings: json.RawMessage(`{"cpu_embeddings":false}`),
	})
	require.NoError(t, err)
	projection := projectionBatch[0]
	spec := projection.RequestSpec("test", "agents", nil)
	require.Equal(t, lpxv1alpha1.WorkloadModeV2StrictHybrid, spec.WorkloadMode)
	require.Equal(t, int64(1), spec.Partitions[0].CompilerPartitionID)

	t.Log("Reject a malformed setting before applying the strict-hybrid override")
	_, err = appendModelProjections(nil, ModelProjectionInput{
		Pipeline:      PipelineLPX,
		Models:        []string{"default"},
		BuildSnapshot: normalized,
		ModelSettings: json.RawMessage(`{"cpu_embeddings":"false"}`),
	})
	require.EqualError(t, err, "model settings.cpu_embeddings must be a boolean")
}

func TestProjectModelV2UsesOnlyTheSourceSelectedAdjacentChain(t *testing.T) {
	t.Parallel()

	t.Log("Prepare a selected adjacent chain between unselected physical partitions")
	normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	build := normalized.build
	prefix, suffix := build.Partitions[0], build.Partitions[1]
	prefix.SourcePartitionID, prefix.PartPath = 3, "part-3"
	suffix.SourcePartitionID, suffix.PartPath = 11, "part-11"
	build.Partitions = append([]BuildPartition{prefix}, append(build.Partitions, suffix)...)
	build.SelectedPropSyncChains = [][]int{{7, 8}}

	t.Log("Project with global prop sync disabled")
	projection := projectTestBuild(t, normalized, PipelineSingle, `{"prop_sync":false}`)

	t.Log("Preserve only the source-selected connector and partition identities")
	spec := projection.RequestSpec("test", "agents", nil)
	require.Len(t, spec.Partitions, 2)
	require.Len(t, spec.PropSyncConnectors, 1)
	require.Equal(t, int64(7), spec.Partitions[0].CompilerPartitionID)
	require.Equal(t, int64(8), spec.Partitions[1].CompilerPartitionID)
	require.Equal(t, spec.Partitions[0].ID, spec.PropSyncConnectors[0].FromPartitionID)
	require.Equal(t, spec.Partitions[1].ID, spec.PropSyncConnectors[0].ToPartitionID)

	t.Log("Reject a selected XT edge between valid but incompatible physical topologies")
	topology, err := parse("URSA_V2__Q8__16C__G_97_25__KP_FEC__GHZ_1_0__NO_FPGA")
	require.NoError(t, err)
	build.Partitions[2].Topology = topology
	_, err = appendModelProjections(nil, ModelProjectionInput{
		Pipeline: PipelineSingle, Models: []string{"default"}, BuildSnapshot: normalized,
		ModelSettings: json.RawMessage(`{"prop_sync":false}`),
	})
	require.ErrorContains(t, err, "has incompatible topology at partition ID 8")
}

func TestProjectModelV2PrioritizesSelectedChainWhenPropSyncIsEnabled(t *testing.T) {
	t.Parallel()
	normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	build := normalized.build
	build.CompilationMode = BuildCompilationModeHybrid

	t.Log("Use individually schedulable physical shapes whose collapsed runtime total is not a scheduler shape")
	firstTopology, err := build.Partitions[0].Topology.withChipCount(8)
	require.NoError(t, err)
	secondTopology, err := build.Partitions[1].Topology.withChipCount(64)
	require.NoError(t, err)
	build.Partitions[0].Topology = firstTopology
	build.Partitions[1].Topology = secondTopology
	third := build.Partitions[1]
	third.SourcePartitionID = 11
	third.PartPath = "part-11"
	build.Partitions = append(build.Partitions, third)
	build.SelectedPropSyncChains = [][]int{{7, 8}}

	t.Log("Project the selected chain beside an independent third partition")
	projection := projectTestBuild(t, normalized, PipelineLPX, "")

	t.Log("Project all physical scheduler partitions and only the selected connector")
	spec := projection.RequestSpec("test", "agents", nil)
	require.Len(t, spec.Partitions, 3)
	require.Len(t, spec.PropSyncConnectors, 1)
	require.Equal(t, spec.Partitions[0].ID, spec.PropSyncConnectors[0].FromPartitionID)
	require.Equal(t, spec.Partitions[1].ID, spec.PropSyncConnectors[0].ToPartitionID)

	t.Log("Collapse only the Agent runtime projection of the selected chain")
	data := resolvedPartitionData([]*ModelProjection{projection})
	require.Equal(t, "7\n11", data["partition_ids"])
	require.Equal(t, "9\n8", data["nodes_per_partition"])
	require.Equal(t, "0\n1", data["partition_indices"])
	require.Equal(t, "0\n9", data["partition_node_offsets"])
	require.Equal(t, "part-7\npart-11", data["partition_paths"])
	require.Equal(t, projection.configuredBuild.Partitions[0].Topology.Raw+"\n"+projection.configuredBuild.Partitions[1].Topology.Raw, data["topologies"])
	require.Empty(t, projection.configuredBuild.SelectedPropSyncChains)
	require.Equal(t, 17, projection.agentReplicas)
	require.Equal(t, [][]int{{7, 8}}, build.SelectedPropSyncChains)

	t.Log("Preserve physical output order when selected chains are declared in reverse")
	fourth := build.Partitions[2]
	fourth.SourcePartitionID = 13
	build.Partitions = append(build.Partitions, fourth)
	build.SelectedPropSyncChains = [][]int{{11, 13}, {7, 8}}
	connectors, err := v2Connectors(build, build.Partitions, true, maxPropSyncMultipleLoadSetsOffset)
	require.NoError(t, err)
	require.Len(t, connectors, 2)
	require.Equal(t, "partition-000", connectors[0].FromPartitionID)
	require.Equal(t, "partition-001", connectors[0].ToPartitionID)
	require.Equal(t, "partition-002", connectors[1].FromPartitionID)
	require.Equal(t, "partition-003", connectors[1].ToPartitionID)
	require.NotSame(t, connectors[0].Requirement.MaxInterPartitionOffset, connectors[1].Requirement.MaxInterPartitionOffset)
}

func TestProjectModelV2CollapsesSelectedChainAfterOmittingEmbeddingPrefix(t *testing.T) {
	t.Parallel()
	normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	build := normalized.build
	build.CompilationMode = BuildCompilationModeHybrid

	t.Log("Model the normalized adjacent 0/1/2 chain whose standalone embedding root is omitted")
	topology, err := build.Partitions[0].Topology.withChipCount(8)
	require.NoError(t, err)
	partitions := make([]BuildPartition, 3)
	for sourceID := range partitions {
		partitions[sourceID] = build.Partitions[0]
		partitions[sourceID].SourcePartitionID = sourceID
		partitions[sourceID].PartPath = fmt.Sprintf("part-%d", sourceID)
		partitions[sourceID].Topology = topology
	}
	build.Partitions = partitions
	build.SelectedPropSyncChains = [][]int{{0, 1, 2}}
	build.StandaloneTokenEmbeddings = true
	build.SupportsCPUEmbeddings = true

	t.Log("Project after omitting the standalone embedding prefix")
	projection := projectTestBuild(t, normalized, PipelineLPX, "")

	t.Log("Keep physical scheduler partitions 1/2 while collapsing their Agent runtime projection")
	spec := projection.RequestSpec("test", "agents", nil)
	require.Len(t, spec.Partitions, 2)
	require.Equal(t, int64(1), spec.Partitions[0].CompilerPartitionID)
	require.Equal(t, int64(2), spec.Partitions[1].CompilerPartitionID)
	require.Len(t, spec.PropSyncConnectors, 1)
	require.Equal(t, 2, projection.agentReplicas)
	data := resolvedPartitionData([]*ModelProjection{projection})
	require.Equal(t, "1", data["partition_ids"])
	require.Equal(t, "2", data["nodes_per_partition"])
	require.Equal(t, "part-1", data["partition_paths"])
	require.Equal(t, 16, projection.configuredBuild.Partitions[0].Topology.ChipCount)
	require.Empty(t, projection.configuredBuild.SelectedPropSyncChains)
	require.Equal(t, [][]int{{0, 1, 2}}, build.SelectedPropSyncChains)
}

func TestProjectModelV2PreservesAgentReplicasWhenCollapsingSubHostPartitions(t *testing.T) {
	t.Parallel()

	t.Log("Prepare two selected V2 sub-host partitions")
	normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	build := normalized.build
	build.CompilationMode = BuildCompilationModeHybrid

	for index := range build.Partitions {
		topology, err := build.Partitions[index].Topology.withChipCount(2)
		require.NoError(t, err)
		build.Partitions[index].Topology = topology
	}
	build.SelectedPropSyncChains = [][]int{{7, 8}}

	t.Log("Project and collapse the selected runtime chain")
	projection := projectTestBuild(t, normalized, PipelineLPX, "")

	t.Log("Preserve both physical scheduler partitions and Agent replicas")
	require.Len(t, projection.RequestSpec("test", "agents", nil).Partitions, 2)
	require.Equal(t, 2, projection.agentReplicas)

	t.Log("Keep both physical Agent endpoints in the collapsed C4 runtime partition")
	data := resolvedPartitionData([]*ModelProjection{projection})
	require.Equal(t, "2", data["nodes_per_partition"])
	require.Equal(t, "0", data["partition_node_offsets"])
	require.Equal(t, 2, projection.configuredBuild.Partitions[0].effectiveNodeCount())
	require.Equal(t, 4, projection.configuredBuild.Partitions[0].Topology.ChipCount)
}
