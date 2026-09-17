/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
)

func TestProjectModelV3HybridBuildProjectsSelectedPropSyncWithoutGlobalCoupling(t *testing.T) {
	t.Log("Create one LPX manifest with two selected LPU partitions and one CUDA artifact")
	fixture := newV3CompilerFixture()
	fixture.compilationMode = manifestcapnp.CompilationMode_lpx
	fixture.partitions[0].topology = "another-opaque-v3-topology"
	second := fixture.partitions[0]
	second.id = 2
	fixture.partitions = append(fixture.partitions, second, testV3CapnpPartition{
		id: 3, deviceType: manifestcapnp.DeviceType_cuda,
	})
	fixture.selectedPropSyncChains = [][]uint32{{1, 2}}
	buildDir := writeCompilerFixture(t, fixture)

	t.Log("Project hybrid while keeping manifest-selected links independent of global prop sync")
	intent := ModelProjectionInput{
		Pipeline:      PipelineLPX,
		Models:        []string{"default"},
		BuildSnapshot: normalizeTestSnapshot(t, acquireTestSnapshot(t, buildDir)),
	}
	projectionBatch, err := appendModelProjections(nil, intent)
	require.NoError(t, err)
	projection := projectionBatch[0]
	require.Equal(t, BuildCompilationModeHybrid, projection.configuredBuild.CompilationMode)
	require.Equal(t, int64(8192), projection.configuredBuild.runtimeSettings["sequence_length"])
	require.Equal(t, false, projection.configuredBuild.runtimeSettings["prop_sync"])
	digest, err := workloadSetDigest([]*ModelProjection{projection})
	require.NoError(t, err)
	require.Equal(t, projection.Digest(), digest, "one model is the aggregate digest base case")

	t.Log("Project only the two LPU artifacts and their selected chain into the hybrid scheduler request")
	cyborgRef := &lpxv1alpha1.PodCliqueReference{Name: "decode"}
	spec := projection.RequestSpec(&MaterializationPlan{CyborgClique: cyborgRef.Name}, "agents")
	require.Equal(t, lpxv1alpha1.WorkloadModeV3HxStrictHybrid, spec.WorkloadMode)
	require.Len(t, spec.Partitions, 2)
	require.Equal(t, int64(1), spec.Partitions[0].CompilerPartitionID)
	require.Equal(t, int64(2), spec.Partitions[1].CompilerPartitionID)
	require.Len(t, spec.PropSyncConnectors, 1)
	require.Equal(t, spec.Partitions[0].ID, spec.PropSyncConnectors[0].FromPartitionID)
	require.Equal(t, spec.Partitions[1].ID, spec.PropSyncConnectors[0].ToPartitionID)

	t.Log("Bind workload references while mapping compiler IDs to model ordinals")
	require.Equal(t, "agents", spec.NodeLocal.AgentPodCliqueRef.Name)
	require.Equal(t, "default", spec.NodeLocal.Model)
	require.Equal(t, cyborgRef, spec.CyborgPodCliqueRef)
	require.Len(t, spec.NodeLocal.PartitionMappings, 2)
	for index, partition := range spec.Partitions {
		require.Equal(t, int64(index), partition.Ordinal)
		require.Equal(t, int64(index), spec.NodeLocal.PartitionMappings[index].ModelPartitionID)
		require.Equal(t, partition.ID, spec.NodeLocal.PartitionMappings[index].PartitionID)
	}

	t.Log("Mutating returned fields must not change the projection or caller-owned reference")
	before := spec.DeepCopy()
	(*spec.Partitions[0].Extent)[0] = 0
	(*spec.PropSyncConnectors[0].Requirement.Connections)[0].FromLogicalDevice = 99
	(*spec.PropSyncConnectors[0].Requirement.AcceptableLaneMultiplicities)[0] = 99
	spec.AllocationMetadata.Raw[0] = ' '
	spec.CyborgPodCliqueRef.Name = "changed"
	require.Equal(t, before.CyborgPodCliqueRef, cyborgRef)
	require.Equal(t, *before, projection.RequestSpec(&MaterializationPlan{CyborgClique: cyborgRef.Name}, "agents"))
	require.Nil(t, projection.RequestSpec(&MaterializationPlan{}, "agents").CyborgPodCliqueRef)

	t.Log("Remove the manifest's selected chain without synthesizing hybrid connectors")
	fixture.selectedPropSyncChains = nil
	writeTestV3CapnpManifest(t, buildDir, fixture)
	intent.BuildSnapshot = normalizeTestSnapshot(t, acquireTestSnapshot(t, buildDir))
	projectionBatch, err = appendModelProjections(nil, intent)
	require.NoError(t, err)
	projection = projectionBatch[0]
	require.Empty(t, projection.RequestSpec(&MaterializationPlan{}, "agents").PropSyncConnectors)
}

func TestProjectModelV3RejectsInvalidPropSyncChains(t *testing.T) {
	t.Log("Define invalid selected chains on otherwise valid HX builds")
	tests := []struct {
		name        string
		partitions  int
		numLPUNodes uint32
		chains      [][]uint32
		wantErr     string
	}{
		{
			name:       "selected prop-sync chain references missing partition",
			partitions: 1, numLPUNodes: 2, chains: [][]uint32{{1, 2}},
			wantErr: "references missing partition ID 2",
		},
		{
			name:       "selected prop-sync chain is not forward-adjacent",
			partitions: 3, numLPUNodes: 6, chains: [][]uint32{{1, 3}},
			wantErr: "is not forward-adjacent at partition ID 3",
		},
		{
			name:       "missing edge evidence precedes overlap validation",
			partitions: 2, numLPUNodes: 4, chains: [][]uint32{{1, 1}, {1, 3}},
			wantErr: "selected V3 prop-sync chain 1 references missing partition ID 3",
		},
		{
			name:       "selected prop-sync chains overlap",
			partitions: 3, numLPUNodes: 6, chains: [][]uint32{{1, 2}, {2, 3}},
			wantErr: "overlaps partition ID 2",
		},
		{
			name:       "selected prop-sync chain repeats a partition",
			partitions: 1, numLPUNodes: 2, chains: [][]uint32{{1, 1}},
			wantErr: "overlaps partition ID 1",
		},
		{
			name:       "selected prop-sync chain is incomplete",
			partitions: 3, numLPUNodes: 6, chains: [][]uint32{{1, 2}},
			wantErr: "complete adjacent prop-sync connector chain",
		},
		{
			name:       "LPU-only partitions require a complete prop-sync chain",
			partitions: 2, numLPUNodes: 2,
			wantErr: "complete adjacent prop-sync connector chain",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Acquire and normalize the real manifest before testing its selected chain")
			fixture := newV3CompilerFixture()
			for id := 2; id <= test.partitions; id++ {
				partition := fixture.partitions[0]
				partition.id = uint32(id)
				fixture.partitions = append(fixture.partitions, partition)
			}
			fixture.numLPUNodes = test.numLPUNodes
			fixture.selectedPropSyncChains = test.chains
			normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeCompilerFixture(t, fixture)))

			t.Log("Reject the selected chain at the projector boundary")
			_, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineSingle, Models: []string{"default"}, BuildSnapshot: normalized,
			})
			require.ErrorContains(t, err, test.wantErr)
		})
	}
}

func TestProjectModelV3ProjectsSelectedPropSyncChain(t *testing.T) {
	t.Log("Create a two-partition V3 build with one selected adjacent prop-sync chain")
	fixture := newV3CompilerFixture()
	second := fixture.partitions[0]
	second.id = 2
	fixture.partitions = append(fixture.partitions, second)
	fixture.numLPUNodes = 4
	fixture.selectedPropSyncChains = [][]uint32{{1, 2}}
	buildDir := writeCompilerFixture(t, fixture)
	snapshot := acquireTestSnapshot(t, buildDir)
	intent := ModelProjectionInput{
		Pipeline: PipelineSingle, Models: []string{"default"}, BuildSnapshot: normalizeTestSnapshot(t, snapshot),
	}

	t.Log("Project the selected chain through the LPU-only runtime")
	projectionBatch, err := appendModelProjections(nil, intent)
	require.NoError(t, err)
	projection := projectionBatch[0]

	t.Log("Project both partitions and one HX prop-sync connector")
	spec := projection.RequestSpec(&MaterializationPlan{}, "agents")
	require.Len(t, spec.Partitions, 2)
	require.Len(t, spec.PropSyncConnectors, 1)
	connector := spec.PropSyncConnectors[0]
	require.Equal(t, spec.Partitions[0].ID, connector.FromPartitionID)
	require.Equal(t, spec.Partitions[1].ID, connector.ToPartitionID)
	require.Equal(t, lpxv1alpha1.PropSyncConnectorKindHxPropSyncV1, connector.Requirement.Kind)
	require.NotNil(t, connector.Requirement.AcceptableLaneMultiplicities)
	require.Equal(t, []int64{4, 2, 1}, *connector.Requirement.AcceptableLaneMultiplicities)
	require.NotNil(t, connector.Requirement.Connections)
	require.Len(t, *connector.Requirement.Connections, 16)
	for logicalDevice, connection := range *connector.Requirement.Connections {
		require.Equal(t, int64(logicalDevice), connection.FromLogicalDevice)
		require.Equal(t, int64(logicalDevice), connection.ToLogicalDevice)
	}

	t.Log("Encode the same connector topology in allocation metadata")
	require.JSONEq(t, `{
		"arch":"lp30",
		"topology":"lyra",
		"metadata_version":1,
		"partition_info":{
			"num_partitions":2,
			"1":{"device":"lpu","allocation":[16,1,1,1]},
			"2":{"device":"lpu","allocation":[16,1,1,1]}
		},
		"prop_sync_info":{"version":1,"prop_sync_pairs":[{
			"source_partition":1,
			"dest_partition":2,
			"connections":[[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15]],
			"num_supported_lanes":[4,2,1]
		}]}
	}`, string(spec.AllocationMetadata.Raw))

	t.Log("Reject model settings that disable a manifest-selected chain")
	intent.ModelSettings = json.RawMessage(`{"prop_sync":false}`)
	_, err = appendModelProjections(nil, intent)
	require.ErrorContains(t, err, "manifest-selected prop-sync chains require settings.prop_sync=true")

	t.Log("Reject global prop sync without manifest-selected evidence in an LPU-only build")
	fixture.selectedPropSyncChains = nil
	intent.BuildSnapshot = normalizeTestSnapshot(t, acquireTestSnapshot(t, writeCompilerFixture(t, fixture)))
	intent.ModelSettings = json.RawMessage(`{"prop_sync":true}`)
	_, err = appendModelProjections(nil, intent)
	require.ErrorContains(t, err, "settings.prop_sync=true requires a manifest-selected prop-sync chain")

	t.Log("Extend the native selected chain across three HX artifacts with distinct opaque topology names")
	third := fixture.partitions[0]
	third.id, third.topology = 3, "other-opaque-hx-topology"
	fixture.partitions = append(fixture.partitions, third)
	fixture.numLPUNodes = 6
	fixture.selectedPropSyncChains = [][]uint32{{1, 2, 3}}
	writeTestV3CapnpManifest(t, buildDir, fixture)
	intent.BuildSnapshot = normalizeTestSnapshot(t, acquireTestSnapshot(t, buildDir))
	intent.ModelSettings = nil
	projectionBatch, err = appendModelProjections(nil, intent)
	require.NoError(t, err)
	projection = projectionBatch[0]

	t.Log("Advance each native scheduler connector to the next physical partition")
	spec = projection.RequestSpec(&MaterializationPlan{}, "agents")
	require.Len(t, spec.PropSyncConnectors, 2)
	require.Equal(t, "partition-000", spec.PropSyncConnectors[0].FromPartitionID)
	require.Equal(t, "partition-001", spec.PropSyncConnectors[0].ToPartitionID)
	require.Equal(t, "partition-001", spec.PropSyncConnectors[1].FromPartitionID)
	require.Equal(t, "partition-002", spec.PropSyncConnectors[1].ToPartitionID)
}

func TestProjectModelV3UsesMultiNodePropSyncBoundary(t *testing.T) {
	t.Log("Create adjacent V3 partitions whose source boundary begins at logical device 16")
	fixture := newV3CompilerFixture()
	second := fixture.partitions[0]
	second.id = 2
	fixture.partitions[0].topology = v3HXTopologyFamily
	fixture.partitions[0].topologyFamily = v3HXTopologyFamily
	fixture.partitions[0].partitionShape = []uint32{16, 2, 1, 1}
	fixture.partitions[0].numChips = 32
	fixture.partitions = append(fixture.partitions, second)
	fixture.numLPUNodes = 3
	fixture.selectedPropSyncChains = [][]uint32{{1, 2}}
	buildDir := writeCompilerFixture(t, fixture)
	snapshot := acquireTestSnapshot(t, buildDir)

	t.Log("Project the multi-node prop-sync chain")
	projection := projectTestBuild(t, normalizeTestSnapshot(t, snapshot), PipelineSingle, "")

	t.Log("Project logical connections from the source partition's final node")
	spec := projection.RequestSpec(&MaterializationPlan{}, "agents")
	connections := *spec.PropSyncConnectors[0].Requirement.Connections
	require.Equal(t, lpxv1alpha1.HxLogicalConnection{FromLogicalDevice: 16, ToLogicalDevice: 0}, connections[0])
	require.Equal(t, lpxv1alpha1.HxLogicalConnection{FromLogicalDevice: 31, ToLogicalDevice: 15}, connections[15])
	require.Contains(t, string(spec.AllocationMetadata.Raw), `"connections":[[16,0]`)
}

func TestProjectModelV3UsesTopologyMetadataAndTracksManifestDigest(t *testing.T) {
	t.Log("Create and project a one-node V3 topology-metadata manifest")
	fixture := newV3CompilerFixture()
	fixture.partitions[0].topology = v3HXTopologyFamily
	fixture.partitions[0].topologyFamily = v3HXTopologyFamily
	fixture.partitions[0].partitionShape = []uint32{16, 1, 1, 1}
	fixture.numLPUNodes = 1
	buildDir := writeCompilerFixture(t, fixture)
	firstSnapshot := acquireTestSnapshot(t, buildDir)
	intent := ModelProjectionInput{
		Pipeline: PipelineSingle, Models: []string{"default"}, BuildSnapshot: normalizeTestSnapshot(t, firstSnapshot),
	}
	firstBatch, err := appendModelProjections(nil, intent)
	require.NoError(t, err)
	first := firstBatch[0]

	t.Log("Retain the acquired locator and exact single-partition allocation metadata")
	require.Equal(t, "file://"+buildDir, intent.BuildSnapshot.build.Path)
	require.Equal(t, intent.BuildSnapshot.build.Path, first.configuredBuild.Path)
	firstSpec := first.RequestSpec(&MaterializationPlan{}, "agents")
	require.JSONEq(t, `{
		"arch":"lp30",
		"topology":"lyra",
		"metadata_version":1,
		"partition_info":{
			"num_partitions":1,
			"1":{"device":"lpu","allocation":[16,1,1,1]}
		},
		"prop_sync_info":{"version":1,"prop_sync_pairs":[]}
	}`, string(firstSpec.AllocationMetadata.Raw))

	t.Log("Change the immutable manifest to a two-node topology and project again")
	fixture.partitions[0].partitionShape = []uint32{16, 2, 1, 1}
	fixture.partitions[0].numChips = 32
	fixture.numLPUNodes = 2
	writeTestV3CapnpManifest(t, buildDir, fixture)
	secondSnapshot := acquireTestSnapshot(t, buildDir)
	intent.BuildSnapshot = normalizeTestSnapshot(t, secondSnapshot)
	secondBatch, err := appendModelProjections(nil, intent)
	require.NoError(t, err)
	second := secondBatch[0]

	t.Log("Track the immutable manifest change in snapshot and projection digests")
	require.NotEqual(t, firstSnapshot.contentID, secondSnapshot.contentID)
	require.NotEqual(t, first.Digest(), second.Digest())
	secondSpec := second.RequestSpec(&MaterializationPlan{}, "agents")

	t.Log("Project each manifest extent and the updated runtime partition data")
	require.Equal(t, []int64{16, 1, 1, 1}, *firstSpec.Partitions[0].Extent)
	require.Equal(t, []int64{16, 2, 1, 1}, *secondSpec.Partitions[0].Extent)
	data := resolvedPartitionData([]*ModelProjection{second})
	require.Equal(t, "part-1", data["partition_paths"])
	require.Equal(t, v3HXTopologyFamily, data["topologies"])
	require.Empty(t, firstSpec.PropSyncConnectors)
	require.Empty(t, secondSpec.PropSyncConnectors)
}
