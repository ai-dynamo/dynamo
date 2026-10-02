/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
)

func TestProjectModelHXHybridBuildProjectsSelectedPropSyncWithoutGlobalCoupling(t *testing.T) {
	t.Log("Create one LPX manifest with two selected LPU partitions and one CUDA artifact")
	fixture := newV3CompilerFixture()
	fixture.compilationMode = manifestcapnp.CompilationMode_lpx
	second := fixture.partitions[0]
	second.id = 2
	fixture.partitions = append(fixture.partitions, second, testV3CapnpPartition{
		id: 3, deviceType: manifestcapnp.DeviceType_cuda,
	})
	fixture.selectedPropSyncChains = [][]uint32{{1, 2}}
	buildDir := writeCompilerFixture(t, fixture)

	t.Log("Project hybrid while keeping manifest-selected links independent of global prop sync")
	projection := projectTestModel(t, acquireTestSnapshot(t, buildDir), PipelineHybrid)
	require.Equal(t, compilationModeHybrid, projection.component.configuredBuild.compilationMode)
	digest, err := workloadSetDigest([]*Model{projection})
	require.NoError(t, err)
	require.Equal(t, projection.Digest(), digest, "one model is the aggregate digest base case")

	t.Log("Project only the two LPU artifacts and their selected chain into the hybrid scheduler request")
	workload := newTestWorkload(t, []*Model{projection}, 2, "test-pcs")
	spec := workload.RequestSpec(projection, 1)
	require.Equal(t, lpxv1alpha1.WorkloadModeV3HxStrictHybrid, spec.WorkloadMode)
	require.Len(t, spec.Partitions, 2)
	require.Equal(t, int64(1), spec.Partitions[0].CompilerPartitionID)
	require.Equal(t, int64(2), spec.Partitions[1].CompilerPartitionID)
	require.Len(t, spec.PropSyncConnectors, 1)
	require.Equal(t, spec.Partitions[0].ID, spec.PropSyncConnectors[0].FromPartitionID)
	require.Equal(t, spec.Partitions[1].ID, spec.PropSyncConnectors[0].ToPartitionID)

	t.Log("Bind the replica's Grove cliques while mapping compiler IDs to model ordinals")
	require.Equal(t, &lpxv1alpha1.PodCliqueScalingGroupReference{Name: "test-pcs-0-lpx", ReplicaIndex: 1},
		spec.MaterializationTarget.PodCliqueScalingGroupRef)
	require.Equal(t, "test-pcs-0-lpx-1-agt", spec.NodeLocal.AgentPodCliqueRef.Name)
	require.Equal(t, "default", spec.NodeLocal.Model)
	require.Equal(t, &lpxv1alpha1.PodCliqueReference{Name: "test-pcs-0-lpx-1-cond"}, spec.CyborgPodCliqueRef)
	require.Len(t, spec.NodeLocal.PartitionMappings, 2)
	for index, partition := range spec.Partitions {
		require.Equal(t, int64(index), partition.Ordinal)
		require.Equal(t, int64(index), spec.NodeLocal.PartitionMappings[index].ModelPartitionID)
		require.Equal(t, partition.ID, spec.NodeLocal.PartitionMappings[index].PartitionID)
	}

	t.Log("Mutating returned fields must not change the projection")
	before := spec.DeepCopy()
	(*spec.Partitions[0].Extent)[0] = 0
	(*spec.PropSyncConnectors[0].Requirement.Connections)[0].FromLogicalDevice = 99
	(*spec.PropSyncConnectors[0].Requirement.AcceptableLaneMultiplicities)[0] = 99
	spec.AllocationMetadata.Raw[0] = ' '
	spec.CyborgPodCliqueRef.Name = "changed"
	require.Equal(t, *before, workload.RequestSpec(projection, 1))
	require.Nil(t, projection.requestSpec().CyborgPodCliqueRef)

	t.Log("Remove the manifest's selected chain without synthesizing hybrid connectors")
	fixture.selectedPropSyncChains = nil
	writeTestV3CapnpManifest(t, buildDir, fixture)
	projection = projectTestModel(t, acquireTestSnapshot(t, buildDir), PipelineHybrid)
	require.Empty(t, projection.requestSpec().PropSyncConnectors)
}

func TestProjectModelHXRejectsInvalidPropSyncChains(t *testing.T) {
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
			normalized := acquireTestSnapshot(t, writeCompilerFixture(t, fixture))

			t.Log("Reject the selected chain at the projector boundary")
			_, err := projectComponent(testRenderComponentName, "", normalized, PipelineSingle, []string{"default"})
			require.ErrorContains(t, err, test.wantErr)
		})
	}
}

func TestProjectModelHXProjectsSelectedPropSyncChain(t *testing.T) {
	t.Log("Create a two-partition HX build with one selected adjacent prop-sync chain")
	fixture := newV3CompilerFixture()
	second := fixture.partitions[0]
	second.id = 2
	fixture.partitions = append(fixture.partitions, second)
	fixture.numLPUNodes = 4
	fixture.selectedPropSyncChains = [][]uint32{{1, 2}}
	buildDir := writeCompilerFixture(t, fixture)

	t.Log("Project the selected chain through the LPU-only runtime")
	projection := projectTestModel(t, acquireTestSnapshot(t, buildDir), PipelineSingle)

	t.Log("Project both partitions and one HX prop-sync connector")
	spec := projection.requestSpec()
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

	t.Log("Extend the native selected chain across three HX artifacts")
	third := fixture.partitions[0]
	third.id = 3
	fixture.partitions = append(fixture.partitions, third)
	fixture.numLPUNodes = 6
	fixture.selectedPropSyncChains = [][]uint32{{1, 2, 3}}
	writeTestV3CapnpManifest(t, buildDir, fixture)
	projection = projectTestModel(t, acquireTestSnapshot(t, buildDir), PipelineSingle)

	t.Log("Advance each native scheduler connector to the next physical partition")
	spec = projection.requestSpec()
	require.Len(t, spec.PropSyncConnectors, 2)
	require.Equal(t, "partition-000", spec.PropSyncConnectors[0].FromPartitionID)
	require.Equal(t, "partition-001", spec.PropSyncConnectors[0].ToPartitionID)
	require.Equal(t, "partition-001", spec.PropSyncConnectors[1].FromPartitionID)
	require.Equal(t, "partition-002", spec.PropSyncConnectors[1].ToPartitionID)
}

func TestProjectModelHXUsesMultiNodePropSyncBoundary(t *testing.T) {
	t.Log("Create adjacent HX partitions whose source boundary begins at logical device 16")
	fixture := newV3CompilerFixture()
	second := fixture.partitions[0]
	second.id = 2
	fixture.partitions[0].topologyFamily = hxTopologyFamily
	fixture.partitions[0].partitionShape = []uint32{16, 2, 1, 1}
	fixture.partitions[0].numChips = 32
	fixture.partitions = append(fixture.partitions, second)
	fixture.numLPUNodes = 3
	fixture.selectedPropSyncChains = [][]uint32{{1, 2}}
	buildDir := writeCompilerFixture(t, fixture)
	snapshot := acquireTestSnapshot(t, buildDir)

	t.Log("Project the multi-node prop-sync chain")
	projection := projectTestModel(t, snapshot, PipelineSingle)

	t.Log("Project logical connections from the source partition's final node")
	spec := projection.requestSpec()
	connections := *spec.PropSyncConnectors[0].Requirement.Connections
	require.Equal(t, lpxv1alpha1.HxLogicalConnection{FromLogicalDevice: 16, ToLogicalDevice: 0}, connections[0])
	require.Equal(t, lpxv1alpha1.HxLogicalConnection{FromLogicalDevice: 31, ToLogicalDevice: 15}, connections[15])
	require.Contains(t, string(spec.AllocationMetadata.Raw), `"connections":[[16,0]`)
}

func TestProjectModelHXUsesTopologyMetadataAndTracksManifestDigest(t *testing.T) {
	t.Log("Create and project a one-node HX topology-metadata manifest")
	fixture := newV3CompilerFixture()
	fixture.partitions[0].topologyFamily = hxTopologyFamily
	fixture.partitions[0].partitionShape = []uint32{16, 1, 1, 1}
	fixture.numLPUNodes = 1
	buildDir := writeCompilerFixture(t, fixture)
	firstSnapshot := acquireTestSnapshot(t, buildDir)
	first := projectTestModel(t, firstSnapshot, PipelineSingle)

	t.Log("Retain the acquired locator and exact single-partition allocation metadata")
	require.Equal(t, "file://"+buildDir, firstSnapshot.path)
	require.Equal(t, firstSnapshot.path, first.component.configuredBuild.path)
	firstSpec := first.requestSpec()
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
	second := projectTestModel(t, secondSnapshot, PipelineSingle)

	t.Log("Track the immutable manifest change in snapshot and projection digests")
	require.NotEqual(t, firstSnapshot.contentID, secondSnapshot.contentID)
	require.NotEqual(t, first.Digest(), second.Digest())
	secondSpec := second.requestSpec()

	t.Log("Project each manifest extent and the updated runtime partition data")
	require.Equal(t, []int64{16, 1, 1, 1}, *firstSpec.Partitions[0].Extent)
	require.Equal(t, []int64{16, 2, 1, 1}, *secondSpec.Partitions[0].Extent)
	data := resolvedPartitionData([]*Model{second})
	require.Equal(t, "part-1", data["partition_paths"])
	require.Empty(t, firstSpec.PropSyncConnectors)
	require.Empty(t, secondSpec.PropSyncConnectors)
}
