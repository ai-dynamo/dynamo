/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
)

// projectTestUnits projects one model of build, choosing the pipeline from its
// CUDA partitions as workload resolution does.
func projectTestUnits(build *Build, selection *dynamov1beta1.LPXLocalPartitions) ([]*ModelProjection, error) {
	pipeline := PipelineSingle
	if build.hasCUDAPartitions {
		pipeline = PipelineHybrid
	}
	return appendModelProjections(nil, ModelProjectionInput{
		Pipeline: pipeline, Models: []string{"default"}, RuntimeBuildRef: "model-build",
		Build: build, LocalPartitions: selection,
	})
}

func TestUnitProjection(t *testing.T) {
	t.Parallel()
	const (
		lp20 = deploymentcapnpv1.LpuArchitecture_lp20
		lp30 = deploymentcapnpv1.LpuArchitecture_lp30
	)
	lp20Hybrid := testUnitBundle{
		architecture: lp20, devicesPerNode: 8, hybrid: true, chains: [][]uint32{{1, 2}},
		partitions: testUnitPartitions(16, nil, 0, 1, 2, 3),
		units: []testLPUUnit{
			{reservations: []testLPUReservation{testNodesReservation(2, 0)}},
			{reservations: []testLPUReservation{testNodesReservation(2, 1), testNodesReservation(2, 2)}, connectors: []testLPUConnector{{}}},
			{reservations: []testLPUReservation{testNodesReservation(2, 3)}},
		},
	}
	xtGap := lpxv1alpha1.PropSyncConnectorRequirement{Kind: lpxv1alpha1.PropSyncConnectorKindXt8888Gap, MaxInterPartitionOffset: new(int64(0))}
	for _, test := range []struct {
		name           string
		bundle         testUnitBundle
		selection      *dynamov1beta1.LPXLocalPartitions
		wantErr        string
		wantPartitions []lpxv1alpha1.PartitionRequest
		wantConnectors []lpxv1alpha1.PropSyncConnectorRequest
		wantAgents     int
		wantLocalIDs   []int
		// wantRows lists the /configs rows of an LPU-only projection, which a
		// speculative workload renders when its other build lacks units.
		wantRows map[string]string
	}{
		{
			name: "LP20 chain joined by an xt8888Gap connector",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, chains: [][]uint32{{1, 2}}, partitions: testUnitPartitions(8, nil, 1, 2),
				units: []testLPUUnit{{reservations: []testLPUReservation{testNodesReservation(1, 1), testNodesReservation(1, 2)}, connectors: []testLPUConnector{{}}}},
			},
			wantPartitions: []lpxv1alpha1.PartitionRequest{
				{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 1, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC8)},
				{ID: "partition-001", Ordinal: 1, CompilerPartitionID: 2, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC8)},
			},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{{FromPartitionID: "partition-000", ToPartitionID: "partition-001", Requirement: xtGap}},
			wantAgents:     2,
			wantRows:       map[string]string{"nodes_per_partition": "1\n1", "partition_ids": "1\n2", "partition_node_offsets": "0\n1", "partition_paths": "part-1\npart-2"},
		},
		{
			name: "LP20 packed chain requests every chip of its hosts",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, chains: [][]uint32{{1, 2, 3, 4, 5, 6}}, partitions: testUnitPartitions(4, nil, 1, 2, 3, 4, 5, 6),
				units: testSingleUnits(testNodesReservation(3, 1, 2, 3, 4, 5, 6)),
			},
			wantPartitions: []lpxv1alpha1.PartitionRequest{{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 1, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC24)}},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{},
			wantAgents:     3,
			wantRows: map[string]string{
				"nodes_per_partition": "1\n1\n1\n1\n1\n1", "partition_ids": "1\n2\n3\n4\n5\n6",
				"partition_node_offsets": "0\n1\n2\n3\n4\n5", "partition_paths": "part-1\npart-2\npart-3\npart-4\npart-5\npart-6",
			},
		},
		{
			name: "LPU-only CPU embeddings keep partition 0 off LPUs",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, cpuEmbeddings: true, chains: [][]uint32{{1, 2}}, partitions: testUnitPartitions(8, nil, 0, 1, 2),
				units: []testLPUUnit{{reservations: []testLPUReservation{testNodesReservation(1, 1), testNodesReservation(1, 2)}, connectors: []testLPUConnector{{}}}},
			},
			wantPartitions: []lpxv1alpha1.PartitionRequest{
				{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 1, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC8)},
				{ID: "partition-001", Ordinal: 1, CompilerPartitionID: 2, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC8)},
			},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{{FromPartitionID: "partition-000", ToPartitionID: "partition-001", Requirement: xtGap}},
			wantAgents:     2,
			wantRows:       map[string]string{"nodes_per_partition": "1\n1", "partition_ids": "1\n2", "partition_node_offsets": "0\n1", "partition_paths": "part-1\npart-2"},
		},
		{
			name: "LP30 single partition requests its HX extent",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(32, []uint32{16, 1, 2, 1}, 1),
				units: testSingleUnits(testHXReservation(1, 2, 1, 1)),
			},
			wantPartitions: []lpxv1alpha1.PartitionRequest{{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 1, Extent: &[]int64{16, 1, 2, 1}}},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{},
			wantAgents:     2,
			wantRows:       map[string]string{"nodes_per_partition": "2", "partition_ids": "1", "partition_node_offsets": "0", "partition_paths": "part-1"},
		},
		{
			name: "LP30 partial selection requests its one partition",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partSelect: []uint32{3}, partitions: testUnitPartitions(16, []uint32{16, 1, 1, 1}, 3),
				units: testSingleUnits(testHXReservation(1, 1, 1, 3)),
			},
			wantPartitions: []lpxv1alpha1.PartitionRequest{{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 3, Extent: &[]int64{16, 1, 1, 1}}},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{},
			wantAgents:     1,
			wantRows:       map[string]string{"nodes_per_partition": "1", "partition_ids": "3", "partition_node_offsets": "0", "partition_paths": "part-3"},
		},
		{
			name: "LP30 packed chain requests the packed hosts' extent",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, chains: [][]uint32{{1, 2, 3, 4}}, partitions: testUnitPartitions(8, []uint32{8, 1, 1, 1}, 1, 2, 3, 4),
				units: testSingleUnits(testHXReservation(2, 1, 1, 1, 2, 3, 4)),
			},
			wantPartitions: []lpxv1alpha1.PartitionRequest{{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 1, Extent: &[]int64{16, 2, 1, 1}}},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{},
			wantAgents:     2,
			wantRows:       map[string]string{"nodes_per_partition": "1\n1\n1\n1", "partition_ids": "1\n2\n3\n4", "partition_node_offsets": "0\n1\n2\n3", "partition_paths": "part-1\npart-2\npart-3\npart-4"},
		},
		{
			name: "LP30 chain carries the compiler's device pairs",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, chains: [][]uint32{{1, 2}},
				partitions: []testUnitPartition{{id: 1, numChips: 32, shape: []uint32{16, 1, 2, 1}}, {id: 2, numChips: 16, shape: []uint32{16, 1, 1, 1}}},
				units: []testLPUUnit{{
					reservations: []testLPUReservation{testHXReservation(1, 2, 1, 1), testHXReservation(1, 1, 1, 2)},
					connectors:   []testLPUConnector{{hx: true, pairs: [][2]uint32{{16, 0}, {31, 15}}, lanes: []uint8{4, 2}, dimension: 1}},
				}},
			},
			wantPartitions: []lpxv1alpha1.PartitionRequest{
				{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 1, Extent: &[]int64{16, 1, 2, 1}},
				{ID: "partition-001", Ordinal: 1, CompilerPartitionID: 2, Extent: &[]int64{16, 1, 1, 1}},
			},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{{
				FromPartitionID: "partition-000", ToPartitionID: "partition-001",
				Requirement: lpxv1alpha1.PropSyncConnectorRequirement{
					Kind:                         lpxv1alpha1.PropSyncConnectorKindHxPropSyncV1,
					Connections:                  &[]lpxv1alpha1.HxLogicalConnection{{FromLogicalDevice: 16, ToLogicalDevice: 0}, {FromLogicalDevice: 31, ToLogicalDevice: 15}},
					AcceptableLaneMultiplicities: &[]int64{4, 2},
				},
			}},
			wantAgents: 3,
			wantRows:   map[string]string{"nodes_per_partition": "2\n1", "partition_ids": "1\n2", "partition_node_offsets": "0\n2", "partition_paths": "part-1\npart-2"},
		},
		{
			name:      "LP20 hybrid keeps remote units",
			bundle:    lp20Hybrid,
			selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{0}},
			wantPartitions: []lpxv1alpha1.PartitionRequest{
				{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 1, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC16)},
				{ID: "partition-001", Ordinal: 1, CompilerPartitionID: 2, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC16)},
				{ID: "partition-002", Ordinal: 2, CompilerPartitionID: 3, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC16)},
			},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{{FromPartitionID: "partition-000", ToPartitionID: "partition-001", Requirement: xtGap}},
			wantAgents:     6,
			wantLocalIDs:   []int{0},
		},
		{
			name:      "LP20 hybrid runs a selected chain root on the GPU",
			bundle:    lp20Hybrid,
			selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{1}},
			wantPartitions: []lpxv1alpha1.PartitionRequest{
				{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 0, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC16)},
				{ID: "partition-001", Ordinal: 1, CompilerPartitionID: 3, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC16)},
			},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{},
			wantAgents:     4,
			wantLocalIDs:   []int{1},
		},
		{
			name:         "every unit on the GPU leaves no Agents",
			bundle:       lp20Hybrid,
			selection:    &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeAll},
			wantAgents:   0,
			wantLocalIDs: []int{0, 1, 3},
		},
		{
			name:      "chain member is not a unit root",
			bundle:    lp20Hybrid,
			selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{2}},
			wantErr:   "localPartitions references partition 2 of the prop-sync chain that starts at partition 1; select 1",
		},
		{
			name:      "local partition outside every unit",
			bundle:    lp20Hybrid,
			selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{4}},
			wantErr:   "localPartitions references partition 4, which the build does not contain",
		},
		{
			name: "LP30 hybrid keeps one reservation per chain member",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, hybrid: true, chains: [][]uint32{{2, 3}},
				partitions: testUnitPartitions(16, []uint32{16, 1, 1, 1}, 1, 2, 3, 4),
				units: []testLPUUnit{
					{reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1)}},
					{
						reservations: []testLPUReservation{testHXReservation(1, 1, 1, 2), testHXReservation(1, 1, 1, 3)},
						connectors:   []testLPUConnector{{hx: true, pairs: [][2]uint32{{0, 0}, {1, 1}}, lanes: []uint8{4, 2, 1}, dimension: 1}},
					},
					{reservations: []testLPUReservation{testHXReservation(1, 1, 1, 4)}},
				},
			},
			selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{1}},
			wantPartitions: []lpxv1alpha1.PartitionRequest{
				{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 2, Extent: &[]int64{16, 1, 1, 1}},
				{ID: "partition-001", Ordinal: 1, CompilerPartitionID: 3, Extent: &[]int64{16, 1, 1, 1}},
				{ID: "partition-002", Ordinal: 2, CompilerPartitionID: 4, Extent: &[]int64{16, 1, 1, 1}},
			},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{{
				FromPartitionID: "partition-000", ToPartitionID: "partition-001",
				Requirement: lpxv1alpha1.PropSyncConnectorRequirement{
					Kind:                         lpxv1alpha1.PropSyncConnectorKindHxPropSyncV1,
					Connections:                  &[]lpxv1alpha1.HxLogicalConnection{{FromLogicalDevice: 0, ToLogicalDevice: 0}, {FromLogicalDevice: 1, ToLogicalDevice: 1}},
					AcceptableLaneMultiplicities: &[]int64{4, 2, 1},
				},
			}},
			wantAgents:   3,
			wantLocalIDs: []int{1},
		},
		{
			name: "LP20 LPU-only partitions without a chain",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(8, nil, 1, 2),
				units: testSingleUnits(testNodesReservation(1, 1), testNodesReservation(1, 2)),
			},
			wantErr: "unsupported LPX runtime: LPU-only workloads require exactly one lpuUnits entry per model; this bundle has 2",
		},
		{
			name: "LP30 LPU-only partitions without a chain",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(16, []uint32{16, 1, 1, 1}, 1, 2),
				units: testSingleUnits(testHXReservation(1, 1, 1, 1), testHXReservation(1, 1, 1, 2)),
			},
			wantErr: "unsupported LPX runtime: LPU-only workloads require exactly one lpuUnits entry per model; this bundle has 2",
		},
		{
			name: "LPU-only local partitions",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(8, nil, 1),
				units: testSingleUnits(testNodesReservation(1, 1)),
			},
			selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{1}},
			wantErr:   "unsupported LPX runtime: localPartitions requires a hybrid build with a Cyborg conductor",
		},
		{
			name: "LP20 hosts without a registered XT shape",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(72, nil, 1),
				units: testSingleUnits(testNodesReservation(9, 1)),
			},
			wantErr: "partition 1 has unregistered XT shape c72",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Decode the units and project the build for its pipeline")
			build, err := acquireTestUnitBundle(t, test.bundle)
			require.NoError(t, err)
			projections, err := projectTestUnits(build, test.selection)
			if test.wantErr != "" {
				require.EqualError(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
			projection := projections[0]

			t.Log("Request one LPX partition per kept reservation with the units' connectors and no compiler envelope")
			spec := projection.RequestSpec(&MaterializationPlan{}, "agents")
			if test.wantPartitions == nil {
				test.wantPartitions = []lpxv1alpha1.PartitionRequest{}
			}
			if test.wantConnectors == nil {
				test.wantConnectors = []lpxv1alpha1.PropSyncConnectorRequest{}
			}
			require.Equal(t, test.wantPartitions, spec.Partitions)
			require.Equal(t, test.wantConnectors, spec.PropSyncConnectors)
			require.JSONEq(t, `{}`, string(spec.AllocationMetadata.Raw))
			require.Equal(t, test.wantAgents, projection.AgentReplicas())
			require.Equal(t, test.wantLocalIDs, projection.localPartitionIDs)

			t.Log("Render legacy /configs rows from LPU-only units for a workload that still reads them")
			if !test.bundle.hybrid {
				require.Equal(t, test.wantRows, resolvedPartitionData(projections))
			}
		})
	}
}

func TestUnitProjectionBounds(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name           string
		devicesPerNode uint32
		units          int
		chain          bool
		wantErr        string
	}{
		// The legacy projection caps a request at 8192 device pairs to bound its
		// compiler envelope; units send none, and the LPR CRD bounds connectors.
		{name: "HX connectors beyond the legacy device-pair cap", devicesPerNode: 4097, units: 3, chain: true},
		{name: "most LPX partitions", devicesPerNode: 16, units: maxLPXPartitions},
		{name: "too many LPX partitions", devicesPerNode: 16, units: maxLPXPartitions + 1, wantErr: "LPX projection has 257 partitions, limit is 1..256"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Place one-host LP30 partitions in single units or one chain joined device by device")
			bundle := testUnitBundle{architecture: deploymentcapnpv1.LpuArchitecture_lp30, devicesPerNode: test.devicesPerNode, hybrid: true}
			chain := testLPUUnit{}
			for id := range uint32(test.units) {
				bundle.partitions = append(bundle.partitions, testUnitPartition{id: id, numChips: test.devicesPerNode, shape: []uint32{test.devicesPerNode, 1, 1, 1}})
				if !test.chain {
					bundle.units = append(bundle.units, testSingleUnits(testHXReservation(1, 1, 1, id))...)
					continue
				}
				chain.reservations = append(chain.reservations, testHXReservation(1, 1, 1, id))
				if id > 0 {
					chain.connectors = append(chain.connectors, testHXConventionConnector(1, test.devicesPerNode))
				}
			}
			if test.chain {
				bundle.units = []testLPUUnit{chain}
				bundle.chains = [][]uint32{make([]uint32, test.units)}
				for id := range bundle.chains[0] {
					bundle.chains[0][id] = uint32(id)
				}
			}
			build, err := acquireTestUnitBundle(t, bundle)
			require.NoError(t, err)

			t.Log("Bound the request's partitions, not its device pairs")
			projections, err := projectTestUnits(build, nil)
			if test.wantErr != "" {
				require.EqualError(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
			require.Len(t, projections[0].partitions, test.units)
		})
	}
}

func TestUnitProjectionDigest(t *testing.T) {
	t.Parallel()

	t.Log("Decode an LP30 chain and the same chain with different compiler device pairs")
	bundle := testUnitBundle{
		architecture: deploymentcapnpv1.LpuArchitecture_lp30, devicesPerNode: 16, hybrid: true, chains: [][]uint32{{1, 2}},
		partitions: testUnitPartitions(16, []uint32{16, 1, 1, 1}, 1, 2, 3),
		units: []testLPUUnit{
			{reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1), testHXReservation(1, 1, 1, 2)}, connectors: []testLPUConnector{testHXConventionConnector(1, 16)}},
			{reservations: []testLPUReservation{testHXReservation(1, 1, 1, 3)}},
		},
	}
	build, err := acquireTestUnitBundle(t, bundle)
	require.NoError(t, err)
	bundle.units[0].connectors = []testLPUConnector{{hx: true, pairs: [][2]uint32{{0, 0}}, lanes: []uint8{4, 2, 1}, dimension: 1}}
	otherPairs, err := acquireTestUnitBundle(t, bundle)
	require.NoError(t, err)
	legacy := *build
	legacy.lpuUnits = nil

	t.Log("Project units, other units, a local selection, and the legacy derivation")
	projected, err := projectTestUnits(build, nil)
	require.NoError(t, err)
	repeated, err := projectTestUnits(build, nil)
	require.NoError(t, err)
	otherProjected, err := projectTestUnits(otherPairs, nil)
	require.NoError(t, err)
	local, err := projectTestUnits(build, &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{3}})
	require.NoError(t, err)
	legacyProjected, err := projectTestUnits(&legacy, nil)
	require.NoError(t, err)

	t.Log("Bind the units and the local selection into a digest distinct from the legacy derivation")
	digest := projected[0].Digest()
	require.Equal(t, digest, repeated[0].Digest())
	require.NotEqual(t, digest, otherProjected[0].Digest())
	require.NotEqual(t, digest, local[0].Digest())
	require.NotEqual(t, digest, legacyProjected[0].Digest())
}

// requireLegacyParity projects build through its units and through the legacy
// derivation with the units ignored. Both must request the same LPX partitions
// and connectors, size the same Agents, and select the same GPU-local partitions.
// LPU-only projections must also render the same /configs rows, which a
// speculative workload still reads when its other build lacks units. Only HX
// allocation metadata differs: units send none.
func requireLegacyParity(t *testing.T, build *Build, selection *dynamov1beta1.LPXLocalPartitions) {
	t.Helper()
	require.NotNil(t, build.lpuUnits)
	legacyBuild := *build
	legacyBuild.lpuUnits = nil

	unitProjections, err := projectTestUnits(build, selection)
	require.NoError(t, err)
	legacyProjections, err := projectTestUnits(&legacyBuild, selection)
	require.NoError(t, err)
	unit, legacy := unitProjections[0], legacyProjections[0]

	unitSpec := unit.RequestSpec(&MaterializationPlan{}, "agents")
	legacySpec := legacy.RequestSpec(&MaterializationPlan{}, "agents")
	require.JSONEq(t, `{}`, string(unitSpec.AllocationMetadata.Raw))
	if build.family == BuildFamilyHX {
		legacySpec.AllocationMetadata = unitSpec.AllocationMetadata
	}
	require.Equal(t, legacySpec, unitSpec)
	require.Equal(t, legacy.AgentReplicas(), unit.AgentReplicas())
	require.Equal(t, legacy.localPartitionIDs, unit.localPartitionIDs)
	if !build.hasCUDAPartitions {
		require.Equal(t, resolvedPartitionData(legacyProjections), resolvedPartitionData(unitProjections))
	}
}

func TestUnitProjectionMatchesLegacy(t *testing.T) {
	t.Parallel()
	const (
		lp20 = deploymentcapnpv1.LpuArchitecture_lp20
		lp30 = deploymentcapnpv1.LpuArchitecture_lp30
	)
	allLocal := &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeAll}
	lp20Singles := testUnitBundle{architecture: lp20, devicesPerNode: 8, hybrid: true, partitions: testUnitPartitions(8, nil, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)}
	for id := range uint32(17) {
		lp20Singles.units = append(lp20Singles.units, testSingleUnits(testNodesReservation(1, id))...)
	}
	lp20Chain := testUnitBundle{
		architecture: lp20, devicesPerNode: 8, hybrid: true, chains: [][]uint32{{1, 2}}, partitions: testUnitPartitions(16, nil, 0, 1, 2, 3),
		units: []testLPUUnit{
			{reservations: []testLPUReservation{testNodesReservation(2, 0)}},
			{reservations: []testLPUReservation{testNodesReservation(2, 1), testNodesReservation(2, 2)}, connectors: []testLPUConnector{{}}},
			{reservations: []testLPUReservation{testNodesReservation(2, 3)}},
		},
	}
	lp30Chain := testUnitBundle{
		architecture: lp30, devicesPerNode: 16, hybrid: true, chains: [][]uint32{{2, 3}}, partitions: testUnitPartitions(16, []uint32{16, 1, 1, 1}, 1, 2, 3, 4),
		units: []testLPUUnit{
			{reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1)}},
			{reservations: []testLPUReservation{testHXReservation(1, 1, 1, 2), testHXReservation(1, 1, 1, 3)}, connectors: []testLPUConnector{testHXConventionConnector(1, 16)}},
			{reservations: []testLPUReservation{testHXReservation(1, 1, 1, 4)}},
		},
	}
	lp30Singles := testUnitBundle{architecture: lp30, devicesPerNode: 16, hybrid: true, partitions: testUnitPartitions(16, []uint32{16, 1, 1, 1}, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)}
	for id := range uint32(14) {
		lp30Singles.units = append(lp30Singles.units, testSingleUnits(testHXReservation(1, 1, 1, id))...)
	}
	for _, test := range []struct {
		name      string
		bundle    testUnitBundle
		selection *dynamov1beta1.LPXLocalPartitions
	}{
		{
			name: "LP20 LPU-only chain after a CPU embedding partition",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, cpuEmbeddings: true, chains: [][]uint32{{1, 2}}, partitions: testUnitPartitions(8, nil, 0, 1, 2),
				units: []testLPUUnit{{reservations: []testLPUReservation{testNodesReservation(1, 1), testNodesReservation(1, 2)}, connectors: []testLPUConnector{{}}}},
			},
		},
		{
			name: "LP20 LPU-only partition across two hosts",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(16, nil, 1),
				units: testSingleUnits(testNodesReservation(2, 1)),
			},
		},
		{
			name: "LP20 LPU-only partition smaller than a node",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(4, nil, 1),
				units: testSingleUnits(testNodesReservation(1, 1)),
			},
		},
		{
			name: "LP20 four two-chip stages packed on one host",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, chains: [][]uint32{{1, 2, 3, 4}}, partitions: testUnitPartitions(2, nil, 1, 2, 3, 4),
				units: testSingleUnits(testNodesReservation(1, 1, 2, 3, 4)),
			},
		},
		{
			name: "LP20 six four-chip stages packed on three hosts",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, chains: [][]uint32{{1, 2, 3, 4, 5, 6}}, partitions: testUnitPartitions(4, nil, 1, 2, 3, 4, 5, 6),
				units: testSingleUnits(testNodesReservation(3, 1, 2, 3, 4, 5, 6)),
			},
		},
		{
			name: "LP20 packed stages leave the last host partly used",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, chains: [][]uint32{{1, 2, 3}}, partitions: testUnitPartitions(4, nil, 1, 2, 3),
				units: testSingleUnits(testNodesReservation(2, 1, 2, 3)),
			},
		},
		{name: "LP20 hybrid single units", bundle: lp20Singles},
		{name: "LP20 hybrid single units with local partitions", bundle: lp20Singles, selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{0, 15, 16}}},
		{name: "LP20 hybrid chain", bundle: lp20Chain},
		{name: "LP20 hybrid chain with partition 0 local", bundle: lp20Chain, selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{0}}},
		{name: "LP20 hybrid chain local", bundle: lp20Chain, selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{1}}},
		{name: "LP20 hybrid all local", bundle: lp20Chain, selection: allLocal},
		{
			name: "LP30 LPU-only sixteen-chip partition",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(16, []uint32{16, 1, 1, 1}, 1),
				units: testSingleUnits(testHXReservation(1, 1, 1, 1)),
			},
		},
		{
			name: "LP30 LPU-only subnode partition",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(8, []uint32{8, 1, 1, 1}, 1),
				units: testSingleUnits(testHXReservation(1, 1, 1, 1)),
			},
		},
		{
			name: "LP30 one-tray partition without topology metadata",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(8, nil, 1),
				units: testSingleUnits(testHXReservation(1, 1, 1, 1)),
			},
		},
		{
			name: "LP30 packed chain without topology metadata",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, chains: [][]uint32{{1, 2, 3, 4}}, partitions: testUnitPartitions(8, nil, 1, 2, 3, 4),
				units: testSingleUnits(testHXReservation(2, 1, 1, 1, 2, 3, 4)),
			},
		},
		{
			name: "LP30 four four-chip stages packed on one host",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, chains: [][]uint32{{1, 2, 3, 4}}, partitions: testUnitPartitions(4, []uint32{4, 1, 1, 1}, 1, 2, 3, 4),
				units: testSingleUnits(testHXReservation(1, 1, 1, 1, 2, 3, 4)),
			},
		},
		{
			name: "LP30 four eight-chip stages packed on two hosts",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, chains: [][]uint32{{1, 2, 3, 4}}, partitions: testUnitPartitions(8, []uint32{8, 1, 1, 1}, 1, 2, 3, 4),
				units: testSingleUnits(testHXReservation(2, 1, 1, 1, 2, 3, 4)),
			},
		},
		{
			name: "LP30 LPU-only chain of two-tray partitions",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, chains: [][]uint32{{1, 2}}, partitions: testUnitPartitions(32, []uint32{16, 1, 2, 1}, 1, 2),
				units: []testLPUUnit{{
					reservations: []testLPUReservation{testHXReservation(1, 2, 1, 1), testHXReservation(1, 2, 1, 2)},
					connectors:   []testLPUConnector{testHXConventionConnector(2, 16)},
				}},
			},
		},
		{name: "LP30 hybrid chain", bundle: lp30Chain},
		{name: "LP30 hybrid chain with partition 1 local", bundle: lp30Chain, selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{1}}},
		{name: "LP30 hybrid chain local", bundle: lp30Chain, selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{2}}},
		{name: "LP30 hybrid all local", bundle: lp30Chain, selection: allLocal},
		{name: "LP30 hybrid fourteen single units", bundle: lp30Singles},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Decode the synthetic bundle and compare unit placement with the legacy derivation")
			build, err := acquireTestUnitBundle(t, test.bundle)
			require.NoError(t, err)
			requireLegacyParity(t, build, test.selection)
		})
	}
}
