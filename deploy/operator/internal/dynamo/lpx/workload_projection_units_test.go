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
		architecture: lp20, devicesPerNode: 8, hybrid: true,
		partitions: testUnitPartitions(16, 0, 1, 2, 3),
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
	}{
		{
			name: "LP20 chain joined by an xt8888Gap connector",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(8, 1, 2),
				units: []testLPUUnit{{reservations: []testLPUReservation{testNodesReservation(1, 1), testNodesReservation(1, 2)}, connectors: []testLPUConnector{{}}}},
			},
			wantPartitions: []lpxv1alpha1.PartitionRequest{
				{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 1, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC8)},
				{ID: "partition-001", Ordinal: 1, CompilerPartitionID: 2, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC8)},
			},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{{FromPartitionID: "partition-000", ToPartitionID: "partition-001", Requirement: xtGap}},
			wantAgents:     2,
		},
		{
			name: "LP20 packed chain requests every chip of its hosts",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(4, 1, 2, 3, 4, 5, 6),
				units: testSingleUnits(testNodesReservation(3, 1, 2, 3, 4, 5, 6)),
			},
			wantPartitions: []lpxv1alpha1.PartitionRequest{{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 1, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC24)}},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{},
			wantAgents:     3,
		},
		{
			name: "LPU-only CPU embeddings keep partition 0 off LPUs",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, cpuEmbeddings: true, partitions: testUnitPartitions(8, 0, 1, 2),
				units: []testLPUUnit{{reservations: []testLPUReservation{testNodesReservation(1, 1), testNodesReservation(1, 2)}, connectors: []testLPUConnector{{}}}},
			},
			wantPartitions: []lpxv1alpha1.PartitionRequest{
				{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 1, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC8)},
				{ID: "partition-001", Ordinal: 1, CompilerPartitionID: 2, XtShape: new(lpxv1alpha1.Xt8888PartitionShapeC8)},
			},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{{FromPartitionID: "partition-000", ToPartitionID: "partition-001", Requirement: xtGap}},
			wantAgents:     2,
		},
		{
			name: "LP30 single partition requests its HX extent",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(32, 1),
				units: testSingleUnits(testHXReservation(1, 2, 1, 1)),
			},
			wantPartitions: []lpxv1alpha1.PartitionRequest{{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 1, Extent: &[]int64{16, 1, 2, 1}}},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{},
			wantAgents:     2,
		},
		{
			name: "LP30 partial selection requests its one partition",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partSelect: []uint32{3}, partitions: testUnitPartitions(16, 3),
				units: testSingleUnits(testHXReservation(1, 1, 1, 3)),
			},
			wantPartitions: []lpxv1alpha1.PartitionRequest{{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 3, Extent: &[]int64{16, 1, 1, 1}}},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{},
			wantAgents:     1,
		},
		{
			name: "LP30 packed chain requests the packed hosts' extent",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(8, 1, 2, 3, 4),
				units: testSingleUnits(testHXReservation(2, 1, 1, 1, 2, 3, 4)),
			},
			wantPartitions: []lpxv1alpha1.PartitionRequest{{ID: "partition-000", Ordinal: 0, CompilerPartitionID: 1, Extent: &[]int64{16, 2, 1, 1}}},
			wantConnectors: []lpxv1alpha1.PropSyncConnectorRequest{},
			wantAgents:     2,
		},
		{
			name: "LP30 chain carries the compiler's device pairs",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16,
				partitions: []testUnitPartition{{id: 1, numChips: 32}, {id: 2, numChips: 16}},
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
				architecture: lp30, devicesPerNode: 16, hybrid: true,
				partitions: testUnitPartitions(16, 1, 2, 3, 4),
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
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(8, 1, 2),
				units: testSingleUnits(testNodesReservation(1, 1), testNodesReservation(1, 2)),
			},
			wantErr: "unsupported LPX runtime: LPU-only workloads require exactly one lpuUnits entry per model; this bundle has 2",
		},
		{
			name: "LP30 LPU-only partitions without a chain",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(16, 1, 2),
				units: testSingleUnits(testHXReservation(1, 1, 1, 1), testHXReservation(1, 1, 1, 2)),
			},
			wantErr: "unsupported LPX runtime: LPU-only workloads require exactly one lpuUnits entry per model; this bundle has 2",
		},
		{
			name: "LPU-only local partitions",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(8, 1),
				units: testSingleUnits(testNodesReservation(1, 1)),
			},
			selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{1}},
			wantErr:   "unsupported LPX runtime: localPartitions requires a hybrid build with a Cyborg conductor",
		},
		{
			name: "LP20 hosts without a registered XT shape",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(72, 1),
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
		// The request carries no compiler envelope, so the LPR CRD alone bounds connectors.
		{name: "HX connectors with more than 8192 device pairs", devicesPerNode: 4097, units: 3, chain: true},
		{name: "most LPX partitions", devicesPerNode: 16, units: maxLPXPartitions},
		{name: "too many LPX partitions", devicesPerNode: 16, units: maxLPXPartitions + 1, wantErr: "LPX projection has 257 partitions, limit is 1..256"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Place one-host LP30 partitions in single units or one chain joined device by device")
			bundle := testUnitBundle{architecture: deploymentcapnpv1.LpuArchitecture_lp30, devicesPerNode: test.devicesPerNode, hybrid: true}
			chain := testLPUUnit{}
			for id := range uint32(test.units) {
				bundle.partitions = append(bundle.partitions, testUnitPartition{id: id, numChips: test.devicesPerNode})
				if !test.chain {
					bundle.units = append(bundle.units, testSingleUnits(testHXReservation(1, 1, 1, id))...)
					continue
				}
				chain.reservations = append(chain.reservations, testHXReservation(1, 1, 1, id))
				if id > 0 {
					chain.connectors = append(chain.connectors, testHXConventionConnector(test.devicesPerNode))
				}
			}
			if test.chain {
				bundle.units = []testLPUUnit{chain}
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
		architecture: deploymentcapnpv1.LpuArchitecture_lp30, devicesPerNode: 16, hybrid: true,
		partitions: testUnitPartitions(16, 1, 2, 3),
		units: []testLPUUnit{
			{reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1), testHXReservation(1, 1, 1, 2)}, connectors: []testLPUConnector{testHXConventionConnector(16)}},
			{reservations: []testLPUReservation{testHXReservation(1, 1, 1, 3)}},
		},
	}
	build, err := acquireTestUnitBundle(t, bundle)
	require.NoError(t, err)
	bundle.units[0].connectors = []testLPUConnector{{hx: true, pairs: [][2]uint32{{0, 0}}, lanes: []uint8{4, 2, 1}, dimension: 1}}
	otherPairs, err := acquireTestUnitBundle(t, bundle)
	require.NoError(t, err)

	t.Log("Project units, other units, and a local selection")
	projected, err := projectTestUnits(build, nil)
	require.NoError(t, err)
	repeated, err := projectTestUnits(build, nil)
	require.NoError(t, err)
	otherProjected, err := projectTestUnits(otherPairs, nil)
	require.NoError(t, err)
	local, err := projectTestUnits(build, &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{3}})
	require.NoError(t, err)

	t.Log("Bind the units and the local selection into the digest")
	digest := projected[0].Digest()
	require.Equal(t, digest, repeated[0].Digest())
	require.NotEqual(t, digest, otherProjected[0].Digest())
	require.NotEqual(t, digest, local[0].Digest())
}
