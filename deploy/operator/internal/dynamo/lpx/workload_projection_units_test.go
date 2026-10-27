/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
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
		},
		{
			name: "LP30 chain carries the compiler's device pairs",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, chains: [][]uint32{{1, 2}},
				partitions: []testUnitPartition{{id: 1, numChips: 32, shape: []uint32{16, 1, 2, 1}}, {id: 2, numChips: 16, shape: []uint32{16, 1, 1, 1}}},
				units: []testLPUUnit{{
					reservations: []testLPUReservation{testHXReservation(1, 2, 1, 1), testHXReservation(1, 1, 1, 2)},
					connectors:   []testLPUConnector{{hx: true, pairs: [][2]uint32{{16, 0}, {31, 15}}, lanes: []uint8{4, 2}}},
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
				architecture: lp30, devicesPerNode: 16, hybrid: true, chains: [][]uint32{{2, 3}},
				partitions: testUnitPartitions(16, []uint32{16, 1, 1, 1}, 1, 2, 3, 4),
				units: []testLPUUnit{
					{reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1)}},
					{
						reservations: []testLPUReservation{testHXReservation(1, 1, 1, 2), testHXReservation(1, 1, 1, 3)},
						connectors:   []testLPUConnector{{hx: true, pairs: [][2]uint32{{0, 0}, {1, 1}}, lanes: []uint8{4, 2, 1}}},
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
		{name: "HX connectors at the device-pair budget", devicesPerNode: 4096, units: 3, chain: true},
		{name: "HX connectors over the device-pair budget", devicesPerNode: 4097, units: 3, chain: true, wantErr: "HX prop-sync graph exceeds 8192 logical connections"},
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

			t.Log("Bound the request's partitions and device pairs")
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
	bundle.units[0].connectors = []testLPUConnector{{hx: true, pairs: [][2]uint32{{0, 0}}, lanes: []uint8{4, 2, 1}}}
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
// Only HX allocation metadata differs: units send none.
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
		// wantUnitsErr is the units path's refusal of a bundle whose partitions
		// the legacy derivation deploys.
		wantUnitsErr string
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
			name: "LP30 one-node partition without topology metadata sized in nodes",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(8, nil, 1),
				units: testSingleUnits(testNodesReservation(1, 1)),
			},
			wantUnitsErr: "config.lpuUnits[0].reservations[0].size.nodes sizes an LP30 reservation",
		},
		{
			name: "LP30 packed chain without topology metadata sized in nodes",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, chains: [][]uint32{{1, 2, 3, 4}}, partitions: testUnitPartitions(8, nil, 1, 2, 3, 4),
				units: testSingleUnits(testNodesReservation(2, 1, 2, 3, 4)),
			},
			wantUnitsErr: "config.lpuUnits[0].reservations[0].size.nodes sizes an LP30 reservation",
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

			if test.wantUnitsErr != "" {
				t.Log("Refuse the units, which have no LPX encoding")
				_, err := acquireTestUnitBundle(t, test.bundle)
				require.ErrorIs(t, err, ErrUnsupportedRuntime)
				require.ErrorContains(t, err, test.wantUnitsErr)

				t.Log("Deploy the same partitions through the legacy derivation")
				legacyBundle := test.bundle
				legacyBundle.units = nil
				build, err := acquireTestUnitBundle(t, legacyBundle)
				require.NoError(t, err)
				_, err = projectTestUnits(build, test.selection)
				require.NoError(t, err)
				return
			}

			t.Log("Decode the synthetic bundle and compare unit placement with the legacy derivation")
			build, err := acquireTestUnitBundle(t, test.bundle)
			require.NoError(t, err)
			requireLegacyParity(t, build, test.selection)
		})
	}
}

// placementFixtureRoot holds the vendored copy of the shared lpx_placement
// contract fixtures. Copy the canonical set into it unchanged; never edit it here.
const placementFixtureRoot = "testdata/lpx_placement"

// placementFixtureIndex is the fixture set's fixtures.json: one entry per
// fixture directory with the sha256 of every file in it.
type placementFixtureIndex struct {
	Fixtures []struct {
		Name  string            `json:"name"`
		Files map[string]string `json:"files"`
	} `json:"fixtures"`
}

// readPlacementFixtureIndex reads the vendored fixtures.json, which must list
// at least one fixture.
func readPlacementFixtureIndex(t *testing.T) placementFixtureIndex {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(placementFixtureRoot, "fixtures.json"))
	require.NoError(t, err)
	var index placementFixtureIndex
	require.NoError(t, json.Unmarshal(data, &index))
	require.NotEmpty(t, index.Fixtures)
	return index
}

// expectedPlacement is a fixture's hand-reviewed expected-placement.json. The
// operator checks the fields it owns; runtime-only views such as novaStages and
// per-pod ranks are ignored. A null lpr means the units path refuses the bundle
// for the refusal reason. legacy is the legacy projection of the same bundle and
// scenario with the units ignored; its configMap and lpuServers record the
// runtime ConfigMaps the operator no longer renders and are ignored.
type expectedPlacement struct {
	CompilationMode string `json:"compilationMode"`
	Architecture    string `json:"architecture"`
	DevicesPerNode  int    `json:"devicesPerNode"`
	LocalPartitions *struct {
		Mode string  `json:"mode"`
		IDs  []int64 `json:"ids"`
	} `json:"localPartitions"`
	Units             []expectedPlacementUnit `json:"units"`
	LocalPartitionIDs []int                   `json:"localPartitionIds"`
	LPR               *expectedPlacementLPR   `json:"lpr"`
	Refusal           *struct {
		Operator string `json:"operator"`
	} `json:"refusal"`
	AgentCount int `json:"agentCount"`
	AgentPods  []struct {
		PodIndex       int    `json:"podIndex"`
		LPRPartitionID string `json:"lprPartitionId"`
	} `json:"agentPods"`
	Legacy struct {
		LPR        *expectedPlacementLPR `json:"lpr"`
		AgentCount int                   `json:"agentCount"`
	} `json:"legacy"`
}

// expectedPlacementLPR is a reviewed LPX pipeline request placement. The units
// path lpr omits allocationMetadata because units always send {}.
type expectedPlacementLPR struct {
	Partitions         []lpxv1alpha1.PartitionRequest         `json:"partitions"`
	PropSyncConnectors []lpxv1alpha1.PropSyncConnectorRequest `json:"propSyncConnectors"`
	AllocationMetadata json.RawMessage                        `json:"allocationMetadata"`
}

// expectedPlacementUnit mirrors one config.lpuUnits entry with capnp field names.
type expectedPlacementUnit struct {
	Reservations []struct {
		PartitionIDs []int `json:"partitionIds"`
		Size         struct {
			Nodes *int `json:"nodes"`
			HX    *struct {
				TraysPerHemisphere int64 `json:"traysPerHemisphere"`
				Hemispheres        int64 `json:"hemispheres"`
				Racks              int64 `json:"racks"`
			} `json:"hx"`
		} `json:"size"`
	} `json:"reservations"`
	Connectors []struct {
		Xt8888Gap *struct {
			MaxInterPartitionOffset int64 `json:"maxInterPartitionOffset"`
		} `json:"xt8888Gap"`
		HxPropSyncV1 *struct {
			Connections []struct {
				FromDevice int64 `json:"fromDevice"`
				ToDevice   int64 `json:"toDevice"`
			} `json:"connections"`
			AcceptableLaneMultiplicities []int64 `json:"acceptableLaneMultiplicities"`
		} `json:"hxPropSyncV1"`
	} `json:"connectors"`
}

// TestPlacementFixturesMatchIndex requires the vendored lpx_placement set to be
// an unedited copy of the shared set: the index and README, one directory per
// indexed fixture, and in each exactly the indexed files with their sha256.
func TestPlacementFixturesMatchIndex(t *testing.T) {
	t.Parallel()

	t.Log("Read the fixture index and the vendored directory")
	index := readPlacementFixtureIndex(t)
	entries, err := os.ReadDir(placementFixtureRoot)
	require.NoError(t, err)

	t.Log("Require the index, the README and exactly one directory per indexed fixture")
	var rootFiles, fixtureDirs, indexed []string
	for _, entry := range entries {
		if entry.IsDir() {
			fixtureDirs = append(fixtureDirs, entry.Name())
		} else {
			rootFiles = append(rootFiles, entry.Name())
		}
	}
	for _, fixture := range index.Fixtures {
		indexed = append(indexed, fixture.Name)
	}
	require.ElementsMatch(t, []string{"README.md", "fixtures.json"}, rootFiles)
	require.ElementsMatch(t, indexed, fixtureDirs)

	for _, fixture := range index.Fixtures {
		t.Run(fixture.Name, func(t *testing.T) {
			t.Parallel()

			t.Log("Require exactly the indexed files, including the deployment contract and the reviewed placement")
			dir := filepath.Join(placementFixtureRoot, fixture.Name)
			files, err := os.ReadDir(dir)
			require.NoError(t, err)
			names := make([]string, 0, len(files))
			for _, file := range files {
				names = append(names, file.Name())
			}
			require.ElementsMatch(t, slices.Collect(maps.Keys(fixture.Files)), names)
			require.Contains(t, fixture.Files, gbuildDeploymentV1CapnpFile)
			require.Contains(t, fixture.Files, "expected-placement.json")

			t.Log("Require every vendored file to have its indexed sha256")
			for name, want := range fixture.Files {
				data, err := os.ReadFile(filepath.Join(dir, name))
				require.NoError(t, err)
				sum := sha256.Sum256(data)
				require.Equal(t, want, hex.EncodeToString(sum[:]), "sha256 of %s/%s", fixture.Name, name)
			}
		})
	}
}

// TestUnitProjectionMatchesPlacementFixtures projects every compiler-written
// bundle of the shared lpx_placement set for its scenario through the legacy
// derivation with the units ignored and through the units, and checks each
// against the fixture's hand-reviewed placement.
func TestUnitProjectionMatchesPlacementFixtures(t *testing.T) {
	t.Parallel()
	workloadModes := map[string]lpxv1alpha1.WorkloadMode{
		"lp20/lpuOnly": lpxv1alpha1.WorkloadModeV2LPUOnly,
		"lp20/lpx":     lpxv1alpha1.WorkloadModeV2StrictHybrid,
		"lp30/lpuOnly": lpxv1alpha1.WorkloadModeV3HxLPUOnly,
		"lp30/lpx":     lpxv1alpha1.WorkloadModeV3HxStrictHybrid,
	}
	targetFamilies := map[string]lpxv1alpha1.TargetFamily{
		"lp20": lpxv1alpha1.TargetFamilyXt8888,
		"lp30": lpxv1alpha1.TargetFamilyHx16x8x2x3,
	}
	for _, fixture := range readPlacementFixtureIndex(t).Fixtures {
		t.Run(fixture.Name, func(t *testing.T) {
			t.Parallel()

			t.Log("Read the reviewed placement and resolve its scenario")
			dir, err := filepath.Abs(filepath.Join(placementFixtureRoot, fixture.Name))
			require.NoError(t, err)
			data, err := os.ReadFile(filepath.Join(dir, "expected-placement.json"))
			require.NoError(t, err)
			var expected expectedPlacement
			require.NoError(t, json.Unmarshal(data, &expected))
			workloadMode := workloadModes[expected.Architecture+"/"+expected.CompilationMode]
			targetFamily := targetFamilies[expected.Architecture]
			require.NotEmpty(t, workloadMode, "architecture %q, compilationMode %q", expected.Architecture, expected.CompilationMode)
			hybrid := expected.CompilationMode == "lpx"
			var selection *dynamov1beta1.LPXLocalPartitions
			if scenario := expected.LocalPartitions; scenario != nil {
				switch scenario.Mode {
				case "all":
					selection = &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeAll}
				case "ids":
					selection = &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: scenario.IDs}
				default:
					require.Failf(t, "unsupported localPartitions mode", "%q", scenario.Mode)
				}
			}

			t.Log("Acquire the bundle and decode the compiler's units")
			registry, err := NewModelRegistry("", nil)
			require.NoError(t, err)
			build, err := registry.AcquireBuild(t.Context(), dir)
			require.NoError(t, err)
			require.Equal(t, string(targetFamily), string(build.family))
			require.Equal(t, expected.DevicesPerNode, build.partitions[0].devicesPerNode)
			require.Equal(t, hybrid, build.hasCUDAPartitions)
			wantUnits := make([]lpuUnit, len(expected.Units))
			for index, unit := range expected.Units {
				wantUnits[index] = unit.lpuUnit(t, build.family, expected.DevicesPerNode)
			}
			require.Equal(t, wantUnits, build.lpuUnits)

			t.Log("Reproduce the legacy section, allocation metadata included, with the units ignored")
			require.NotNil(t, expected.Legacy.LPR)
			legacyBuild := *build
			legacyBuild.lpuUnits = nil
			legacyProjections, err := projectTestUnits(&legacyBuild, selection)
			require.NoError(t, err)
			legacySpec := legacyProjections[0].RequestSpec(&MaterializationPlan{}, "agents")
			requirePlacementRequest(t, *expected.Legacy.LPR, workloadMode, targetFamily, legacySpec)
			require.JSONEq(t, string(expected.Legacy.LPR.AllocationMetadata), string(legacySpec.AllocationMetadata.Raw))
			require.Equal(t, expected.Legacy.AgentCount, legacyProjections[0].AgentReplicas())

			t.Log("Project the units, or refuse the bundle for the reviewed reason")
			projections, err := projectTestUnits(build, selection)
			if expected.LPR == nil {
				require.NotNil(t, expected.Refusal)
				require.ErrorIs(t, err, ErrUnsupportedRuntime)
				require.ErrorContains(t, err, expected.Refusal.Operator)
				return
			}
			require.NoError(t, err)
			projection := projections[0]

			t.Log("Request the reviewed LPX partitions and connectors without allocation metadata for the reviewed Agents")
			spec := projection.RequestSpec(&MaterializationPlan{}, "agents")
			requirePlacementRequest(t, *expected.LPR, workloadMode, targetFamily, spec)
			require.JSONEq(t, `{}`, string(spec.AllocationMetadata.Raw))
			require.Equal(t, expected.AgentCount, projection.AgentReplicas())
			require.Equal(t, expected.LocalPartitionIDs, append([]int{}, projection.localPartitionIDs...))

			t.Log("Place each reviewed Agent pod in its LPX partition, walking partitions in Agent order")
			wantPodPartitions := make([]string, 0, len(expected.AgentPods))
			for index, pod := range expected.AgentPods {
				require.Equal(t, index, pod.PodIndex)
				wantPodPartitions = append(wantPodPartitions, pod.LPRPartitionID)
			}
			podPartitions := make([]string, 0, projection.AgentReplicas())
			for ordinal, partition := range projection.partitions {
				for range partition.effectiveNodeCount() {
					podPartitions = append(podPartitions, spec.Partitions[ordinal].ID)
				}
			}
			require.Equal(t, wantPodPartitions, podPartitions)

			if !hybrid {
				return
			}

			t.Log("Render the hybrid workload with one Cyborg client per runtime I/O endpoint")
			pcs := renderTestPCS(true)
			pcs.Spec.Template.Cliques[0].Spec.Replicas = build.ioFPGACount * build.ioFanoutFactor
			rendered, err := renderSelectedForTest(pcs, projections, RenderInput{
				Stages: map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}},
			})
			require.NoError(t, err)

			t.Log("Publish the local unit roots to Agents and Cyborg, and the Agent template to Cyborg")
			roots := make([]string, len(expected.LocalPartitionIDs))
			for index, id := range expected.LocalPartitionIDs {
				roots[index] = strconv.Itoa(id)
			}
			localRoots := corev1.EnvVar{Name: localPartitionIDsEnv, Value: strings.Join(roots, ",")}
			agent := namedClique(t, rendered, testAgentTemplateName)
			require.Equal(t, int32(expected.AgentCount), agent.Spec.Replicas)
			require.Equal(t, []corev1.EnvVar{localRoots}, placementEnv(agent.Spec.PodSpec.Containers[0].Env))
			require.Equal(t, []corev1.EnvVar{{Name: allocationEnvVar, Value: testAgentTemplateName}, localRoots},
				placementEnv(namedClique(t, rendered, "cond").Spec.PodSpec.Containers[0].Env))
		})
	}
}

// requirePlacementRequest requires spec to request the reviewed LPX partitions
// and connectors for workloadMode and targetFamily.
func requirePlacementRequest(
	t *testing.T,
	want expectedPlacementLPR,
	workloadMode lpxv1alpha1.WorkloadMode,
	targetFamily lpxv1alpha1.TargetFamily,
	spec lpxv1alpha1.LPUPipelineRequestSpec,
) {
	t.Helper()
	require.Equal(t, workloadMode, spec.WorkloadMode)
	require.Equal(t, targetFamily, spec.TargetFamily)
	require.Equal(t, want.Partitions, spec.Partitions)
	if len(want.PropSyncConnectors) == 0 {
		require.Empty(t, spec.PropSyncConnectors)
		return
	}
	require.Equal(t, want.PropSyncConnectors, spec.PropSyncConnectors)
}

// lpuUnit converts the reviewed unit to the decoded form for family and
// devicesPerNode by SPEC rule 5: an LP20 reservation runs on size.nodes hosts,
// and an LP30 reservation on the product of its size.hx dimensions with extent
// [devicesPerNode, traysPerHemisphere, hemispheres, racks]. The operator refuses
// an LP30 reservation sized in nodes, and no reviewed fixture has one.
func (u expectedPlacementUnit) lpuUnit(t *testing.T, family BuildFamily, devicesPerNode int) lpuUnit {
	t.Helper()
	unit := lpuUnit{Connectors: []lpxv1alpha1.PropSyncConnectorRequirement{}}
	for _, reservation := range u.Reservations {
		decoded := lpuReservation{PartitionIDs: reservation.PartitionIDs}
		switch nodes, hx := reservation.Size.Nodes, reservation.Size.HX; {
		case family == BuildFamilyXT && nodes != nil && hx == nil:
			decoded.Hosts = *nodes
		case family == BuildFamilyHX && nodes == nil && hx != nil:
			decoded.Hosts = int(hx.TraysPerHemisphere * hx.Hemispheres * hx.Racks)
			decoded.HXExtent = []int64{int64(devicesPerNode), hx.TraysPerHemisphere, hx.Hemispheres, hx.Racks}
		default:
			require.Failf(t, "unsupported reviewed reservation size", "%s reservation of partitions %v", family, reservation.PartitionIDs)
		}
		unit.Reservations = append(unit.Reservations, decoded)
	}
	for _, connector := range u.Connectors {
		if gap := connector.Xt8888Gap; gap != nil {
			unit.Connectors = append(unit.Connectors, lpxv1alpha1.PropSyncConnectorRequirement{
				Kind: lpxv1alpha1.PropSyncConnectorKindXt8888Gap, MaxInterPartitionOffset: new(gap.MaxInterPartitionOffset),
			})
			continue
		}
		require.NotNil(t, connector.HxPropSyncV1, "reviewed connector has no variant")
		connections := make([]lpxv1alpha1.HxLogicalConnection, 0, len(connector.HxPropSyncV1.Connections))
		for _, pair := range connector.HxPropSyncV1.Connections {
			connections = append(connections, lpxv1alpha1.HxLogicalConnection{FromLogicalDevice: pair.FromDevice, ToLogicalDevice: pair.ToDevice})
		}
		unit.Connectors = append(unit.Connectors, lpxv1alpha1.PropSyncConnectorRequirement{
			Kind:                         lpxv1alpha1.PropSyncConnectorKindHxPropSyncV1,
			Connections:                  &connections,
			AcceptableLaneMultiplicities: new(connector.HxPropSyncV1.AcceptableLaneMultiplicities),
		})
	}
	return unit
}
