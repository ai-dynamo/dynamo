/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"
)

func TestPropSyncProjection(t *testing.T) {
	t.Parallel()

	t.Log("Cover separate and packed reservations through the same compiler-manifest path")
	for _, test := range []struct {
		name          string
		family        BuildFamily
		widths        []uint32
		nodeWidths    []uint32
		nodes         uint32
		cpu           bool
		runtimeStart  int
		chains        [][]uint32
		pipeline      Pipeline
		topologies    []string
		ids           []uint32
		shape         *lpxv1alpha1.Xt8888PartitionShape
		extent        *[]int64
		omitMetadata  bool
		physicalNodes int32
		wantError     string
	}{
		{name: "selected subset", family: BuildFamilyXT, widths: []uint32{8, 4, 4}, ids: []uint32{0, 1, 2}, chains: [][]uint32{{1, 2}}, nodes: 3, runtimeStart: 1, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 1},
		{name: "CPU embeddings inside chain", family: BuildFamilyXT, widths: []uint32{8, 4, 4}, ids: []uint32{0, 1, 2}, chains: [][]uint32{{0, 1, 2}}, nodes: 1, cpu: true, runtimeStart: 1, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 1},
		{name: "host embedding node count", family: BuildFamilyXT, widths: []uint32{8, 4, 4}, ids: []uint32{0, 1, 2}, chains: [][]uint32{{0, 1, 2}}, nodes: 2, cpu: true, runtimeStart: 1, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 1},
		{name: "unrelated node count", family: BuildFamilyXT, widths: []uint32{8, 4, 4}, ids: []uint32{0, 1, 2}, chains: [][]uint32{{1, 2}}, nodes: 4, cpu: true, wantError: "deployment.numLpuNodes"},
		{name: "LPU embeddings cannot pack", family: BuildFamilyXT, widths: []uint32{8, 4, 4}, ids: []uint32{0, 1, 2}, chains: [][]uint32{{0, 1, 2}}, nodes: 1, wantError: "deployment.numLpuNodes"},
		{name: "invalid selected chain", family: BuildFamilyXT, widths: []uint32{8, 4, 4}, ids: []uint32{0, 1, 2}, chains: [][]uint32{{1, 3}}, nodes: 1, cpu: true, wantError: "not contiguous"},
		{family: BuildFamilyXT, name: "XT unselected stages", widths: []uint32{4, 4}, chains: [][]uint32{}},
		{family: BuildFamilyXT, name: "XT whole node chain", widths: []uint32{8, 8}},
		{family: BuildFamilyXT, name: "XT hybrid separate reservations", widths: []uint32{4, 4}, pipeline: PipelineLPX},
		{family: BuildFamilyXT, name: "XT hybrid packed node count", widths: []uint32{4, 4}, pipeline: PipelineLPX, nodes: 1, wantError: "LPU partitions use 2 LPU nodes, want deployment.numLpuNodes 1"},
		{family: BuildFamilyHX, name: "HX unrelated packed node count", widths: []uint32{8, 8}, nodes: 3, wantError: "deployment.numLpuNodes"},
		{family: BuildFamilyHX, name: "HX mixed subnode widths", widths: []uint32{8, 4}},
		{family: BuildFamilyHX, name: "HX narrower nonstandard destination", widths: []uint32{8, 4}, nodeWidths: []uint32{8, 4}, wantError: "destination partition 2 has 4 devices, need 8"},
		{family: BuildFamilyHX, name: "HX wider destination", widths: []uint32{8, 16}, nodeWidths: []uint32{8, 16}},
		{family: BuildFamilyHX, name: "HX smaller nodes with enough destination capacity", widths: []uint32{16, 16}, nodeWidths: []uint32{16, 8}},
		{family: BuildFamilyXT, name: "XT mixed subnode widths", widths: []uint32{4, 2}},
		{family: BuildFamilyXT, name: "XT mixed node widths", widths: []uint32{4, 4}, nodeWidths: []uint32{8, 16}},
		{family: BuildFamilyXT, name: "XT width does not divide node", widths: []uint32{3, 3}},
		{family: BuildFamilyXT, name: "XT different topology names", widths: []uint32{4, 4}, topologies: []string{"topology-a", "topology-b"}},
		{family: BuildFamilyXT, name: "XT packed twelve-node chain", widths: slices.Repeat([]uint32{4}, 24), nodes: 12, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC96), physicalNodes: 12},
		{family: BuildFamilyXT, name: "XT unregistered whole-node shape", widths: []uint32{72}, wantError: "unregistered XT shape c72"},
		{family: BuildFamilyXT, name: "XT unregistered packed shape", widths: []uint32{1, 1}, nodeWidths: []uint32{3, 3}, wantError: "unregistered XT shape c3"},
		{family: BuildFamilyHX, name: "HX nine nodes", widths: slices.Repeat([]uint32{8}, 18), wantError: "144 does not fit a supported HX extent"},
		{family: BuildFamilyHX, name: "HX eighteen nodes", widths: slices.Repeat([]uint32{8}, 36), wantError: "288 does not fit a supported HX extent"},
		{family: BuildFamilyHX, name: "HX more than two racks", widths: slices.Repeat([]uint32{8}, 66), wantError: "528 does not fit a supported HX extent"},
		{family: BuildFamilyHX, name: "HX incomplete chain", widths: []uint32{8, 8, 8}, chains: [][]uint32{{1, 2}}, wantError: "complete adjacent prop-sync connector chain"},
		{family: BuildFamilyHX, name: "HX hybrid", widths: []uint32{8, 8}, pipeline: PipelineLPX, wantError: "requires an LPU-only workload"},
		{family: BuildFamilyHX, name: "HX hybrid unselected stages", widths: []uint32{8, 8}, pipeline: PipelineLPX, chains: [][]uint32{}},
		{family: BuildFamilyHX, name: "HX hybrid partial chain", widths: []uint32{8, 8, 8}, pipeline: PipelineLPX, chains: [][]uint32{{1, 2}}},
		{family: BuildFamilyHX, name: "HX aggregate connection budget", widths: []uint32{4096, 4096, 4096}, nodeWidths: []uint32{4096, 4096, 4096}},
		{family: BuildFamilyHX, name: "HX aggregate connection budget exceeded", widths: []uint32{4097, 4097, 4097}, nodeWidths: []uint32{4097, 4097, 4097}, wantError: "exceeds 8192 logical connections"},
		{family: BuildFamilyHX, name: "HX maximum manifest node width", widths: []uint32{math.MaxUint32, math.MaxUint32}, nodeWidths: []uint32{math.MaxUint32, math.MaxUint32}, wantError: "exceeds 8192 logical connections"},
		{family: BuildFamilyHX, name: "HX wide singleton without connectors", widths: []uint32{math.MaxUint32}, nodeWidths: []uint32{math.MaxUint32}},
		{
			name: "2x4c", family: BuildFamilyXT, widths: []uint32{4, 4}, nodes: 1, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 1,
		},
		{
			name: "4x2c", family: BuildFamilyXT, widths: []uint32{2, 2, 2, 2}, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 1,
		},
		{
			name: "6x4c", family: BuildFamilyXT, widths: []uint32{4, 4, 4, 4, 4, 4}, nodes: 3, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC24), physicalNodes: 3,
		},
		{
			name: "partial last owner", family: BuildFamilyXT, widths: []uint32{4, 4, 4}, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC16), physicalNodes: 2,
		},
		{
			name: "LP30 2x8c without topology metadata", family: BuildFamilyHX, widths: []uint32{8, 8}, extent: ptr.To([]int64{16, 1, 1, 1}), physicalNodes: 1, omitMetadata: true,
		},
		{
			name: "LP30 4x4c", family: BuildFamilyHX, widths: []uint32{4, 4, 4, 4}, nodes: 1, extent: ptr.To([]int64{16, 1, 1, 1}), physicalNodes: 1,
		},
		{
			name: "LP30 partial last node in manifest order", family: BuildFamilyHX, widths: []uint32{8, 8, 8}, ids: []uint32{11, 3, 9}, nodes: 2, extent: ptr.To([]int64{16, 2, 1, 1}), physicalNodes: 2,
		},
		{name: "XT single subnode partition", family: BuildFamilyXT, widths: []uint32{2}, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 1},
		{name: "LP30 single subnode stage", family: BuildFamilyHX, widths: []uint32{8}, extent: ptr.To([]int64{16, 1, 1, 1}), physicalNodes: 1},
		{name: "LP30 two hemispheres", family: BuildFamilyHX, widths: slices.Repeat([]uint32{8}, 20), extent: ptr.To([]int64{16, 5, 2, 1}), physicalNodes: 10},
		{name: "LP30 two racks", family: BuildFamilyHX, widths: slices.Repeat([]uint32{8}, 64), extent: ptr.To([]int64{16, 8, 2, 2}), physicalNodes: 32},
		{name: "XT four-chip nodes", family: BuildFamilyXT, widths: []uint32{2, 2, 2}, nodeWidths: []uint32{4, 4, 4}, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 2},
		{name: "XT single stage across four-chip nodes", family: BuildFamilyXT, widths: []uint32{8}, nodeWidths: []uint32{4}, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 2},
		{name: "XT sixteen-node stage", family: BuildFamilyXT, widths: []uint32{128}, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC128), physicalNodes: 16},
		{name: "XT thirty-two-chip nodes", family: BuildFamilyXT, widths: slices.Repeat([]uint32{4}, 18), nodeWidths: slices.Repeat([]uint32{32}, 18), shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC96), physicalNodes: 3},
		{name: "LP30 eight-chip nodes", family: BuildFamilyHX, widths: []uint32{4, 4, 4}, nodeWidths: []uint32{8, 8, 8}, extent: ptr.To([]int64{8, 2, 1, 1}), physicalNodes: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Encode source stages, manifest geometry and the selected chain")
			fixture := newV2CompilerFixture()
			if test.family == BuildFamilyHX {
				fixture = newV3CompilerFixture()
			}
			width := fixture.partitions[0].devicesPerNode
			fixture.numLPUNodes, fixture.cpuEmbeddings = 0, test.cpu
			fixture.partitions = make([]testV3CapnpPartition, len(test.widths))
			fixture.selectedPropSyncChains = [][]uint32{make([]uint32, len(test.widths))}
			for index, chips := range test.widths {
				partition := &fixture.partitions[index]
				*partition = testV3CapnpPartition{
					id: uint32(index + 1), deviceType: manifestcapnp.DeviceType_lpu, numChips: chips, devicesPerNode: width,
					topology: registryTestTopology,
				}
				if index < len(test.ids) {
					partition.id = test.ids[index]
				}
				if index < len(test.nodeWidths) {
					partition.devicesPerNode = test.nodeWidths[index]
				}
				if test.family == BuildFamilyHX {
					partition.topology = fmt.Sprintf("hx-stage-%d", partition.id)
					partition.topologyFamily = hxTopologyFamily
					partition.partitionShape = []uint32{min(chips, partition.devicesPerNode), max(1, chips/partition.devicesPerNode), 1, 1}
				}
				if index < len(test.topologies) {
					partition.topology = test.topologies[index]
				}
				if test.omitMetadata {
					partition.topologyFamily, partition.partitionShape = "", nil
				}
				fixture.selectedPropSyncChains[0][index] = partition.id
				fixture.numLPUNodes += max(1, chips/partition.devicesPerNode)
			}
			if len(test.widths) == 1 {
				fixture.selectedPropSyncChains = nil
			}
			if test.nodes != 0 {
				fixture.numLPUNodes = test.nodes
			}
			if test.chains != nil {
				fixture.selectedPropSyncChains = test.chains
			}
			pipeline := test.pipeline
			if pipeline == "" {
				pipeline = PipelineSingle
			}
			if pipeline == PipelineLPX {
				fixture.compilationMode = manifestcapnp.CompilationMode_lpx
			}

			t.Log("Project each geometry once and reject unsupported workloads")
			buildDir := writeCompilerFixture(t, fixture)
			snapshot := acquireTestSnapshot(t, buildDir)
			normalized, err := normalizeBuildSnapshot(snapshot)
			var projections []*ModelProjection
			if err == nil {
				projections, err = appendModelProjections(nil, ModelProjectionInput{
					Pipeline: pipeline, Models: []string{"default"}, BuildSnapshot: normalized,
				})
			}
			if test.wantError != "" {
				require.ErrorContains(t, err, test.wantError)
				return
			}
			require.NoError(t, err)
			projection := projections[0]
			require.Equal(t, normalizeTestSnapshot(t, snapshot).build, normalized.build)
			original := normalized.build.Partitions

			if test.physicalNodes == 0 {
				t.Log("Retain separate reservations and HX device-index connectors when packing does not apply")
				require.Equal(t, original, projection.partitions)
				require.EqualValues(t, fixture.numLPUNodes, projection.agentReplicas)
				edges := 0
				for _, chain := range fixture.selectedPropSyncChains {
					edges += len(chain) - 1
				}
				require.Len(t, projection.connectors, edges)
				if test.family == BuildFamilyHX {
					for _, connector := range projection.connectors {
						for device, connection := range *connector.Requirement.Connections {
							require.Equal(t, lpxv1alpha1.HxLogicalConnection{FromLogicalDevice: int64(device), ToLogicalDevice: int64(device)}, connection)
						}
					}
				}
				return
			}

			t.Log("Pack one physical shape, preserve every logical source stage and remove internal connectors")
			require.EqualValues(t, test.physicalNodes, projection.agentReplicas)
			require.Equal(t, original[test.runtimeStart:], projection.configuredBuild.Partitions)
			request := projection.RequestSpec(&MaterializationPlan{}, "agents")
			require.Len(t, request.Partitions, 1)
			require.EqualValues(t, test.family, request.TargetFamily)
			require.Equal(t, test.shape, request.Partitions[0].XtShape)
			require.Equal(t, test.extent, request.Partitions[0].Extent)
			require.EqualValues(t, fixture.partitions[test.runtimeStart].id, request.Partitions[0].CompilerPartitionID)
			require.Empty(t, request.PropSyncConnectors)
			require.Equal(t, []lpxv1alpha1.NodeLocalPartitionMapping{{ModelPartitionID: 0, PartitionID: "partition-000"}}, request.NodeLocal.PartitionMappings)
			if test.family == BuildFamilyHX {
				extentJSON, err := json.Marshal(test.extent)
				require.NoError(t, err)
				require.JSONEq(t, fmt.Sprintf(`{"arch":"lp30","topology":"lyra","metadata_version":1,
					"partition_info":{"num_partitions":1,"%d":{"device":"lpu","allocation":%s}},
					"prop_sync_info":{"version":1,"prop_sync_pairs":[]}}`, fixture.partitions[0].id, extentJSON), string(request.AllocationMetadata.Raw))
			}
		})
	}
}
