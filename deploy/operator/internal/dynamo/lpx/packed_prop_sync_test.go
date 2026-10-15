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
		family        buildTopology
		widths        []uint32
		nodeWidth     uint32
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
		{name: "selected subset", family: xtBuildTopology, widths: []uint32{8, 4, 4}, ids: []uint32{0, 1, 2}, chains: [][]uint32{{1, 2}}, runtimeStart: 1, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 1},
		{name: "CPU embeddings inside chain", family: xtBuildTopology, widths: []uint32{8, 4, 4}, ids: []uint32{0, 1, 2}, chains: [][]uint32{{0, 1, 2}}, cpu: true, runtimeStart: 1, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 1},
		{name: "invalid selected chain", family: xtBuildTopology, widths: []uint32{8, 4, 4}, ids: []uint32{0, 1, 2}, chains: [][]uint32{{1, 3}}, cpu: true, wantError: "not contiguous"},
		{family: xtBuildTopology, name: "XT unselected stages", widths: []uint32{4, 4}, chains: [][]uint32{}},
		{family: xtBuildTopology, name: "XT whole node chain", widths: []uint32{8, 8}},
		{family: xtBuildTopology, name: "XT hybrid separate reservations", widths: []uint32{4, 4}, pipeline: PipelineHybrid},
		{family: hxBuildTopology, name: "HX mixed subnode widths", widths: []uint32{8, 4}},
		{family: xtBuildTopology, name: "XT mixed subnode widths", widths: []uint32{4, 2}},
		{family: xtBuildTopology, name: "XT width does not divide node", widths: []uint32{3, 3}},
		{family: xtBuildTopology, name: "XT different topology names", widths: []uint32{4, 4}, topologies: []string{"topology-a", "topology-b"}},
		{family: xtBuildTopology, name: "XT packed twelve-node chain", widths: slices.Repeat([]uint32{4}, 24), shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC96), physicalNodes: 12},
		{family: xtBuildTopology, name: "XT unregistered whole-node shape", widths: []uint32{72}, wantError: "unregistered XT shape c72"},
		{family: xtBuildTopology, name: "XT unregistered packed shape", widths: []uint32{1, 1}, nodeWidth: 3, wantError: "unregistered XT shape c3"},
		{family: hxBuildTopology, name: "HX nine nodes", widths: slices.Repeat([]uint32{8}, 18), wantError: "144 does not fit a supported HX extent"},
		{family: hxBuildTopology, name: "HX eighteen nodes", widths: slices.Repeat([]uint32{8}, 36), wantError: "288 does not fit a supported HX extent"},
		{family: hxBuildTopology, name: "HX more than two racks", widths: slices.Repeat([]uint32{8}, 66), wantError: "528 does not fit a supported HX extent"},
		{family: hxBuildTopology, name: "HX incomplete chain", widths: []uint32{8, 8, 8}, chains: [][]uint32{{1, 2}}, wantError: "complete adjacent prop-sync connector chain"},
		{family: hxBuildTopology, name: "HX hybrid", widths: []uint32{8, 8}, pipeline: PipelineHybrid, wantError: "requires an LPU-only workload"},
		{family: hxBuildTopology, name: "HX hybrid unselected stages", widths: []uint32{8, 8}, pipeline: PipelineHybrid, chains: [][]uint32{}},
		{family: hxBuildTopology, name: "HX hybrid partial chain", widths: []uint32{8, 8, 8}, pipeline: PipelineHybrid, chains: [][]uint32{{1, 2}}},
		{family: hxBuildTopology, name: "HX aggregate connection budget", widths: []uint32{4096, 4096, 4096}, nodeWidth: 4096},
		{family: hxBuildTopology, name: "HX aggregate connection budget exceeded", widths: []uint32{4097, 4097, 4097}, nodeWidth: 4097, wantError: "exceeds 8192 logical connections"},
		{family: hxBuildTopology, name: "HX maximum manifest node width", widths: []uint32{math.MaxUint32, math.MaxUint32}, nodeWidth: math.MaxUint32, wantError: "exceeds 8192 logical connections"},
		{family: hxBuildTopology, name: "HX wide singleton without connectors", widths: []uint32{math.MaxUint32}, nodeWidth: math.MaxUint32},
		{
			name: "2x4c", family: xtBuildTopology, widths: []uint32{4, 4}, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 1,
		},
		{
			name: "4x2c", family: xtBuildTopology, widths: []uint32{2, 2, 2, 2}, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 1,
		},
		{
			name: "6x4c", family: xtBuildTopology, widths: []uint32{4, 4, 4, 4, 4, 4}, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC24), physicalNodes: 3,
		},
		{
			name: "partial last owner", family: xtBuildTopology, widths: []uint32{4, 4, 4}, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC16), physicalNodes: 2,
		},
		{
			name: "LP30 2x8c without topology metadata", family: hxBuildTopology, widths: []uint32{8, 8}, extent: ptr.To([]int64{16, 1, 1, 1}), physicalNodes: 1, omitMetadata: true,
		},
		{
			name: "LP30 4x4c", family: hxBuildTopology, widths: []uint32{4, 4, 4, 4}, extent: ptr.To([]int64{16, 1, 1, 1}), physicalNodes: 1,
		},
		{
			name: "LP30 partial last node in manifest order", family: hxBuildTopology, widths: []uint32{8, 8, 8}, ids: []uint32{11, 3, 9}, extent: ptr.To([]int64{16, 2, 1, 1}), physicalNodes: 2,
		},
		{name: "XT single subnode partition", family: xtBuildTopology, widths: []uint32{2}, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 1},
		{name: "LP30 single subnode stage", family: hxBuildTopology, widths: []uint32{8}, extent: ptr.To([]int64{16, 1, 1, 1}), physicalNodes: 1},
		{name: "LP30 two hemispheres", family: hxBuildTopology, widths: slices.Repeat([]uint32{8}, 20), extent: ptr.To([]int64{16, 5, 2, 1}), physicalNodes: 10},
		{name: "LP30 two racks", family: hxBuildTopology, widths: slices.Repeat([]uint32{8}, 64), extent: ptr.To([]int64{16, 8, 2, 2}), physicalNodes: 32},
		{name: "XT four-chip nodes", family: xtBuildTopology, widths: []uint32{2, 2, 2}, nodeWidth: 4, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 2},
		{name: "XT single stage across four-chip nodes", family: xtBuildTopology, widths: []uint32{8}, nodeWidth: 4, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC8), physicalNodes: 2},
		{name: "XT sixteen-node stage", family: xtBuildTopology, widths: []uint32{128}, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC128), physicalNodes: 16},
		{name: "XT thirty-two-chip nodes", family: xtBuildTopology, widths: slices.Repeat([]uint32{4}, 18), nodeWidth: 32, shape: ptr.To(lpxv1alpha1.Xt8888PartitionShapeC96), physicalNodes: 3},
		{name: "LP30 eight-chip nodes", family: hxBuildTopology, widths: []uint32{4, 4, 4}, nodeWidth: 8, extent: ptr.To([]int64{8, 2, 1, 1}), physicalNodes: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Encode source stages, manifest geometry and the selected chain")
			fixture := newV2CompilerFixture()
			if test.family == hxBuildTopology {
				fixture = newV3CompilerFixture()
			}
			width := fixture.partitions[0].devicesPerNode
			if test.nodeWidth != 0 {
				width = test.nodeWidth
			}
			fixture.cpuEmbeddings = test.cpu
			wantAgentReplicas := uint32(0)
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
				if test.family == hxBuildTopology {
					partition.topology = fmt.Sprintf("hx-stage-%d", partition.id)
					partition.topologyFamily = hxTopologyFamily
					partition.partitionShape = []uint32{min(chips, width), max(1, chips/width), 1, 1}
				}
				if index < len(test.topologies) {
					partition.topology = test.topologies[index]
				}
				if test.omitMetadata {
					partition.topologyFamily, partition.partitionShape = "", nil
				}
				fixture.selectedPropSyncChains[0][index] = partition.id
				wantAgentReplicas += max(1, chips/width)
			}
			if len(test.widths) == 1 {
				fixture.selectedPropSyncChains = nil
			}
			if test.chains != nil {
				fixture.selectedPropSyncChains = test.chains
			}
			pipeline := test.pipeline
			if pipeline == "" {
				pipeline = PipelineSingle
			}
			if pipeline == PipelineHybrid {
				fixture.compilationMode = manifestcapnp.CompilationMode_lpx
			}

			t.Log("Project each geometry once and reject unsupported workloads")
			buildDir := writeCompilerFixture(t, fixture)
			snapshot := acquireTestSnapshot(t, buildDir)
			normalized := snapshot
			before := *snapshot
			before.partitions = slices.Clone(snapshot.partitions)
			projections, err := appendModelProjections(nil, modelProjectionInput{
				Pipeline: pipeline, Models: []string{"default"}, Build: normalized,
			})
			if test.wantError != "" {
				require.ErrorContains(t, err, test.wantError)
				return
			}
			require.NoError(t, err)
			projection := projections[0]
			require.Equal(t, before, *normalized)
			original := normalized.partitions

			if test.physicalNodes == 0 {
				t.Log("Retain separate reservations and HX device-index connectors when packing does not apply")
				require.Equal(t, original, projection.partitions)
				require.EqualValues(t, wantAgentReplicas, projection.agentReplicas)
				edges := 0
				for _, chain := range fixture.selectedPropSyncChains {
					edges += len(chain) - 1
				}
				require.Len(t, projection.connectors, edges)
				if test.family == hxBuildTopology {
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
			require.Equal(t, original[test.runtimeStart:], projection.configuredBuild.partitions)
			request := projection.RequestSpec(&MaterializationPlan{}, "agents")
			require.Len(t, request.Partitions, 1)
			require.Equal(t, test.family.target(), request.TargetFamily)
			require.Equal(t, test.shape, request.Partitions[0].XtShape)
			require.Equal(t, test.extent, request.Partitions[0].Extent)
			require.EqualValues(t, fixture.partitions[test.runtimeStart].id, request.Partitions[0].CompilerPartitionID)
			require.Empty(t, request.PropSyncConnectors)
			require.Equal(t, []lpxv1alpha1.NodeLocalPartitionMapping{{ModelPartitionID: 0, PartitionID: "partition-000"}}, request.NodeLocal.PartitionMappings)
			if test.family == hxBuildTopology {
				extentJSON, err := json.Marshal(test.extent)
				require.NoError(t, err)
				require.JSONEq(t, fmt.Sprintf(`{"arch":"lp30","topology":"lyra","metadata_version":1,
					"partition_info":{"num_partitions":1,"%d":{"device":"lpu","allocation":%s}},
					"prop_sync_info":{"version":1,"prop_sync_pairs":[]}}`, fixture.partitions[0].id, extentJSON), string(request.AllocationMetadata.Raw))
			}
		})
	}
}
