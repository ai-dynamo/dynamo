/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"capnproto.org/go/capnp/v3"
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	"github.com/stretchr/testify/require"
)

func TestBuildPartitionFromManifestV2SelectsFamily(t *testing.T) {
	t.Parallel()

	t.Log("Define XT and HX classification outcomes from actual revision-2 artifacts")
	tests := []struct {
		name, topology, family   string
		numChips, devicesPerNode uint32
		extent, wantExtent       []uint32
		architectures            [][]string
		wantErr                  string
	}{
		{name: "opaque XT", architectures: [][]string{{"polaris"}}, topology: " opaque__xt ", numChips: 8, devicesPerNode: 8},
		{name: "multi-node XT uses numChips", architectures: [][]string{{"polaris"}}, topology: registryTestTopology, numChips: 16, devicesPerNode: 8},
		{name: "incomplete multi-node XT", architectures: [][]string{{"polaris"}}, topology: registryTestTopology, numChips: 9, devicesPerNode: 8, wantErr: "not divisible by 8"},
		{name: "empty XT topology", architectures: [][]string{{"polaris"}}, topology: " ", numChips: 8, devicesPerNode: 8, wantErr: "topology must not be empty"},
		{name: "unsafe XT topology", architectures: [][]string{{"polaris"}}, topology: "opaque\rname", numChips: 8, devicesPerNode: 8, wantErr: "must not contain NUL bytes or line breaks"},
		{name: "XT with HX fixture name and node width", topology: v3OpaqueTopology, numChips: 16, devicesPerNode: 16, architectures: [][]string{{"polaris"}}},
		{name: "missing programs", topology: registryTestTopology, architectures: [][]string{}, wantErr: "programs are missing or invalid"},
		{name: "empty program", topology: registryTestTopology, architectures: [][]string{{}}, wantErr: "program chips are missing or invalid"},
		{name: "unknown architecture", topology: registryTestTopology, architectures: [][]string{{"v3u"}}, wantErr: "unsupported chip architecture"},
		{name: "metadata-less HX opaque topology", topology: " opaque__hx ", numChips: 16, devicesPerNode: 16, wantExtent: []uint32{16, 1, 1, 1}},
		{name: "metadata HX opaque topology", topology: v3OpaqueTopology, numChips: 16, devicesPerNode: 16, family: " " + hxTopologyFamily + " ", extent: []uint32{16, 1, 1, 1}, wantExtent: []uint32{16, 1, 1, 1}},
		{name: "metadata HX with XT fixture name", topology: registryTestTopology, numChips: 8, devicesPerNode: 8, family: hxTopologyFamily, extent: []uint32{8, 1, 1, 1}, wantExtent: []uint32{8, 1, 1, 1}},
		{name: "full HX geometry", topology: v3OpaqueTopology, numChips: 512, devicesPerNode: 16, family: hxTopologyFamily, extent: []uint32{16, 8, 2, 2}, wantExtent: []uint32{16, 8, 2, 2}},
		{name: "unknown HX family", topology: v3OpaqueTopology, numChips: 16, devicesPerNode: 16, family: "unknown", extent: []uint32{16, 1, 1, 1}, wantErr: "topologyMetadata.topologyFamily"},
		{name: "missing HX extent", topology: v3OpaqueTopology, numChips: 16, devicesPerNode: 16, family: hxTopologyFamily, wantErr: "unsupported HX extent"},
		{name: "unsupported HX extent", topology: v3OpaqueTopology, numChips: 64, devicesPerNode: 16, family: hxTopologyFamily, extent: []uint32{16, 1, 2, 2}, wantErr: "unsupported HX extent"},
		{name: "metadata-less HX four-chip node", topology: v3OpaqueTopology, numChips: 4, devicesPerNode: 4, wantExtent: []uint32{4, 1, 1, 1}},
		{name: "metadata-less HX subnode", topology: v3OpaqueTopology, numChips: 8, devicesPerNode: 16, wantExtent: []uint32{16, 1, 1, 1}},
		{name: "metadata-less HX needs geometry for multiple nodes", topology: v3OpaqueTopology, numChips: 32, devicesPerNode: 16, wantErr: "without topologyMetadata requires a one-node stage"},
		{name: "metadata-less HX zero chips", topology: v3OpaqueTopology, devicesPerNode: 16, wantErr: "chip count divides 16"},
		{name: "metadata-less HX nondivisor", topology: v3OpaqueTopology, numChips: 3, devicesPerNode: 16, wantErr: "chip count divides 16"},
		{name: "metadata HX zero chips", topology: v3OpaqueTopology, devicesPerNode: 16, family: hxTopologyFamily, extent: []uint32{0, 1, 1, 1}, wantErr: "unsupported HX extent"},
		{name: "metadata HX nondivisor", topology: v3OpaqueTopology, numChips: 3, devicesPerNode: 16, family: hxTopologyFamily, extent: []uint32{3, 1, 1, 1}, wantErr: "unsupported HX extent"},
		{name: "partial HX shape cannot span nodes", topology: v3OpaqueTopology, numChips: 16, devicesPerNode: 16, family: hxTopologyFamily, extent: []uint32{8, 2, 1, 1}, wantErr: "unsupported HX extent"},
		{name: "partial HX chip count mismatch", topology: v3OpaqueTopology, numChips: 8, devicesPerNode: 16, family: hxTopologyFamily, extent: []uint32{4, 1, 1, 1}, wantErr: "contains 4 chips, want numChips 8"},
		{name: "HX missing node width", topology: v3OpaqueTopology, numChips: 16, family: hxTopologyFamily, extent: []uint32{16, 1, 1, 1}, wantErr: "devicesPerNode must be >= 1"},
		{name: "metadata-less HX unsafe topology", topology: v3OpaqueTopology + "\n", numChips: 16, devicesPerNode: 16, wantErr: "must not contain NUL bytes or line breaks"},
		{name: "metadata HX NUL topology", topology: hxTopologyFamily + "\x00other", numChips: 16, devicesPerNode: 16, family: hxTopologyFamily, extent: []uint32{16, 1, 1, 1}, wantErr: "topology must not contain NUL bytes or line breaks"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Construct a manifest partition with the selected geometry")
			_, segment := capnp.NewSingleSegmentMessage(nil)
			raw, err := manifestcapnp.NewRootPartitionInfo(segment)
			require.NoError(t, err)
			setManifestV2LPUArtifact(t, raw, 7)
			detail, err := raw.Detail().Lpu()
			require.NoError(t, err)
			require.NoError(t, detail.SetTopology(test.topology))
			detail.SetNumChips(test.numChips)
			detail.SetDevicesPerNode(test.devicesPerNode)
			architectures := test.architectures
			if architectures == nil {
				architectures = [][]string{{"polarisB0"}}
			}
			setManifestChipArchitectures(t, detail, architectures...)
			if test.family != "" {
				metadata, err := detail.NewTopologyMetadata()
				require.NoError(t, err)
				require.NoError(t, metadata.SetTopologyFamily(test.family))
				extent, err := metadata.NewPartitionShape(int32(len(test.extent)))
				require.NoError(t, err)
				for index, value := range test.extent {
					extent.Set(index, value)
				}
			}

			t.Log("Decode exactly the supported geometry")
			partition, err := buildPartitionFromManifestV2(raw)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				require.Equal(t, buildPartition{}, partition)
				return
			}
			require.NoError(t, err)
			require.Equal(t, 7, partition.sourcePartitionID)
			require.Equal(t, "part-0", partition.partPath)
			require.Equal(t, test.topology, partition.topology)
			require.EqualValues(t, test.numChips, partition.numChips)
			require.EqualValues(t, test.devicesPerNode, partition.devicesPerNode)
			require.Len(t, partition.hxExtent, len(test.wantExtent))
			for index, value := range test.wantExtent {
				require.EqualValues(t, value, partition.hxExtent[index])
			}
		})
	}
}

func TestManifestPartitionsRequireUniformDeviceCounts(t *testing.T) {
	t.Parallel()

	t.Log("Cover uniform and mixed device counts for both LPU families")
	for _, test := range []struct {
		name       string
		family     BuildFamily
		nodeWidths [2]int
		wantErr    string
	}{
		{name: "XT uniform", family: BuildFamilyXT, nodeWidths: [2]int{8, 8}},
		{name: "HX uniform", family: BuildFamilyHX, nodeWidths: [2]int{16, 16}},
		{name: "XT mixed", family: BuildFamilyXT, nodeWidths: [2]int{8, 16}, wantErr: "LPU partition 3 has devicesPerNode 16, want 8"},
		{name: "HX mixed", family: BuildFamilyHX, nodeWidths: [2]int{16, 8}, wantErr: "LPU partition 3 has devicesPerNode 8, want 16"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Build partitions with different chip counts and manifest node widths")
			partitions := []buildPartition{
				{sourcePartitionID: 7, numChips: test.nodeWidths[0], devicesPerNode: test.nodeWidths[0]},
				{sourcePartitionID: 3, numChips: 2 * test.nodeWidths[1], devicesPerNode: test.nodeWidths[1]},
			}
			if test.family == BuildFamilyHX {
				partitions[0].hxExtent = []int64{int64(test.nodeWidths[0]), 1, 1, 1}
				partitions[1].hxExtent = []int64{int64(test.nodeWidths[1]), 2, 1, 1}
			}

			t.Log("Reject mixed device counts before deriving runtime projections")
			family, err := classifyManifestPartitions(gbuildManifestV2CapnpFile, partitions, false)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.family, family)
			require.Equal(t, 3, partitions[0].effectiveNodeCount()+partitions[1].effectiveNodeCount())
		})
	}
}

func TestManifestPartitionFamilyOrdering(t *testing.T) {
	t.Log("Classify HX partitions while preserving manifest order")
	hx := []buildPartition{
		{sourcePartitionID: 7, numChips: 16, devicesPerNode: 16, hxExtent: []int64{16, 1, 1, 1}},
		{sourcePartitionID: 3, numChips: 16, devicesPerNode: 16, hxExtent: []int64{16, 1, 1, 1}},
	}
	family, err := classifyManifestPartitions(gbuildManifestV2CapnpFile, hx, false)
	require.NoError(t, err)
	require.Equal(t, BuildFamilyHX, family)
	require.Equal(t, []int{7, 3}, []int{hx[0].sourcePartitionID, hx[1].sourcePartitionID})

	t.Log("Classify XT partitions while sorting by source partition identity")
	xt := []buildPartition{
		{sourcePartitionID: 7, numChips: 8, devicesPerNode: 8},
		{sourcePartitionID: 3, numChips: 8, devicesPerNode: 8},
	}
	family, err = classifyManifestPartitions(gbuildManifestV2CapnpFile, xt, false)
	require.NoError(t, err)
	require.Equal(t, BuildFamilyXT, family)
	require.Equal(t, []int{3, 7}, []int{xt[0].sourcePartitionID, xt[1].sourcePartitionID})
}
