/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBuildPartitionFromManifestV2SelectsFamily(t *testing.T) {
	t.Parallel()

	t.Log("Define XT and HX classification outcomes from actual revision-2 artifacts")
	tests := []struct {
		name, topology, family   string
		numChips, devicesPerNode uint32
		extent, wantExtent       []uint32
		wantCompatible           bool
		wantErr                  string
	}{
		{name: "ordinary XT", topology: registryTestTopology, numChips: 8, devicesPerNode: 8},
		{name: "multi-node XT", topology: strings.Replace(registryTestTopology, "8C", "16C", 1), numChips: 16, devicesPerNode: 8},
		{name: "metadata-less HX opaque topology", topology: " " + v3OpaqueTopology + " ", numChips: 16, devicesPerNode: 16, wantExtent: []uint32{16, 1, 1, 1}, wantCompatible: true},
		{name: "metadata-less HX XT-looking topology", topology: registryTestTopology, numChips: 16, devicesPerNode: 16, wantExtent: []uint32{16, 1, 1, 1}, wantCompatible: true},
		{name: "metadata HX opaque topology", topology: v3OpaqueTopology, numChips: 16, devicesPerNode: 16, family: " " + hxTopologyFamily + " ", extent: []uint32{16, 1, 1, 1}, wantExtent: []uint32{16, 1, 1, 1}},
		{name: "metadata HX XT-looking topology", topology: registryTestTopology, numChips: 16, devicesPerNode: 16, family: hxTopologyFamily, extent: []uint32{16, 1, 1, 1}, wantExtent: []uint32{16, 1, 1, 1}},
		{name: "full HX geometry", topology: v3OpaqueTopology, numChips: 512, devicesPerNode: 16, family: hxTopologyFamily, extent: []uint32{16, 8, 2, 2}, wantExtent: []uint32{16, 8, 2, 2}},
		{name: "metadata forbids XT fallback", topology: registryTestTopology, numChips: 8, devicesPerNode: 8, family: hxTopologyFamily, extent: []uint32{16, 1, 1, 1}, wantErr: "contains 16 chips, want numChips 8"},
		{name: "unknown HX family", topology: v3OpaqueTopology, numChips: 16, devicesPerNode: 16, family: "unknown", extent: []uint32{16, 1, 1, 1}, wantErr: "topologyMetadata.topologyFamily"},
		{name: "missing HX extent", topology: v3OpaqueTopology, numChips: 16, devicesPerNode: 16, family: hxTopologyFamily, wantErr: "unsupported HX extent"},
		{name: "unsupported HX extent", topology: v3OpaqueTopology, numChips: 64, devicesPerNode: 16, family: hxTopologyFamily, extent: []uint32{16, 1, 2, 2}, wantErr: "unsupported HX extent"},
		{name: "HX devices per node mismatch", topology: v3OpaqueTopology, numChips: 16, devicesPerNode: 8, family: hxTopologyFamily, extent: []uint32{16, 1, 1, 1}, wantErr: "does not match devicesPerNode 8"},
		{name: "opaque topology needs HX geometry", topology: v3OpaqueTopology, numChips: 16, devicesPerNode: 8, wantErr: "parsing"},
		{name: "metadata-less HX wrong chip count", topology: v3OpaqueTopology, numChips: 8, devicesPerNode: 16, wantErr: "does not match expected pattern"},
		{name: "metadata-less HX empty topology", topology: " ", numChips: 16, devicesPerNode: 16, wantErr: "topology must not be empty"},
		{name: "metadata-less HX unsafe topology", topology: v3OpaqueTopology + "\n", numChips: 16, devicesPerNode: 16, wantErr: "must not contain NUL bytes or line breaks"},
		{name: "metadata HX NUL topology", topology: hxTopologyFamily + "\x00other", numChips: 16, devicesPerNode: 16, family: hxTopologyFamily, extent: []uint32{16, 1, 1, 1}, wantErr: "topology must not contain NUL bytes or line breaks"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Construct a manifest partition with the selected geometry")
			manifest := newManifestV2ContractFixture(t)
			artifacts, err := manifest.Artifacts()
			require.NoError(t, err)
			partitions, err := artifacts.Partitions()
			require.NoError(t, err)
			raw := partitions.At(0)
			setManifestV2LPUArtifact(t, raw, 7)
			detail, err := raw.Detail().Lpu()
			require.NoError(t, err)
			require.NoError(t, detail.SetTopology(test.topology))
			detail.SetNumChips(test.numChips)
			detail.SetDevicesPerNode(test.devicesPerNode)
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

			t.Log("Decode exactly the supported geometry and preserve the compatibility flag")
			partition, compatible, err := buildPartitionFromManifestV2(raw)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				require.Equal(t, BuildPartition{}, partition)
				require.False(t, compatible)
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.wantCompatible, compatible)
			require.Equal(t, 7, partition.SourcePartitionID)
			require.Equal(t, "part-0", partition.PartPath)
			require.Equal(t, strings.TrimSpace(test.topology), partition.Topology.Raw)
			require.EqualValues(t, test.numChips, partition.Topology.ChipCount)
			require.Len(t, partition.HXExtent, len(test.wantExtent))
			for index, value := range test.wantExtent {
				require.EqualValues(t, value, partition.HXExtent[index])
			}

			t.Log("Reject unsafe paths for every supported partition family")
			require.NoError(t, detail.SetPath("../outside"))
			partition, compatible, err = buildPartitionFromManifestV2(raw)
			require.ErrorContains(t, err, "path")
			require.Equal(t, BuildPartition{}, partition)
			require.False(t, compatible)
		})
	}
}

func TestManifestPartitionFamilyOrdering(t *testing.T) {
	t.Log("Classify HX partitions while preserving manifest order")
	hx := []BuildPartition{{SourcePartitionID: 7, HXExtent: []int64{16, 1, 1, 1}}, {SourcePartitionID: 3, HXExtent: []int64{16, 1, 1, 1}}}
	family, _, _, err := classifyManifestPartitions(gbuildManifestV2CapnpFile, hx, false)
	require.NoError(t, err)
	require.Equal(t, BuildFamilyHX, family)
	require.Equal(t, []int{7, 3}, []int{hx[0].SourcePartitionID, hx[1].SourcePartitionID})

	t.Log("Classify XT partitions while sorting by source partition identity")
	xt := []BuildPartition{{SourcePartitionID: 7}, {SourcePartitionID: 3}}
	family, _, _, err = classifyManifestPartitions(gbuildManifestV2CapnpFile, xt, false)
	require.NoError(t, err)
	require.Equal(t, BuildFamilyXT, family)
	require.Equal(t, []int{3, 7}, []int{xt[0].SourcePartitionID, xt[1].SourcePartitionID})
}
