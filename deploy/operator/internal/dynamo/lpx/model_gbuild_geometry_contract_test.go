/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGBuildGeometryContract(t *testing.T) {
	t.Log("Read the offline GBuild corpus and its geometry index")
	root := filepath.Join("testdata", "gbuild_geometry_contract")
	data, err := os.ReadFile(filepath.Join(root, "fixtures.json"))
	require.NoError(t, err)
	var index struct {
		Fixtures []struct {
			Name       string `json:"name"`
			SHA256     string `json:"sha256"`
			Partitions []struct {
				ID             int      `json:"id"`
				Topology       string   `json:"topology"`
				NumChips       int      `json:"num_chips"`
				DevicesPerNode int      `json:"devices_per_node"`
				SourceExtent   []uint32 `json:"source_extent"`
			} `json:"partitions"`
			SelectedChain []int `json:"selected_chain"`
		} `json:"fixtures"`
	}
	require.NoError(t, json.Unmarshal(data, &index))

	t.Log("Keep the scheduling and runtime expectations independent of the projector")
	want := map[string]struct {
		ids            []int
		chips          []int
		extents        [][]int64
		nodes, offsets string
		agents         int
		contentID      string
	}{
		"subnode8": {
			[]int{7}, []int{8}, [][]int64{{16, 1, 1, 1}}, "1", "0", 1,
			"sha256:89dbe9188c9040e09208c5285892f0b16a6814b6b3dbe1c337d27cf1bdec2585",
		},
		"cross_hemi32": {
			[]int{7}, []int{32}, [][]int64{{16, 1, 2, 1}}, "2", "0", 2,
			"sha256:b066521fbbf5fa728a5d436990a97e2c33ac3e2d16b176b91038ad5acf7fd7ec",
		},
		"rack256": {
			[]int{7}, []int{256}, [][]int64{{16, 8, 2, 1}}, "16", "0", 16,
			"sha256:f1f9b6d2d0ed0957f261dad2660150d6fd753abff79ff9f35bd5cd7608ca2d10",
		},
		"connected_chain": {
			[]int{7, 8}, []int{32, 16}, [][]int64{{16, 1, 2, 1}, {16, 1, 1, 1}}, "2\n1", "0\n2", 3,
			"sha256:e26ec31a28f2189eedde280da4aeac6d50063a740aab65e1d13432ac893e1441",
		},
	}
	require.Len(t, index.Fixtures, len(want))
	seen := make(map[string]bool)
	for _, fixture := range index.Fixtures {
		t.Run(fixture.Name, func(t *testing.T) {
			t.Log("Verify the frozen bytes and their compiler-owned facts before projection")
			expected, ok := want[fixture.Name]
			require.True(t, ok)
			require.False(t, seen[fixture.Name], "duplicate corpus entry")
			seen[fixture.Name] = true
			fixtureRoot, err := filepath.Abs(filepath.Join(root, fixture.Name))
			require.NoError(t, err)
			payload, err := os.ReadFile(filepath.Join(fixtureRoot, gbuildDeploymentV1CapnpFile))
			require.NoError(t, err)
			require.LessOrEqual(t, len(payload), 64*1024)
			require.Equal(t, fixture.SHA256, fmt.Sprintf("%x", sha256.Sum256(payload)))
			deployment, err := decodeGbuildDeploymentV1(payload)
			require.NoError(t, err)
			raw, err := deployment.Partitions()
			require.NoError(t, err)
			require.Equal(t, len(expected.ids), raw.Len())
			require.Len(t, fixture.Partitions, len(expected.ids))
			for i, source := range fixture.Partitions {
				ref, err := raw.At(i).Partition()
				require.NoError(t, err)
				require.EqualValues(t, expected.ids[i], ref.PartitionId())
				require.Equal(t, expected.ids[i], source.ID)
				require.Equal(t, expected.chips[i], source.NumChips)
				require.Equal(t, 16, source.DevicesPerNode)
				detail, err := raw.At(i).Detail().Lpu()
				require.NoError(t, err)
				topology, err := detail.Topology()
				require.NoError(t, err)
				require.Equal(t, v3OpaqueTopology, topology)
				require.Equal(t, v3OpaqueTopology, source.Topology)
				metadata, err := detail.TopologyMetadata()
				require.NoError(t, err)
				family, err := metadata.TopologyFamily()
				require.NoError(t, err)
				require.Equal(t, "HX16x8x2", family)
				shape, err := metadata.PartitionShape()
				require.NoError(t, err)
				require.Equal(t, 4, shape.Len())
				require.Len(t, source.SourceExtent, 4)
				for axis, value := range source.SourceExtent {
					require.Equal(t, value, shape.At(axis))
				}
			}

			t.Log("Acquire the fixture's exact source identity and project the LPU-only workload")
			build := acquireTestSnapshot(t, fixtureRoot)
			require.Equal(t, expected.contentID, build.contentID)
			require.Equal(t, build.contentID, acquireTestSnapshot(t, fixtureRoot).contentID)
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineSingle, Models: []string{"default"}, Build: build,
			})
			require.NoError(t, err)
			require.Len(t, projections, 1)
			projection := projections[0]
			require.Equal(t, expected.agents, projection.AgentReplicas())
			require.Equal(t, expected.contentID, projection.CompilerSnapshotDigest())
			spec := projection.RequestSpec(&MaterializationPlan{}, "agents")
			require.Len(t, spec.Partitions, len(expected.ids))
			require.Len(t, spec.NodeLocal.PartitionMappings, len(expected.ids))
			var ids, paths []string
			for i, partition := range spec.Partitions {
				require.EqualValues(t, expected.ids[i], partition.CompilerPartitionID)
				require.EqualValues(t, i, partition.Ordinal)
				require.Equal(t, fmt.Sprintf("partition-%03d", i), partition.ID)
				require.NotNil(t, partition.Extent)
				require.Equal(t, expected.extents[i], *partition.Extent)
				require.EqualValues(t, i, spec.NodeLocal.PartitionMappings[i].ModelPartitionID)
				require.Equal(t, partition.ID, spec.NodeLocal.PartitionMappings[i].PartitionID)
				require.Equal(t, expected.chips[i], build.partitions[i].numChips)
				require.Equal(t, expected.chips[i], projection.configuredBuild.partitions[i].numChips)
				require.Equal(t, fixture.Partitions[i].Topology, build.partitions[i].topology)
				require.Equal(t, fixture.Partitions[i].Topology, projection.configuredBuild.partitions[i].topology)
				ids = append(ids, strconv.Itoa(expected.ids[i]))
				paths = append(paths, fmt.Sprintf("part-%d", expected.ids[i]))
			}

			t.Log("Preserve aligned runtime rows and global Agent rank offsets")
			require.Equal(t, map[string]string{
				"nodes_per_partition":    expected.nodes,
				"partition_ids":          strings.Join(ids, "\n"),
				"partition_node_offsets": expected.offsets,
				"partition_paths":        strings.Join(paths, "\n"),
			}, resolvedPartitionData(projections))

			t.Log("Honor the compiler-selected chain at the final logical-node boundary")
			if fixture.Name != "connected_chain" {
				require.Empty(t, fixture.SelectedChain)
				require.Empty(t, spec.PropSyncConnectors)
				return
			}
			require.Equal(t, []int{7, 8}, fixture.SelectedChain)
			require.Len(t, spec.PropSyncConnectors, 1)
			connector := spec.PropSyncConnectors[0]
			require.Equal(t, "partition-000", connector.FromPartitionID)
			require.Equal(t, "partition-001", connector.ToPartitionID)
			require.NotNil(t, connector.Requirement.AcceptableLaneMultiplicities)
			require.Equal(t, []int64{4, 2, 1}, *connector.Requirement.AcceptableLaneMultiplicities)
			require.NotNil(t, connector.Requirement.Connections)
			require.Len(t, *connector.Requirement.Connections, 16)
			for i, connection := range *connector.Requirement.Connections {
				require.EqualValues(t, 16+i, connection.FromLogicalDevice)
				require.EqualValues(t, i, connection.ToLogicalDevice)
			}
		})
	}
}
