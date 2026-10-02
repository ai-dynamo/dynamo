/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// The fixture is shared with lpu-monorepo/Groq/IE/rust/lpx-runtime/tests/fixtures/partitions.json.
// It compares the prior runtime descriptors with the manifest resolver's contract.
func TestRuntimeDescriptorConformance(t *testing.T) {
	t.Parallel()
	t.Log("Read the shared native-runtime conformance cases")
	data, err := os.ReadFile("testdata/runtime-partitions.json")
	require.NoError(t, err)
	var cases []struct {
		Name      string `json:"name"`
		Artifacts []struct {
			ID      int     `json:"id"`
			Path    string  `json:"path"`
			Chips   int     `json:"chips"`
			HXShape []int64 `json:"hx_shape"`
			Devices int     `json:"devices"`
		} `json:"artifacts"`
		Chains   [][]int `json:"chains"`
		Selected []int   `json:"selected"`
		Hybrid   bool    `json:"hybrid"`
		Expected []struct {
			ID    int    `json:"source_partition_id"`
			Path  string `json:"relative_path"`
			Nodes int    `json:"node_count"`
		} `json:"expected"`
		Error string `json:"error"`
	}
	require.NoError(t, json.Unmarshal(data, &cases))
	for _, test := range cases {
		if test.Error != "" {
			continue
		}
		t.Run(test.Name, func(t *testing.T) {
			t.Log("Construct physical compiler partitions using the shared geometry")
			build := Build{Family: BuildFamilyXT}
			for _, artifact := range test.Artifacts {
				topology, err := parse("URSA_V2__Q8__" + strconv.Itoa(artifact.Chips) + "C__G_96_25__KP_FEC__GHZ_1_0__NO_FPGA")
				require.NoError(t, err)
				part := BuildPartition{SourcePartitionID: artifact.ID, PartPath: artifact.Path, Topology: topology, HXExtent: artifact.HXShape}
				if artifact.Devices == 16 {
					build.Family = BuildFamilyHX
					if len(part.HXExtent) == 0 {
						part.HXExtent = []int64{16, 1, 1, 1}
					}
				}
				build.Partitions = append(build.Partitions, part)
			}

			t.Log("Compare selected descriptors after the operator's existing chain collapse")
			selected := Build{Family: build.Family}
			for _, id := range test.Selected {
				index := slices.IndexFunc(build.Partitions, func(part BuildPartition) bool { return part.SourcePartitionID == id })
				require.NotEqual(t, -1, index)
				part := build.Partitions[index]
				if test.Hybrid && build.Family == BuildFamilyXT {
					for _, chain := range test.Chains {
						if chain[0] == id {
							part, err = collapseSelectedPropSyncChain(chain, build.Partitions[index:index+len(chain)])
							require.NoError(t, err)
						}
					}
				}
				selected.Partitions = append(selected.Partitions, part)
			}
			projection := &ModelProjection{configuredBuild: selected, pipeline: PipelineLPX}
			legacy := resolvedPartitionData([]*ModelProjection{projection})
			var ids, paths, counts, offsets []string
			offset := 0
			for _, expected := range test.Expected {
				ids = append(ids, strconv.Itoa(expected.ID))
				paths = append(paths, expected.Path)
				counts = append(counts, strconv.Itoa(expected.Nodes))
				offsets = append(offsets, strconv.Itoa(offset))
				offset += expected.Nodes
			}
			require.Equal(t, strings.Join(ids, ","), runtimePartitionIDs(projection))
			require.Equal(t, strings.Join(ids, "\n"), legacy["partition_ids"])
			require.Equal(t, strings.Join(paths, "\n"), legacy["partition_paths"])
			require.Equal(t, strings.Join(counts, "\n"), legacy["nodes_per_partition"])
			require.Equal(t, strings.Join(offsets, "\n"), legacy["partition_node_offsets"])

			if build.Family == BuildFamilyXT {
				t.Log("Compare old leader suffixes with the same descriptor offsets across replicas")
				workload := &Workload{modelProjections: []*ModelProjection{projection}}
				plan := &MaterializationPlan{ResourcePrefix: "fixture", ScalingGroupTemplate: "work-b", Agents: []ExpectedAgent{{TemplateName: "draft1"}}}
				legacyHosts, _, err := workload.renderCyborgConfigMap(plan)
				require.NoError(t, err)
				for _, replica := range []string{"0", "3"} {
					var hosts []string
					for _, offset := range offsets {
						hosts = append(hosts, "work-b-"+replica+"-draft1-"+offset)
					}
					require.Equal(t, strings.Join(hosts, "\n"), strings.ReplaceAll(legacyHosts.Data["lpu_servers"], "${GROVE_PCSG_INDEX}", replica))
				}
			}
		})
	}
}
