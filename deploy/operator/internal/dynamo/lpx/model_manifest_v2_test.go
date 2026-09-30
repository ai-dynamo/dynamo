/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	"github.com/stretchr/testify/require"
)

// manifestV2Fields exposes the mutable sections of a contract fixture.
type manifestV2Fields struct {
	manifest   manifestcapnpv2.Manifest
	deployment manifestcapnpv2.DeploymentInfo
	program    manifestcapnpv2.ProgramConfig
	runtimeIO  manifestcapnpv2.RuntimeIoConfig
	artifacts  manifestcapnpv2.ArtifactInfo
	lpu        manifestcapnpv2.LpuPartitionArtifact
}

func newManifestV2Fields(t *testing.T) manifestV2Fields {
	t.Helper()
	fields := manifestV2Fields{manifest: newManifestV2ContractFixture(t)}
	var err error
	fields.deployment, err = fields.manifest.Deployment()
	require.NoError(t, err)
	fields.program, err = fields.deployment.Program()
	require.NoError(t, err)
	fields.runtimeIO, err = fields.deployment.RuntimeIo()
	require.NoError(t, err)
	fields.artifacts, err = fields.manifest.Artifacts()
	require.NoError(t, err)
	partitions, err := fields.artifacts.Partitions()
	require.NoError(t, err)
	fields.lpu, err = partitions.At(0).Detail().Lpu()
	require.NoError(t, err)
	return fields
}

func TestBuildFromGbuildManifestV2ProjectsRuntimeIO(t *testing.T) {
	t.Parallel()

	t.Log("Project the complete build contract from a valid revision 2 manifest")
	fields := newManifestV2Fields(t)
	build, err := buildFromGbuildManifestV2("gs://models/build", fields.manifest)
	require.NoError(t, err)
	require.EqualValues(t, 4, build.ioFPGACount)
	require.EqualValues(t, 2, build.ioFanoutFactor)
	require.Len(t, build.partitions, 1)

	t.Log("Accept compat FPGA I/O mode without changing the normalized build contract")
	fields.runtimeIO.SetReserved1(1)
	compat, err := buildFromGbuildManifestV2("gs://models/build", fields.manifest)
	require.NoError(t, err)
	require.Equal(t, build, compat)
}

func TestBuildFromGbuildManifestV2ValidatesContract(t *testing.T) {
	t.Parallel()

	withEmbeddings := func(path string, cpu, standalone bool) func(*testing.T, manifestV2Fields) {
		return func(t *testing.T, fields manifestV2Fields) {
			fields.deployment.SetNumLpuNodes(2)
			fields.program.SetSupportsCpuEmbeddings(cpu)
			fields.program.SetStandaloneTokenEmbeddings(standalone)
			partitions, err := fields.artifacts.NewPartitions(2)
			require.NoError(t, err)
			setManifestV2LPUArtifact(t, partitions.At(0), 0)
			setManifestV2LPUArtifact(t, partitions.At(1), 1)
			if path != "" {
				runtimeAssets, err := fields.artifacts.NewRuntimeAssets()
				require.NoError(t, err)
				require.NoError(t, runtimeAssets.SetTokenEmbeddingsPath(path))
			}
		}
	}

	t.Log("Define one contract mutation per case against an otherwise valid revision-2 manifest")
	for _, test := range []struct {
		name    string
		mutate  func(*testing.T, manifestV2Fields)
		wantErr string
	}{
		{
			name: "unsupported contract revision",
			mutate: func(_ *testing.T, f manifestV2Fields) {
				f.manifest.SetContractRevision(f.manifest.ContractRevision() + 1)
			},
			wantErr: "contractRevision",
		},
		{
			name:    "positive batch size",
			mutate:  func(_ *testing.T, f manifestV2Fields) { f.program.SetBatchSize(0) },
			wantErr: "deployment.program.batchSize must be >= 1, got 0",
		},
		{
			name:    "batch divisibility",
			mutate:  func(_ *testing.T, f manifestV2Fields) { f.program.SetBatchSize(3) },
			wantErr: "deployment.program.batchSize 3 must be divisible by deployment.runtimeIo.ioFpgaCount 4",
		},
		{
			name:    "fanout divisibility",
			mutate:  func(_ *testing.T, f manifestV2Fields) { f.runtimeIO.SetFanoutFactor(3) },
			wantErr: "deployment.program.batchSize per endpoint 2 must be divisible by deployment.runtimeIo.fanoutFactor 3",
		},
		{
			name:    "zero I/O FPGA count",
			mutate:  func(_ *testing.T, f manifestV2Fields) { f.runtimeIO.SetIoFpgaCount(0) },
			wantErr: "deployment.runtimeIo.ioFpgaCount must be in [1,",
		},
		{
			name:    "zero fanout factor",
			mutate:  func(_ *testing.T, f manifestV2Fields) { f.runtimeIO.SetFanoutFactor(0) },
			wantErr: "deployment.runtimeIo.fanoutFactor must be in [1,",
		},
		{
			name:    "host I/O with multiple endpoints",
			mutate:  func(_ *testing.T, f manifestV2Fields) { f.runtimeIO.SetProtocol(runtimeIOProtocolHost) },
			wantErr: "host runtime I/O requires ioFpgaCount 1, got 4",
		},
		{
			name:    "unsupported I/O protocol",
			mutate:  func(_ *testing.T, f manifestV2Fields) { f.runtimeIO.SetProtocol(runtimeIOProtocolMultiEndpoint + 1) },
			wantErr: "deployment.runtimeIo.protocol 2 is not supported",
		},
		{
			name:    "unsupported I/O mode",
			mutate:  func(_ *testing.T, f manifestV2Fields) { f.runtimeIO.SetReserved1(runtimeIOMaxMode + 1) },
			wantErr: "deployment.runtimeIo mode 3 is not supported",
		},
		{
			name: "partition traversal",
			mutate: func(t *testing.T, f manifestV2Fields) {
				require.NoError(t, f.lpu.SetPath("../outside"))
			},
			wantErr: "LPU partition 0 path",
		},
		{
			name:    "missing numChips",
			mutate:  func(_ *testing.T, f manifestV2Fields) { f.lpu.SetNumChips(0) },
			wantErr: "numChips must be >= 1",
		},
		{
			name:    "missing devicesPerNode",
			mutate:  func(_ *testing.T, f manifestV2Fields) { f.lpu.SetDevicesPerNode(0) },
			wantErr: "devicesPerNode must be >= 1",
		},
		{
			name:    "node count uses manifest device density",
			mutate:  func(_ *testing.T, f manifestV2Fields) { f.lpu.SetDevicesPerNode(4) },
			wantErr: "LPU partitions use 2 LPU nodes, want deployment.numLpuNodes 1",
		},
		{
			name: "non-LPU artifact for lpuOnly",
			mutate: func(t *testing.T, f manifestV2Fields) {
				f.deployment.SetCompilationMode(manifestcapnpv2.CompilationMode_lpuOnly)
				partitions, err := f.artifacts.NewPartitions(2)
				require.NoError(t, err)
				setManifestV2LPUArtifact(t, partitions.At(0), 0)
				cudaRef, err := partitions.At(1).NewPartition()
				require.NoError(t, err)
				cudaRef.SetPartitionId(1)
				cudaRef.SetDeviceType(manifestcapnpv2.DeviceType_cuda)
			},
			wantErr: "deployment.compilationMode lpuOnly requires every artifact partition to use deviceType lpu",
		},
		{
			name:    "runtime asset line break",
			mutate:  withEmbeddings("runtime\nasset", true, false),
			wantErr: "artifacts.runtimeAssets.tokenEmbeddingsPath",
		},
		{
			name:    "asset without CPU embeddings",
			mutate:  withEmbeddings("runtime/text_embeddings.npz", false, false),
			wantErr: "requires supportsCpuEmbeddings=true",
		},
		{
			name:    "missing standalone asset",
			mutate:  withEmbeddings("", true, true),
			wantErr: "is required when standaloneTokenEmbeddings=true",
		},
		{
			name:   "valid standalone asset",
			mutate: withEmbeddings("runtime/text_embeddings.npz", true, true),
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Accept only the safe and internally consistent revision-2 contract")
			fields := newManifestV2Fields(t)
			test.mutate(t, fields)
			_, err := buildFromGbuildManifestV2("gs://models/build", fields.manifest)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
		})
	}
}

func TestAcquireBuildRejectsInvalidHXArtifacts(t *testing.T) {
	t.Log("Define malformed HX artifact inventories")
	for _, test := range []struct {
		name    string
		mutate  func(*testV3CapnpFixture)
		wantErr string
	}{
		{
			name:    "duplicate LPU partition ID",
			mutate:  func(f *testV3CapnpFixture) { f.partitions = append(f.partitions, f.partitions[0]) },
			wantErr: "repeats LPU partition ID 1",
		},
		{
			name: "topology metadata node-count mismatch",
			mutate: func(f *testV3CapnpFixture) {
				f.partitions[0].topologyFamily = hxTopologyFamily
				f.partitions[0].partitionShape = []uint32{16, 1, 1, 1}
			},
			wantErr: "deployment.numLpuNodes = 2, but partition extents require 1 LPU nodes",
		},
		{
			name:    "partial build",
			mutate:  func(f *testV3CapnpFixture) { f.partSelect = true },
			wantErr: "partSelect builds are not supported",
		},
		{
			name:    "empty selected prop-sync chain",
			mutate:  func(f *testV3CapnpFixture) { f.selectedPropSyncChains = [][]uint32{{}} },
			wantErr: "deployment.selectedPropSyncChains[0] must contain at least two partitionIds",
		},
		{
			name:    "singleton selected prop-sync chain",
			mutate:  func(f *testV3CapnpFixture) { f.selectedPropSyncChains = [][]uint32{{1}} },
			wantErr: "deployment.selectedPropSyncChains[0] must contain at least two partitionIds",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Acquire stable manifest bytes with the selected malformed HX inventory")
			fixture := newV3CompilerFixture()
			test.mutate(&fixture)
			buildDir := writeCompilerFixture(t, fixture)

			t.Log("Reject malformed HX artifacts before publishing the acquired snapshot")
			snapshot, err := (&defaultModelRegistry{}).AcquireBuild(t.Context(), buildDir)
			require.ErrorIs(t, err, errInvalidBuildManifest)
			require.ErrorContains(t, err, test.wantErr)
			require.Nil(t, snapshot)
		})
	}
}

func TestBuildPartitionFromManifestV2SelectsFamily(t *testing.T) {
	t.Parallel()

	t.Log("Define XT and HX classification outcomes from manifest geometry and topology metadata")
	tests := []struct {
		name, family             string
		numChips, devicesPerNode uint32
		extent, wantExtent       []uint32
		wantNodes                int
		wantCompatible           bool
		wantErr                  string
	}{
		{name: "XT single node", numChips: 8, devicesPerNode: 8, wantNodes: 1},
		{name: "multi-node XT", numChips: 16, devicesPerNode: 8, wantNodes: 2},
		{name: "manifest device density", numChips: 8, devicesPerNode: 4, wantNodes: 2},
		{name: "sub-host XT geometry", numChips: 8, devicesPerNode: 16, wantNodes: 1},
		{name: "nonintegral XT geometry", numChips: 9, devicesPerNode: 8, wantErr: "9 chips, not divisible by 8 LPU devices per node"},
		{name: "metadata-less HX", numChips: 16, devicesPerNode: 16, wantExtent: []uint32{16, 1, 1, 1}, wantCompatible: true, wantNodes: 1},
		{name: "metadata HX", numChips: 16, devicesPerNode: 16, family: " " + hxTopologyFamily + " ", extent: []uint32{16, 1, 1, 1}, wantExtent: []uint32{16, 1, 1, 1}, wantNodes: 1},
		{name: "full HX geometry", numChips: 512, devicesPerNode: 16, family: hxTopologyFamily, extent: []uint32{16, 8, 2, 2}, wantExtent: []uint32{16, 8, 2, 2}, wantNodes: 32},
		{name: "metadata forbids XT fallback", numChips: 8, devicesPerNode: 8, family: hxTopologyFamily, extent: []uint32{16, 1, 1, 1}, wantErr: "contains 16 chips, want numChips 8"},
		{name: "unknown HX family", numChips: 16, devicesPerNode: 16, family: "unknown", extent: []uint32{16, 1, 1, 1}, wantErr: "topologyMetadata.topologyFamily"},
		{name: "missing HX extent", numChips: 16, devicesPerNode: 16, family: hxTopologyFamily, wantErr: "unsupported HX extent"},
		{name: "unsupported HX extent", numChips: 64, devicesPerNode: 16, family: hxTopologyFamily, extent: []uint32{16, 1, 2, 2}, wantErr: "unsupported HX extent"},
		{name: "HX devices per node mismatch", numChips: 16, devicesPerNode: 8, family: hxTopologyFamily, extent: []uint32{16, 1, 1, 1}, wantErr: "does not match devicesPerNode 8"},
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
				require.Equal(t, buildPartition{}, partition)
				require.False(t, compatible)
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.wantCompatible, compatible)
			require.Equal(t, 7, partition.sourcePartitionID)
			require.Equal(t, "part-0", partition.partPath)
			require.EqualValues(t, test.numChips, partition.numChips)
			require.EqualValues(t, test.devicesPerNode, partition.devicesPerNode)
			require.Equal(t, test.wantNodes, partition.effectiveNodeCount())
			require.Len(t, partition.hxExtent, len(test.wantExtent))
			for index, value := range test.wantExtent {
				require.EqualValues(t, value, partition.hxExtent[index])
			}

			t.Log("Reject unsafe paths for every supported partition family")
			require.NoError(t, detail.SetPath("../outside"))
			partition, compatible, err = buildPartitionFromManifestV2(raw)
			require.ErrorContains(t, err, "path")
			require.Equal(t, buildPartition{}, partition)
			require.False(t, compatible)
		})
	}
}

func TestManifestPartitionFamilyOrdering(t *testing.T) {
	t.Log("Classify HX partitions while preserving manifest order")
	hx := []buildPartition{
		{sourcePartitionID: 7, numChips: 16, devicesPerNode: 16, hxExtent: []int64{16, 1, 1, 1}},
		{sourcePartitionID: 3, numChips: 16, devicesPerNode: 16, hxExtent: []int64{16, 1, 1, 1}},
	}
	family, _, _, err := classifyManifestPartitions(hx, false)
	require.NoError(t, err)
	require.Equal(t, hxFamily, family)
	require.Equal(t, []int{7, 3}, []int{hx[0].sourcePartitionID, hx[1].sourcePartitionID})

	t.Log("Classify XT partitions while sorting by source partition identity")
	xt := []buildPartition{
		{sourcePartitionID: 7, numChips: 8, devicesPerNode: 8},
		{sourcePartitionID: 3, numChips: 8, devicesPerNode: 8},
	}
	family, _, _, err = classifyManifestPartitions(xt, false)
	require.NoError(t, err)
	require.Equal(t, xtFamily, family)
	require.Equal(t, []int{3, 7}, []int{xt[0].sourcePartitionID, xt[1].sourcePartitionID})
}

func TestBuildFromGbuildManifestV2ValidatesPartSelect(t *testing.T) {
	t.Parallel()

	t.Log("Define valid and invalid revision-2 partSelect inventories")
	tests := []struct {
		name        string
		selectedIDs []uint32
		wantErr     string
	}{
		{name: "valid", selectedIDs: []uint32{1, 0}},
		{
			name:        "duplicate",
			selectedIDs: []uint32{0, 0},
			wantErr:     "artifacts.partSelect has duplicate LPU partition id 0",
		},
		{
			name:        "unpackaged",
			selectedIDs: []uint32{0, 2},
			wantErr:     "artifacts.partSelect references unpackaged LPU partition id 2",
		},
		{
			name:        "incomplete",
			selectedIDs: []uint32{0},
			wantErr:     "artifacts.partSelect selects 1 LPU partitions, but artifacts package 2",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Build an unsorted two-partition V2 artifact inventory and explicit selector")
			manifest := newManifestV2ContractFixture(t)
			deployment, err := manifest.Deployment()
			require.NoError(t, err)
			deployment.SetNumLpuNodes(2)
			artifacts, err := manifest.Artifacts()
			require.NoError(t, err)
			partitions, err := artifacts.NewPartitions(2)
			require.NoError(t, err)
			setManifestV2LPUArtifact(t, partitions.At(0), 1)
			setManifestV2LPUArtifact(t, partitions.At(1), 0)
			partSelect, err := artifacts.NewPartSelect()
			require.NoError(t, err)
			selected, err := partSelect.NewPartitions(int32(len(test.selectedIDs)))
			require.NoError(t, err)
			for index, id := range test.selectedIDs {
				selected.At(index).SetPartitionId(id)
				selected.At(index).SetDeviceType(manifestcapnpv2.DeviceType_lpu)
			}

			t.Log("Validate against the sorted unique XT inventory produced by classification")
			build, err := buildFromGbuildManifestV2("gs://models/build", manifest)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
			require.Len(t, build.partitions, 2)
			require.Equal(t, 0, build.partitions[0].sourcePartitionID)
			require.Equal(t, 1, build.partitions[1].sourcePartitionID)
		})
	}
}
