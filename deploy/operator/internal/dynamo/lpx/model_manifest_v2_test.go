/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"capnproto.org/go/capnp/v3"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/capnp/gbuild_manifest/v2"
	"github.com/stretchr/testify/require"
)

func TestBuildFromGbuildManifestV2ProjectsRuntimeIO(t *testing.T) {
	t.Parallel()

	t.Log("Build a valid revision 2 manifest")
	manifest := newManifestV2ContractFixture(t)

	t.Log("Project the complete build contract")
	build, err := buildFromGbuildManifestV2("gs://models/build", manifest)
	require.NoError(t, err)
	require.EqualValues(t, 4, build.IOFPGACount)
	require.EqualValues(t, 2, build.IOFanoutFactor)
	require.Len(t, build.Partitions, 1)

	t.Log("Accept compat FPGA I/O mode without changing the normalized build contract")
	deployment, err := manifest.Deployment()
	require.NoError(t, err)
	runtimeIO, err := deployment.RuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetReserved1(1)
	compat, err := buildFromGbuildManifestV2("gs://models/build", manifest)
	require.NoError(t, err)
	require.Equal(t, build, compat)

	t.Log("Reject an unsupported contract revision before lowering the manifest")
	manifest.SetContractRevision(manifest.ContractRevision() + 1)
	_, err = buildFromGbuildManifestV2("gs://models/build", manifest)
	require.ErrorContains(t, err, "contractRevision")
}

func TestNormalizeBuildSnapshotRejectsInvalidHXArtifacts(t *testing.T) {
	t.Log("Define malformed HX artifact inventories")
	tests := []struct {
		name    string
		wantErr string
	}{
		{name: "duplicate LPU partition ID", wantErr: "repeats LPU partition ID 1"},
		{name: "topology metadata node-count mismatch", wantErr: "deployment.numLpuNodes = 2, but partition extents require 1 LPU nodes"},
		{name: "partial build", wantErr: "partSelect builds are not supported"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Acquire stable manifest bytes with the selected malformed HX inventory")
			fixture := newV3CompilerFixture()
			switch test.name {
			case "duplicate LPU partition ID":
				fixture.partitions = append(fixture.partitions, fixture.partitions[0])
			case "topology metadata node-count mismatch":
				fixture.partitions[0].topology = v3HXTopologyFamily
				fixture.partitions[0].topologyFamily = v3HXTopologyFamily
				fixture.partitions[0].partitionShape = []uint32{16, 1, 1, 1}
			case "partial build":
				fixture.partSelect = true
			}
			snapshot := acquireTestSnapshot(t, writeCompilerFixture(t, fixture))

			t.Log("Reject malformed HX artifacts during manifest normalization")
			_, err := normalizeBuildSnapshot(snapshot)
			require.ErrorContains(t, err, test.wantErr)
		})
	}
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
			require.Len(t, build.Partitions, 2)
			require.Equal(t, 0, build.Partitions[0].SourcePartitionID)
			require.Equal(t, 1, build.Partitions[1].SourcePartitionID)
		})
	}
}

func TestBuildFromGbuildManifestV2ValidatesRuntimeInvariants(t *testing.T) {
	t.Parallel()

	t.Log("Define invalid revision-2 runtime I/O contracts")
	tests := []struct {
		name                             string
		batchSize, ioCount, fanoutFactor uint32
		wantErr                          string
	}{
		{
			name: "positive batch size", batchSize: 0, ioCount: 4, fanoutFactor: 2,
			wantErr: "deployment.program.batchSize must be >= 1, got 0",
		},
		{
			name: "batch divisibility", batchSize: 3, ioCount: 4, fanoutFactor: 2,
			wantErr: "deployment.program.batchSize 3 must be divisible by deployment.runtimeIo.ioFpgaCount 4",
		},
		{
			name: "fanout divisibility", batchSize: 8, ioCount: 4, fanoutFactor: 3,
			wantErr: "deployment.program.batchSize per endpoint 2 must be divisible by deployment.runtimeIo.fanoutFactor 3",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Build a manifest with malformed runtime metadata")
			manifest := newManifestV2ContractFixture(t)
			deployment, err := manifest.Deployment()
			require.NoError(t, err)
			program, err := deployment.Program()
			require.NoError(t, err)
			program.SetBatchSize(test.batchSize)
			runtimeIO, err := deployment.RuntimeIo()
			require.NoError(t, err)
			runtimeIO.SetIoFpgaCount(test.ioCount)
			runtimeIO.SetFanoutFactor(test.fanoutFactor)

			t.Log("Reject it before scheduler-facing artifact projection")
			_, err = buildFromGbuildManifestV2("gs://models/build", manifest)
			require.ErrorContains(t, err, test.wantErr)
		})
	}
}

func TestBuildFromGbuildManifestV2ValidatesV2OnlyContracts(t *testing.T) {
	t.Parallel()

	type geometryFixture struct {
		numChips, devicesPerNode, numNodes uint32
	}
	type embeddingFixture struct {
		path            string
		cpu, standalone bool
	}

	t.Log("Define revision-2 path, XT geometry, node-count, and embedding contracts")
	tests := []struct {
		name, partitionPath, tokenizerPath, wantErr, wantAsset string
		geometry                                               *geometryFixture
		embedding                                              *embeddingFixture
	}{
		{name: "partition traversal", partitionPath: "../outside", wantErr: "LPU partition 0 path"},
		{name: "absolute tokenizer", tokenizerPath: "/outside", wantErr: "model.tokenizer.path"},
		{
			name: "runtime asset line break", embedding: &embeddingFixture{path: "runtime\nasset", cpu: true},
			wantErr: "artifacts.runtimeAssets.tokenEmbeddingsPath",
		},
		{name: "missing numChips", geometry: &geometryFixture{devicesPerNode: 8, numNodes: 1}, wantErr: "numChips must be >= 1"},
		{
			name: "topology and numChips mismatch", geometry: &geometryFixture{numChips: 16, devicesPerNode: 8, numNodes: 2},
			wantErr: "topology chip count 8 does not match numChips 16",
		},
		{name: "missing devicesPerNode", geometry: &geometryFixture{numChips: 8, numNodes: 1}, wantErr: "devicesPerNode must be >= 1"},
		{
			name: "deployment node count mismatch", geometry: &geometryFixture{numChips: 8, devicesPerNode: 8, numNodes: 2},
			wantErr: "LPU partitions use 1 LPU nodes, want deployment.numLpuNodes 2",
		},
		{
			name: "asset without CPU embeddings", embedding: &embeddingFixture{path: "runtime/text_embeddings.npz"},
			wantErr: "requires supportsCpuEmbeddings=true",
		},
		{
			name: "missing standalone asset", embedding: &embeddingFixture{cpu: true, standalone: true},
			wantErr: "is required when standaloneTokenEmbeddings=true",
		},
		{
			name: "valid standalone asset", embedding: &embeddingFixture{path: "runtime/text_embeddings.npz", cpu: true, standalone: true},
			wantAsset: "runtime/text_embeddings.npz",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Apply one contract mutation to an otherwise valid revision-2 manifest")
			manifest := newManifestV2ContractFixture(t)
			if test.partitionPath != "" || test.geometry != nil {
				artifacts, err := manifest.Artifacts()
				require.NoError(t, err)
				partitions, err := artifacts.Partitions()
				require.NoError(t, err)
				detail, err := partitions.At(0).Detail().Lpu()
				require.NoError(t, err)
				if test.partitionPath != "" {
					require.NoError(t, detail.SetPath(test.partitionPath))
				}
				if test.geometry != nil {
					detail.SetNumChips(test.geometry.numChips)
					detail.SetDevicesPerNode(test.geometry.devicesPerNode)
					deployment, err := manifest.Deployment()
					require.NoError(t, err)
					deployment.SetNumLpuNodes(test.geometry.numNodes)
				}
			}
			if test.tokenizerPath != "" {
				model, err := manifest.NewModel()
				require.NoError(t, err)
				tokenizer, err := model.NewTokenizer()
				require.NoError(t, err)
				require.NoError(t, tokenizer.SetPath(test.tokenizerPath))
			}
			if test.embedding != nil {
				deployment, err := manifest.Deployment()
				require.NoError(t, err)
				deployment.SetNumLpuNodes(2)
				program, err := deployment.Program()
				require.NoError(t, err)
				program.SetSupportsCpuEmbeddings(test.embedding.cpu)
				program.SetStandaloneTokenEmbeddings(test.embedding.standalone)
				artifacts, err := manifest.Artifacts()
				require.NoError(t, err)
				partitions, err := artifacts.NewPartitions(2)
				require.NoError(t, err)
				setManifestV2LPUArtifact(t, partitions.At(0), 0)
				setManifestV2LPUArtifact(t, partitions.At(1), 1)
				if test.embedding.path != "" {
					runtimeAssets, err := artifacts.NewRuntimeAssets()
					require.NoError(t, err)
					require.NoError(t, runtimeAssets.SetTokenEmbeddingsPath(test.embedding.path))
				}
			}

			t.Log("Accept only the safe and internally consistent revision-2 contract")
			build, err := buildFromGbuildManifestV2("gs://models/build", manifest)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.wantAsset, build.RuntimeTokenEmbeddingsPath)
		})
	}
}

func TestNormalizedV2BuildTreatsManifestBuildIDAsOpaque(t *testing.T) {
	t.Parallel()

	t.Log("Define relocated revision-2 build references and opaque manifest identities")
	tests := []struct {
		name    string
		ref     string
		buildID string
	}{
		{
			name:    "local copy",
			ref:     "file:///models/copied-build",
			buildID: "original-producer-id",
		},
		{
			name:    "different GCS bucket and path",
			ref:     "gs://new-bucket/relocated/renamed-build",
			buildID: "original-producer-id",
		},
		{
			name: "missing producer identity",
			ref:  "file:///models/build-without-producer-id",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Create a revision-2 snapshot under its deployment locator")
			manifest := newManifestV2ContractFixture(t)
			if test.buildID != "" {
				build, err := capnp.NewStruct(manifest.Segment(), capnp.ObjectSize{DataSize: 8, PointerCount: 12})
				require.NoError(t, err)
				require.NoError(t, build.SetText(11, test.buildID))
				require.NoError(t, manifest.SetReserved3(build.ToPtr()))
			}
			payload, err := manifest.Message().Marshal()
			require.NoError(t, err)
			snapshot := &BuildSnapshot{
				ref:           test.ref,
				contentID:     "sha256:test",
				manifestBytes: payload,
			}

			t.Log("Use the deployment locator without interpreting manifest build.buildId")
			normalized, err := normalizeBuildSnapshot(snapshot)
			require.NoError(t, err)
			require.Equal(t, test.ref, normalized.build.Path)
		})
	}
}

func TestBuildFromGbuildManifestV2RejectsNonLPUArtifactForLPUOnly(t *testing.T) {
	t.Parallel()

	t.Log("Build an lpuOnly manifest with one LPU and one CUDA artifact")
	manifest := newManifestV2ContractFixture(t)
	deployment, err := manifest.Deployment()
	require.NoError(t, err)
	deployment.SetCompilationMode(manifestcapnpv2.CompilationMode_lpuOnly)
	artifacts, err := manifest.Artifacts()
	require.NoError(t, err)
	partitions, err := artifacts.NewPartitions(2)
	require.NoError(t, err)
	setManifestV2LPUArtifact(t, partitions.At(0), 0)
	cudaRef, err := partitions.At(1).NewPartition()
	require.NoError(t, err)
	cudaRef.SetPartitionId(1)
	cudaRef.SetDeviceType(manifestcapnpv2.DeviceType_cuda)

	t.Log("Reject the mixed-device lpuOnly artifact set")
	_, err = buildFromGbuildManifestV2("gs://bucket/registry/build-id", manifest)
	require.ErrorContains(t, err, "deployment.compilationMode lpuOnly requires every artifact partition to use deviceType lpu")
}
