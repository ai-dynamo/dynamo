/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
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

	t.Log("Reject an unsupported contract revision before lowering the manifest")
	manifest.SetContractRevision(manifest.ContractRevision() + 1)
	_, err = buildFromGbuildManifestV2("gs://models/build", manifest)
	require.ErrorContains(t, err, "contractRevision")
}

func TestManifestReadersIgnoreExecutionMetadata(t *testing.T) {
	for _, field := range []string{"batch size", "node count", "compilation mode", "protocol", "FPGA mode", "runtime assets", "partition selection"} {
		t.Run(field, func(t *testing.T) {
			t.Log("Normalize equivalent legacy and split deployment contracts")
			legacy := newManifestV2ContractFixture(t)
			split := newDeploymentV1ContractFixture(t)
			want, err := buildFromGbuildManifestV2("gs://models/build", legacy)
			require.NoError(t, err)
			legacyConfig, err := legacy.Deployment()
			require.NoError(t, err)
			splitConfig, err := split.Config()
			require.NoError(t, err)
			legacyArtifacts, err := legacy.Artifacts()
			require.NoError(t, err)

			t.Log("Change metadata that does not determine Dynamo deployment output")
			switch field {
			case "batch size":
				legacyProgram, err := legacyConfig.Program()
				require.NoError(t, err)
				legacyProgram.SetBatchSize(7)
				splitProgram, err := splitConfig.Program()
				require.NoError(t, err)
				splitProgram.SetBatchSize(7)
			case "node count":
				legacyConfig.SetNumLpuNodes(99)
				splitConfig.SetNumLpuNodes(99)
			case "compilation mode":
				legacyConfig.SetCompilationMode(manifestcapnpv2.CompilationMode(99))
				splitConfig.SetCompilationMode(deploymentcapnpv1.CompilationMode(99))
			case "protocol":
				legacyIO, err := legacyConfig.RuntimeIo()
				require.NoError(t, err)
				legacyIO.SetProtocol(99)
				splitIO, err := splitConfig.RuntimeIo()
				require.NoError(t, err)
				splitIO.SetProtocol(deploymentcapnpv1.RuntimeIoProtocol(99))
			case "FPGA mode":
				legacyIO, err := legacyConfig.RuntimeIo()
				require.NoError(t, err)
				legacyIO.SetReserved1(99)
				splitIO, err := splitConfig.RuntimeIo()
				require.NoError(t, err)
				splitIO.SetFpgaMode(deploymentcapnpv1.FpgaIoMode(99))
			case "runtime assets":
				legacyAssets, err := legacyArtifacts.NewRuntimeAssets()
				require.NoError(t, err)
				require.NoError(t, legacyAssets.SetTokenEmbeddingsPath("../runtime-only"))
				splitAssets, err := split.NewRuntimeAssets()
				require.NoError(t, err)
				require.NoError(t, splitAssets.SetTokenEmbeddingsPath("../runtime-only"))
			case "partition selection":
				_, err := legacyArtifacts.NewPartSelect()
				require.NoError(t, err)
				_, err = split.NewPartSelect()
				require.NoError(t, err)
			}

			t.Log("Preserve the normalized deployment from both readers")
			gotLegacy, err := buildFromGbuildManifestV2("gs://models/build", legacy)
			require.NoError(t, err)
			require.Equal(t, want, gotLegacy)
			gotSplit, err := buildFromGbuildDeploymentV1("gs://models/build", split)
			require.NoError(t, err)
			require.Equal(t, want, gotSplit)
		})
	}
}

func TestAcquireBuildSnapshotRejectsInvalidHXArtifacts(t *testing.T) {
	t.Log("Define malformed HX artifact inventories")
	tests := []struct {
		name    string
		wantErr string
	}{
		{name: "duplicate LPU partition ID", wantErr: "repeats LPU partition ID 1"},
		{name: "partial build", wantErr: "partSelect builds are not supported"},
		{name: "empty selected prop-sync chain", wantErr: "deployment.selectedPropSyncChains[0] must contain at least two partitionIds"},
		{name: "singleton selected prop-sync chain", wantErr: "deployment.selectedPropSyncChains[0] must contain at least two partitionIds"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Acquire stable manifest bytes with the selected malformed HX inventory")
			fixture := newV3CompilerFixture()
			switch test.name {
			case "duplicate LPU partition ID":
				fixture.partitions = append(fixture.partitions, fixture.partitions[0])
			case "partial build":
				fixture.partSelect = true
			case "empty selected prop-sync chain":
				fixture.selectedPropSyncChains = [][]uint32{{}}
			case "singleton selected prop-sync chain":
				fixture.selectedPropSyncChains = [][]uint32{{1}}
			}
			buildDir := writeCompilerFixture(t, fixture)

			t.Log("Reject malformed HX artifacts before publishing the acquired snapshot")
			snapshot, err := (&defaultModelRegistry{}).AcquireBuildSnapshot(t.Context(), buildDir)
			require.ErrorIs(t, err, errInvalidBuildManifest)
			require.ErrorContains(t, err, test.wantErr)
			require.Nil(t, snapshot)
		})
	}
}

func TestBuildFromGbuildManifestV2ValidatesPartitionGeometry(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name, path, wantErr      string
		numChips, devicesPerNode uint32
	}{
		{"partition traversal", "../outside", "LPU partition 0 path", 8, 8},
		{"missing chips", "part-0", "numChips must be >= 1", 0, 8},
		{"missing node width", "part-0", "devicesPerNode must be >= 1", 8, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Set an invalid partition field used by Dynamo")
			manifest := newManifestV2ContractFixture(t)
			artifacts, err := manifest.Artifacts()
			require.NoError(t, err)
			partitions, err := artifacts.Partitions()
			require.NoError(t, err)
			detail, err := partitions.At(0).Detail().Lpu()
			require.NoError(t, err)
			require.NoError(t, detail.SetPath(test.path))
			detail.SetNumChips(test.numChips)
			detail.SetDevicesPerNode(test.devicesPerNode)

			t.Log("Reject the invalid scheduler or runtime configuration input")
			_, err = buildFromGbuildManifestV2("gs://models/build", manifest)
			require.ErrorContains(t, err, test.wantErr)
		})
	}
}

func TestBuildFromGbuildManifestV2DetectsCUDAPartitions(t *testing.T) {
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

	t.Log("Retain CUDA requirements independently of the compilation mode label")
	build, err := buildFromGbuildManifestV2("gs://bucket/registry/build-id", manifest)
	require.NoError(t, err)
	require.True(t, build.HasCUDAPartitions)
	require.Len(t, build.Partitions, 1)
}
