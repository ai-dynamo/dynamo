/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"os"
	"path/filepath"
	"testing"

	modelpb "github.com/ai-dynamo/modelexpress/modelexpress_client/go/gen/modelexpress/model"

	"capnproto.org/go/capnp/v3"
	commoncapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/common/v1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func newDeploymentV1ContractFixture(t *testing.T) deploymentcapnpv1.Deployment {
	t.Helper()
	_, seg := capnp.NewSingleSegmentMessage(nil)
	deployment, err := deploymentcapnpv1.NewRootDeployment(seg)
	require.NoError(t, err)
	deployment.SetContractRevision(deploymentcapnpv1.CurrentContractRevision)
	identity, err := deployment.NewIdentity()
	require.NoError(t, err)
	require.NoError(t, identity.SetBuildId("build-id"))
	config, err := deployment.NewConfig()
	require.NoError(t, err)
	config.SetCompilationMode(deploymentcapnpv1.CompilationMode_lpx)
	config.SetNumLpuNodes(1)
	program, err := config.NewProgram()
	require.NoError(t, err)
	program.SetBatchSize(8)
	runtimeIO, err := config.NewRuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetProtocol(deploymentcapnpv1.RuntimeIoProtocol_fpgaRoce)
	runtimeIO.SetFpgaMode(deploymentcapnpv1.FpgaIoMode_dibDeb)
	runtimeIO.SetIoFpgaCount(4)
	runtimeIO.SetFanoutFactor(2)
	partitions, err := deployment.NewPartitions(1)
	require.NoError(t, err)
	partition, err := partitions.At(0).NewPartition()
	require.NoError(t, err)
	partition.SetDeviceType(commoncapnpv1.DeviceType_lpu)
	detail, err := partitions.At(0).Detail().NewLpu()
	require.NoError(t, err)
	require.NoError(t, detail.SetPath("part-0"))
	require.NoError(t, detail.SetTopology(registryTestTopology))
	detail.SetArchitecture(deploymentcapnpv1.LpuArchitecture_lp20)
	detail.SetNumChips(8)
	detail.SetDevicesPerNode(8)
	execution, err := deployment.NewExecution()
	require.NoError(t, err)
	require.NoError(t, execution.SetPath("execution.v1.capnp.bin"))
	digest := sha256.Sum256([]byte("opaque runtime bytes"))
	require.NoError(t, execution.SetSha256(digest[:]))
	return deployment
}

func TestDeploymentContractSelectsOneFormat(t *testing.T) {
	t.Parallel()
	for _, tt := range []struct {
		name    string
		files   []string
		want    buildContractFormat
		wantErr string
	}{
		{name: "legacy", files: []string{gbuildManifestV2CapnpFile}, want: buildContractManifestV2},
		{name: "split", files: []string{gbuildDeploymentV1CapnpFile, "execution.v1.capnp.bin"}, want: buildContractDeploymentV1},
		{name: "both", files: []string{gbuildDeploymentV1CapnpFile, gbuildManifestV2CapnpFile}, wantErr: "ambiguous"},
		{name: "JSON only", files: []string{gbuildManifestJSONFile}, wantErr: "missing"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Select the unique binary entry point without a preference or fallback")
			format, err := selectBuildContract(tt.files)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, format)
		})
	}
}

func TestDeploymentContractPreservesNormalizedBuild(t *testing.T) {
	t.Log("Create equivalent legacy and deployment-only scheduler contracts")
	legacy, err := buildFromGbuildManifestV2("gs://models/build", newManifestV2ContractFixture(t))
	require.NoError(t, err)
	deployment := newDeploymentV1ContractFixture(t)
	data, err := deployment.Message().Marshal()
	require.NoError(t, err)

	t.Log("Decode the deployment without importing execution or report types")
	decoded, err := decodeGbuildDeploymentV1(data)
	require.NoError(t, err)
	build, err := buildFromGbuildDeploymentV1("gs://models/build", decoded, []string{gbuildDeploymentV1CapnpFile, "execution.v1.capnp.bin"})
	require.NoError(t, err)
	require.Equal(t, legacy, build)
}

func TestDeploymentContractRejectsInvalidFacts(t *testing.T) {
	for _, name := range []string{"revision", "future revision", "build ID", "I/O protocol", "FPGA mode", "zero endpoints", "zero fanout", "host endpoints", "batch splitting", "execution path", "execution self-reference", "execution digest", "execution inventory", "unknown device", "mismatched detail", "embedding support", "missing embeddings", "missing architecture", "unknown architecture"} {
		t.Run(name, func(t *testing.T) {
			t.Log("Alter one typed deployment fact and retain the other valid fields")
			deployment := newDeploymentV1ContractFixture(t)
			inventory := []string{gbuildDeploymentV1CapnpFile, "execution.v1.capnp.bin"}
			config, err := deployment.Config()
			require.NoError(t, err)
			runtimeIO, err := config.RuntimeIo()
			require.NoError(t, err)
			execution, err := deployment.Execution()
			require.NoError(t, err)
			wantErr := "execution"
			switch name {
			case "missing architecture", "unknown architecture":
				partitions, err := deployment.Partitions()
				require.NoError(t, err)
				detail, err := partitions.At(0).Detail().Lpu()
				require.NoError(t, err)
				architecture := deploymentcapnpv1.LpuArchitecture_unspecified
				if name == "unknown architecture" {
					architecture = deploymentcapnpv1.LpuArchitecture(99)
				}
				detail.SetArchitecture(architecture)
				wantErr = "unsupported chip architecture"
			case "revision":
				deployment.SetContractRevision(0)
				wantErr = "contractRevision"
			case "future revision":
				deployment.SetContractRevision(99)
				wantErr = "contractRevision"
			case "build ID":
				identity, err := deployment.Identity()
				require.NoError(t, err)
				require.NoError(t, identity.SetBuildId(""))
				wantErr = "identity.buildId"
			case "zero endpoints":
				runtimeIO.SetIoFpgaCount(0)
				wantErr = "ioFpgaCount"
			case "zero fanout":
				runtimeIO.SetFanoutFactor(0)
				wantErr = "fanoutFactor"
			case "host endpoints":
				runtimeIO.SetProtocol(deploymentcapnpv1.RuntimeIoProtocol_host)
				wantErr = "host runtime I/O"
			case "execution self-reference":
				require.NoError(t, execution.SetPath(gbuildDeploymentV1CapnpFile))
			case "unknown device", "mismatched detail":
				partitions, err := deployment.Partitions()
				require.NoError(t, err)
				ref, err := partitions.At(0).Partition()
				require.NoError(t, err)
				if name == "unknown device" {
					ref.SetDeviceType(commoncapnpv1.DeviceType(99))
					wantErr = "deviceType"
				} else {
					ref.SetDeviceType(commoncapnpv1.DeviceType_cuda)
					wantErr = "mismatched detail"
				}
			case "embedding support":
				assets, err := deployment.NewRuntimeAssets()
				require.NoError(t, err)
				require.NoError(t, assets.SetTokenEmbeddingsPath("runtime/embeddings.npz"))
				wantErr = "requires supportsCpuEmbeddings"
			case "missing embeddings":
				program, err := config.Program()
				require.NoError(t, err)
				program.SetSupportsCpuEmbeddings(true)
				program.SetStandaloneTokenEmbeddings(true)
				wantErr = "tokenEmbeddingsPath is required"
			case "I/O protocol":
				runtimeIO.SetProtocol(deploymentcapnpv1.RuntimeIoProtocol(99))
				wantErr = "protocol"
			case "FPGA mode":
				runtimeIO.SetFpgaMode(deploymentcapnpv1.FpgaIoMode(99))
				wantErr = "fpgaMode"
			case "batch splitting":
				program, err := config.Program()
				require.NoError(t, err)
				program.SetBatchSize(7)
				wantErr = "divisible"
			case "execution path":
				require.NoError(t, execution.SetPath("../execution.v1.capnp.bin"))
			case "execution digest":
				require.NoError(t, execution.SetSha256([]byte("short")))
			case "execution inventory":
				inventory = []string{gbuildDeploymentV1CapnpFile}
			}

			t.Log("Reject the malformed scheduler input before producing a build")
			_, err = buildFromGbuildDeploymentV1("gs://models/build", deployment, inventory)
			require.ErrorContains(t, err, wantErr)
		})
	}
}

func TestDeploymentContractAcquisitionAndRuntimePath(t *testing.T) {
	for _, test := range []struct {
		name         string
		architecture deploymentcapnpv1.LpuArchitecture
		family       BuildFamily
		topology     string
		chips        uint32
	}{
		{"LP20", deploymentcapnpv1.LpuArchitecture_lp20, BuildFamilyXT, registryTestTopology, 8},
		{"LP30", deploymentcapnpv1.LpuArchitecture_lp30, BuildFamilyHX, hxTopologyFamily, 16},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Materialize deployment and opaque execution without a report or JSON")
			root := t.TempDir()
			deployment := newDeploymentV1ContractFixture(t)
			partitions, err := deployment.Partitions()
			require.NoError(t, err)
			detail, err := partitions.At(0).Detail().Lpu()
			require.NoError(t, err)
			detail.SetArchitecture(test.architecture)
			require.NoError(t, detail.SetTopology(test.topology))
			detail.SetNumChips(test.chips)
			detail.SetDevicesPerNode(test.chips)
			payload, err := deployment.Message().Marshal()
			require.NoError(t, err)
			require.NoError(t, os.WriteFile(filepath.Join(root, gbuildDeploymentV1CapnpFile), payload, 0o600))
			require.NoError(t, os.WriteFile(filepath.Join(root, "execution.v1.capnp.bin"), []byte("opaque runtime bytes"), 0o600))
			registry, err := NewModelRegistry("", nil)
			require.NoError(t, err)

			t.Log("Acquire and normalize exactly the scheduler contract")
			snapshot, err := registry.AcquireBuildSnapshot(t.Context(), root)
			require.NoError(t, err)
			require.Equal(t, payload, snapshot.manifestBytes)
			require.Equal(t, buildContractDeploymentV1, snapshot.format)
			normalized, err := normalizeBuildSnapshot(snapshot)
			require.NoError(t, err)
			require.Equal(t, buildContractDeploymentV1, normalized.format)
			require.EqualValues(t, 4, normalized.build.IOFPGACount)
			require.Equal(t, test.family, normalized.build.Family)

			t.Log("Pass the selected deployment file to Cyborg before authored environment references")
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineLPX, Models: []string{"default"}, BuildSnapshot: normalized,
			})
			require.NoError(t, err)
			require.Len(t, projections, 1)
			projection := projections[0]
			require.Equal(t, buildContractDeploymentV1, projection.contractFormat)
			require.Equal(t, test.family, projection.configuredBuild.Family)
			container := corev1.Container{Env: []corev1.EnvVar{{Name: "MODEL_PATH", Value: "$(GBUILD_MANIFEST_PATH)"}}}
			require.NoError(t, applyCyborgManifestPath(&container, projection, "/models"))
			require.Equal(t, []corev1.EnvVar{
				{Name: gbuildManifestPathEnv, Value: filepath.Join(root, gbuildDeploymentV1CapnpFile)},
				{Name: "MODEL_PATH", Value: "$(GBUILD_MANIFEST_PATH)"},
			}, container.Env)

			t.Log("Bind exact deployment bytes to snapshot identity")
			identity, err := deployment.Identity()
			require.NoError(t, err)
			require.NoError(t, identity.SetBuildId("different-build"))
			changed, err := deployment.Message().Marshal()
			require.NoError(t, err)
			require.NoError(t, os.WriteFile(filepath.Join(root, gbuildDeploymentV1CapnpFile), changed, 0o600))
			second, err := registry.AcquireBuildSnapshot(t.Context(), root)
			require.NoError(t, err)
			require.NotEqual(t, snapshot.contentID, second.contentID)
		})
	}
}

func TestDeploymentContractCorruptionNeverUsesLegacy(t *testing.T) {
	t.Log("Publish a corrupt split contract beside an otherwise valid legacy contract")
	root := t.TempDir()
	writeManifestV2Payload(t, root, manifestV2Payload(t))
	require.NoError(t, os.WriteFile(filepath.Join(root, gbuildDeploymentV1CapnpFile), []byte("corrupt"), 0o600))
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)

	t.Log("Reject ambiguous publication before any legacy decoder can accept it")
	_, err = registry.AcquireBuildSnapshot(t.Context(), root)
	require.ErrorIs(t, err, ErrBuildSnapshotInconsistent)
	require.ErrorContains(t, err, "ambiguous")
}

func TestDeploymentContractFencesRemoteBytes(t *testing.T) {
	t.Log("Serve differing deployment reads with a stable inventory")
	client := &fakeModelServiceClient{
		list: &modelpb.ModelFileList{Files: []*modelpb.ModelFileInfo{
			{RelativePath: gbuildDeploymentV1CapnpFile}, {RelativePath: "execution.v1.capnp.bin"},
		}},
		fileStreams: []*fakeModelFileStream{
			{chunks: modelFileChunks(gbuildDeploymentV1CapnpFile, "first")},
			{chunks: modelFileChunks(gbuildDeploymentV1CapnpFile, "second")},
		},
	}
	registry, err := NewModelRegistry("gs://bucket/registry", client)
	require.NoError(t, err)

	t.Log("Reject changes between the two bounded reads")
	_, err = registry.AcquireBuildSnapshot(t.Context(), "model/build")
	require.ErrorIs(t, err, ErrBuildSnapshotInconsistent)
	require.ErrorContains(t, err, "compiler metadata deployment.v1.capnp.bin changed while acquiring")
	require.Len(t, client.listRequests, 2)
	require.Len(t, client.filesRequests, 2)
	for _, request := range client.filesRequests {
		require.Equal(t, []string{gbuildDeploymentV1CapnpFile}, request.GetFileSelector().GetPaths())
	}
}

func TestDeploymentContractBoundsAcquisition(t *testing.T) {
	t.Log("Create a sparse deployment file beyond the existing metadata budget")
	root := t.TempDir()
	file, err := os.Create(filepath.Join(root, gbuildDeploymentV1CapnpFile))
	require.NoError(t, err)
	require.NoError(t, file.Truncate(int64(maxBuildSnapshotMetadataBytes)+1))
	require.NoError(t, file.Close())
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)

	t.Log("Reject oversized metadata before a decoder or runtime sees it")
	_, err = registry.AcquireBuildSnapshot(t.Context(), root)
	require.ErrorIs(t, err, ErrBuildSnapshotInconsistent)
	require.ErrorIs(t, err, errBuildFileTooLarge)
}

func TestDeploymentContractSupportedGeometry(t *testing.T) {
	for _, name := range []string{"host XT", "HX metadata", "HX legacy geometry", "HX subnode", "XT sixteen-chip node", "partial XT", "hybrid CPU", "CPU embedding asset"} {
		t.Run(name, func(t *testing.T) {
			t.Log("Configure one supported geometry on a deployment-only fixture")
			deployment := newDeploymentV1ContractFixture(t)
			config, err := deployment.Config()
			require.NoError(t, err)
			partitions, err := deployment.Partitions()
			require.NoError(t, err)
			detail, err := partitions.At(0).Detail().Lpu()
			require.NoError(t, err)
			wantFamily := BuildFamilyXT
			wantNodes := 1
			wantIO := int32(4)
			switch name {
			case "host XT":
				runtimeIO, err := config.RuntimeIo()
				require.NoError(t, err)
				runtimeIO.SetProtocol(deploymentcapnpv1.RuntimeIoProtocol_host)
				runtimeIO.SetFpgaMode(deploymentcapnpv1.FpgaIoMode_simple)
				runtimeIO.SetIoFpgaCount(1)
				wantIO = 1
			case "HX metadata", "HX legacy geometry", "HX subnode":
				detail.SetArchitecture(deploymentcapnpv1.LpuArchitecture_lp30)
				wantFamily = BuildFamilyHX
				require.NoError(t, detail.SetTopology(hxTopologyFamily))
				detail.SetNumChips(16)
				detail.SetDevicesPerNode(16)
				if name == "HX subnode" {
					detail.SetNumChips(8)
				}
				if name == "HX metadata" {
					wantNodes = 2
					config.SetNumLpuNodes(2)
					detail.SetNumChips(32)
					metadata, err := detail.NewTopologyMetadata()
					require.NoError(t, err)
					require.NoError(t, metadata.SetTopologyFamily(hxTopologyFamily))
					shape, err := metadata.NewPartitionShape(4)
					require.NoError(t, err)
					for index, value := range []uint32{16, 2, 1, 1} {
						shape.Set(index, value)
					}
				}
			case "XT sixteen-chip node":
				detail.SetNumChips(16)
				detail.SetDevicesPerNode(16)
			case "partial XT":
				config.SetNumLpuNodes(4)
				ref, err := partitions.At(0).Partition()
				require.NoError(t, err)
				ref.SetPartitionId(3)
				selection, err := deployment.NewPartSelect()
				require.NoError(t, err)
				selected, err := selection.NewPartitions(1)
				require.NoError(t, err)
				selected.At(0).SetDeviceType(commoncapnpv1.DeviceType_lpu)
				selected.At(0).SetPartitionId(3)
			case "hybrid CPU":
				expanded, err := deployment.NewPartitions(2)
				require.NoError(t, err)
				require.NoError(t, expanded.Set(0, partitions.At(0)))
				ref, err := expanded.At(1).NewPartition()
				require.NoError(t, err)
				ref.SetDeviceType(commoncapnpv1.DeviceType_cpu)
				_, err = expanded.At(1).Detail().NewCpu()
				require.NoError(t, err)
			case "CPU embedding asset":
				program, err := config.Program()
				require.NoError(t, err)
				program.SetSupportsCpuEmbeddings(true)
				assets, err := deployment.NewRuntimeAssets()
				require.NoError(t, err)
				require.NoError(t, assets.SetTokenEmbeddingsPath("runtime/embeddings.npz"))
			}

			t.Log("Preserve scheduler geometry while excluding non-LPU execution details")
			build, err := buildFromGbuildDeploymentV1("gs://models/build", deployment, []string{gbuildDeploymentV1CapnpFile, "execution.v1.capnp.bin"})
			require.NoError(t, err)
			require.Equal(t, wantFamily, build.Family)
			require.Equal(t, wantIO, build.IOFPGACount)
			require.Len(t, build.Partitions, 1)
			if wantFamily == BuildFamilyHX {
				require.Equal(t, []int64{16, int64(wantNodes), 1, 1}, build.Partitions[0].HXExtent)
			} else {
				require.Equal(t, wantNodes, build.Partitions[0].effectiveNodeCount())
			}
		})
	}
}

func TestDeploymentContractRejectsCorruptNewOnly(t *testing.T) {
	t.Log("Acquire stable corrupt deployment bytes with no legacy entry point")
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, gbuildDeploymentV1CapnpFile), []byte("corrupt"), 0o600))
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)
	snapshot, err := registry.AcquireBuildSnapshot(t.Context(), root)
	require.NoError(t, err)

	t.Log("Reject the selected deployment during normalization")
	_, err = normalizeBuildSnapshot(snapshot)
	require.ErrorContains(t, err, "parsing deployment.v1.capnp.bin")
}
