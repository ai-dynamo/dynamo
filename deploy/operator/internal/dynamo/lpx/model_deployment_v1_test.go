/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
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
	config, err := deployment.NewConfig()
	require.NoError(t, err)
	config.SetCompilationMode(deploymentcapnpv1.CompilationMode_lpx)
	_, err = config.NewProgram()
	require.NoError(t, err)
	runtimeIO, err := config.NewRuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetProtocol(deploymentcapnpv1.RuntimeIoProtocol_fpgaRoce)
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
	for _, test := range []struct {
		name           string
		devicesPerNode uint32
		wantNodes      int
	}{
		{name: "one node", devicesPerNode: 8, wantNodes: 1},
		{name: "manifest device density", devicesPerNode: 4, wantNodes: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Create equivalent legacy and deployment-only contracts with the manifest device density")
			manifest := newManifestV2ContractFixture(t)
			artifacts, err := manifest.Artifacts()
			require.NoError(t, err)
			legacyPartitions, err := artifacts.Partitions()
			require.NoError(t, err)
			legacyDetail, err := legacyPartitions.At(0).Detail().Lpu()
			require.NoError(t, err)
			legacyDetail.SetDevicesPerNode(test.devicesPerNode)
			legacy, err := buildFromGbuildManifestV2("gs://models/build", manifest)
			require.NoError(t, err)
			deployment := newDeploymentV1ContractFixture(t)
			partitions, err := deployment.Partitions()
			require.NoError(t, err)
			detail, err := partitions.At(0).Detail().Lpu()
			require.NoError(t, err)
			detail.SetDevicesPerNode(test.devicesPerNode)
			data, err := deployment.Message().Marshal()
			require.NoError(t, err)

			t.Log("Derive the same node count from partition geometry in both formats")
			decoded, err := decodeGbuildDeploymentV1(data)
			require.NoError(t, err)
			build, err := buildFromGbuildDeploymentV1("gs://models/build", decoded)
			require.NoError(t, err)
			require.Equal(t, legacy, build)
			require.EqualValues(t, test.devicesPerNode, build.Partitions[0].DevicesPerNode)
			require.Equal(t, test.wantNodes, build.Partitions[0].effectiveNodeCount())
		})
	}
}

func TestDeploymentContractIgnoresRuntimeProtocol(t *testing.T) {
	t.Log("Normalize a deployment with FPGA runtime I/O")
	deployment := newDeploymentV1ContractFixture(t)
	want, err := buildFromGbuildDeploymentV1("gs://models/build", deployment)
	require.NoError(t, err)
	config, err := deployment.Config()
	require.NoError(t, err)
	runtimeIO, err := config.RuntimeIo()
	require.NoError(t, err)

	for _, test := range []struct {
		name     string
		protocol deploymentcapnpv1.RuntimeIoProtocol
	}{
		{"host", deploymentcapnpv1.RuntimeIoProtocol_host},
		{"unknown", deploymentcapnpv1.RuntimeIoProtocol(99)},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Preserve the normalized build when only the runtime protocol changes")
			runtimeIO.SetProtocol(test.protocol)
			got, err := buildFromGbuildDeploymentV1("gs://models/build", deployment)
			require.NoError(t, err)
			require.Equal(t, want, got)
		})
	}
}

func TestDeploymentContractRejectsInvalidFacts(t *testing.T) {
	for _, name := range []string{"revision", "future revision", "zero endpoints", "zero fanout", "unknown device", "missing architecture", "unknown architecture"} {
		t.Run(name, func(t *testing.T) {
			t.Log("Alter one typed deployment fact and retain the other valid fields")
			deployment := newDeploymentV1ContractFixture(t)
			config, err := deployment.Config()
			require.NoError(t, err)
			runtimeIO, err := config.RuntimeIo()
			require.NoError(t, err)
			var wantErr string
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
			case "zero endpoints":
				runtimeIO.SetIoFpgaCount(0)
				wantErr = "ioFpgaCount"
			case "zero fanout":
				runtimeIO.SetFanoutFactor(0)
				wantErr = "fanoutFactor"
			case "unknown device":
				partitions, err := deployment.Partitions()
				require.NoError(t, err)
				ref, err := partitions.At(0).Partition()
				require.NoError(t, err)
				ref.SetDeviceType(commoncapnpv1.DeviceType(99))
				wantErr = "deviceType"
			}

			t.Log("Reject the malformed scheduler input before producing a build")
			_, err = buildFromGbuildDeploymentV1("gs://models/build", deployment)
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
			require.Equal(t, buildContractDeploymentV1, snapshot.format)
			normalized := snapshot
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
			identity, err := deployment.NewIdentity()
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
	for _, entry := range []string{"corrupt file", "directory"} {
		t.Run(entry, func(t *testing.T) {
			t.Log("Publish an invalid split entry beside an otherwise valid legacy contract")
			root := t.TempDir()
			writeManifestV2Payload(t, root, manifestV2Payload(t))
			path := filepath.Join(root, gbuildDeploymentV1CapnpFile)
			switch entry {
			case "corrupt file":
				require.NoError(t, os.WriteFile(path, []byte("corrupt"), 0o600))
			case "directory":
				require.NoError(t, os.Mkdir(path, 0o700))
			}
			registry, err := NewModelRegistry("", nil)
			require.NoError(t, err)

			t.Log("Reject ambiguous publication before any legacy decoder can accept it")
			_, err = registry.AcquireBuildSnapshot(t.Context(), root)
			require.ErrorIs(t, err, errInvalidBuildManifest)
			require.ErrorContains(t, err, "ambiguous")
		})
	}
}

func TestDeploymentContractReadsRemoteBytesOnce(t *testing.T) {
	t.Log("Serve one deployment contract with a separate opaque execution artifact")
	payload, err := newDeploymentV1ContractFixture(t).Message().Marshal()
	require.NoError(t, err)
	client := &fakeModelServiceClient{
		list:        &modelpb.ModelFileList{Files: []*modelpb.ModelFileInfo{{RelativePath: gbuildDeploymentV1CapnpFile}}},
		fileStreams: []*fakeModelFileStream{{chunks: modelFileChunks(gbuildDeploymentV1CapnpFile, string(payload))}},
	}
	registry, err := NewModelRegistry("gs://bucket/registry", client)
	require.NoError(t, err)
	snapshot, err := registry.AcquireBuildSnapshot(t.Context(), "model/build")
	require.NoError(t, err)
	require.Equal(t, buildContractDeploymentV1, snapshot.format)
	require.Len(t, client.listRequests, 2)
	require.Equal(t, []string{gbuildManifestV2CapnpFile}, client.listRequests[0].GetFileSelector().GetPaths())
	require.Equal(t, []string{gbuildDeploymentV1CapnpFile}, client.listRequests[1].GetFileSelector().GetPaths())
	require.Len(t, client.filesRequests, 1)
	require.Equal(t, []string{gbuildDeploymentV1CapnpFile}, client.filesRequests[0].GetFileSelector().GetPaths())
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
	require.ErrorIs(t, err, errBuildFileTooLarge)
}

func TestDeploymentContractSupportedGeometry(t *testing.T) {
	const hybridCUDA = "hybrid CUDA"
	for _, name := range []string{"host XT", "HX metadata", "HX legacy geometry", "HX subnode", "XT sixteen-chip node", "partial XT", "hybrid CPU", hybridCUDA, "CPU embeddings"} {
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
				ref, err := partitions.At(0).Partition()
				require.NoError(t, err)
				ref.SetPartitionId(3)
				selection, err := deployment.NewPartSelect()
				require.NoError(t, err)
				selected, err := selection.NewPartitions(1)
				require.NoError(t, err)
				selected.At(0).SetDeviceType(commoncapnpv1.DeviceType_lpu)
				selected.At(0).SetPartitionId(3)
			case "hybrid CPU", hybridCUDA:
				expanded, err := deployment.NewPartitions(2)
				require.NoError(t, err)
				require.NoError(t, expanded.Set(0, partitions.At(0)))
				ref, err := expanded.At(1).NewPartition()
				require.NoError(t, err)
				if name == hybridCUDA {
					ref.SetDeviceType(commoncapnpv1.DeviceType_cuda)
					_, err = expanded.At(1).Detail().NewCuda()
				} else {
					ref.SetDeviceType(commoncapnpv1.DeviceType_cpu)
					_, err = expanded.At(1).Detail().NewCpu()
				}
				require.NoError(t, err)
			case "CPU embeddings":
				program, err := config.Program()
				require.NoError(t, err)
				program.SetSupportsCpuEmbeddings(true)
			}

			t.Log("Preserve scheduler geometry while excluding non-LPU execution details")
			build, err := buildFromGbuildDeploymentV1("gs://models/build", deployment)
			require.NoError(t, err)
			require.Equal(t, wantFamily, build.Family)
			require.Equal(t, wantIO, build.IOFPGACount)
			require.Equal(t, name == "CPU embeddings", build.SupportsCPUEmbeddings)
			require.Equal(t, name == hybridCUDA, build.HasCUDAPartitions)
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
	_, err = registry.AcquireBuildSnapshot(t.Context(), root)
	require.ErrorContains(t, err, "parsing deployment.v1.capnp.bin")
}
