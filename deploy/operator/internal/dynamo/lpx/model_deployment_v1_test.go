/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

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

// testUnitBundle describes a deployment.v1 contract with compiler-owned LPU units.
type testUnitBundle struct {
	architecture   deploymentcapnpv1.LpuArchitecture
	devicesPerNode uint32
	// hybrid compiles in lpx mode and adds one CUDA partition after the LPU partitions.
	hybrid  bool
	lpxMode bool
	// cpuEmbeddings sets supportsCpuEmbeddings and standaloneTokenEmbeddings.
	cpuEmbeddings bool
	partitions    []testUnitPartition
	chains        [][]uint32
	// partSelect lists the selected LPU partitions; nil writes a full build.
	partSelect []uint32
	// units is written as config.lpuUnits; nil leaves the field unset.
	units []testLPUUnit
}

type testUnitPartition struct {
	id       uint32
	numChips uint32
	// shape is the HX16x8x2 partitionShape; nil omits topologyMetadata.
	shape []uint32
	// topology names the compiled topology; empty uses registryTestTopology.
	topology string
}

type testLPUUnit struct {
	reservations []testLPUReservation
	connectors   []testLPUConnector
}

type testLPUReservation struct {
	ids   []uint32
	nodes uint32
	// hx is {traysPerHemisphere, hemispheres, racks}; nil sizes the reservation in nodes.
	hx []uint8
}

type testLPUConnector struct {
	// hx selects hxPropSyncV1 with device pairs; otherwise xt8888Gap.
	hx    bool
	pairs [][2]uint32
}

// testUnitPartitions returns LPU partitions that share one geometry.
func testUnitPartitions(numChips uint32, shape []uint32, ids ...uint32) []testUnitPartition {
	partitions := make([]testUnitPartition, len(ids))
	for index, id := range ids {
		partitions[index] = testUnitPartition{id: id, numChips: numChips, shape: shape}
	}
	return partitions
}

func testNodesReservation(nodes uint32, ids ...uint32) testLPUReservation {
	return testLPUReservation{ids: ids, nodes: nodes}
}

func testHXReservation(traysPerHemisphere, hemispheres, racks uint8, ids ...uint32) testLPUReservation {
	return testLPUReservation{ids: ids, hx: []uint8{traysPerHemisphere, hemispheres, racks}}
}

// testSingleUnits places each reservation in its own unit.
func testSingleUnits(reservations ...testLPUReservation) []testLPUUnit {
	units := make([]testLPUUnit, len(reservations))
	for index, reservation := range reservations {
		units[index] = testLPUUnit{reservations: []testLPUReservation{reservation}}
	}
	return units
}

// testHXConventionConnector joins a source reservation's last host, device by
// device, to the destination's first host, with lanes [4, 2, 1].
func testHXConventionConnector(sourceHosts, devicesPerNode uint32) testLPUConnector {
	connector := testLPUConnector{hx: true}
	for device := range devicesPerNode {
		connector.pairs = append(connector.pairs, [2]uint32{(sourceHosts-1)*devicesPerNode + device, device})
	}
	return connector
}

// encodeTestUnitBundle serializes the bundle as deployment.v1.capnp.bin bytes.
func encodeTestUnitBundle(t *testing.T, bundle testUnitBundle) []byte {
	t.Helper()
	message, segment := capnp.NewSingleSegmentMessage(nil)
	deployment, err := deploymentcapnpv1.NewRootDeployment(segment)
	require.NoError(t, err)
	deployment.SetContractRevision(deploymentcapnpv1.CurrentContractRevision)

	t.Log("Encode the program, runtime I/O and selected chains")
	config, err := deployment.NewConfig()
	require.NoError(t, err)
	config.SetCompilationMode(deploymentcapnpv1.CompilationMode_lpuOnly)
	if bundle.hybrid || bundle.lpxMode {
		config.SetCompilationMode(deploymentcapnpv1.CompilationMode_lpx)
	}
	program, err := config.NewProgram()
	require.NoError(t, err)
	program.SetSupportsCpuEmbeddings(bundle.cpuEmbeddings)
	program.SetStandaloneTokenEmbeddings(bundle.cpuEmbeddings)
	runtimeIO, err := config.NewRuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetIoFpgaCount(1)
	runtimeIO.SetFanoutFactor(1)
	if bundle.chains != nil {
		chains, err := config.NewSelectedPropSyncChains(int32(len(bundle.chains)))
		require.NoError(t, err)
		for chainIndex, chain := range bundle.chains {
			ids, err := chains.At(chainIndex).NewPartitionIds(int32(len(chain)))
			require.NoError(t, err)
			for index, id := range chain {
				ids.Set(index, id)
			}
		}
	}

	t.Log("Encode the compiler-owned units")
	if bundle.units != nil {
		units, err := config.NewLpuUnits(int32(len(bundle.units)))
		require.NoError(t, err)
		for unitIndex, unit := range bundle.units {
			reservations, err := units.At(unitIndex).NewReservations(int32(len(unit.reservations)))
			require.NoError(t, err)
			for index, reservation := range unit.reservations {
				ids, err := reservations.At(index).NewPartitionIds(int32(len(reservation.ids)))
				require.NoError(t, err)
				for idIndex, id := range reservation.ids {
					ids.Set(idIndex, id)
				}
				size := reservations.At(index).Size()
				if reservation.hx == nil {
					size.SetNodes(reservation.nodes)
					continue
				}
				extent, err := size.NewHx()
				require.NoError(t, err)
				extent.SetTraysPerHemisphere(reservation.hx[0])
				extent.SetHemispheres(reservation.hx[1])
				extent.SetRacks(reservation.hx[2])
			}
			connectors, err := units.At(unitIndex).NewConnectors(int32(len(unit.connectors)))
			require.NoError(t, err)
			for index, connector := range unit.connectors {
				if !connector.hx {
					connectors.At(index).SetXt8888Gap()
					continue
				}
				hx, err := connectors.At(index).NewHxPropSyncV1()
				require.NoError(t, err)
				pairs, err := hx.NewConnections(int32(len(connector.pairs)))
				require.NoError(t, err)
				for pairIndex, pair := range connector.pairs {
					pairs.At(pairIndex).SetFromDevice(pair[0])
					pairs.At(pairIndex).SetToDevice(pair[1])
				}
			}
		}
	}

	t.Log("Encode the partition selection, the LPU partitions and the hybrid CUDA partition")
	if bundle.partSelect != nil {
		selection, err := deployment.NewPartSelect()
		require.NoError(t, err)
		selected, err := selection.NewPartitions(int32(len(bundle.partSelect)))
		require.NoError(t, err)
		for index, id := range bundle.partSelect {
			selected.At(index).SetDeviceType(commoncapnpv1.DeviceType_lpu)
			selected.At(index).SetPartitionId(id)
		}
	}
	count := len(bundle.partitions)
	if bundle.hybrid {
		count++
	}
	partitions, err := deployment.NewPartitions(int32(count))
	require.NoError(t, err)
	lastID := uint32(0)
	for index, partition := range bundle.partitions {
		ref, err := partitions.At(index).NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(commoncapnpv1.DeviceType_lpu)
		ref.SetPartitionId(partition.id)
		lastID = max(lastID, partition.id)
		detail, err := partitions.At(index).Detail().NewLpu()
		require.NoError(t, err)
		require.NoError(t, detail.SetPath(fmt.Sprintf("part-%d", partition.id)))
		topology := partition.topology
		if topology == "" {
			topology = registryTestTopology
		}
		require.NoError(t, detail.SetTopology(topology))
		detail.SetArchitecture(bundle.architecture)
		detail.SetNumChips(partition.numChips)
		detail.SetDevicesPerNode(bundle.devicesPerNode)
		if partition.shape != nil {
			metadata, err := detail.NewTopologyMetadata()
			require.NoError(t, err)
			require.NoError(t, metadata.SetTopologyFamily(hxTopologyFamily))
			shape, err := metadata.NewPartitionShape(int32(len(partition.shape)))
			require.NoError(t, err)
			for dimension, extent := range partition.shape {
				shape.Set(dimension, extent)
			}
		}
	}
	if bundle.hybrid {
		ref, err := partitions.At(len(bundle.partitions)).NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(commoncapnpv1.DeviceType_cuda)
		ref.SetPartitionId(lastID + 1)
		_, err = partitions.At(len(bundle.partitions)).Detail().NewCuda()
		require.NoError(t, err)
	}

	data, err := message.Marshal()
	require.NoError(t, err)
	return data
}

// acquireTestUnitBundle publishes the bundle in a fresh build directory and
// acquires it through the model registry.
func acquireTestUnitBundle(t *testing.T, bundle testUnitBundle) (*Build, error) {
	t.Helper()
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, gbuildDeploymentV1CapnpFile), encodeTestUnitBundle(t, bundle), 0o600))
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)
	return registry.AcquireBuild(t.Context(), root)
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
			require.EqualValues(t, test.devicesPerNode, build.partitions[0].devicesPerNode)
			require.Equal(t, test.wantNodes, build.partitions[0].effectiveNodeCount())
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
			snapshot, err := registry.AcquireBuild(t.Context(), root)
			require.NoError(t, err)
			require.Equal(t, buildContractDeploymentV1, snapshot.contractFormat)
			normalized := snapshot
			require.NoError(t, err)
			require.Equal(t, buildContractDeploymentV1, normalized.contractFormat)
			require.EqualValues(t, 4, normalized.ioFPGACount)
			require.Equal(t, test.family, normalized.family)

			t.Log("Pass the selected deployment file to Cyborg before authored environment references")
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineHybrid, Models: []string{"default"}, Build: normalized,
			})
			require.NoError(t, err)
			require.Len(t, projections, 1)
			projection := projections[0]
			require.Equal(t, buildContractDeploymentV1, projection.contractFormat)
			require.Equal(t, test.family, projection.configuredBuild.family)
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
			second, err := registry.AcquireBuild(t.Context(), root)
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
			_, err = registry.AcquireBuild(t.Context(), root)
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
		files: map[string]*fakeModelFileStream{
			gbuildDeploymentV1CapnpFile: {chunks: modelFileChunks(gbuildDeploymentV1CapnpFile, string(payload))},
		},
	}
	registry, err := NewModelRegistry("gs://bucket/registry", client)
	require.NoError(t, err)
	snapshot, err := registry.AcquireBuild(t.Context(), "model/build")
	require.NoError(t, err)
	require.Equal(t, buildContractDeploymentV1, snapshot.contractFormat)
	require.Empty(t, client.listRequests)
	require.Len(t, client.filesRequests, 2)
	require.Equal(t, []string{gbuildManifestV2CapnpFile}, client.filesRequests[0].GetFileSelector().GetPaths())
	require.Equal(t, []string{gbuildDeploymentV1CapnpFile}, client.filesRequests[1].GetFileSelector().GetPaths())
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
	_, err = registry.AcquireBuild(t.Context(), root)
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
			require.Equal(t, wantFamily, build.family)
			require.Equal(t, wantIO, build.ioFPGACount)
			require.Equal(t, name == "CPU embeddings", build.supportsCPUEmbeddings)
			require.Equal(t, name == hybridCUDA, build.hasCUDAPartitions)
			require.Len(t, build.partitions, 1)
			if wantFamily == BuildFamilyHX {
				require.Equal(t, []int64{16, int64(wantNodes), 1, 1}, build.partitions[0].hxExtent)
			} else {
				require.Equal(t, wantNodes, build.partitions[0].effectiveNodeCount())
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
	_, err = registry.AcquireBuild(t.Context(), root)
	require.ErrorContains(t, err, "parsing deployment.v1.capnp.bin")
}

func TestDeploymentContractTopologyFamilies(t *testing.T) {
	t.Log("Accept canonical topology metadata and reject unsupported family names")
	for _, family := range []string{"HX16x8x2", "16x8x2x3", " HX16x8x2 ", " 16x8x2x3 ", "", "hx16x8x2", "HX16x8x2x3", "unknown"} {
		for _, geometry := range []struct {
			name    string
			chips   uint32
			shape   []uint32
			want    []int64
			wantErr string
		}{
			{"cross hemisphere", 32, []uint32{16, 1, 2, 1}, []int64{16, 1, 2, 1}, ""},
			{"subnode", 8, []uint32{8, 1, 1, 1}, []int64{16, 1, 1, 1}, ""},
			{"cross rack", 32, []uint32{16, 1, 1, 2}, nil, "unsupported HX extent"},
			{"count mismatch", 16, []uint32{8, 1, 1, 1}, nil, "contains 8 chips, want numChips 16"},
		} {
			t.Run(family+"/"+geometry.name, func(t *testing.T) {
				t.Log("Encode a split deployment contract with explicit LP30 geometry")
				deployment := newDeploymentV1ContractFixture(t)
				partitions, err := deployment.Partitions()
				require.NoError(t, err)
				detail, err := partitions.At(0).Detail().Lpu()
				require.NoError(t, err)
				detail.SetArchitecture(deploymentcapnpv1.LpuArchitecture_lp30)
				detail.SetDevicesPerNode(16)
				detail.SetNumChips(geometry.chips)
				metadata, err := detail.NewTopologyMetadata()
				require.NoError(t, err)
				require.NoError(t, metadata.SetTopologyFamily(family))
				shape, err := metadata.NewPartitionShape(int32(len(geometry.shape)))
				require.NoError(t, err)
				for i, value := range geometry.shape {
					shape.Set(i, value)
				}
				data, err := deployment.Message().Marshal()
				require.NoError(t, err)

				t.Log("Decode through the split reader and preserve family and shape rejection")
				decoded, err := decodeGbuildDeploymentV1(data)
				require.NoError(t, err)
				build, err := buildFromGbuildDeploymentV1("gs://models/build", decoded)
				switch family {
				case "HX16x8x2", " HX16x8x2 ":
					if geometry.wantErr != "" {
						require.ErrorContains(t, err, geometry.wantErr)
						return
					}
				default:
					require.ErrorContains(t, err, "topologyMetadata.topologyFamily")
					return
				}
				require.NoError(t, err)
				require.Equal(t, geometry.want, build.partitions[0].hxExtent)
				require.EqualValues(t, geometry.chips, build.partitions[0].numChips)
			})
		}
	}
}

func TestDeploymentUnitsContract(t *testing.T) {
	t.Parallel()
	xt := testUnitBundle{architecture: deploymentcapnpv1.LpuArchitecture_lp20, devicesPerNode: 8,
		partitions: testUnitPartitions(8, nil, 0, 1),
		units:      []testLPUUnit{{reservations: []testLPUReservation{testNodesReservation(1, 0), testNodesReservation(1, 1)}, connectors: []testLPUConnector{{}}}}}
	hx := testUnitBundle{architecture: deploymentcapnpv1.LpuArchitecture_lp30, devicesPerNode: 16,
		partitions: testUnitPartitions(16, nil, 0, 1),
		units:      []testLPUUnit{{reservations: []testLPUReservation{testHXReservation(1, 1, 1, 0), testHXReservation(1, 1, 1, 1)}, connectors: []testLPUConnector{testHXConventionConnector(1, 16)}}}}
	for _, tc := range []struct {
		name    string
		base    testUnitBundle
		change  func(*testUnitBundle)
		wantErr string
	}{
		{name: "XT units need no legacy chain", base: xt},
		{name: "HX units need no legacy topology", base: hx},
		{name: "legacy chain cannot override units", base: xt, change: func(b *testUnitBundle) { b.chains = [][]uint32{{99, 99}} }},
		{name: "missing partition", base: xt, change: func(b *testUnitBundle) { b.units = testSingleUnits(testNodesReservation(1, 0)) }, wantErr: "omits LPU partition 1"},
		{name: "missing zero is not a generic exception", base: xt, change: func(b *testUnitBundle) { b.units = testSingleUnits(testNodesReservation(1, 1)) }, wantErr: "omits LPU partition 0"},
		{name: "CPU standalone zero can be omitted", base: xt, change: func(b *testUnitBundle) { b.cpuEmbeddings = true; b.units = testSingleUnits(testNodesReservation(1, 1)) }},
		{name: "CPU standalone zero can be present", base: xt, change: func(b *testUnitBundle) { b.cpuEmbeddings = true }},
		{name: "CUDA build cannot omit zero", base: xt, change: func(b *testUnitBundle) {
			b.hybrid = true
			b.cpuEmbeddings = true
			b.units = testSingleUnits(testNodesReservation(1, 1))
		}, wantErr: "omits LPU partition 0"},
		{name: "duplicate coverage", base: xt, change: func(b *testUnitBundle) {
			b.units = testSingleUnits(testNodesReservation(1, 0), testNodesReservation(1, 0))
		}, wantErr: "repeats partition 0"},
		{name: "unknown partition", base: xt, change: func(b *testUnitBundle) { b.units = testSingleUnits(testNodesReservation(1, 99)) }, wantErr: "not a deployed LPU partition"},
		{name: "empty unit", base: xt, change: func(b *testUnitBundle) { b.units = []testLPUUnit{{}} }, wantErr: "reservations must not be empty"},
		{name: "empty partition list", base: xt, change: func(b *testUnitBundle) { b.units = testSingleUnits(testNodesReservation(1)) }, wantErr: "partitionIds must not be empty"},
		{name: "extra host", base: xt, change: func(b *testUnitBundle) { b.units = testSingleUnits(testNodesReservation(3, 0, 1)) }, wantErr: "reserves 3 hosts for 16 chips; want 2"},
		{name: "missing host", base: xt, change: func(b *testUnitBundle) { b.units = testSingleUnits(testNodesReservation(1, 0, 1)) }, wantErr: "reserves 1 hosts for 16 chips; want 2"},
		{name: "packed without CUDA", base: xt, change: func(b *testUnitBundle) { b.units = testSingleUnits(testNodesReservation(2, 0, 1)) }},
		{name: "packed with CUDA", base: xt, change: func(b *testUnitBundle) { b.hybrid = true; b.units = testSingleUnits(testNodesReservation(2, 0, 1)) }, wantErr: "requires a bundle without CUDA"},
		{name: "packed beside reservation", base: xt, change: func(b *testUnitBundle) {
			b.units = []testLPUUnit{{reservations: []testLPUReservation{testNodesReservation(2, 0, 1), testNodesReservation(1, 0)}, connectors: []testLPUConnector{{}}}}
		}, wantErr: "only reservation"},
		{name: "connector count", base: xt, change: func(b *testUnitBundle) {
			b.units = []testLPUUnit{{reservations: []testLPUReservation{testNodesReservation(1, 0), testNodesReservation(1, 1)}}}
		}, wantErr: "0 connectors for 2 reservations"},
		{name: "HX connector on XT", base: xt, change: func(b *testUnitBundle) { b.units = hx.units }, wantErr: "LP20 reservations use size.nodes"},
		{name: "HX nodes greater than one", base: hx, change: func(b *testUnitBundle) { b.units = testSingleUnits(testNodesReservation(2, 0, 1)) }},
		{name: "HX nodes form retains another width", base: hx, change: func(b *testUnitBundle) {
			b.devicesPerNode = 8
			b.partitions = testUnitPartitions(8, nil, 0)
			b.units = testSingleUnits(testNodesReservation(1, 0))
		}},
		{name: "HX nodes one allows partial host", base: hx, change: func(b *testUnitBundle) {
			b.partitions = testUnitPartitions(8, nil, 0)
			b.units = testSingleUnits(testNodesReservation(1, 0))
		}},
		{name: "HX device pair out of bounds", base: hx, change: func(b *testUnitBundle) {
			b.units = []testLPUUnit{{reservations: hx.units[0].reservations, connectors: []testLPUConnector{{hx: true, pairs: [][2]uint32{{16, 0}}}}}}
		}, wantErr: "outside the joined reservations"},
		{name: "HX repeated device pair", base: hx, change: func(b *testUnitBundle) {
			b.units = []testLPUUnit{{reservations: hx.units[0].reservations, connectors: []testLPUConnector{{hx: true, pairs: [][2]uint32{{0, 0}, {0, 0}}}}}}
		}, wantErr: "repeats device pair"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Decode a serialized compiler contract without derived legacy placement")
			bundle := tc.base
			if tc.change != nil {
				tc.change(&bundle)
			}
			build, err := acquireTestUnitBundle(t, bundle)
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
				return
			}
			require.NoError(t, err)
			require.Len(t, build.lpuUnits, len(bundle.units))
			for i, unit := range bundle.units {
				for j, reservation := range unit.reservations {
					want := make([]int, len(reservation.ids))
					for k, id := range reservation.ids {
						want[k] = int(id)
					}
					require.Equal(t, want, build.lpuUnits[i].Reservations[j].PartitionIDs)
				}
			}
		})
	}
}

func TestDeploymentUnitsPreserveEncodedOrder(t *testing.T) {
	t.Parallel()
	t.Log("Read units whose unit and partition order differs from numeric ID order")
	bundle := testUnitBundle{architecture: deploymentcapnpv1.LpuArchitecture_lp20, devicesPerNode: 8,
		partitions: testUnitPartitions(8, nil, 0, 1, 2), units: testSingleUnits(testNodesReservation(2, 2, 0), testNodesReservation(1, 1))}
	build, err := acquireTestUnitBundle(t, bundle)
	require.NoError(t, err)
	require.Equal(t, []int{2, 0}, build.lpuUnits[0].Reservations[0].PartitionIDs)
	require.Equal(t, []int{1}, build.lpuUnits[1].Reservations[0].PartitionIDs)
}

func TestUnitsPlacementIgnoresClearedLegacyFields(t *testing.T) {
	t.Log("Decode a units chain while all legacy placement fields remain populated")
	bundle := testUnitBundle{architecture: deploymentcapnpv1.LpuArchitecture_lp30, devicesPerNode: 16, lpxMode: true,
		chains: [][]uint32{{0, 1}}, partitions: testUnitPartitions(16, []uint32{16, 1, 1, 1}, 0, 1),
		units: []testLPUUnit{{reservations: []testLPUReservation{testHXReservation(1, 1, 1, 0), testHXReservation(1, 1, 1, 1)}, connectors: []testLPUConnector{testHXConventionConnector(1, 16)}}}}
	message, err := capnp.Unmarshal(encodeTestUnitBundle(t, bundle))
	require.NoError(t, err)
	original, err := deploymentcapnpv1.ReadRootDeployment(message)
	require.NoError(t, err)
	writable, _ := capnp.NewSingleSegmentMessage(nil)
	require.NoError(t, writable.SetRoot(original.ToPtr()))
	deployment, err := deploymentcapnpv1.ReadRootDeployment(writable)
	require.NoError(t, err)
	config, err := deployment.Config()
	require.NoError(t, err)
	config.SetNumLpuNodes(2)
	before, err := buildFromGbuildDeploymentV1("fixture", deployment)
	require.NoError(t, err)
	beforeProjection, err := projectTestUnits(before, nil)
	require.NoError(t, err)

	t.Log("Clear every legacy placement field without changing units or partition execution artifacts")
	config.SetNumLpuNodes(0)
	require.NoError(t, config.SetSelectedPropSyncChains(deploymentcapnpv1.PropSyncChain_List{}))
	partitions, err := deployment.Partitions()
	require.NoError(t, err)
	for i := 0; i < partitions.Len(); i++ {
		detail, err := partitions.At(i).Detail().Lpu()
		require.NoError(t, err)
		require.NoError(t, detail.SetTopologyMetadata(deploymentcapnpv1.TopologyMetadata{}))
	}
	after, err := buildFromGbuildDeploymentV1("fixture", deployment)
	require.NoError(t, err)
	afterProjection, err := projectTestUnits(after, nil)
	require.NoError(t, err)

	t.Log("Keep the complete scheduler request and Agent count unchanged")
	require.Equal(t, before.lpuUnits, after.lpuUnits)
	require.Equal(t, beforeProjection[0].AgentReplicas(), afterProjection[0].AgentReplicas())
	require.Equal(t, beforeProjection[0].RequestSpec(&MaterializationPlan{}, "agents"), afterProjection[0].RequestSpec(&MaterializationPlan{}, "agents"))
}
