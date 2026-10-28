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
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
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
	setTestUnits(t, deployment, testSingleUnits(testNodesReservation(1, 0)))
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
	hybrid bool
	// lpx compiles in lpx mode without adding a CUDA partition.
	lpx bool
	// pipelineName is the program name; empty leaves it unset.
	pipelineName string
	// extraPartitions are CPU or CUDA partitions written after the LPU partitions.
	extraPartitions []testNonLPUPartition
	// cpuEmbeddings sets supportsCpuEmbeddings and standaloneTokenEmbeddings.
	cpuEmbeddings bool
	partitions    []testUnitPartition
	// partSelect lists the selected LPU partitions; nil writes a full build.
	partSelect []uint32
	// units is written as config.lpuUnits; nil leaves the field unset.
	units []testLPUUnit
}

type testUnitPartition struct {
	id       uint32
	numChips uint32
	// topology names the compiled topology; empty uses registryTestTopology.
	topology string
}

type testNonLPUPartition struct {
	id         uint32
	deviceType commoncapnpv1.DeviceType
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
	// hx selects hxPropSyncV1 with pairs, lanes and dimension; otherwise xt8888Gap with offset.
	hx     bool
	offset uint8
	pairs  [][2]uint32
	lanes  []uint8
	// dimension is the hxPropSyncV1 topologyDimension; zero leaves it unset.
	dimension uint8
}

// testUnitPartitions returns LPU partitions that share one geometry.
func testUnitPartitions(numChips uint32, ids ...uint32) []testUnitPartition {
	partitions := make([]testUnitPartition, len(ids))
	for index, id := range ids {
		partitions[index] = testUnitPartition{id: id, numChips: numChips}
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

// testHXConventionConnector joins two one-host reservations device by device,
// with lanes [4, 2, 1].
func testHXConventionConnector(devicesPerNode uint32) testLPUConnector {
	connector := testLPUConnector{hx: true, lanes: []uint8{4, 2, 1}, dimension: hxDefaultPropSyncTopologyDimension}
	for device := range devicesPerNode {
		connector.pairs = append(connector.pairs, [2]uint32{device, device})
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

	t.Log("Encode the program and runtime I/O")
	config, err := deployment.NewConfig()
	require.NoError(t, err)
	config.SetCompilationMode(deploymentcapnpv1.CompilationMode_lpuOnly)
	if bundle.hybrid || bundle.lpx {
		config.SetCompilationMode(deploymentcapnpv1.CompilationMode_lpx)
	}
	program, err := config.NewProgram()
	require.NoError(t, err)
	program.SetSupportsCpuEmbeddings(bundle.cpuEmbeddings)
	program.SetStandaloneTokenEmbeddings(bundle.cpuEmbeddings)
	if bundle.pipelineName != "" {
		require.NoError(t, program.SetProgramName(bundle.pipelineName))
	}
	runtimeIO, err := config.NewRuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetIoFpgaCount(1)
	runtimeIO.SetFanoutFactor(1)
	if bundle.units != nil {
		setTestUnits(t, deployment, bundle.units)
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
	count := len(bundle.partitions) + len(bundle.extraPartitions)
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
	}
	for index, extra := range bundle.extraPartitions {
		partition := partitions.At(len(bundle.partitions) + index)
		ref, err := partition.NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(extra.deviceType)
		ref.SetPartitionId(extra.id)
		lastID = max(lastID, extra.id)
		switch extra.deviceType {
		case commoncapnpv1.DeviceType_cuda:
			_, err = partition.Detail().NewCuda()
		case commoncapnpv1.DeviceType_cpu:
			_, err = partition.Detail().NewCpu()
		default:
			t.Fatalf("unsupported extra test device type %d", extra.deviceType)
		}
		require.NoError(t, err)
	}
	if bundle.hybrid {
		partition := partitions.At(count - 1)
		ref, err := partition.NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(commoncapnpv1.DeviceType_cuda)
		ref.SetPartitionId(lastID + 1)
		_, err = partition.Detail().NewCuda()
		require.NoError(t, err)
	}

	data, err := message.Marshal()
	require.NoError(t, err)
	return data
}

// setTestUnits replaces the deployment's config.lpuUnits.
func setTestUnits(t *testing.T, deployment deploymentcapnpv1.Deployment, testUnits []testLPUUnit) {
	t.Helper()
	config, err := deployment.Config()
	require.NoError(t, err)
	units, err := config.NewLpuUnits(int32(len(testUnits)))
	require.NoError(t, err)
	for unitIndex, unit := range testUnits {
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
				gap, err := connectors.At(index).NewXt8888Gap()
				require.NoError(t, err)
				gap.SetMaxInterPartitionOffset(connector.offset)
				continue
			}
			hx, err := connectors.At(index).NewHxPropSyncV1()
			require.NoError(t, err)
			hx.SetTopologyDimension(connector.dimension)
			pairs, err := hx.NewConnections(int32(len(connector.pairs)))
			require.NoError(t, err)
			for pairIndex, pair := range connector.pairs {
				pairs.At(pairIndex).SetFromDevice(pair[0])
				pairs.At(pairIndex).SetToDevice(pair[1])
			}
			lanes, err := hx.NewAcceptableLaneMultiplicities(int32(len(connector.lanes)))
			require.NoError(t, err)
			for laneIndex, lane := range connector.lanes {
				lanes.Set(laneIndex, lane)
			}
		}
	}
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
	for _, name := range []string{"revision", "future revision", "zero endpoints", "zero fanout", "unknown device", "missing architecture", "unknown architecture", "zero chips", "no units"} {
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
			case "zero chips":
				partitions, err := deployment.Partitions()
				require.NoError(t, err)
				detail, err := partitions.At(0).Detail().Lpu()
				require.NoError(t, err)
				detail.SetNumChips(0)
				wantErr = "numChips must be >= 1"
			case "no units":
				setTestUnits(t, deployment, []testLPUUnit{})
				wantErr = "records no config.lpuUnits"
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
		{"LP30", deploymentcapnpv1.LpuArchitecture_lp30, BuildFamilyHX, "HX16x8x2", 16},
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
			if test.family == BuildFamilyHX {
				setTestUnits(t, deployment, testSingleUnits(testHXReservation(1, 1, 1, 0)))
			}
			payload, err := deployment.Message().Marshal()
			require.NoError(t, err)
			require.NoError(t, os.WriteFile(filepath.Join(root, gbuildDeploymentV1CapnpFile), payload, 0o600))
			require.NoError(t, os.WriteFile(filepath.Join(root, "execution.v1.capnp.bin"), []byte("opaque runtime bytes"), 0o600))
			registry, err := NewModelRegistry("", nil)
			require.NoError(t, err)

			t.Log("Acquire and normalize exactly the scheduler contract")
			snapshot, err := registry.AcquireBuild(t.Context(), root)
			require.NoError(t, err)
			normalized := snapshot
			require.EqualValues(t, 4, normalized.ioFPGACount)
			require.Equal(t, test.family, normalized.family)

			t.Log("Pass the selected deployment file to Cyborg before authored environment references")
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineHybrid, Models: []string{"default"}, Build: normalized,
			})
			require.NoError(t, err)
			require.Len(t, projections, 1)
			projection := projections[0]
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
	require.Equal(t, BuildFamilyXT, snapshot.family)
	require.Empty(t, client.listRequests)
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
	_, err = registry.AcquireBuild(t.Context(), root)
	require.ErrorIs(t, err, errBuildFileTooLarge)
}

func TestDeploymentContractSupportedGeometry(t *testing.T) {
	const hybridCUDA = "hybrid CUDA"
	for _, name := range []string{"host XT", "HX", "HX subnode", "XT sixteen-chip node", "partial XT", "hybrid CPU", hybridCUDA, "CPU embeddings"} {
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
			wantChips, wantDevicesPerNode := 8, 8
			wantIO := int32(4)
			wantID := 0
			switch name {
			case "host XT":
				runtimeIO, err := config.RuntimeIo()
				require.NoError(t, err)
				runtimeIO.SetProtocol(deploymentcapnpv1.RuntimeIoProtocol_host)
				runtimeIO.SetIoFpgaCount(1)
				wantIO = 1
			case "HX", "HX subnode":
				detail.SetArchitecture(deploymentcapnpv1.LpuArchitecture_lp30)
				wantFamily = BuildFamilyHX
				detail.SetNumChips(16)
				detail.SetDevicesPerNode(16)
				wantChips, wantDevicesPerNode = 16, 16
				if name == "HX subnode" {
					detail.SetNumChips(8)
					wantChips = 8
				}
				setTestUnits(t, deployment, testSingleUnits(testHXReservation(1, 1, 1, 0)))
			case "XT sixteen-chip node":
				detail.SetNumChips(16)
				detail.SetDevicesPerNode(16)
				wantChips, wantDevicesPerNode = 16, 16
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
				setTestUnits(t, deployment, testSingleUnits(testNodesReservation(1, 3)))
				wantID = 3
			case "hybrid CPU", hybridCUDA:
				expanded, err := deployment.NewPartitions(2)
				require.NoError(t, err)
				require.NoError(t, expanded.Set(0, partitions.At(0)))
				ref, err := expanded.At(1).NewPartition()
				require.NoError(t, err)
				ref.SetPartitionId(1)
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

			t.Log("Keep the LPU chip geometry while excluding non-LPU execution details")
			build, err := buildFromGbuildDeploymentV1("gs://models/build", deployment)
			require.NoError(t, err)
			require.Equal(t, wantFamily, build.family)
			require.Equal(t, wantIO, build.ioFPGACount)
			require.Equal(t, name == "CPU embeddings", build.supportsCPUEmbeddings)
			require.Equal(t, name == hybridCUDA, build.hasCUDAPartitions)
			require.Equal(t, []buildPartition{{sourcePartitionID: wantID, numChips: wantChips, devicesPerNode: wantDevicesPerNode}}, build.partitions)
			require.Len(t, build.lpuUnits, 1)
		})
	}
}

func TestDeploymentContractRejectsCorruptNewOnly(t *testing.T) {
	t.Log("Acquire stable corrupt deployment bytes")
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, gbuildDeploymentV1CapnpFile), []byte("corrupt"), 0o600))
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)
	_, err = registry.AcquireBuild(t.Context(), root)
	require.ErrorContains(t, err, "parsing deployment.v1.capnp.bin")
}

func TestDeploymentContractLPUUnits(t *testing.T) {
	t.Parallel()
	const (
		lp20 = deploymentcapnpv1.LpuArchitecture_lp20
		lp30 = deploymentcapnpv1.LpuArchitecture_lp30
	)
	lp20Partitions := testUnitPartitions(8, 1, 2, 3)
	lp30Partitions := testUnitPartitions(16, 1, 2)
	lp20Units := []testLPUUnit{
		{reservations: []testLPUReservation{testNodesReservation(1, 1), testNodesReservation(1, 2)}, connectors: []testLPUConnector{{}}},
		{reservations: []testLPUReservation{testNodesReservation(1, 3)}},
	}
	eighteenStages := make([]uint32, 18)
	for index := range eighteenStages {
		eighteenStages[index] = uint32(index + 1)
	}
	for _, test := range []struct {
		name    string
		bundle  testUnitBundle
		want    []lpuUnit
		wantErr string
	}{
		{
			name:    "absent units",
			bundle:  testUnitBundle{architecture: lp20, devicesPerNode: 8, partitions: lp20Partitions},
			wantErr: "unsupported LPX runtime: deployment.v1.capnp.bin records no config.lpuUnits",
		},
		{
			name:    "empty units",
			bundle:  testUnitBundle{architecture: lp20, devicesPerNode: 8, partitions: lp20Partitions, units: []testLPUUnit{}},
			wantErr: "unsupported LPX runtime: deployment.v1.capnp.bin records no config.lpuUnits",
		},
		{
			name:   "LP20 chain and single partition",
			bundle: testUnitBundle{architecture: lp20, devicesPerNode: 8, partitions: lp20Partitions, units: lp20Units},
			want: []lpuUnit{
				{
					Reservations: []lpuReservation{{PartitionIDs: []int{1}, Hosts: 1}, {PartitionIDs: []int{2}, Hosts: 1}},
					Connectors: []lpxv1alpha1.PropSyncConnectorRequirement{
						{Kind: lpxv1alpha1.PropSyncConnectorKindXt8888Gap, MaxInterPartitionOffset: new(int64(0))},
					},
				},
				{Reservations: []lpuReservation{{PartitionIDs: []int{3}, Hosts: 1}}, Connectors: []lpxv1alpha1.PropSyncConnectorRequirement{}},
			},
		},
		{
			name: "LP30 chain with reservation-local device pairs",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16,
				partitions: []testUnitPartition{{id: 1, numChips: 32}, {id: 2, numChips: 16}},
				units: []testLPUUnit{{
					reservations: []testLPUReservation{testHXReservation(1, 2, 1, 1), testHXReservation(1, 1, 1, 2)},
					connectors:   []testLPUConnector{{hx: true, pairs: [][2]uint32{{16, 0}, {31, 15}}, lanes: []uint8{4, 2, 1}, dimension: 1}},
				}},
			},
			want: []lpuUnit{{
				Reservations: []lpuReservation{
					{PartitionIDs: []int{1}, Hosts: 2, HXExtent: []int64{16, 1, 2, 1}},
					{PartitionIDs: []int{2}, Hosts: 1, HXExtent: []int64{16, 1, 1, 1}},
				},
				Connectors: []lpxv1alpha1.PropSyncConnectorRequirement{{
					Kind:                         lpxv1alpha1.PropSyncConnectorKindHxPropSyncV1,
					Connections:                  &[]lpxv1alpha1.HxLogicalConnection{{FromLogicalDevice: 16, ToLogicalDevice: 0}, {FromLogicalDevice: 31, ToLogicalDevice: 15}},
					AcceptableLaneMultiplicities: &[]int64{4, 2, 1},
				}},
			}},
		},
		{
			name: "LP30 packed chain shares one reservation",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16,
				partitions: testUnitPartitions(8, 1, 2, 3, 4),
				units:      testSingleUnits(testHXReservation(2, 1, 1, 1, 2, 3, 4)),
			},
			want: []lpuUnit{{
				Reservations: []lpuReservation{{PartitionIDs: []int{1, 2, 3, 4}, Hosts: 2, HXExtent: []int64{16, 2, 1, 1}}},
				Connectors:   []lpxv1alpha1.PropSyncConnectorRequirement{},
			}},
		},
		{
			name: "LP30 two full racks",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(512, 1),
				units: testSingleUnits(testHXReservation(8, 2, 2, 1)),
			},
			want: []lpuUnit{{
				Reservations: []lpuReservation{{PartitionIDs: []int{1}, Hosts: 32, HXExtent: []int64{16, 8, 2, 2}}},
				Connectors:   []lpxv1alpha1.PropSyncConnectorRequirement{},
			}},
		},
		{
			name: "LP30 partial selection placed by units",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partSelect: []uint32{3}, partitions: testUnitPartitions(16, 3),
				units: testSingleUnits(testHXReservation(1, 1, 1, 3)),
			},
			want: []lpuUnit{{
				Reservations: []lpuReservation{{PartitionIDs: []int{3}, Hosts: 1, HXExtent: []int64{16, 1, 1, 1}}},
				Connectors:   []lpxv1alpha1.PropSyncConnectorRequirement{},
			}},
		},
		{
			name: "LP30 partial selection without units",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partSelect: []uint32{3}, partitions: testUnitPartitions(16, 3),
			},
			wantErr: "records no config.lpuUnits",
		},
		{
			name: "hybrid chain of small stages keeps one reservation per stage",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, hybrid: true, partitions: testUnitPartitions(4, 1, 2),
				units: []testLPUUnit{{reservations: []testLPUReservation{testNodesReservation(1, 1), testNodesReservation(1, 2)}, connectors: []testLPUConnector{{}}}},
			},
			want: []lpuUnit{{
				Reservations: []lpuReservation{{PartitionIDs: []int{1}, Hosts: 1}, {PartitionIDs: []int{2}, Hosts: 1}},
				Connectors: []lpxv1alpha1.PropSyncConnectorRequirement{
					{Kind: lpxv1alpha1.PropSyncConnectorKindXt8888Gap, MaxInterPartitionOffset: new(int64(0))},
				},
			}},
		},
		{
			name: "LPU-only CPU embeddings leave partition 0 out of units",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, cpuEmbeddings: true, partitions: testUnitPartitions(8, 0, 1),
				units: testSingleUnits(testNodesReservation(1, 1)),
			},
			want: []lpuUnit{{Reservations: []lpuReservation{{PartitionIDs: []int{1}, Hosts: 1}}, Connectors: []lpxv1alpha1.PropSyncConnectorRequirement{}}},
		},
		{
			name: "LPU-only CPU embeddings place partition 0",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, cpuEmbeddings: true, partitions: testUnitPartitions(8, 0, 1),
				units: testSingleUnits(testNodesReservation(1, 0), testNodesReservation(1, 1)),
			},
			wantErr: "deployment.v1.capnp.bin config.lpuUnits places LPU partition 0, which an LPU-only build with CPU token embeddings runs on the CPU",
		},
		{
			name: "hybrid CPU embeddings still place partition 0",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, hybrid: true, cpuEmbeddings: true, partitions: testUnitPartitions(8, 0, 1),
				units: testSingleUnits(testNodesReservation(1, 1)),
			},
			wantErr: "deployment.v1.capnp.bin config.lpuUnits omits LPU partition 0",
		},
		{
			name: "LPU-only build without CPU embeddings places partition 0",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(8, 0, 1),
				units: testSingleUnits(testNodesReservation(1, 1)),
			},
			wantErr: "deployment.v1.capnp.bin config.lpuUnits omits LPU partition 0",
		},
		{
			name: "uncovered partition",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: lp20Partitions,
				units: testSingleUnits(testNodesReservation(1, 1), testNodesReservation(1, 2)),
			},
			wantErr: "deployment.v1.capnp.bin config.lpuUnits omits LPU partition 3",
		},
		{
			name: "unknown partition",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: lp20Partitions,
				units: testSingleUnits(testNodesReservation(1, 1), testNodesReservation(1, 2), testNodesReservation(1, 3), testNodesReservation(1, 9)),
			},
			wantErr: "deployment.v1.capnp.bin config.lpuUnits[3].reservations[0].partitionIds references partition 9, which is not a deployed LPU partition",
		},
		{
			name: "CUDA partition",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, hybrid: true, partitions: testUnitPartitions(8, 1, 2),
				units: testSingleUnits(testNodesReservation(1, 1), testNodesReservation(1, 2), testNodesReservation(1, 3)),
			},
			wantErr: "config.lpuUnits[2].reservations[0].partitionIds references partition 3, which is not a deployed LPU partition",
		},
		{
			name: "partition in two reservations",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: lp20Partitions,
				units: testSingleUnits(testNodesReservation(1, 1), testNodesReservation(1, 2), testNodesReservation(1, 3), testNodesReservation(1, 3)),
			},
			wantErr: "config.lpuUnits[3].reservations[0].partitionIds repeats partition 3",
		},
		{
			name: "units out of root order",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: lp20Partitions,
				units: testSingleUnits(testNodesReservation(1, 2), testNodesReservation(1, 1), testNodesReservation(1, 3)),
			},
			wantErr: "config.lpuUnits[1] root partition 1 does not follow root partition 2; units must ascend by root",
		},
		{
			name: "unit skips a partition ID",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: lp20Partitions,
				units: []testLPUUnit{
					{reservations: []testLPUReservation{testNodesReservation(1, 1), testNodesReservation(1, 3)}, connectors: []testLPUConnector{{}}},
					{reservations: []testLPUReservation{testNodesReservation(1, 2)}},
				},
			},
			wantErr: "config.lpuUnits[0].reservations[1].partitionIds lists partition 3 after 1; a unit holds consecutive ascending partition IDs",
		},
		{
			name: "unit without reservations",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: lp20Partitions,
				units: []testLPUUnit{{}},
			},
			wantErr: "config.lpuUnits[0].reservations must not be empty",
		},
		{
			name: "reservation without partitions",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: lp20Partitions,
				units: testSingleUnits(testNodesReservation(1)),
			},
			wantErr: "config.lpuUnits[0].reservations[0].partitionIds must not be empty",
		},
		{
			name: "zero nodes",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: lp20Partitions,
				units: testSingleUnits(testNodesReservation(0, 1), testNodesReservation(1, 2), testNodesReservation(1, 3)),
			},
			wantErr: "config.lpuUnits[0].reservations[0].size.nodes must be in [1, 2147483647], got 0",
		},
		{
			name: "nodes beyond the Agent replica range",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(8, 1),
				units: testSingleUnits(testNodesReservation(2147483648, 1)),
			},
			wantErr: "config.lpuUnits[0].reservations[0].size.nodes must be in [1, 2147483647], got 2147483648",
		},
		{
			name: "LP20 reservation sized as an HX extent",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(8, 1),
				units: testSingleUnits(testHXReservation(1, 1, 1, 1)),
			},
			wantErr: "config.lpuUnits[0].reservations[0].size.hx sizes an LP30 reservation; LP20 reservations use size.nodes",
		},
		{
			name: "zero HX dimension",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(16, 1),
				units: testSingleUnits(testHXReservation(1, 0, 1, 1)),
			},
			wantErr: "config.lpuUnits[0].reservations[0].size.hx {traysPerHemisphere: 1, hemispheres: 0, racks: 1} is not a supported HX extent",
		},
		{
			name: "HX extent across racks without full hemispheres",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(16, 1),
				units: testSingleUnits(testHXReservation(1, 1, 2, 1)),
			},
			wantErr: "{traysPerHemisphere: 1, hemispheres: 1, racks: 2} is not a supported HX extent",
		},
		{
			name: "reservation smaller than its partition",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(16, 1),
				units: testSingleUnits(testNodesReservation(1, 1)),
			},
			wantErr: "config.lpuUnits[0].reservations[0] reserves 8 devices for 16 chips",
		},
		{
			name: "LP20 reservation larger than its partition keeps the compiler's hosts",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(16, 1),
				units: testSingleUnits(testNodesReservation(3, 1)),
			},
			want: []lpuUnit{{Reservations: []lpuReservation{{PartitionIDs: []int{1}, Hosts: 3}}, Connectors: []lpxv1alpha1.PropSyncConnectorRequirement{}}},
		},
		{
			name: "packed stages on too few hosts",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(4, 1, 2, 3),
				units: testSingleUnits(testNodesReservation(1, 1, 2, 3)),
			},
			wantErr: "config.lpuUnits[0].reservations[0] reserves 8 devices for 12 chips",
		},
		{
			name: "LP30 extent comes from the compiler's reservation alone",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(32, 1),
				units: testSingleUnits(testHXReservation(2, 1, 1, 1)),
			},
			want: []lpuUnit{{
				Reservations: []lpuReservation{{PartitionIDs: []int{1}, Hosts: 2, HXExtent: []int64{16, 2, 1, 1}}},
				Connectors:   []lpxv1alpha1.PropSyncConnectorRequirement{},
			}},
		},
		{
			name: "LP30 one-node partition without HX metadata on a two-device node sized in nodes",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 2, partitions: testUnitPartitions(2, 1),
				units: testSingleUnits(testNodesReservation(1, 1)),
			},
			wantErr: "unsupported LPX runtime: deployment.v1.capnp.bin config.lpuUnits[0].reservations[0].size.nodes sizes an LP30 reservation, " +
				"which an HX LPU pipeline request cannot express; LP30 reservations use size.hx",
		},
		{
			name: "LP30 packed chain on hosts that no HX extent holds",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: testUnitPartitions(8, eighteenStages...),
				units: testSingleUnits(testNodesReservation(9, eighteenStages...)),
			},
			wantErr: "unsupported LPX runtime: deployment.v1.capnp.bin config.lpuUnits[0].reservations[0].size.nodes sizes an LP30 reservation, " +
				"which an HX LPU pipeline request cannot express; LP30 reservations use size.hx",
		},
		{
			name: "hybrid chain packed onto shared hosts",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, hybrid: true, partitions: testUnitPartitions(4, 1, 2),
				units: testSingleUnits(testNodesReservation(1, 1, 2)),
			},
			wantErr: "config.lpuUnits[0].reservations[0] packs partitions [1,2] onto shared hosts, which only LPU-only bundles without CUDA partitions do",
		},
		{
			name: "packed reservation beside another reservation",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: testUnitPartitions(4, 1, 2, 3),
				units: []testLPUUnit{{reservations: []testLPUReservation{testNodesReservation(1, 1, 2), testNodesReservation(1, 3)}, connectors: []testLPUConnector{{}}}},
			},
			wantErr: "config.lpuUnits[0].reservations[0] packs partitions [1,2] beside other reservations; a packed chain is its unit's only reservation",
		},
		{
			name: "missing connector",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: lp20Partitions,
				units: []testLPUUnit{
					{reservations: []testLPUReservation{testNodesReservation(1, 1), testNodesReservation(1, 2)}},
					{reservations: []testLPUReservation{testNodesReservation(1, 3)}},
				},
			},
			wantErr: "config.lpuUnits[0] has 0 connectors for 2 reservations, want 1",
		},
		{
			name: "LP20 unit with an HX connector",
			bundle: testUnitBundle{
				architecture: lp20, devicesPerNode: 8, partitions: lp20Partitions,
				units: []testLPUUnit{
					{reservations: []testLPUReservation{testNodesReservation(1, 1), testNodesReservation(1, 2)}, connectors: []testLPUConnector{testHXConventionConnector(8)}},
					{reservations: []testLPUReservation{testNodesReservation(1, 3)}},
				},
			},
			wantErr: "config.lpuUnits[0].connectors[0] is hxPropSyncV1; LP20 units use xt8888Gap",
		},
		{
			name: "LP30 unit with an XT connector",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: lp30Partitions,
				units: []testLPUUnit{{reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1), testHXReservation(1, 1, 1, 2)}, connectors: []testLPUConnector{{}}}},
			},
			wantErr: "config.lpuUnits[0].connectors[0] is xt8888Gap; LP30 units use hxPropSyncV1",
		},
		{
			name: "source device outside its reservation",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: lp30Partitions,
				units: []testLPUUnit{{
					reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1), testHXReservation(1, 1, 1, 2)},
					connectors:   []testLPUConnector{{hx: true, pairs: [][2]uint32{{16, 0}}, lanes: []uint8{4}}},
				}},
			},
			wantErr: "config.lpuUnits[0].connectors[0].hxPropSyncV1.connections[0] 16->0 is outside the joined reservations' 16 and 16 devices",
		},
		{
			name: "destination device outside its reservation",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: lp30Partitions,
				units: []testLPUUnit{{
					reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1), testHXReservation(1, 1, 1, 2)},
					connectors:   []testLPUConnector{{hx: true, pairs: [][2]uint32{{15, 15}, {0, 16}}, lanes: []uint8{4}}},
				}},
			},
			wantErr: "connections[1] 0->16 is outside the joined reservations' 16 and 16 devices",
		},
		{
			name: "repeated device pair",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: lp30Partitions,
				units: []testLPUUnit{{
					reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1), testHXReservation(1, 1, 1, 2)},
					connectors:   []testLPUConnector{{hx: true, pairs: [][2]uint32{{0, 0}, {0, 0}}, lanes: []uint8{4}}},
				}},
			},
			wantErr: "connections[1] repeats device pair 0->0",
		},
		{
			name: "no device pairs",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: lp30Partitions,
				units: []testLPUUnit{{
					reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1), testHXReservation(1, 1, 1, 2)},
					connectors:   []testLPUConnector{{hx: true, lanes: []uint8{4}}},
				}},
			},
			wantErr: "config.lpuUnits[0].connectors[0].hxPropSyncV1.connections must not be empty",
		},
		{
			name: "no lane multiplicities",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: lp30Partitions,
				units: []testLPUUnit{{
					reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1), testHXReservation(1, 1, 1, 2)},
					connectors:   []testLPUConnector{{hx: true, pairs: [][2]uint32{{0, 0}}}},
				}},
			},
			wantErr: "config.lpuUnits[0].connectors[0].hxPropSyncV1.acceptableLaneMultiplicities must not be empty",
		},
		{
			name: "zero lane multiplicity",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: lp30Partitions,
				units: []testLPUUnit{{
					reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1), testHXReservation(1, 1, 1, 2)},
					connectors:   []testLPUConnector{{hx: true, pairs: [][2]uint32{{0, 0}}, lanes: []uint8{4, 0}}},
				}},
			},
			wantErr: "acceptableLaneMultiplicities[1] must be positive",
		},
		{
			name: "LP30 boundary compiled across hemispheres",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: lp30Partitions,
				units: []testLPUUnit{{
					reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1), testHXReservation(1, 1, 1, 2)},
					connectors:   []testLPUConnector{{hx: true, pairs: [][2]uint32{{0, 0}}, lanes: []uint8{4, 2, 1}, dimension: 2}},
				}},
			},
			wantErr: "unsupported LPX runtime: deployment.v1.capnp.bin config.lpuUnits[0].connectors[0].hxPropSyncV1.topologyDimension = 2; " +
				"an LPU pipeline request cannot carry a topology dimension, so the operator deploys only boundaries compiled at " +
				"the HX16x8x2 default dimension 1 (trays of one hemisphere)",
		},
		{
			name: "LP30 boundary without a topology dimension",
			bundle: testUnitBundle{
				architecture: lp30, devicesPerNode: 16, partitions: lp30Partitions,
				units: []testLPUUnit{{
					reservations: []testLPUReservation{testHXReservation(1, 1, 1, 1), testHXReservation(1, 1, 1, 2)},
					connectors:   []testLPUConnector{{hx: true, pairs: [][2]uint32{{0, 0}}, lanes: []uint8{4, 2, 1}}},
				}},
			},
			wantErr: "unsupported LPX runtime: deployment.v1.capnp.bin config.lpuUnits[0].connectors[0].hxPropSyncV1.topologyDimension = 0;",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Decode the deployment contract and its units")
			build, err := acquireTestUnitBundle(t, test.bundle)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)

			t.Log("Keep validated units in Agent order")
			require.Equal(t, test.want, build.lpuUnits)
		})
	}
}
