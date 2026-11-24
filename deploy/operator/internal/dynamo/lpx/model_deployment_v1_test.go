/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"os"
	"path/filepath"
	"testing"

	commoncapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/common/v1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/testfixture"
	"github.com/stretchr/testify/require"
)

func newDeploymentV1ContractFixture(t *testing.T) deploymentcapnpv1.Deployment {
	t.Helper()
	return testfixture.NewDeployment(t, []testfixture.Partition{{ID: 0, DeviceType: commoncapnpv1.DeviceType_lpu, Architecture: deploymentcapnpv1.LpuArchitecture_lp20, DevicesPerNode: 8}}, testSingleUnits(testNodesReservation(1, 0)), 4, 2)
}

func deploymentPayload(t *testing.T) []byte {
	t.Helper()
	payload, err := newDeploymentV1ContractFixture(t).Message().Marshal()
	require.NoError(t, err)
	return payload
}

func writeDeploymentPayload(t *testing.T, buildDir string, payload []byte) {
	t.Helper()
	require.NoError(t, os.MkdirAll(buildDir, 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, gbuildDeploymentV1CapnpFile), payload, 0o600))
}

// testUnitBundle describes a deployment.v1 contract with compiler-owned LPU units.
type testUnitBundle struct {
	architecture   deploymentcapnpv1.LpuArchitecture
	devicesPerNode uint32
	ioFPGACount    uint32
	ioFanoutFactor uint32
	hybrid         bool
	partitions     []testUnitPartition
	// units is written as config.lpuUnits; nil leaves the field unset.
	units []testLPUUnit
}

type testUnitPartition struct {
	id uint32
}

type testLPUUnit = testfixture.Unit
type testLPUReservation = testfixture.Reservation
type testLPUConnector = testfixture.Connector

// testUnitPartitions returns the authored LPU partition IDs.
func testUnitPartitions(ids ...uint32) []testUnitPartition {
	partitions := make([]testUnitPartition, len(ids))
	for index, id := range ids {
		partitions[index] = testUnitPartition{id: id}
	}
	return partitions
}

func testNodesReservation(nodes uint32, ids ...uint32) testLPUReservation {
	return testLPUReservation{IDs: ids, Nodes: nodes}
}

func testHXReservation(traysPerHemisphere, hemispheres, racks uint8, ids ...uint32) testLPUReservation {
	return testLPUReservation{IDs: ids, HX: []uint8{traysPerHemisphere, hemispheres, racks}}
}

// testSingleUnits places each reservation in its own unit.
func testSingleUnits(reservations ...testLPUReservation) []testLPUUnit {
	units := make([]testLPUUnit, len(reservations))
	for index, reservation := range reservations {
		units[index] = testLPUUnit{Reservations: []testLPUReservation{reservation}}
	}
	return units
}

// testHXConventionConnector joins a source reservation's last host, device by
// device, to the destination's first host, with lanes [4, 2, 1].
func testHXConventionConnector(sourceHosts, devicesPerNode uint32) testLPUConnector {
	connector := testLPUConnector{HX: true}
	for device := range devicesPerNode {
		connector.Pairs = append(connector.Pairs, [2]uint32{(sourceHosts-1)*devicesPerNode + device, device})
	}
	return connector
}

// encodeTestUnitBundle serializes the bundle as deployment.v1.capnp.bin bytes.
func encodeTestUnitBundle(t *testing.T, bundle testUnitBundle) []byte {
	t.Helper()
	partitions := make([]testfixture.Partition, 0, len(bundle.partitions)+1)
	lastID := uint32(0)
	for _, partition := range bundle.partitions {
		partitions = append(partitions, testfixture.Partition{ID: partition.id, DeviceType: commoncapnpv1.DeviceType_lpu, Architecture: bundle.architecture, DevicesPerNode: bundle.devicesPerNode})
		lastID = max(lastID, partition.id)
	}
	if bundle.hybrid {
		partitions = append(partitions, testfixture.Partition{ID: lastID + 1, DeviceType: commoncapnpv1.DeviceType_cuda})
	}
	deployment := testfixture.NewDeployment(t, partitions, bundle.units, max(1, bundle.ioFPGACount), max(1, bundle.ioFanoutFactor))
	data, err := deployment.Message().Marshal()
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
	for _, name := range []string{"revision", "future revision", "zero endpoints", "zero fanout", "unknown device", "mismatched detail", "missing architecture", "unknown architecture"} {
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
				wantErr = "unsupported LPU architecture"
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
			case "mismatched detail":
				partitions, err := deployment.Partitions()
				require.NoError(t, err)
				_, err = partitions.At(0).Detail().NewCuda()
				require.NoError(t, err)
				wantErr = "LPU partition 0 has mismatched detail"
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
