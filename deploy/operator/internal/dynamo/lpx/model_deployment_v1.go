/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"math"

	"capnproto.org/go/capnp/v3"
	commoncapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/common/v1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
)

const gbuildDeploymentV1CapnpFile = "deployment.v1.capnp.bin"

// decodeGbuildDeploymentV1 reads the scheduler contract without execution/report schemas.
func decodeGbuildDeploymentV1(data []byte) (deploymentcapnpv1.Deployment, error) {
	msg, err := capnp.Unmarshal(data)
	if err != nil {
		return deploymentcapnpv1.Deployment{}, fmt.Errorf("parsing %s: %w", gbuildDeploymentV1CapnpFile, err)
	}
	deployment, err := deploymentcapnpv1.ReadRootDeployment(msg)
	if err != nil {
		return deploymentcapnpv1.Deployment{}, fmt.Errorf("reading %s root: %w", gbuildDeploymentV1CapnpFile, err)
	}
	return deployment, nil
}

// buildFromGbuildDeploymentV1 decodes supported compiler input using the acquired canonical build reference.
func buildFromGbuildDeploymentV1(buildRef string, manifest deploymentcapnpv1.Deployment) (*Build, error) {
	// The operator reads contract revision 1 only.
	if manifest.ContractRevision() != deploymentcapnpv1.CurrentContractRevision {
		return nil, fmt.Errorf(
			"%s contractRevision = %d, want %d",
			gbuildDeploymentV1CapnpFile,
			manifest.ContractRevision(),
			deploymentcapnpv1.CurrentContractRevision,
		)
	}

	// The config holds the units and the runtime I/O.
	if !manifest.HasConfig() {
		return nil, fmt.Errorf("%s is missing config", gbuildDeploymentV1CapnpFile)
	}
	deployment, err := manifest.Config()
	if err != nil {
		return nil, fmt.Errorf("reading %s config: %w", gbuildDeploymentV1CapnpFile, err)
	}

	// Placement comes from the units alone; a bundle without units needs a recompile.
	rawUnits, err := deployment.LpuUnits()
	if err != nil {
		return nil, fmt.Errorf("reading %s config.lpuUnits: %w", gbuildDeploymentV1CapnpFile, err)
	}
	if rawUnits.Len() == 0 {
		return nil, fmt.Errorf("%s config.lpuUnits is empty; recompile the model with LPU units", gbuildDeploymentV1CapnpFile)
	}

	// An invalid record of a remaining build fact fails the acquisition.
	ioFPGACount, ioFanoutFactor, err := runtimeIOFromDeploymentV1(deployment)
	if err != nil {
		return nil, err
	}
	checkpoint, err := checkpointFromDeploymentV1(manifest)
	if err != nil {
		return nil, err
	}
	architecture, devicesPerNode, hasCUDAPartitions, err := deviceMetadataFromDeploymentV1(manifest)
	if err != nil {
		return nil, err
	}

	// Keep the generated unit list and its backing message until projection is complete.
	return &Build{
		path:              buildRef,
		architecture:      architecture,
		devicesPerNode:    devicesPerNode,
		hasCUDAPartitions: hasCUDAPartitions,
		lpuUnits:          rawUnits,
		ioFPGACount:       ioFPGACount,
		ioFanoutFactor:    ioFanoutFactor,
		checkpoint:        checkpoint,
	}, nil
}

func lpuConnectorFromDeploymentV1(raw deploymentcapnpv1.LpuConnector, subject string) (lpxv1alpha1.PropSyncConnectorRequirement, error) {
	switch raw.Which() {
	case deploymentcapnpv1.LpuConnector_Which_xt8888Gap:
		offset := int64(0)
		return lpxv1alpha1.PropSyncConnectorRequirement{Kind: lpxv1alpha1.PropSyncConnectorKindXt8888Gap, MaxInterPartitionOffset: &offset}, nil
	case deploymentcapnpv1.LpuConnector_Which_hxPropSyncV1:
		connector, err := raw.HxPropSyncV1()
		if err != nil {
			return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("reading %s.hxPropSyncV1: %w", subject, err)
		}
		pairs, err := connector.Connections()
		if err != nil {
			return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("reading %s.connections: %w", subject, err)
		}

		connections := make([]lpxv1alpha1.HxLogicalConnection, pairs.Len())
		for i := range pairs.Len() {
			pair := pairs.At(i)
			connections[i] = lpxv1alpha1.HxLogicalConnection{FromLogicalDevice: int64(pair.FromDevice()), ToLogicalDevice: int64(pair.ToDevice())}
		}
		lanes := []int64{4, 2, 1}
		return lpxv1alpha1.PropSyncConnectorRequirement{Kind: lpxv1alpha1.PropSyncConnectorKindHxPropSyncV1, Connections: &connections, AcceptableLaneMultiplicities: &lanes}, nil
	default:
		return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("%s has unsupported variant %d", subject, raw.Which())
	}
}

func runtimeIOFromDeploymentV1(deployment deploymentcapnpv1.DeploymentInfo) (int32, int32, error) {
	if !deployment.HasRuntimeIo() {
		return 0, 0, fmt.Errorf("%s config.runtimeIo is missing", gbuildDeploymentV1CapnpFile)
	}
	runtimeIO, err := deployment.RuntimeIo()
	if err != nil {
		return 0, 0, fmt.Errorf("reading %s config.runtimeIo: %w", gbuildDeploymentV1CapnpFile, err)
	}
	count := runtimeIO.IoFpgaCount()
	if count == 0 || count > math.MaxInt32 {
		return 0, 0, fmt.Errorf("%s config.runtimeIo.ioFpgaCount must be in [1, %d], got %d", gbuildDeploymentV1CapnpFile, math.MaxInt32, count)
	}

	// Fanout is a separate positive client count for each physical endpoint.
	fanoutFactor := runtimeIO.FanoutFactor()
	if fanoutFactor == 0 || fanoutFactor > math.MaxInt32 {
		return 0, 0, fmt.Errorf("%s config.runtimeIo.fanoutFactor must be in [1, %d], got %d", gbuildDeploymentV1CapnpFile, math.MaxInt32, fanoutFactor)
	}
	return int32(count), int32(fanoutFactor), nil
}

// deviceMetadataFromDeploymentV1 returns the LPU architecture and the device count per node of
// the build, and whether the build has a CUDA partition. The compiler gives all LPU partitions
// one architecture and one count, so the last LPU partition supplies the two values.
func deviceMetadataFromDeploymentV1(artifacts deploymentcapnpv1.Deployment) (lpuArchitecture, int, bool, error) {
	rawPartitions, err := artifacts.Partitions()
	if err != nil {
		return 0, 0, false, fmt.Errorf("reading %s partitions: %w", gbuildDeploymentV1CapnpFile, err)
	}

	// The device type of a partition selects the fact that the build keeps.
	var (
		architecture      lpuArchitecture
		devicesPerNode    int
		hasCUDAPartitions bool
	)
	for index := range rawPartitions.Len() {
		raw := rawPartitions.At(index)
		ref, err := raw.Partition()
		if err != nil {
			return 0, 0, false, fmt.Errorf("reading %s artifact partition ref: %w", gbuildDeploymentV1CapnpFile, err)
		}

		switch ref.DeviceType() {
		case commoncapnpv1.DeviceType_lpu:
			// The generated accessor panics on a detail of a different device type.
			if !raw.Detail().HasLpu() {
				return 0, 0, false, fmt.Errorf("LPU partition %d has mismatched detail", ref.PartitionId())
			}
			detail, err := raw.Detail().Lpu()
			if err != nil {
				return 0, 0, false, fmt.Errorf("reading %s LPU partition %d detail: %w", gbuildDeploymentV1CapnpFile, ref.PartitionId(), err)
			}

			// Only LP20 and LP30 have a scheduler target family.
			architecture = detail.Architecture()
			switch architecture {
			case lp20, lp30:
			default:
				return 0, 0, false, fmt.Errorf("unsupported LPU architecture %q", architecture.String())
			}

			devicesPerNode = int(detail.DevicesPerNode())
		case commoncapnpv1.DeviceType_cuda:
			hasCUDAPartitions = true
		case commoncapnpv1.DeviceType_cpu:
		default:
			return 0, 0, false, fmt.Errorf("partition %d has unsupported deviceType %d", ref.PartitionId(), ref.DeviceType())
		}
	}

	return architecture, devicesPerNode, hasCUDAPartitions, nil
}
