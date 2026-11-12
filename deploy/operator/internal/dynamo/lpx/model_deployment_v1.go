/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"math"
	"slices"

	"capnproto.org/go/capnp/v3"
	commoncapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/common/v1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
)

const gbuildDeploymentV1CapnpFile = "deployment.v1.capnp.bin"

// buildContractFormat selects a supported compiler contract and runtime entry file.
type buildContractFormat uint8

const (
	buildContractUnknown buildContractFormat = iota
	buildContractManifestV2
	buildContractDeploymentV1
)

// selectBuildContract selects exactly one entry point from a normalized inventory.
func selectBuildContract(inventory []string) (buildContractFormat, error) {
	legacy := slices.Contains(inventory, gbuildManifestV2CapnpFile)
	split := slices.Contains(inventory, gbuildDeploymentV1CapnpFile)
	switch {
	case legacy && split:
		return buildContractUnknown, fmt.Errorf("ambiguous build contracts: both %s and %s", gbuildManifestV2CapnpFile, gbuildDeploymentV1CapnpFile)
	case legacy:
		return buildContractManifestV2, nil
	case split:
		return buildContractDeploymentV1, nil
	default:
		return buildContractUnknown, fmt.Errorf("missing build contract: require %s or %s", gbuildManifestV2CapnpFile, gbuildDeploymentV1CapnpFile)
	}
}

// filename returns the selected runtime entry point; unknown formats are invalid.
func (format buildContractFormat) filename() (string, error) {
	switch format {
	case buildContractManifestV2:
		return gbuildManifestV2CapnpFile, nil
	case buildContractDeploymentV1:
		return gbuildDeploymentV1CapnpFile, nil
	default:
		return "", fmt.Errorf("unsupported build contract format %d", format)
	}
}

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
	if manifest.ContractRevision() != deploymentcapnpv1.CurrentContractRevision {
		return nil, fmt.Errorf(
			"%s contractRevision = %d, want %d",
			gbuildDeploymentV1CapnpFile,
			manifest.ContractRevision(),
			deploymentcapnpv1.CurrentContractRevision,
		)
	}
	if !manifest.HasConfig() {
		return nil, fmt.Errorf("%s is missing config", gbuildDeploymentV1CapnpFile)
	}
	deployment, err := manifest.Config()
	if err != nil {
		return nil, fmt.Errorf("reading %s config: %w", gbuildDeploymentV1CapnpFile, err)
	}
	if !deployment.HasProgram() {
		return nil, fmt.Errorf("%s config.program is missing", gbuildDeploymentV1CapnpFile)
	}
	program, err := deployment.Program()
	if err != nil {
		return nil, fmt.Errorf("reading %s config.program: %w", gbuildDeploymentV1CapnpFile, err)
	}
	rawUnits, err := deployment.LpuUnits()
	if err != nil {
		return nil, fmt.Errorf("reading %s config.lpuUnits: %w", gbuildDeploymentV1CapnpFile, err)
	}
	var chains [][]int
	if rawUnits.Len() == 0 {
		chains, err = selectedPropSyncChainsFromDeploymentV1(deployment)
		if err != nil {
			return nil, err
		}
	}
	ioFPGACount, ioFanoutFactor, err := runtimeIOFromDeploymentV1(deployment)
	if err != nil {
		return nil, err
	}

	checkpoint, checkpointErr := checkpointFromDeploymentV1(manifest)

	build := &Build{
		path:                      buildRef,
		selectedPropSyncChains:    chains,
		standaloneTokenEmbeddings: program.StandaloneTokenEmbeddings(),
		supportsCPUEmbeddings:     program.SupportsCpuEmbeddings(),
		ioFPGACount:               ioFPGACount,
		ioFanoutFactor:            ioFanoutFactor,
		checkpoint:                checkpoint,
		checkpointErr:             checkpointErr,
	}

	// Complete the normalized build with scheduler-facing LPU artifacts.
	if err := addLPUArtifactsFromDeploymentV1(manifest, build, rawUnits.Len() != 0); err != nil {
		return nil, err
	}

	// Keep the generated list and its backing message until projection is complete.
	build.lpuUnits = rawUnits

	return build, nil
}

func unitRoot(unit deploymentcapnpv1.LpuUnit) (int, error) {
	reservations, err := unit.Reservations()
	if err != nil {
		return 0, fmt.Errorf("reading unit reservations: %w", err)
	}
	if reservations.Len() == 0 {
		return 0, fmt.Errorf("lpuUnit has no root partition")
	}
	return reservationRoot(reservations.At(0))
}

func reservationRoot(reservation deploymentcapnpv1.LpuReservation) (int, error) {
	ids, err := reservation.PartitionIds()
	if err != nil {
		return 0, fmt.Errorf("reading reservation partition IDs: %w", err)
	}
	if ids.Len() == 0 {
		return 0, fmt.Errorf("lpuUnits reservation has no root partition")
	}
	return int(ids.At(0)), nil
}

func reservationSize(raw deploymentcapnpv1.LpuReservation, devicesPerNode int) (int, []int64, error) {
	size := raw.Size()
	switch size.Which() {
	case deploymentcapnpv1.LpuReservation_size_Which_nodes:
		nodes := size.Nodes()
		if nodes == 0 || nodes > math.MaxInt32 {
			return 0, nil, fmt.Errorf("lpuUnits size.nodes must be in [1, %d], got %d", math.MaxInt32, nodes)
		}
		return int(nodes), nil, nil
	case deploymentcapnpv1.LpuReservation_size_Which_hx:
		extent, err := size.Hx()
		if err != nil {
			return 0, nil, fmt.Errorf("reading lpuUnits size.hx: %w", err)
		}
		trays, hemispheres, racks := int64(extent.TraysPerHemisphere()), int64(extent.Hemispheres()), int64(extent.Racks())
		return int(trays * hemispheres * racks), []int64{int64(devicesPerNode), trays, hemispheres, racks}, nil
	default:
		return 0, nil, fmt.Errorf("lpuUnits size has unsupported variant %d", size.Which())
	}
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

func selectedPropSyncChainsFromDeploymentV1(deployment deploymentcapnpv1.DeploymentInfo) ([][]int, error) {
	if !deployment.HasSelectedPropSyncChains() {
		return nil, nil
	}
	rawChains, err := deployment.SelectedPropSyncChains()
	if err != nil {
		return nil, fmt.Errorf("reading %s config.selectedPropSyncChains: %w", gbuildDeploymentV1CapnpFile, err)
	}
	chains := make([][]int, 0, rawChains.Len())
	for chainIndex := 0; chainIndex < rawChains.Len(); chainIndex++ {
		rawChain := rawChains.At(chainIndex)
		if !rawChain.HasPartitionIds() {
			return nil, fmt.Errorf("%s config.selectedPropSyncChains[%d].partitionIds is missing", gbuildDeploymentV1CapnpFile, chainIndex)
		}
		rawIDs, err := rawChain.PartitionIds()
		if err != nil {
			return nil, fmt.Errorf("reading %s config.selectedPropSyncChains[%d].partitionIds: %w", gbuildDeploymentV1CapnpFile, chainIndex, err)
		}

		chainPath := fmt.Sprintf("%s config.selectedPropSyncChains[%d]", gbuildDeploymentV1CapnpFile, chainIndex)
		chain, err := decodePropSyncChain(rawIDs, chainPath)
		if err != nil {
			return nil, err
		}
		chains = append(chains, chain)
	}
	return chains, nil
}

func addLPUArtifactsFromDeploymentV1(
	artifacts deploymentcapnpv1.Deployment,
	build *Build,
	units bool,
) error {
	rawPartitions, err := artifacts.Partitions()
	if err != nil {
		return fmt.Errorf("reading %s partitions: %w", gbuildDeploymentV1CapnpFile, err)
	}
	partitions := make([]buildPartition, 0, rawPartitions.Len())
	for index := 0; index < rawPartitions.Len(); index++ {
		partition, err := buildPartitionFromDeploymentV1(rawPartitions.At(index), units)
		if err != nil {
			return err
		}
		// Project LPU artifacts while retaining whether the build also requires CUDA.
		if partition.partPath != "" {
			partitions = append(partitions, partition)
		} else {
			ref, err := rawPartitions.At(index).Partition()
			if err != nil {
				return fmt.Errorf("reading %s artifact partition ref: %w", gbuildDeploymentV1CapnpFile, err)
			}
			build.hasCUDAPartitions = build.hasCUDAPartitions || ref.DeviceType() == commoncapnpv1.DeviceType_cuda
		}
	}
	if len(partitions) == 0 {
		return fmt.Errorf("%s artifacts contain no LPU partitions", gbuildDeploymentV1CapnpFile)
	}
	// Units already encode inventory order and ownership. One Agent template
	// still requires a common target family and device width across the build.
	family := partitions[0].family
	if units {
		for _, partition := range partitions {
			if partition.family != family {
				return fmt.Errorf("%s mixes XT and HX LPU partitions", gbuildDeploymentV1CapnpFile)
			}
			if partition.devicesPerNode != partitions[0].devicesPerNode {
				return fmt.Errorf("all LPU partitions must use the same devicesPerNode")
			}
		}
	} else {
		family, err = classifyManifestPartitions(gbuildDeploymentV1CapnpFile, partitions, artifacts.HasPartSelect())
		if err != nil {
			return err
		}
	}

	// Publish the scheduler-facing artifact projection.
	build.partitions = partitions
	build.family = family

	return nil
}

func buildPartitionFromDeploymentV1(raw deploymentcapnpv1.PartitionDeployment, units bool) (buildPartition, error) {
	if !raw.HasPartition() {
		return buildPartition{}, fmt.Errorf("%s artifact partition is missing partition ref", gbuildDeploymentV1CapnpFile)
	}
	ref, err := raw.Partition()
	if err != nil {
		return buildPartition{}, fmt.Errorf("reading %s artifact partition ref: %w", gbuildDeploymentV1CapnpFile, err)
	}
	switch ref.DeviceType() {
	case commoncapnpv1.DeviceType_cuda, commoncapnpv1.DeviceType_cpu:
		return buildPartition{}, nil
	case commoncapnpv1.DeviceType_lpu:
		if raw.Detail().Which() != deploymentcapnpv1.PartitionDeployment_detail_Which_lpu || !raw.Detail().HasLpu() {
			return buildPartition{}, fmt.Errorf("LPU partition %d has mismatched detail", ref.PartitionId())
		}
	default:
		return buildPartition{}, fmt.Errorf("partition %d has unsupported deviceType %d", ref.PartitionId(), ref.DeviceType())
	}
	detail, err := raw.Detail().Lpu()
	if err != nil {
		return buildPartition{}, fmt.Errorf("reading %s LPU partition %d detail: %w", gbuildDeploymentV1CapnpFile, ref.PartitionId(), err)
	}
	family := BuildFamilyXT
	switch detail.Architecture() {
	case deploymentcapnpv1.LpuArchitecture_lp20:
	case deploymentcapnpv1.LpuArchitecture_lp30:
		family = BuildFamilyHX
	default:
		return buildPartition{}, fmt.Errorf("%s LPU partition %d unsupported chip architecture %d", gbuildDeploymentV1CapnpFile, ref.PartitionId(), detail.Architecture())
	}
	if units {
		width, err := positiveManifestUInt32ToInt("LPU partition devicesPerNode", detail.DevicesPerNode())
		if err != nil {
			return buildPartition{}, err
		}
		path, err := detail.Path()
		if err != nil {
			return buildPartition{}, fmt.Errorf("reading LPU partition path: %w", err)
		}
		path, err = cleanManifestRelativeBuildPath("LPU partition path", path)
		if err != nil {
			return buildPartition{}, err
		}
		return buildPartition{sourcePartitionID: int(ref.PartitionId()), partPath: path, devicesPerNode: width, family: family}, nil
	}
	return buildLPUArtifact(gbuildDeploymentV1CapnpFile, ref.PartitionId(), detail, family)
}
