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

// buildFromGbuildDeploymentV1 validates compiler input using the acquired canonical build reference.
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
	chains, err := selectedPropSyncChainsFromDeploymentV1(deployment)
	if err != nil {
		return nil, err
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
	if err := addLPUArtifactsFromDeploymentV1(manifest, build); err != nil {
		return nil, err
	}

	// Validate compiler-owned placement against the normalized LPU partitions.
	build.lpuUnits, err = lpuUnitsFromDeploymentV1(deployment, build)
	if err != nil {
		return nil, err
	}
	return build, nil
}

// lpuUnitsFromDeploymentV1 decodes and validates the compiler-owned placement
// units of a build whose LPU partitions and chains are already normalized. An
// absent or empty list returns nil: its writer predates units. build is not mutated.
func lpuUnitsFromDeploymentV1(deployment deploymentcapnpv1.DeploymentInfo, build *Build) ([]lpuUnit, error) {
	if !deployment.HasLpuUnits() {
		return nil, nil
	}
	rawUnits, err := deployment.LpuUnits()
	if err != nil {
		return nil, fmt.Errorf("reading %s config.lpuUnits: %w", gbuildDeploymentV1CapnpFile, err)
	}
	if rawUnits.Len() == 0 {
		return nil, nil
	}

	// Index the deployed LPU partitions that reservations may name.
	partitions := make(map[int]buildPartition, len(build.partitions))
	for _, partition := range build.partitions {
		partitions[partition.sourcePartitionID] = partition
	}

	// Decode units in Agent order, which ascends by root partition.
	placed := make(map[int]bool, len(build.partitions))
	units := make([]lpuUnit, 0, rawUnits.Len())
	for index := range rawUnits.Len() {
		subject := fmt.Sprintf("%s config.lpuUnits[%d]", gbuildDeploymentV1CapnpFile, index)
		unit, err := lpuUnitFromDeploymentV1(rawUnits.At(index), subject, build, partitions, placed)
		if err != nil {
			return nil, err
		}
		if index > 0 && unit.root() <= units[index-1].root() {
			return nil, fmt.Errorf("%s root partition %d does not follow root partition %d; units must ascend by root",
				subject, unit.root(), units[index-1].root())
		}
		units = append(units, unit)
	}

	// Every deployed LPU partition runs in a unit, except the standalone
	// token-embedding partition 0 that an LPU-only build runs on the CPU.
	cpuEmbeddingPartition := deployment.CompilationMode() == deploymentcapnpv1.CompilationMode_lpuOnly &&
		build.supportsCPUEmbeddings && build.standaloneTokenEmbeddings
	for _, partition := range build.partitions {
		id := partition.sourcePartitionID
		if !placed[id] && (id != 0 || !cpuEmbeddingPartition) {
			return nil, fmt.Errorf("%s config.lpuUnits omits LPU partition %d", gbuildDeploymentV1CapnpFile, id)
		}
	}

	// Units restate the selected chains while both fields exist.
	if err := validateLPUUnitChains(units, build.selectedPropSyncChains); err != nil {
		return nil, err
	}
	return units, nil
}

// lpuUnitFromDeploymentV1 decodes one unit of the normalized build and records
// its partitions in placed. partitions indexes the build's LPU partitions by ID.
func lpuUnitFromDeploymentV1(
	raw deploymentcapnpv1.LpuUnit,
	subject string,
	build *Build,
	partitions map[int]buildPartition,
	placed map[int]bool,
) (lpuUnit, error) {
	rawReservations, err := raw.Reservations()
	if err != nil {
		return lpuUnit{}, fmt.Errorf("reading %s.reservations: %w", subject, err)
	}
	if rawReservations.Len() == 0 {
		return lpuUnit{}, fmt.Errorf("%s.reservations must not be empty", subject)
	}
	rawConnectors, err := raw.Connectors()
	if err != nil {
		return lpuUnit{}, fmt.Errorf("reading %s.connectors: %w", subject, err)
	}
	if rawConnectors.Len() != rawReservations.Len()-1 {
		return lpuUnit{}, fmt.Errorf("%s has %d connectors for %d reservations, want %d",
			subject, rawConnectors.Len(), rawReservations.Len(), rawReservations.Len()-1)
	}

	// Place each reservation's consecutive partitions exactly once on enough devices.
	family, devicesPerNode := build.family, build.partitions[0].devicesPerNode
	unit := lpuUnit{
		Reservations: make([]lpuReservation, 0, rawReservations.Len()),
		Connectors:   make([]lpxv1alpha1.PropSyncConnectorRequirement, 0, rawConnectors.Len()),
	}
	previousID := -1
	for index := range rawReservations.Len() {
		reservationSubject := fmt.Sprintf("%s.reservations[%d]", subject, index)
		reservation, err := lpuReservationFromDeploymentV1(rawReservations.At(index), reservationSubject, family, devicesPerNode)
		if err != nil {
			return lpuUnit{}, err
		}
		chips := 0
		for _, id := range reservation.PartitionIDs {
			partition, deployed := partitions[id]
			if !deployed {
				return lpuUnit{}, fmt.Errorf("%s.partitionIds references partition %d, which is not a deployed LPU partition", reservationSubject, id)
			}
			if placed[id] {
				return lpuUnit{}, fmt.Errorf("%s.partitionIds repeats partition %d", reservationSubject, id)
			}
			if previousID >= 0 && id != previousID+1 {
				return lpuUnit{}, fmt.Errorf("%s.partitionIds lists partition %d after %d; a unit holds consecutive ascending partition IDs",
					reservationSubject, id, previousID)
			}
			placed[id] = true
			previousID = id
			chips += partition.numChips
		}
		if devices := reservation.Hosts * devicesPerNode; chips > devices {
			return lpuUnit{}, fmt.Errorf("%s reserves %d devices for %d chips", reservationSubject, devices, chips)
		}
		unit.Reservations = append(unit.Reservations, reservation)
	}

	// Each connector joins adjacent reservations with the family's requirement.
	for index := range rawConnectors.Len() {
		requirement, err := lpuConnectorFromDeploymentV1(
			rawConnectors.At(index),
			fmt.Sprintf("%s.connectors[%d]", subject, index),
			family,
			int64(unit.Reservations[index].Hosts*devicesPerNode),
			int64(unit.Reservations[index+1].Hosts*devicesPerNode),
		)
		if err != nil {
			return lpuUnit{}, err
		}
		unit.Connectors = append(unit.Connectors, requirement)
	}
	return unit, nil
}

// lpuReservationFromDeploymentV1 decodes one reservation's partitions and size.
// The partition IDs are validated by the caller.
func lpuReservationFromDeploymentV1(
	raw deploymentcapnpv1.LpuReservation,
	subject string,
	family BuildFamily,
	devicesPerNode int,
) (lpuReservation, error) {
	rawIDs, err := raw.PartitionIds()
	if err != nil {
		return lpuReservation{}, fmt.Errorf("reading %s.partitionIds: %w", subject, err)
	}
	if rawIDs.Len() == 0 {
		return lpuReservation{}, fmt.Errorf("%s.partitionIds must not be empty", subject)
	}
	reservation := lpuReservation{PartitionIDs: make([]int, rawIDs.Len())}
	for index := range rawIDs.Len() {
		reservation.PartitionIDs[index] = int(rawIDs.At(index))
	}

	// LP20 sizes reservations in nodes; LP30 may also use an HX extent.
	size := raw.Size()
	switch size.Which() {
	case deploymentcapnpv1.LpuReservation_size_Which_nodes:
		// Bound hosts to the Agent replica range, which also keeps device counts in int64.
		nodes := size.Nodes()
		if nodes == 0 || nodes > math.MaxInt32 {
			return lpuReservation{}, fmt.Errorf("%s.size.nodes must be in [1, %d], got %d", subject, math.MaxInt32, nodes)
		}
		reservation.Hosts = int(nodes)
	case deploymentcapnpv1.LpuReservation_size_Which_hx:
		if family != BuildFamilyHX {
			return lpuReservation{}, fmt.Errorf("%s.size.hx sizes an LP30 reservation; LP20 reservations use size.nodes", subject)
		}
		extent, err := size.Hx()
		if err != nil {
			return lpuReservation{}, fmt.Errorf("reading %s.size.hx: %w", subject, err)
		}
		trays, hemispheres, racks := int64(extent.TraysPerHemisphere()), int64(extent.Hemispheres()), int64(extent.Racks())
		if !supportedHXExtent(trays, hemispheres, racks) {
			return lpuReservation{}, fmt.Errorf("%s.size.hx {traysPerHemisphere: %d, hemispheres: %d, racks: %d} is not a supported HX extent",
				subject, trays, hemispheres, racks)
		}
		reservation.Hosts = int(trays * hemispheres * racks)
		reservation.HXExtent = []int64{int64(devicesPerNode), trays, hemispheres, racks}
	default:
		return lpuReservation{}, fmt.Errorf("%s.size has unsupported variant %d", subject, size.Which())
	}
	return reservation, nil
}

// lpuConnectorFromDeploymentV1 decodes one connector into the scheduler
// requirement for the build family. Device pairs index the joined reservations'
// sourceDevices and destinationDevices.
func lpuConnectorFromDeploymentV1(
	raw deploymentcapnpv1.LpuConnector,
	subject string,
	family BuildFamily,
	sourceDevices int64,
	destinationDevices int64,
) (lpxv1alpha1.PropSyncConnectorRequirement, error) {
	switch raw.Which() {
	case deploymentcapnpv1.LpuConnector_Which_xt8888Gap:
		if family != BuildFamilyXT {
			return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("%s is xt8888Gap; LP30 units use hxPropSyncV1", subject)
		}
		gap, err := raw.Xt8888Gap()
		if err != nil {
			return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("reading %s.xt8888Gap: %w", subject, err)
		}
		offset := int64(gap.MaxInterPartitionOffset())
		return lpxv1alpha1.PropSyncConnectorRequirement{
			Kind:                    lpxv1alpha1.PropSyncConnectorKindXt8888Gap,
			MaxInterPartitionOffset: &offset,
		}, nil
	case deploymentcapnpv1.LpuConnector_Which_hxPropSyncV1:
		if family != BuildFamilyHX {
			return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("%s is hxPropSyncV1; LP20 units use xt8888Gap", subject)
		}
		return hxPropSyncRequirementFromDeploymentV1(raw, subject+".hxPropSyncV1", sourceDevices, destinationDevices)
	default:
		return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("%s has unsupported variant %d", subject, raw.Which())
	}
}

// hxPropSyncRequirementFromDeploymentV1 decodes the device pairs and lane
// widths of an hxPropSyncV1 connector.
func hxPropSyncRequirementFromDeploymentV1(
	raw deploymentcapnpv1.LpuConnector,
	subject string,
	sourceDevices int64,
	destinationDevices int64,
) (lpxv1alpha1.PropSyncConnectorRequirement, error) {
	connector, err := raw.HxPropSyncV1()
	if err != nil {
		return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("reading %s: %w", subject, err)
	}

	// Each distinct pair names a device on each joined reservation.
	rawConnections, err := connector.Connections()
	if err != nil {
		return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("reading %s.connections: %w", subject, err)
	}
	if rawConnections.Len() == 0 {
		return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("%s.connections must not be empty", subject)
	}
	connections := make([]lpxv1alpha1.HxLogicalConnection, rawConnections.Len())
	seen := make(map[lpxv1alpha1.HxLogicalConnection]bool, rawConnections.Len())
	for index := range rawConnections.Len() {
		pair := rawConnections.At(index)
		connection := lpxv1alpha1.HxLogicalConnection{
			FromLogicalDevice: int64(pair.FromDevice()),
			ToLogicalDevice:   int64(pair.ToDevice()),
		}
		if connection.FromLogicalDevice >= sourceDevices || connection.ToLogicalDevice >= destinationDevices {
			return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf(
				"%s.connections[%d] %d->%d is outside the joined reservations' %d and %d devices",
				subject, index, connection.FromLogicalDevice, connection.ToLogicalDevice, sourceDevices, destinationDevices)
		}
		if seen[connection] {
			return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("%s.connections[%d] repeats device pair %d->%d",
				subject, index, connection.FromLogicalDevice, connection.ToLogicalDevice)
		}
		seen[connection] = true
		connections[index] = connection
	}

	// Lane multiplicities are positive and listed in preference order.
	rawLanes, err := connector.AcceptableLaneMultiplicities()
	if err != nil {
		return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("reading %s.acceptableLaneMultiplicities: %w", subject, err)
	}
	if rawLanes.Len() == 0 {
		return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("%s.acceptableLaneMultiplicities must not be empty", subject)
	}
	lanes := make([]int64, rawLanes.Len())
	for index := range rawLanes.Len() {
		if rawLanes.At(index) == 0 {
			return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf("%s.acceptableLaneMultiplicities[%d] must be positive", subject, index)
		}
		lanes[index] = int64(rawLanes.At(index))
	}
	return lpxv1alpha1.PropSyncConnectorRequirement{
		Kind:                         lpxv1alpha1.PropSyncConnectorKindHxPropSyncV1,
		Connections:                  &connections,
		AcceptableLaneMultiplicities: &lanes,
	}, nil
}

// validateLPUUnitChains requires every selected chain to equal exactly one
// unit's partitions and every multi-partition unit to be a selected chain.
func validateLPUUnitChains(units []lpuUnit, chains [][]int) error {
	unitByRoot := make(map[int]int, len(units))
	for index, unit := range units {
		unitByRoot[unit.root()] = index
	}
	matched := make([]bool, len(units))
	for chainIndex, chain := range chains {
		index, found := unitByRoot[chain[0]]
		if !found || !slices.Equal(units[index].partitionIDs(), chain) {
			return fmt.Errorf("%s config.selectedPropSyncChains[%d] %s does not equal the partitions of one lpuUnits entry",
				gbuildDeploymentV1CapnpFile, chainIndex, formatPropSyncChain(chain))
		}
		if matched[index] {
			return fmt.Errorf("%s config.selectedPropSyncChains[%d] %s repeats an earlier chain",
				gbuildDeploymentV1CapnpFile, chainIndex, formatPropSyncChain(chain))
		}
		matched[index] = true
	}
	for index, unit := range units {
		if ids := unit.partitionIDs(); len(ids) > 1 && !matched[index] {
			return fmt.Errorf("%s config.lpuUnits[%d] groups partitions %s, which is not a selected prop-sync chain",
				gbuildDeploymentV1CapnpFile, index, formatPropSyncChain(ids))
		}
	}
	return nil
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
) error {
	rawPartitions, err := artifacts.Partitions()
	if err != nil {
		return fmt.Errorf("reading %s partitions: %w", gbuildDeploymentV1CapnpFile, err)
	}
	partitions := make([]buildPartition, 0, rawPartitions.Len())
	for index := 0; index < rawPartitions.Len(); index++ {
		partition, err := buildPartitionFromDeploymentV1(rawPartitions.At(index))
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
	partialSelection := artifacts.HasPartSelect()
	family, err := classifyManifestPartitions(gbuildDeploymentV1CapnpFile, partitions, partialSelection)
	if err != nil {
		return err
	}

	// Publish the scheduler-facing artifact projection.
	build.partitions = partitions
	build.family = family

	return nil
}

func buildPartitionFromDeploymentV1(raw deploymentcapnpv1.PartitionDeployment) (buildPartition, error) {
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
	switch detail.Architecture() {
	case deploymentcapnpv1.LpuArchitecture_lp20:
		return buildLPUArtifact(gbuildDeploymentV1CapnpFile, ref.PartitionId(), detail, BuildFamilyXT)
	case deploymentcapnpv1.LpuArchitecture_lp30:
		return buildLPUArtifact(gbuildDeploymentV1CapnpFile, ref.PartitionId(), detail, BuildFamilyHX)
	default:
		return buildPartition{}, fmt.Errorf("%s LPU partition %d unsupported chip architecture %d", gbuildDeploymentV1CapnpFile, ref.PartitionId(), detail.Architecture())
	}
}
