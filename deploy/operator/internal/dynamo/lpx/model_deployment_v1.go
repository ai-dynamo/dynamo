/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"capnproto.org/go/capnp/v3"
	commoncapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/common/v1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
)

const gbuildDeploymentV1CapnpFile = "deployment.v1.capnp.bin"

// hxDefaultPropSyncTopologyDimension is HX16x8x2 INTRA_HALF_RACK, the dimension
// GBuild compiles a prop-sync boundary at unless its preset names another: trays
// of one hemisphere, the only tray-to-tray links with four lanes.
const hxDefaultPropSyncTopologyDimension = 1

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
	ioFPGACount, ioFanoutFactor, err := runtimeIOFromDeploymentV1(deployment)
	if err != nil {
		return nil, err
	}

	checkpoint, checkpointErr := checkpointFromDeploymentV1(manifest)

	build := &Build{
		path:                      buildRef,
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
	// Units list exactly the packaged partitions, so partSelect builds need no
	// separate handling.
	build.lpuUnits, err = lpuUnitsFromDeploymentV1(deployment, build)
	if err != nil {
		return nil, err
	}
	return build, nil
}

// lpuUnitsFromDeploymentV1 decodes and validates the compiler-owned placement
// units of a build whose LPU partitions are already normalized. The compiler
// owns placement, so a build without units has none the operator can deploy.
// build is not mutated.
func lpuUnitsFromDeploymentV1(deployment deploymentcapnpv1.DeploymentInfo, build *Build) ([]lpuUnit, error) {
	var rawUnits deploymentcapnpv1.LpuUnit_List
	if deployment.HasLpuUnits() {
		var err error
		rawUnits, err = deployment.LpuUnits()
		if err != nil {
			return nil, fmt.Errorf("reading %s config.lpuUnits: %w", gbuildDeploymentV1CapnpFile, err)
		}
	}
	if rawUnits.Len() == 0 {
		return nil, fmt.Errorf("%w: %s records no config.lpuUnits; the operator deploys only builds whose compiler "+
			"records LPU placement units, so recompile the build with a GBuild that writes them",
			ErrUnsupportedRuntime, gbuildDeploymentV1CapnpFile)
	}

	// Index the deployed LPU partitions that reservations may name.
	partitions := make(map[int]buildPartition, len(build.partitions))
	for _, partition := range build.partitions {
		partitions[partition.sourcePartitionID] = partition
	}

	// Decode units in Agent order, which ascends by root partition. Only an
	// LPU-only bundle without CUDA partitions packs chains onto shared hosts.
	lpuOnly := deployment.CompilationMode() == deploymentcapnpv1.CompilationMode_lpuOnly
	packing := lpuOnly && !build.hasCUDAPartitions
	placed := make(map[int]bool, len(build.partitions))
	units := make([]lpuUnit, 0, rawUnits.Len())
	for index := range rawUnits.Len() {
		subject := fmt.Sprintf("%s config.lpuUnits[%d]", gbuildDeploymentV1CapnpFile, index)
		unit, err := lpuUnitFromDeploymentV1(rawUnits.At(index), subject, build, partitions, placed, packing)
		if err != nil {
			return nil, err
		}
		if index > 0 && unit.root() <= units[index-1].root() {
			return nil, fmt.Errorf("%s root partition %d does not follow root partition %d; units must ascend by root",
				subject, unit.root(), units[index-1].root())
		}
		units = append(units, unit)
	}

	// Every deployed LPU partition runs in exactly one unit, except the standalone
	// token-embedding partition 0, which an LPU-only build runs on the CPU instead.
	cpuTokenEmbeddings := lpuOnly && build.supportsCPUEmbeddings && build.standaloneTokenEmbeddings
	for _, partition := range build.partitions {
		id := partition.sourcePartitionID
		switch {
		case id == 0 && cpuTokenEmbeddings && placed[id]:
			return nil, fmt.Errorf("%s config.lpuUnits places LPU partition 0, which an LPU-only build with CPU token embeddings runs on the CPU",
				gbuildDeploymentV1CapnpFile)
		case !placed[id] && (id != 0 || !cpuTokenEmbeddings):
			return nil, fmt.Errorf("%s config.lpuUnits omits LPU partition %d", gbuildDeploymentV1CapnpFile, id)
		}
	}

	return units, nil
}

// lpuUnitFromDeploymentV1 decodes one unit of the normalized build and records
// its partitions in placed. partitions indexes the build's LPU partitions by ID.
// packing reports whether the bundle may pack a chain into one reservation: only
// LPU-only bundles without CUDA partitions do. The compiler sizes reservations;
// decoding checks only that each holds its partitions' chips.
func lpuUnitFromDeploymentV1(
	raw deploymentcapnpv1.LpuUnit,
	subject string,
	build *Build,
	partitions map[int]buildPartition,
	placed map[int]bool,
	packing bool,
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

	// Place each reservation's consecutive partitions exactly once.
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

		// A packed chain shares one reservation in an LPU-only bundle.
		ids := formatPropSyncChain(reservation.PartitionIDs)
		if len(reservation.PartitionIDs) > 1 && !packing {
			return lpuUnit{}, fmt.Errorf("%s packs partitions %s onto shared hosts, which only LPU-only bundles without CUDA partitions do",
				reservationSubject, ids)
		}
		if len(reservation.PartitionIDs) > 1 && rawReservations.Len() > 1 {
			return lpuUnit{}, fmt.Errorf("%s packs partitions %s beside other reservations; a packed chain is its unit's only reservation",
				reservationSubject, ids)
		}

		// The compiler sizes the reservation; it must hold its partitions' chips.
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
// An LP30 reservation sized in nodes is refused: an HX LPU pipeline request
// takes only an extent. The caller validates the partition IDs and the size
// against the partitions.
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

	// LP20 sizes reservations in nodes and LP30 in an HX extent.
	size := raw.Size()
	switch size.Which() {
	case deploymentcapnpv1.LpuReservation_size_Which_nodes:
		// An HX LPU pipeline request has no node-count form, so LP30 nodes fail closed.
		if family == BuildFamilyHX {
			return lpuReservation{}, fmt.Errorf("%w: %s.size.nodes sizes an LP30 reservation, which an HX LPU pipeline request cannot express; LP30 reservations use size.hx",
				ErrUnsupportedRuntime, subject)
		}

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

		// LPX validation accepts only these shapes; the CRD types extent as four
		// small integers and does not. Refusing at decode fails the build before
		// any LPR is created, and caps a reservation at 32 hosts: unchecked uint8
		// dimensions give up to 255^3 hosts each, and 256 such reservations
		// overflow the int32 Agent replica count.
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

	// An LPU pipeline request has no topology dimension and one lane list for every
	// HX connector, so LPX cannot be told to keep a boundary at another dimension.
	// Dimension 0 is a writer that predates the field.
	if dimension := connector.TopologyDimension(); dimension != hxDefaultPropSyncTopologyDimension {
		return lpxv1alpha1.PropSyncConnectorRequirement{}, fmt.Errorf(
			"%w: %s.topologyDimension = %d; an LPU pipeline request cannot carry a topology dimension, so the operator "+
				"deploys only boundaries compiled at the HX16x8x2 default dimension %d (trays of one hemisphere)",
			ErrUnsupportedRuntime, subject, dimension, hxDefaultPropSyncTopologyDimension)
	}
	return lpxv1alpha1.PropSyncConnectorRequirement{
		Kind:                         lpxv1alpha1.PropSyncConnectorKindHxPropSyncV1,
		Connections:                  &connections,
		AcceptableLaneMultiplicities: &lanes,
	}, nil
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

// addLPUArtifactsFromDeploymentV1 normalizes the build's LPU partitions into
// build, in ID order, and records whether it also requires CUDA partitions.
func addLPUArtifactsFromDeploymentV1(
	artifacts deploymentcapnpv1.Deployment,
	build *Build,
) error {
	rawPartitions, err := artifacts.Partitions()
	if err != nil {
		return fmt.Errorf("reading %s partitions: %w", gbuildDeploymentV1CapnpFile, err)
	}
	partitions := make([]buildPartition, 0, rawPartitions.Len())
	seen := make(map[int]bool, rawPartitions.Len())
	for index := 0; index < rawPartitions.Len(); index++ {
		raw := rawPartitions.At(index)
		if !raw.HasPartition() {
			return fmt.Errorf("%s artifact partition is missing partition ref", gbuildDeploymentV1CapnpFile)
		}
		ref, err := raw.Partition()
		if err != nil {
			return fmt.Errorf("reading %s artifact partition ref: %w", gbuildDeploymentV1CapnpFile, err)
		}

		// Project LPU artifacts while retaining whether the build also requires CUDA.
		switch ref.DeviceType() {
		case commoncapnpv1.DeviceType_cuda:
			build.hasCUDAPartitions = true
			continue
		case commoncapnpv1.DeviceType_cpu:
			continue
		case commoncapnpv1.DeviceType_lpu:
		default:
			return fmt.Errorf("partition %d has unsupported deviceType %d", ref.PartitionId(), ref.DeviceType())
		}
		partition, family, err := buildPartitionFromDeploymentV1(raw, ref.PartitionId())
		if err != nil {
			return err
		}

		// One Agent template serves the build, so every LPU partition shares one
		// family and node width.
		if len(partitions) == 0 {
			build.family = family
		} else if family != build.family {
			return fmt.Errorf("%s mixes XT and HX LPU partitions", gbuildDeploymentV1CapnpFile)
		}
		if seen[partition.sourcePartitionID] {
			return fmt.Errorf("%s has duplicate LPU partition id %d", gbuildDeploymentV1CapnpFile, partition.sourcePartitionID)
		}
		seen[partition.sourcePartitionID] = true
		if len(partitions) > 0 && partition.devicesPerNode != partitions[0].devicesPerNode {
			return fmt.Errorf("%s LPU partition %d has devicesPerNode %d, want %d; all LPU partitions must use the same devicesPerNode",
				gbuildDeploymentV1CapnpFile, partition.sourcePartitionID, partition.devicesPerNode, partitions[0].devicesPerNode)
		}
		partitions = append(partitions, partition)
	}
	if len(partitions) == 0 {
		return fmt.Errorf("%s artifacts contain no LPU partitions", gbuildDeploymentV1CapnpFile)
	}
	slices.SortFunc(partitions, func(a, b buildPartition) int { return a.sourcePartitionID - b.sourcePartitionID })
	build.partitions = partitions
	return nil
}

// buildPartitionFromDeploymentV1 decodes the chip geometry of one LPU partition
// and the family its architecture selects. Placement comes from lpuUnits, so the
// partition's topology and compiled extent are not read.
func buildPartitionFromDeploymentV1(raw deploymentcapnpv1.PartitionDeployment, partitionID uint32) (buildPartition, BuildFamily, error) {
	if raw.Detail().Which() != deploymentcapnpv1.PartitionDeployment_detail_Which_lpu || !raw.Detail().HasLpu() {
		return buildPartition{}, "", fmt.Errorf("LPU partition %d has mismatched detail", partitionID)
	}
	detail, err := raw.Detail().Lpu()
	if err != nil {
		return buildPartition{}, "", fmt.Errorf("reading %s LPU partition %d detail: %w", gbuildDeploymentV1CapnpFile, partitionID, err)
	}

	// Each partition selects the hardware family from its explicit architecture.
	var family BuildFamily
	switch detail.Architecture() {
	case deploymentcapnpv1.LpuArchitecture_lp20:
		family = BuildFamilyXT
	case deploymentcapnpv1.LpuArchitecture_lp30:
		family = BuildFamilyHX
	default:
		return buildPartition{}, "", fmt.Errorf("%s LPU partition %d unsupported chip architecture %d",
			gbuildDeploymentV1CapnpFile, partitionID, detail.Architecture())
	}

	subject := fmt.Sprintf("%s LPU partition %d", gbuildDeploymentV1CapnpFile, partitionID)
	numChips, err := positiveManifestUInt32ToInt(subject+" numChips", detail.NumChips())
	if err != nil {
		return buildPartition{}, "", err
	}
	devicesPerNode, err := positiveManifestUInt32ToInt(subject+" devicesPerNode", detail.DevicesPerNode())
	if err != nil {
		return buildPartition{}, "", err
	}
	return buildPartition{sourcePartitionID: int(partitionID), numChips: numChips, devicesPerNode: devicesPerNode}, family, nil
}

func positiveManifestUInt32ToInt(field string, value uint32) (int, error) {
	if value == 0 {
		return 0, fmt.Errorf("%s must be >= 1, got 0", field)
	}
	return int(value), nil
}

// supportedHXExtent reports whether whole-node HX dimensions name a scheduler
// shape: up to eight trays in one or two hemispheres of one rack, or two full racks.
func supportedHXExtent(traysPerHemisphere, hemispheres, racks int64) bool {
	if traysPerHemisphere < 1 || traysPerHemisphere > 8 {
		return false
	}
	return ((hemispheres == 1 || hemispheres == 2) && racks == 1) ||
		(traysPerHemisphere == 8 && hemispheres == 2 && racks == 2)
}

func formatPropSyncChain(chain []int) string {
	ids := make([]string, 0, len(chain))
	for _, partitionID := range chain {
		ids = append(ids, strconv.Itoa(partitionID))
	}
	return "[" + strings.Join(ids, ",") + "]"
}
