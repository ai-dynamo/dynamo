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

	// Validate compiler-owned placement against the normalized LPU partitions.
	build.lpuUnits, err = lpuUnitsFromDeploymentV1(deployment, build)
	if err != nil {
		return nil, err
	}

	// A partial LP30 selection deploys only through units, which list exactly the
	// packaged partitions; the legacy HX projection requires the whole build.
	if build.family == BuildFamilyHX && manifest.HasPartSelect() && build.lpuUnits == nil {
		return nil, fmt.Errorf("V3 %s partSelect builds are not supported", gbuildDeploymentV1CapnpFile)
	}
	return build, nil
}

// lpuUnitsFromDeploymentV1 decodes and validates the compiler-owned placement
// units of a build whose LPU artifacts are already normalized. An
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

	// Preserve encoded order for Agent, reservation and partition placement.
	// Only a bundle without CUDA partitions may pack stages onto shared hosts.
	lpuOnly := deployment.CompilationMode() == deploymentcapnpv1.CompilationMode_lpuOnly
	switch deployment.CompilationMode() {
	case deploymentcapnpv1.CompilationMode_lpuOnly, deploymentcapnpv1.CompilationMode_lpx:
	default:
		return nil, fmt.Errorf("%s config.compilationMode is unsupported: %d", gbuildDeploymentV1CapnpFile, deployment.CompilationMode())
	}
	packing := !build.hasCUDAPartitions
	placed := make(map[int]bool, len(build.partitions))
	units := make([]lpuUnit, 0, rawUnits.Len())
	for index := range rawUnits.Len() {
		subject := fmt.Sprintf("%s config.lpuUnits[%d]", gbuildDeploymentV1CapnpFile, index)
		unit, err := lpuUnitFromDeploymentV1(rawUnits.At(index), subject, build, partitions, placed, packing)
		if err != nil {
			return nil, err
		}

		units = append(units, unit)
	}

	// Every deployed LPU partition runs in exactly one unit, except the standalone
	// token-embedding partition 0, which an LPU-only build runs on the CPU instead.
	cpuTokenEmbeddings := lpuOnly && build.supportsCPUEmbeddings && build.standaloneTokenEmbeddings
	for _, partition := range build.partitions {
		id := partition.sourcePartitionID
		if !placed[id] && (id != 0 || !cpuTokenEmbeddings) {
			return nil, fmt.Errorf("%s config.lpuUnits omits LPU partition %d", gbuildDeploymentV1CapnpFile, id)
		}
	}

	return units, nil
}

// lpuUnitFromDeploymentV1 decodes one unit of the normalized build and records
// its partitions in placed. partitions indexes the build's LPU partitions by ID.
// packing reports whether the bundle has no CUDA partitions. The compiler
// sizes reservations; decoding checks exact host capacity and unique coverage.
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

	// Place each reservation's encoded partitions exactly once.
	family, devicesPerNode := build.family, build.partitions[0].devicesPerNode
	unit := lpuUnit{
		Reservations: make([]lpuReservation, 0, rawReservations.Len()),
		Connectors:   make([]lpxv1alpha1.PropSyncConnectorRequirement, 0, rawConnectors.Len()),
	}
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
			placed[id] = true
			chips += partition.numChips
		}

		// Packed stages share the sole reservation of a unit without CUDA.
		ids := formatPropSyncChain(reservation.PartitionIDs)
		if len(reservation.PartitionIDs) > 1 && !packing {
			return lpuUnit{}, fmt.Errorf("%s packs partitions %s onto shared hosts, which requires a bundle without CUDA partitions",
				reservationSubject, ids)
		}
		if len(reservation.PartitionIDs) > 1 && rawReservations.Len() > 1 {
			return lpuUnit{}, fmt.Errorf("%s packs partitions %s beside other reservations; a packed chain is its unit's only reservation",
				reservationSubject, ids)
		}

		// Reservation capacity is exact, including the final partially filled host.
		wantHosts := (chips + devicesPerNode - 1) / devicesPerNode
		if reservation.Hosts != wantHosts {
			return lpuUnit{}, fmt.Errorf("%s reserves %d hosts for %d chips; want %d", reservationSubject, reservation.Hosts, chips, wantHosts)
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
// The decoder preserves the size form. The projection applies the operator's
// supported target profiles separately from the compiler contract.
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

		if trays == 0 || hemispheres == 0 || racks == 0 {
			return lpuReservation{}, fmt.Errorf("%s.size.hx dimensions must be positive", subject)
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
		offset := int64(0)
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

	// Lane preferences are operator policy, not part of the placement schema.
	lanes := []int64{4, 2, 1}
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
	// buildFromGbuildDeploymentV1 checks partSelect once it knows whether units place the build.
	family, err := classifyManifestPartitions(gbuildDeploymentV1CapnpFile, partitions, false)
	if err != nil {
		return err
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
	if units {
		family := BuildFamilyXT
		switch detail.Architecture() {
		case deploymentcapnpv1.LpuArchitecture_lp20:
		case deploymentcapnpv1.LpuArchitecture_lp30:
			family = BuildFamilyHX
		default:
			return buildPartition{}, fmt.Errorf("LPU partition %d has unsupported architecture %d", ref.PartitionId(), detail.Architecture())
		}
		chips, err := positiveManifestUInt32ToInt("LPU partition numChips", detail.NumChips())
		if err != nil {
			return buildPartition{}, err
		}
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
		return buildPartition{sourcePartitionID: int(ref.PartitionId()), partPath: path, numChips: chips, devicesPerNode: width, family: family}, nil
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
