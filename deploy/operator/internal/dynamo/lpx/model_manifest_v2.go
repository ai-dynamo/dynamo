/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strings"

	"capnproto.org/go/capnp/v3"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
)

const (
	gbuildManifestV2CapnpFile             = "manifest.v2.capnp.bin"
	runtimeIOProtocolHost          uint16 = 0
	runtimeIOProtocolMultiEndpoint uint16 = 1
	runtimeIOMaxMode               uint16 = 2
	hxTopologyFamily                      = "16x8x2x3"
)

func decodeGbuildManifestV2(data []byte) (manifestcapnpv2.Manifest, error) {
	msg, err := capnp.Unmarshal(data)
	if err != nil {
		return manifestcapnpv2.Manifest{}, fmt.Errorf("parsing %s: %w", gbuildManifestV2CapnpFile, err)
	}
	manifest, err := manifestcapnpv2.ReadRootManifest(msg)
	if err != nil {
		return manifestcapnpv2.Manifest{}, fmt.Errorf("reading %s root: %w", gbuildManifestV2CapnpFile, err)
	}
	return manifest, nil
}

// buildFromGbuildManifestV2 validates compiler input using the acquired canonical build reference.
func buildFromGbuildManifestV2(buildRef string, manifest manifestcapnpv2.Manifest) (*Build, error) {
	if manifest.ContractRevision() != manifestcapnpv2.CurrentContractRevision {
		return nil, fmt.Errorf(
			"%s contractRevision = %d, want %d",
			gbuildManifestV2CapnpFile,
			manifest.ContractRevision(),
			manifestcapnpv2.CurrentContractRevision,
		)
	}
	if !manifest.HasDeployment() {
		return nil, fmt.Errorf("%s is missing deployment", gbuildManifestV2CapnpFile)
	}
	deployment, err := manifest.Deployment()
	if err != nil {
		return nil, fmt.Errorf("reading %s deployment: %w", gbuildManifestV2CapnpFile, err)
	}
	var compilationMode compilationMode
	switch deployment.CompilationMode() {
	case manifestcapnpv2.CompilationMode_lpuOnly:
		compilationMode = compilationModeLPUOnly
	case manifestcapnpv2.CompilationMode_lpx:
		compilationMode = compilationModeHybrid
	default:
		return nil, fmt.Errorf("%s deployment.compilationMode %q is not supported", gbuildManifestV2CapnpFile, deployment.CompilationMode().String())
	}
	if !deployment.HasProgram() {
		return nil, fmt.Errorf("%s deployment.program is missing", gbuildManifestV2CapnpFile)
	}
	program, err := deployment.Program()
	if err != nil {
		return nil, fmt.Errorf("reading %s deployment.program: %w", gbuildManifestV2CapnpFile, err)
	}
	batchSize, err := positiveManifestUInt32ToInt(
		fmt.Sprintf("%s deployment.program.batchSize", gbuildManifestV2CapnpFile),
		program.BatchSize(),
	)
	if err != nil {
		return nil, err
	}
	chains, err := selectedPropSyncChainsFromManifestV2(deployment)
	if err != nil {
		return nil, err
	}
	ioFPGACount, ioFanoutFactor, err := runtimeIOFromManifestV2(deployment)
	if err != nil {
		return nil, err
	}
	// Reject an incomplete split-I/O batch before any runtime path consumes it.
	if batchSize%int(ioFPGACount) != 0 {
		return nil, fmt.Errorf(
			"%s deployment.program.batchSize %d must be divisible by deployment.runtimeIo.ioFpgaCount %d",
			gbuildManifestV2CapnpFile,
			batchSize,
			ioFPGACount,
		)
	}

	// Reject incomplete client-owned transaction regions after endpoint splitting.
	perEndpointBatchSize := batchSize / int(ioFPGACount)
	if perEndpointBatchSize%int(ioFanoutFactor) != 0 {
		return nil, fmt.Errorf(
			"%s deployment.program.batchSize per endpoint %d must be divisible by deployment.runtimeIo.fanoutFactor %d",
			gbuildManifestV2CapnpFile,
			perEndpointBatchSize,
			ioFanoutFactor,
		)
	}
	if !manifest.HasArtifacts() {
		return nil, fmt.Errorf("%s is missing artifacts", gbuildManifestV2CapnpFile)
	}
	artifacts, err := manifest.Artifacts()
	if err != nil {
		return nil, fmt.Errorf("reading %s artifacts: %w", gbuildManifestV2CapnpFile, err)
	}
	runtimeTokenEmbeddingsPath, err := runtimeTokenEmbeddingsPathFromManifestV2(artifacts)
	if err != nil {
		return nil, err
	}
	// Validate runtime embedding assets before projecting scheduler-facing artifacts.
	if runtimeTokenEmbeddingsPath != "" && !program.SupportsCpuEmbeddings() {
		return nil, fmt.Errorf("%s artifacts.runtimeAssets.tokenEmbeddingsPath requires supportsCpuEmbeddings=true", gbuildManifestV2CapnpFile)
	}
	if program.SupportsCpuEmbeddings() && program.StandaloneTokenEmbeddings() && runtimeTokenEmbeddingsPath == "" {
		return nil, fmt.Errorf("%s artifacts.runtimeAssets.tokenEmbeddingsPath is required when standaloneTokenEmbeddings=true", gbuildManifestV2CapnpFile)
	}

	build := &Build{
		path:                      buildRef,
		compilationMode:           compilationMode,
		selectedPropSyncChains:    chains,
		standaloneTokenEmbeddings: program.StandaloneTokenEmbeddings(),
		supportsCPUEmbeddings:     program.SupportsCpuEmbeddings(),
		ioFPGACount:               ioFPGACount,
		ioFanoutFactor:            ioFanoutFactor,
	}

	// Complete the normalized build with scheduler-facing LPU artifacts.
	if err := addLPUArtifactsFromManifestV2(artifacts, deployment, build); err != nil {
		return nil, err
	}
	return build, nil
}

func runtimeIOFromManifestV2(deployment manifestcapnpv2.DeploymentInfo) (int32, int32, error) {
	if !deployment.HasRuntimeIo() {
		return 0, 0, fmt.Errorf("%s deployment.runtimeIo is missing", gbuildManifestV2CapnpFile)
	}
	runtimeIO, err := deployment.RuntimeIo()
	if err != nil {
		return 0, 0, fmt.Errorf("reading %s deployment.runtimeIo: %w", gbuildManifestV2CapnpFile, err)
	}
	count := runtimeIO.IoFpgaCount()
	if count == 0 || count > math.MaxInt32 {
		return 0, 0, fmt.Errorf("%s deployment.runtimeIo.ioFpgaCount must be in [1, %d], got %d", gbuildManifestV2CapnpFile, math.MaxInt32, count)
	}

	// Fanout is a separate positive client count for each physical endpoint.
	fanoutFactor := runtimeIO.FanoutFactor()
	if fanoutFactor == 0 || fanoutFactor > math.MaxInt32 {
		return 0, 0, fmt.Errorf("%s deployment.runtimeIo.fanoutFactor must be in [1, %d], got %d", gbuildManifestV2CapnpFile, math.MaxInt32, fanoutFactor)
	}
	switch runtimeIO.Protocol() {
	case runtimeIOProtocolHost:
		if count != 1 {
			return 0, 0, fmt.Errorf("%s host runtime I/O requires ioFpgaCount 1, got %d", gbuildManifestV2CapnpFile, count)
		}
	case runtimeIOProtocolMultiEndpoint:
	default:
		return 0, 0, fmt.Errorf("%s deployment.runtimeIo.protocol %d is not supported", gbuildManifestV2CapnpFile, runtimeIO.Protocol())
	}
	if mode := runtimeIO.Reserved1(); mode > runtimeIOMaxMode {
		return 0, 0, fmt.Errorf("%s deployment.runtimeIo mode %d is not supported", gbuildManifestV2CapnpFile, mode)
	}
	return int32(count), int32(fanoutFactor), nil
}

func selectedPropSyncChainsFromManifestV2(deployment manifestcapnpv2.DeploymentInfo) ([][]int, error) {
	if !deployment.HasSelectedPropSyncChains() {
		return nil, nil
	}
	rawChains, err := deployment.SelectedPropSyncChains()
	if err != nil {
		return nil, fmt.Errorf("reading %s deployment.selectedPropSyncChains: %w", gbuildManifestV2CapnpFile, err)
	}
	chains := make([][]int, 0, rawChains.Len())
	for chainIndex := 0; chainIndex < rawChains.Len(); chainIndex++ {
		rawChain := rawChains.At(chainIndex)
		if !rawChain.HasPartitionIds() {
			return nil, fmt.Errorf("%s deployment.selectedPropSyncChains[%d].partitionIds is missing", gbuildManifestV2CapnpFile, chainIndex)
		}
		rawIDs, err := rawChain.PartitionIds()
		if err != nil {
			return nil, fmt.Errorf("reading %s deployment.selectedPropSyncChains[%d].partitionIds: %w", gbuildManifestV2CapnpFile, chainIndex, err)
		}
		if rawIDs.Len() < 2 {
			return nil, fmt.Errorf("%s deployment.selectedPropSyncChains[%d] must contain at least two partitionIds", gbuildManifestV2CapnpFile, chainIndex)
		}
		chain := make([]int, rawIDs.Len())
		for index := range chain {
			chain[index] = int(rawIDs.At(index))
		}
		chains = append(chains, chain)
	}
	return chains, nil
}

func addLPUArtifactsFromManifestV2(
	artifacts manifestcapnpv2.ArtifactInfo,
	deployment manifestcapnpv2.DeploymentInfo,
	build *Build,
) error {
	rawPartitions, err := artifacts.Partitions()
	if err != nil {
		return fmt.Errorf("reading %s artifacts.partitions: %w", gbuildManifestV2CapnpFile, err)
	}
	partitions := make([]buildPartition, 0, rawPartitions.Len())
	hxDoubleNodeCount := true
	for index := 0; index < rawPartitions.Len(); index++ {
		partition, compatible, err := buildPartitionFromManifestV2(rawPartitions.At(index))
		if err != nil {
			return err
		}
		// Nonempty paths mark LPU artifacts; only metadata-less HX partitions permit the historical doubled node count.
		if partition.partPath != "" {
			partitions = append(partitions, partition)
			hxDoubleNodeCount = hxDoubleNodeCount && compatible
		}
	}
	// An LPU-only build cannot silently discard packaged CUDA or CPU partitions.
	if build.compilationMode == compilationModeLPUOnly && len(partitions) != rawPartitions.Len() {
		return fmt.Errorf(
			"%s deployment.compilationMode lpuOnly requires every artifact partition to use deviceType lpu",
			gbuildManifestV2CapnpFile,
		)
	}
	if len(partitions) == 0 {
		return fmt.Errorf("%s artifacts contain no LPU partitions", gbuildManifestV2CapnpFile)
	}
	partialSelection := artifacts.HasPartSelect()
	family, packagedNodes, partitionZeroNodes, err := classifyManifestPartitions(partitions, partialSelection)
	if err == nil && family == BuildFamilyXT {
		err = validateManifestV2PartSelect(artifacts, partitions)
	}
	if err != nil {
		return err
	}

	// Publish the artifact projection before validating the complete deployment geometry.
	build.partitions = partitions
	build.family = family

	want, err := positiveManifestUInt32ToInt(
		fmt.Sprintf("%s deployment.numLpuNodes", gbuildManifestV2CapnpFile),
		deployment.NumLpuNodes(),
	)
	if err != nil {
		return err
	}
	return validateManifestPartitionNodeCount(want, build, partialSelection, hxDoubleNodeCount, packagedNodes, partitionZeroNodes)
}

func buildPartitionFromManifestV2(raw manifestcapnpv2.PartitionInfo) (buildPartition, bool, error) {
	if !raw.HasPartition() {
		return buildPartition{}, false, fmt.Errorf("%s artifact partition is missing partition ref", gbuildManifestV2CapnpFile)
	}
	ref, err := raw.Partition()
	if err != nil {
		return buildPartition{}, false, fmt.Errorf("reading %s artifact partition ref: %w", gbuildManifestV2CapnpFile, err)
	}
	if ref.DeviceType() != manifestcapnpv2.DeviceType_lpu {
		return buildPartition{}, false, nil
	}
	if raw.Detail().Which() != manifestcapnpv2.PartitionInfo_detail_Which_lpu || !raw.Detail().HasLpu() {
		return buildPartition{}, false, fmt.Errorf("%s artifact partition %d has deviceType lpu without LPU detail", gbuildManifestV2CapnpFile, ref.PartitionId())
	}
	detail, err := raw.Detail().Lpu()
	if err != nil {
		return buildPartition{}, false, fmt.Errorf("reading %s LPU partition %d detail: %w", gbuildManifestV2CapnpFile, ref.PartitionId(), err)
	}
	path, err := detail.Path()
	if err != nil {
		return buildPartition{}, false, fmt.Errorf("reading %s LPU partition %d path: %w", gbuildManifestV2CapnpFile, ref.PartitionId(), err)
	}

	// Runtime topology names belong to the manifest and are interpreted by the runtime.
	subject := fmt.Sprintf("%s LPU partition %d", gbuildManifestV2CapnpFile, ref.PartitionId())

	// Metadata selects HX; only the 16-chip, 16-device HX shape can omit it.
	var partition buildPartition
	var compatible bool
	if detail.HasTopologyMetadata() || (detail.NumChips() == 16 && detail.DevicesPerNode() == 16) {
		partition, compatible, err = buildHXPartition(subject, detail)
	} else {
		partition, err = buildXTPartition(subject, detail)
	}
	if err != nil {
		return buildPartition{}, false, err
	}

	// Validate shared artifact fields once, after the selected geometry is accepted.
	partition.partPath, err = cleanManifestRelativeBuildPath(subject+" path", path)
	if err != nil {
		return buildPartition{}, false, err
	}
	partition.sourcePartitionID = int(ref.PartitionId())
	return partition, compatible, nil
}

func validateManifestV2PartSelect(artifacts manifestcapnpv2.ArtifactInfo, partitions []buildPartition) error {
	if !artifacts.HasPartSelect() {
		return nil
	}
	partSelect, err := artifacts.PartSelect()
	if err != nil {
		return fmt.Errorf("reading %s artifacts.partSelect: %w", gbuildManifestV2CapnpFile, err)
	}
	selected, err := partSelect.Partitions()
	if err != nil {
		return fmt.Errorf("reading %s artifacts.partSelect.partitions: %w", gbuildManifestV2CapnpFile, err)
	}
	if selected.Len() == 0 {
		return fmt.Errorf("%s artifacts.partSelect.partitions is empty", gbuildManifestV2CapnpFile)
	}
	selectedLPU := make(map[int]struct{}, len(partitions))
	for index := 0; index < selected.Len(); index++ {
		ref := selected.At(index)
		if ref.DeviceType() != manifestcapnpv2.DeviceType_lpu {
			continue
		}
		id := int(ref.PartitionId())
		if _, duplicate := selectedLPU[id]; duplicate {
			return fmt.Errorf("%s artifacts.partSelect has duplicate LPU partition id %d", gbuildManifestV2CapnpFile, id)
		}
		partitionIndex := sort.Search(len(partitions), func(i int) bool {
			return partitions[i].sourcePartitionID >= id
		})
		if partitionIndex == len(partitions) || partitions[partitionIndex].sourcePartitionID != id {
			return fmt.Errorf("%s artifacts.partSelect references unpackaged LPU partition id %d", gbuildManifestV2CapnpFile, id)
		}
		selectedLPU[id] = struct{}{}
	}
	if len(selectedLPU) != len(partitions) {
		return fmt.Errorf("%s artifacts.partSelect selects %d LPU partitions, but artifacts package %d", gbuildManifestV2CapnpFile, len(selectedLPU), len(partitions))
	}
	return nil
}

func runtimeTokenEmbeddingsPathFromManifestV2(artifacts manifestcapnpv2.ArtifactInfo) (string, error) {
	if !artifacts.HasRuntimeAssets() {
		return "", nil
	}
	runtimeAssets, err := artifacts.RuntimeAssets()
	if err != nil {
		return "", fmt.Errorf("reading %s artifacts.runtimeAssets: %w", gbuildManifestV2CapnpFile, err)
	}
	if !runtimeAssets.HasTokenEmbeddingsPath() {
		return "", nil
	}
	rawPath, err := runtimeAssets.TokenEmbeddingsPath()
	if err != nil {
		return "", fmt.Errorf("reading %s artifacts.runtimeAssets.tokenEmbeddingsPath: %w", gbuildManifestV2CapnpFile, err)
	}
	return cleanManifestRelativeBuildPath(
		fmt.Sprintf("%s artifacts.runtimeAssets.tokenEmbeddingsPath", gbuildManifestV2CapnpFile),
		rawPath,
	)
}

func classifyManifestPartitions(partitions []buildPartition, partSelect bool) (BuildFamily, int, int, error) {
	family := BuildFamilyXT
	packagedNodes, partitionZeroNodes := 0, 0
	seen := make(map[int]struct{}, len(partitions))
	for _, partition := range partitions {
		partitionFamily := BuildFamilyXT
		if len(partition.hxExtent) != 0 {
			partitionFamily = BuildFamilyHX
		}
		if len(seen) != 0 && family != partitionFamily {
			return "", 0, 0, fmt.Errorf("%s mixes XT and HX LPU partitions", gbuildManifestV2CapnpFile)
		}
		family = partitionFamily
		if _, duplicate := seen[partition.sourcePartitionID]; duplicate {
			if family == BuildFamilyHX {
				return "", 0, 0, fmt.Errorf("V3 %s repeats LPU partition ID %d", gbuildManifestV2CapnpFile, partition.sourcePartitionID)
			}
			return "", 0, 0, fmt.Errorf("%s has duplicate LPU partition id %d", gbuildManifestV2CapnpFile, partition.sourcePartitionID)
		}
		seen[partition.sourcePartitionID] = struct{}{}

		// Retain both deployment geometries using each partition's manifest device count.
		nodes := partition.effectiveNodeCount()
		packagedNodes += nodes
		if partition.sourcePartitionID == 0 {
			partitionZeroNodes += nodes
		}
	}
	if family == BuildFamilyHX {
		if partSelect {
			return "", 0, 0, fmt.Errorf("V3 %s partSelect builds are not supported", gbuildManifestV2CapnpFile)
		}
		return family, packagedNodes, partitionZeroNodes, nil
	}
	sort.Slice(partitions, func(i, j int) bool { return partitions[i].sourcePartitionID < partitions[j].sourcePartitionID })
	return family, packagedNodes, partitionZeroNodes, nil
}

func buildXTPartition(subject string, raw manifestcapnpv2.LpuPartitionArtifact) (buildPartition, error) {
	// Read partition geometry directly from the manifest's numeric fields.
	numChips, err := positiveManifestUInt32ToInt(subject+" numChips", raw.NumChips())
	if err != nil {
		return buildPartition{}, err
	}
	devicesPerNode, err := positiveManifestUInt32ToInt(subject+" devicesPerNode", raw.DevicesPerNode())
	if err != nil {
		return buildPartition{}, err
	}

	// Multi-node partitions must fill whole nodes at the declared device density.
	if numChips > devicesPerNode && numChips%devicesPerNode != 0 {
		return buildPartition{}, fmt.Errorf("%s has %d chips, not divisible by %d LPU devices per node", subject, numChips, devicesPerNode)
	}
	return buildPartition{numChips: numChips, devicesPerNode: devicesPerNode}, nil
}

func buildHXPartition(subject string, raw manifestcapnpv2.LpuPartitionArtifact) (buildPartition, bool, error) {
	// Retain device density alongside the HX extent validated below.
	partition := buildPartition{
		numChips:       int(raw.NumChips()),
		devicesPerNode: int(raw.DevicesPerNode()),
	}
	if !raw.HasTopologyMetadata() {
		// The caller selects metadata-less HX only for the 16-chip, 16-device case.
		partition.hxExtent = []int64{16, 1, 1, 1}
		return partition, true, nil
	}

	// Decode HX metadata directly and report read errors at their source.
	metadata, err := raw.TopologyMetadata()
	if err != nil {
		return buildPartition{}, false, fmt.Errorf("reading V3 %s topologyMetadata: %w", subject, err)
	}
	family, err := metadata.TopologyFamily()
	if err != nil {
		return buildPartition{}, false, fmt.Errorf("reading V3 %s topologyMetadata.topologyFamily: %w", subject, err)
	}
	if strings.TrimSpace(family) != hxTopologyFamily {
		return buildPartition{}, false, fmt.Errorf("V3 %s topologyMetadata.topologyFamily = %q, want %q", subject, strings.TrimSpace(family), hxTopologyFamily)
	}
	shape, err := metadata.PartitionShape()
	if err != nil {
		return buildPartition{}, false, fmt.Errorf("reading V3 %s topologyMetadata.partitionShape: %w", subject, err)
	}
	extent := make([]int64, shape.Len())
	for index := range extent {
		extent[index] = int64(shape.At(index))
	}

	// Preserve every supported HX shape and its chip/device consistency checks.
	if len(extent) != 4 || extent[0] != 16 || extent[1] < 1 || extent[1] > 8 ||
		(!((extent[2] == 1 || extent[2] == 2) && extent[3] == 1) && !(extent[1] == 8 && extent[2] == 2 && extent[3] == 2)) {
		return buildPartition{}, false, fmt.Errorf("V3 %s has unsupported HX extent %v", subject, extent)
	}
	count := extent[0] * extent[1] * extent[2] * extent[3]
	if count != int64(raw.NumChips()) {
		return buildPartition{}, false, fmt.Errorf("V3 %s topologyMetadata.partitionShape contains %d chips, want numChips %d", subject, count, raw.NumChips())
	}
	if extent[0] != int64(raw.DevicesPerNode()) {
		return buildPartition{}, false, fmt.Errorf("V3 %s topologyMetadata.partitionShape first dimension %d does not match devicesPerNode %d", subject, extent[0], raw.DevicesPerNode())
	}
	partition.hxExtent = extent
	return partition, false, nil
}

func cleanManifestRelativeBuildPath(field, rawPath string) (string, error) {
	if strings.ContainsAny(rawPath, "\x00\r\n") {
		return "", fmt.Errorf("%s %q must not contain NUL bytes or line breaks", field, rawPath)
	}
	assetPath := strings.TrimSpace(rawPath)
	if assetPath == "" {
		return "", fmt.Errorf("%s is empty", field)
	}
	if filepath.IsAbs(assetPath) {
		return "", fmt.Errorf("%s %q must be relative and stay within build directory", field, rawPath)
	}
	assetPath = filepath.ToSlash(filepath.Clean(assetPath))
	if assetPath == "." || assetPath == ".." || strings.HasPrefix(assetPath, "../") {
		return "", fmt.Errorf("%s %q must be relative and stay within build directory", field, rawPath)
	}
	return assetPath, nil
}

func validateManifestPartitionNodeCount(
	want int,
	build *Build,
	partialSelection bool,
	hxDoubleNodeCount bool,
	packagedNodes, partitionZeroNodes int,
) error {
	// Compare the declaration with the packaged and host-embedding partition inventories.
	hostEmbeddingNodes := packagedNodes
	if build.supportsCPUEmbeddings && build.standaloneTokenEmbeddings {
		hostEmbeddingNodes -= partitionZeroNodes
	}
	if build.family == BuildFamilyHX {
		if want == packagedNodes || (hxDoubleNodeCount && want == 2*packagedNodes) {
			return nil
		}
		return fmt.Errorf("V3 %s deployment.numLpuNodes = %d, but partition extents require %d LPU nodes", gbuildManifestV2CapnpFile, want, packagedNodes)
	}

	// A host-embedding deployment must retain at least one model partition.
	if hostEmbeddingNodes == 0 {
		return fmt.Errorf("%s LPU partitions use 0 LPU nodes, want deployment.numLpuNodes %d", gbuildManifestV2CapnpFile, want)
	}

	// Accept either supported deployment mode without discarding packaged partitions.
	if want == packagedNodes || want == hostEmbeddingNodes {
		return nil
	}

	// partSelect artifacts contain only the selected partitions, while numLpuNodes
	// describes the complete deployment geometry.
	if partialSelection && want >= hostEmbeddingNodes {
		return nil
	}

	// Keep the existing error concise when both deployment modes use the same node count.
	if packagedNodes == hostEmbeddingNodes {
		return fmt.Errorf("%s LPU partitions use %d LPU nodes, want deployment.numLpuNodes %d", gbuildManifestV2CapnpFile, packagedNodes, want)
	}

	return fmt.Errorf(
		"%s LPU partitions use %d packaged LPU nodes or %d with host embeddings, want deployment.numLpuNodes %d",
		gbuildManifestV2CapnpFile,
		packagedNodes,
		hostEmbeddingNodes,
		want,
	)
}

func positiveManifestUInt32ToInt(field string, value uint32) (int, error) {
	if value == 0 {
		return 0, fmt.Errorf("%s must be >= 1, got 0", field)
	}
	return int(value), nil
}
