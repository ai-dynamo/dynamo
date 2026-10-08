/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"capnproto.org/go/capnp/v3"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
)

const hxTopologyFamily = "16x8x2x3"

func classifyManifestPartitions(filename string, partitions []BuildPartition, partSelect bool) (BuildFamily, int, int, error) {
	family := BuildFamilyXT
	packagedNodes, partitionZeroNodes := 0, 0
	seen := make(map[int]struct{}, len(partitions))
	for _, partition := range partitions {
		partitionFamily := BuildFamilyXT
		if len(partition.HXExtent) != 0 {
			partitionFamily = BuildFamilyHX
		}
		if len(seen) != 0 && family != partitionFamily {
			return "", 0, 0, fmt.Errorf("%s mixes XT and HX LPU partitions", filename)
		}
		family = partitionFamily
		if _, duplicate := seen[partition.SourcePartitionID]; duplicate {
			if family == BuildFamilyHX {
				return "", 0, 0, fmt.Errorf("V3 %s repeats LPU partition ID %d", filename, partition.SourcePartitionID)
			}
			return "", 0, 0, fmt.Errorf("%s has duplicate LPU partition id %d", filename, partition.SourcePartitionID)
		}
		seen[partition.SourcePartitionID] = struct{}{}

		// Retain both deployment geometries during the mandatory artifact traversal.
		nodes := partition.effectiveNodeCount()
		packagedNodes += nodes
		if partition.SourcePartitionID == 0 {
			partitionZeroNodes += nodes
		}
	}
	if family == BuildFamilyHX {
		if partSelect {
			return "", 0, 0, fmt.Errorf("V3 %s partSelect builds are not supported", filename)
		}
		return family, packagedNodes, partitionZeroNodes, nil
	}
	sort.Slice(partitions, func(i, j int) bool { return partitions[i].SourcePartitionID < partitions[j].SourcePartitionID })
	return family, packagedNodes, partitionZeroNodes, nil
}

func buildXTPartition(subject, topology string, raw manifestcapnpv2.LpuPartitionArtifact) (BuildPartition, error) {
	// Derive XT geometry from numeric manifest fields, never the topology name.
	numChips, err := positiveManifestUInt32ToInt(subject+" numChips", raw.NumChips())
	if err != nil {
		return BuildPartition{}, err
	}
	devicesPerNode, err := positiveManifestUInt32ToInt(subject+" devicesPerNode", raw.DevicesPerNode())
	if err != nil {
		return BuildPartition{}, err
	}
	if numChips > devicesPerNode && numChips%devicesPerNode != 0 {
		return BuildPartition{}, fmt.Errorf("%s has %d chips, not divisible by %d LPU devices per node", subject, numChips, devicesPerNode)
	}
	return BuildPartition{Topology: Topology{Raw: topology, ChipCount: numChips}, DevicesPerNode: devicesPerNode}, nil
}

func buildHXPartition(subject, topology string, raw manifestcapnpv2.LpuPartitionArtifact) (BuildPartition, bool, error) {
	devicesPerNode, err := positiveManifestUInt32ToInt(subject+" devicesPerNode", raw.DevicesPerNode())
	if err != nil {
		return BuildPartition{}, false, err
	}
	partition := BuildPartition{Topology: Topology{Raw: topology, ChipCount: int(raw.NumChips())}, DevicesPerNode: devicesPerNode}
	if !raw.HasTopologyMetadata() {
		chips := partition.Topology.ChipCount
		if chips <= 0 || chips > devicesPerNode || devicesPerNode%chips != 0 {
			return BuildPartition{}, false, fmt.Errorf("V3 %s without topologyMetadata requires a one-node stage whose chip count divides %d", subject, devicesPerNode)
		}
		partition.HXExtent = []int64{int64(devicesPerNode), 1, 1, 1}
		return partition, chips == devicesPerNode, nil
	}

	// Decode HX metadata directly and report read errors at their source.
	metadata, err := raw.TopologyMetadata()
	if err != nil {
		return BuildPartition{}, false, fmt.Errorf("reading V3 %s topologyMetadata: %w", subject, err)
	}
	family, err := metadata.TopologyFamily()
	if err != nil {
		return BuildPartition{}, false, fmt.Errorf("reading V3 %s topologyMetadata.topologyFamily: %w", subject, err)
	}
	if strings.TrimSpace(family) != hxTopologyFamily {
		return BuildPartition{}, false, fmt.Errorf("V3 %s topologyMetadata.topologyFamily = %q, want %q", subject, strings.TrimSpace(family), hxTopologyFamily)
	}
	shape, err := metadata.PartitionShape()
	if err != nil {
		return BuildPartition{}, false, fmt.Errorf("reading V3 %s topologyMetadata.partitionShape: %w", subject, err)
	}
	extent := make([]int64, shape.Len())
	for index := range extent {
		extent[index] = int64(shape.At(index))
	}

	// A partial-node shape describes its logical chips; physical HX reservations use full nodes.
	subnode := len(extent) == 4 && extent[0] > 0 && extent[0] < int64(devicesPerNode) &&
		int64(devicesPerNode)%extent[0] == 0 && extent[1] == 1 && extent[2] == 1 && extent[3] == 1
	if !subnode && (len(extent) != 4 || extent[0] != int64(devicesPerNode) || extent[1] < 1 || extent[1] > 8 ||
		(!((extent[2] == 1 || extent[2] == 2) && extent[3] == 1) && !(extent[1] == 8 && extent[2] == 2 && extent[3] == 2))) {
		return BuildPartition{}, false, fmt.Errorf("V3 %s has unsupported HX extent %v", subject, extent)
	}

	// Match the source chip count before rounding its scheduler reservation up to a whole node.
	count := extent[0] * extent[1] * extent[2] * extent[3]
	if count != int64(raw.NumChips()) {
		return BuildPartition{}, false, fmt.Errorf("V3 %s topologyMetadata.partitionShape contains %d chips, want numChips %d", subject, count, raw.NumChips())
	}
	extent[0] = int64(devicesPerNode)
	partition.HXExtent = extent
	return partition, false, nil
}

func buildCompilationMode(filename, mode string) (BuildCompilationMode, error) {
	switch mode {
	case string(BuildCompilationModeLPUOnly):
		return BuildCompilationModeLPUOnly, nil
	case string(BuildCompilationModeHybrid):
		return BuildCompilationModeHybrid, nil
	default:
		return BuildCompilationModeUnknown, fmt.Errorf("%s deployment.compilationMode %q is not supported", filename, mode)
	}
}

// decodePropSyncChain normalizes one selected-chain list from the compiler manifest.
func decodePropSyncChain(rawIDs capnp.UInt32List, chainPath string) ([]int, error) {
	if rawIDs.Len() < 2 {
		return nil, fmt.Errorf("%s must contain at least two partitionIds", chainPath)
	}

	partitionIDs := make([]int, rawIDs.Len())
	for index := range rawIDs.Len() {
		partitionIDs[index] = int(rawIDs.At(index))
	}
	return partitionIDs, nil
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
	filename string,
	want int,
	build *Build,
	partialSelection bool,
	hxDoubleNodeCount bool,
	packagedNodes, partitionZeroNodes int,
) error {
	if build.CompilationMode == BuildCompilationModeLPUOnly && len(build.SelectedPropSyncChains) == 1 {
		configured := *build
		if configured.Family == BuildFamilyXT {
			if err := configured.selectXTRuntimePartitions(); err != nil {
				return err
			}
		}
		packed := packPropSyncPartitions(configured.Partitions)
		if len(packed) < len(configured.Partitions) && want == packed[0].effectiveNodeCount() {
			return nil
		}
	}

	// Compare the declaration with the packaged and host-embedding partition inventories.
	hostEmbeddingNodes := packagedNodes
	if build.SupportsCPUEmbeddings && build.StandaloneTokenEmbeddings {
		hostEmbeddingNodes -= partitionZeroNodes
	}
	if build.Family == BuildFamilyHX {
		if want == packagedNodes || (hxDoubleNodeCount && want == 2*packagedNodes) {
			return nil
		}
		return fmt.Errorf("V3 %s deployment.numLpuNodes = %d, but partition extents require %d LPU nodes", filename, want, packagedNodes)
	}

	// A host-embedding deployment must retain at least one model partition.
	if hostEmbeddingNodes == 0 {
		return fmt.Errorf("%s LPU partitions use 0 LPU nodes, want deployment.numLpuNodes %d", filename, want)
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
		return fmt.Errorf("%s LPU partitions use %d LPU nodes, want deployment.numLpuNodes %d", filename, packagedNodes, want)
	}

	return fmt.Errorf(
		"%s LPU partitions use %d packaged LPU nodes or %d with host embeddings, want deployment.numLpuNodes %d",
		filename,
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
