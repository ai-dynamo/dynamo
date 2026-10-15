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
)

const hxTopologyFamily = "16x8x2x3"

func classifyManifestPartitions(filename string, partitions []BuildPartition, partSelect bool) (BuildFamily, error) {
	family := BuildFamilyXT
	seen := make(map[int]struct{}, len(partitions))
	for _, partition := range partitions {
		partitionFamily := BuildFamilyXT
		if len(partition.HXExtent) != 0 {
			partitionFamily = BuildFamilyHX
		}
		if len(seen) != 0 && family != partitionFamily {
			return "", fmt.Errorf("%s mixes XT and HX LPU partitions", filename)
		}
		family = partitionFamily
		if _, duplicate := seen[partition.SourcePartitionID]; duplicate {
			if family == BuildFamilyHX {
				return "", fmt.Errorf("V3 %s repeats LPU partition ID %d", filename, partition.SourcePartitionID)
			}
			return "", fmt.Errorf("%s has duplicate LPU partition id %d", filename, partition.SourcePartitionID)
		}
		seen[partition.SourcePartitionID] = struct{}{}

		// One Agent template serves the build; mixed node widths require Dynamo rendering support.
		if partition.DevicesPerNode != partitions[0].DevicesPerNode {
			return "", fmt.Errorf("%s LPU partition %d has devicesPerNode %d, want %d; all LPU partitions must use the same devicesPerNode",
				filename, partition.SourcePartitionID, partition.DevicesPerNode, partitions[0].DevicesPerNode)
		}
	}
	if family == BuildFamilyHX {
		if partSelect {
			return "", fmt.Errorf("V3 %s partSelect builds are not supported", filename)
		}
		return family, nil
	}
	sort.Slice(partitions, func(i, j int) bool { return partitions[i].SourcePartitionID < partitions[j].SourcePartitionID })
	return family, nil
}

func buildXTPartition(subject, topology string, raw interface {
	NumChips() uint32
	DevicesPerNode() uint32
}) (BuildPartition, error) {
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
	return BuildPartition{topology: topology, NumChips: numChips, DevicesPerNode: devicesPerNode}, nil
}

type lpuTopologyMetadata interface {
	TopologyFamily() (string, error)
	PartitionShape() (capnp.UInt32List, error)
}

// Both generated families expose the same physical LPU geometry.
type lpuPartitionArtifact[M lpuTopologyMetadata] interface {
	NumChips() uint32
	DevicesPerNode() uint32
	HasTopologyMetadata() bool
	TopologyMetadata() (M, error)
	Topology() (string, error)
	Path() (string, error)
}

func buildHXPartition[M lpuTopologyMetadata, P lpuPartitionArtifact[M]](subject, topology string, raw P) (BuildPartition, error) {
	devicesPerNode, err := positiveManifestUInt32ToInt(subject+" devicesPerNode", raw.DevicesPerNode())
	if err != nil {
		return BuildPartition{}, err
	}
	partition := BuildPartition{topology: topology, NumChips: int(raw.NumChips()), DevicesPerNode: devicesPerNode}
	if !raw.HasTopologyMetadata() {
		chips := partition.NumChips
		if chips <= 0 || chips > devicesPerNode || devicesPerNode%chips != 0 {
			return BuildPartition{}, fmt.Errorf("V3 %s without topologyMetadata requires a one-node stage whose chip count divides %d", subject, devicesPerNode)
		}
		partition.HXExtent = []int64{int64(devicesPerNode), 1, 1, 1}
		return partition, nil
	}

	// Decode HX metadata directly and report read errors at their source.
	metadata, err := raw.TopologyMetadata()
	if err != nil {
		return BuildPartition{}, fmt.Errorf("reading V3 %s topologyMetadata: %w", subject, err)
	}
	family, err := metadata.TopologyFamily()
	if err != nil {
		return BuildPartition{}, fmt.Errorf("reading V3 %s topologyMetadata.topologyFamily: %w", subject, err)
	}
	if strings.TrimSpace(family) != hxTopologyFamily {
		return BuildPartition{}, fmt.Errorf("V3 %s topologyMetadata.topologyFamily = %q, want %q", subject, strings.TrimSpace(family), hxTopologyFamily)
	}
	shape, err := metadata.PartitionShape()
	if err != nil {
		return BuildPartition{}, fmt.Errorf("reading V3 %s topologyMetadata.partitionShape: %w", subject, err)
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
		return BuildPartition{}, fmt.Errorf("V3 %s has unsupported HX extent %v", subject, extent)
	}

	// Match the source chip count before rounding its scheduler reservation up to a whole node.
	count := extent[0] * extent[1] * extent[2] * extent[3]
	if count != int64(raw.NumChips()) {
		return BuildPartition{}, fmt.Errorf("V3 %s topologyMetadata.partitionShape contains %d chips, want numChips %d", subject, count, raw.NumChips())
	}
	extent[0] = int64(devicesPerNode)
	partition.HXExtent = extent
	return partition, nil
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

func positiveManifestUInt32ToInt(field string, value uint32) (int, error) {
	if value == 0 {
		return 0, fmt.Errorf("%s must be >= 1, got 0", field)
	}
	return int(value), nil
}

func buildLPUArtifact[M lpuTopologyMetadata, P lpuPartitionArtifact[M]](filename string, partitionID uint32, raw P, family BuildFamily) (BuildPartition, error) {
	topology, err := raw.Topology()
	if err != nil {
		return BuildPartition{}, fmt.Errorf("reading %s LPU partition %d topology: %w", filename, partitionID, err)
	}
	path, err := raw.Path()
	if err != nil {
		return BuildPartition{}, fmt.Errorf("reading %s LPU partition %d path: %w", filename, partitionID, err)
	}

	// Topology names are opaque but must fit one runtime configuration line.
	subject := fmt.Sprintf("%s LPU partition %d", filename, partitionID)
	if strings.TrimSpace(topology) == "" {
		return BuildPartition{}, fmt.Errorf("%s topology must not be empty", subject)
	}
	if strings.ContainsAny(topology, "\x00\r\n") {
		return BuildPartition{}, fmt.Errorf("%s topology must not contain NUL bytes or line breaks: %q", subject, topology)
	}

	// Each format selects the hardware family from its explicit architecture.
	var partition BuildPartition
	switch family {
	case BuildFamilyHX:
		partition, err = buildHXPartition(subject, topology, raw)
	case BuildFamilyXT:
		partition, err = buildXTPartition(subject, topology, raw)
	default:
		return BuildPartition{}, fmt.Errorf("%s unsupported hardware family %q", subject, family)
	}
	if err != nil {
		return BuildPartition{}, err
	}

	// Validate shared artifact fields once, after the selected geometry is accepted.
	partition.PartPath, err = cleanManifestRelativeBuildPath(subject+" path", path)
	if err != nil {
		return BuildPartition{}, err
	}
	partition.SourcePartitionID = int(partitionID)
	return partition, nil
}
