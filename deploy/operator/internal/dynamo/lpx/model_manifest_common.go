/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"path/filepath"
	"strings"

	"capnproto.org/go/capnp/v3"
)

const hxTopologyFamily = "16x8x2x3"

func classifyManifestPartitions(filename string, partitions []buildPartition, partSelect bool) (buildTopology, error) {
	family := xtBuildTopology
	seen := make(map[int]struct{}, len(partitions))
	for _, partition := range partitions {
		partitionFamily := xtBuildTopology
		if len(partition.hxExtent) != 0 {
			partitionFamily = hxBuildTopology
		}
		if len(seen) != 0 && family != partitionFamily {
			return nil, fmt.Errorf("%s mixes XT and HX LPU partitions", filename)
		}
		family = partitionFamily
		if _, duplicate := seen[partition.sourcePartitionID]; duplicate {
			if family == hxBuildTopology {
				return nil, fmt.Errorf("V3 %s repeats LPU partition ID %d", filename, partition.sourcePartitionID)
			}
			return nil, fmt.Errorf("%s has duplicate LPU partition id %d", filename, partition.sourcePartitionID)
		}
		seen[partition.sourcePartitionID] = struct{}{}

		// One Agent template serves the build; mixed node widths require Dynamo rendering support.
		if partition.devicesPerNode != partitions[0].devicesPerNode {
			return nil, fmt.Errorf("%s LPU partition %d has devicesPerNode %d, want %d; all LPU partitions must use the same devicesPerNode",
				filename, partition.sourcePartitionID, partition.devicesPerNode, partitions[0].devicesPerNode)
		}
	}
	if err := family.normalizePartitions(filename, partitions, partSelect); err != nil {
		return nil, err
	}
	return family, nil
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

func buildLPUArtifact[M lpuTopologyMetadata, P lpuPartitionArtifact[M]](filename string, partitionID uint32, raw P, family buildTopology) (buildPartition, error) {
	topology, err := raw.Topology()
	if err != nil {
		return buildPartition{}, fmt.Errorf("reading %s LPU partition %d topology: %w", filename, partitionID, err)
	}
	path, err := raw.Path()
	if err != nil {
		return buildPartition{}, fmt.Errorf("reading %s LPU partition %d path: %w", filename, partitionID, err)
	}

	// Topology names are opaque but must fit one runtime configuration line.
	subject := fmt.Sprintf("%s LPU partition %d", filename, partitionID)
	if strings.TrimSpace(topology) == "" {
		return buildPartition{}, fmt.Errorf("%s topology must not be empty", subject)
	}
	if strings.ContainsAny(topology, "\x00\r\n") {
		return buildPartition{}, fmt.Errorf("%s topology must not contain NUL bytes or line breaks: %q", subject, topology)
	}

	// Each schema selects the same behavior from its explicit architecture.
	partition, err := family.decodePartition(subject, topology, topologyArtifact[M]{raw})
	if err != nil {
		return buildPartition{}, err
	}

	// Validate shared artifact fields once, after the selected geometry is accepted.
	partition.partPath, err = cleanManifestRelativeBuildPath(subject+" path", path)
	if err != nil {
		return buildPartition{}, err
	}
	partition.sourcePartitionID = int(partitionID)
	return partition, nil
}
