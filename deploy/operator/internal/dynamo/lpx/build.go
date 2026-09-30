/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"net/url"
	"path/filepath"
	"strings"
)

// BuildFamily identifies the physical LPU target family of a normalized build.
type BuildFamily string

const (
	// BuildFamilyXT identifies the XT8888 LPU target family.
	BuildFamilyXT BuildFamily = "xt8888"
	// BuildFamilyHX identifies the HX16x8x2x3 LPU target family.
	BuildFamilyHX BuildFamily = "hx16x8x2x3"
)

// Build is the registry's source-independent view of an LPU build.
//
// The version 2 Cap'n Proto manifest populates this shape before deployment code
// derives placement and replica counts.
type Build struct {
	// Path is the absolute file or GCS reference of the build payload.
	Path string
	// Family is the physical LPU target family.
	Family BuildFamily
	// HasCUDAPartitions retains GPU requirements excluded from the LPU partition projection.
	HasCUDAPartitions bool
	// Partitions contains the normalized physical compiler partitions.
	Partitions []BuildPartition
	// SelectedPropSyncChains contains source partition IDs grouped into selected prop-sync chains.
	SelectedPropSyncChains [][]int
	// StandaloneTokenEmbeddings reports whether token embeddings occupy a standalone partition.
	StandaloneTokenEmbeddings bool
	// SupportsCPUEmbeddings reports whether standalone token embeddings may run on the CPU.
	SupportsCPUEmbeddings bool
	// IOFPGACount is the number of I/O FPGA endpoints described by the build.
	IOFPGACount int32
	// IOFanoutFactor is the number of clients assigned to each I/O FPGA transaction.
	IOFanoutFactor int32
}

// BuildPartition describes one normalized physical compiler partition.
type BuildPartition struct {
	// SourcePartitionID is the compiler partition id used in artifacts and
	// selected prop-sync chains. It is not the slice index after sorting/filtering.
	SourcePartitionID int
	// PartPath is the nonempty relative gas-dir fragment under the build payload.
	PartPath string
	// Topology is the scheduler-facing node shape for this partition.
	Topology Topology
	// DevicesPerNode is the number of chips per physical node in the compiler manifest.
	DevicesPerNode int
	// HXExtent is the scheduler-facing four-dimensional HX allocation.
	HXExtent []int64

	// runtimeNodeCount sums physical endpoints for a collapsed hybrid chain;
	// Topology retains the original root stage.
	runtimeNodeCount int
}

// Topology holds an opaque compiler name and a chip count.
type Topology struct {
	ChipCount int
	Raw       string
}

// effectiveNodeCount returns the number of Agent endpoints assigned to the
// partition. Source partitions derive it from topology; collapsed runtime
// partitions preserve the sum of their physical scheduler endpoints.
func (p BuildPartition) effectiveNodeCount() int {
	if p.runtimeNodeCount > 0 {
		return p.runtimeNodeCount
	}
	return max(1, p.Topology.ChipCount/p.DevicesPerNode)
}

// buildRuntimePath resolves a snapshot reference to its runtime filesystem path.
// A safe relative runtimeRef remaps file-URL snapshots under modelStoragePath;
// otherwise the snapshot reference remains authoritative.
func buildRuntimePath(buildPath, runtimeRef, modelStoragePath string) (string, error) {
	snapshotURL, snapshotErr := url.Parse(buildPath)
	runtimeRef = strings.TrimSpace(runtimeRef)
	runtimeURL, runtimeErr := url.Parse(runtimeRef)
	if snapshotErr == nil && runtimeErr == nil && snapshotURL.Scheme == BuildSchemeFile &&
		runtimeRef != "" && runtimeURL.Scheme == "" && !filepath.IsAbs(runtimeRef) {
		cleaned := filepath.Clean(runtimeRef)
		if cleaned != "." && cleaned != ".." && !strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
			buildPath = (&url.URL{Scheme: BuildSchemeFile, Path: filepath.Join(modelStoragePath, cleaned)}).String()
		}
	}

	buildURL, err := parseBuildRef(buildPath)
	if err != nil {
		return "", fmt.Errorf("parse build path %q: %w", buildPath, err)
	}

	switch buildURL.Scheme {
	case BuildSchemeFile:
		return buildURL.Path, nil
	case BuildSchemeGCS:
		if modelStoragePath == "" {
			return "", fmt.Errorf("model storage path is empty")
		}

		objectPath := strings.TrimPrefix(buildURL.Path, "/")
		if objectPath == "" {
			return "", fmt.Errorf("invalid GCS build path %q: missing object path", buildPath)
		}

		// Validate every object path segment before joining the accepted path once.
		for component := range strings.SplitSeq(objectPath, "/") {
			if component == "" || component == "." || component == ".." {
				return "", fmt.Errorf("invalid GCS build path %q: bad path segment %q", buildPath, component)
			}
		}

		return filepath.Join(modelStoragePath, "gcs", buildURL.Host, objectPath), nil
	default:
		return "", fmt.Errorf("unsupported build path scheme %q", buildURL.Scheme)
	}
}
