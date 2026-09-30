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

// compilationMode records the compiler-authored execution mode in a normalized build.
type compilationMode string

const (
	// compilationModeLPUOnly represents a build executed entirely on LPUs.
	compilationModeLPUOnly compilationMode = "lpuOnly"
	// compilationModeHybrid retains the model manifest's historical "lpx" value.
	compilationModeHybrid compilationMode = "lpx"
)

// Build is the registry's source-independent view of an LPU build.
//
// The version 2 Cap'n Proto manifest populates this shape before deployment code
// derives placement and replica counts.
type Build struct {
	// contentID is the digest of the compiler manifest that produced the build.
	contentID string
	// path is the absolute file or GCS reference of the build payload.
	path string
	// family is the physical LPU target family.
	family *family
	// compilationMode selects the compiler-authored LPU-only or hybrid artifact mode.
	compilationMode compilationMode
	// partitions contains the normalized physical compiler partitions.
	partitions []buildPartition
	// selectedPropSyncChains contains source partition IDs grouped into selected prop-sync chains.
	selectedPropSyncChains [][]int
	// standaloneTokenEmbeddings reports whether token embeddings occupy a standalone partition.
	standaloneTokenEmbeddings bool
	// supportsCPUEmbeddings reports whether standalone token embeddings may run on the CPU.
	supportsCPUEmbeddings bool
	// ioFPGACount is the number of I/O FPGA endpoints described by the build.
	ioFPGACount int32
	// ioFanoutFactor is the number of clients assigned to each I/O FPGA transaction.
	ioFanoutFactor int32
}

// buildPartition describes one normalized physical compiler partition.
type buildPartition struct {
	// sourcePartitionID is the compiler partition id used in artifacts and
	// selected prop-sync chains. It is not the slice index after sorting/filtering.
	sourcePartitionID int
	// partPath is the nonempty relative gas-dir fragment under the build payload.
	partPath string
	// numChips is the positive LPU chip count declared by the manifest.
	numChips int
	// devicesPerNode is the positive number of LPU devices per node declared by the manifest.
	devicesPerNode int
	// hxExtent is the scheduler-facing four-dimensional HX allocation.
	hxExtent []int64

	// runtimeNodeCount overrides the node count derived from manifest geometry after
	// selected prop-sync partitions are collapsed for the LPU runtime. Sub-host
	// partitions still occupy one scheduler endpoint each, so their combined
	// chip count alone cannot recover the number of scheduled Agent pods.
	runtimeNodeCount int
}

// effectiveNodeCount returns the number of Agent endpoints assigned to the
// partition. Source partitions derive it from manifest geometry; collapsed runtime
// partitions preserve the sum of their physical scheduler endpoints.
// The partition must have validated chip and device counts.
func (p buildPartition) effectiveNodeCount() int {
	if p.runtimeNodeCount > 0 {
		return p.runtimeNodeCount
	}
	return max(1, p.numChips/p.devicesPerNode)
}

// buildRuntimePath resolves a snapshot reference to its runtime filesystem path.
// A safe relative runtimeRef remaps file-URL snapshots under modelStoragePath;
// otherwise the snapshot reference remains authoritative.
func buildRuntimePath(buildPath, runtimeRef, modelStoragePath string) (string, error) {
	// Remap file snapshots from the operator's cache to the runtime's model mount.
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

	// Validate the selected reference before resolving the final filesystem path.
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
