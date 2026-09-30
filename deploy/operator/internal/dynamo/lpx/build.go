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

// BuildCompilationMode records the compiler-authored execution mode in a normalized build.
type BuildCompilationMode string

// BuildFamily identifies the physical LPU target family of a normalized build.
type BuildFamily string

const (
	// BuildCompilationModeUnknown represents a build whose compilation mode is not known.
	BuildCompilationModeUnknown BuildCompilationMode = ""
	// BuildCompilationModeLPUOnly represents a build executed entirely on LPUs.
	BuildCompilationModeLPUOnly BuildCompilationMode = "lpuOnly"
	// BuildCompilationModeHybrid retains the model manifest's historical "lpx" value.
	BuildCompilationModeHybrid BuildCompilationMode = "lpx"
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
	// CompilationMode selects the compiler-authored LPU-only or hybrid artifact mode.
	CompilationMode BuildCompilationMode
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
	// NumChips is the positive LPU chip count declared by the manifest.
	NumChips int
	// DevicesPerNode is the positive number of LPU devices per node declared by the manifest.
	DevicesPerNode int
	// HXExtent is the scheduler-facing four-dimensional HX allocation.
	HXExtent []int64

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
func (p BuildPartition) effectiveNodeCount() int {
	if p.runtimeNodeCount > 0 {
		return p.runtimeNodeCount
	}
	return max(1, p.NumChips/p.DevicesPerNode)
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
