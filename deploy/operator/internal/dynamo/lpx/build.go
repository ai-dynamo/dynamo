/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"
)

const lpuChipsPerNode = 8

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
// derives container args, replica counts, prop-sync settings, and tokenizer defaults.
type Build struct {
	// Path is the absolute file or GCS reference of the build payload.
	Path string
	// Family is the physical LPU target family.
	Family BuildFamily
	// CompilationMode selects the compiler-authored LPU-only or hybrid artifact mode.
	CompilationMode BuildCompilationMode
	// Partitions contains the normalized physical compiler partitions.
	Partitions []BuildPartition
	// BatchSize is the normalized batch size for the build.
	BatchSize int
	// SelectedPropSyncChains contains source partition IDs grouped into selected prop-sync chains.
	SelectedPropSyncChains [][]int
	// StandaloneTokenEmbeddings reports whether token embeddings occupy a standalone partition.
	StandaloneTokenEmbeddings bool
	// SupportsCPUEmbeddings reports whether standalone token embeddings may run on the CPU.
	SupportsCPUEmbeddings bool
	// RuntimeTokenizerPath is the tokenizer path relative to the build payload.
	RuntimeTokenizerPath string
	// RuntimeTokenEmbeddingsPath is the token-embeddings path relative to the build payload.
	RuntimeTokenEmbeddingsPath string
	// IOFPGACount is the number of I/O FPGA endpoints described by the build.
	IOFPGACount int32
	// IOFanoutFactor is the number of clients assigned to each I/O FPGA transaction.
	IOFanoutFactor int32
	// runtimeSettings holds immutable normalized defaults on a snapshot and
	// configured runtime settings shared by immutable component projections.
	runtimeSettings map[string]any
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
	// HXExtent is the scheduler-facing four-dimensional HX allocation.
	HXExtent []int64

	// runtimeNodeCount overrides the node count derived from Topology after
	// selected prop-sync partitions are collapsed for the LPU runtime. Sub-host
	// partitions still occupy one scheduler endpoint each, so their combined
	// chip count alone cannot recover the number of scheduled Agent pods.
	runtimeNodeCount int
}

// effectiveNodeCount returns the number of Agent endpoints assigned to the
// partition. Source partitions derive it from topology; collapsed runtime
// partitions preserve the sum of their physical scheduler endpoints.
func (p BuildPartition) effectiveNodeCount() int {
	if p.runtimeNodeCount > 0 {
		return p.runtimeNodeCount
	}
	return p.Topology.Replicas()
}

// omitStandaloneEmbeddingPartition removes host-only embedding work when the
// build manifest explicitly identifies it as safe to run outside the LPU.
func (b *Build) omitStandaloneEmbeddingPartition() {
	if !b.SupportsCPUEmbeddings || !b.StandaloneTokenEmbeddings {
		return
	}

	// Normalized XT partitions are sorted by source ID; keep the retained interval independently owned.
	if len(b.Partitions) > 1 && b.Partitions[0].SourcePartitionID == 0 {
		b.Partitions = slices.Clone(b.Partitions[1:])
	}
}

func buildRuntimePath(buildPath string, modelStoragePath string) (string, error) {
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
