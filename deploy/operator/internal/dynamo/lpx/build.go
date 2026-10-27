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

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
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
// The validated compiler contract populates this shape before deployment code
// derives placement and replica counts.
type Build struct {
	// contentID identifies the exact compiler contract used to acquire this build.
	contentID      string
	contractFormat buildContractFormat
	// path is the absolute file or GCS reference of the build payload.
	path string
	// family is the physical LPU target family.
	family BuildFamily
	// hasCUDAPartitions retains GPU requirements excluded from the LPU partition projection.
	hasCUDAPartitions bool
	// partitions contains the normalized physical compiler partitions.
	partitions []buildPartition
	// selectedPropSyncChains contains source partition IDs grouped into selected prop-sync chains.
	selectedPropSyncChains [][]int
	// lpuUnits is the validated compiler-owned LPU placement in Agent order. Nil
	// means the compiler predates units, so projection derives placement from
	// partitions and selectedPropSyncChains.
	lpuUnits []lpuUnit
	// standaloneTokenEmbeddings reports whether token embeddings occupy a standalone partition.
	standaloneTokenEmbeddings bool
	// supportsCPUEmbeddings reports whether standalone token embeddings may run on the CPU.
	supportsCPUEmbeddings bool
	// ioFPGACount is the number of I/O FPGA endpoints described by the build.
	ioFPGACount int32
	// ioFanoutFactor is the number of clients assigned to each I/O FPGA transaction.
	ioFanoutFactor int32
	// checkpoint is the validated source checkpoint that the deployment contract records, or nil.
	checkpoint *Checkpoint
	// checkpointErr rejects the recorded checkpoint. It fails only a workload
	// that would project the checkpoint, so other workloads keep the build.
	checkpointErr error
}

// buildPartition describes one normalized physical compiler partition.
type buildPartition struct {
	// sourcePartitionID is the compiler partition id used in artifacts and
	// selected prop-sync chains. It is not the slice index after sorting/filtering.
	sourcePartitionID int
	// partPath is the nonempty relative gas-dir fragment under the build payload.
	partPath string
	// numChips is the chip count declared by the compiler manifest.
	numChips int
	// topology identifies compatible subnode stages for PropSync packing.
	topology string
	// devicesPerNode is the number of chips per physical node in the compiler manifest.
	devicesPerNode int
	// hxExtent is the scheduler-facing four-dimensional HX allocation.
	hxExtent []int64

	// runtimeNodeCount sums physical endpoints for a collapsed hybrid chain;
	// numChips retains the original root stage.
	runtimeNodeCount int
}

// lpuUnit is one compiler-owned runtime group: a selected prop-sync chain or a
// single LPU partition. Its root, the first partition of its first reservation,
// names the unit in localPartitions and LPX_LOCAL_PARTITION_IDS. The exported
// fields define the unit's digest encoding.
type lpuUnit struct {
	// Reservations lists the unit's scheduler partitions in Agent order.
	Reservations []lpuReservation `json:"reservations"`
	// Connectors[i] joins Reservations[i] to Reservations[i+1].
	Connectors []lpxv1alpha1.PropSyncConnectorRequirement `json:"connectors"`
}

// lpuReservation is the set of hosts that one LPU partition, or several packed
// stages of one chain, run on.
type lpuReservation struct {
	// PartitionIDs lists consecutive ascending compiler partition IDs.
	PartitionIDs []int `json:"partitionIds"`
	// Hosts is the reservation's Agent count.
	Hosts int `json:"hosts"`
	// HXExtent is the LP30 scheduler extent [devicesPerNode, traysPerHemisphere,
	// hemispheres, racks] from size.hx. Nil on LP20.
	HXExtent []int64 `json:"hxExtent,omitempty"`
}

// root returns the unit's first partition ID.
func (u lpuUnit) root() int {
	return u.Reservations[0].PartitionIDs[0]
}

// partitionIDs returns the unit's partition IDs across its reservations.
func (u lpuUnit) partitionIDs() []int {
	var ids []int
	for _, reservation := range u.Reservations {
		ids = append(ids, reservation.PartitionIDs...)
	}
	return ids
}

// effectiveNodeCount returns the number of Agent endpoints assigned to the
// partition. Source partitions derive it from topology; collapsed runtime
// partitions preserve the sum of their physical scheduler endpoints.
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
