/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// validateSelectedPropSyncGraph returns validated forward-adjacent edge source positions in selected-chain order.
func validateSelectedPropSyncGraph(
	partitions []buildPartition,
	chains [][]int,
	subject string,
) ([]int, error) {
	// Empty chain sets have no references or edges to validate.
	if len(chains) == 0 {
		return []int{}, nil
	}

	// Index every physical partition once for both reference validation and connector projection.
	partitionPositions := make(map[int]int, len(partitions))
	for position, partition := range partitions {
		partitionPositions[partition.sourcePartitionID] = position
	}

	// Reject malformed references across every chain before evaluating relationships between valid members.
	for chainIndex, chain := range chains {
		for _, partitionID := range chain {
			if _, present := partitionPositions[partitionID]; !present {
				return nil, fmt.Errorf("%s %d references missing partition ID %d", subject, chainIndex, partitionID)
			}
		}
	}

	// Enforce disjoint forward-adjacent chains and record each ordered physical edge by source position.
	edgePositions := make([]int, 0)
	for chainIndex, chain := range chains {
		previousPosition := partitionPositions[chain[0]]
		for memberIndex, partitionID := range chain {
			position, unused := partitionPositions[partitionID]
			if !unused {
				return nil, fmt.Errorf("%s %d overlaps partition ID %d", subject, chainIndex, partitionID)
			}
			delete(partitionPositions, partitionID)
			if memberIndex == 0 {
				continue
			}

			if position != previousPosition+1 {
				return nil, fmt.Errorf("%s %d is not forward-adjacent at partition ID %d", subject, chainIndex, partitionID)
			}
			edgePositions = append(edgePositions, previousPosition)
			previousPosition = position
		}
	}
	return edgePositions, nil
}

func (b *Build) selectXTRuntimePartitions() error {
	if err := b.consumeRuntimeSelectedPropSyncChain(); err != nil {
		return err
	}
	if b.supportsCPUEmbeddings && b.standaloneTokenEmbeddings &&
		len(b.partitions) > 1 && b.partitions[0].sourcePartitionID == 0 {
		b.partitions = b.partitions[1:]
	}
	return nil
}

func (b *Build) consumeRuntimeSelectedPropSyncChain() error {
	if len(b.selectedPropSyncChains) == 0 {
		return nil
	}
	if len(b.selectedPropSyncChains) != 1 {
		return fmt.Errorf("LPU-only runtime requires exactly one selected prop-sync chain, got %d", len(b.selectedPropSyncChains))
	}
	chain := b.selectedPropSyncChains[0]

	// Locate the selected chain in normalized physical partition order.
	firstSelected := sort.Search(len(b.partitions), func(index int) bool {
		return b.partitions[index].sourcePartitionID >= chain[0]
	})

	// A reached member has a contiguous prefix, so modular distance below its index identifies a duplicate.
	for memberIndex, partitionID := range chain {
		if uint(partitionID)-uint(chain[0]) < uint(memberIndex) {
			return fmt.Errorf("LPU-only selected prop-sync chain %s contains duplicate partition id %d", formatPropSyncChain(chain), partitionID)
		}
		if memberIndex > 0 && partitionID != chain[memberIndex-1]+1 {
			return fmt.Errorf("LPU-only selected prop-sync chain %s is not contiguous at partition id %d", formatPropSyncChain(chain), partitionID)
		}

		partitionIndex := firstSelected + memberIndex
		if partitionIndex >= len(b.partitions) || b.partitions[partitionIndex].sourcePartitionID != partitionID {
			return fmt.Errorf("LPU-only selected prop-sync chain %s references missing partition id %d", formatPropSyncChain(chain), partitionID)
		}
	}

	b.partitions = b.partitions[firstSelected : firstSelected+len(chain)]
	b.selectedPropSyncChains = nil
	return nil
}

// packPropSyncPartitions combines small stages into one whole-node reservation.
// With 8 chips/node: [4, 4, 4] -> [16]. With 16 chips/node: [8, 8, 8] -> [32].
// The result keeps the first stage's ID; input stages are unchanged.
// Partitions must be nonempty and normalized, with positive chip counts and node widths.
func packPropSyncPartitions(partitions []buildPartition) []buildPartition {
	// Shared-node stages need matching chip counts and, for XT, identical topology
	// names. Decoding guarantees that every partition of a build has one node width.
	root := partitions[0]
	chips, chipsPerNode := root.numChips, root.devicesPerNode
	if len(partitions) < 2 || chips >= chipsPerNode || chipsPerNode%chips != 0 {
		return partitions
	}
	for _, partition := range partitions[1:] {
		if partition.numChips != chips ||
			(len(partition.hxExtent) == 0 && partition.topology != root.topology) {
			return partitions
		}
	}

	// Reserve the final node in full even when some of its chips are unused.
	root.numChips = ((len(partitions)*chips + chipsPerNode - 1) / chipsPerNode) * chipsPerNode
	return []buildPartition{root}
}

func formatPropSyncChain(chain []int) string {
	ids := make([]string, 0, len(chain))
	for _, partitionID := range chain {
		ids = append(ids, strconv.Itoa(partitionID))
	}
	return "[" + strings.Join(ids, ",") + "]"
}
