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

// collapseSelectedPropSyncChain projects validated nonempty contiguous physical partitions onto their runtime root.
func collapseSelectedPropSyncChain(partitions []buildPartition) buildPartition {
	root := partitions[0]
	totalChipCount := 0
	totalNodeCount := 0
	for _, partition := range partitions {
		totalChipCount += partition.numChips
		totalNodeCount += partition.effectiveNodeCount()
	}

	return buildPartition{
		sourcePartitionID: root.sourcePartitionID,
		partPath:          root.partPath,
		numChips:          totalChipCount,
		devicesPerNode:    root.devicesPerNode,
		runtimeNodeCount:  totalNodeCount,
	}
}

func formatPropSyncChain(chain []int) string {
	ids := make([]string, 0, len(chain))
	for _, partitionID := range chain {
		ids = append(ids, strconv.Itoa(partitionID))
	}
	return "[" + strings.Join(ids, ",") + "]"
}
