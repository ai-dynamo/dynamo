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
	partitions []BuildPartition,
	chains [][]int,
	subject string,
	requireCompatibleTopology bool,
) ([]int, error) {
	// Index every physical partition once for both reference validation and connector projection.
	partitionPositions := make(map[int]int, len(partitions))
	for position, partition := range partitions {
		partitionPositions[partition.SourcePartitionID] = position
	}

	// Reject malformed references across every chain before evaluating relationships between valid members.
	for chainIndex, chain := range chains {
		if len(chain) < 2 {
			return nil, fmt.Errorf("%s %d must contain at least two partition IDs", subject, chainIndex)
		}
		for _, partitionID := range chain {
			if _, present := partitionPositions[partitionID]; !present {
				return nil, fmt.Errorf("%s %d references missing partition ID %d", subject, chainIndex, partitionID)
			}
		}
	}

	// Enforce disjoint forward-adjacent chains and record each ordered physical edge by source position.
	edgePositions := make([]int, 0)
	for chainIndex, chain := range chains {
		rootPosition := partitionPositions[chain[0]]
		previousPosition := rootPosition
		for memberIndex, partitionID := range chain {
			position, unused := partitionPositions[partitionID]
			if !unused {
				return nil, fmt.Errorf("%s %d overlaps partition ID %d", subject, chainIndex, partitionID)
			}
			delete(partitionPositions, partitionID)
			if requireCompatibleTopology &&
				!partitions[rootPosition].Topology.compatibleWith(partitions[position].Topology) {
				return nil, fmt.Errorf("%s %d has incompatible topology at partition ID %d", subject, chainIndex, partitionID)
			}
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
	if b == nil || len(b.SelectedPropSyncChains) == 0 {
		return nil
	}
	if len(b.SelectedPropSyncChains) != 1 {
		return fmt.Errorf("LPU-only runtime requires exactly one selected prop-sync chain, got %d", len(b.SelectedPropSyncChains))
	}
	chain := b.SelectedPropSyncChains[0]
	if len(chain) < 2 {
		return fmt.Errorf("LPU-only selected prop-sync chain %s must contain at least two partition ids", formatPropSyncChain(chain))
	}

	// Locate the selected chain in normalized physical partition order.
	firstSelected := sort.Search(len(b.Partitions), func(index int) bool {
		return b.Partitions[index].SourcePartitionID >= chain[0]
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
		if partitionIndex >= len(b.Partitions) || b.Partitions[partitionIndex].SourcePartitionID != partitionID {
			return fmt.Errorf("LPU-only selected prop-sync chain %s references missing partition id %d", formatPropSyncChain(chain), partitionID)
		}
	}

	b.Partitions = b.Partitions[firstSelected : firstSelected+len(chain)]
	b.SelectedPropSyncChains = nil
	return nil
}

// collapseSelectedPropSyncChain projects validated nonempty contiguous physical partitions onto their runtime root.
func collapseSelectedPropSyncChain(chain []int, partitions []BuildPartition) (BuildPartition, error) {
	root := partitions[0]
	totalChipCount := 0
	totalNodeCount := 0
	for _, partition := range partitions {
		totalChipCount += partition.Topology.ChipCount
		totalNodeCount += partition.effectiveNodeCount()
	}

	topology, err := root.Topology.withChipCount(totalChipCount)
	if err != nil {
		return BuildPartition{}, fmt.Errorf("selected prop-sync chain %s cannot form combined topology: %w", formatPropSyncChain(chain), err)
	}
	return BuildPartition{
		SourcePartitionID: root.SourcePartitionID,
		PartPath:          root.PartPath,
		Topology:          topology,
		runtimeNodeCount:  totalNodeCount,
	}, nil
}

func formatPropSyncChain(chain []int) string {
	ids := make([]string, 0, len(chain))
	for _, partitionID := range chain {
		ids = append(ids, strconv.Itoa(partitionID))
	}
	return "[" + strings.Join(ids, ",") + "]"
}
