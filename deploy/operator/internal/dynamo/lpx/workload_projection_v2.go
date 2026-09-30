/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"fmt"
	"slices"
	"sort"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
)

const v2ProjectionVersion = "v2-xt-node-local/v1"

// projectV2Component derives the XT component shared by every model and writes
// its model-independent digest fields.
func projectV2Component(intent ModelProjectionInput, fields digestTranscript) (ModelProjection, error) {
	source := intent.BuildSnapshot.build
	configured := *source

	// Apply the selected chain and CPU embedding placement before deriving scheduler requests.
	if intent.Pipeline != PipelineHybrid {
		if err := configured.consumeRuntimeSelectedPropSyncChain(); err != nil {
			return ModelProjection{}, fmt.Errorf("resolving configured V2 build: %w", err)
		}

		// Omit host-only embeddings by retaining a view of the immutable source partitions.
		if configured.SupportsCPUEmbeddings && configured.StandaloneTokenEmbeddings &&
			len(configured.Partitions) > 1 && configured.Partitions[0].SourcePartitionID == 0 {
			configured.Partitions = configured.Partitions[1:]
		}
	}

	// Bind the V2 workload and runtime contract into projection identity before partition validation.
	fields.field("input-embeddings-on-gpu", []byte{1})
	bindHybridRuntimeIO(fields, intent.Pipeline, configured.IOFPGACount, configured.IOFanoutFactor)

	// Bind validated physical partitions into projection identity while counting runtime endpoints.
	partitions := configured.Partitions
	agentReplicas := 0
	for index, partition := range partitions {
		compilerID := uint32(partition.SourcePartitionID)
		shape, err := xtShape(partition)
		if err != nil {
			return ModelProjection{}, fmt.Errorf("V2 compiler partition %d: %w", compilerID, err)
		}
		endpoints := partition.effectiveNodeCount()
		agentReplicas += endpoints
		fields.uint32Field("compiler-partition-id", compilerID)
		fields.uint32Field("model-partition-id", uint32(index))
		fields.intField("endpoint-count", int64(endpoints))
		fields.field("xt-shape", []byte(shape))
	}
	connectors, err := v2Connectors(source, partitions)
	if err != nil {
		return ModelProjection{}, err
	}
	// Preserve physical scheduler partitions while collapsing selected chains only in LPU runtime state.
	if intent.Pipeline == PipelineHybrid && len(configured.SelectedPropSyncChains) != 0 {
		runtimeChainByRoot := make(map[int][]int, len(configured.SelectedPropSyncChains))
		for _, chain := range configured.SelectedPropSyncChains {
			runtimeChainByRoot[chain[0]] = chain
		}
		collapsed := make([]BuildPartition, 0, len(partitions))
		for partitionIndex := 0; partitionIndex < len(partitions); {
			partition := partitions[partitionIndex]
			chain, selected := runtimeChainByRoot[partition.SourcePartitionID]
			if !selected {
				collapsed = append(collapsed, partition)
				partitionIndex++
				continue
			}
			chainEnd := partitionIndex + len(chain)
			collapsed = append(collapsed, collapseSelectedPropSyncChain(partitions[partitionIndex:chainEnd]))
			partitionIndex = chainEnd
		}
		configured.Partitions = collapsed
		configured.SelectedPropSyncChains = nil
	}

	for _, connector := range connectors {
		encoded, _ := json.Marshal(connector)
		fields.field("connector", encoded)
	}
	allocationMetadata := json.RawMessage(`{}`)
	fields.field("allocation-metadata", allocationMetadata)

	return ModelProjection{
		configuredBuild:    configured,
		allocationMetadata: allocationMetadata,
		partitions:         partitions,
		connectors:         connectors,
		agentReplicas:      agentReplicas,
	}, nil
}

func xtShape(partition BuildPartition) (lpxv1alpha1.Xt8888PartitionShape, error) {
	// XT8888 scheduler shapes require eight physical devices per host.
	if partition.DevicesPerNode != 8 {
		return "", fmt.Errorf("devicesPerNode %d, want 8 for XT8888 scheduler shapes", partition.DevicesPerNode)
	}

	// Reserve one whole physical host for compiler partitions that use fewer than eight chips.
	chipCount := partition.NumChips
	if chipCount > 0 && chipCount < 8 {
		return lpxv1alpha1.Xt8888PartitionShapeC8, nil
	}

	// Reject partial and unregistered whole-host shapes before deriving their LPX names.
	if chipCount < 8 || chipCount%8 != 0 || (chipCount > 64 && chipCount != 96 && chipCount != 128) {
		return "", fmt.Errorf("chip count %d is not a registered XT8888 partition shape", chipCount)
	}
	return lpxv1alpha1.Xt8888PartitionShape(fmt.Sprintf("c%d", chipCount)), nil
}

// v2Connectors requires a normalized nonnil build and a nonempty contiguous
// interval of its physical partitions. Runtime chain collapse happens afterward.
func v2Connectors(
	build *Build,
	partitions []BuildPartition,
) ([]lpxv1alpha1.PropSyncConnectorRequest, error) {
	// Only compiler-selected relationships impose placement constraints.
	if len(build.SelectedPropSyncChains) == 0 {
		return []lpxv1alpha1.PropSyncConnectorRequest{}, nil
	}

	// Validate explicit chains before ordering their scheduler edges.
	edgePositions, err := validateSelectedPropSyncGraph(
		build.Partitions,
		build.SelectedPropSyncChains,
		"selected prop-sync chain",
	)
	if err != nil {
		return nil, err
	}

	// Scheduler output follows physical order, not chain declaration order.
	slices.Sort(edgePositions)

	// Rebase selected physical edges into the retained partition interval.
	start := sort.Search(len(build.Partitions), func(index int) bool {
		return build.Partitions[index].SourcePartitionID >= partitions[0].SourcePartitionID
	})
	connectors := make([]lpxv1alpha1.PropSyncConnectorRequest, 0, len(edgePositions))
	for _, position := range edgePositions {
		i := position - start
		if i < 0 {
			continue
		}
		if i+1 >= len(partitions) {
			break
		}
		offset := int64(0)
		connectors = append(connectors, lpxv1alpha1.PropSyncConnectorRequest{
			FromPartitionID: fmt.Sprintf("partition-%03d", i),
			ToPartitionID:   fmt.Sprintf("partition-%03d", i+1),
			Requirement: lpxv1alpha1.PropSyncConnectorRequirement{
				Kind:                    lpxv1alpha1.PropSyncConnectorKindXt8888Gap,
				MaxInterPartitionOffset: &offset,
			},
		})
	}
	return connectors, nil
}
