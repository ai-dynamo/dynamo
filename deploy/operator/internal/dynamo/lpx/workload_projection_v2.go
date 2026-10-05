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

//nolint:gocyclo // V2 projection validates one complete transformation.
func appendV2ModelProjections(dst []*ModelProjection, intent ModelProjectionInput) ([]*ModelProjection, error) {
	configured := *intent.BuildSnapshot.build
	usesResolvedRuntime := intent.Pipeline == PipelineSingle ||
		intent.Pipeline == PipelineSpecDecode
	ioFPGACount, ioFanoutFactor := configured.IOFPGACount, configured.IOFanoutFactor
	connectorBuild := intent.BuildSnapshot.build

	// Apply the selected chain and CPU embedding placement before deriving scheduler requests.
	if usesResolvedRuntime {
		if err := configured.selectXTRuntimePartitions(); err != nil {
			return nil, fmt.Errorf("resolving configured V2 build: %w", err)
		}
	}

	allocationMetadata := json.RawMessage(`{}`)

	// Combine packed physical reservations without changing the runtime's logical stages.
	partitions := configured.Partitions
	if usesResolvedRuntime && len(connectorBuild.SelectedPropSyncChains) != 0 {
		partitions = packPropSyncPartitions(partitions)
	}

	transcripts := newModelProjectionTranscripts(intent, v2ProjectionVersion)
	for index := range transcripts {
		transcript := &transcripts[index]
		transcript.field("input-embeddings-on-gpu", []byte{1})
		bindHybridRuntimeIO(transcript, intent.Pipeline, ioFPGACount, ioFanoutFactor)
	}

	// Bind validated physical partitions into projection identity while counting runtime endpoints.
	agentReplicas := 0
	for index, partition := range partitions {
		compilerID := uint32(partition.SourcePartitionID)
		shape, endpoints, shapeErr := xtShape(partition)
		if shapeErr != nil {
			return nil, shapeErr
		}
		agentReplicas += int(endpoints)
		for modelIndex := range transcripts {
			transcripts[modelIndex].uint32Field("compiler-partition-id", compilerID)
			transcripts[modelIndex].uint32Field("model-partition-id", uint32(index))
			transcripts[modelIndex].intField("endpoint-count", endpoints)
			transcripts[modelIndex].field("xt-shape", []byte(shape))
		}
	}

	connectors, err := v2Connectors(connectorBuild, partitions)
	if err != nil {
		return nil, err
	}

	// Preserve physical scheduler partitions while collapsing selected chains only in LPU runtime state.
	if intent.Pipeline == PipelineLPX && len(configured.SelectedPropSyncChains) != 0 {
		runtimeChainByRoot := make(map[int][]int, len(connectorBuild.SelectedPropSyncChains))
		for _, chain := range connectorBuild.SelectedPropSyncChains {
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
			for _, stage := range partitions[partitionIndex:chainEnd] {
				partition.runtimeNodeCount += stage.effectiveNodeCount()
			}
			collapsed = append(collapsed, partition)
			partitionIndex = chainEnd
		}
		configured.Partitions = collapsed
		configured.SelectedPropSyncChains = nil
	}

	// Encode shared connectors once without changing any model's digest field order.
	for _, connector := range connectors {
		encoded, _ := json.Marshal(connector)
		for index := range transcripts {
			transcripts[index].field("connector", encoded)
		}
	}

	// Publish distinct logical identities backed by the component's immutable configuration.
	for index := range transcripts {
		transcript := &transcripts[index]
		transcript.field("allocation-metadata", allocationMetadata)

		dst = append(dst, &ModelProjection{
			digest:                 transcript.sum(),
			compilerSnapshotDigest: intent.BuildSnapshot.contentID,
			runtimeBuildRef:        intent.RuntimeBuildRef,
			model:                  intent.Models[index],
			pipeline:               intent.Pipeline,
			configuredBuild:        configured,
			allocationMetadata:     allocationMetadata,
			partitions:             partitions,
			connectors:             connectors,
			agentReplicas:          agentReplicas,
		})
	}
	return dst, nil
}

func xtShape(partition BuildPartition) (lpxv1alpha1.Xt8888PartitionShape, int64, error) {
	// These chip capacities name registered XT shapes; node width comes from the manifest.
	chips := max(partition.Topology.ChipCount, partition.DevicesPerNode)
	switch chips {
	case 8, 16, 24, 32, 40, 48, 56, 64, 96, 128:
		return lpxv1alpha1.Xt8888PartitionShape(fmt.Sprintf("c%d", chips)), int64(partition.effectiveNodeCount()), nil
	default:
		return "", 0, fmt.Errorf("partition %d has unregistered XT shape c%d", uint32(partition.SourcePartitionID), chips)
	}
}

// v2Connectors validates source chains before emitting edges between reservations.
// build is normalized and nonnil; partitions is nonempty and its source IDs form
// a contiguous interval of build.Partitions (only the root for a packed chain).
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
	connectors := make([]lpxv1alpha1.PropSyncConnectorRequest, 0, min(len(edgePositions), len(partitions)-1))
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
