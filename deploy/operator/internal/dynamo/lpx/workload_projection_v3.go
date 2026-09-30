/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
)

// projectV3Component derives the HX component shared by every model and writes
// its model-independent digest fields.
func projectV3Component(source *Build, pipeline Pipeline, fields digestTranscript) (*component, error) {
	configured := *source
	allocationMetadata, connectors, err := projectV3PropSync(configured.partitions, configured.selectedPropSyncChains, pipeline)
	if err != nil {
		return nil, err
	}
	// Selected chains are represented by the allocation metadata and connectors.
	configured.selectedPropSyncChains = nil

	// Count runtime endpoints while binding ordered partitions into projection identity.
	// Manifest validation guarantees numChips/devicesPerNode equals the HX extent's node count.
	partitionRequests := make([]lpxv1alpha1.PartitionRequest, len(configured.partitions))
	fields.field("v3-envelope-schema", []byte("dynamo.lpx.v3-capnp/v1"))
	agentReplicas := 0
	for index, partition := range configured.partitions {
		extent := slices.Clone(partition.hxExtent)
		partitionRequests[index] = newPartitionRequest(index, partition)
		partitionRequests[index].Extent = &extent
		agentReplicas += partition.effectiveNodeCount()
		fields.intField("ordered-compiler-id-index", int64(index))
		fields.uint32Field("ordered-compiler-id", uint32(partition.sourcePartitionID))
	}
	fields.field("allocation-metadata", allocationMetadata)
	// Bind the Cyborg runtime contract into hybrid projection identity.
	bindHybridRuntimeIO(fields, pipeline, configured.ioFPGACount, configured.ioFanoutFactor)

	return &component{
		configuredBuild:    configured,
		allocationMetadata: allocationMetadata,
		partitionRequests:  partitionRequests,
		connectors:         connectors,
		agentReplicas:      agentReplicas,
	}, nil
}

func projectV3PropSync(
	partitions []buildPartition,
	chains [][]int,
	pipeline Pipeline,
) (json.RawMessage, []lpxv1alpha1.PropSyncConnectorRequest, error) {
	edgePositions, err := validateSelectedPropSyncGraph(partitions, chains, "selected V3 prop-sync chain")
	if err != nil {
		return nil, nil, err
	}
	if pipeline != PipelineHybrid && len(edgePositions) != len(partitions)-1 {
		return nil, nil, fmt.Errorf("V3 LPU-only workloads require a complete adjacent prop-sync connector chain")
	}

	// Project each physical partition into the HX allocation metadata envelope.
	partitionInfo := make(map[string]any, len(partitions)+1)
	partitionInfo["num_partitions"] = len(partitions)
	for _, partition := range partitions {
		compilerID := uint32(partition.sourcePartitionID)
		partitionInfo[strconv.FormatUint(uint64(compilerID), 10)] = map[string]any{
			"device":     "lpu",
			"allocation": partition.hxExtent,
		}
	}

	// Project validated edges into runtime metadata and scheduler connector order.
	propSyncPairs := make([]any, 0)
	connectors := make([]lpxv1alpha1.PropSyncConnectorRequest, 0, len(edgePositions))
	for _, fromPosition := range edgePositions {
		source := partitions[fromPosition]
		destinationID := partitions[fromPosition+1].sourcePartitionID
		logicalConnections := make([]lpxv1alpha1.HxLogicalConnection, source.devicesPerNode)
		connections := make([][2]int64, source.devicesPerNode)
		// Connections leave from the source partition's final node.
		sourceOffset := int64(source.numChips - source.devicesPerNode)
		for logicalDevice := range logicalConnections {
			from := sourceOffset + int64(logicalDevice)
			logicalConnections[logicalDevice] = lpxv1alpha1.HxLogicalConnection{
				FromLogicalDevice: from,
				ToLogicalDevice:   int64(logicalDevice),
			}
			connections[logicalDevice] = [2]int64{from, int64(logicalDevice)}
		}
		acceptableLaneMultiplicities := []int64{4, 2, 1}
		propSyncPairs = append(propSyncPairs, map[string]any{
			"source_partition":    source.sourcePartitionID,
			"dest_partition":      destinationID,
			"connections":         connections,
			"num_supported_lanes": acceptableLaneMultiplicities,
		})
		connectors = append(connectors, lpxv1alpha1.PropSyncConnectorRequest{
			FromPartitionID: fmt.Sprintf("partition-%03d", fromPosition),
			ToPartitionID:   fmt.Sprintf("partition-%03d", fromPosition+1),
			Requirement: lpxv1alpha1.PropSyncConnectorRequirement{
				Kind:                         lpxv1alpha1.PropSyncConnectorKindHxPropSyncV1,
				Connections:                  &logicalConnections,
				AcceptableLaneMultiplicities: &acceptableLaneMultiplicities,
			},
		})
	}

	// Encode the same validated graph for the LPU runtime allocation contract.
	metadata, _ := json.Marshal(map[string]any{
		"arch":             "lp30",
		"topology":         "lyra",
		"metadata_version": 1,
		"partition_info":   partitionInfo,
		"prop_sync_info": map[string]any{
			"version":         1,
			"prop_sync_pairs": propSyncPairs,
		},
	})
	return metadata, connectors, nil
}
