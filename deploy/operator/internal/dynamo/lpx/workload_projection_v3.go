/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"fmt"
	"strconv"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
)

const (
	v3CompilerEnvelopeSchema = "dynamo.lpx.v3-capnp/v1"
	v3ProjectionVersion      = "v3-hx-capnp/v1"
	v3LPUDevice              = "lpu"
)

// projectV3Component derives the HX component shared by every model and writes
// its model-independent digest fields.
func projectV3Component(intent ModelProjectionInput, fields digestTranscript) (ModelProjection, error) {
	configured := *intent.BuildSnapshot.build
	allocationMetadata, connectors, err := projectV3PropSync(configured.Partitions, configured.SelectedPropSyncChains, intent.Pipeline)
	if err != nil {
		return ModelProjection{}, err
	}
	// Selected chains are represented by the allocation metadata and connectors.
	configured.SelectedPropSyncChains = nil

	// Count runtime endpoints while binding ordered partitions into projection identity.
	// Manifest validation guarantees NumChips/DevicesPerNode equals the HX extent's node count.
	fields.field("v3-envelope-schema", []byte(v3CompilerEnvelopeSchema))
	agentReplicas := 0
	for index, partition := range configured.Partitions {
		agentReplicas += partition.effectiveNodeCount()
		fields.intField("ordered-compiler-id-index", int64(index))
		fields.uint32Field("ordered-compiler-id", uint32(partition.SourcePartitionID))
	}
	fields.field("allocation-metadata", allocationMetadata)
	// Bind the Cyborg runtime contract into hybrid projection identity.
	bindHybridRuntimeIO(fields, intent.Pipeline, configured.IOFPGACount, configured.IOFanoutFactor)

	return ModelProjection{
		configuredBuild:    configured,
		allocationMetadata: allocationMetadata,
		partitions:         configured.Partitions,
		connectors:         connectors,
		agentReplicas:      agentReplicas,
	}, nil
}

func projectV3PropSync(
	partitions []BuildPartition,
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

	// Project each physical partition into the V3 allocation metadata envelope.
	partitionInfo := make(map[string]any, len(partitions)+1)
	partitionInfo["num_partitions"] = len(partitions)
	for _, partition := range partitions {
		compilerID := uint32(partition.SourcePartitionID)
		partitionInfo[strconv.FormatUint(uint64(compilerID), 10)] = map[string]any{
			"device":     v3LPUDevice,
			"allocation": partition.HXExtent,
		}
	}

	// Project validated edges into runtime metadata and scheduler connector order.
	propSyncPairs := make([]any, 0)
	connectors := make([]lpxv1alpha1.PropSyncConnectorRequest, 0, len(edgePositions))
	for _, fromPosition := range edgePositions {
		source := partitions[fromPosition]
		destinationID := partitions[fromPosition+1].SourcePartitionID
		logicalConnections := make([]lpxv1alpha1.HxLogicalConnection, source.DevicesPerNode)
		connections := make([][2]int64, source.DevicesPerNode)
		// Connections leave from the source partition's final node.
		sourceOffset := int64(source.NumChips - source.DevicesPerNode)
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
			"source_partition":    source.SourcePartitionID,
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
