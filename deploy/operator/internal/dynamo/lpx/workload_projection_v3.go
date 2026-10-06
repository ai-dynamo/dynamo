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

func appendV3ModelProjections(dst []*ModelProjection, intent ModelProjectionInput) ([]*ModelProjection, error) {
	runtimeBuild := *intent.BuildSnapshot.build
	manifestPartitions := runtimeBuild.Partitions

	ioFPGACount, ioFanoutFactor := runtimeBuild.IOFPGACount, runtimeBuild.IOFanoutFactor

	edgePositions, err := validateSelectedPropSyncGraph(manifestPartitions, runtimeBuild.SelectedPropSyncChains, "selected V3 prop-sync chain")
	if err != nil {
		return nil, err
	}
	if intent.Pipeline != PipelineLPX && len(edgePositions) != len(manifestPartitions)-1 {
		return nil, fmt.Errorf("V3 LPU-only workloads require a complete adjacent prop-sync connector chain")
	}

	partitions := manifestPartitions
	if len(edgePositions) == len(manifestPartitions)-1 {
		partitions = packPropSyncPartitions(manifestPartitions)
	}
	if len(partitions) < len(manifestPartitions) {
		if intent.Pipeline == PipelineLPX {
			if intent.LocalPartitions != nil {
				return nil, fmt.Errorf("%w: localPartitions cannot split packed HX prop-sync partitions, which require an LPU-only workload", ErrUnsupportedRuntime)
			}
			return nil, fmt.Errorf("packed HX prop-sync requires an LPU-only workload")
		}
		edgePositions = nil

		partition := &partitions[0]
		nodes, width := int64(partition.effectiveNodeCount()), int64(partition.DevicesPerNode)
		switch {
		case nodes <= 8:
			partition.HXExtent = []int64{width, nodes, 1, 1}
		case nodes <= 16 && nodes%2 == 0:
			partition.HXExtent = []int64{width, nodes / 2, 2, 1}
		case nodes == 32:
			partition.HXExtent = []int64{width, 8, 2, 2}
		default:
			return nil, fmt.Errorf("packed chain chip count %d does not fit a supported HX extent", partition.Topology.ChipCount)
		}
	}

	// Keep GPU-local runtime partitions out of every LPU reservation and runtime record.
	// Selection requires a hybrid build, which is never packed, so partitions still
	// match the manifest order.
	remote, localPartitionIDs, err := selectRemotePartitions(intent.LocalPartitions, intent.Pipeline, intent.BuildSnapshot.build, partitions)
	if err != nil {
		return nil, err
	}
	if len(localPartitionIDs) > 0 {
		edgePositions = retainedPropSyncEdges(partitions, remote, edgePositions)
		partitions = remote
		runtimeBuild.Partitions = remote
	}

	// Selected chains are represented by the allocation metadata and connectors below.
	runtimeBuild.SelectedPropSyncChains = nil
	allocationMetadata, connectors, err := projectV3PropSync(partitions, edgePositions)
	if err != nil {
		return nil, err
	}

	// Initialize independent model hashes after validating the shared component geometry.
	transcripts := newModelProjectionTranscripts(intent, v3ProjectionVersion)
	for index := range transcripts {
		transcripts[index].field("v3-envelope-schema", []byte(v3CompilerEnvelopeSchema))
	}
	if len(localPartitionIDs) > 0 {
		encoded, _ := json.Marshal(localPartitionIDs)
		for index := range transcripts {
			transcripts[index].field("local-partition-ids", encoded)
		}
	}

	// Count runtime endpoints while binding ordered partitions into projection identity.
	agentReplicas := 0
	for index, partition := range partitions {
		agentReplicas += partition.effectiveNodeCount()
		for modelIndex := range transcripts {
			transcripts[modelIndex].intField("ordered-compiler-id-index", int64(index))
			transcripts[modelIndex].uint32Field("ordered-compiler-id", uint32(partition.SourcePartitionID))
		}
	}

	// Publish distinct logical identities backed by the component's immutable configuration.
	for index := range transcripts {
		transcript := &transcripts[index]
		transcript.field("allocation-metadata", allocationMetadata)
		// Bind the Cyborg runtime contract into hybrid projection identity.
		bindHybridRuntimeIO(transcript, intent.Pipeline, ioFPGACount, ioFanoutFactor)

		dst = append(dst, &ModelProjection{
			digest:                 transcript.sum(),
			compilerSnapshotDigest: intent.BuildSnapshot.contentID,
			runtimeBuildRef:        intent.RuntimeBuildRef,
			model:                  intent.Models[index],
			pipeline:               intent.Pipeline,
			configuredBuild:        runtimeBuild,
			allocationMetadata:     allocationMetadata,
			partitions:             partitions,
			connectors:             connectors,
			agentReplicas:          agentReplicas,
			localPartitionIDs:      localPartitionIDs,
		})
	}
	return dst, nil
}

// retainedPropSyncEdges rebases selected edges from positions in partitions to
// positions in remote, which holds a build-ordered subset of partitions. It keeps
// declaration order and drops every edge with a GPU-local endpoint. A selected
// chain moves to the GPU as one runtime partition, so both endpoints of an edge
// share one placement and a retained edge joins adjacent retained positions.
func retainedPropSyncEdges(partitions, remote []BuildPartition, edgePositions []int) []int {
	retainedOrdinal := make(map[int]int, len(remote))
	for ordinal, partition := range remote {
		retainedOrdinal[partition.SourcePartitionID] = ordinal
	}
	retained := make([]int, 0, len(edgePositions))
	for _, position := range edgePositions {
		from, fromRetained := retainedOrdinal[partitions[position].SourcePartitionID]
		_, toRetained := retainedOrdinal[partitions[position+1].SourcePartitionID]
		if fromRetained && toRetained {
			retained = append(retained, from)
		}
	}
	return retained
}

// Each edgePositions entry is a source index whose destination is position+1.
func projectV3PropSync(partitions []BuildPartition, edgePositions []int) (json.RawMessage, []lpxv1alpha1.PropSyncConnectorRequest, error) {
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
	const maxConnections = 8192
	remainingConnections := maxConnections
	propSyncPairs := make([]any, 0)
	connectors := make([]lpxv1alpha1.PropSyncConnectorRequest, 0, len(edgePositions))
	for _, fromPosition := range edgePositions {
		source := partitions[fromPosition]
		destination := partitions[fromPosition+1]
		if source.DevicesPerNode > remainingConnections {
			return nil, nil, fmt.Errorf("HX prop-sync graph exceeds %d logical connections", maxConnections)
		}
		remainingConnections -= source.DevicesPerNode
		if capacity := destination.effectiveNodeCount() * destination.DevicesPerNode; source.DevicesPerNode > capacity {
			return nil, nil, fmt.Errorf("HX prop-sync destination partition %d has %d devices, need %d", destination.SourcePartitionID, capacity, source.DevicesPerNode)
		}

		logicalConnections := make([]lpxv1alpha1.HxLogicalConnection, source.DevicesPerNode)
		connections := make([][2]int64, source.DevicesPerNode)
		sourceOffset := int64((source.effectiveNodeCount() - 1) * source.DevicesPerNode)
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
			"dest_partition":      destination.SourcePartitionID,
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
