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
	v3MaxSWADKVCBlocksDraft  = "max_swa_dkvc_blocks_draft"
	v3LPUDevice              = "lpu"
	v3HXLogicalDeviceCount   = 16
)

func appendV3ModelProjections(dst []*ModelProjection, intent ModelProjectionInput, modelSettings map[string]any) ([]*ModelProjection, error) {
	runtimeBuild := *intent.BuildSnapshot.build
	manifestPartitions := runtimeBuild.Partitions
	selectedPropSyncChains := runtimeBuild.SelectedPropSyncChains

	ioFPGACount, ioFanoutFactor := runtimeBuild.IOFPGACount, runtimeBuild.IOFanoutFactor
	// V3 topology and selected prop-sync chains are projected below. Do not feed
	// them through the runtime-settings consumer: the runtime projection
	// intentionally contains no artifact partitions.
	runtimeBuild.Partitions = nil
	runtimeBuild.SelectedPropSyncChains = nil
	if err := resolveBuildSettings(&runtimeBuild, modelSettings); err != nil {
		return nil, fmt.Errorf("resolving configured V3 runtime build: %w", err)
	}
	propSyncEnabled, ok := runtimeBuild.runtimeSettings["prop_sync"].(bool)
	if !ok {
		return nil, fmt.Errorf("configured V3 runtime settings.prop_sync must be a boolean")
	}
	if runtimeBuild.CompilationMode == BuildCompilationModeLPUOnly {
		if len(selectedPropSyncChains) > 0 && !propSyncEnabled {
			return nil, fmt.Errorf("V3 manifest-selected prop-sync chains require settings.prop_sync=true")
		}
		if len(selectedPropSyncChains) == 0 && propSyncEnabled {
			return nil, fmt.Errorf("V3 settings.prop_sync=true requires a manifest-selected prop-sync chain")
		}
	}

	allocationMetadata, connectors, err := projectV3PropSync(manifestPartitions, selectedPropSyncChains, intent.Pipeline)
	if err != nil {
		return nil, err
	}

	// Initialize independent model hashes after validating the shared component geometry.
	transcripts := newModelProjectionTranscripts(intent, v3ProjectionVersion)
	for index := range transcripts {
		transcripts[index].field("v3-envelope-schema", []byte(v3CompilerEnvelopeSchema))
	}

	// Count runtime endpoints while binding ordered partitions into projection identity.
	agentReplicas := 0
	for index, partition := range manifestPartitions {
		agentReplicas += int(partition.HXExtent[1] * partition.HXExtent[2] * partition.HXExtent[3])
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
			digest:             transcript.sum(),
			runtimeBuildRef:    intent.RuntimeBuildRef,
			model:              intent.Models[index],
			pipeline:           intent.Pipeline,
			configuredBuild:    runtimeBuild,
			allocationMetadata: allocationMetadata,
			partitions:         manifestPartitions,
			connectors:         connectors,
			agentReplicas:      agentReplicas,
		})
	}
	return dst, nil
}

func projectV3PropSync(
	partitions []BuildPartition,
	chains [][]int,
	pipeline Pipeline,
) (json.RawMessage, []lpxv1alpha1.PropSyncConnectorRequest, error) {
	edgePositions, err := validateSelectedPropSyncGraph(partitions, chains, "selected V3 prop-sync chain", false)
	if err != nil {
		return nil, nil, err
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
		endpointCount := source.HXExtent[1] * source.HXExtent[2] * source.HXExtent[3]
		logicalConnections := make([]lpxv1alpha1.HxLogicalConnection, v3HXLogicalDeviceCount)
		connections := make([][2]int64, v3HXLogicalDeviceCount)
		sourceOffset := source.HXExtent[0]*endpointCount - v3HXLogicalDeviceCount
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
	if pipeline != PipelineLPX && len(connectors) != len(partitions)-1 {
		return nil, nil, fmt.Errorf("V3 LPU-only workloads require a complete adjacent prop-sync connector chain")
	}
	return metadata, connectors, nil
}
