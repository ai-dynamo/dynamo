/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"fmt"
	"slices"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
)

const unitsProjectionVersion = "lpu-units-node-local/v1"

// appendUnitModelProjections projects a build whose compiler recorded lpuUnits.
// The units supply reservations, sizes and connectors, so chains, packing and
// extents are not derived here. The runtime rows in configuredBuild and
// propSyncEdges reproduce what the legacy projection renders for the same
// bundle, so runtimes that read /configs keep working. intent.Build has units.
func appendUnitModelProjections(dst []*ModelProjection, intent ModelProjectionInput) ([]*ModelProjection, error) {
	build := intent.Build
	hybrid := intent.Pipeline == PipelineHybrid

	// An LPU-only conductor launches exactly one unit for each model. The legacy
	// projection accepts several LP20 partitions without a chain, but LPX then
	// rejects the incomplete LPU-only chain, so units refuse them here.
	if !hybrid && len(build.lpuUnits) != 1 {
		return nil, fmt.Errorf("%w: LPU-only workloads require exactly one lpuUnits entry per model; this bundle has %d",
			ErrUnsupportedRuntime, len(build.lpuUnits))
	}
	units, localPartitionIDs, err := selectRemoteUnits(intent.LocalPartitions, intent.Pipeline, build.lpuUnits)
	if err != nil {
		return nil, err
	}

	// Index the source partitions that reservations and runtime rows name.
	sourcePartitions := make(map[int]buildPartition, len(build.partitions))
	for _, partition := range build.partitions {
		sourcePartitions[partition.sourcePartitionID] = partition
	}
	devicesPerNode := build.partitions[0].devicesPerNode

	// Request one LPX partition per kept reservation, in Agent order, and join
	// each unit's adjacent reservations.
	var (
		partitions    = make([]buildPartition, 0, len(units))
		connectors    = make([]lpxv1alpha1.PropSyncConnectorRequest, 0)
		rows          = make([]buildPartition, 0, len(units))
		propSyncEdges = make([]int, 0)
		agentReplicas = 0
		connections   = 0
	)
	for _, unit := range units {
		first := len(partitions)
		unitHosts := 0
		for _, reservation := range unit.Reservations {
			partition := sourcePartitions[reservation.PartitionIDs[0]]
			partition.numChips = reservation.Hosts * devicesPerNode
			partition.hxExtent = slices.Clone(reservation.HXExtent)
			if build.family == BuildFamilyXT {
				if _, err := xtShape(partition); err != nil {
					return nil, err
				}
			}
			partitions = append(partitions, partition)
			unitHosts += reservation.Hosts
		}
		for index, requirement := range unit.Connectors {
			if requirement.Connections != nil {
				connections += len(*requirement.Connections)
				if connections > maxHXPropSyncConnections {
					return nil, fmt.Errorf("HX prop-sync graph exceeds %d logical connections", maxHXPropSyncConnections)
				}
			}
			connector := lpxv1alpha1.PropSyncConnectorRequest{
				FromPartitionID: fmt.Sprintf("partition-%03d", first+index),
				ToPartitionID:   fmt.Sprintf("partition-%03d", first+index+1),
			}
			requirement.DeepCopyInto(&connector.Requirement)
			connectors = append(connectors, connector)
			if build.family == BuildFamilyHX {
				propSyncEdges = append(propSyncEdges, first+index)
			}
		}
		agentReplicas += unitHosts

		// Legacy XT hybrid rows run each unit as one runtime partition; every
		// other runtime reads one row per partition, where packed stages, which
		// decoding guarantees are smaller than a node, each count one shared host.
		if hybrid && build.family == BuildFamilyXT {
			row := sourcePartitions[unit.root()]
			row.runtimeNodeCount = unitHosts
			rows = append(rows, row)
			continue
		}
		for _, reservation := range unit.Reservations {
			for _, id := range reservation.PartitionIDs {
				row := sourcePartitions[id]
				row.runtimeNodeCount = reservation.Hosts
				if len(reservation.PartitionIDs) > 1 {
					row.runtimeNodeCount = 1
				}
				rows = append(rows, row)
			}
		}
	}
	runtimeBuild := *build
	runtimeBuild.partitions = rows
	runtimeBuild.selectedPropSyncChains = nil

	// The scheduler reads placement from partitions and connectors alone.
	allocationMetadata := json.RawMessage(`{}`)

	// Bind the compiler's units and the local selection into each model's identity.
	encodedUnits, _ := json.Marshal(build.lpuUnits)
	transcripts := newModelProjectionTranscripts(intent, unitsProjectionVersion)
	for index := range transcripts {
		transcript := &transcripts[index]
		transcript.field("lpu-units", encodedUnits)
		if len(localPartitionIDs) > 0 {
			encoded, _ := json.Marshal(localPartitionIDs)
			transcript.field("local-partition-ids", encoded)
		}
		transcript.field("allocation-metadata", allocationMetadata)
		bindHybridRuntimeIO(transcript, intent.Pipeline, build.ioFPGACount, build.ioFanoutFactor)

		dst = append(dst, &ModelProjection{
			digest:                 transcript.sum(),
			compilerSnapshotDigest: build.contentID,
			contractFormat:         build.contractFormat,
			runtimeBuildRef:        intent.RuntimeBuildRef,
			model:                  intent.Models[index],
			pipeline:               intent.Pipeline,
			configuredBuild:        runtimeBuild,
			allocationMetadata:     allocationMetadata,
			partitions:             partitions,
			connectors:             connectors,
			propSyncEdges:          propSyncEdges,
			agentReplicas:          agentReplicas,
			localPartitionIDs:      localPartitionIDs,
		})
	}
	return dst, nil
}
