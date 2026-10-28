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

// appendUnitModelProjections projects a build from its compiler-owned lpuUnits.
// The units supply reservations, sizes and connectors, so chains, packing and
// extents are not derived here.
func appendUnitModelProjections(dst []*ModelProjection, intent ModelProjectionInput) ([]*ModelProjection, error) {
	build := intent.Build
	hybrid := intent.Pipeline == PipelineHybrid

	// An LPU-only conductor launches exactly one unit for each model; LPX rejects
	// an LPU-only request whose partitions do not form one chain.
	if !hybrid && len(build.lpuUnits) != 1 {
		return nil, fmt.Errorf("%w: LPU-only workloads require exactly one lpuUnits entry per model; this bundle has %d",
			ErrUnsupportedRuntime, len(build.lpuUnits))
	}
	units, localPartitionIDs, err := selectRemoteUnits(intent.LocalPartitions, intent.Pipeline, build.lpuUnits)
	if err != nil {
		return nil, err
	}

	// Index the source partitions that reservations name.
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
		agentReplicas = 0
	)
	for _, unit := range units {
		first := len(partitions)
		for _, reservation := range unit.Reservations {
			partition := sourcePartitions[reservation.PartitionIDs[0]]
			partition.numChips = reservation.Hosts * devicesPerNode
			partition.hxExtent = slices.Clone(reservation.HXExtent)

			// The LPR CRD admits xtShape only from its closed enum (c8..c64, c96,
			// c128), so a host count without a registered shape has no request
			// encoding. The enum also caps an LP20 reservation at 128 devices,
			// which keeps 256 reservations' Agent replicas within int32.
			if build.family == BuildFamilyXT {
				if _, err := xtShape(partition); err != nil {
					return nil, err
				}
			}
			partitions = append(partitions, partition)
			agentReplicas += reservation.Hosts
		}
		for index, requirement := range unit.Connectors {
			connector := lpxv1alpha1.PropSyncConnectorRequest{
				FromPartitionID: fmt.Sprintf("partition-%03d", first+index),
				ToPartitionID:   fmt.Sprintf("partition-%03d", first+index+1),
			}
			requirement.DeepCopyInto(&connector.Requirement)
			connectors = append(connectors, connector)
		}
	}

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
		bindHybridRuntimeIO(transcript, intent.Pipeline, build.ioFPGACount, build.ioFanoutFactor)

		dst = append(dst, &ModelProjection{
			digest:                 transcript.sum(),
			compilerSnapshotDigest: build.contentID,
			runtimeBuildRef:        intent.RuntimeBuildRef,
			model:                  intent.Models[index],
			pipeline:               intent.Pipeline,
			configuredBuild:        *build,
			partitions:             partitions,
			connectors:             connectors,
			agentReplicas:          agentReplicas,
			localPartitionIDs:      localPartitionIDs,
		})
	}
	return dst, nil
}

// xtShape names the registered XT shape of an LP20 request partition.
func xtShape(partition buildPartition) (lpxv1alpha1.Xt8888PartitionShape, error) {
	// These chip capacities name registered XT shapes; node width comes from the manifest.
	chips := max(partition.numChips, partition.devicesPerNode)
	switch chips {
	case 8, 16, 24, 32, 40, 48, 56, 64, 96, 128:
		return lpxv1alpha1.Xt8888PartitionShape(fmt.Sprintf("c%d", chips)), nil
	default:
		return "", fmt.Errorf("partition %d has unregistered XT shape c%d", uint32(partition.sourcePartitionID), chips)
	}
}
