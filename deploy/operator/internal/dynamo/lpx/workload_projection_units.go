/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"

	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
)

// Bump this domain when projection behavior changes for the same compiler bytes.
const unitsProjectionVersion = "lpu-units-node-local/v1"

func appendUnitModelProjections(dst []*ModelProjection, intent ModelProjectionInput) ([]*ModelProjection, error) {
	build := intent.Build
	hybrid := intent.Pipeline == PipelineHybrid

	// The LPU-only conductor can launch only one unit per model.
	if !hybrid && build.lpuUnits.Len() != 1 {
		return nil, fmt.Errorf("%w: LPU-only workloads require exactly one lpuUnits entry per model; this bundle has %d",
			ErrUnsupportedRuntime, build.lpuUnits.Len())
	}
	units, localPartitionIDs, err := selectRemoteUnits(intent.LocalPartitions, intent.Pipeline, build.lpuUnits)
	if err != nil {
		return nil, err
	}

	devicesPerNode := build.partitions[0].devicesPerNode
	if build.family == BuildFamilyXT && devicesPerNode != 8 {
		return nil, fmt.Errorf("%w: LP20 nodes projection requires devicesPerNode=8, got %d", ErrUnsupportedRuntime, devicesPerNode)
	}

	var (
		partitions    = make([]buildPartition, 0, len(units))
		connectors    = make([]lpxv1alpha1.PropSyncConnectorRequest, 0)
		agentReplicas = 0
	)
	for _, unit := range units {
		first := len(partitions)
		reservations, err := unit.Reservations()
		if err != nil {
			return nil, fmt.Errorf("reading unit reservations: %w", err)
		}
		for i := range reservations.Len() {
			partition, hosts, err := projectUnitReservation(reservations.At(i), build.family, devicesPerNode)
			if err != nil {
				return nil, err
			}
			partitions = append(partitions, partition)
			agentReplicas += hosts
		}
		rawConnectors, err := unit.Connectors()
		if err != nil {
			return nil, fmt.Errorf("reading unit connectors: %w", err)
		}
		for index := range rawConnectors.Len() {
			requirement, err := lpuConnectorFromDeploymentV1(rawConnectors.At(index), fmt.Sprintf("lpuUnits connector %d", index))
			if err != nil {
				return nil, err
			}
			connector := lpxv1alpha1.PropSyncConnectorRequest{
				FromPartitionID: fmt.Sprintf("partition-%03d", first+index),
				ToPartitionID:   fmt.Sprintf("partition-%03d", first+index+1),
				Requirement:     requirement,
			}
			connectors = append(connectors, connector)
		}
	}

	// The scheduler reads placement from partitions and connectors alone.
	allocationMetadata := json.RawMessage(`{}`)

	// The compiler content ID covers placement and runtime I/O. Only operator
	// inputs need separate fields in the coordination revision.
	for index, model := range intent.Models {
		transcript := digestTranscript{hash: sha256.New()}
		transcript.field("schema", []byte(unitsProjectionVersion))
		transcript.field("build-content-id", []byte(build.contentID))
		transcript.field("model", []byte(model))
		transcript.field("pipeline", []byte(intent.Pipeline))
		if len(localPartitionIDs) > 0 {
			encoded, _ := json.Marshal(localPartitionIDs)
			transcript.field("local-partition-ids", encoded)
		}

		dst = append(dst, &ModelProjection{
			digest:                 transcript.sum(),
			compilerSnapshotDigest: build.contentID,
			contractFormat:         build.contractFormat,
			runtimeBuildRef:        intent.RuntimeBuildRef,
			model:                  intent.Models[index],
			pipeline:               intent.Pipeline,
			configuredBuild:        *build,
			allocationMetadata:     allocationMetadata,
			partitions:             partitions,
			connectors:             connectors,
			agentReplicas:          agentReplicas,
			localPartitionIDs:      localPartitionIDs,
		})
	}
	return dst, nil
}

func projectUnitReservation(reservation deploymentcapnpv1.LpuReservation, family BuildFamily, devicesPerNode int) (buildPartition, int, error) {
	root, err := reservationRoot(reservation)
	if err != nil {
		return buildPartition{}, 0, err
	}
	partition := buildPartition{sourcePartitionID: root, devicesPerNode: devicesPerNode}
	hosts, extent, err := reservationSize(reservation, devicesPerNode)
	if err != nil {
		return buildPartition{}, 0, err
	}
	if family == BuildFamilyXT && extent != nil {
		return buildPartition{}, 0, fmt.Errorf("%w: LP20 reservations require size.nodes, not size.hx", ErrUnsupportedRuntime)
	}
	partition.numChips = hosts * devicesPerNode
	partition.hxExtent = extent
	if family == BuildFamilyHX {
		if len(partition.hxExtent) == 0 {
			if hosts != 1 || devicesPerNode != 16 {
				return buildPartition{}, 0, fmt.Errorf("%w: LP30 size.nodes requires one host with devicesPerNode=16; use a supported size.hx", ErrUnsupportedRuntime)
			}
			partition.hxExtent = []int64{16, 1, 1, 1}
		}
	}

	// The LPR CRD admits xtShape only from its closed enum (c8..c64, c96,
	// c128), so a host count without a registered shape has no request
	// encoding.
	if family == BuildFamilyXT {
		if _, err := xtShape(partition); err != nil {
			return buildPartition{}, 0, err
		}
	}
	return partition, hosts, nil
}
