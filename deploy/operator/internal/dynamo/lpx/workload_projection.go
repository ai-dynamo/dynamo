/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"

	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
)

const (
	// Bump this domain when projection behavior changes for the same compiler bytes.
	unitsProjectionVersion = "lpu-units-node-local/v1"

	maxLPXPartitions = 256
)

// projectModels returns one component's immutable model projections.
// intent.Build contains a normalized, non-nil build; Models is nonempty
// and Pipeline is selected by the validated resolver. Inputs are not mutated.
func projectModels(intent ModelProjectionInput) ([]*ModelProjection, error) {
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

	// Each reservation is one scheduler partition; connectors join a unit's partitions in order.
	var (
		partitions    = make([]lpxv1alpha1.PartitionRequest, 0, len(units))
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
			partition, hosts, err := projectUnitReservation(reservations.At(i), build.devicesPerNode)
			if err != nil {
				return nil, err
			}
			partition.ID, partition.Ordinal = partitionID(len(partitions)), int64(len(partitions))
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
			connectors = append(connectors, lpxv1alpha1.PropSyncConnectorRequest{
				FromPartitionID: partitionID(first + index),
				ToPartitionID:   partitionID(first + index + 1),
				Requirement:     requirement,
			})
		}
	}

	// Bound the request and the local selection before runtime expansion. Only a local selection may leave no remote partition.
	if bounded := len(partitions) + len(localPartitionIDs); bounded == 0 || bounded > maxLPXPartitions {
		return nil, fmt.Errorf("LPX projection has %d partitions, limit is 1..%d", bounded, maxLPXPartitions)
	}

	// The Agent count becomes the int32 replica count of a PodClique.
	if agentReplicas > math.MaxInt32 {
		return nil, fmt.Errorf("LPX projection has %d Agents, limit is %d", agentReplicas, math.MaxInt32)
	}

	// Only a hybrid build's Cyborg conductor consumes a checkpoint.
	var checkpoint *Checkpoint
	if hybrid {
		checkpoint = build.checkpoint
	}

	// The compiler content ID covers placement and runtime I/O. Only operator
	// inputs need separate fields in the coordination revision.
	projections := make([]*ModelProjection, 0, len(intent.Models))
	for _, model := range intent.Models {
		transcript := digestTranscript{hash: sha256.New()}
		transcript.field("schema", []byte(unitsProjectionVersion))
		transcript.field("build-content-id", []byte(build.contentID))
		transcript.field("model", []byte(model))
		transcript.field("pipeline", []byte(intent.Pipeline))
		if len(localPartitionIDs) > 0 {
			encoded, _ := json.Marshal(localPartitionIDs)
			transcript.field("local-partition-ids", encoded)
		}

		projections = append(projections, &ModelProjection{
			digest:            transcript.sum(),
			runtimeBuildRef:   intent.RuntimeBuildRef,
			model:             model,
			stage:             intent.Stage,
			pipeline:          intent.Pipeline,
			configuredBuild:   *build,
			partitions:        partitions,
			connectors:        connectors,
			agentReplicas:     agentReplicas,
			localPartitionIDs: localPartitionIDs,
			checkpoint:        checkpoint,
		})
	}
	return projections, nil
}

// partitionID names the scheduler partition at index within one model's request.
func partitionID(index int) string {
	return fmt.Sprintf("partition-%03d", index)
}

// projectUnitReservation returns the scheduler shape of one reservation and its host count.
// The caller assigns the partition identity.
func projectUnitReservation(reservation deploymentcapnpv1.LpuReservation, devicesPerNode int) (lpxv1alpha1.PartitionRequest, int, error) {
	size := reservation.Size()
	switch size.Which() {
	case deploymentcapnpv1.LpuReservation_size_Which_nodes:
		nodes := size.Nodes()
		if nodes == 0 || nodes > math.MaxInt32 {
			return lpxv1alpha1.PartitionRequest{}, 0, fmt.Errorf("lpuUnits size.nodes must be in [1, %d], got %d", math.MaxInt32, nodes)
		}
		shape := lpxv1alpha1.Xt8888PartitionShape(fmt.Sprintf("c%d", int(nodes)*devicesPerNode))
		return lpxv1alpha1.PartitionRequest{XtShape: &shape}, int(nodes), nil
	case deploymentcapnpv1.LpuReservation_size_Which_hx:
		hx, err := size.Hx()
		if err != nil {
			return lpxv1alpha1.PartitionRequest{}, 0, fmt.Errorf("reading lpuUnits size.hx: %w", err)
		}
		trays, hemispheres, racks := int64(hx.TraysPerHemisphere()), int64(hx.Hemispheres()), int64(hx.Racks())
		extent := []int64{int64(devicesPerNode), trays, hemispheres, racks}
		return lpxv1alpha1.PartitionRequest{Extent: &extent}, int(trays * hemispheres * racks), nil
	default:
		return lpxv1alpha1.PartitionRequest{}, 0, fmt.Errorf("lpuUnits size has unsupported variant %d", size.Which())
	}
}
