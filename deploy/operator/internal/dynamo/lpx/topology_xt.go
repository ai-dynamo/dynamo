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
	corev1 "k8s.io/api/core/v1"
)

const xtProjectionVersion = "xt-node-local/v1"

//nolint:gocyclo // XT projection validates one complete transformation.
func (xtTopology) project(intent modelProjectionInput, transcript digestTranscript) (Model, error) {
	configured := *intent.Build
	usesResolvedRuntime := intent.Pipeline == PipelineSingle ||
		intent.Pipeline == PipelineSpecDecode
	ioFPGACount, ioFanoutFactor := configured.ioFPGACount, configured.ioFanoutFactor
	connectorBuild := intent.Build

	// Apply the selected chain and CPU embedding placement before deriving scheduler requests.
	if usesResolvedRuntime {
		if err := configured.selectXTRuntimePartitions(); err != nil {
			return Model{}, fmt.Errorf("resolving configured XT build: %w", err)
		}
	}

	allocationMetadata := json.RawMessage(`{}`)

	// Combine packed physical reservations without changing the runtime's logical stages.
	partitions := configured.partitions
	if usesResolvedRuntime && len(connectorBuild.selectedPropSyncChains) != 0 {
		partitions = packPropSyncPartitions(partitions)
	}

	transcript.field("input-embeddings-on-gpu", []byte{1})
	bindHybridRuntimeIO(transcript, intent.Pipeline, ioFPGACount, ioFanoutFactor)

	partitions, localPartitionIDs, err := selectRemotePartitions(intent.LocalPartitions, intent.Pipeline, &configured, partitions)
	if err != nil {
		return Model{}, err
	}
	if len(localPartitionIDs) > 0 {
		encoded, _ := json.Marshal(localPartitionIDs)
		transcript.field("local-partition-ids", encoded)
		configured.partitions = partitions
	}

	// Bind validated physical partitions into projection identity while counting runtime endpoints.
	agentReplicas := 0
	for index, partition := range partitions {
		compilerID := uint32(partition.sourcePartitionID)
		shape, shapeErr := xtShape(partition)
		if shapeErr != nil {
			return Model{}, shapeErr
		}
		endpoints := int64(partition.effectiveNodeCount())
		agentReplicas += int(endpoints)
		transcript.uint32Field("compiler-partition-id", compilerID)
		transcript.uint32Field("model-partition-id", uint32(index))
		transcript.intField("endpoint-count", endpoints)
		transcript.field("xt-shape", []byte(shape))
	}

	connectors, err := xtConnectors(connectorBuild, partitions)
	if err != nil {
		return Model{}, err
	}

	// Preserve physical scheduler partitions while collapsing selected chains only in LPU runtime state.
	if intent.Pipeline == PipelineHybrid && len(configured.selectedPropSyncChains) != 0 {
		runtimeChainByRoot := make(map[int][]int, len(connectorBuild.selectedPropSyncChains))
		for _, chain := range connectorBuild.selectedPropSyncChains {
			runtimeChainByRoot[chain[0]] = chain
		}
		collapsed := make([]buildPartition, 0, len(partitions))
		for partitionIndex := 0; partitionIndex < len(partitions); {
			partition := partitions[partitionIndex]
			chain, selected := runtimeChainByRoot[partition.sourcePartitionID]
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
		configured.partitions = collapsed
		configured.selectedPropSyncChains = nil
	}

	// Encode shared connectors once.
	for _, connector := range connectors {
		encoded, _ := json.Marshal(connector)
		transcript.field("connector", encoded)
	}

	transcript.field("allocation-metadata", allocationMetadata)
	return Model{
		configuredBuild:    configured,
		allocationMetadata: allocationMetadata,
		partitions:         partitions,
		connectors:         connectors,
		agentReplicas:      agentReplicas,
		localPartitionIDs:  localPartitionIDs,
	}, nil
}

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

// xtConnectors validates source chains before emitting edges between reservations.
// build is normalized and nonnil; partitions holds the retained reservations in
// build order. An edge is emitted only when both endpoints are retained, so a
// packed chain's non-root members and GPU-local partitions produce no edge.
func xtConnectors(
	build *Build,
	partitions []buildPartition,
) ([]lpxv1alpha1.PropSyncConnectorRequest, error) {
	// Only compiler-selected relationships impose placement constraints.
	if len(build.selectedPropSyncChains) == 0 {
		return []lpxv1alpha1.PropSyncConnectorRequest{}, nil
	}

	// Validate explicit chains before ordering their scheduler edges.
	edgePositions, err := validateSelectedPropSyncGraph(
		build.partitions,
		build.selectedPropSyncChains,
		"selected prop-sync chain",
	)
	if err != nil {
		return nil, err
	}

	// Scheduler output follows physical order, not chain declaration order.
	slices.Sort(edgePositions)

	retainedOrdinal := make(map[int]int, len(partitions))
	for ordinal, partition := range partitions {
		retainedOrdinal[partition.sourcePartitionID] = ordinal
	}
	connectors := make([]lpxv1alpha1.PropSyncConnectorRequest, 0, len(edgePositions))
	for _, position := range edgePositions {
		from, fromRetained := retainedOrdinal[build.partitions[position].sourcePartitionID]
		to, toRetained := retainedOrdinal[build.partitions[position+1].sourcePartitionID]
		if !fromRetained || !toRetained {
			continue
		}
		offset := int64(0)
		connectors = append(connectors, lpxv1alpha1.PropSyncConnectorRequest{
			FromPartitionID: fmt.Sprintf("partition-%03d", from),
			ToPartitionID:   fmt.Sprintf("partition-%03d", to),
			Requirement: lpxv1alpha1.PropSyncConnectorRequirement{
				Kind:                    lpxv1alpha1.PropSyncConnectorKindXt8888Gap,
				MaxInterPartitionOffset: &offset,
			},
		})
	}
	return connectors, nil
}

func (xtTopology) decodePartition(subject, topology string, raw topologyGeometry) (buildPartition, error) {
	// Derive XT geometry from numeric manifest fields, never the topology name.
	numChips, err := positiveManifestUInt32ToInt(subject+" numChips", raw.NumChips())
	if err != nil {
		return buildPartition{}, err
	}
	devicesPerNode, err := positiveManifestUInt32ToInt(subject+" devicesPerNode", raw.DevicesPerNode())
	if err != nil {
		return buildPartition{}, err
	}
	if numChips > devicesPerNode && numChips%devicesPerNode != 0 {
		return buildPartition{}, fmt.Errorf("%s has %d chips, not divisible by %d LPU devices per node", subject, numChips, devicesPerNode)
	}
	return buildPartition{topology: topology, numChips: numChips, devicesPerNode: devicesPerNode}, nil
}

// xtTopology describes XT8888 devices, called V2 by the scheduler wire API.
type xtTopology struct{}

func (xtTopology) String() string                   { return string(lpxv1alpha1.TargetFamilyXt8888) }
func (xtTopology) target() lpxv1alpha1.TargetFamily { return lpxv1alpha1.TargetFamilyXt8888 }
func (xtTopology) lpuResource() corev1.ResourceName { return v2LPUResourceName }
func (xtTopology) projectionVersion() string        { return xtProjectionVersion }
func (xtTopology) workloadMode(pipeline Pipeline) lpxv1alpha1.WorkloadMode {
	if pipeline == PipelineHybrid {
		return lpxv1alpha1.WorkloadModeV2StrictHybrid
	}
	return lpxv1alpha1.WorkloadModeV2LPUOnly
}
func (xtTopology) normalizePartitions(_ string, partitions []buildPartition, _ bool) error {
	sort.Slice(partitions, func(i, j int) bool { return partitions[i].sourcePartitionID < partitions[j].sourcePartitionID })
	return nil
}
func (xtTopology) partitionRequest(index int, partition buildPartition) lpxv1alpha1.PartitionRequest {
	request := newPartitionRequest(index, partition)
	shape, _ := xtShape(partition)
	request.XtShape = &shape
	return request
}
