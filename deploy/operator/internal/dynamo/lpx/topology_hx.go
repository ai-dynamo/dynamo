/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	corev1 "k8s.io/api/core/v1"
)

const (
	hxProjectionVersion = "hx-node-local/v1"
	hxLPUDevice         = "lpu"
)

func (hxTopology) project(intent modelProjectionInput, transcript digestTranscript) (Model, error) {
	runtimeBuild := *intent.Build
	manifestPartitions := runtimeBuild.partitions

	ioFPGACount, ioFanoutFactor := runtimeBuild.ioFPGACount, runtimeBuild.ioFanoutFactor

	edgePositions, err := validateSelectedPropSyncGraph(manifestPartitions, runtimeBuild.selectedPropSyncChains, "selected HX prop-sync chain")
	if err != nil {
		return Model{}, err
	}
	if intent.Pipeline != PipelineHybrid && len(edgePositions) != len(manifestPartitions)-1 {
		return Model{}, fmt.Errorf("HX LPU-only workloads require a complete adjacent prop-sync connector chain")
	}

	partitions := manifestPartitions
	if len(edgePositions) == len(manifestPartitions)-1 {
		partitions = packPropSyncPartitions(manifestPartitions)
	}
	if len(partitions) < len(manifestPartitions) {
		if intent.Pipeline == PipelineHybrid {
			if intent.LocalPartitions != nil {
				return Model{}, fmt.Errorf("%w: localPartitions cannot split packed HX prop-sync partitions, which require an LPU-only workload", ErrUnsupportedRuntime)
			}
			return Model{}, fmt.Errorf("packed HX prop-sync requires an LPU-only workload")
		}
		edgePositions = nil

		partition := &partitions[0]
		nodes, width := int64(partition.effectiveNodeCount()), int64(partition.devicesPerNode)
		switch {
		case nodes <= 8:
			partition.hxExtent = []int64{width, nodes, 1, 1}
		case nodes <= 16 && nodes%2 == 0:
			partition.hxExtent = []int64{width, nodes / 2, 2, 1}
		case nodes == 32:
			partition.hxExtent = []int64{width, 8, 2, 2}
		default:
			return Model{}, fmt.Errorf("packed chain chip count %d does not fit a supported HX extent", partition.numChips)
		}
	}

	// Keep GPU-local runtime partitions out of every LPU reservation and runtime record.
	// Selection requires a hybrid build, which is never packed, so partitions still
	// match the manifest order.
	remote, localPartitionIDs, err := selectRemotePartitions(intent.LocalPartitions, intent.Pipeline, intent.Build, partitions)
	if err != nil {
		return Model{}, err
	}
	if len(localPartitionIDs) > 0 {
		edgePositions = retainedPropSyncEdges(partitions, remote, edgePositions)
		partitions = remote
		runtimeBuild.partitions = remote
	}

	// Selected chains are represented by the allocation metadata and connectors below.
	runtimeBuild.selectedPropSyncChains = nil
	allocationMetadata, connectors, err := projectHXPropSync(partitions, edgePositions)
	if err != nil {
		return Model{}, err
	}

	// Record identity fields after validating the shared component geometry.
	if len(localPartitionIDs) > 0 {
		encoded, _ := json.Marshal(localPartitionIDs)
		transcript.field("local-partition-ids", encoded)
	}

	// Count runtime endpoints while binding ordered partitions into projection identity.
	agentReplicas := 0
	for index, partition := range partitions {
		agentReplicas += partition.effectiveNodeCount()
		transcript.intField("ordered-compiler-id-index", int64(index))
		transcript.uint32Field("ordered-compiler-id", uint32(partition.sourcePartitionID))
	}

	transcript.field("allocation-metadata", allocationMetadata)
	bindHybridRuntimeIO(transcript, intent.Pipeline, ioFPGACount, ioFanoutFactor)
	return Model{
		configuredBuild:    runtimeBuild,
		allocationMetadata: allocationMetadata,
		partitions:         partitions,
		connectors:         connectors,
		propSyncEdges:      edgePositions,
		agentReplicas:      agentReplicas,
		localPartitionIDs:  localPartitionIDs,
	}, nil
}

// retainedPropSyncEdges rebases selected edges from positions in partitions to
// positions in remote, which holds a build-ordered subset of partitions. It keeps
// declaration order and drops every edge with a GPU-local endpoint. A selected
// chain moves to the GPU as one runtime partition, so both endpoints of an edge
// share one placement and a retained edge joins adjacent retained positions.
func retainedPropSyncEdges(partitions, remote []buildPartition, edgePositions []int) []int {
	retainedOrdinal := make(map[int]int, len(remote))
	for ordinal, partition := range remote {
		retainedOrdinal[partition.sourcePartitionID] = ordinal
	}
	retained := make([]int, 0, len(edgePositions))
	for _, position := range edgePositions {
		from, fromRetained := retainedOrdinal[partitions[position].sourcePartitionID]
		_, toRetained := retainedOrdinal[partitions[position+1].sourcePartitionID]
		if fromRetained && toRetained {
			retained = append(retained, from)
		}
	}
	return retained
}

// Each edgePositions entry is a source index whose destination is position+1.
func projectHXPropSync(partitions []buildPartition, edgePositions []int) (json.RawMessage, []lpxv1alpha1.PropSyncConnectorRequest, error) {
	// Project each physical partition into the HX allocation metadata envelope.
	partitionInfo := make(map[string]any, len(partitions)+1)
	partitionInfo["num_partitions"] = len(partitions)
	for _, partition := range partitions {
		compilerID := uint32(partition.sourcePartitionID)
		partitionInfo[strconv.FormatUint(uint64(compilerID), 10)] = map[string]any{
			"device":     hxLPUDevice,
			"allocation": partition.hxExtent,
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
		if source.devicesPerNode > remainingConnections {
			return nil, nil, fmt.Errorf("HX prop-sync graph exceeds %d logical connections", maxConnections)
		}
		remainingConnections -= source.devicesPerNode

		logicalConnections := make([]lpxv1alpha1.HxLogicalConnection, source.devicesPerNode)
		connections := make([][2]int64, source.devicesPerNode)
		sourceOffset := int64((source.effectiveNodeCount() - 1) * source.devicesPerNode)
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
			"dest_partition":      destination.sourcePartitionID,
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

func (hxTopology) decodePartition(subject, topology string, raw topologyGeometry) (buildPartition, error) {
	devicesPerNode, err := positiveManifestUInt32ToInt(subject+" devicesPerNode", raw.DevicesPerNode())
	if err != nil {
		return buildPartition{}, err
	}
	partition := buildPartition{topology: topology, numChips: int(raw.NumChips()), devicesPerNode: devicesPerNode}
	if !raw.HasTopologyMetadata() {
		chips := partition.numChips
		if chips <= 0 || chips > devicesPerNode || devicesPerNode%chips != 0 {
			return buildPartition{}, fmt.Errorf("V3 %s without topologyMetadata requires a one-node stage whose chip count divides %d", subject, devicesPerNode)
		}
		partition.hxExtent = []int64{int64(devicesPerNode), 1, 1, 1}
		return partition, nil
	}

	// Decode HX metadata directly and report read errors at their source.
	metadata, err := raw.metadata()
	if err != nil {
		return buildPartition{}, fmt.Errorf("reading V3 %s topologyMetadata: %w", subject, err)
	}
	family, err := metadata.TopologyFamily()
	if err != nil {
		return buildPartition{}, fmt.Errorf("reading V3 %s topologyMetadata.topologyFamily: %w", subject, err)
	}
	if strings.TrimSpace(family) != hxTopologyFamily {
		return buildPartition{}, fmt.Errorf("V3 %s topologyMetadata.topologyFamily = %q, want %q", subject, strings.TrimSpace(family), hxTopologyFamily)
	}
	shape, err := metadata.PartitionShape()
	if err != nil {
		return buildPartition{}, fmt.Errorf("reading V3 %s topologyMetadata.partitionShape: %w", subject, err)
	}
	extent := make([]int64, shape.Len())
	for index := range extent {
		extent[index] = int64(shape.At(index))
	}

	// A partial-node shape describes its logical chips; physical HX reservations use full nodes.
	subnode := len(extent) == 4 && extent[0] > 0 && extent[0] < int64(devicesPerNode) &&
		int64(devicesPerNode)%extent[0] == 0 && extent[1] == 1 && extent[2] == 1 && extent[3] == 1
	if !subnode && (len(extent) != 4 || extent[0] != int64(devicesPerNode) || extent[1] < 1 || extent[1] > 8 ||
		(!((extent[2] == 1 || extent[2] == 2) && extent[3] == 1) && !(extent[1] == 8 && extent[2] == 2 && extent[3] == 2))) {
		return buildPartition{}, fmt.Errorf("V3 %s has unsupported HX extent %v", subject, extent)
	}

	// Match the source chip count before rounding its scheduler reservation up to a whole node.
	count := extent[0] * extent[1] * extent[2] * extent[3]
	if count != int64(raw.NumChips()) {
		return buildPartition{}, fmt.Errorf("V3 %s topologyMetadata.partitionShape contains %d chips, want numChips %d", subject, count, raw.NumChips())
	}
	extent[0] = int64(devicesPerNode)
	partition.hxExtent = extent
	return partition, nil
}

// hxTopology describes HX16x8x2x3 devices, called V3 by the scheduler wire API.
type hxTopology struct{}

func (hxTopology) String() string                   { return string(lpxv1alpha1.TargetFamilyHx16x8x2x3) }
func (hxTopology) target() lpxv1alpha1.TargetFamily { return lpxv1alpha1.TargetFamilyHx16x8x2x3 }
func (hxTopology) lpuResource() corev1.ResourceName { return v3LPUResourceName }
func (hxTopology) projectionVersion() string        { return hxProjectionVersion }
func (hxTopology) workloadMode(pipeline Pipeline) lpxv1alpha1.WorkloadMode {
	if pipeline == PipelineHybrid {
		return lpxv1alpha1.WorkloadModeV3HxStrictHybrid
	}
	return lpxv1alpha1.WorkloadModeV3HxLPUOnly
}
func (hxTopology) normalizePartitions(filename string, _ []buildPartition, partialSelection bool) error {
	if partialSelection {
		return fmt.Errorf("V3 %s partSelect builds are not supported", filename)
	}
	return nil
}
func (hxTopology) partitionRequest(index int, partition buildPartition) lpxv1alpha1.PartitionRequest {
	request := newPartitionRequest(index, partition)
	extent := append([]int64(nil), partition.hxExtent...)
	request.Extent = &extent
	return request
}
