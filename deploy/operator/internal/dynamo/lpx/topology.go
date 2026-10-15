/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	corev1 "k8s.io/api/core/v1"
)

// buildTopology owns one hardware family's decoding, scheduler, and runtime
// behavior. Both compiler readers select it from explicit chip architecture;
// topology names remain opaque and only constrain homogeneous PropSync packing.
type buildTopology interface {
	String() string
	target() lpxv1alpha1.TargetFamily
	decodePartition(subject, name string, raw topologyGeometry) (buildPartition, error)
	normalizePartitions(filename string, partitions []buildPartition, partialSelection bool) error
	project(intent modelProjectionInput, fields digestTranscript) (Model, error)
	projectionVersion() string
	partitionRequest(index int, partition buildPartition) lpxv1alpha1.PartitionRequest
	workloadMode(pipeline Pipeline) lpxv1alpha1.WorkloadMode
	lpuResource() corev1.ResourceName
}

var (
	xtBuildTopology buildTopology = xtTopology{}
	hxBuildTopology buildTopology = hxTopology{}
)

// topologyGeometry erases only the generated schema type. Metadata is decoded
// lazily so XT artifacts do not start requiring HX-specific fields.
type topologyGeometry interface {
	NumChips() uint32
	DevicesPerNode() uint32
	HasTopologyMetadata() bool
	metadata() (lpuTopologyMetadata, error)
}

type topologyArtifact[M lpuTopologyMetadata] struct{ lpuPartitionArtifact[M] }

func (a topologyArtifact[M]) metadata() (lpuTopologyMetadata, error) { return a.TopologyMetadata() }
