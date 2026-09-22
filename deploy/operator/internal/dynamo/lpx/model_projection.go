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
	"k8s.io/apimachinery/pkg/runtime"
)

const (
	// WorkloadDigestAnnotation records the immutable Dynamo workload projection digest on rendered objects.
	WorkloadDigestAnnotation = "scheduling.lpu.nvidia.com/dynamo-workload-digest"
	// DGDGenerationAnnotation records the source DGD generation on rendered objects.
	DGDGenerationAnnotation = "scheduling.lpu.nvidia.com/dgd-generation"
	// DGDUIDAnnotation records the source DGD UID on rendered objects.
	DGDUIDAnnotation = "scheduling.lpu.nvidia.com/dgd-uid"
	// WorkloadModeAnnotation records the projected LPX workload mode on rendered objects.
	WorkloadModeAnnotation = "scheduling.lpu.nvidia.com/workload-mode"
)

// WorkloadDigest is the SHA-256 identity of a projected LPX workload.
type WorkloadDigest [32]byte

// String returns the digest in canonical sha256-prefixed hexadecimal form.
func (d WorkloadDigest) String() string {
	return fmt.Sprintf("sha256:%x", d[:])
}

// ModelProjectionInput is the complete producer-owned input to one component's
// immutable model projections. It excludes Grove materialization and scheduler output.
type ModelProjectionInput struct {
	// Pipeline selects the DGD pipeline shape being projected.
	Pipeline Pipeline
	// Models contains the validated, nonempty logical model identities in canonical order.
	Models []string
	// RuntimeBuildRef is the build reference projected into runtime configuration.
	RuntimeBuildRef string
	// BuildSnapshot is the normalized immutable build input.
	BuildSnapshot NormalizedBuildSnapshot
}

// ModelProjection holds scheduler request inputs and runtime rendering state
// derived for one model. Completed component data is shared read-only; only the
// resolver assigns the stage before publishing the immutable projections.
type ModelProjection struct {
	digest                 WorkloadDigest
	compilerSnapshotDigest string
	runtimeBuildRef        string
	model                  string
	stage                  string
	pipeline               Pipeline
	configuredBuild        Build
	allocationMetadata     json.RawMessage
	// partitions retains immutable physical build evidence before runtime collapse.
	partitions    []BuildPartition
	connectors    []lpxv1alpha1.PropSyncConnectorRequest
	agentReplicas int
}

// Digest returns the immutable projection digest. The receiver must be non-nil.
func (p *ModelProjection) Digest() WorkloadDigest {
	return p.digest
}

// CompilerSnapshotDigest returns the immutable compiler snapshot identity.
func (p *ModelProjection) CompilerSnapshotDigest() string {
	return p.compilerSnapshotDigest
}

// Model returns the logical model identity. The receiver must be non-nil.
func (p *ModelProjection) Model() string {
	return p.model
}

// RequestSpec returns a fresh node-local request for one Grove scaling-group
// replica. The receiver and plan must be non-nil; neither input is mutated.
func (p *ModelProjection) RequestSpec(
	plan *MaterializationPlan,
	agentPodCliqueName string,
) lpxv1alpha1.LPUPipelineRequestSpec {
	connectors := make([]lpxv1alpha1.PropSyncConnectorRequest, len(p.connectors))
	for i := range p.connectors {
		p.connectors[i].DeepCopyInto(&connectors[i])
	}
	partitions := make([]lpxv1alpha1.PartitionRequest, len(p.partitions))
	mappings := make([]lpxv1alpha1.NodeLocalPartitionMapping, len(p.partitions))
	for index, partition := range p.partitions {
		partitionID := fmt.Sprintf("partition-%03d", index)
		request := lpxv1alpha1.PartitionRequest{
			ID:                  partitionID,
			Ordinal:             int64(index),
			CompilerPartitionID: int64(uint32(partition.SourcePartitionID)),
		}
		if p.configuredBuild.Family == BuildFamilyXT {
			shape, _, _ := xtShape(partition.Topology.ChipCount)
			request.XtShape = &shape
		}
		if partition.HXExtent != nil {
			extent := slices.Clone(partition.HXExtent)
			request.Extent = &extent
		}
		partitions[index] = request
		mappings[index] = lpxv1alpha1.NodeLocalPartitionMapping{
			ModelPartitionID: int64(index),
			PartitionID:      partitionID,
		}
	}
	spec := lpxv1alpha1.LPUPipelineRequestSpec{
		AllocationMetadata: runtime.RawExtension{Raw: slices.Clone(p.allocationMetadata)},
		MaterializationTarget: lpxv1alpha1.MaterializationTarget{
			PodCliqueSetReplicaIndex: 0,
			PodCliqueScalingGroupRef: &lpxv1alpha1.PodCliqueScalingGroupReference{
				Name: plan.LPXScalingGroup, ReplicaIndex: int64(plan.ReplicaIndex),
			},
		},
		RepairPolicy:       &lpxv1alpha1.RepairPolicy{Mode: lpxv1alpha1.RepairPolicyModeSamePlacement},
		PropSyncConnectors: connectors,
		TargetFamily:       lpxv1alpha1.TargetFamily(p.configuredBuild.Family),
		WorkloadMode:       p.schedulerWorkloadMode(),
		ExecutionBackend:   lpxv1alpha1.ExecutionBackendNodeLocal,
		NodeLocal: &lpxv1alpha1.NodeLocalRequest{
			AgentPodCliqueRef: lpxv1alpha1.PodCliqueReference{Name: agentPodCliqueName},
			Model:             p.model,
			PartitionMappings: mappings,
		},
		Partitions: partitions,
	}
	if plan.CyborgClique != "" {
		spec.CyborgPodCliqueRef = &lpxv1alpha1.PodCliqueReference{Name: plan.CyborgClique}
	}
	return spec
}

// schedulerWorkloadMode translates the normalized build family and pipeline at the LPX wire boundary.
func (p *ModelProjection) schedulerWorkloadMode() lpxv1alpha1.WorkloadMode {
	// HX and XT have separate wire values for the same two runtime shapes.
	if p.configuredBuild.Family == BuildFamilyHX {
		if p.pipeline == PipelineLPX {
			return lpxv1alpha1.WorkloadModeV3HxStrictHybrid
		}
		return lpxv1alpha1.WorkloadModeV3HxLPUOnly
	}
	if p.pipeline == PipelineLPX {
		return lpxv1alpha1.WorkloadModeV2StrictHybrid
	}
	return lpxv1alpha1.WorkloadModeV2LPUOnly
}
