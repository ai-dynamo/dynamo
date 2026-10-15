/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"fmt"
	"slices"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
)

const (
	// DeploymentNameAnnotation routes rendered-object events to their LPXGraphDeployment.
	DeploymentNameAnnotation = "lpx.nvidia.com/deployment-name"
	// WorkloadDigestAnnotation records the immutable Dynamo workload projection digest on rendered objects.
	WorkloadDigestAnnotation = "scheduling.lpu.nvidia.com/dynamo-workload-digest"
	// WorkloadModeAnnotation records the projected LPX workload mode on rendered objects.
	WorkloadModeAnnotation = "scheduling.lpu.nvidia.com/workload-mode"
)

// WorkloadDigest is the SHA-256 identity of a projected LPX workload.
type WorkloadDigest [32]byte

// String returns the digest in canonical sha256-prefixed hexadecimal form.
func (d WorkloadDigest) String() string {
	return fmt.Sprintf("sha256:%x", d[:])
}

// modelProjectionInput is the complete producer-owned input to one component's
// immutable model projections. It excludes Grove materialization and scheduler output.
type modelProjectionInput struct {
	// Pipeline selects the DGD pipeline shape being projected.
	Pipeline Pipeline
	// Models contains the validated, nonempty logical model identities in canonical order.
	Models []string
	// RuntimeBuildRef is the build reference projected into runtime configuration.
	RuntimeBuildRef string
	// Build is the normalized immutable build input.
	Build *Build
	// LocalPartitions selects hybrid runtime partitions that run on the Cyborg GPU.
	// Nil runs every partition on LPUs.
	LocalPartitions *dynamov1beta1.LPXLocalPartitions
}

// Model holds scheduler request inputs and runtime rendering state
// derived for one model. Completed component data is shared read-only; only the
// resolver assigns the stage before publishing the immutable projections.
type Model struct {
	digest                 WorkloadDigest
	compilerSnapshotDigest string
	runtimeBuildRef        string
	name                   string
	componentName          string
	pipeline               Pipeline
	configuredBuild        Build
	contractFormat         buildContractFormat
	allocationMetadata     json.RawMessage
	// partitions describes the partitions requested from LPX. Packing can combine
	// several configuredBuild.partitions into one LPX partition.
	partitions []buildPartition
	connectors []lpxv1alpha1.PropSyncConnectorRequest
	// propSyncEdges lists each position in partitions whose successor continues
	// a selected prop-sync chain. Only unpacked HX projections set it; XT
	// collapses selected chains in configuredBuild instead.
	propSyncEdges []int
	agentReplicas int
	// localPartitionIDs lists the runtime partitions that run on the Cyborg GPU, in build order.
	localPartitionIDs []int
}

// AgentReplicas returns the number of LPU Agent Pods that the projection requires.
func (p *Model) AgentReplicas() int {
	return p.agentReplicas
}

// Digest returns the immutable projection digest. The receiver must be non-nil.
func (p *Model) Digest() WorkloadDigest {
	return p.digest
}

// CompilerSnapshotDigest returns the immutable compiler snapshot identity.
func (p *Model) CompilerSnapshotDigest() string {
	return p.compilerSnapshotDigest
}

// Name returns the logical model identity. The receiver must be non-nil.
func (p *Model) Name() string {
	return p.name
}

// Component returns the DGD component that supplies this model.
func (p *Model) Component() string {
	return p.componentName
}

// RequestSpec returns a fresh node-local request for one Grove scaling-group
// replica. The receiver and plan must be non-nil; neither input is mutated.
func (p *Model) RequestSpec(
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
		request := p.configuredBuild.topology.partitionRequest(index, partition)
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
		TargetFamily:       p.configuredBuild.topology.target(),
		WorkloadMode:       p.schedulerWorkloadMode(),
		ExecutionBackend:   lpxv1alpha1.ExecutionBackendNodeLocal,
		NodeLocal: &lpxv1alpha1.NodeLocalRequest{
			AgentPodCliqueRef: lpxv1alpha1.PodCliqueReference{Name: agentPodCliqueName},
			Model:             p.name,
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
func (p *Model) schedulerWorkloadMode() lpxv1alpha1.WorkloadMode {
	return p.configuredBuild.topology.workloadMode(p.pipeline)
}

func newPartitionRequest(index int, partition buildPartition) lpxv1alpha1.PartitionRequest {
	return lpxv1alpha1.PartitionRequest{
		ID:                  fmt.Sprintf("partition-%03d", index),
		Ordinal:             int64(index),
		CompilerPartitionID: int64(uint32(partition.sourcePartitionID)),
	}
}
