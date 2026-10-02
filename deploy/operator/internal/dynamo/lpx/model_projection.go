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
	// DeploymentNameAnnotation routes rendered-object events to their LPXGraphDeployment.
	DeploymentNameAnnotation = "lpx.nvidia.com/deployment-name"
	// WorkloadDigestAnnotation records the immutable Dynamo workload projection digest on rendered objects.
	WorkloadDigestAnnotation = "scheduling.lpu.nvidia.com/dynamo-workload-digest"
	// WorkloadModeAnnotation records the projected LPX workload mode on rendered objects.
	WorkloadModeAnnotation = "scheduling.lpu.nvidia.com/workload-mode"
	// ServingLabel selects each workload's conductor Pods for its discovery Service.
	ServingLabel = "lpx.nvidia.com/serving"
)

// WorkloadDigest is the SHA-256 identity of a projected LPX workload.
type WorkloadDigest [32]byte

// String returns the digest in canonical sha256-prefixed hexadecimal form.
func (d WorkloadDigest) String() string {
	return fmt.Sprintf("sha256:%x", d[:])
}

// component is one DGD component's compiled LPU runtime, shared read-only by
// the models it serves.
type component struct {
	// name is the DGD component that supplies the models.
	name string
	// runtimeBuildRef is the authored build ID, used to resolve runtime model paths.
	runtimeBuildRef string
	// pipeline is the runtime shape the build was projected for.
	pipeline Pipeline
	// configuredBuild is the runtime view of the build, including XT's trimmed or
	// collapsed partitions.
	configuredBuild    Build
	allocationMetadata json.RawMessage
	// partitionRequests are the physical scheduler partitions before runtime collapse.
	partitionRequests []lpxv1alpha1.PartitionRequest
	connectors        []lpxv1alpha1.PropSyncConnectorRequest
	// agentReplicas is the number of Agent pods that serve each model.
	agentReplicas int
}

// Model is one runtime model: default, draftN, or target. Each model has its own
// Agent clique and one scheduler request per workload replica.
type Model struct {
	name      string
	digest    WorkloadDigest
	component *component
	// agentTemplate is the model's Agent clique template, assigned with the workload's names.
	agentTemplate string
}

// Name returns the logical model identity. The receiver must be non-nil.
func (m *Model) Name() string {
	return m.name
}

// Component returns the DGD component that supplies this model.
func (m *Model) Component() string {
	return m.component.name
}

// Digest returns the immutable model digest. The receiver must be non-nil.
func (m *Model) Digest() WorkloadDigest {
	return m.digest
}

// CompilerSnapshotDigest returns the immutable compiler manifest identity.
func (m *Model) CompilerSnapshotDigest() string {
	return m.component.configuredBuild.contentID
}

// RequestSpec returns a fresh node-local request for model in one scaling-group
// replica. model must belong to w; neither is mutated.
func (w *Workload) RequestSpec(model *Model, replica int32) lpxv1alpha1.LPUPipelineRequestSpec {
	spec := model.requestSpec()
	spec.MaterializationTarget = lpxv1alpha1.MaterializationTarget{
		PodCliqueSetReplicaIndex: 0,
		PodCliqueScalingGroupRef: &lpxv1alpha1.PodCliqueScalingGroupReference{
			Name: w.scalingGroup, ReplicaIndex: int64(replica),
		},
	}
	spec.NodeLocal.AgentPodCliqueRef = lpxv1alpha1.PodCliqueReference{Name: w.cliqueName(model.agentTemplate, replica)}
	if w.Pipeline() == PipelineHybrid {
		spec.CyborgPodCliqueRef = &lpxv1alpha1.PodCliqueReference{Name: w.ConductorCliqueName(replica)}
	}
	return spec
}

// requestSpec returns a fresh request for the model's placement and runtime
// contract, without its Grove materialization target.
func (m *Model) requestSpec() lpxv1alpha1.LPUPipelineRequestSpec {
	connectors := make([]lpxv1alpha1.PropSyncConnectorRequest, len(m.component.connectors))
	for i := range m.component.connectors {
		m.component.connectors[i].DeepCopyInto(&connectors[i])
	}
	partitions := make([]lpxv1alpha1.PartitionRequest, len(m.component.partitionRequests))
	mappings := make([]lpxv1alpha1.NodeLocalPartitionMapping, len(m.component.partitionRequests))
	for index := range m.component.partitionRequests {
		m.component.partitionRequests[index].DeepCopyInto(&partitions[index])
		mappings[index] = lpxv1alpha1.NodeLocalPartitionMapping{
			ModelPartitionID: partitions[index].Ordinal,
			PartitionID:      partitions[index].ID,
		}
	}
	return lpxv1alpha1.LPUPipelineRequestSpec{
		AllocationMetadata: runtime.RawExtension{Raw: slices.Clone(m.component.allocationMetadata)},
		RepairPolicy:       &lpxv1alpha1.RepairPolicy{Mode: lpxv1alpha1.RepairPolicyModeSamePlacement},
		PropSyncConnectors: connectors,
		TargetFamily:       m.component.configuredBuild.family.target,
		WorkloadMode:       m.component.configuredBuild.family.workloadMode(m.component.pipeline),
		ExecutionBackend:   lpxv1alpha1.ExecutionBackendNodeLocal,
		NodeLocal: &lpxv1alpha1.NodeLocalRequest{
			Model:             m.name,
			PartitionMappings: mappings,
		},
		Partitions: partitions,
	}
}

// newPartitionRequest identifies a physical scheduler partition by its position in the build.
func newPartitionRequest(index int, partition buildPartition) lpxv1alpha1.PartitionRequest {
	return lpxv1alpha1.PartitionRequest{
		ID:                  fmt.Sprintf("partition-%03d", index),
		Ordinal:             int64(index),
		CompilerPartitionID: int64(uint32(partition.sourcePartitionID)),
	}
}
