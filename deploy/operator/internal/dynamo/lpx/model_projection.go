/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"fmt"
	"slices"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
)

const (
	// WorkloadDigestAnnotation records the immutable Dynamo workload projection digest on rendered objects.
	WorkloadDigestAnnotation = "scheduling.lpu.nvidia.com/dynamo-workload-digest"
	// DGDGenerationAnnotation records the source DGD generation on rendered objects.
	DGDGenerationAnnotation = "scheduling.lpu.nvidia.com/dgd-generation"
	// DGDUIDAnnotation records the source DGD UID on rendered objects.
	DGDUIDAnnotation = "scheduling.lpu.nvidia.com/dgd-uid"
	// ExecutionBackendAnnotation selects the LPX execution backend on rendered objects.
	ExecutionBackendAnnotation = commonconsts.KubeAnnotationLPXExecutionBackend
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
	// ModelSettings contains the optional per-model JSON settings.
	ModelSettings json.RawMessage
}

// ModelProjection holds scheduler request inputs and runtime rendering state
// derived for one model. Completed component data is shared read-only; only the
// resolver assigns the stage before publishing the immutable projections.
type ModelProjection struct {
	digest             WorkloadDigest
	runtimeBuildRef    string
	model              string
	stage              string
	pipeline           Pipeline
	configuredBuild    Build
	allocationMetadata json.RawMessage
	// partitions retains immutable physical build evidence before runtime collapse.
	partitions    []BuildPartition
	connectors    []lpxv1alpha1.PropSyncConnectorRequest
	agentReplicas int
}

// Digest returns the immutable projection digest. The receiver must be non-nil.
func (p *ModelProjection) Digest() WorkloadDigest {
	return p.digest
}

// Model returns the logical model identity. The receiver must be non-nil.
func (p *ModelProjection) Model() string {
	return p.model
}

// RequestSpec returns a fresh node-local LPR spec. The materialization owner
// supplies only workload references; scheduler-owned placement fields do not
// exist in this request arm. The receiver must be non-nil. A nil cyborgRef is
// supported and omits the Cyborg reference. Neither input is mutated.
func (p *ModelProjection) RequestSpec(
	namespace string,
	podGangName string,
	cyborgRef *lpxv1alpha1.CyborgPodCliqueReference,
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
	ns := namespace
	spec := lpxv1alpha1.LPUPipelineRequestSpec{
		AllocationMetadata: runtime.RawExtension{Raw: slices.Clone(p.allocationMetadata)},
		PodGangRef:         lpxv1alpha1.NamespacedName{Name: podGangName, Namespace: &ns},
		PropSyncConnectors: connectors,
		TargetFamily:       lpxv1alpha1.TargetFamily(p.configuredBuild.Family),
		WorkloadMode:       p.schedulerWorkloadMode(),
		ExecutionBackend:   lpxv1alpha1.ExecutionBackendNodeLocal,
		NodeLocal: &lpxv1alpha1.NodeLocalRequest{
			Model:             p.model,
			PartitionMappings: mappings,
		},
		Partitions: partitions,
	}
	if cyborgRef != nil {
		spec.CyborgPodCliqueRef = cyborgRef.DeepCopy()
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
