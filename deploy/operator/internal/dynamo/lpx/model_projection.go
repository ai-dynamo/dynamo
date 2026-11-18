/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"net/url"
	"path/filepath"
	"strings"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
)

const (
	// DeploymentNameAnnotation routes rendered-object events to their LPXGraphDeployment.
	DeploymentNameAnnotation = "lpx.nvidia.com/deployment-name"
	// WorkloadDigestAnnotation records the immutable Dynamo workload projection digest on rendered objects.
	WorkloadDigestAnnotation = "scheduling.lpu.nvidia.com/dynamo-workload-digest"
	// WorkloadModeAnnotation records the projected LPX workload mode on rendered objects.
	WorkloadModeAnnotation = "scheduling.lpu.nvidia.com/workload-mode"
)

// ModelProjectionInput is the complete producer-owned input to one component's
// immutable model projections. It excludes Grove materialization and scheduler output.
type ModelProjectionInput struct {
	// Pipeline selects the DGD pipeline shape being projected.
	Pipeline Pipeline
	// Models contains the validated, nonempty logical model identities in canonical order.
	Models []string
	// Stage is the name of the DGD component that supplies the models.
	Stage string
	// RuntimeBuildRef is the build reference projected into runtime configuration.
	RuntimeBuildRef string
	// Build is the normalized immutable build input.
	Build *Build
	// LocalPartitions selects hybrid runtime partitions that run on the Cyborg GPU.
	// Nil runs every partition on LPUs.
	LocalPartitions *dynamov1beta1.LPXLocalPartitions
}

// ModelProjection holds scheduler request inputs and runtime rendering state
// derived for one model. It is immutable; the projections of one component
// share their partitions and connectors read-only.
type ModelProjection struct {
	digest          WorkloadDigest
	runtimeBuildRef string
	model           string
	stage           string
	pipeline        Pipeline
	configuredBuild Build
	partitions      []lpxv1alpha1.PartitionRequest
	connectors      []lpxv1alpha1.PropSyncConnectorRequest
	agentReplicas   int
	// localPartitionIDs lists the runtime partitions that run on the Cyborg GPU, in build order.
	localPartitionIDs []int
	// checkpoint is the build checkpoint projected into the Cyborg conductor, or nil.
	// Projecting or dropping it leaves the digest unchanged. The build's
	// contentID still hashes the whole deployment contract, so a rebuild whose
	// recorded checkpoint differs has a new digest and rolls the workload.
	checkpoint *Checkpoint
}

func (p *ModelProjection) Checkpoint() *Checkpoint {
	return p.checkpoint
}

// AgentReplicas returns the number of LPU Agent Pods that the projection requires.
func (p *ModelProjection) AgentReplicas() int {
	return p.agentReplicas
}

// Digest returns the immutable projection digest. The receiver must be non-nil.
func (p *ModelProjection) Digest() WorkloadDigest {
	return p.digest
}

// CompilerSnapshotDigest returns the immutable compiler snapshot identity.
func (p *ModelProjection) CompilerSnapshotDigest() string {
	return p.configuredBuild.contentID
}

// Model returns the logical model identity. The receiver must be non-nil.
func (p *ModelProjection) Model() string {
	return p.model
}

// ComponentName returns the DGD component that supplies this model.
func (p *ModelProjection) ComponentName() string {
	return p.stage
}

// RequestSpec returns a fresh node-local request for one Grove scaling-group
// replica. The receiver and plan must be non-nil; neither input is mutated.
func (p *ModelProjection) RequestSpec(plan *MaterializationPlan, agentPodCliqueName string) lpxv1alpha1.LPUPipelineRequestSpec {
	connectors := make([]lpxv1alpha1.PropSyncConnectorRequest, len(p.connectors))
	for i := range p.connectors {
		p.connectors[i].DeepCopyInto(&connectors[i])
	}

	partitions := make([]lpxv1alpha1.PartitionRequest, len(p.partitions))
	mappings := make([]lpxv1alpha1.NodeLocalPartitionMapping, len(p.partitions))
	for index := range p.partitions {
		p.partitions[index].DeepCopyInto(&partitions[index])
		mappings[index] = lpxv1alpha1.NodeLocalPartitionMapping{
			ModelPartitionID: int64(index),
			PartitionID:      partitions[index].ID,
		}
	}

	spec := lpxv1alpha1.LPUPipelineRequestSpec{
		MaterializationTarget: lpxv1alpha1.MaterializationTarget{
			PodCliqueSetReplicaIndex: 0,
			PodCliqueScalingGroupRef: &lpxv1alpha1.PodCliqueScalingGroupReference{
				Name: plan.LPXScalingGroup, ReplicaIndex: int64(plan.ReplicaIndex),
			},
		},
		RepairPolicy:       &lpxv1alpha1.RepairPolicy{Mode: lpxv1alpha1.RepairPolicyModeSamePlacement},
		PropSyncConnectors: connectors,
		TargetFamily:       p.targetFamily(),
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

// targetFamily translates the normalized build architecture at the LPX wire boundary.
func (p *ModelProjection) targetFamily() lpxv1alpha1.TargetFamily {
	switch p.configuredBuild.architecture {
	case lp30:
		return lpxv1alpha1.TargetFamilyHx16x8x2x3
	default: // LP20
		return lpxv1alpha1.TargetFamilyXt8888
	}
}

// schedulerWorkloadMode translates the normalized build architecture and pipeline at the LPX wire boundary.
func (p *ModelProjection) schedulerWorkloadMode() lpxv1alpha1.WorkloadMode {
	// HX and XT have separate wire values for the same two runtime shapes.
	switch p.configuredBuild.architecture {
	case lp30:
		if p.pipeline == PipelineHybrid {
			return lpxv1alpha1.WorkloadModeV3HxStrictHybrid
		}
		return lpxv1alpha1.WorkloadModeV3HxLPUOnly
	default: // LP20
		if p.pipeline == PipelineHybrid {
			return lpxv1alpha1.WorkloadModeV2StrictHybrid
		}
		return lpxv1alpha1.WorkloadModeV2LPUOnly
	}
}

// buildRuntimePath resolves a snapshot reference to its runtime filesystem path.
// A safe relative runtimeRef remaps file-URL snapshots under modelStoragePath;
// otherwise the snapshot reference remains authoritative.
func (p *ModelProjection) buildRuntimePath(modelStoragePath string) (string, error) {
	var (
		buildPath  = p.configuredBuild.path
		runtimeRef = p.runtimeBuildRef
	)

	snapshotURL, snapshotErr := url.Parse(buildPath)
	runtimeRef = strings.TrimSpace(runtimeRef)
	runtimeURL, runtimeErr := url.Parse(runtimeRef)
	if snapshotErr == nil && runtimeErr == nil && snapshotURL.Scheme == BuildSchemeFile &&
		runtimeRef != "" && runtimeURL.Scheme == "" && !filepath.IsAbs(runtimeRef) {
		cleaned := filepath.Clean(runtimeRef)
		if cleaned != "." && cleaned != ".." && !strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
			buildPath = (&url.URL{Scheme: BuildSchemeFile, Path: filepath.Join(modelStoragePath, cleaned)}).String()
		}
	}

	buildURL, err := parseBuildRef(buildPath)
	if err != nil {
		return "", fmt.Errorf("parse build path %q: %w", buildPath, err)
	}

	switch buildURL.Scheme {
	case BuildSchemeFile:
		return buildURL.Path, nil
	case BuildSchemeGCS:
		if modelStoragePath == "" {
			return "", fmt.Errorf("model storage path is empty")
		}

		objectPath := strings.TrimPrefix(buildURL.Path, "/")
		if objectPath == "" {
			return "", fmt.Errorf("invalid GCS build path %q: missing object path", buildPath)
		}

		// Validate every object path segment before joining the accepted path once.
		for component := range strings.SplitSeq(objectPath, "/") {
			if component == "" || component == "." || component == ".." {
				return "", fmt.Errorf("invalid GCS build path %q: bad path segment %q", buildPath, component)
			}
		}

		return filepath.Join(modelStoragePath, "gcs", buildURL.Host, objectPath), nil
	default:
		return "", fmt.Errorf("unsupported build path scheme %q", buildURL.Scheme)
	}
}
