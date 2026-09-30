/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"fmt"

	"k8s.io/utils/ptr"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

const (
	// MaxSpecDecodeNumDrafts bounds the supported speculative-decoding draft fanout.
	MaxSpecDecodeNumDrafts = 8

	runtimeModelDraft  = "draft"
	runtimeModelTarget = "target"
)

// ErrUnsupportedRuntime identifies input that the supported LPX runtimes cannot materialize.
var ErrUnsupportedRuntime = errors.New("unsupported LPX runtime")

// ErrBuildSnapshotAcquisition identifies retryable failures while reading the
// immutable build snapshot from its backing store.
var ErrBuildSnapshotAcquisition = errors.New("acquiring immutable LPX build snapshot")

// BuildSnapshotSource acquires immutable build inputs by build reference.
type BuildSnapshotSource interface {
	// AcquireBuildSnapshot returns a complete snapshot for id or an error.
	AcquireBuildSnapshot(context.Context, string) (*BuildSnapshot, error)
}

// ResolveWorkload projects one admitted component group against immutable builds.
// dgd and source must be non-nil. componentNames must contain exactly the members
// of one ComponentGroups entry from the admitted dgd, in any order.
// Inputs are read without mutation.
func ResolveWorkload(
	ctx context.Context,
	dgd *dynamov1beta1.DynamoGraphDeployment,
	componentNames []string,
	source BuildSnapshotSource,
) (*Workload, error) {
	// Resolve only this group's native components from the unchanged graph.
	components := make([]*dynamov1beta1.DynamoComponentDeploymentSharedSpec, 0, len(componentNames))
	for _, name := range componentNames {
		components = append(components, dgd.GetComponentByName(name))
	}

	// Preserve draft-before-target runtime identities independently of authored order.
	if len(components) == 2 && components[0].ComponentRole(dynamov1beta1.ComponentRoleLPXConductor) != nil {
		components[0], components[1] = components[1], components[0]
	}

	// Acquire each build and expand its models in canonical runtime order.
	var (
		snapshot NormalizedBuildSnapshot
		pipeline Pipeline

		projections = make([]*ModelProjection, 0, len(components))
	)

	for _, stage := range components {
		model := stage.LPX
		configuredModel := stage.ComponentName

		// An admitted group has at most two components, so only its preceding component can share a build.
		if len(projections) == 0 || projections[len(projections)-1].runtimeBuildRef != model.BuildID {
			rawSnapshot, acquireErr := source.AcquireBuildSnapshot(ctx, model.BuildID)
			if acquireErr != nil {
				return nil, fmt.Errorf(
					"%w for model %q build %q: %w",
					ErrBuildSnapshotAcquisition,
					configuredModel,
					model.BuildID,
					acquireErr,
				)
			}
			normalizedSnapshot, normalizeErr := normalizeBuildSnapshot(rawSnapshot)
			if normalizeErr != nil {
				return nil, fmt.Errorf(
					"normalizing immutable LPX build snapshot for model %q build %q: %w",
					configuredModel,
					model.BuildID,
					normalizeErr,
				)
			}
			snapshot = normalizedSnapshot
		}

		// Validated model cardinality fixes one runtime shape for the selected workload.
		if pipeline == "" {
			pipeline = PipelineSingle
			if len(components) == 2 {
				pipeline = PipelineSpecDecode
			} else if snapshot.build.CompilationMode == BuildCompilationModeHybrid {
				pipeline = PipelineLPX
			}
		}

		if err := validateWorkloadConductor(stage, pipeline, snapshot.build.CompilationMode); err != nil {
			return nil, err
		}

		// Project the component once into the resolver-owned aggregate destination.
		hasConductor := stage.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor) != nil
		modelNames := expandedModelNames(len(components), hasConductor, int(ptr.Deref(stage.Replicas, 1)))
		intent := ModelProjectionInput{
			Pipeline:        pipeline,
			Models:          modelNames,
			RuntimeBuildRef: model.BuildID,
			BuildSnapshot:   snapshot,
		}
		projected, err := appendModelProjections(projections, intent)
		if err != nil {
			return nil, fmt.Errorf("project LPX model %q from build %q: %w", modelNames[0], model.BuildID, err)
		}

		// Component geometry fixes the Agent count independently of draft fanout.
		componentProjections := projected[len(projections):]
		if count := stage.ComponentRole(dynamov1beta1.ComponentRoleLPXAgent).Replicas; count != nil && int(*count) != componentProjections[0].agentReplicas {
			return nil, fmt.Errorf("component %q agent replicas %d must match the compiled count %d", configuredModel, *count, componentProjections[0].agentReplicas)
		}

		// Retain the authored stage association before acquiring the next component.
		for _, projection := range componentProjections {
			projection.stage = stage.ComponentName
		}
		projections = projected
	}

	// The conductor component owns explicit capacity or the initial native seed.
	conductor := components[len(components)-1]
	scalingGroupReplicas := ptr.Deref(conductor.Replicas, ptr.Deref(conductor.MinAvailable, 1))

	// Canonical roles expand into default or draft0..draft7 followed by target.
	digest, err := workloadSetDigest(projections)
	if err != nil {
		return nil, err
	}
	return &Workload{
		modelProjections:     projections,
		digest:               digest,
		scalingGroupReplicas: scalingGroupReplicas,
	}, nil
}

// validateWorkloadConductor checks role requirements that depend on the immutable build.
func validateWorkloadConductor(
	component *dynamov1beta1.DynamoComponentDeploymentSharedSpec,
	pipeline Pipeline,
	compilationMode BuildCompilationMode,
) error {
	conductor := component.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor)
	// Hybrid execution is selected by immutable build metadata, not template presence.
	if compilationMode == BuildCompilationModeHybrid {
		if pipeline == PipelineSpecDecode {
			return fmt.Errorf("%w: the shared speculative runtime requires LPU-only builds", ErrUnsupportedRuntime)
		}
		template := conductor.PodTemplate
		container := common.FindContainerByName(template.Spec.Containers, commonconsts.MainContainerName)
		count, err := EffectiveCyborgGPUCount(container.Resources)
		if err != nil {
			return fmt.Errorf("component %q conductor resources: %w", component.ComponentName, err)
		}
		if count > 0 {
			return nil
		}

		// A Pod claim supplies devices only to containers that reference its local name.
		for _, claim := range container.Resources.Claims {
			for _, podClaim := range template.Spec.ResourceClaims {
				if claim.Name == podClaim.Name {
					return nil
				}
			}
		}
		return fmt.Errorf("component %q conductor main container requires a declared resourceClaim or a positive %s request", component.ComponentName, commonconsts.KubeResourceGPUNvidia)
	}

	// An LPU-only conductor is a singleton even when the workload replica count is larger.
	if conductor != nil && ptr.Deref(conductor.Replicas, 1) != 1 {
		return fmt.Errorf("%w: component %q conductor replicas must be one for LPU-only execution", ErrUnsupportedRuntime, component.ComponentName)
	}

	return nil
}

func expandedModelNames(componentCount int, hasConductor bool, draftCount int) []string {
	// Component membership retains Nova's existing default/draft/target identities.
	if componentCount == 1 {
		return []string{"default"}
	}
	if hasConductor {
		return []string{runtimeModelTarget}
	}
	names := make([]string, draftCount)
	for index := range names {
		names[index] = fmt.Sprintf("draft%d", index)
	}
	return names
}
