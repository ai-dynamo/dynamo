/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
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

// ResolveSelectedWorkload validates and projects the selected LPX workload in dgd.
// dgd and source must be non-nil. The function reads but does not mutate dgd.
func ResolveSelectedWorkload(
	ctx context.Context,
	dgd *dynamov1beta1.DynamoGraphDeployment,
	source BuildSnapshotSource,
) (*SelectedWorkload, error) {
	components, allErrs := validateSelectedIntent(dgd)
	if err := allErrs.ToAggregate(); err != nil {
		return nil, err
	}
	projections := make([]*ModelProjection, 0, len(components))
	var snapshot NormalizedBuildSnapshot
	var pipeline Pipeline
	for index, stage := range components {
		model := stage.LPX
		configuredModel := stage.ComponentName

		// A validated selection has at most two models, so only the preceding projection can share its build.
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

		if err := validateSelectedConductor(dgd, stage, pipeline, snapshot.build.CompilationMode); err != nil {
			return nil, err
		}

		// Project the component once into the resolver-owned aggregate destination.
		modelNames := expandedSelectedModelNames(index, len(components), int(ptr.Deref(stage.Replicas, 1)))
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
	scalingGroupReplicas := int32(1)
	if len(components) == 1 {
		// A sole component can flatten its replica axis into the outer scaling group.
		scalingGroupReplicas = ptr.Deref(components[0].Replicas, 1)
	}
	// Canonical roles expand into default or draft0..draft7 followed by target.
	digest, err := workloadSetDigest(projections)
	if err != nil {
		return nil, err
	}
	return &SelectedWorkload{
		modelProjections:     projections,
		digest:               digest,
		scalingGroupReplicas: scalingGroupReplicas,
	}, nil
}

// validateSelectedConductor checks role requirements that depend on the immutable build.
func validateSelectedConductor(
	dgd *dynamov1beta1.DynamoGraphDeployment,
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

	// Only the LPU-only serving template materializes a renamed conductor container.
	if component != ServingComponent(dgd) {
		return nil
	}

	// Preserve authored indices when reporting conductor name collisions.
	componentIndex := slices.IndexFunc(dgd.Spec.Components, func(candidate dynamov1beta1.DynamoComponentDeploymentSharedSpec) bool {
		return candidate.ComponentName == component.ComponentName
	})
	roleIndex := slices.IndexFunc(component.Roles, func(candidate dynamov1beta1.ComponentRoleSpec) bool {
		return candidate.Name == conductor.Name
	})
	podSpecPath := field.NewPath("spec", "components").Index(componentIndex).Child("roles").Index(roleIndex).Child("podTemplate", "spec")

	// Check both container lists before a selected workload can be published.
	if err := validateRolePodSpecContainerNames(&conductor.PodTemplate.Spec, podSpecPath, dynamov1beta1.ComponentRoleLPXConductor).ToAggregate(); err != nil {
		return err
	}

	// An LPU-only conductor is a singleton even when the engine replica count is larger.
	if ptr.Deref(conductor.Replicas, 1) != 1 {
		return fmt.Errorf("%w: component %q conductor replicas must be one for LPU-only execution", ErrUnsupportedRuntime, component.ComponentName)
	}

	return nil
}
