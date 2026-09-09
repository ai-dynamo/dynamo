/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
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
	component := components[len(components)-1]
	projections := make([]*ModelProjection, 0, len(components))
	var snapshot NormalizedBuildSnapshot
	var pipeline Pipeline
	for index, stage := range components {
		model := stage.LPX
		configuredModel := stage.ComponentName
		var modelSettings json.RawMessage
		if model.Settings != nil {
			modelSettings = model.Settings.Raw
		}
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
			ModelSettings:   modelSettings,
		}
		projected, err := appendModelProjections(projections, intent)
		if err != nil {
			return nil, fmt.Errorf("project LPX model %q from build %q: %w", modelNames[0], model.BuildID, err)
		}

		// Component geometry fixes the Agent count independently of draft fanout.
		componentProjections := projected[len(projections):]
		if count := stage.ComponentRole(dynamov1beta1.ComponentRoleWorker).Replicas; count != nil && int(*count) != componentProjections[0].agentReplicas {
			return nil, fmt.Errorf("component %q agent replicas %d must match the compiled count %d", configuredModel, *count, componentProjections[0].agentReplicas)
		}

		// Retain the authored stage association before acquiring the next component.
		for _, projection := range componentProjections {
			projection.stage = stage.ComponentName
		}
		projections = projected
	}
	replicas := ptr.Deref(component.Replicas, 1)
	if pipeline != PipelineLPX && replicas != 1 {
		return nil, fmt.Errorf("%w: fixed engine replicas above one currently require a hybrid engine", ErrUnsupportedRuntime)
	}

	// Canonical roles expand into default or draft0..draft7 followed by target.
	digest, err := workloadSetDigest(projections)
	if err != nil {
		return nil, err
	}
	return &SelectedWorkload{
		modelProjections: projections,
		digest:           digest,
		engineReplicas:   replicas,
	}, nil
}

// validateSelectedConductor checks role requirements that depend on the immutable build.
func validateSelectedConductor(
	dgd *dynamov1beta1.DynamoGraphDeployment,
	component *dynamov1beta1.DynamoComponentDeploymentSharedSpec,
	pipeline Pipeline,
	compilationMode BuildCompilationMode,
) error {
	conductor := component.ComponentRole(dynamov1beta1.ComponentRoleLeader)
	// Hybrid execution is selected by immutable build metadata, not template presence.
	if compilationMode == BuildCompilationModeHybrid {
		if pipeline == PipelineSpecDecode {
			return fmt.Errorf("%w: the shared speculative runtime requires LPU-only builds", ErrUnsupportedRuntime)
		}
		template := component.ComponentRole(dynamov1beta1.ComponentRoleWorker).PodTemplate
		if conductor != nil && conductor.PodTemplate != nil {
			template = conductor.PodTemplate
		}
		container := findMainContainer(template.Spec.Containers)
		count, err := EffectiveCyborgGPUCount(container.Resources)
		if err != nil {
			return fmt.Errorf("component %q conductor resources: %w", component.ComponentName, err)
		}
		if len(template.Spec.ResourceClaims) == 0 && count <= 0 {
			return fmt.Errorf("component %q conductor requires resourceClaims or a positive %s request", component.ComponentName, commonconsts.KubeResourceGPUNvidia)
		}
		return nil
	}
	if conductor == nil {
		return nil
	}
	if ptr.Deref(conductor.Replicas, 1) != 1 {
		return fmt.Errorf("%w: component %q conductor replicas must be one for LPU-only execution", ErrUnsupportedRuntime, component.ComponentName)
	}
	if conductor.PodTemplate == nil {
		return nil
	}
	// Explicit LPU conductors receive the same appended launch arguments as inherited templates.
	componentIndex := slices.IndexFunc(dgd.Spec.Components, func(candidate dynamov1beta1.DynamoComponentDeploymentSharedSpec) bool {
		return candidate.ComponentName == component.ComponentName
	})
	roleIndex := slices.IndexFunc(component.Roles, func(role dynamov1beta1.ComponentRoleSpec) bool { return role.Name == dynamov1beta1.ComponentRoleLeader })
	for containerIndex := range conductor.PodTemplate.Spec.Containers {
		container := &conductor.PodTemplate.Spec.Containers[containerIndex]
		if container.Name != commonconsts.MainContainerName {
			continue
		}
		containerPath := field.NewPath("spec", "components").Index(componentIndex).Child("roles").Index(roleIndex).Child("podTemplate", "spec", "containers").Index(containerIndex)
		if err := validateAllocationInjectionTargetFields(container, containerPath).ToAggregate(); err != nil {
			return err
		}
	}
	return nil
}
