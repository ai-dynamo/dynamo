/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"

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

// ResolveWorkloads resolves every LPX component group of an admitted dgd against
// immutable builds and names each workload's resources inside PodCliqueSet pcsName.
// dgd and registry must be non-nil. Workloads are ordered by name.
// Inputs are read without mutation.
func ResolveWorkloads(
	ctx context.Context,
	dgd *dynamov1beta1.DynamoGraphDeployment,
	pcsName string,
	registry ModelRegistry,
) ([]*Workload, error) {
	groups := componentGroups(dgd)
	workloads := make([]*Workload, 0, len(groups))
	for _, name := range slices.Sorted(maps.Keys(groups)) {
		workload, err := resolveWorkload(ctx, dgd, groups[name], registry)
		if err != nil {
			return nil, err
		}

		// Only independent workloads need additional names within the shared PCS.
		group := ""
		if len(groups) > 1 {
			group = name
		}
		if err := workload.nameResources(pcsName, group); err != nil {
			return nil, err
		}

		// Compare finalized identities only against previously resolved workloads.
		for _, previous := range workloads {
			if previous.scalingGroupTemplate == workload.scalingGroupTemplate || previous.resourcePrefix == workload.resourcePrefix {
				return nil, fmt.Errorf("LPX components %q and %q resolve to conflicting resource names; rename one component", previous.name, name)
			}
		}
		workloads = append(workloads, workload)
	}
	return workloads, nil
}

// resolveWorkload projects one admitted component group against immutable builds.
// componentNames must contain exactly the members of one componentGroups entry
// from the admitted dgd, in any order.
func resolveWorkload(
	ctx context.Context,
	dgd *dynamov1beta1.DynamoGraphDeployment,
	componentNames []string,
	registry ModelRegistry,
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
		snapshot *Build
		pipeline Pipeline

		models = make([]*Model, 0, len(components))
	)

	for _, stage := range components {
		model := stage.LPX
		configuredModel := stage.ComponentName

		// An admitted group has at most two components, so only its preceding component can share a build.
		if len(models) == 0 || models[len(models)-1].component.runtimeBuildRef != model.BuildID {
			acquired, acquireErr := registry.AcquireBuild(ctx, model.BuildID)
			if acquireErr != nil {
				// Keep compiler validation failures distinct from retryable acquisition failures.
				if errors.Is(acquireErr, errInvalidBuildManifest) {
					return nil, fmt.Errorf("invalid LPX build for model %q build %q: %w", configuredModel, model.BuildID, acquireErr)
				}

				return nil, fmt.Errorf(
					"%w for model %q build %q: %w",
					ErrBuildSnapshotAcquisition,
					configuredModel,
					model.BuildID,
					acquireErr,
				)
			}
			snapshot = acquired
		}

		// Validated model cardinality fixes one runtime shape for the selected workload.
		if pipeline == "" {
			pipeline = PipelineSingle
			if len(components) == 2 {
				pipeline = PipelineSpecDecode
			} else if snapshot.compilationMode == compilationModeHybrid {
				pipeline = PipelineHybrid
			}
		}

		if err := validateWorkloadConductor(stage, pipeline, snapshot.compilationMode); err != nil {
			return nil, err
		}

		// Project the component once, in canonical runtime order.
		hasConductor := stage.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor) != nil
		modelNames := expandedModelNames(len(components), hasConductor, int(ptr.Deref(stage.Replicas, 1)))
		projected, err := projectComponent(stage.ComponentName, model.BuildID, snapshot, pipeline, modelNames)
		if err != nil {
			return nil, fmt.Errorf("project LPX model %q from build %q: %w", modelNames[0], model.BuildID, err)
		}

		// Component geometry fixes the Agent count independently of draft fanout.
		agents := projected[0].component.agentReplicas
		if count := stage.ComponentRole(dynamov1beta1.ComponentRoleLPXAgent).Replicas; count != nil && int(*count) != agents {
			return nil, fmt.Errorf("component %q agent replicas %d must match the compiled count %d", configuredModel, *count, agents)
		}
		models = append(models, projected...)
	}

	// The conductor component owns explicit capacity or the initial native seed.
	conductor := components[len(components)-1]
	scalingGroupReplicas := ptr.Deref(conductor.Replicas, ptr.Deref(conductor.MinAvailable, 1))

	// A Nova conductor is a singleton; Cyborg defaults to one complete client group.
	conductorReplicas := int32(1)
	if pipeline == PipelineHybrid {
		minimum, err := minimumCyborgReplicas(&models[0].component.configuredBuild)
		if err != nil {
			return nil, err
		}
		conductorReplicas = ptr.Deref(conductor.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor).Replicas, minimum)
	}

	// Canonical roles expand into default or draft0..draft7 followed by target.
	digest, err := workloadSetDigest(models)
	if err != nil {
		return nil, err
	}
	return &Workload{
		name:                 conductor.ComponentName,
		minAvailable:         ptr.Deref(conductor.MinAvailable, 1),
		conductorReplicas:    conductorReplicas,
		models:               models,
		digest:               digest,
		scalingGroupReplicas: scalingGroupReplicas,
	}, nil
}

// validateWorkloadConductor checks role requirements that depend on the immutable build.
func validateWorkloadConductor(
	component *dynamov1beta1.DynamoComponentDeploymentSharedSpec,
	pipeline Pipeline,
	compilationMode compilationMode,
) error {
	conductor := component.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor)
	// Hybrid execution is selected by immutable build metadata, not template presence.
	if compilationMode == compilationModeHybrid {
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
