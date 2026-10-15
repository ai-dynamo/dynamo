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

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"

	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
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

// ResolveWorkloads resolves and names every admitted component group before rendering.
// Workloads are returned in serving-component order; inputs are not mutated.
func ResolveWorkloads(ctx context.Context, reader client.Reader, dgd *dynamov1beta1.DynamoGraphDeployment, pcsName string, registry ModelRegistry) ([]*Workload, error) {
	groups := ComponentGroups(dgd)
	workloads := make([]*Workload, 0, len(groups))
	for _, name := range slices.Sorted(maps.Keys(groups)) {
		workload, err := resolveWorkload(ctx, reader, dgd, groups[name], registry)
		if err != nil {
			return nil, err
		}
		group := ""
		if len(groups) > 1 {
			group = name
		}
		workload, err = workload.WithNames(pcsName, group)
		if err != nil {
			return nil, err
		}
		for _, previous := range workloads {
			if previous.ScalingGroup() == workload.ScalingGroup() || previous.ResourcePrefix() == workload.ResourcePrefix() {
				return nil, fmt.Errorf("LPX components %q and %q resolve to conflicting resource names; rename one component", previous.ServingComponentName(), name)
			}
		}
		workloads = append(workloads, workload)
	}
	return workloads, nil
}

// resolveWorkload projects one admitted component group against immutable builds.
// reader, dgd and registry must be non-nil. componentNames must contain exactly the members
// of one ComponentGroups entry from the admitted dgd, in any order.
// Inputs are read without mutation.
func resolveWorkload(
	ctx context.Context,
	reader client.Reader,
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

		projections = make([]*ModelProjection, 0, len(components))
	)

	for _, stage := range components {
		model := stage.LPX
		configuredModel := stage.ComponentName

		// An admitted group has at most two components, so only its preceding component can share a build.
		if len(projections) == 0 || projections[len(projections)-1].runtimeBuildRef != model.BuildID {
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
			} else if snapshot.hasCUDAPartitions {
				pipeline = PipelineHybrid
			}
		}

		if err := validateWorkloadConductor(ctx, reader, dgd.Namespace, stage, pipeline, snapshot.hasCUDAPartitions); err != nil {
			return nil, err
		}

		// Project the component once into the resolver-owned aggregate destination.
		hasConductor := stage.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor) != nil
		modelNames := expandedModelNames(len(components), hasConductor, int(ptr.Deref(stage.Replicas, 1)))
		intent := ModelProjectionInput{
			Pipeline:        pipeline,
			Models:          modelNames,
			RuntimeBuildRef: model.BuildID,
			Build:           snapshot,
			LocalPartitions: model.LocalPartitions(),
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
	scalingGroupReplicas := ptr.Deref(conductor.Replicas, provideroverride.EffectiveGroveMinAvailable(conductor))

	// Canonical roles expand into default or draft0..draft7 followed by target.
	digest, err := workloadSetDigest(projections)
	if err != nil {
		return nil, err
	}
	workload := &Workload{
		conductorReplicas:    1,
		modelProjections:     projections,
		digest:               digest,
		scalingGroupReplicas: scalingGroupReplicas,
	}
	if pipeline == PipelineHybrid {
		minimum, err := workload.MinimumCyborgReplicas()
		if err != nil {
			return nil, err
		}
		workload.conductorReplicas = ptr.Deref(conductor.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor).Replicas, minimum)
	}
	return workload, nil
}

// validateWorkloadConductor checks role requirements that depend on the immutable build.
// reader and component must be non-nil.
func validateWorkloadConductor(
	ctx context.Context,
	reader client.Reader,
	namespace string,
	component *dynamov1beta1.DynamoComponentDeploymentSharedSpec,
	pipeline Pipeline,
	requiresGPU bool,
) error {
	conductor := component.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor)
	// Validate the selected hybrid runtime's GPU resources and claim bindings.
	if requiresGPU {
		if pipeline == PipelineSpecDecode {
			return fmt.Errorf("%w: the shared speculative runtime requires LPU-only builds", ErrUnsupportedRuntime)
		}
		template := conductor.PodTemplate
		container := common.FindContainerByName(template.Spec.Containers, commonconsts.MainContainerName)
		count, err := dra.ResolveGPUCount(ctx, reader, namespace, &template.Spec, container.Resources)
		if err != nil {
			return fmt.Errorf("component %q conductor resources: %w", component.ComponentName, err)
		}
		if count > 0 {
			return nil
		}
		return fmt.Errorf("component %q conductor main container requires a positive GPU count from scalar resources or resourceClaims", component.ComponentName)
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
