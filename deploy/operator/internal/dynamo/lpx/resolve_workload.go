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
	"math"
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

// ResolveWorkloads resolves and names each LPX workload in serving-component order.
// reader, dgd and registry must be non-nil, and dgd must have passed admission.
// Inputs are not mutated.
func ResolveWorkloads(ctx context.Context, reader client.Reader, dgd *dynamov1beta1.DynamoGraphDeployment, pcsName string, registry ModelRegistry) ([]*Workload, error) {
	// Sorted serving-component names fix the workload order for every caller.
	groups := ComponentGroups(dgd)
	workloads := make([]*Workload, 0, len(groups))
	for _, name := range slices.Sorted(maps.Keys(groups)) {
		workload, err := resolveWorkload(ctx, reader, dgd, groups[name], registry, pcsName, len(groups) > 1)
		if err != nil {
			return nil, err
		}

		// Finalized names must be unique among all workloads in the same PCS.
		for _, previous := range workloads {
			if previous.ScalingGroup() == workload.ScalingGroup() || previous.ResourcePrefix() == workload.ResourcePrefix() {
				return nil, fmt.Errorf("LPX components %q and %q resolve to conflicting resource names; rename one component", previous.ServingComponentName(), name)
			}
		}
		workloads = append(workloads, workload)
	}
	return workloads, nil
}

// resolveWorkload projects one admitted component group against immutable builds
// and names it in its PodCliqueSet.
// reader, dgd and registry must be non-nil. componentNames must contain exactly the members
// of one ComponentGroups entry from the admitted dgd, in any order.
// shared reports that the dgd has other component groups in the same PCS.
// Inputs are read without mutation.
func resolveWorkload(
	ctx context.Context,
	reader client.Reader,
	dgd *dynamov1beta1.DynamoGraphDeployment,
	componentNames []string,
	registry ModelRegistry,
	pcsName string,
	shared bool,
) (*Workload, error) {
	// Resolve only this group's native components from the unchanged graph.
	components := make([]*dynamov1beta1.DynamoComponentDeploymentSharedSpec, 0, len(componentNames))
	for _, name := range componentNames {
		components = append(components, dgd.GetComponentByName(name))
	}

	// Preserve draft-before-target runtime identities independently of authored order.
	if len(components) == 2 && ServesEndpoint(components[0]) {
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

		// Project the component's models from its build once.
		modelNames := expandedModelNames(len(components), ServesEndpoint(stage), int(ptr.Deref(stage.Replicas, 1)))
		intent := ModelProjectionInput{
			Pipeline:        pipeline,
			Models:          modelNames,
			Stage:           stage.ComponentName,
			RuntimeBuildRef: model.BuildID,
			Build:           snapshot,
			LocalPartitions: model.LocalPartitions(),
		}
		projected, err := projectModels(intent)
		if err != nil {
			return nil, fmt.Errorf("project LPX model %q from build %q: %w", modelNames[0], model.BuildID, err)
		}

		// Component geometry fixes the Agent count independently of draft fanout.
		if count := stage.ComponentRole(dynamov1beta1.ComponentRoleLPXAgent).Replicas; count != nil && int(*count) != projected[0].agentReplicas {
			return nil, fmt.Errorf("component %q agent replicas %d must match the compiled count %d", configuredModel, *count, projected[0].agentReplicas)
		}
		projections = append(projections, projected...)
	}

	// The serving component owns explicit capacity or the initial native seed, and the Cyborg width.
	serving := components[len(components)-1]
	scalingGroupReplicas := ptr.Deref(serving.Replicas, provideroverride.EffectiveGroveMinAvailable(serving))
	cyborgReplicas, err := resolveCyborgReplicas(serving.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor), pipeline, snapshot)
	if err != nil {
		return nil, err
	}
	return newWorkload(pcsName, shared, projections, scalingGroupReplicas, cyborgReplicas)
}

// resolveCyborgReplicas returns the Cyborg width of a hybrid workload, and zero for any other
// workload, which has no conductor. An omitted width starts with one complete compiled client
// group. role is the validated conductor role of a hybrid workload; build is its non-nil build.
func resolveCyborgReplicas(role *dynamov1beta1.ComponentRoleSpec, pipeline Pipeline, build *Build) (int32, error) {
	if pipeline != PipelineHybrid {
		return 0, nil
	}
	minimum := int64(build.ioFPGACount) * int64(build.ioFanoutFactor)
	if minimum > math.MaxInt32 {
		return 0, fmt.Errorf("minimum Cyborg replicas %d exceeds the PodClique replica limit %d", minimum, math.MaxInt32)
	}
	return ptr.Deref(role.Replicas, int32(minimum)), nil
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
	// An LPU-only engine runs in the Agent Pods with its runtime sidecar, alone or with its draft.
	if agentRuntimeSidecar(component) != nil && pipeline == PipelineHybrid {
		return fmt.Errorf(
			"%w: component %q agent %q sidecar requires an LPU-only build; declare the conductor role instead",
			ErrUnsupportedRuntime, component.ComponentName, commonconsts.RuntimeContainerName,
		)
	}

	conductor := component.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor)

	// Validate the selected hybrid runtime's GPU resources and claim bindings.
	if requiresGPU {
		if pipeline == PipelineSpecDecode {
			return fmt.Errorf("%w: the shared speculative runtime requires LPU-only builds", ErrUnsupportedRuntime)
		}
		template := conductor.PodTemplate
		// The hybrid GPU engine serves only through the LPX sidecar.
		if common.FindContainerByName(template.Spec.InitContainers, commonconsts.RuntimeContainerName) == nil {
			return fmt.Errorf("%w: component %q conductor of a hybrid build requires the LPX sidecar; declare the %q init container",
				ErrUnsupportedRuntime, component.ComponentName, commonconsts.RuntimeContainerName)
		}
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

	// Only a hybrid build has a conductor: the Cyborg that runs its GPU engine.
	if conductor != nil {
		return fmt.Errorf(
			"%w: component %q conductor requires a hybrid build; an LPU-only engine serves from its agent role through the %q sidecar",
			ErrUnsupportedRuntime, component.ComponentName, commonconsts.RuntimeContainerName,
		)
	}

	return nil
}

func expandedModelNames(componentCount int, target bool, draftCount int) []string {
	// Component membership retains the established default/draft/target identities.
	if componentCount == 1 {
		return []string{"default"}
	}
	if target {
		return []string{runtimeModelTarget}
	}
	names := make([]string, draftCount)
	for index := range names {
		names[index] = fmt.Sprintf("%s%d", runtimeModelDraft, index)
	}
	return names
}
