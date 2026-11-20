/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"maps"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// MaxRenderedPodCliqueSetBytes keeps headroom below the API server's request-size ceiling for
// admission metadata and transport overhead.
const MaxRenderedPodCliqueSetBytes = 1 << 20

// PodTemplate retains a fresh role template together with its environment sources.
// A nil Environment explicitly retains legacy rendering.
type PodTemplate struct {
	Template    corev1.PodTemplateSpec
	Environment *semanticenv.ContainerLayers
}

// CliqueTemplate retains an independently rendered GPU clique and its environment sources.
type CliqueTemplate struct {
	Template    *grovev1alpha1.PodCliqueTemplateSpec
	Environment *semanticenv.ContainerLayers
}

// RenderInput contains the fresh stage templates and runtime settings consumed by rendering.
type RenderInput struct {
	// MinAvailable is the minimum number of complete workload replicas in the gang.
	MinAvailable *int32
	// Stages contains an independently merged LPU template for every projected stage.
	Stages map[string]PodTemplate
	// Cyborg supplies the GPU clique for hybrid pipelines; nil for LPU-only workloads.
	Cyborg *CliqueTemplate
}

// WorkloadTemplates contains the clique templates and scaling group contributed by one workload.
type WorkloadTemplates struct {
	Cliques      []*grovev1alpha1.PodCliqueTemplateSpec
	ScalingGroup grovev1alpha1.PodCliqueScalingGroupConfig
}

// RenderNodeLocal renders a workload without constructing a PCS. workload
// must be non-nil and is read without mutation. input must have a stage template for
// every component of the workload, and a Cyborg template for a hybrid workload. The function
// may mutate the reference-backed templates in input; callers must pass fresh owned values.
//
//nolint:gocyclo // Rendering is one transactional validation-and-materialization pass.
func RenderNodeLocal(
	workload *Workload,
	input RenderInput,
) (*WorkloadTemplates, error) {
	projections := workload.modelProjections

	// Keep the hybrid GPU clique before the workload's LPU roles.
	hybrid := workload.Pipeline() == PipelineHybrid
	rendered := &WorkloadTemplates{}
	var cyborg *grovev1alpha1.PodCliqueTemplateSpec
	if hybrid {
		cyborg = input.Cyborg.Template
		rendered.Cliques = append(rendered.Cliques, cyborg)
	}

	workloadDigest := workload.Digest().String()
	agentTemplateNames := make([]string, 0, len(projections))
	for index, projection := range projections {
		if projection.agentReplicas > 0 {
			agentTemplateNames = append(agentTemplateNames, workload.agentTemplate(index))
		}
	}

	// The serving component's Agent template owns storage independently of model order.
	modelStoragePath, err := lpuModelStoragePath(input.Stages[workload.ServingComponentName()].Template.Spec)
	if err != nil {
		return nil, err
	}

	// Resolve every model path before changing a role's authored environment.
	paths, err := modelPaths(projections, modelStoragePath)
	if err != nil {
		return nil, err
	}
	if hybrid {
		input.Cyborg.Environment.PrependPod(&cyborg.Spec.PodSpec, "operator.lpx.modelPaths", paths)
	}
	for _, stage := range input.Stages {
		stage.Environment.PrependPod(&stage.Template.Spec, "operator.lpx.modelPaths", paths)
	}

	// Canonical projections keep each component together; consume its last Agent instance.
	var template PodTemplate
	for index, projection := range projections {
		stage := projection.stage
		if index == 0 || stage != projections[index-1].stage {
			template = input.Stages[stage]

			// A component's projections share one partition selection. When it
			// leaves no partition on LPUs, the unused Agent template needs no
			// placement or storage validation.
			if projection.agentReplicas == 0 {
				continue
			}

			// Every stage's storage must agree with the serving stage on the published model paths.
			storagePath, err := lpuModelStoragePath(template.Template.Spec)
			if err != nil {
				return nil, fmt.Errorf("stage %s: %w", stage, err)
			}
			if storagePath != modelStoragePath {
				return nil, fmt.Errorf("stage %s must use the serving model-storage mount path %q", stage, modelStoragePath)
			}

			// Bind the LPX configuration and scheduling to the Agent.
			applyLocalPartitionIDs(common.FindContainerByName(template.Template.Spec.Containers, commonconsts.MainContainerName), projection, template.Environment)
			configureAgentScheduling(&template.Template.Spec, projection.configuredBuild.architecture, projection.configuredBuild.devicesPerNode)
		}
		if projection.agentReplicas == 0 {
			continue
		}
		podSpec := template.Template.Spec
		if index+1 < len(projections) && stage == projections[index+1].stage {
			podSpec = *podSpec.DeepCopy()
		}

		annotations := roleAnnotations(maps.Clone(template.Template.Annotations), lpxv1alpha1.PodRoleAgent, projection.Digest().String())
		annotations[lpxv1alpha1.PodModelAnnotation] = projection.model
		annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation] = projection.CompilerSnapshotDigest()
		annotations[WorkloadModeAnnotation] = string(projection.schedulerWorkloadMode())
		replicas := int32(projection.agentReplicas)
		if err := template.Environment.Apply(&podSpec); err != nil {
			return nil, fmt.Errorf("LPX role %s: %w", workload.agentTemplate(index), err)
		}
		rendered.Cliques = append(rendered.Cliques, &grovev1alpha1.PodCliqueTemplateSpec{
			Name:        workload.agentTemplate(index),
			Labels:      maps.Clone(template.Template.Labels),
			Annotations: annotations,
			Spec: grovev1alpha1.PodCliqueSpec{
				RoleName:     workload.agentTemplate(index),
				PodSpec:      podSpec,
				Replicas:     replicas,
				MinAvailable: new(replicas),
			},
		})
	}

	if hybrid {
		if err := configureHybridCyborg(
			cyborg,
			projections[0],
			workloadDigest,
			modelStoragePath,
			agentTemplateNames,
			input.Cyborg.Environment,
		); err != nil {
			return nil, fmt.Errorf("stage %s conductor: %w", projections[0].stage, err)
		}

		cyborg.Spec.MinAvailable = new(int32(1))
		if err := input.Cyborg.Environment.Apply(&cyborg.Spec.PodSpec); err != nil {
			return nil, fmt.Errorf("LPX role %s: %w", cyborg.Name, err)
		}
	}

	// Each workload contributes its own scaling group to the shared PCS.
	members := agentTemplateNames
	if hybrid {
		members = append(members, workload.ConductorTemplate())
	}
	rendered.ScalingGroup = grovev1alpha1.PodCliqueScalingGroupConfig{
		Name:         workload.scalingGroupTemplate,
		CliqueNames:  members,
		Annotations:  map[string]string{WorkloadDigestAnnotation: workloadDigest},
		Replicas:     new(ptr.Deref(input.MinAvailable, 1)),
		MinAvailable: new(ptr.Deref(input.MinAvailable, 1)),
	}

	// Keep every LPX role in one backend gang, including the KAI fallback roles.
	for _, clique := range rendered.Cliques {
		clique.Spec.PodSpec.SchedulerName = v1alpha1.LPXSchedulerName
	}

	return rendered, nil
}

// roleAnnotations consumes base, allocating it when nil.
func roleAnnotations(base map[string]string, role string, workloadDigest string) map[string]string {
	annotations := base
	if annotations == nil {
		annotations = make(map[string]string)
	}

	// Remove controller-owned role metadata before stamping canonical values.
	for _, key := range []string{
		WorkloadModeAnnotation,
		lpxv1alpha1.CompilerSnapshotDigestAnnotation,
		lpxv1alpha1.PodModelAnnotation,
		lpxv1alpha1.PodPartitionIDAnnotation,
		lpxv1alpha1.PodRankInPartitionAnnotation,
	} {
		delete(annotations, key)
	}
	annotations[WorkloadDigestAnnotation] = workloadDigest
	annotations[lpxv1alpha1.PodRoleAnnotation] = role
	return annotations
}
