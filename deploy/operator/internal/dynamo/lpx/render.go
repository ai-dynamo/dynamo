/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"maps"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// MaxRenderedPodCliqueSetBytes keeps headroom below the API server's request-size ceiling for
// admission metadata and transport overhead.
const MaxRenderedPodCliqueSetBytes = 1 << 20

// PodTemplate retains a fresh role template together with its environment sources.
// A nil Environment explicitly retains legacy rendering.
type PodTemplate struct {
	Template    corev1.PodTemplateSpec
	Environment semanticenv.ContainerLayers
}

// CliqueTemplate retains an independently rendered GPU clique and its environment sources.
type CliqueTemplate struct {
	Template    *grovev1alpha1.PodCliqueTemplateSpec
	Environment semanticenv.ContainerLayers
}

// RenderInput contains the fresh stage templates and runtime settings consumed by rendering.
type RenderInput struct {
	// MinAvailable is the minimum number of complete workload replicas in the gang.
	MinAvailable *int32
	// Stages contains an independently merged LPU template for every projected stage.
	Stages map[string]PodTemplate
	// Conductor supplies a fresh, independently merged template for the shared LPU
	// conductor when plan.ConductorTemplate is nonempty. Otherwise it is nil;
	// hybrid pipelines supply their independently rendered Cyborg clique below.
	Conductor *PodTemplate
	// Cyborg supplies the GPU clique for hybrid pipelines; nil for LPU-only workloads.
	Cyborg *CliqueTemplate
}

// WorkloadTemplates contains only the templates and resources contributed by one workload.
type WorkloadTemplates struct {
	Cliques      []*grovev1alpha1.PodCliqueTemplateSpec
	ScalingGroup grovev1alpha1.PodCliqueScalingGroupConfig
	Resources    []client.Object
}

// RenderNodeLocal renders a workload without constructing a PCS. workload
// and plan must be non-nil and are read without mutation. The function may mutate
// the reference-backed templates in input; callers must pass fresh owned values.
// The caller assigns namespaces to the returned resources.
//
//nolint:gocyclo // Rendering is one transactional validation-and-materialization pass.
func RenderNodeLocal(
	workload *Workload,
	plan *MaterializationPlan,
	input RenderInput,
) (*WorkloadTemplates, error) {
	projections := workload.modelProjections

	// Require every paired role input before any template mutation takes place.
	for _, projection := range projections {
		if _, found := input.Stages[projection.stage]; !found {
			return nil, fmt.Errorf("LPX rendering requires a template for stage %s", projection.stage)
		}
	}

	// Keep the hybrid GPU clique before the workload's LPU roles.
	hybrid := projections[0].pipeline == PipelineHybrid
	rendered := &WorkloadTemplates{}
	var cyborg *grovev1alpha1.PodCliqueTemplateSpec
	if hybrid {
		if input.Cyborg == nil {
			return nil, fmt.Errorf("LPX rendering requires an explicit Cyborg template")
		}
		cyborg = input.Cyborg.Template
		rendered.Cliques = append(rendered.Cliques, cyborg)
	}

	workloadDigest := workload.Digest().String()
	agentTemplateNames := make([]string, 0, len(plan.Agents))
	for _, agent := range plan.Agents {
		if agent.Replicas > 0 {
			agentTemplateNames = append(agentTemplateNames, agent.TemplateName)
		}
	}
	conductorTemplateName := plan.ConductorTemplate
	allocation := strings.Join(agentTemplateNames, ":")

	// The serving component owns conductor metadata and storage independently of model
	// order. Without a conductor, the serving Agent template owns storage.
	conductorStage := workload.ServingComponentName()
	conductorTemplate := input.Conductor
	storageTemplate := input.Stages[conductorStage].Template
	if plan.ConductorTemplate != "" {
		if conductorTemplate == nil {
			return nil, fmt.Errorf("LPX rendering requires an explicit conductor template")
		}
		storageTemplate = conductorTemplate.Template
	}
	modelStoragePath, err := lpuModelStoragePath(storageTemplate.Spec)
	if err != nil {
		return nil, err
	}

	// Resolve every model path before changing a role's authored environment.
	paths, err := modelPaths(projections, modelStoragePath)
	if err != nil {
		return nil, err
	}
	if conductorTemplate != nil {
		conductorTemplate.Environment.PrependPod(&conductorTemplate.Template.Spec, "operator.lpx.modelPaths", paths)
	}
	if hybrid {
		input.Cyborg.Environment.PrependPod(&cyborg.Spec.PodSpec, "operator.lpx.modelPaths", paths)
	}
	for _, stage := range input.Stages {
		stage.Environment.PrependPod(&stage.Template.Spec, "operator.lpx.modelPaths", paths)
	}

	// Share partition configuration across the workload's roles.
	configMap, configHash, err := renderRuntimeConfigMap(plan.ResourcePrefix+"-lpu", resolvedPartitionData(projections))
	if err != nil {
		return nil, err
	}
	// Every hybrid Cyborg reads its remote Agent addresses, independently of the
	// LPU family. Render the optional Cyborg config and construct final resource order once.
	var (
		cyborgConfigMap  *corev1.ConfigMap
		cyborgConfigHash string
		extraResources   []client.Object
	)
	if hybrid {
		cyborgConfigMap, cyborgConfigHash, err = workload.renderCyborgConfigMap(plan)
		if err != nil {
			return nil, err
		}
		// Preserve the legacy graph order: Cyborg config first, LPU config last.
		extraResources = []client.Object{cyborgConfigMap, configMap}
	} else {
		extraResources = []client.Object{configMap}
	}

	// Consume the independently rendered conductor without copying Agent startup or placement.
	var conductor *grovev1alpha1.PodCliqueTemplateSpec
	if conductorTemplateName != "" {
		annotations := roleAnnotations(conductorTemplate.Template.Annotations, lpxv1alpha1.PodRoleConductor, workloadDigest)
		annotations[v1alpha1.AnnotationExtraResourcesHash] = configHash
		conductor = &grovev1alpha1.PodCliqueTemplateSpec{
			Name:        conductorTemplateName,
			Labels:      conductorTemplate.Template.Labels,
			Annotations: annotations,
			Spec: grovev1alpha1.PodCliqueSpec{
				RoleName:     conductorTemplateName,
				PodSpec:      conductorTemplate.Template.Spec,
				Replicas:     1,
				MinAvailable: new(int32(1)),
				StartsAfter:  agentTemplateNames,
			},
		}
		rendered.Cliques = append(rendered.Cliques, conductor)
	}

	// Canonical projections keep each component together; consume its last Agent instance.
	var template PodTemplate
	for index, projection := range projections {
		stage := projection.stage
		agent := plan.Agents[index]
		if index == 0 || stage != projections[index-1].stage {
			template = input.Stages[stage]
			var conductorSpec *corev1.PodSpec
			if stage == conductorStage && conductor != nil {
				conductorSpec = &conductor.Spec.PodSpec
			}

			// A component's projections share one partition selection. When it
			// leaves no partition on LPUs, shape only the conductor; the unused
			// Agent template needs no placement or storage validation.
			if agent.Replicas == 0 {
				if conductorSpec != nil {
					configureLPUConductorPod(conductorSpec, configMap.Name, allocation, conductorTemplate.Environment)
				}
				continue
			}

			// Agent and conductor storage must agree on the published model paths.
			if !hybrid {
				storagePath, err := lpuModelStoragePath(template.Template.Spec)
				if err != nil {
					return nil, fmt.Errorf("stage %s: %w", stage, err)
				}
				if storagePath != modelStoragePath {
					return nil, fmt.Errorf("stage %s must use the Conductor model-storage mount path %q", stage, modelStoragePath)
				}
			}

			// Bind the LPX configuration and scheduling to the Agent and any conductor.
			var conductorEnvironment semanticenv.ContainerLayers
			if conductorTemplate != nil {
				conductorEnvironment = conductorTemplate.Environment
			}
			configureLPURolePods(&template.Template.Spec, conductorSpec, projection, configMap.Name, allocation, conductorEnvironment)
		}
		if agent.Replicas == 0 {
			continue
		}
		podSpec := template.Template.Spec
		if index+1 < len(projections) && stage == projections[index+1].stage {
			podSpec = *podSpec.DeepCopy()
		}

		annotations := roleAnnotations(maps.Clone(template.Template.Annotations), lpxv1alpha1.PodRoleAgent, projection.Digest().String())
		annotations[v1alpha1.AnnotationExtraResourcesHash] = configHash
		annotations[lpxv1alpha1.PodModelAnnotation] = projection.model
		annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation] = projection.CompilerSnapshotDigest()
		annotations[WorkloadModeAnnotation] = string(projection.schedulerWorkloadMode())
		replicas := int32(agent.Replicas)
		if err := template.Environment.Apply(&podSpec); err != nil {
			return nil, fmt.Errorf("LPX role %s: %w", agent.TemplateName, err)
		}
		rendered.Cliques = append(rendered.Cliques, &grovev1alpha1.PodCliqueTemplateSpec{
			Name:        agent.TemplateName,
			Labels:      maps.Clone(template.Template.Labels),
			Annotations: annotations,
			Spec: grovev1alpha1.PodCliqueSpec{
				RoleName:     agent.TemplateName,
				PodSpec:      podSpec,
				Replicas:     replicas,
				MinAvailable: new(replicas),
			},
		})
	}

	// The conductor's environment is complete after all serving-stage bindings.
	if conductor != nil {
		if err := conductorTemplate.Environment.Apply(&conductor.Spec.PodSpec); err != nil {
			return nil, fmt.Errorf("LPX role %s: %w", conductor.Name, err)
		}
	}

	selectedTemplateNames := agentTemplateNames
	if conductorTemplateName != "" {
		selectedTemplateNames = append([]string{conductorTemplateName}, selectedTemplateNames...)
	}

	if hybrid {
		// Bound GPU hostnames using the rendered width of the last workload replica.
		if err := plan.validatePodHostname("Cyborg", plan.CyborgTemplate, int(cyborg.Spec.Replicas)-1); err != nil {
			return nil, err
		}

		if err := configureHybridCyborg(
			cyborg,
			projections[0],
			workloadDigest,
			modelStoragePath,
			agentTemplateNames,
			cyborgConfigMap,
			cyborgConfigHash,
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
	members := selectedTemplateNames
	if hybrid {
		members = append(members, plan.CyborgTemplate)
	}
	rendered.ScalingGroup = grovev1alpha1.PodCliqueScalingGroupConfig{
		Name:         plan.ScalingGroupTemplate,
		CliqueNames:  members,
		Annotations:  map[string]string{WorkloadDigestAnnotation: workloadDigest},
		Replicas:     new(ptr.Deref(input.MinAvailable, 1)),
		MinAvailable: new(ptr.Deref(input.MinAvailable, 1)),
	}

	// Keep every LPX role in one backend gang, including the KAI fallback roles.
	for _, clique := range rendered.Cliques {
		clique.Spec.PodSpec.SchedulerName = v1alpha1.LPXSchedulerName
	}

	rendered.Resources = extraResources
	return rendered, nil
}

// configureLPURolePods consumes fresh, independently owned Agent and conductor
// specs. Agent and projection are nonnil, and projection has validated partitions;
// nil conductor means no emitted launcher.
func configureLPURolePods(agentPodSpec, conductorPodSpec *corev1.PodSpec, projection *ModelProjection, configMapName, allocation string, conductorEnvironment semanticenv.ContainerLayers) {
	family := projection.configuredBuild.family
	ensureLPUConfigVolume(agentPodSpec, configMapName)
	configureAgentScheduling(agentPodSpec, family, projection.partitions[0].devicesPerNode)
	if conductorPodSpec != nil {
		configureLPUConductorPod(conductorPodSpec, configMapName, allocation, conductorEnvironment)
	}
}

// configureLPUConductorPod shapes the conductor's LPX-owned fields. Placement is
// already resolved.
func configureLPUConductorPod(conductorPodSpec *corev1.PodSpec, configMapName, allocation string, environment semanticenv.ContainerLayers) {
	ensureLPUConfigVolume(conductorPodSpec, configMapName)
	configureNodeLocalConductorRuntime(conductorPodSpec, allocation, environment)
}

// roleAnnotations consumes base, allocating it when nil.
func roleAnnotations(
	base map[string]string,
	role string,
	workloadDigest string,
) map[string]string {
	annotations := workloadAnnotations(base, workloadDigest)
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
	annotations[lpxv1alpha1.PodRoleAnnotation] = role
	return annotations
}

func workloadAnnotations(base map[string]string, digest string) map[string]string {
	if base == nil {
		base = make(map[string]string)
	}
	base[WorkloadDigestAnnotation] = digest
	return base
}
