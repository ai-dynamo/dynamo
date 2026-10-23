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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// MaxRenderedPodCliqueSetBytes keeps headroom below the API server's request-size ceiling for
// admission metadata and transport overhead.
const MaxRenderedPodCliqueSetBytes = 1 << 20

// RenderInput contains the fresh stage templates and runtime settings consumed by rendering.
type RenderInput struct {
	// MinAvailable is the minimum number of complete workload replicas in the gang.
	MinAvailable *int32
	// Stages contains an independently merged LPU template for every projected stage.
	Stages map[string]corev1.PodTemplateSpec
	// Conductor supplies a fresh, independently merged template for the shared LPU
	// conductor when plan.ConductorTemplate is nonempty. Otherwise it is nil;
	// hybrid pipelines supply their independently rendered Cyborg clique below.
	Conductor *corev1.PodTemplateSpec
	// Cyborg supplies the GPU clique for hybrid pipelines; nil for LPU-only workloads.
	Cyborg *grovev1alpha1.PodCliqueTemplateSpec
	// TopologyDeployment enables rank-0 topology discovery when nonempty: it is
	// the DynamoGraphDeployment UID that every Agent and Cyborg container
	// receives as LPX_DEPLOYMENT. Set it only for a hybrid workload whose Agents
	// have replicas and whose Agent and Cyborg templates declare the "runtime"
	// sidecar; empty leaves the rendering unchanged.
	TopologyDeployment string
}

// WorkloadTemplates contains only the templates and resources contributed by one workload.
type WorkloadTemplates struct {
	Cliques      []*grovev1alpha1.PodCliqueTemplateSpec
	ScalingGroup grovev1alpha1.PodCliqueScalingGroupConfig
	// PublishesTopology reports that Pod 0 of each allocation's Agent clique,
	// labeled TopologyPublisherLabel, publishes the allocation's topology.
	PublishesTopology bool
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

	// Keep the hybrid GPU clique before the workload's LPU roles.
	hybrid := projections[0].pipeline == PipelineHybrid
	rendered := &WorkloadTemplates{}
	cyborg := input.Cyborg
	if hybrid {
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
	storageTemplate := input.Stages[conductorStage]
	if plan.ConductorTemplate != "" {
		if conductorTemplate == nil {
			return nil, fmt.Errorf("LPX rendering requires an explicit conductor template")
		}
		storageTemplate = *conductorTemplate
	}
	modelStoragePath, err := lpuModelStoragePath(storageTemplate.Spec)
	if err != nil {
		return nil, err
	}

	// Publish model paths to every container before role-specific bindings are prepended.
	containerLists := make([][]corev1.Container, 0, 2*(len(input.Stages)+2))
	if conductorTemplate != nil {
		containerLists = append(containerLists, conductorTemplate.Spec.Containers, conductorTemplate.Spec.InitContainers)
	}
	if hybrid {
		containerLists = append(containerLists, cyborg.Spec.PodSpec.Containers, cyborg.Spec.PodSpec.InitContainers)
	}
	for _, template := range input.Stages {
		containerLists = append(containerLists, template.Spec.Containers, template.Spec.InitContainers)
	}
	if err := applyModelPaths(containerLists, projections, modelStoragePath); err != nil {
		return nil, err
	}

	// Consume the independently rendered conductor without copying Agent startup or placement.
	var conductor *grovev1alpha1.PodCliqueTemplateSpec
	if conductorTemplateName != "" {
		annotations := roleAnnotations(conductorTemplate.Annotations, lpxv1alpha1.PodRoleConductor, workloadDigest)
		conductor = &grovev1alpha1.PodCliqueTemplateSpec{
			Name:        conductorTemplateName,
			Labels:      conductorTemplate.Labels,
			Annotations: annotations,
			Spec: grovev1alpha1.PodCliqueSpec{
				RoleName:     conductorTemplateName,
				PodSpec:      conductorTemplate.Spec,
				Replicas:     1,
				MinAvailable: ptr.To(int32(1)),
				StartsAfter:  agentTemplateNames,
			},
		}
		rendered.Cliques = append(rendered.Cliques, conductor)
	}

	// Canonical projections keep each component together; consume its last Agent instance.
	var template corev1.PodTemplateSpec
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
					configureLPUConductorPod(conductorSpec, allocation)
				}
				continue
			}

			// Agent and conductor storage must agree on the published model paths.
			if !hybrid {
				storagePath, err := lpuModelStoragePath(template.Spec)
				if err != nil {
					return nil, fmt.Errorf("stage %s: %w", stage, err)
				}
				if storagePath != modelStoragePath {
					return nil, fmt.Errorf("stage %s must use the Conductor model-storage mount path %q", stage, modelStoragePath)
				}
			}

			// Bind the LPX scheduling to the Agent and any conductor.
			configureLPURolePods(&template.Spec, conductorSpec, projection, allocation)

			// The hybrid Agent's runtime sidecar publishes the allocation's topology from Pod 0.
			if input.TopologyDeployment != "" {
				if err := configureTopologyPublisher(&template.Spec, input.TopologyDeployment); err != nil {
					return nil, fmt.Errorf("stage %s: %w", stage, err)
				}
				rendered.PublishesTopology = true
			}
		}
		if agent.Replicas == 0 {
			continue
		}
		podSpec := template.Spec
		if index+1 < len(projections) && stage == projections[index+1].stage {
			podSpec = *podSpec.DeepCopy()
		}

		container := common.FindContainerByName(podSpec.Containers, commonconsts.MainContainerName)
		if err := applyRuntimeSelection(container, projection, modelStoragePath); err != nil {
			return nil, err
		}
		annotations := roleAnnotations(maps.Clone(template.Annotations), lpxv1alpha1.PodRoleAgent, projection.Digest().String())
		annotations[lpxv1alpha1.PodModelAnnotation] = projection.model
		annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation] = projection.CompilerSnapshotDigest()
		annotations[WorkloadModeAnnotation] = string(projection.schedulerWorkloadMode())
		labels := maps.Clone(template.Labels)
		if input.TopologyDeployment != "" {
			if labels == nil {
				labels = make(map[string]string, 1)
			}
			labels[TopologyPublisherLabel] = commonconsts.KubeLabelValueTrue
		}
		replicas := int32(agent.Replicas)
		rendered.Cliques = append(rendered.Cliques, &grovev1alpha1.PodCliqueTemplateSpec{
			Name:        agent.TemplateName,
			Labels:      labels,
			Annotations: annotations,
			Spec: grovev1alpha1.PodCliqueSpec{
				RoleName:     agent.TemplateName,
				PodSpec:      podSpec,
				Replicas:     replicas,
				MinAvailable: ptr.To(replicas),
			},
		})
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

		container := common.FindContainerByName(cyborg.Spec.PodSpec.Containers, commonconsts.MainContainerName)
		if err := configureHybridCyborg(
			cyborg,
			projections[0],
			workloadDigest,
			modelStoragePath,
			agentTemplateNames,
		); err != nil {
			return nil, fmt.Errorf("stage %s conductor: %w", projections[0].stage, err)
		}

		// Each Cyborg reads the Agent leader's topology through its own runtime sidecar.
		if input.TopologyDeployment != "" {
			if err := configureTopologyConsumer(&cyborg.Spec.PodSpec, input.TopologyDeployment); err != nil {
				return nil, fmt.Errorf("stage %s conductor: %w", projections[0].stage, err)
			}
		}

		if err := applyRuntimeSelection(container, projections[0], modelStoragePath); err != nil {
			return nil, err
		}
		applyOwnedEnv(container, []corev1.EnvVar{{
			Name:  allocationEnvVar,
			Value: allocation,
		}})
		cyborg.Spec.MinAvailable = ptr.To(int32(1))
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
		Replicas:     ptr.To(ptr.Deref(input.MinAvailable, 1)),
		MinAvailable: ptr.To(ptr.Deref(input.MinAvailable, 1)),
	}

	// Keep every LPX role in one backend gang, including the KAI fallback roles.
	for _, clique := range rendered.Cliques {
		clique.Spec.PodSpec.SchedulerName = v1alpha1.LPXSchedulerName
	}

	return rendered, nil
}

// configureLPURolePods consumes fresh, independently owned Agent and conductor
// specs. Agent and projection are nonnil, and projection has validated partitions;
// nil conductor means no emitted launcher.
func configureLPURolePods(agentPodSpec, conductorPodSpec *corev1.PodSpec, projection *ModelProjection, allocation string) {
	family := projection.configuredBuild.family
	configureAgentScheduling(agentPodSpec, family, projection.partitions[0].devicesPerNode)
	if conductorPodSpec != nil {
		configureLPUConductorPod(conductorPodSpec, allocation)
	}
}

// configureLPUConductorPod shapes the conductor's LPX-owned fields. Placement is
// already resolved.
func configureLPUConductorPod(conductorPodSpec *corev1.PodSpec, allocation string) {
	configureNodeLocalConductorRuntime(conductorPodSpec, allocation)
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
		v1alpha1.AnnotationExtraResourcesHash,
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
