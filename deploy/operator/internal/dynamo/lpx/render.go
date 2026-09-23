/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// MaxRenderedPodCliqueSetBytes keeps headroom below the API server's request-size ceiling for
// admission metadata and transport overhead.
const MaxRenderedPodCliqueSetBytes = 1 << 20

// StageLabel records the authored LPX component association.
const StageLabel = "lpx.nvidia.com/stage"

// RenderInput contains the fresh stage templates and runtime settings consumed by rendering.
type RenderInput struct {
	// CyborgConfigMap is rendered before the user's Cyborg defaults are merged.
	CyborgConfigMap *corev1.ConfigMap
	// MinAvailable is the minimum number of complete workload replicas in the gang.
	MinAvailable *int32
	// Stages contains an independently merged LPU template for every projected stage.
	Stages map[string]corev1.PodTemplateSpec
	// Conductor supplies a fresh, independently merged template for the shared LPU
	// conductor. Required for non-hybrid pipelines; hybrid pipelines leave it nil
	// and supply their independently rendered Cyborg clique below.
	Conductor *corev1.PodTemplateSpec
	// Cyborg supplies the GPU clique for hybrid pipelines; nil for LPU-only workloads.
	Cyborg *grovev1alpha1.PodCliqueTemplateSpec
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

	// Keep the hybrid GPU clique before the workload's LPU roles.
	hybrid := projections[0].pipeline == PipelineLPX
	rendered := &WorkloadTemplates{}
	cyborg := input.Cyborg
	if hybrid {
		rendered.Cliques = append(rendered.Cliques, cyborg)
	}

	workloadDigest := workload.Digest().String()
	agentTemplateNames := make([]string, 0, len(plan.Agents))
	for _, agent := range plan.Agents {
		agentTemplateNames = append(agentTemplateNames, agent.TemplateName)
	}
	conductorTemplateName := plan.ConductorTemplate
	allocation := strings.Join(agentTemplateNames, ",")
	if projections[0].pipeline == PipelineSpecDecode {
		allocation = strings.Join(agentTemplateNames, ":")
	}

	// The serving component owns conductor metadata and storage independently of model order.
	conductorStage := workload.ServingComponentName()
	conductorTemplate := input.Conductor
	storageTemplate := input.Stages[conductorStage]
	if !hybrid {
		if conductorTemplate == nil {
			return nil, fmt.Errorf("LPX rendering requires an explicit conductor template")
		}
		storageTemplate = *conductorTemplate
	}
	modelStoragePath, err := lpuModelStoragePath(storageTemplate.Spec)
	if err != nil {
		return nil, err
	}
	configMap, err := renderRuntimeConfigMap(plan.ResourcePrefix+"-lpu", resolvedPartitionData(projections))
	if err != nil {
		return nil, err
	}
	v2HybridRuntime := projections[0].configuredBuild.Family == BuildFamilyXT &&
		projections[0].pipeline == PipelineLPX

	// Render the optional Cyborg config and construct final resource order once.
	var (
		cyborgConfigMap *corev1.ConfigMap
		extraResources  []client.Object
	)
	if v2HybridRuntime {
		cyborgConfigMap = input.CyborgConfigMap
		// Preserve the legacy graph order: Cyborg config first, LPU config last.
		extraResources = []client.Object{cyborgConfigMap, configMap}
	} else {
		extraResources = []client.Object{configMap}
	}

	configHash := LPUConfigMapHash(configMap)

	// Consume the independently rendered conductor without copying Agent startup or placement.
	var conductor *grovev1alpha1.PodCliqueTemplateSpec
	if conductorTemplateName != "" {
		container := common.FindContainerByName(conductorTemplate.Spec.Containers, commonconsts.MainContainerName)
		if err := applyModelPaths(container, projections, modelStoragePath); err != nil {
			return nil, err
		}
		annotations := roleAnnotations(conductorTemplate.Annotations, lpxv1alpha1.PodRoleConductor, workloadDigest)
		annotations[commonconsts.AnnotationExtraResourcesHash] = configHash
		conductor = &grovev1alpha1.PodCliqueTemplateSpec{
			Name:        conductorTemplateName,
			Labels:      maps.Clone(conductorTemplate.Labels),
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
		if index == 0 || stage != projections[index-1].stage {
			template = input.Stages[stage]
			storagePath, err := lpuModelStoragePath(template.Spec)
			if err != nil {
				return nil, fmt.Errorf("stage %s: %w", stage, err)
			}
			if storagePath != modelStoragePath {
				return nil, fmt.Errorf("stage %s must use the Conductor model-storage mount path %q", stage, modelStoragePath)
			}

			// Publish the model path before template-owned hybrid Agent bindings.
			if hybrid {
				container := common.FindContainerByName(template.Spec.Containers, commonconsts.MainContainerName)
				if err := applyModelPaths(container, projections, modelStoragePath); err != nil {
					return nil, fmt.Errorf("stage %s: %w", stage, err)
				}
			}
			var conductorSpec *corev1.PodSpec
			if stage == conductorStage && conductor != nil {
				conductorSpec = &conductor.Spec.PodSpec
			}
			if err := configureLPURolePods(&template.Spec, conductorSpec, workload, configMap.Name, allocation); err != nil {
				return nil, fmt.Errorf("stage %s: %w", stage, err)
			}
		}
		podSpec := template.Spec
		if index+1 < len(projections) && stage == projections[index+1].stage {
			podSpec = *podSpec.DeepCopy()
		}

		annotations := roleAnnotations(template.Annotations, lpxv1alpha1.PodRoleAgent, projection.Digest().String())
		annotations[commonconsts.AnnotationExtraResourcesHash] = configHash
		annotations[lpxv1alpha1.PodModelAnnotation] = projection.model
		annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation] = projection.CompilerSnapshotDigest()
		annotations[WorkloadModeAnnotation] = string(projection.schedulerWorkloadMode())
		agent := plan.Agents[index]
		replicas := int32(agent.Replicas)
		rendered.Cliques = append(rendered.Cliques, &grovev1alpha1.PodCliqueTemplateSpec{
			Name:        agent.TemplateName,
			Labels:      maps.Clone(template.Labels),
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

		// Bind the authored HX Cyborg configuration mount to the generated ConfigMap.
		container := common.FindContainerByName(cyborg.Spec.PodSpec.Containers, commonconsts.MainContainerName)
		if cyborgConfigMap == nil && slices.ContainsFunc(container.VolumeMounts,
			func(mount corev1.VolumeMount) bool { return mount.Name == lpuConfigVolumeName }) {
			if err := withLPUConfigVolume(&cyborg.Spec.PodSpec, configMap.Name, true); err != nil {
				return nil, err
			}
		}
		if err := configureHybridCyborg(
			cyborg,
			projections[0],
			workloadDigest,
			modelStoragePath,
			agentTemplateNames,
			cyborgConfigMap,
		); err != nil {
			return nil, err
		}

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
		clique.Spec.PodSpec.SchedulerName = SchedulerName
	}

	rendered.Resources = extraResources
	return rendered, nil
}

// configureLPURolePods consumes fresh, independently owned Agent and conductor
// specs. Agent and workload are nonnil; nil conductor means no emitted launcher.
func configureLPURolePods(agentPodSpec, conductorPodSpec *corev1.PodSpec, workload *Workload, configMapName, allocation string) error {
	if err := withLPUConfigVolume(agentPodSpec, configMapName, workload.BuildFamily() == BuildFamilyXT); err != nil {
		return err
	}
	configureAgentScheduling(agentPodSpec, workload.BuildFamily())
	// Placement is already resolved; shape only the actual conductor's LPX-owned fields.
	if conductorPodSpec != nil {
		stripLPUResources(conductorPodSpec)
		if err := withLPUConfigVolume(conductorPodSpec, configMapName, workload.BuildFamily() == BuildFamilyXT); err != nil {
			return err
		}
		configureNodeLocalConductorRuntime(conductorPodSpec, allocation)
	}
	configureAgentIdentity(agentPodSpec)

	return nil
}

func roleAnnotations(
	base map[string]string,
	role string,
	workloadDigest string,
) map[string]string {
	annotations := workloadAnnotations(maps.Clone(base), workloadDigest)
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

func appendUnique(existing []string, values ...string) []string {
	for _, value := range values {
		if !slices.Contains(existing, value) {
			existing = append(existing, value)
		}
	}
	return existing
}
