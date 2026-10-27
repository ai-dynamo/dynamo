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

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
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

// RenderInput contains the fresh stage templates and runtime settings consumed by rendering.
type RenderInput struct {
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

	// Runtimes place builds without compiler-owned LPU units from the /configs
	// ConfigMaps. A workload whose every build records units reads placement from
	// the builds and the environment instead, so it gets no ConfigMap.
	var (
		configMap        *corev1.ConfigMap
		configHash       string
		cyborgConfigMap  *corev1.ConfigMap
		cyborgConfigHash string
		extraResources   []client.Object
	)
	if !workload.placesLPUUnits() {
		configMap, configHash, err = renderRuntimeConfigMap(plan.ResourcePrefix+"-lpu", resolvedPartitionData(projections))
		if err != nil {
			return nil, err
		}
		extraResources = []client.Object{configMap}
	}

	// Every legacy hybrid Cyborg reads its remote Agent addresses, independently
	// of the LPU family. Preserve the legacy order: Cyborg config first, LPU config last.
	if hybrid && configMap != nil {
		cyborgConfigMap, cyborgConfigHash, err = workload.renderCyborgConfigMap(plan)
		if err != nil {
			return nil, err
		}
		extraResources = []client.Object{cyborgConfigMap, configMap}
	}

	// Consume the independently rendered conductor without copying Agent startup or placement.
	var conductor *grovev1alpha1.PodCliqueTemplateSpec
	if conductorTemplateName != "" {
		container := common.FindContainerByName(conductorTemplate.Spec.Containers, commonconsts.MainContainerName)
		if err := applyModelPaths(container, projections, modelStoragePath); err != nil {
			return nil, err
		}
		annotations := roleAnnotations(conductorTemplate.Annotations, lpxv1alpha1.PodRoleConductor, workloadDigest)
		if configMap != nil {
			annotations[v1alpha1.AnnotationExtraResourcesHash] = configHash
		}
		conductor = &grovev1alpha1.PodCliqueTemplateSpec{
			Name:        conductorTemplateName,
			Labels:      conductorTemplate.Labels,
			Annotations: annotations,
			Spec: grovev1alpha1.PodCliqueSpec{
				RoleName:     conductorTemplateName,
				PodSpec:      conductorTemplate.Spec,
				Replicas:     1,
				MinAvailable: new(int32(1)),
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
			// Agent template is neither mutated nor validated.
			if agent.Replicas == 0 {
				if conductorSpec != nil {
					configureLPUConductorPod(conductorSpec, configMap, allocation)
				}
				continue
			}

			// Publish the model path, and for unit placement the GPU-local units,
			// before template-owned hybrid Agent bindings.
			if hybrid {
				container := common.FindContainerByName(template.Spec.Containers, commonconsts.MainContainerName)
				if err := applyModelPaths(container, projections, modelStoragePath); err != nil {
					return nil, fmt.Errorf("stage %s: %w", stage, err)
				}
				if projection.placesLPUUnits() {
					applyLocalPartitionIDs(container, projection)
				}
			} else {
				storagePath, err := lpuModelStoragePath(template.Spec)
				if err != nil {
					return nil, fmt.Errorf("stage %s: %w", stage, err)
				}
				if storagePath != modelStoragePath {
					return nil, fmt.Errorf("stage %s must use the Conductor model-storage mount path %q", stage, modelStoragePath)
				}
			}
			configureLPURolePods(&template.Spec, conductorSpec, projection, configMap, allocation)
		}
		if agent.Replicas == 0 {
			continue
		}
		podSpec := template.Spec
		if index+1 < len(projections) && stage == projections[index+1].stage {
			podSpec = *podSpec.DeepCopy()
		}

		annotations := roleAnnotations(maps.Clone(template.Annotations), lpxv1alpha1.PodRoleAgent, projection.Digest().String())
		if configMap != nil {
			annotations[v1alpha1.AnnotationExtraResourcesHash] = configHash
		}
		annotations[lpxv1alpha1.PodModelAnnotation] = projection.model
		annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation] = projection.CompilerSnapshotDigest()
		annotations[WorkloadModeAnnotation] = string(projection.schedulerWorkloadMode())
		replicas := int32(agent.Replicas)
		rendered.Cliques = append(rendered.Cliques, &grovev1alpha1.PodCliqueTemplateSpec{
			Name:        agent.TemplateName,
			Labels:      maps.Clone(template.Labels),
			Annotations: annotations,
			Spec: grovev1alpha1.PodCliqueSpec{
				RoleName:     agent.TemplateName,
				PodSpec:      podSpec,
				Replicas:     replicas,
				MinAvailable: new(replicas),
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

		if err := configureHybridCyborg(
			cyborg,
			projections[0],
			workloadDigest,
			modelStoragePath,
			plan.Agents[0].TemplateName,
			agentTemplateNames,
			cyborgConfigMap,
			cyborgConfigHash,
		); err != nil {
			return nil, err
		}

		cyborg.Spec.MinAvailable = new(int32(1))
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

	// Keep every LPX role in one backend gang, including the KAI fallback roles,
	// and refuse a config mount that no rendered volume backs.
	for _, clique := range rendered.Cliques {
		clique.Spec.PodSpec.SchedulerName = v1alpha1.LPXSchedulerName
		if err := requireDeclaredConfigVolume(clique); err != nil {
			return nil, err
		}
	}

	rendered.Resources = extraResources
	return rendered, nil
}

// configureLPURolePods consumes fresh, independently owned Agent and conductor
// specs. Agent and projection are nonnil, and projection has validated partitions;
// nil conductor means no emitted launcher. A nil configMap means the workload's
// runtimes read placement from LPU units, so no config volume is added.
func configureLPURolePods(
	agentPodSpec, conductorPodSpec *corev1.PodSpec,
	projection *ModelProjection,
	configMap *corev1.ConfigMap,
	allocation string,
) {
	family := projection.configuredBuild.family
	if configMap != nil {
		ensureLPUConfigVolume(agentPodSpec, configMap.Name)
	}
	configureAgentScheduling(agentPodSpec, family, projection.partitions[0].devicesPerNode)
	if conductorPodSpec != nil {
		configureLPUConductorPod(conductorPodSpec, configMap, allocation)
	}
}

// configureLPUConductorPod shapes the conductor's LPX-owned fields. Placement is
// already resolved. A nil configMap means no config volume, as for
// configureLPURolePods.
func configureLPUConductorPod(conductorPodSpec *corev1.PodSpec, configMap *corev1.ConfigMap, allocation string) {
	if configMap != nil {
		ensureLPUConfigVolume(conductorPodSpec, configMap.Name)
	}
	configureNodeLocalConductorRuntime(conductorPodSpec, allocation)
}

// requireDeclaredConfigVolume rejects a nonnil clique whose containers mount a
// config volume that its PodSpec lacks. Kubernetes would reject every Pod of the
// clique; rendering fails instead, so the controller keeps the published PCS.
// The operator adds that volume only for workloads that read /configs.
func requireDeclaredConfigVolume(clique *grovev1alpha1.PodCliqueTemplateSpec) error {
	spec := &clique.Spec.PodSpec
	if slices.ContainsFunc(spec.Volumes, func(volume corev1.Volume) bool { return volume.Name == lpuConfigVolumeName }) {
		return nil
	}
	for _, container := range slices.Concat(spec.InitContainers, spec.Containers) {
		if slices.ContainsFunc(container.VolumeMounts, func(mount corev1.VolumeMount) bool { return mount.Name == lpuConfigVolumeName }) {
			return fmt.Errorf(
				"LPX role %s container %s mounts volume %q, which its pod template does not declare; "+
					"the operator provides /configs only for builds without lpuUnits",
				clique.Name, container.Name, lpuConfigVolumeName,
			)
		}
	}
	return nil
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
