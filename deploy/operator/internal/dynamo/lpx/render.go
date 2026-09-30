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

// Render appends the workload's cliques and scaling group to pcs and returns its
// runtime resources. agents holds each component's Dynamo-defaulted Agent template;
// conductor is the Dynamo-defaulted conductor template: the Nova conductor for
// LPU-only pipelines and the Cyborg worker for hybrid pipelines. pcs is modified only
// when rendering succeeds; the workload is read without mutation. The function may
// mutate the reference-backed templates; callers must pass fresh owned values. The
// caller assigns namespaces to the returned resources.
//
//nolint:gocyclo // Rendering is one transactional validation-and-materialization pass.
func (w *Workload) Render(
	pcs *grovev1alpha1.PodCliqueSet,
	agents map[string]corev1.PodTemplateSpec,
	conductor corev1.PodTemplateSpec,
) ([]client.Object, error) {
	models := w.models
	hybrid := w.Pipeline() == PipelineHybrid
	workloadDigest := w.Digest().String()
	agentTemplateNames := make([]string, 0, len(models))
	for _, model := range models {
		agentTemplateNames = append(agentTemplateNames, model.agentTemplate)
	}

	// Nova's conductor owns model storage; hybrid Agents and Cyborg share the serving Agent's mount.
	storageTemplate := conductor
	if hybrid {
		storageTemplate = agents[w.name]
	}
	modelStoragePath, err := lpuModelStoragePath(storageTemplate.Spec)
	if err != nil {
		return nil, err
	}
	configMap, configHash, err := renderRuntimeConfigMap(w.resourcePrefix+"-lpu", resolvedPartitionData(models))
	if err != nil {
		return nil, err
	}

	// Render the optional Cyborg config and construct final resource order once.
	var (
		cyborgConfigMap  *corev1.ConfigMap
		cyborgConfigHash string
	)
	resources := []client.Object{configMap}
	if hybrid && models[0].component.configuredBuild.family.cyborgServerConfig {
		cyborgConfigMap, cyborgConfigHash, err = w.renderCyborgConfigMap()
		if err != nil {
			return nil, err
		}
		// Preserve the legacy graph order: Cyborg config first, LPU config last.
		resources = []client.Object{cyborgConfigMap, configMap}
	}

	// The conductor starts after every Agent and precedes them in the clique list.
	conductorClique := &grovev1alpha1.PodCliqueTemplateSpec{
		Name:   w.conductorTemplate,
		Labels: conductor.Labels,
		Spec: grovev1alpha1.PodCliqueSpec{
			RoleName:     w.conductorTemplate,
			PodSpec:      conductor.Spec,
			Replicas:     w.conductorReplicas,
			MinAvailable: ptr.To(int32(1)),
			StartsAfter:  slices.Clone(agentTemplateNames),
		},
	}
	cliques := []*grovev1alpha1.PodCliqueTemplateSpec{conductorClique}
	if !hybrid {
		container := common.FindContainerByName(conductorClique.Spec.PodSpec.Containers, commonconsts.MainContainerName)
		if err := applyModelPaths(container, models, modelStoragePath); err != nil {
			return nil, err
		}
		conductorClique.Annotations = roleAnnotations(conductor.Annotations, lpxv1alpha1.PodRoleConductor, workloadDigest)
		conductorClique.Annotations[v1alpha1.AnnotationExtraResourcesHash] = configHash
	}

	// Canonical models keep each component together; consume its last Agent instance.
	allocation := strings.Join(agentTemplateNames, ":")
	var template corev1.PodTemplateSpec
	for index, model := range models {
		stage := model.component.name
		if index == 0 || stage != models[index-1].component.name {
			template = agents[stage]

			// Publish the model path before template-owned hybrid Agent bindings.
			if hybrid {
				container := common.FindContainerByName(template.Spec.Containers, commonconsts.MainContainerName)
				if err := applyModelPaths(container, models, modelStoragePath); err != nil {
					return nil, fmt.Errorf("stage %s: %w", stage, err)
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
			var conductorSpec *corev1.PodSpec
			if stage == w.name && !hybrid {
				conductorSpec = &conductorClique.Spec.PodSpec
			}
			if err := configureLPURolePods(&template.Spec, conductorSpec, model, configMap.Name, allocation); err != nil {
				return nil, fmt.Errorf("stage %s: %w", stage, err)
			}
		}
		podSpec := template.Spec
		if index+1 < len(models) && stage == models[index+1].component.name {
			podSpec = *podSpec.DeepCopy()
		}

		annotations := roleAnnotations(maps.Clone(template.Annotations), lpxv1alpha1.PodRoleAgent, model.Digest().String())
		annotations[v1alpha1.AnnotationExtraResourcesHash] = configHash
		annotations[lpxv1alpha1.PodModelAnnotation] = model.name
		annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation] = model.CompilerSnapshotDigest()
		annotations[WorkloadModeAnnotation] = string(model.component.configuredBuild.family.workloadMode(model.component.pipeline))
		replicas := int32(model.component.agentReplicas)
		cliques = append(cliques, &grovev1alpha1.PodCliqueTemplateSpec{
			Name:        model.agentTemplate,
			Labels:      maps.Clone(template.Labels),
			Annotations: annotations,
			Spec: grovev1alpha1.PodCliqueSpec{
				RoleName:     model.agentTemplate,
				PodSpec:      podSpec,
				Replicas:     replicas,
				MinAvailable: ptr.To(replicas),
			},
		})
	}

	// Nova's conductor leads the scaling group; Cyborg workers follow their Agents.
	members := append([]string{w.conductorTemplate}, agentTemplateNames...)
	if hybrid {
		members = append(slices.Clone(agentTemplateNames), w.conductorTemplate)

		// Bind the authored HX Cyborg configuration mount to the generated ConfigMap.
		container := common.FindContainerByName(conductorClique.Spec.PodSpec.Containers, commonconsts.MainContainerName)
		if cyborgConfigMap == nil && slices.ContainsFunc(container.VolumeMounts,
			func(mount corev1.VolumeMount) bool { return mount.Name == lpuConfigVolumeName }) {
			if err := withLPUConfigVolume(&conductorClique.Spec.PodSpec, configMap.Name, true); err != nil {
				return nil, err
			}
		}
		conductorClique.Annotations = conductor.Annotations
		if err := configureHybridCyborg(conductorClique, models[0], workloadDigest, modelStoragePath, cyborgConfigMap, cyborgConfigHash); err != nil {
			return nil, err
		}
	}

	// Each workload contributes its own scaling group to the shared PCS.
	scalingGroup := grovev1alpha1.PodCliqueScalingGroupConfig{
		Name:         w.scalingGroupTemplate,
		CliqueNames:  members,
		Annotations:  map[string]string{WorkloadDigestAnnotation: workloadDigest},
		Replicas:     ptr.To(w.minAvailable),
		MinAvailable: ptr.To(w.minAvailable),
	}

	// Keep every LPX role in one backend gang; discovery selects only the conductor.
	for _, clique := range cliques {
		clique.Spec.PodSpec.SchedulerName = v1alpha1.LPXSchedulerName
		delete(clique.Labels, ServingLabel)
		if clique.Name == w.conductorTemplate {
			if clique.Labels == nil {
				clique.Labels = make(map[string]string)
			}
			clique.Labels[ServingLabel] = commonconsts.KubeLabelValueTrue
		} else {
			delete(clique.Labels, commonconsts.KubeLabelDynamoDiscoveryEnabled)
			delete(clique.Labels, commonconsts.KubeLabelDynamoDiscoveryBackend)
			delete(clique.Labels, commonconsts.KubeLabelDynamoBaseModelHash)
		}
	}

	pcs.Spec.Template.Cliques = append(pcs.Spec.Template.Cliques, cliques...)
	pcs.Spec.Template.PodCliqueScalingGroupConfigs = append(pcs.Spec.Template.PodCliqueScalingGroupConfigs, scalingGroup)
	return resources, nil
}

// configureLPURolePods consumes fresh, independently owned Agent and conductor
// specs. Agent and projection are nonnil, and projection has validated partitions;
// nil conductor means no emitted launcher.
func configureLPURolePods(agentPodSpec, conductorPodSpec *corev1.PodSpec, model *Model, configMapName, allocation string) error {
	// Use this component's manifest geometry for its Agent resources and configuration mount.
	buildFamily := model.component.configuredBuild.family
	if err := withLPUConfigVolume(agentPodSpec, configMapName, buildFamily.configOverrides); err != nil {
		return err
	}
	configureAgentScheduling(agentPodSpec, buildFamily, model.component.configuredBuild.partitions[0].devicesPerNode)

	// Placement is already resolved; shape only the actual conductor's LPX-owned fields.
	if conductorPodSpec != nil {
		if err := withLPUConfigVolume(conductorPodSpec, configMapName, buildFamily.configOverrides); err != nil {
			return err
		}
		configureNodeLocalConductorRuntime(conductorPodSpec, allocation)
	}

	return nil
}

// roleAnnotations consumes base, allocating it when nil.
func roleAnnotations(
	base map[string]string,
	role string,
	workloadDigest string,
) map[string]string {
	annotations := base
	if annotations == nil {
		annotations = make(map[string]string)
	}
	annotations[WorkloadDigestAnnotation] = workloadDigest
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
