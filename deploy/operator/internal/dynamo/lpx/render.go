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

const (
	// StageLabel records the authored LPX component association.
	StageLabel = "lpx.nvidia.com/stage"
	// ExecutionRoleLabel distinguishes internal LPU and GPU Pod templates.
	ExecutionRoleLabel = "lpx.nvidia.com/execution-role"
)

// RenderInput contains the fresh stage templates and runtime settings consumed by rendering.
type RenderInput struct {
	// CyborgConfigMap is rendered before the user's Cyborg defaults are merged.
	CyborgConfigMap *corev1.ConfigMap
	// MinAvailable is the minimum number of complete engine replicas in the gang.
	MinAvailable *int32
	// Stages contains an independently merged LPU template for every projected stage.
	Stages map[string]corev1.PodTemplateSpec
	// Conductor supplies a fresh, independently merged template for the shared LPU
	// conductor. Nil reuses the serving component's agent template. Hybrid pipelines
	// do not emit this conductor.
	Conductor *corev1.PodTemplateSpec
	// SSHSecretName names the Secret containing runtime SSH credentials.
	SSHSecretName string
}

// RenderSelectedNodeLocal consumes fresh graph and render inputs. pcs, workload,
// and plan must be non-nil. The function mutates pcs and may mutate the
// reference-backed stage and conductor template contents in input; callers must
// pass independently owned values. pcs contains only the optional Cyborg clique
// rendered from the same workload, with no scaling groups. workload and plan are
// read without mutation.
//
//nolint:gocyclo // Rendering is one transactional validation-and-materialization pass.
func RenderSelectedNodeLocal(
	pcs *grovev1alpha1.PodCliqueSet,
	workload *SelectedWorkload,
	plan *MaterializationPlan,
	input RenderInput,
) ([]client.Object, error) {
	projections := workload.modelProjections

	// Hybrid input owns one GPU clique before the LPU roles are appended.
	hybrid := projections[0].pipeline == PipelineLPX
	var cyborg *grovev1alpha1.PodCliqueTemplateSpec
	if hybrid {
		cyborg = pcs.Spec.Template.Cliques[0]
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
	namespace := pcs.Namespace

	// The serving component owns conductor metadata and storage independently of model order.
	conductorStage := projections[len(projections)-1].stage
	conductorTemplate := input.Stages[conductorStage]
	if input.Conductor != nil {
		conductorTemplate = *input.Conductor
	}
	modelStorage, err := lpuModelStorageBinding(conductorTemplate.Spec)
	if err != nil {
		return nil, err
	}
	modelStorage.volume = *modelStorage.volume.DeepCopy()
	modelStorage.mount = *modelStorage.mount.DeepCopy()
	configMap, err := renderLPUConfigMap(namespace, plan.PodCliqueSetName, modelStorage.mount.MountPath, projections, plan.Agents)
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

	// Consume an explicit conductor; separate an inherited one before consuming its Agent.
	var conductor *grovev1alpha1.PodCliqueTemplateSpec
	if conductorTemplateName != "" {
		conductorSpec := conductorTemplate.Spec
		if input.Conductor == nil {
			conductorSpec = *conductorSpec.DeepCopy()
			conductorSpec.Affinity = nil
		}
		annotations := roleAnnotations(conductorTemplate.Annotations, lpxv1alpha1.PodRoleConductor, workloadDigest)
		annotations[commonconsts.AnnotationExtraResourcesHash] = configHash
		conductor = &grovev1alpha1.PodCliqueTemplateSpec{
			Name:        conductorTemplateName,
			Labels:      maps.Clone(conductorTemplate.Labels),
			Annotations: annotations,
			Spec: grovev1alpha1.PodCliqueSpec{
				RoleName:     conductorTemplateName,
				PodSpec:      conductorSpec,
				Replicas:     1,
				MinAvailable: ptr.To(int32(1)),
				StartsAfter:  agentTemplateNames,
			},
		}
		pcs.Spec.Template.Cliques = append(pcs.Spec.Template.Cliques, conductor)
	}

	// Canonical projections keep each component together; consume its last Agent instance.
	var template corev1.PodTemplateSpec
	for index, projection := range projections {
		stage := projection.stage
		if index == 0 || stage != projections[index-1].stage {
			template = input.Stages[stage]
			storage, err := lpuModelStorageBinding(template.Spec)
			if err != nil {
				return nil, fmt.Errorf("stage %s: %w", stage, err)
			}
			if storage.mount.MountPath != modelStorage.mount.MountPath {
				return nil, fmt.Errorf("stage %s must use the Conductor model-storage mount path %q", stage, modelStorage.mount.MountPath)
			}
			var conductorSpec *corev1.PodSpec
			if stage == conductorStage && conductor != nil {
				conductorSpec = &conductor.Spec.PodSpec
			}
			if err := configureLPURolePods(&template.Spec, conductorSpec, workload, configMap.Name, allocation, input.SSHSecretName); err != nil {
				return nil, fmt.Errorf("stage %s: %w", stage, err)
			}
		}
		podSpec := template.Spec
		if index+1 < len(projections) && stage == projections[index+1].stage {
			podSpec = *podSpec.DeepCopy()
		}

		// Resolve per-model placeholders only after separating siblings' backing data.
		for containerIndex := range podSpec.Containers {
			env := podSpec.Containers[containerIndex].Env
			for envIndex := range env {
				if env[envIndex].Name == lpuModelNameEnvVar {
					env[envIndex] = corev1.EnvVar{Name: lpuModelNameEnvVar, Value: projection.Model()}
				}
			}
		}
		annotations := roleAnnotations(template.Annotations, lpxv1alpha1.PodRoleAgent, projection.Digest().String())
		annotations[commonconsts.AnnotationExtraResourcesHash] = configHash
		annotations[lpxv1alpha1.PodModelAnnotation] = projection.model
		annotations[WorkloadModeAnnotation] = string(projection.schedulerWorkloadMode())
		agent := plan.Agents[index]
		replicas := int32(agent.Replicas)
		pcs.Spec.Template.Cliques = append(pcs.Spec.Template.Cliques, &grovev1alpha1.PodCliqueTemplateSpec{
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

	pcs.Annotations = workloadAnnotations(pcs.Annotations, workloadDigest)
	// Node-local uses canonical annotation absence for backward compatibility.
	delete(pcs.Annotations, ExecutionBackendAnnotation)
	selectedTemplateNames := agentTemplateNames
	if conductorTemplateName != "" {
		selectedTemplateNames = append([]string{conductorTemplateName}, selectedTemplateNames...)
	}

	if hybrid {
		// Bound GPU hostnames using the rendered width of the last engine replica.
		if err := plan.validatePodHostname("Cyborg", plan.CyborgTemplate, int(cyborg.Spec.Replicas)-1); err != nil {
			return nil, err
		}

		// HX Cyborg may inherit its Agent's configuration mount.
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
			modelStorage,
			agentTemplateNames,
			cyborgConfigMap,
		); err != nil {
			return nil, err
		}
	}

	// The LPX-only PCS owns one scaling group for the complete engine.
	members := selectedTemplateNames
	if hybrid {
		members = append(members, plan.CyborgTemplate)
	}
	pcs.Spec.Template.PodCliqueScalingGroupConfigs = []grovev1alpha1.PodCliqueScalingGroupConfig{{
		Name:         lpxScalingGroupTemplateName,
		CliqueNames:  members,
		Annotations:  map[string]string{WorkloadDigestAnnotation: workloadDigest},
		Replicas:     ptr.To(plan.Replicas),
		MinAvailable: ptr.To(ptr.Deref(input.MinAvailable, 1)),
	}}

	explicit := grovev1alpha1.CliqueStartupTypeExplicit
	pcs.Spec.Template.StartupType = &explicit
	return extraResources, nil
}

// configureLPURolePods consumes fresh, independently owned Agent and conductor
// specs. Agent and workload are nonnil; nil conductor means no emitted launcher.
func configureLPURolePods(agentPodSpec, conductorPodSpec *corev1.PodSpec, workload *SelectedWorkload, configMapName, allocation, sshSecretName string) error {
	if err := withLPUConfigVolume(agentPodSpec, configMapName, workload.BuildFamily() == BuildFamilyXT); err != nil {
		return err
	}
	configureAgentScheduling(agentPodSpec, workload.BuildFamily())
	// Placement is already resolved; shape only the actual conductor's LPX-owned fields.
	if conductorPodSpec != nil {
		conductorPodSpec.SchedulerName = corev1.DefaultSchedulerName
		stripLPUResources(conductorPodSpec)
		if err := withLPUConfigVolume(conductorPodSpec, configMapName, workload.BuildFamily() == BuildFamilyXT); err != nil {
			return err
		}
	}
	if workload.Pipeline() == PipelineSpecDecode || workload.Pipeline() == PipelineLPX {
		agent := common.FindContainerByName(agentPodSpec.Containers, commonconsts.MainContainerName)
		setContainerEnv(agent, false, corev1.EnvVar{Name: lpuModelNameEnvVar})
	}

	if workload.Pipeline() == PipelineLPX {
		// Extra Agents beyond one per runtime partition imply multi-node SSH setup.
		projection := workload.modelProjections[0]
		partitionCount := len(projection.partitions)
		if workload.BuildFamily() == BuildFamilyXT {
			partitionCount = len(projection.configuredBuild.Partitions)
		}
		if err := configureDirectHybridAgentRuntime(
			agentPodSpec,
			configMapName,
			sshSecretName,
			projection.agentReplicas > partitionCount,
		); err != nil {
			return err
		}
	} else {
		// Both node-local roles use the operator-managed key, including an agent-only draft.
		if strings.TrimSpace(sshSecretName) == "" {
			return fmt.Errorf("node-local LPU runtime requires an MPI SSH secret name")
		}
		if conductorPodSpec != nil {
			if err := configureNodeLocalConductorRuntime(conductorPodSpec, workload.BuildFamily(), allocation, sshSecretName); err != nil {
				return err
			}
		}
		if err := configureNodeLocalAgentRuntime(agentPodSpec, workload.BuildFamily(), workload.Pipeline() == PipelineSingle, sshSecretName); err != nil {
			return err
		}
	}

	ensureLPUNodeTolerations(agentPodSpec)
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
		ExecutionBackendAnnotation,
		WorkloadModeAnnotation,
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
