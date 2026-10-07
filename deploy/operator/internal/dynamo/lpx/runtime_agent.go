/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"slices"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// configureNodeLocalConductorRuntime consumes a fresh conductor PodSpec with a validated main container.
func configureNodeLocalConductorRuntime(
	conductorPodSpec *corev1.PodSpec,
	allocation string,
) {
	// Bind placement data without interpreting the template's executable or arguments.
	conductor := common.FindContainerByName(conductorPodSpec.Containers, commonconsts.MainContainerName)
	conductor.Name = dynamov1beta1.ComponentRoleLPXConductor

	// Kubernetes expands environment references in order; publish allocation before authored bindings.
	env := make([]corev1.EnvVar, 0, len(conductor.Env)+1)
	env = append(env, corev1.EnvVar{Name: allocationEnvVar, Value: allocation})
	for _, variable := range conductor.Env {
		if variable.Name != allocationEnvVar {
			env = append(env, variable)
		}
	}
	conductor.Env = env

	retargetMainContainerReferences(conductorPodSpec, conductor)
}

// applyModelPaths binds nonempty canonical projections into a fresh runtime container.
func applyModelPaths(container *corev1.Container, projections []*ModelProjection, modelStoragePath string) error {
	type modelPathBinding struct {
		name       string
		projection *ModelProjection
	}
	bindings := []modelPathBinding{{"LPX_MODEL_PATH", projections[0]}}
	if projections[0].pipeline == PipelineSpecDecode {
		bindings[0].name = "LPX_DRAFT_MODEL_PATH"
		bindings = append(bindings, modelPathBinding{"LPX_TARGET_MODEL_PATH", projections[len(projections)-1]})
	}

	// Resolve all paths before publishing authoritative values ahead of authored references.
	env := make([]corev1.EnvVar, 0, len(container.Env)+len(bindings))
	for _, binding := range bindings {
		path, err := buildRuntimePath(lpuRuntimeBuildRef(binding.projection, modelStoragePath), modelStoragePath)
		if err != nil {
			return fmt.Errorf("resolve %s: %w", binding.name, err)
		}
		env = append(env, corev1.EnvVar{Name: binding.name, Value: path})
	}
	for _, variable := range container.Env {
		if !slices.ContainsFunc(bindings, func(binding modelPathBinding) bool { return binding.name == variable.Name }) {
			env = append(env, variable)
		}
	}
	container.Env = env
	return nil
}

// configureAgentIdentity names the main runtime and retargets its container references.
// agentPodSpec must be non-nil and contain the validated main container.
func configureAgentIdentity(agentPodSpec *corev1.PodSpec) {
	agent := common.FindContainerByName(agentPodSpec.Containers, commonconsts.MainContainerName)
	agent.Name = lpuAgentContainerName
	retargetMainContainerReferences(agentPodSpec, agent)
}

// retargetMainContainerReferences requires its non-nil target to point into the non-nil PodSpec's Containers.
func retargetMainContainerReferences(podSpec *corev1.PodSpec, target *corev1.Container) {
	targetName := target.Name
	// Keep discovery identity and explicit resource selectors valid anywhere in the Pod after renaming main.
	for index := range podSpec.Containers {
		container := &podSpec.Containers[index]
		retargetEnvReferences(container.Env, targetName, container == target)
	}
	for index := range podSpec.InitContainers {
		retargetEnvReferences(podSpec.InitContainers[index].Env, targetName, false)
	}
	for index := range podSpec.EphemeralContainers {
		retargetEnvReferences(podSpec.EphemeralContainers[index].Env, targetName, false)
	}

	// Retarget both standalone and projected downwardAPI resource files.
	for index := range podSpec.Volumes {
		volume := &podSpec.Volumes[index]
		if volume.DownwardAPI != nil {
			retargetResourceFieldReferencesInDownwardAPI(volume.DownwardAPI.Items, targetName)
		}
		if volume.Projected == nil {
			continue
		}
		for sourceIndex := range volume.Projected.Sources {
			downwardAPI := volume.Projected.Sources[sourceIndex].DownwardAPI
			if downwardAPI != nil {
				retargetResourceFieldReferencesInDownwardAPI(downwardAPI.Items, targetName)
			}
		}
	}
}

func retargetEnvReferences(env []corev1.EnvVar, targetName string, retargetDiscoveryIdentity bool) {
	// Replace the target's discovery opt-in before retargeting explicit resource selectors.
	for index := range env {
		if retargetDiscoveryIdentity && env[index].Name == "CONTAINER_NAME" {
			env[index] = corev1.EnvVar{Name: "CONTAINER_NAME", Value: targetName}
		}
		source := env[index].ValueFrom
		if source == nil || source.ResourceFieldRef == nil {
			continue
		}
		if source.ResourceFieldRef.ContainerName == commonconsts.MainContainerName {
			source.ResourceFieldRef.ContainerName = targetName
		}
	}
}

func retargetResourceFieldReferencesInDownwardAPI(items []corev1.DownwardAPIVolumeFile, targetName string) {
	// DownwardAPI files require the renamed container's exact Kubernetes name.
	for index := range items {
		selector := items[index].ResourceFieldRef
		if selector != nil && selector.ContainerName == commonconsts.MainContainerName {
			selector.ContainerName = targetName
		}
	}
}
