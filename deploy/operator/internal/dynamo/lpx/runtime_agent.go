/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// configureDirectHybridAgentRuntime lowers an LPX-scheduled PodSpec into the
// direct agent runtime used by conductorless hybrid workloads.
func configureDirectHybridAgentRuntime(
	agentPodSpec *corev1.PodSpec,
	lpuConfigMapName string,
) {
	agent := configureAgentIdentity(agentPodSpec)

	agent.Env = append(agent.Env,
		corev1.EnvVar{
			Name: "GAS_DIR",
			ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{
				LocalObjectReference: corev1.LocalObjectReference{Name: lpuConfigMapName},
				Key:                  "gas_dir",
			}},
		},
	)
}

// configureNodeLocalConductorRuntime consumes a fresh conductor PodSpec with a validated main container.
func configureNodeLocalConductorRuntime(
	conductorPodSpec *corev1.PodSpec,
	allocation string,
) {
	// Bind placement data without interpreting the template's executable or arguments.
	conductor := common.FindContainerByName(conductorPodSpec.Containers, commonconsts.MainContainerName)
	conductor.Name = dynamov1beta1.ComponentRoleLPXConductor
	setContainerEnv(conductor, corev1.EnvVar{Name: allocationEnvVar, Value: allocation})
	retargetMainContainerReferences(conductorPodSpec, conductor)
}

// configureAgentIdentity names the main runtime and retargets its container references.
// agentPodSpec must be non-nil and contain the validated main container.
func configureAgentIdentity(agentPodSpec *corev1.PodSpec) *corev1.Container {
	agent := common.FindContainerByName(agentPodSpec.Containers, commonconsts.MainContainerName)
	agent.Name = lpuAgentContainerName
	retargetMainContainerReferences(agentPodSpec, agent)
	return agent
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
