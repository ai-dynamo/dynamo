/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"slices"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
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
	applyAllocation(common.FindContainerByName(conductorPodSpec.Containers, commonconsts.MainContainerName), allocation)
}

// applyAllocation publishes the Agent PodClique template names into one fresh
// runtime container, replacing any authored value.
func applyAllocation(container *corev1.Container, allocation string) {
	// Kubernetes expands environment references in order; publish allocation before authored bindings.
	env := make([]corev1.EnvVar, 0, len(container.Env)+1)
	env = append(env, corev1.EnvVar{Name: allocationEnvVar, Value: allocation})
	for _, variable := range container.Env {
		if variable.Name != allocationEnvVar {
			env = append(env, variable)
		}
	}
	container.Env = env
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
		projection := binding.projection
		path, err := buildRuntimePath(projection.configuredBuild.path, projection.runtimeBuildRef, modelStoragePath)
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

// lpuModelStoragePath returns the mount path of the model-storage volume in the
// PodSpec's main container: the root under which runtimes find compiled builds.
func lpuModelStoragePath(spec corev1.PodSpec) (string, error) {
	container := common.FindContainerByName(spec.Containers, commonconsts.MainContainerName)
	mountIndex := slices.IndexFunc(container.VolumeMounts, func(mount corev1.VolumeMount) bool {
		return mount.Name == v1alpha1.ModelStorageVolumeName
	})
	if mountIndex < 0 {
		return "", fmt.Errorf(
			"selected LPX main container requires model storage volume mount %q",
			v1alpha1.ModelStorageVolumeName,
		)
	}
	mount := container.VolumeMounts[mountIndex]
	if strings.TrimSpace(mount.MountPath) == "" {
		return "", fmt.Errorf("model storage volume %q has no mount path", mount.Name)
	}
	volumeIndex := slices.IndexFunc(spec.Volumes, func(volume corev1.Volume) bool { return volume.Name == mount.Name })
	if volumeIndex < 0 {
		return "", fmt.Errorf("selected LPX podTemplate has no model storage volume %q", mount.Name)
	}
	return mount.MountPath, nil
}
