/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"slices"

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

	// Kubernetes expands environment references in order; publish allocation before authored bindings.
	env := make([]corev1.EnvVar, 0, len(conductor.Env)+1)
	env = append(env, corev1.EnvVar{Name: allocationEnvVar, Value: allocation})
	for _, variable := range conductor.Env {
		if variable.Name != allocationEnvVar {
			env = append(env, variable)
		}
	}
	conductor.Env = env
}

// applyModelPaths resolves nonempty canonical projections and rewrites each container's Env in place.
func applyModelPaths(containerLists [][]corev1.Container, projections []*ModelProjection, modelStoragePath string) error {
	// Single-model workloads publish one path; speculative workloads publish the first draft and final target.
	type modelPathBinding struct {
		name       string
		projection *ModelProjection
	}
	bindings := []modelPathBinding{{"LPX_MODEL_PATH", projections[0]}}
	if projections[0].pipeline == PipelineSpecDecode {
		bindings[0].name = "LPX_DRAFT_MODEL_PATH"
		bindings = append(bindings, modelPathBinding{"LPX_TARGET_MODEL_PATH", projections[len(projections)-1]})
	}

	// Resolve all paths once before changing any container's environment.
	paths := make([]corev1.EnvVar, 0, len(bindings))
	for _, binding := range bindings {
		projection := binding.projection
		path, err := buildRuntimePath(projection.configuredBuild.path, projection.runtimeBuildRef, modelStoragePath)
		if err != nil {
			return fmt.Errorf("resolve %s: %w", binding.name, err)
		}
		paths = append(paths, corev1.EnvVar{Name: binding.name, Value: path})
	}

	// Regular and init containers receive authoritative paths ahead of authored references.
	for _, containers := range containerLists {
		for i := range containers {
			container := &containers[i]
			env := make([]corev1.EnvVar, 0, len(paths)+len(container.Env))
			env = append(env, paths...)
			for _, variable := range container.Env {
				if !slices.ContainsFunc(paths, func(path corev1.EnvVar) bool { return path.Name == variable.Name }) {
					env = append(env, variable)
				}
			}
			container.Env = env
		}
	}
	return nil
}
