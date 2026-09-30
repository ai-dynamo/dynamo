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
		path, err := buildRuntimePath(projection.configuredBuild.Path, projection.runtimeBuildRef, modelStoragePath)
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
