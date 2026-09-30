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
	// Speculative decoding binds the first draft and final target; other workloads bind one model.
	names := []string{"LPX_MODEL_PATH"}
	if projections[0].pipeline == PipelineSpecDecode {
		names = []string{"LPX_DRAFT_MODEL_PATH", "LPX_TARGET_MODEL_PATH"}
		projections = []*ModelProjection{projections[0], projections[len(projections)-1]}
	}

	// Resolve all paths before publishing authoritative values ahead of authored references.
	env := make([]corev1.EnvVar, 0, len(container.Env)+len(names))
	for index, name := range names {
		projection := projections[index]
		path, err := buildRuntimePath(projection.configuredBuild.Path, projection.runtimeBuildRef, modelStoragePath)
		if err != nil {
			return fmt.Errorf("resolve %s: %w", name, err)
		}
		env = append(env, corev1.EnvVar{Name: name, Value: path})
	}
	for _, variable := range container.Env {
		if !slices.Contains(names, variable.Name) {
			env = append(env, variable)
		}
	}
	container.Env = env
	return nil
}
