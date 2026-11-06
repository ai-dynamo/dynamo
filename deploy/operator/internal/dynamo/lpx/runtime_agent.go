/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	corev1 "k8s.io/api/core/v1"
)

// configureNodeLocalConductorRuntime consumes a fresh conductor PodSpec with a validated main container.
func configureNodeLocalConductorRuntime(
	conductorPodSpec *corev1.PodSpec,
	allocation string,
	environment semanticenv.ContainerLayers,
) {
	// Bind placement data without interpreting the template's executable or arguments.
	conductor := common.FindContainerByName(conductorPodSpec.Containers, commonconsts.MainContainerName)

	// Kubernetes expands environment references in order; publish allocation before authored bindings.
	environment.Prepend(conductor, "operator.lpx.allocation", []corev1.EnvVar{{Name: allocationEnvVar, Value: allocation}})
}

// modelPaths resolves every binding for nonempty canonical projections before any Pod is changed.
func modelPaths(projections []*ModelProjection, modelStoragePath string) ([]corev1.EnvVar, error) {
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
			return nil, fmt.Errorf("resolve %s: %w", binding.name, err)
		}
		paths = append(paths, corev1.EnvVar{Name: binding.name, Value: path})
	}

	return paths, nil
}
