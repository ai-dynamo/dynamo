/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

const (
	v2LPUResourceName corev1.ResourceName = "lpu.nvidia.com/lpu"
	v3LPUResourceName corev1.ResourceName = "nvidia.com/lpu"

	allocationEnvVar = "LPX_ALLOCATION"
)

// configureAgentScheduling consumes a fresh, nonnil Agent PodSpec containing main.
// devicesPerNode is the device count from the manifest; the compiler makes it positive.
func configureAgentScheduling(agent *corev1.PodSpec, architecture lpuArchitecture, devicesPerNode int) {
	// The build reader accepts only LP20 and LP30 for an LPU partition.
	name := v2LPUResourceName
	if architecture == lp30 {
		name = v3LPUResourceName
	}

	// Bind the manifest's device count on main while preserving all other authored resources.
	container := common.FindContainerByName(agent.Containers, commonconsts.MainContainerName)
	amount := *resource.NewQuantity(int64(devicesPerNode), resource.DecimalSI)
	if container.Resources.Requests == nil {
		container.Resources.Requests = make(corev1.ResourceList)
	}
	if container.Resources.Limits == nil {
		container.Resources.Limits = make(corev1.ResourceList)
	}
	container.Resources.Requests[name], container.Resources.Limits[name] = amount, amount
}

// configureNodeLocalConductorRuntime consumes a fresh conductor PodSpec with a validated main container.
func configureNodeLocalConductorRuntime(conductorPodSpec *corev1.PodSpec, allocation string) {
	// Bind placement data without interpreting the template's executable or arguments.
	prependEnv(common.FindContainerByName(conductorPodSpec.Containers, commonconsts.MainContainerName), corev1.EnvVar{Name: allocationEnvVar, Value: allocation})
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
		path, err := projection.buildRuntimePath(modelStoragePath)
		if err != nil {
			return fmt.Errorf("resolve %s: %w", binding.name, err)
		}
		paths = append(paths, corev1.EnvVar{Name: binding.name, Value: path})
	}

	// Regular and init containers receive authoritative paths ahead of authored references.
	for _, containers := range containerLists {
		for i := range containers {
			prependEnv(&containers[i], paths...)
		}
	}

	return nil
}
