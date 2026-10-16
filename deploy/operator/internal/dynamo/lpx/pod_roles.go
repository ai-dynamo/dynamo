/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"strings"

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
// devicesPerNode is the positive device count from the manifest.
func configureAgentScheduling(
	agent *corev1.PodSpec,
	targetFamily BuildFamily,
	devicesPerNode int,
) {
	// Model projection has already restricted the target family to XT or HX.
	name := v2LPUResourceName
	if targetFamily == BuildFamilyHX {
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

// stripLPUResources removes authored LPU requests and limits from a non-agent PodSpec,
// so only agents hold trays.
func stripLPUResources(spec *corev1.PodSpec) {
	for index := range spec.InitContainers {
		stripLPUResourceRequirements(&spec.InitContainers[index].Resources)
	}
	for index := range spec.Containers {
		stripLPUResourceRequirements(&spec.Containers[index].Resources)
	}
	for index := range spec.EphemeralContainers {
		stripLPUResourceRequirements(&spec.EphemeralContainers[index].Resources)
	}
	if spec.Resources != nil {
		stripLPUResourceList(spec.Resources.Limits)
		stripLPUResourceList(spec.Resources.Requests)
	}
}

func stripLPUResourceRequirements(resources *corev1.ResourceRequirements) {
	stripLPUResourceList(resources.Limits)
	stripLPUResourceList(resources.Requests)
}

func stripLPUResourceList(resources corev1.ResourceList) {
	for name := range resources {
		if strings.HasPrefix(string(name), "lpu.nvidia.com/") || name == v3LPUResourceName {
			delete(resources, name)
		}
	}
}
