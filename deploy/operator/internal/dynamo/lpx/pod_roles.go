/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

const (
	v2LPUResourceName corev1.ResourceName = "lpu.nvidia.com/lpu"
	v3LPUResourceName corev1.ResourceName = "nvidia.com/lpu"

	lpuAgentContainerName = "agent"
	allocationEnvVar      = "LPX_ALLOCATION"
)

// configureAgentScheduling consumes a fresh, nonnil Agent PodSpec containing main.
func configureAgentScheduling(
	agent *corev1.PodSpec,
	targetFamily BuildFamily,
) {
	// Preserve Agent placement while replacing authored LPU resources with the selected device.
	stripLPUResources(agent)
	agent.SchedulerName = v1alpha1.LPXSchedulerName
	container := common.FindContainerByName(agent.Containers, commonconsts.MainContainerName)
	// Model projection has already restricted the target family to XT or HX.
	name, amount := v2LPUResourceName, resource.MustParse("8")
	if targetFamily == BuildFamilyHX {
		name, amount = v3LPUResourceName, resource.MustParse("16")
	}
	if container.Resources.Requests == nil {
		container.Resources.Requests = make(corev1.ResourceList)
	}
	if container.Resources.Limits == nil {
		container.Resources.Limits = make(corev1.ResourceList)
	}
	container.Resources.Requests[name], container.Resources.Limits[name] = amount, amount
}

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
