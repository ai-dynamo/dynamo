/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
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
	// Preserve authored resources while binding the selected device on Agent main.
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
