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
	allocationEnvVar = "LPX_ALLOCATION"
)

// configureAgentScheduling consumes a fresh, nonnil Agent PodSpec containing main.
// devicesPerNode is the positive device count from the manifest.
func configureAgentScheduling(
	agent *corev1.PodSpec,
	targetFamily *family,
	devicesPerNode int,
) {
	// Bind the manifest's device count on main while preserving all other authored resources.
	container := common.FindContainerByName(agent.Containers, commonconsts.MainContainerName)
	amount := *resource.NewQuantity(int64(devicesPerNode), resource.DecimalSI)
	if container.Resources.Requests == nil {
		container.Resources.Requests = make(corev1.ResourceList)
	}
	if container.Resources.Limits == nil {
		container.Resources.Limits = make(corev1.ResourceList)
	}
	container.Resources.Requests[targetFamily.lpuResource], container.Resources.Limits[targetFamily.lpuResource] = amount, amount
}
