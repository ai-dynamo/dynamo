/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import "github.com/ai-dynamo/dynamo/deploy/operator/internal/common"

const (
	KubeAnnotationDeploymentStrategy                    = "nvidia.com/deployment-strategy"
	KubeAnnotationDeploymentRollingUpdateMaxSurge       = "nvidia.com/deployment-rolling-update-max-surge"
	KubeAnnotationDeploymentRollingUpdateMaxUnavailable = "nvidia.com/deployment-rolling-update-max-unavailable"
)

// deploymentStrategyFromAnnotations returns the effective strategy shared by
// the DCD's Kubernetes Deployment and the DGD's cross-generation coordinator.
// Preserve the existing RollingUpdate fallback for missing or unrecognized
// values so both controllers interpret the annotation identically.
func deploymentStrategyFromAnnotations(annotations map[string]string) common.DeploymentStrategy {
	if common.DeploymentStrategy(annotations[KubeAnnotationDeploymentStrategy]) == common.DeploymentStrategyRecreate {
		return common.DeploymentStrategyRecreate
	}
	return common.DeploymentStrategyRollingUpdate
}
