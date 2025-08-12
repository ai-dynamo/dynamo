/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// ComponentDefaults interface defines how defaults should be provided
type ComponentDefaults interface {
	// GetBaseContainer returns the base container configuration for this component type
	// The backendFramework parameter may be empty for components that don't need backend-specific config
	// The numberOfNodes parameter indicates the total number of nodes in the deployment
	GetBaseContainer(backendFramework BackendFramework, numberOfNodes int32) (corev1.Container, error)
}

// ComponentDefaultsFactory creates appropriate defaults based on component type and number of nodes
func ComponentDefaultsFactory(componentType string, numberOfNodes int32) ComponentDefaults {
	switch componentType {
	case commonconsts.ComponentTypeFrontend:
		return NewFrontendDefaults()
	case commonconsts.ComponentTypeWorker:
		return NewWorkerDefaults()
	default:
		return &BaseComponentDefaults{}
	}
}

// BaseComponentDefaults provides common defaults shared by all components
type BaseComponentDefaults struct{}

func (b *BaseComponentDefaults) GetBaseContainer(backendFramework BackendFramework, numberOfNodes int32) (corev1.Container, error) {
	return b.getCommonContainer(), nil
}

func (b *BaseComponentDefaults) getCommonContainer() corev1.Container {
	container := corev1.Container{
		Name: "main",
	}

	return container
}
