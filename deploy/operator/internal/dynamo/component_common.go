/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// ComponentDefaults interface defines how defaults should be provided
type ComponentDefaults interface {
	// GetBaseContainer returns the base container configuration for this component type
	// The numberOfNodes parameter indicates the total number of nodes in the deployment
	GetBaseContainer(context ComponentContext) (corev1.Container, error)

	// GetBasePodSpec returns the base pod spec configuration for this component type
	// The numberOfNodes parameter indicates the total number of nodes in the deployment
	GetBasePodSpec(context ComponentContext) (corev1.PodSpec, error)
}

// ComponentDefaultsFactory creates appropriate defaults based on component type and number of nodes
func ComponentDefaultsFactory(componentType string) ComponentDefaults {
	switch componentType {
	case commonconsts.ComponentTypeFrontend:
		return NewFrontendDefaults()
	case commonconsts.ComponentTypeWorker, commonconsts.ComponentTypePrefill, commonconsts.ComponentTypeDecode:
		return NewWorkerDefaults()
	case commonconsts.ComponentTypePlanner:
		return NewPlannerDefaults()
	default:
		return &BaseComponentDefaults{}
	}
}

// BaseComponentDefaults provides common defaults shared by all components
type BaseComponentDefaults struct{}

type ComponentContext struct {
	numberOfNodes                  int32
	DynamoNamespace                string
	ComponentType                  string
	ParentGraphDeploymentName      string
	ParentGraphDeploymentNamespace string
	DiscoveryBackend               string
}

func (b *BaseComponentDefaults) GetBaseContainer(context ComponentContext) (corev1.Container, error) {
	return b.getCommonContainer(context), nil
}

func (b *BaseComponentDefaults) GetBasePodSpec(context ComponentContext) (corev1.PodSpec, error) {
	return b.getCommonPodSpec(), nil
}

func (b *BaseComponentDefaults) getCommonPodSpec() corev1.PodSpec {
	return corev1.PodSpec{
		TerminationGracePeriodSeconds: ptr.To(int64(60)),
		RestartPolicy:                 corev1.RestartPolicyAlways,
	}
}

func (b *BaseComponentDefaults) getCommonContainer(context ComponentContext) corev1.Container {
	container := corev1.Container{
		Name: commonconsts.MainContainerName,
		Command: []string{
			"/bin/sh",
			"-c",
		},
	}
	container.Env = []corev1.EnvVar{
		{
			Name:  commonconsts.DynamoNamespaceEnvVar,
			Value: context.DynamoNamespace,
		},
		{
			Name:  commonconsts.DynamoComponentEnvVar,
			Value: context.ComponentType,
		},
		{
			Name:  "DYN_PARENT_DGD_K8S_NAME",
			Value: context.ParentGraphDeploymentName,
		},
		{
			Name:  "DYN_PARENT_DGD_K8S_NAMESPACE",
			Value: context.ParentGraphDeploymentNamespace,
		},
		{
			Name: "POD_NAME",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.name",
				},
			},
		},
		{
			Name: "POD_NAMESPACE",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.namespace",
				},
			},
		},
		{
			Name: "POD_UID",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.uid",
				},
			},
		},
	}

	// Set discovery backend env var to "kubernetes" unless explicitly set to "etcd"
	if context.DiscoveryBackend != "etcd" {
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  commonconsts.DynamoDiscoveryBackendEnvVar,
			Value: "kubernetes",
		})
	}

	return container
}
