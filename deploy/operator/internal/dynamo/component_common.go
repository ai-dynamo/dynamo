/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	controller_common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
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
	case commonconsts.ComponentTypeEPP:
		return NewEPPDefaults()
	default:
		return &BaseComponentDefaults{}
	}
}

// BaseComponentDefaults provides common defaults shared by all components
type BaseComponentDefaults struct{}

// DiscoveryContext holds resolved discovery settings for a component.
type DiscoveryContext struct {
	Backend configv1alpha1.DiscoveryBackend
	Mode    configv1alpha1.KubeDiscoveryMode
}

// NewDiscoveryContext resolves discovery settings from operator config and component annotations.
func NewDiscoveryContext(defaultBackend configv1alpha1.DiscoveryBackend, annotations map[string]string) DiscoveryContext {
	return DiscoveryContext{
		Backend: controller_common.GetDiscoveryBackend(defaultBackend, annotations),
		Mode:    controller_common.GetKubeDiscoveryMode(annotations),
	}
}

type ComponentContext struct {
	numberOfNodes                  int32
	DynamoNamespace                string
	ComponentType                  string
	ParentGraphDeploymentName      string
	ParentGraphDeploymentNamespace string
	Discovery                      DiscoveryContext
	EPPConfig                      *v1beta1.EPPConfig
	WorkerHashSuffix               string
	// MainContainerName is the component's effective main-container name
	// (spec.mainContainerName, defaulting to "main").
	MainContainerName string
	// ContainerName is the name of the container being generated from this
	// context. Empty means the component's main container; the frontend
	// sidecar merge sets it to the sidecar's name so identity env vars
	// reflect the container they are injected into.
	ContainerName string
}

// mainContainerName returns the context's main-container name, defaulting to
// the well-known "main" when the context was built without one (tests,
// legacy call sites).
func (c ComponentContext) mainContainerName() string {
	if c.MainContainerName == "" {
		return commonconsts.MainContainerName
	}
	return c.MainContainerName
}

// containerName returns the name of the container being generated: the
// explicit ContainerName when set (e.g. a frontend sidecar), otherwise the
// component's main container.
func (c ComponentContext) containerName() string {
	if c.ContainerName == "" {
		return c.mainContainerName()
	}
	return c.ContainerName
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
		Name: context.containerName(),
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
	if context.Discovery.Backend != "etcd" {
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  commonconsts.DynamoDiscoveryBackendEnvVar,
			Value: "kubernetes",
		})
	}

	if context.Discovery.Mode == configv1alpha1.KubeDiscoveryModeContainer {
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  "CONTAINER_NAME",
			Value: context.containerName(),
		})
		// Tell the runtime which container name carries main-container
		// (pod-level) identity, so a renamed main container keeps the same
		// discovery identity as the default "main". Only injected for
		// non-default names: the runtime falls back to "main" when the env
		// var is unset, and skipping it keeps pod specs of existing
		// default-named deployments byte-identical across operator upgrades
		// (no restart churn).
		if mainContainerName := context.mainContainerName(); mainContainerName != commonconsts.MainContainerName {
			container.Env = append(container.Env, corev1.EnvVar{
				Name:  commonconsts.DynamoMainContainerEnvVar,
				Value: mainContainerName,
			})
		}
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  "DYN_KUBE_DISCOVERY_MODE",
			Value: string(configv1alpha1.KubeDiscoveryModeContainer),
		})
	}

	return container
}
