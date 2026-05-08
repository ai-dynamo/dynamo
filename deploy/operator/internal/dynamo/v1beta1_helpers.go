/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

func ComponentsByName(dgd *v1beta1.DynamoGraphDeployment) map[string]*v1beta1.DynamoComponentDeploymentSharedSpec {
	if dgd == nil {
		return map[string]*v1beta1.DynamoComponentDeploymentSharedSpec{}
	}

	components := make(map[string]*v1beta1.DynamoComponentDeploymentSharedSpec, len(dgd.Spec.Components))
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		components[component.ComponentName] = component
	}
	return components
}

func GetPodTemplateAnnotations(component *v1beta1.DynamoComponentDeploymentSharedSpec) map[string]string {
	if component == nil || component.PodTemplate == nil {
		return nil
	}
	return component.PodTemplate.Annotations
}

func GetPodTemplateLabels(component *v1beta1.DynamoComponentDeploymentSharedSpec) map[string]string {
	if component == nil || component.PodTemplate == nil {
		return nil
	}
	return component.PodTemplate.Labels
}

func GetMainContainer(component *v1beta1.DynamoComponentDeploymentSharedSpec) *corev1.Container {
	if component == nil || component.PodTemplate == nil {
		return nil
	}
	for i := range component.PodTemplate.Spec.Containers {
		if component.PodTemplate.Spec.Containers[i].Name == commonconsts.MainContainerName {
			return &component.PodTemplate.Spec.Containers[i]
		}
	}
	return nil
}

func GetMainContainerResources(component *v1beta1.DynamoComponentDeploymentSharedSpec) corev1.ResourceRequirements {
	if main := GetMainContainer(component); main != nil {
		return main.Resources
	}
	return corev1.ResourceRequirements{}
}

func GetGPUMemoryService(component *v1beta1.DynamoComponentDeploymentSharedSpec) *v1beta1.GPUMemoryServiceSpec {
	if component == nil || component.Experimental == nil {
		return nil
	}
	return component.Experimental.GPUMemoryService
}

func GetCheckpoint(component *v1beta1.DynamoComponentDeploymentSharedSpec) *v1beta1.ComponentCheckpointConfig {
	if component == nil || component.Experimental == nil {
		return nil
	}
	return component.Experimental.Checkpoint
}

func GetDCDComponentName(dcd *v1beta1.DynamoComponentDeployment) string {
	if dcd == nil {
		return ""
	}
	if dcd.Spec.ComponentName != "" {
		return dcd.Spec.ComponentName
	}
	if dcd.Labels != nil {
		if componentName := dcd.Labels[commonconsts.KubeLabelDynamoComponent]; componentName != "" {
			return componentName
		}
	}
	return dcd.Name
}

func GetDCDDynamoNamespace(dcd *v1beta1.DynamoComponentDeployment) string {
	if dcd == nil {
		return ""
	}
	if dcd.Labels != nil {
		if dynamoNamespace := dcd.Labels[commonconsts.KubeLabelDynamoNamespace]; dynamoNamespace != "" {
			return dynamoNamespace
		}
	}
	parentName := dcd.GetParentGraphDeploymentName()
	if parentName == "" {
		parentName = dcd.Name
	}
	return v1beta1.ComputeDynamoNamespace(dcd.Spec.GlobalDynamoNamespace, dcd.GetNamespace(), parentName)
}

func mergeLowPriorityMetadata(dst, src map[string]string) map[string]string {
	if len(src) == 0 {
		return dst
	}
	if dst == nil {
		dst = map[string]string{}
	}
	for k, v := range src {
		if _, exists := dst[k]; !exists {
			dst[k] = v
		}
	}
	return dst
}
