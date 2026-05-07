/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"encoding/json"
	"maps"
	"strings"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

const (
	preservedDCDAnnotationPrefix           = "nvidia.com/dcd-"
	preservedDCDAlphaSpecAnnotation        = "nvidia.com/dcd-spec"
	preservedDCDServiceNameAnnotation      = preservedDCDAnnotationPrefix + "service-name"
	preservedDCDDynamoNamespaceAnnotation  = preservedDCDAnnotationPrefix + "dynamo-namespace"
	preservedDCDSubComponentTypeAnnotation = preservedDCDAnnotationPrefix + "sub-component-type"
	PreservedDCDAnnotationsAnnotation      = preservedDCDAnnotationPrefix + "annotations"
	PreservedDCDLabelsAnnotation           = preservedDCDAnnotationPrefix + "labels"
	PreservedDCDIngressAnnotation          = preservedDCDAnnotationPrefix + "ingress"
)

func IsPreservedDCDAnnotation(key string) bool {
	return strings.HasPrefix(key, preservedDCDAnnotationPrefix)
}

func ComponentsByName(dgd *v1beta1.DynamoGraphDeployment) map[string]*v1beta1.DynamoComponentDeploymentSharedSpec {
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
	if serviceName := dcd.GetAnnotations()[preservedDCDServiceNameAnnotation]; serviceName != "" {
		return serviceName
	}
	return dcd.Name
}

func GetDCDDynamoNamespace(dcd *v1beta1.DynamoComponentDeployment) string {
	if dcd == nil {
		return ""
	}
	if spec := getDCDPreservedAlphaSpec(dcd); spec != nil {
		if spec.DynamoNamespace != nil {
			return *spec.DynamoNamespace
		}
	} else if dynamoNamespace := dcd.GetAnnotations()[preservedDCDDynamoNamespaceAnnotation]; dynamoNamespace != "" {
		return dynamoNamespace
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

func GetDCDSubComponentType(dcd *v1beta1.DynamoComponentDeployment) string {
	if dcd == nil {
		return ""
	}
	if spec := getDCDPreservedAlphaSpec(dcd); spec != nil {
		return spec.SubComponentType
	}
	if subComponentType := dcd.GetAnnotations()[preservedDCDSubComponentTypeAnnotation]; subComponentType != "" {
		return subComponentType
	}
	return ""
}

func GetDCDPreservedAlphaAnnotations(dcd *v1beta1.DynamoComponentDeployment) map[string]string {
	if spec := getDCDPreservedAlphaSpec(dcd); spec != nil {
		return maps.Clone(spec.Annotations)
	}
	if values := getDCDPreservedStringMapAnnotation(dcd, PreservedDCDAnnotationsAnnotation); values != nil {
		return values
	}
	return nil
}

func GetDCDPreservedAlphaLabels(dcd *v1beta1.DynamoComponentDeployment) map[string]string {
	if spec := getDCDPreservedAlphaSpec(dcd); spec != nil {
		return maps.Clone(spec.Labels)
	}
	if values := getDCDPreservedStringMapAnnotation(dcd, PreservedDCDLabelsAnnotation); values != nil {
		return values
	}
	return nil
}

func GetDCDPreservedAlphaIngressSpec(dcd *v1beta1.DynamoComponentDeployment) (IngressSpec, bool, error) {
	if dcd == nil {
		return IngressSpec{}, false, nil
	}
	spec := getDCDPreservedAlphaSpec(dcd)
	if spec != nil {
		if spec.Ingress == nil {
			return IngressSpec{}, false, nil
		}
		data, err := json.Marshal(spec.Ingress)
		if err != nil {
			return IngressSpec{}, false, err
		}
		var ingressSpec IngressSpec
		if err := json.Unmarshal(data, &ingressSpec); err != nil {
			return IngressSpec{}, false, err
		}
		return ingressSpec, true, nil
	}
	raw := dcd.GetAnnotations()[PreservedDCDIngressAnnotation]
	if raw == "" {
		return IngressSpec{}, false, nil
	}
	var ingressSpec IngressSpec
	if err := json.Unmarshal([]byte(raw), &ingressSpec); err != nil {
		return IngressSpec{}, false, err
	}
	return ingressSpec, true, nil
}

func getDCDPreservedStringMapAnnotation(dcd *v1beta1.DynamoComponentDeployment, key string) map[string]string {
	if dcd == nil {
		return nil
	}
	raw := dcd.GetAnnotations()[key]
	if raw == "" {
		return nil
	}
	var values map[string]string
	if err := json.Unmarshal([]byte(raw), &values); err != nil {
		return nil
	}
	return values
}

func getDCDPreservedAlphaSpec(dcd *v1beta1.DynamoComponentDeployment) *v1alpha1.DynamoComponentDeploymentSharedSpec {
	if dcd == nil {
		return nil
	}
	raw := dcd.GetAnnotations()[preservedDCDAlphaSpecAnnotation]
	if raw == "" {
		return nil
	}
	var envelope struct {
		Spec *v1alpha1.DynamoComponentDeploymentSpec `json:"spec"`
	}
	if err := json.Unmarshal([]byte(raw), &envelope); err == nil && envelope.Spec != nil {
		return &envelope.Spec.DynamoComponentDeploymentSharedSpec
	}
	var spec v1alpha1.DynamoComponentDeploymentSpec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		return nil
	}
	return &spec.DynamoComponentDeploymentSharedSpec
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
