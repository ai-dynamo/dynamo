/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"encoding/json"
	"maps"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

// ComponentsByName returns the graph deployment components indexed by their
// stable v1beta1 component name.
func ComponentsByName(dgd *v1beta1.DynamoGraphDeployment) map[string]*v1beta1.DynamoComponentDeploymentSharedSpec {
	if dgd == nil {
		return nil
	}
	components := make(map[string]*v1beta1.DynamoComponentDeploymentSharedSpec, len(dgd.Spec.Components))
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		components[component.ComponentName] = component
	}
	return components
}

// GetPodTemplateAnnotations returns the component pod-template annotations, if
// a pod template is present.
func GetPodTemplateAnnotations(component *v1beta1.DynamoComponentDeploymentSharedSpec) map[string]string {
	if component == nil || component.PodTemplate == nil {
		return nil
	}
	return component.PodTemplate.Annotations
}

// GetPodTemplateLabels returns the component pod-template labels, if a pod
// template is present.
func GetPodTemplateLabels(component *v1beta1.DynamoComponentDeploymentSharedSpec) map[string]string {
	if component == nil || component.PodTemplate == nil {
		return nil
	}
	return component.PodTemplate.Labels
}

func ensurePodTemplate(component *v1beta1.DynamoComponentDeploymentSharedSpec) *corev1.PodTemplateSpec {
	if component.PodTemplate == nil {
		component.PodTemplate = &corev1.PodTemplateSpec{}
	}
	if component.PodTemplate.Labels == nil {
		component.PodTemplate.Labels = map[string]string{}
	}
	if component.PodTemplate.Annotations == nil {
		component.PodTemplate.Annotations = map[string]string{}
	}
	return component.PodTemplate
}

func ensureMainContainer(podTemplate *corev1.PodTemplateSpec) *corev1.Container {
	for i := range podTemplate.Spec.Containers {
		if podTemplate.Spec.Containers[i].Name == commonconsts.MainContainerName {
			return &podTemplate.Spec.Containers[i]
		}
	}
	podTemplate.Spec.Containers = append([]corev1.Container{{Name: commonconsts.MainContainerName}}, podTemplate.Spec.Containers...)
	return &podTemplate.Spec.Containers[0]
}

// GetMainContainer returns the well-known main container from the component pod
// template, if one exists.
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

// GetMainContainerResources returns the main container resources, or an empty
// resource requirements struct when no main container exists.
func GetMainContainerResources(component *v1beta1.DynamoComponentDeploymentSharedSpec) corev1.ResourceRequirements {
	if main := GetMainContainer(component); main != nil {
		return main.Resources
	}
	return corev1.ResourceRequirements{}
}

// GetDCDKubeLabels returns the labels rendered onto Kubernetes workloads for a
// DCD before controller-specific role labels are added.
func GetDCDKubeLabels(dcd *v1beta1.DynamoComponentDeployment) map[string]string {
	labels := map[string]string{}
	if dcd == nil {
		return labels
	}

	objectLabels := dcd.GetLabels()
	maps.Copy(labels, objectLabels)
	maps.Copy(labels, GetDCDPreservedAlphaLabels(dcd))
	maps.Copy(labels, GetPodTemplateLabels(&dcd.Spec.DynamoComponentDeploymentSharedSpec))
	AddBaseModelLabel(labels, dcd.Spec.ModelRef)
	if subComponentType := GetDCDSubComponentType(dcd); subComponentType != "" {
		labels[commonconsts.KubeLabelDynamoSubComponentType] = subComponentType
	}
	if componentName := GetDCDComponentName(dcd); componentName != "" {
		labels[commonconsts.KubeLabelDynamoComponent] = componentName
	}
	if dynamoNamespace := GetDCDDynamoNamespace(dcd); dynamoNamespace != "" {
		labels[commonconsts.KubeLabelDynamoNamespace] = dynamoNamespace
	}
	for _, key := range []string{
		commonconsts.KubeLabelDynamoComponent,
		commonconsts.KubeLabelDynamoNamespace,
		commonconsts.KubeLabelDynamoGraphDeploymentName,
		commonconsts.KubeLabelDynamoWorkerHash,
		commonconsts.KubeLabelDynamoComponentClass,
	} {
		if value := objectLabels[key]; value != "" {
			labels[key] = value
		}
	}
	return labels
}

// GetDCDKubeAnnotations returns the annotations rendered onto Kubernetes
// workloads for a DCD.
func GetDCDKubeAnnotations(dcd *v1beta1.DynamoComponentDeployment) map[string]string {
	annotations := map[string]string{}
	if dcd == nil {
		return annotations
	}

	maps.Copy(annotations, GetDCDPreservedAlphaAnnotations(dcd))
	maps.Copy(annotations, GetPodTemplateAnnotations(&dcd.Spec.DynamoComponentDeploymentSharedSpec))
	AddBaseModelAnnotation(annotations, dcd.Spec.ModelRef)
	delete(annotations, commonconsts.KubeAnnotationDynamoOperatorOriginVersion)
	return annotations
}

// GetGPUMemoryService returns the component GPU memory service config from the
// v1beta1 experimental block.
func GetGPUMemoryService(component *v1beta1.DynamoComponentDeploymentSharedSpec) *v1beta1.GPUMemoryServiceSpec {
	if component == nil || component.Experimental == nil {
		return nil
	}
	return component.Experimental.GPUMemoryService
}

// GetCheckpoint returns the component checkpoint config from the v1beta1
// experimental block.
func GetCheckpoint(component *v1beta1.DynamoComponentDeploymentSharedSpec) *v1beta1.ComponentCheckpointConfig {
	if component == nil || component.Experimental == nil {
		return nil
	}
	return component.Experimental.Checkpoint
}

// ToAlphaCheckpointConfig converts a v1beta1 checkpoint config into the
// controller's v1alpha1 compatibility shape.
func ToAlphaCheckpointConfig(src *v1beta1.ComponentCheckpointConfig) *v1alpha1.ServiceCheckpointConfig {
	if src == nil {
		return nil
	}
	dst := &v1alpha1.ServiceCheckpointConfig{}
	v1alpha1.ConvertToServiceCheckpointConfig(src, dst)
	return dst
}

// ToAlphaCheckpointIdentity converts a v1beta1 checkpoint identity into the
// controller's v1alpha1 compatibility shape.
func ToAlphaCheckpointIdentity(src *v1beta1.DynamoCheckpointIdentity) *v1alpha1.DynamoCheckpointIdentity {
	if src == nil {
		return nil
	}
	dst := &v1alpha1.DynamoCheckpointIdentity{}
	v1alpha1.ConvertToDynamoCheckpointIdentity(src, dst)
	return dst
}

// ToAlphaGPUMemoryService converts a v1beta1 GPU memory service config into
// the controller's v1alpha1 compatibility shape.
func ToAlphaGPUMemoryService(src *v1beta1.GPUMemoryServiceSpec) *v1alpha1.GPUMemoryServiceSpec {
	if src == nil {
		return nil
	}
	dst := &v1alpha1.GPUMemoryServiceSpec{}
	v1alpha1.ConvertToGPUMemoryServiceSpec(src, dst)
	return dst
}

// ToBetaSharedMemorySize converts the v1alpha1 shared-memory compatibility
// shape into the v1beta1 scalar field.
func ToBetaSharedMemorySize(src *v1alpha1.SharedMemorySpec) *resource.Quantity {
	if src == nil || (!src.Disabled && src.Size.IsZero()) {
		return nil
	}
	dst := &resource.Quantity{}
	v1alpha1.ConvertFromSharedMemorySpec(src, dst)
	return dst
}

// GetDCDComponentName returns the stable component identity for a standalone
// DCD, including any alpha compatibility value restored by API conversion.
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
	if spec := getDCDAlphaSharedSpec(dcd); spec != nil && spec.ServiceName != "" {
		return spec.ServiceName
	}
	return dcd.Name
}

// GetDCDDynamoNamespace returns the Dynamo namespace for a standalone DCD,
// including any alpha compatibility value restored by API conversion.
func GetDCDDynamoNamespace(dcd *v1beta1.DynamoComponentDeployment) string {
	if dcd == nil {
		return ""
	}
	if spec := getDCDAlphaSharedSpec(dcd); spec != nil {
		if spec.DynamoNamespace != nil {
			return *spec.DynamoNamespace
		}
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

// GetDCDSubComponentType returns the alpha subcomponent type restored by API
// conversion, when one was preserved.
func GetDCDSubComponentType(dcd *v1beta1.DynamoComponentDeployment) string {
	if dcd == nil {
		return ""
	}
	if spec := getDCDAlphaSharedSpec(dcd); spec != nil {
		return spec.SubComponentType
	}
	return ""
}

// GetDCDWorkloadComponentType returns the component type that existing
// Kubernetes workloads should be rendered with. Converted v1alpha1 DCDs for
// disaggregated workers preserve the old selector contract:
// component-type=worker plus sub-component-type=prefill/decode.
func GetDCDWorkloadComponentType(dcd *v1beta1.DynamoComponentDeployment) string {
	if legacyType := legacyAlphaWorkloadComponentType(dcd); legacyType != "" {
		return legacyType
	}
	if dcd == nil {
		return ""
	}
	return string(dcd.Spec.ComponentType)
}

func legacyAlphaWorkloadComponentType(dcd *v1beta1.DynamoComponentDeployment) string {
	if dcd == nil || dcd.Labels == nil {
		return ""
	}
	labels := dcd.Labels
	if labels[commonconsts.KubeLabelDynamoGraphDeploymentName] == "" ||
		labels[commonconsts.KubeLabelDynamoWorkerHash] == "" {
		return ""
	}
	legacyType := labels[commonconsts.KubeLabelDynamoComponentType]
	specType := string(dcd.Spec.ComponentType)
	if legacyType != commonconsts.ComponentTypeWorker {
		return ""
	}
	if specType != commonconsts.ComponentTypePrefill && specType != commonconsts.ComponentTypeDecode {
		return ""
	}
	if subComponentType := labels[commonconsts.KubeLabelDynamoSubComponentType]; subComponentType != "" && subComponentType != specType {
		return ""
	}
	return legacyType
}

// GetDCDPreservedAlphaAnnotations returns alpha compatibility annotations
// restored by API conversion.
func GetDCDPreservedAlphaAnnotations(dcd *v1beta1.DynamoComponentDeployment) map[string]string {
	if spec := getDCDAlphaSharedSpec(dcd); spec != nil {
		return spec.Annotations
	}
	return nil
}

// GetDCDPreservedAlphaLabels returns alpha compatibility labels restored by
// API conversion.
func GetDCDPreservedAlphaLabels(dcd *v1beta1.DynamoComponentDeployment) map[string]string {
	if spec := getDCDAlphaSharedSpec(dcd); spec != nil {
		return spec.Labels
	}
	return nil
}

// GetDCDPreservedAlphaIngressSpec returns an alpha ingress compatibility shape
// restored by API conversion.
func GetDCDPreservedAlphaIngressSpec(dcd *v1beta1.DynamoComponentDeployment) (IngressSpec, bool, error) {
	if dcd == nil {
		return IngressSpec{}, false, nil
	}
	alpha, err := convertDCDToAlpha(dcd)
	if err != nil {
		return IngressSpec{}, false, err
	}
	if alpha == nil || alpha.Spec.Ingress == nil {
		return IngressSpec{}, false, nil
	}
	data, err := json.Marshal(alpha.Spec.Ingress)
	if err != nil {
		return IngressSpec{}, false, err
	}
	var ingressSpec IngressSpec
	if err := json.Unmarshal(data, &ingressSpec); err != nil {
		return IngressSpec{}, false, err
	}
	return ingressSpec, true, nil
}

func getDCDAlphaSharedSpec(dcd *v1beta1.DynamoComponentDeployment) *v1alpha1.DynamoComponentDeploymentSharedSpec {
	alpha, err := convertDCDToAlpha(dcd)
	if err != nil || alpha == nil {
		return nil
	}
	return &alpha.Spec.DynamoComponentDeploymentSharedSpec
}

func convertDCDToAlpha(dcd *v1beta1.DynamoComponentDeployment) (*v1alpha1.DynamoComponentDeployment, error) {
	if dcd == nil {
		return nil, nil
	}
	alpha := &v1alpha1.DynamoComponentDeployment{}
	if err := alpha.ConvertFrom(dcd); err != nil {
		return nil, err
	}
	return alpha.DeepCopy(), nil
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
