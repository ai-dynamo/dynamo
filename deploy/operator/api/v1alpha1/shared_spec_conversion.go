/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Shared-spec conversion helpers used by both DGD (per-service) and DCD.
//
// DynamoComponentDeploymentSharedSpec is the payload that differs most between
// v1alpha1 and v1beta1: v1alpha1 has ten per-service pod-configuration fields,
// v1beta1 replaces them with a single corev1.PodTemplateSpec plus a few
// first-class fields. The heavy lifting (podTemplate <-> flat fields and the
// experimental block) lives here so that the DGD and DCD conversion entry
// points stay thin.
//
// Round-trip fidelity
//
// Every v1beta1 field that has no v1alpha1 equivalent, and every v1alpha1
// field that has no v1beta1 equivalent, is saved by the caller in a sparse
// typed spec/status annotation. This file only restores from the typed
// preserved value it receives; it does not read or write annotation side
// channels.

package v1alpha1

import (
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"slices"

	"github.com/imdario/mergo"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiconversion "k8s.io/apimachinery/pkg/conversion"
	"k8s.io/utils/ptr"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	// mainContainerName is the well-known container name in v1beta1 podTemplate
	// that receives the operator's default merges. Duplicated from
	// v1beta1.MainContainerName so this file has no reverse dependency ordering
	// concerns at build time.
	mainContainerName = "main"

	// defaultFrontendSidecarContainerName is the v1alpha1 default container
	// name synthesised by buildPodTemplateTo when a v1alpha1 FrontendSidecarSpec
	// is converted into a podTemplate sidecar + FrontendSidecar name reference.
	defaultFrontendSidecarContainerName = "sidecar-frontend"

	// defaultGPUResourceName is the v1alpha1 default when a user sets
	// Resources.{Requests,Limits}.GPU without specifying GPUType.
	defaultGPUResourceName = corev1.ResourceName("nvidia.com/gpu")

	annotationTrue = "true"
)

// setAnnOnObj is a convenience for object-level preservation annotations.
func setAnnOnObj(obj metav1.Object, key, value string) {
	anns := obj.GetAnnotations()
	if anns == nil {
		anns = map[string]string{}
	}
	anns[key] = value
	obj.SetAnnotations(anns)
}

func setJSONAnnOnObj(obj metav1.Object, key string, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("marshal %s annotation: %w", key, err)
	}
	setAnnOnObj(obj, key, string(data))
	return nil
}

func getAnnFromObj(obj metav1.Object, key string) (string, bool) {
	anns := obj.GetAnnotations()
	if anns == nil {
		return "", false
	}
	v, ok := anns[key]
	return v, ok
}

func getJSONAnnFromObj[T any](obj metav1.Object, key string) (T, bool, error) {
	var out T
	raw, ok := getAnnFromObj(obj, key)
	if !ok || raw == "" {
		return out, false, nil
	}
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return out, false, fmt.Errorf("unmarshal %s annotation: %w", key, err)
	}
	return out, true, nil
}

func delAnnFromObj(obj metav1.Object, key string) {
	anns := obj.GetAnnotations()
	if anns == nil {
		return
	}
	delete(anns, key)
	if len(anns) == 0 {
		obj.SetAnnotations(nil)
	} else {
		obj.SetAnnotations(anns)
	}
}

// ---------------------------------------------------------------------------
// Shared-spec conversion entry points
// ---------------------------------------------------------------------------

func fillSharedAlphaOnlyFromPreserved(dst *DynamoComponentDeploymentSharedSpec, preserved *DynamoComponentDeploymentSharedSpec, mainContainerPresent bool) {
	if dst == nil || preserved == nil {
		return
	}
	restoreSharedAlphaOnlySimpleFields(dst, preserved)
	restoreSharedAlphaOnlyPodFields(dst, preserved, mainContainerPresent)
	restoreSharedAlphaOnlyDisabledFeatures(dst, preserved)
}

func restoreSharedAlphaOnlySimpleFields(dst *DynamoComponentDeploymentSharedSpec, preserved *DynamoComponentDeploymentSharedSpec) {
	if dst.SubComponentType == "" {
		dst.SubComponentType = preserved.SubComponentType
	}
	if dst.DynamoNamespace == nil && preserved.DynamoNamespace != nil {
		dst.DynamoNamespace = ptr.To(*preserved.DynamoNamespace)
	}
	if dst.Autoscaling == nil && preserved.Autoscaling != nil {
		cp := *preserved.Autoscaling
		dst.Autoscaling = &cp
	}
	if dst.Ingress == nil && preserved.Ingress != nil {
		cp := *preserved.Ingress
		dst.Ingress = &cp
	}
	if len(dst.Annotations) == 0 {
		dst.Annotations = maps.Clone(preserved.Annotations)
	}
	if len(dst.Labels) == 0 {
		dst.Labels = maps.Clone(preserved.Labels)
	}
}

func restoreSharedAlphaOnlyPodFields(dst *DynamoComponentDeploymentSharedSpec, preserved *DynamoComponentDeploymentSharedSpec, mainContainerPresent bool) {
	if shouldRestorePreservedSharedMemory(dst.SharedMemory, preserved.SharedMemory) {
		dst.SharedMemory = preserved.SharedMemory.DeepCopy()
	}
	if mainContainerPresent && shouldRestorePreservedResources(dst.Resources, preserved.Resources) {
		dst.Resources = preserved.Resources.DeepCopy()
	}
	restoreEnvFromSecretFromPreserved(dst, preserved, mainContainerPresent)
	if dst.ExtraPodMetadata == nil && extraPodMetadataNeedsPreservation(preserved.ExtraPodMetadata) {
		dst.ExtraPodMetadata = preserved.ExtraPodMetadata.DeepCopy()
	}
	if dst.ExtraPodSpec == nil && shouldRestorePreservedExtraPodSpec(dst, preserved) {
		cp := *preserved.ExtraPodSpec.DeepCopy()
		dst.ExtraPodSpec = &cp
	}
	restoreMainContainerFieldOrigins(dst, preserved, mainContainerPresent)
	if dst.ExtraPodSpec != nil && dst.ExtraPodSpec.MainContainer != nil &&
		dst.ExtraPodSpec.MainContainer.Name == "" &&
		preserved.ExtraPodSpec != nil && preserved.ExtraPodSpec.MainContainer != nil {
		dst.ExtraPodSpec.MainContainer.Name = preserved.ExtraPodSpec.MainContainer.Name
	}
}

func restoreSharedAlphaOnlyDisabledFeatures(dst *DynamoComponentDeploymentSharedSpec, preserved *DynamoComponentDeploymentSharedSpec) {
	if dst.ScalingAdapter == nil && preserved.ScalingAdapter != nil && !preserved.ScalingAdapter.Enabled {
		dst.ScalingAdapter = preserved.ScalingAdapter.DeepCopy()
	}
	if dst.GPUMemoryService == nil && preserved.GPUMemoryService != nil && !preserved.GPUMemoryService.Enabled {
		dst.GPUMemoryService = preserved.GPUMemoryService.DeepCopy()
	}
	if dst.Failover == nil && preserved.Failover != nil && !preserved.Failover.Enabled {
		dst.Failover = preserved.Failover.DeepCopy()
	}
	if dst.Checkpoint == nil && preserved.Checkpoint != nil && !preserved.Checkpoint.Enabled {
		dst.Checkpoint = preserved.Checkpoint.DeepCopy()
	}
}

func saveSharedAlphaOnlySpec(src, save *DynamoComponentDeploymentSharedSpec, includeOriginSplits bool) {
	if src == nil || save == nil {
		return
	}
	hasSave := false

	if sharedSubComponentTypeNeedsSave(src) {
		save.SubComponentType = src.SubComponentType
		hasSave = true
	}
	if src.DynamoNamespace != nil {
		save.DynamoNamespace = ptr.To(*src.DynamoNamespace)
		hasSave = true
	}
	if src.Autoscaling != nil {
		save.Autoscaling = src.Autoscaling.DeepCopy()
		hasSave = true
	}
	if src.Ingress != nil {
		save.Ingress = src.Ingress.DeepCopy()
		hasSave = true
	}
	if len(src.Annotations) > 0 {
		save.Annotations = maps.Clone(src.Annotations)
		hasSave = true
	}
	if len(src.Labels) > 0 {
		save.Labels = maps.Clone(src.Labels)
		hasSave = true
	}
	if src.EnvFromSecret != nil {
		save.EnvFromSecret = ptr.To(*src.EnvFromSecret)
		hasSave = true
	}
	if resourcesNeedPreservation(src.Resources) {
		save.Resources = src.Resources.DeepCopy()
		hasSave = true
	}
	if len(src.VolumeMounts) > 0 && !volumeMountsRoundTripThroughHub(src.VolumeMounts) {
		save.VolumeMounts = slices.Clone(src.VolumeMounts)
		hasSave = true
	}
	if sharedMemoryNeedsPreservation(src.SharedMemory) {
		save.SharedMemory = src.SharedMemory.DeepCopy()
		hasSave = true
	}
	if extraPodMetadataNeedsPreservation(src.ExtraPodMetadata) {
		save.ExtraPodMetadata = src.ExtraPodMetadata.DeepCopy()
		hasSave = true
	}
	if extraPodSpecNeedsPreservation(src.ExtraPodSpec) ||
		alphaFrontendSidecarConflictNeedsPreservation(src.FrontendSidecar, src.ExtraPodSpec) {
		save.ExtraPodSpec = src.ExtraPodSpec.DeepCopy()
		hasSave = true
	}
	if src.ScalingAdapter != nil && !src.ScalingAdapter.Enabled {
		save.ScalingAdapter = src.ScalingAdapter.DeepCopy()
		hasSave = true
	}
	if src.FrontendSidecar != nil {
		save.FrontendSidecar = src.FrontendSidecar.DeepCopy()
		hasSave = true
	}
	if src.GPUMemoryService != nil && !src.GPUMemoryService.Enabled {
		save.GPUMemoryService = src.GPUMemoryService.DeepCopy()
		hasSave = true
	}
	if src.Failover != nil && !src.Failover.Enabled {
		save.Failover = src.Failover.DeepCopy()
		hasSave = true
	}
	if includeOriginSplits || hasSave || sharedMainContainerFieldOriginsNeedSave(src) {
		saveSharedMainContainerOrigins(src, save)
	}
}

func sharedAlphaSpecSaveIsZero(save *DynamoComponentDeploymentSharedSpec) bool {
	return save == nil || apiequality.Semantic.DeepEqual(*save, DynamoComponentDeploymentSharedSpec{})
}

func sharedComponentTypeToHub(src *DynamoComponentDeploymentSharedSpec) v1beta1.ComponentType {
	if src == nil {
		return ""
	}
	if sharedSubComponentTypePromotesToHubComponentType(src) {
		return v1beta1.ComponentType(src.SubComponentType)
	}
	return v1beta1.ComponentType(src.ComponentType)
}

func sharedComponentTypeFromHub(src v1beta1.ComponentType) (componentType string, subComponentType string) {
	switch src {
	case v1beta1.ComponentTypePrefill, v1beta1.ComponentTypeDecode:
		return string(v1beta1.ComponentTypeWorker), string(src)
	default:
		return string(src), ""
	}
}

func sharedSubComponentTypeNeedsSave(src *DynamoComponentDeploymentSharedSpec) bool {
	return src != nil &&
		src.SubComponentType != "" &&
		!sharedSubComponentTypePromotesToHubComponentType(src)
}

func sharedSubComponentTypePromotesToHubComponentType(src *DynamoComponentDeploymentSharedSpec) bool {
	return src != nil &&
		src.ComponentType == string(v1beta1.ComponentTypeWorker) &&
		(src.SubComponentType == string(v1beta1.ComponentTypePrefill) ||
			src.SubComponentType == string(v1beta1.ComponentTypeDecode))
}

func sharedMainContainerFieldOriginsNeedSave(src *DynamoComponentDeploymentSharedSpec) bool {
	if src == nil || src.ExtraPodSpec == nil || src.ExtraPodSpec.MainContainer == nil {
		return false
	}
	main := src.ExtraPodSpec.MainContainer

	// These fields decompose back into dedicated v1alpha1 fields unless the
	// sparse save records that they originated from ExtraPodSpec.MainContainer.
	return main.Name != "" ||
		len(main.Env) > 0 ||
		!resourceRequirementsEqual(main.Resources, corev1.ResourceRequirements{}) ||
		len(main.VolumeMounts) > 0 ||
		main.LivenessProbe != nil ||
		main.ReadinessProbe != nil
}

func saveSharedMainContainerOrigins(src, save *DynamoComponentDeploymentSharedSpec) {
	if src == nil || save == nil || src.ExtraPodSpec == nil || src.ExtraPodSpec.MainContainer == nil {
		return
	}
	main := src.ExtraPodSpec.MainContainer

	if main.Name != "" {
		ensureExtraPodSpecMainContainer(save).Name = main.Name
	}
	if len(src.Envs) > 0 || len(main.Env) > 0 {
		save.Envs = slices.Clone(src.Envs)
		ensureExtraPodSpecMainContainer(save).Env = slices.Clone(main.Env)
	}
	if src.EnvFromSecret != nil || len(main.EnvFrom) > 0 {
		if src.EnvFromSecret != nil {
			save.EnvFromSecret = ptr.To(*src.EnvFromSecret)
		}
		ensureExtraPodSpecMainContainer(save).EnvFrom = slices.Clone(main.EnvFrom)
	}
	if src.Resources != nil || !resourceRequirementsEqual(main.Resources, corev1.ResourceRequirements{}) {
		save.Resources = src.Resources.DeepCopy()
		ensureExtraPodSpecMainContainer(save).Resources = *main.Resources.DeepCopy()
	}
	if len(src.VolumeMounts) > 0 || len(main.VolumeMounts) > 0 {
		save.VolumeMounts = slices.Clone(src.VolumeMounts)
		ensureExtraPodSpecMainContainer(save).VolumeMounts = cloneNativeVolumeMounts(main.VolumeMounts)
	}
	if src.LivenessProbe != nil || main.LivenessProbe != nil {
		save.LivenessProbe = src.LivenessProbe.DeepCopy()
		ensureExtraPodSpecMainContainer(save).LivenessProbe = main.LivenessProbe.DeepCopy()
	}
	if src.ReadinessProbe != nil || main.ReadinessProbe != nil {
		save.ReadinessProbe = src.ReadinessProbe.DeepCopy()
		ensureExtraPodSpecMainContainer(save).ReadinessProbe = main.ReadinessProbe.DeepCopy()
	}
}

func sharedMemoryNeedsPreservation(src *SharedMemorySpec) bool {
	if src == nil {
		return false
	}
	if src.Disabled {
		return !src.Size.IsZero()
	}
	return src.Size.IsZero()
}

func shouldRestorePreservedSharedMemory(dst, preserved *SharedMemorySpec) bool {
	if !sharedMemoryNeedsPreservation(preserved) {
		return false
	}
	if dst == nil {
		return true
	}
	if preserved.Disabled {
		return dst.Disabled && dst.Size.Sign() == 0
	}
	return false
}

func resourcesNeedPreservation(src *Resources) bool {
	return src != nil && !reflect.DeepEqual(src, resourcesFromNative(resourcesToNative(src)))
}

func shouldRestorePreservedResources(dst, preserved *Resources) bool {
	if !resourcesNeedPreservation(preserved) {
		return false
	}
	return resourceRequirementsEqual(resourcesToNativeOrZero(dst), resourcesToNative(preserved))
}

func resourcesToNativeOrZero(src *Resources) corev1.ResourceRequirements {
	if src == nil {
		return corev1.ResourceRequirements{}
	}
	return resourcesToNative(src)
}

func extraPodMetadataNeedsPreservation(src *ExtraPodMetadata) bool {
	return src != nil && len(src.Annotations) == 0 && len(src.Labels) == 0
}

func saveSharedHubOnlySpec(src *v1beta1.DynamoComponentDeploymentSharedSpec, converted *DynamoComponentDeploymentSharedSpec, save *v1beta1.DynamoComponentDeploymentSharedSpec) error {
	if src == nil || save == nil {
		return nil
	}
	if sharedFrontendSidecarNeedsPreservation(src, converted) {
		save.FrontendSidecar = ptr.To(*src.FrontendSidecar)
	}
	if src.PodTemplate != nil {
		if err := saveSharedPodTemplateHubOnlyFields(src, converted, save); err != nil {
			return err
		}
	}
	if experimentalIsHubOnlyShape(src.Experimental) {
		save.Experimental = src.Experimental.DeepCopy()
	}
	return nil
}

func sharedFrontendSidecarNeedsPreservation(src *v1beta1.DynamoComponentDeploymentSharedSpec, converted *DynamoComponentDeploymentSharedSpec) bool {
	if src == nil || src.FrontendSidecar == nil {
		return false
	}
	if converted == nil {
		return true
	}
	return converted.FrontendSidecar == nil
}

func saveSharedPodTemplateHubOnlyFields(src *v1beta1.DynamoComponentDeploymentSharedSpec, converted *DynamoComponentDeploymentSharedSpec, save *v1beta1.DynamoComponentDeploymentSharedSpec) error {
	projected, err := buildSharedPodTemplateFromAlpha(converted, false, false)
	if err != nil {
		return err
	}
	frontendSidecar := ""
	if save.FrontendSidecar != nil {
		frontendSidecar = *save.FrontendSidecar
	}
	save.PodTemplate = sparseSharedHubOnlyPodTemplate(src.PodTemplate, projected, frontendSidecar)
	return nil
}

func sparseSharedHubOnlyPodTemplate(src, projected *corev1.PodTemplateSpec, frontendSidecar string) *corev1.PodTemplateSpec {
	if src == nil {
		return nil
	}
	meta := sharedHubOnlyPodTemplateMetadata(src)

	// Keep metadata, generated-shape markers, and container order as separate
	// save signals so metadata-only preservation cannot freeze live sidecar order.
	needsMetadataSave := !apiequality.Semantic.DeepEqual(meta, metav1.ObjectMeta{})
	needsGeneratedMainMarker := sharedHubOnlyPodTemplateGeneratedMainNeedsSave(src, projected)
	needsEmptyPodTemplateMarker := projected == nil && podTemplateIsZero(src)
	needsContainerOrderSave := sharedHubOnlyPodTemplateContainerOrderNeedsSave(src, projected)
	needsFrontendSidecarKey := frontendSidecar != "" && podTemplateHasContainer(src, frontendSidecar)
	if !needsMetadataSave && !needsGeneratedMainMarker && !needsEmptyPodTemplateMarker &&
		!needsContainerOrderSave && !needsFrontendSidecarKey &&
		!sharedHubOnlyPodTemplateContainerFieldsNeedSave(src, projected) {
		return nil
	}
	out := &corev1.PodTemplateSpec{}
	if needsMetadataSave {
		out.ObjectMeta = meta
	}
	saveSharedHubOnlyPodTemplateContainers(src, projected, out, needsContainerOrderSave, frontendSidecar)
	if !needsMetadataSave && len(out.Spec.Containers) == 0 && !needsGeneratedMainMarker && !needsEmptyPodTemplateMarker {
		return nil
	}
	// A non-nil empty PodTemplate preserves that the hub intentionally had no
	// generated "main" container, even when the v1alpha1 projection creates one.
	return out
}

func sharedHubOnlyPodTemplateMetadata(src *corev1.PodTemplateSpec) metav1.ObjectMeta {
	meta := *src.ObjectMeta.DeepCopy()
	meta.Labels = nil
	meta.Annotations = nil
	return meta
}

func sharedHubOnlyPodTemplateGeneratedMainNeedsSave(src, projected *corev1.PodTemplateSpec) bool {
	if src == nil || projected == nil {
		return false
	}
	return !hasContainerNamed(src.Spec.Containers, mainContainerName) &&
		hasContainerNamed(projected.Spec.Containers, mainContainerName)
}

func sharedHubOnlyPodTemplateContainerOrderNeedsSave(src, projected *corev1.PodTemplateSpec) bool {
	if src == nil || projected == nil {
		return false
	}
	srcNames := podTemplateContainerNames(src)
	projectedNames := podTemplateContainerNames(projected)
	if !hasContainerNamed(src.Spec.Containers, mainContainerName) {
		projectedNames = slices.DeleteFunc(projectedNames, func(name string) bool {
			return name == mainContainerName
		})
	}
	return !slices.Equal(srcNames, projectedNames)
}

func sharedHubOnlyPodTemplateContainerFieldsNeedSave(src, projected *corev1.PodTemplateSpec) bool {
	if src == nil {
		return false
	}
	if projected == nil {
		return len(src.Spec.Containers) > 0
	}
	return !apiequality.Semantic.DeepEqual(src.Spec.Containers, projected.Spec.Containers)
}

func podTemplateContainerNames(podTemplate *corev1.PodTemplateSpec) []string {
	if podTemplate == nil {
		return nil
	}
	names := make([]string, 0, len(podTemplate.Spec.Containers))
	for _, container := range podTemplate.Spec.Containers {
		names = append(names, container.Name)
	}
	return names
}

func saveSharedHubOnlyPodTemplateContainers(src, projected, save *corev1.PodTemplateSpec, preserveOrder bool, frontendSidecar string) {
	projectedContainers := []corev1.Container(nil)
	if projected != nil {
		projectedContainers = projected.Spec.Containers
	}
	for _, srcContainer := range src.Spec.Containers {
		savedContainer := corev1.Container{Name: srcContainer.Name}
		projectedContainerFound := false
		var projectedContainer corev1.Container
		if found, ok := findContainerByName(projectedContainers, srcContainer.Name); ok {
			projectedContainerFound = true
			projectedContainer = found
			saveSharedHubOnlyContainerFields(&savedContainer, srcContainer, projectedContainer)
		} else if !containerHasOnlyName(srcContainer) {
			savedContainer = *srcContainer.DeepCopy()
		}
		if preserveOrder ||
			srcContainer.Name == frontendSidecar ||
			sharedGeneratedMainContainerKeyNeedsSave(srcContainer, projectedContainer, projectedContainerFound) ||
			!containerHasOnlyName(savedContainer) {
			save.Spec.Containers = append(save.Spec.Containers, savedContainer)
		}
	}
}

func sharedGeneratedMainContainerKeyNeedsSave(src, projected corev1.Container, projectedContainerFound bool) bool {
	if src.Name != mainContainerName {
		return false
	}
	if !projectedContainerFound {
		return containerHasOnlyName(src)
	}
	if containerHasOnlyName(src) {
		return true
	}
	for _, projectedMount := range projected.VolumeMounts {
		if _, ok := findPreservedVolumeMount(src.VolumeMounts, projectedMount); !ok {
			return true
		}
	}
	return false
}

func saveSharedHubOnlyContainerFields(save *corev1.Container, src, projected corev1.Container) {
	if src.Name != mainContainerName {
		if !apiequality.Semantic.DeepEqual(src, projected) {
			saveSharedHubOnlyFrontendSidecarContainerFields(save, src, projected)
		}
		return
	}
	if len(src.VolumeMounts) == 0 {
		return
	}
	save.VolumeMounts = sparseSharedHubOnlyVolumeMounts(src.VolumeMounts, projected.VolumeMounts)
}

func saveSharedHubOnlyFrontendSidecarContainerFields(save *corev1.Container, src, projected corev1.Container) {
	*save = *src.DeepCopy()
	clearSharedFrontendSidecarRepresentedContainerFields(save, projected.EnvFrom)
}

func clearSharedFrontendSidecarRepresentedContainerFields(container *corev1.Container, projectedEnvFrom []corev1.EnvFromSource) {
	container.Image = ""
	container.Args = nil
	container.Env = nil
	container.EnvFrom = sparseSharedHubOnlyFrontendSidecarEnvFrom(container.EnvFrom, projectedEnvFrom)
}

func sparseSharedHubOnlyFrontendSidecarEnvFrom(src, projected []corev1.EnvFromSource) []corev1.EnvFromSource {
	if apiequality.Semantic.DeepEqual(src, projected) {
		return nil
	}
	return cloneNativeEnvFromSources(src)
}

func sparseSharedHubOnlyVolumeMounts(src, projected []corev1.VolumeMount) []corev1.VolumeMount {
	out := make([]corev1.VolumeMount, 0, len(src))
	for _, srcMount := range src {
		savedMount := corev1.VolumeMount{
			Name:      srcMount.Name,
			MountPath: srcMount.MountPath,
		}
		if projectedMount, ok := findPreservedVolumeMount(projected, srcMount); !ok ||
			!apiequality.Semantic.DeepEqual(srcMount, projectedMount) {
			savedMount = *srcMount.DeepCopy()
		}
		out = append(out, savedMount)
	}
	return out
}

func sharedHubSpecSaveIsZero(save *v1beta1.DynamoComponentDeploymentSharedSpec) bool {
	if save == nil {
		return true
	}
	normalized := save.DeepCopy()
	// ComponentName is the list-map key for DGD component preservation. It
	// identifies the sparse save entry, but is not itself hub-only payload.
	normalized.ComponentName = ""
	return apiequality.Semantic.DeepEqual(*normalized, v1beta1.DynamoComponentDeploymentSharedSpec{})
}

// ---------------------------------------------------------------------------
// Shared-memory
// ---------------------------------------------------------------------------

// ConvertFromSharedMemorySpec converts the shared-memory struct into the
// v1beta1 scalar representation.
func ConvertFromSharedMemorySpec(src *SharedMemorySpec, dst *resource.Quantity) {
	if src.Disabled {
		*dst = resource.MustParse("0")
		return
	}
	*dst = src.Size
}

// ConvertToSharedMemorySpec converts the v1beta1 shared-memory scalar
// representation into the shared-memory struct.
func ConvertToSharedMemorySpec(src *resource.Quantity, dst *SharedMemorySpec) {
	if src.Sign() == 0 {
		// Canonical v1beta1 "size=0" <-> v1alpha1 Disabled=true. See
		// ConvertFromSharedMemorySpec for the forward direction.
		//
		// Size carries the incoming canonical Quantity value (not the Go zero
		// value) so that every apply produces a spec that is reflect.DeepEqual
		// to what's in etcd. A bare Quantity{} and a JSON-round-tripped
		// Quantity serialize identically to "0" but differ in internal state
		// (Format, cached string), and the API server's generation-bump check
		// uses DeepEqual -- so emitting a bare Quantity{} here would cause
		// every `kubectl apply` to bump `.metadata.generation` even though the
		// spec is byte-identical.
		*dst = SharedMemorySpec{Disabled: true, Size: *src}
		return
	}
	*dst = SharedMemorySpec{Size: *src}
}

// ---------------------------------------------------------------------------
// Volume mounts and compilation cache
// ---------------------------------------------------------------------------
//
// v1alpha1 carries PVC bindings in two slots: a flat VolumeMounts list (with
// a per-entry UseAsCompilationCache flag) and ExtraPodSpec.MainContainer.
// v1beta1 consolidates volume mounts into the podTemplate main container and
// hoists the "cache" flag into a first-class CompilationCacheConfig field.
//
// Provenance rule for the round-trip:
//   - CompilationCacheConfig round-trips as a single flagged v1alpha1
//     VolumeMount (UseAsCompilationCache=true).
//   - All other container-level volumeMounts live on
//     ExtraPodSpec.MainContainer in v1alpha1 (they were set via the
//     extraPodSpec escape hatch anyway, so placing them there mirrors the
//     v1alpha1 reconcile-merge behaviour in graph.go).

// convertVolumeMountsFromHub is the v1beta1 -> v1alpha1 inverse: it synthesises
// a single flagged entry in dst.VolumeMounts when the v1beta1 side declares
// a CompilationCacheConfig. Non-cache mounts on the main container are
// preserved through decomposePodTemplateFromHub's ExtraPodSpec.MainContainer copy,
// not here.
func convertVolumeMountsFromHub(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst *DynamoComponentDeploymentSharedSpec) {
	if src.CompilationCache == nil {
		return
	}
	dst.VolumeMounts = []VolumeMount{{
		Name:                  src.CompilationCache.PVCName,
		MountPoint:            src.CompilationCache.MountPath,
		UseAsCompilationCache: true,
	}}
}

// Restore alpha-only cache flags only when live beta still matches their lossy projection.
func restoreSharedPreservedFlatVolumeMounts(dst, preserved *DynamoComponentDeploymentSharedSpec, src *v1beta1.DynamoComponentDeploymentSharedSpec) {
	if dst == nil || preserved == nil || src == nil || volumeMountsRoundTripThroughHub(preserved.VolumeMounts) {
		return
	}
	if !firstPreservedCompilationCacheMatches(src.CompilationCache, preserved.VolumeMounts) {
		return
	}
	if !volumeMountsEqual(dst.VolumeMounts, visiblePreservedVolumeMountProjection(src, preserved.VolumeMounts)) {
		return
	}
	restorablePreserved := restorablePreservedCompilationCacheMounts(src, preserved)
	dst.VolumeMounts = mergePreservedCompilationCacheVolumeMounts(restorablePreserved, dst.VolumeMounts)
}

func firstPreservedCompilationCacheMatches(compilationCache *v1beta1.CompilationCacheConfig, mounts []VolumeMount) bool {
	for _, mount := range mounts {
		if !mount.UseAsCompilationCache {
			continue
		}
		return compilationCacheMatchesVolumeMount(compilationCache, mount)
	}
	return compilationCache == nil
}

func compilationCacheMatchesVolumeMount(compilationCache *v1beta1.CompilationCacheConfig, mount VolumeMount) bool {
	return compilationCache != nil &&
		compilationCache.PVCName == mount.Name &&
		compilationCache.MountPath == mount.MountPoint
}

func restorablePreservedCompilationCacheMounts(src *v1beta1.DynamoComponentDeploymentSharedSpec, preserved *DynamoComponentDeploymentSharedSpec) []VolumeMount {
	main, mainPresent := sharedMainContainer(src)
	secondaryMountsProjected := mainPresent || hasPodTemplateContent(preserved, false)
	firstCompilationCacheSeen := false
	live := make([]VolumeMount, 0, len(preserved.VolumeMounts))
	for _, mount := range preserved.VolumeMounts {
		if !mount.UseAsCompilationCache {
			continue
		}
		if !firstCompilationCacheSeen {
			firstCompilationCacheSeen = true
			if compilationCacheMatchesVolumeMount(src.CompilationCache, mount) {
				live = append(live, mount)
			}
			continue
		}
		// Secondary cache flags have no beta field. When alpha content creates a
		// beta main container, its matching mount is their observable
		// representation and absence means deletion. Cache-only alpha objects do
		// not create a pod template, so those flags remain sparse-preserved state.
		if !secondaryMountsProjected || nativeVolumeMountHasNamePath(main.VolumeMounts, mount.Name, mount.MountPoint) {
			live = append(live, mount)
		}
	}
	return live
}

func visiblePreservedVolumeMountProjection(src *v1beta1.DynamoComponentDeploymentSharedSpec, mounts []VolumeMount) []VolumeMount {
	projected := make([]VolumeMount, 0, len(mounts))
	firstCompilationCacheSeen := false
	for _, mount := range mounts {
		if !mount.UseAsCompilationCache {
			continue
		}
		if firstCompilationCacheSeen {
			continue
		}
		projected = append(projected, mount)
		firstCompilationCacheSeen = true
	}
	if main, ok := sharedMainContainer(src); ok && len(main.VolumeMounts) > 0 {
		projected = appendMissingVolumeMounts(projected, volumeMountsFromNative(main.VolumeMounts))
	}
	return projected
}

func sharedMainContainer(src *v1beta1.DynamoComponentDeploymentSharedSpec) (corev1.Container, bool) {
	if src == nil || src.PodTemplate == nil {
		return corev1.Container{}, false
	}
	return findContainerByName(src.PodTemplate.Spec.Containers, mainContainerName)
}

func volumeMountsEqual(a, b []VolumeMount) bool {
	return slices.EqualFunc(a, b, func(left, right VolumeMount) bool {
		return left.Name == right.Name &&
			left.MountPoint == right.MountPoint &&
			left.UseAsCompilationCache == right.UseAsCompilationCache
	})
}

func mergePreservedCompilationCacheVolumeMounts(preserved, current []VolumeMount) []VolumeMount {
	out := make([]VolumeMount, 0, len(preserved)+len(current))
	for _, mount := range preserved {
		if mount.UseAsCompilationCache && !flatVolumeMountHasNamePath(out, mount.Name, mount.MountPoint) {
			out = append(out, mount)
		}
	}
	for _, mount := range current {
		if !flatVolumeMountHasNamePath(out, mount.Name, mount.MountPoint) {
			out = append(out, mount)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Experimental (gpuMemoryService, failover, checkpoint)
// ---------------------------------------------------------------------------

// v1alpha1 and v1beta1 use different serialized values for GMS modes:
//
//   - v1alpha1: "intraPod", "interPod"
//   - v1beta1:  "IntraPod", "InterPod"
//
// conversion-gen only casts the named string types and would therefore emit
// values rejected by the target version's CRD schema. Unknown values are kept
// verbatim so conversion does not silently discard data.
func gmsModeToV1beta1(mode GPUMemoryServiceMode) v1beta1.GPUMemoryServiceMode {
	switch mode {
	case GMSModeIntraPod:
		return v1beta1.GMSModeIntraPod
	case GMSModeInterPod:
		return v1beta1.GMSModeInterPod
	default:
		return v1beta1.GPUMemoryServiceMode(mode)
	}
}

func gmsModeFromV1beta1(mode v1beta1.GPUMemoryServiceMode) GPUMemoryServiceMode {
	switch mode {
	case v1beta1.GMSModeIntraPod:
		return GMSModeIntraPod
	case v1beta1.GMSModeInterPod:
		return GMSModeInterPod
	default:
		return GPUMemoryServiceMode(mode)
	}
}

func convertExperimentalToHub(src *DynamoComponentDeploymentSharedSpec, dst *v1beta1.DynamoComponentDeploymentSharedSpec, s apiconversion.Scope) error {
	var exp *v1beta1.ExperimentalSpec
	ensureExp := func() *v1beta1.ExperimentalSpec {
		if exp == nil {
			exp = &v1beta1.ExperimentalSpec{}
		}
		return exp
	}

	if src.GPUMemoryService != nil && src.GPUMemoryService.Enabled {
		ensureExp().GPUMemoryService = &v1beta1.GPUMemoryServiceSpec{}
		if err := Convert_v1alpha1_GPUMemoryServiceSpec_To_v1beta1_GPUMemoryServiceSpec(src.GPUMemoryService, exp.GPUMemoryService, s); err != nil {
			return err
		}
	}

	if src.Failover != nil && src.Failover.Enabled {
		ensureExp().Failover = &v1beta1.FailoverSpec{}
		if err := Convert_v1alpha1_FailoverSpec_To_v1beta1_FailoverSpec(src.Failover, exp.Failover, s); err != nil {
			return err
		}
	}

	if src.Checkpoint != nil {
		ensureExp().Checkpoint = &v1beta1.ComponentCheckpointConfig{}
		if err := Convert_v1alpha1_ServiceCheckpointConfig_To_v1beta1_ComponentCheckpointConfig(src.Checkpoint, exp.Checkpoint, s); err != nil {
			return err
		}
	}

	dst.Experimental = exp
	return nil
}

func convertExperimentalFromHub(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst *DynamoComponentDeploymentSharedSpec, s apiconversion.Scope) error {
	if src.Experimental != nil && src.Experimental.GPUMemoryService != nil {
		dst.GPUMemoryService = &GPUMemoryServiceSpec{}
		if err := Convert_v1beta1_GPUMemoryServiceSpec_To_v1alpha1_GPUMemoryServiceSpec(src.Experimental.GPUMemoryService, dst.GPUMemoryService, s); err != nil {
			return err
		}
	}

	if src.Experimental != nil && src.Experimental.Failover != nil {
		dst.Failover = &FailoverSpec{}
		if err := Convert_v1beta1_FailoverSpec_To_v1alpha1_FailoverSpec(src.Experimental.Failover, dst.Failover, s); err != nil {
			return err
		}
	}

	if src.Experimental != nil && src.Experimental.Checkpoint != nil {
		dst.Checkpoint = &ServiceCheckpointConfig{}
		if err := Convert_v1beta1_ComponentCheckpointConfig_To_v1alpha1_ServiceCheckpointConfig(src.Experimental.Checkpoint, dst.Checkpoint, s); err != nil {
			return err
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// podTemplate (the big one)
// ---------------------------------------------------------------------------

// buildPodTemplateToHub composes the v1beta1 podTemplate from v1alpha1's flat
// fields (Resources, Envs, Probes, EnvFromSecret, ExtraPodSpec,
// ExtraPodMetadata, FrontendSidecar) following the same merge precedence the
// v1alpha1 controller uses at reconcile time: ExtraPodSpec.MainContainer wins
// over dedicated fields, except for env and volumeMounts which are additive.
func buildPodTemplateToHub(src *DynamoComponentDeploymentSharedSpec, dst *v1beta1.DynamoComponentDeploymentSharedSpec, podTemplateOrigin bool) error {
	podTpl, err := buildSharedPodTemplateFromAlpha(src, podTemplateOrigin, false)
	if err != nil {
		return err
	}
	if podTpl == nil {
		return nil
	}

	setFrontendSidecarReferenceToHub(src, dst)
	dst.PodTemplate = podTpl
	return nil
}

func hasPodTemplateContent(src *DynamoComponentDeploymentSharedSpec, podTemplateOrigin bool) bool {
	return src.Resources != nil ||
		len(src.Envs) > 0 ||
		src.EnvFromSecret != nil ||
		hasPodTemplateVolumeMounts(src.VolumeMounts) ||
		src.LivenessProbe != nil ||
		src.ReadinessProbe != nil ||
		!extraPodSpecIsZero(src.ExtraPodSpec) ||
		src.ExtraPodMetadata != nil ||
		src.FrontendSidecar != nil ||
		podTemplateOrigin
}

func shouldBuildPodTemplate(src *DynamoComponentDeploymentSharedSpec, podTemplateOrigin, hasFrontendSidecarRef bool) bool {
	return hasPodTemplateContent(src, podTemplateOrigin) || hasFrontendSidecarRef
}

func buildBasePodTemplate(src *DynamoComponentDeploymentSharedSpec) *corev1.PodTemplateSpec {
	podTpl := &corev1.PodTemplateSpec{}
	if src.ExtraPodMetadata != nil {
		if len(src.ExtraPodMetadata.Annotations) > 0 {
			podTpl.Annotations = maps.Clone(src.ExtraPodMetadata.Annotations)
		}
		if len(src.ExtraPodMetadata.Labels) > 0 {
			podTpl.Labels = maps.Clone(src.ExtraPodMetadata.Labels)
		}
	}
	if src.ExtraPodSpec != nil && src.ExtraPodSpec.PodSpec != nil {
		podTpl.Spec = *src.ExtraPodSpec.PodSpec.DeepCopy()
	}
	return podTpl
}

func mergeExtraPodSpecMainContainer(src *DynamoComponentDeploymentSharedSpec, mainBase *corev1.Container) error {
	if src.ExtraPodSpec == nil || src.ExtraPodSpec.MainContainer == nil {
		return nil
	}
	main := src.ExtraPodSpec.MainContainer.DeepCopy()
	baseEnvs := mainBase.Env
	dedicatedVolumeMounts := slices.Clone(mainBase.VolumeMounts)
	// Name must be "main" regardless of what MainContainer carried.
	main.Name = mainContainerName
	if err := mergo.Merge(mainBase, *main, mergo.WithOverride); err != nil {
		return fmt.Errorf("merge main container: %w", err)
	}
	mainBase.Env = mergeEnvs(baseEnvs, main.Env)
	// The v1alpha1 renderer merged extraPodSpec.mainContainer first, then
	// appended the dedicated service-level mounts. Preserve that ordering rather
	// than allowing mergo to replace the VolumeMounts slice.
	mainBase.VolumeMounts = append(slices.Clone(main.VolumeMounts), dedicatedVolumeMounts...)
	// StartupProbe has no dedicated v1alpha1 field; take it verbatim.
	if main.StartupProbe != nil {
		mainBase.StartupProbe = main.StartupProbe
	}
	return nil
}

func buildSharedPodTemplateFromAlpha(src *DynamoComponentDeploymentSharedSpec, podTemplateOrigin, hasFrontendSidecarRef bool) (*corev1.PodTemplateSpec, error) {
	if src == nil || !shouldBuildPodTemplate(src, podTemplateOrigin, hasFrontendSidecarRef) {
		return nil, nil
	}
	podTpl := buildBasePodTemplate(src)

	// Main container: base from dedicated fields.
	mainBase := buildMainContainerFromDedicated(src)

	// Merge ExtraPodSpec.MainContainer on top, except for fields with legacy
	// additive behavior handled by mergeExtraPodSpecMainContainer.
	if err := mergeExtraPodSpecMainContainer(src, &mainBase); err != nil {
		return nil, err
	}
	mainBase.Name = mainContainerName

	// Assemble containers: main first, then non-main user sidecars.
	containers := []corev1.Container{mainBase}
	for _, ctr := range podTpl.Spec.Containers {
		if ctr.Name != mainContainerName {
			containers = append(containers, ctr)
		}
	}
	podTpl.Spec.Containers = containers

	if src.FrontendSidecar != nil {
		appendFrontendSidecar(podTpl, src.FrontendSidecar)
	}
	return podTpl, nil
}

func setFrontendSidecarReferenceToHub(src *DynamoComponentDeploymentSharedSpec, dst *v1beta1.DynamoComponentDeploymentSharedSpec) {
	if src.FrontendSidecar != nil {
		dst.FrontendSidecar = ptr.To(defaultFrontendSidecarContainerName)
	}
}

// buildMainContainerFromDedicated collects the v1alpha1 flat fields into a
// corev1.Container named "main".
func buildMainContainerFromDedicated(src *DynamoComponentDeploymentSharedSpec) corev1.Container {
	ctr := corev1.Container{Name: mainContainerName}
	if src.Resources != nil {
		ctr.Resources = resourcesToNative(src.Resources)
	}
	if len(src.Envs) > 0 {
		ctr.Env = slices.Clone(src.Envs)
	}
	if src.EnvFromSecret != nil && *src.EnvFromSecret != "" {
		ctr.EnvFrom = append(ctr.EnvFrom, corev1.EnvFromSource{
			SecretRef: &corev1.SecretEnvSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: *src.EnvFromSecret},
			},
		})
	}
	if src.LivenessProbe != nil {
		ctr.LivenessProbe = src.LivenessProbe.DeepCopy()
	}
	if src.ReadinessProbe != nil {
		ctr.ReadinessProbe = src.ReadinessProbe.DeepCopy()
	}
	for _, vm := range src.VolumeMounts {
		mp := vm.MountPoint
		ctr.VolumeMounts = append(ctr.VolumeMounts, corev1.VolumeMount{
			Name:      vm.Name,
			MountPath: mp,
		})
	}
	return ctr
}

// decomposePodTemplateFromHub inverts buildPodTemplateToHub.
func decomposePodTemplateFromHub(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst, restored *DynamoComponentDeploymentSharedSpec) error {
	if src.PodTemplate == nil {
		decomposeMissingPodTemplate(src, dst, restored)
		return nil
	}

	podTpl := src.PodTemplate.DeepCopy()

	// ExtraPodMetadata from podTemplate.metadata.
	restoreExtraPodMetadataFromPodTemplate(podTpl, dst)

	// Pick out the main container; leave everything else as podSpec sidecars.
	var main *corev1.Container
	other := make([]corev1.Container, 0, len(podTpl.Spec.Containers))
	for i := range podTpl.Spec.Containers {
		if podTpl.Spec.Containers[i].Name == mainContainerName && main == nil {
			m := podTpl.Spec.Containers[i].DeepCopy()
			main = m
			continue
		}
		other = append(other, podTpl.Spec.Containers[i])
	}

	// FrontendSidecar name references have no native v1alpha1 representation.
	// Keep v1beta1-first references in the sparse hub save; restore
	// v1alpha1-origin sidecars from the sparse spoke save.
	other = restoreFrontendSidecarFromPodTemplate(src, dst, restored, other)

	restoreDedicatedFieldsFromMain(main, dst)

	// Put everything non-main into ExtraPodSpec. The main-container fields
	// that v1alpha1 can represent directly have already been moved into their
	// dedicated fields and cleared from main, so ExtraPodSpec only carries the
	// true escape-hatch remainder.
	podSpecCopy := podTpl.Spec.DeepCopy()
	podSpecCopy.Containers = other
	// The forward path (buildPodTemplateToHub) always emits a "main" container,
	// even when v1alpha1 had no main-container fields set (e.g. only
	// FrontendSidecar triggered hasAny). Skip recording it on the v1alpha1
	// side when every field other than Name is zero-valued, so that
	// ConvertFrom does not hallucinate an empty MainContainer.
	var mainCopy *corev1.Container
	if main != nil {
		m := main.DeepCopy()
		m.Name = "" // v1alpha1 MainContainer has no Name (it is always "main").
		if !containerIsEmpty(m) {
			mainCopy = m
		}
	}
	if !podSpecIsZero(podSpecCopy) || mainCopy != nil {
		eps := &ExtraPodSpec{}
		if !podSpecIsZero(podSpecCopy) {
			eps.PodSpec = podSpecCopy
		}
		if mainCopy != nil {
			eps.MainContainer = mainCopy
		}
		dst.ExtraPodSpec = eps
	}

	return nil
}

func decomposeMissingPodTemplate(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst, restored *DynamoComponentDeploymentSharedSpec) {
	if fs, ok := restoredDefaultFrontendSidecar(src, restored); ok {
		dst.FrontendSidecar = fs.DeepCopy()
	}
}

func restoreExtraPodMetadataFromPodTemplate(podTpl *corev1.PodTemplateSpec, dst *DynamoComponentDeploymentSharedSpec) {
	hasMetadata := len(podTpl.Annotations) > 0 || len(podTpl.Labels) > 0
	if hasMetadata {
		dst.ExtraPodMetadata = extraPodMetadataFromPodTemplate(podTpl)
	}
}

func extraPodMetadataFromPodTemplate(podTpl *corev1.PodTemplateSpec) *ExtraPodMetadata {
	return &ExtraPodMetadata{
		Annotations: maps.Clone(podTpl.Annotations),
		Labels:      maps.Clone(podTpl.Labels),
	}
}

func restoreFrontendSidecarFromPodTemplate(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst, restored *DynamoComponentDeploymentSharedSpec, other []corev1.Container) []corev1.Container {
	if src.FrontendSidecar == nil {
		return other
	}
	sidecarName := *src.FrontendSidecar
	if fs, ok := restoredDefaultFrontendSidecar(src, restored); ok {
		ctr, found := findContainerByName(other, sidecarName)
		if found {
			dst.FrontendSidecar = frontendSidecarSpecFromContainer(ctr, fs)
			return slices.DeleteFunc(other, func(candidate corev1.Container) bool {
				return candidate.Name == sidecarName
			})
		}
		dst.FrontendSidecar = fs.DeepCopy()
		return other
	}
	return other
}

func restoredDefaultFrontendSidecar(src *v1beta1.DynamoComponentDeploymentSharedSpec, restored *DynamoComponentDeploymentSharedSpec) (*FrontendSidecarSpec, bool) {
	if src == nil ||
		src.FrontendSidecar == nil ||
		*src.FrontendSidecar != defaultFrontendSidecarContainerName ||
		restored == nil ||
		restored.FrontendSidecar == nil {
		return nil, false
	}
	return restored.FrontendSidecar, true
}

func restoreSharedHubOnlyFields(dst, preserved *v1beta1.DynamoComponentDeploymentSharedSpec, src *DynamoComponentDeploymentSharedSpec) error {
	if dst == nil || preserved == nil {
		return nil
	}
	podTemplate, err := restoreSharedPodTemplateHubOnlyFields(preserved, dst.PodTemplate, dst.CompilationCache, src)
	if err != nil {
		return err
	}
	dst.PodTemplate = podTemplate
	restoreSharedHubOnlyFrontendSidecar(dst, preserved)
	if dst.Experimental == nil && experimentalIsHubOnlyShape(preserved.Experimental) {
		dst.Experimental = preserved.Experimental.DeepCopy()
	}
	return nil
}

func restoreSharedHubProjection(src *DynamoComponentDeploymentSharedSpec, dst, preserved *v1beta1.DynamoComponentDeploymentSharedSpec) error {
	if preserved == nil {
		return nil
	}
	if preserved.PodTemplate != nil {
		if err := buildPodTemplateToHub(src, dst, true); err != nil {
			return err
		}
	}
	return restoreSharedHubOnlyFields(dst, preserved, src)
}

func restoreSharedAlphaProjection(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst, preserved *DynamoComponentDeploymentSharedSpec) error {
	if preserved == nil {
		return nil
	}

	// The standard conversion decomposes the live pod template without origin
	// data. Clear every pod-derived field before applying the preserved origin so
	// the second projection is idempotent and cannot retain stale baseline data.
	dst.ExtraPodMetadata = nil
	dst.ExtraPodSpec = nil
	dst.Resources = nil
	dst.Envs = nil
	dst.EnvFromSecret = nil
	dst.VolumeMounts = nil
	dst.LivenessProbe = nil
	dst.ReadinessProbe = nil
	dst.FrontendSidecar = nil
	convertVolumeMountsFromHub(src, dst)
	if err := decomposePodTemplateFromHub(src, dst, preserved); err != nil {
		return err
	}
	// Restore lossy cache flags before field-origin reconstruction so every
	// compilation-cache mount is excluded from main-container origin matching.
	restoreSharedPreservedFlatVolumeMounts(dst, preserved, src)
	fillSharedAlphaOnlyFromPreserved(dst, preserved, sharedHasMainContainer(src))
	pruneEmptyExtraPodSpec(dst, preserved)
	return nil
}

func restoreSharedHubOnlyFrontendSidecar(dst, preserved *v1beta1.DynamoComponentDeploymentSharedSpec) {
	if dst.FrontendSidecar != nil || preserved.FrontendSidecar == nil {
		return
	}
	if sharedFrontendSidecarHasPreservedPodTemplateContainer(preserved) &&
		!podTemplateHasContainer(dst.PodTemplate, *preserved.FrontendSidecar) {
		return
	}
	dst.FrontendSidecar = ptr.To(*preserved.FrontendSidecar)
}

func sharedFrontendSidecarHasPreservedPodTemplateContainer(preserved *v1beta1.DynamoComponentDeploymentSharedSpec) bool {
	if preserved.PodTemplate == nil || preserved.FrontendSidecar == nil {
		return false
	}
	return podTemplateHasContainer(preserved.PodTemplate, *preserved.FrontendSidecar)
}

func podTemplateHasContainer(podTemplate *corev1.PodTemplateSpec, name string) bool {
	if podTemplate == nil {
		return false
	}
	_, ok := findContainerByName(podTemplate.Spec.Containers, name)
	return ok
}

func restoreSharedPodTemplateHubOnlyFields(preserved *v1beta1.DynamoComponentDeploymentSharedSpec, semantic *corev1.PodTemplateSpec, compilationCache *v1beta1.CompilationCacheConfig, src *DynamoComponentDeploymentSharedSpec) (*corev1.PodTemplateSpec, error) {
	if preserved == nil || preserved.PodTemplate == nil {
		if semantic == nil {
			return nil, nil
		}
		return semantic.DeepCopy(), nil
	}
	out := &corev1.PodTemplateSpec{}
	if semantic != nil {
		out = semantic.DeepCopy()
	}
	dropGeneratedCompilationCacheMount(out, preserved.PodTemplate, compilationCache, src)
	dropGeneratedMainContainer(out, preserved.PodTemplate, compilationCache, src)
	restoreSharedPodTemplateMissingHubOnlyContainers(out, preserved.PodTemplate, src)
	if err := restoreSharedPodTemplateExistingHubOnlyContainers(out, preserved.PodTemplate, src); err != nil {
		return nil, err
	}
	restoreSharedPodTemplateContainerOrder(out, preserved.PodTemplate)
	restoreSharedHubOnlyPodTemplateMetadata(&out.ObjectMeta, preserved.PodTemplate.ObjectMeta)
	restoreSharedHubOnlyFlatVolumeMountFields(out, preserved.PodTemplate, src)
	if podTemplateIsZero(preserved.PodTemplate) && podTemplateIsZero(out) {
		return out, nil
	}
	return nilIfEmptyPodTemplate(out), nil
}

func restoreSharedPodTemplateMissingHubOnlyContainers(dst, preserved *corev1.PodTemplateSpec, src *DynamoComponentDeploymentSharedSpec) {
	if dst == nil || preserved == nil {
		return
	}
	for _, savedContainer := range preserved.Spec.Containers {
		if containerHasOnlyName(savedContainer) ||
			podTemplateHasContainer(dst, savedContainer.Name) ||
			preservedGeneratedFrontendSidecarWasDeleted(savedContainer, src) {
			continue
		}
		dst.Spec.Containers = append(dst.Spec.Containers, *savedContainer.DeepCopy())
	}
}

func preservedGeneratedFrontendSidecarWasDeleted(savedContainer corev1.Container, src *DynamoComponentDeploymentSharedSpec) bool {
	// The default sidecar container is generated from v1alpha1 FrontendSidecar.
	// If that live field is gone, the preserved hub-only remainder is stale.
	return savedContainer.Name == defaultFrontendSidecarContainerName &&
		(src == nil || src.FrontendSidecar == nil)
}

func restoreSharedPodTemplateExistingHubOnlyContainers(dst, preserved *corev1.PodTemplateSpec, src *DynamoComponentDeploymentSharedSpec) error {
	if dst == nil || preserved == nil || src == nil || src.FrontendSidecar == nil {
		return nil
	}
	savedContainer, ok := findContainerByName(preserved.Spec.Containers, defaultFrontendSidecarContainerName)
	if !ok || containerHasOnlyName(savedContainer) {
		return nil
	}
	for i := range dst.Spec.Containers {
		if dst.Spec.Containers[i].Name != defaultFrontendSidecarContainerName {
			continue
		}
		return restoreSharedHubOnlyFrontendSidecarContainerFields(&dst.Spec.Containers[i], savedContainer, src.FrontendSidecar)
	}
	return nil
}

func restoreSharedHubOnlyFrontendSidecarContainerFields(dst *corev1.Container, preserved corev1.Container, src *FrontendSidecarSpec) error {
	saved := preserved.DeepCopy()
	saved.Name = ""
	saved.Image = ""
	saved.Args = nil
	saved.Env = nil
	if len(saved.EnvFrom) > 0 {
		envFrom, ok := restoreSharedHubOnlyFrontendSidecarEnvFrom(dst.EnvFrom, saved.EnvFrom, src)
		if !ok {
			saved.EnvFrom = nil
		} else {
			saved.EnvFrom = envFrom
		}
	}
	if err := mergo.Merge(dst, *saved, mergo.WithOverride); err != nil {
		return fmt.Errorf("restore frontend sidecar hub-only container fields: %w", err)
	}
	return nil
}

func restoreSharedHubOnlyFrontendSidecarEnvFrom(dst, preserved []corev1.EnvFromSource, src *FrontendSidecarSpec) ([]corev1.EnvFromSource, bool) {
	if src == nil || src.EnvFromSecret == nil || *src.EnvFromSecret == "" {
		return cloneNativeEnvFromSources(preserved), true
	}
	if len(dst) != 1 || len(preserved) != 1 ||
		dst[0].SecretRef == nil || preserved[0].SecretRef == nil ||
		dst[0].SecretRef.Name != *src.EnvFromSecret ||
		preserved[0].SecretRef.Name != *src.EnvFromSecret {
		return nil, false
	}
	out := cloneNativeEnvFromSources(preserved)
	out[0].SecretRef.Name = dst[0].SecretRef.Name
	return out, true
}

func restoreSharedPodTemplateContainerOrder(dst, preserved *corev1.PodTemplateSpec) {
	if dst == nil ||
		preserved == nil ||
		len(preserved.Spec.Containers) < 2 ||
		len(preserved.Spec.Containers) != len(dst.Spec.Containers) {
		return
	}
	for _, container := range dst.Spec.Containers {
		if !hasContainerNamed(preserved.Spec.Containers, container.Name) {
			return
		}
	}
	remaining := slices.Clone(dst.Spec.Containers)
	out := make([]corev1.Container, 0, len(dst.Spec.Containers))
	for _, savedContainer := range preserved.Spec.Containers {
		for i := range remaining {
			if remaining[i].Name != savedContainer.Name {
				continue
			}
			out = append(out, remaining[i])
			remaining = slices.Delete(remaining, i, i+1)
			break
		}
	}
	out = append(out, remaining...)
	dst.Spec.Containers = out
}

func dropGeneratedMainContainer(dst, preserved *corev1.PodTemplateSpec, compilationCache *v1beta1.CompilationCacheConfig, src *DynamoComponentDeploymentSharedSpec) {
	if hasContainerNamed(preserved.Spec.Containers, mainContainerName) {
		return
	}
	if src != nil && resourcesNeedPreservation(src.Resources) {
		return
	}
	for i := range dst.Spec.Containers {
		if dst.Spec.Containers[i].Name != mainContainerName {
			continue
		}
		if mainContainerHasOnlyGeneratedFields(dst.Spec.Containers[i], compilationCache) {
			dst.Spec.Containers = slices.Delete(dst.Spec.Containers, i, i+1)
			if len(dst.Spec.Containers) == 0 {
				dst.Spec.Containers = nil
			}
			return
		}
	}
}

func mainContainerHasOnlyGeneratedFields(container corev1.Container, compilationCache *v1beta1.CompilationCacheConfig) bool {
	cp := container.DeepCopy()
	cp.Name = ""
	if compilationCache != nil {
		cp.VolumeMounts = slices.DeleteFunc(cp.VolumeMounts, func(mount corev1.VolumeMount) bool {
			return mount.Name == compilationCache.PVCName && mount.MountPath == compilationCache.MountPath
		})
	}
	return containerIsEmpty(cp)
}

func containerHasOnlyName(container corev1.Container) bool {
	cp := container.DeepCopy()
	cp.Name = ""
	return containerIsEmpty(cp)
}

func dropGeneratedCompilationCacheMount(dst, preserved *corev1.PodTemplateSpec, compilationCache *v1beta1.CompilationCacheConfig, src *DynamoComponentDeploymentSharedSpec) {
	if compilationCache == nil || preservedHasVolumeMount(preserved, compilationCache.PVCName, compilationCache.MountPath) ||
		sourceExtraPodMainContainerVolumeMountMatches(srcExtraPodSpec(src), corev1.VolumeMount{Name: compilationCache.PVCName, MountPath: compilationCache.MountPath}) {
		return
	}
	for i := range dst.Spec.Containers {
		if dst.Spec.Containers[i].Name != mainContainerName {
			continue
		}
		dst.Spec.Containers[i].VolumeMounts = slices.DeleteFunc(dst.Spec.Containers[i].VolumeMounts, func(mount corev1.VolumeMount) bool {
			return mount.Name == compilationCache.PVCName && mount.MountPath == compilationCache.MountPath
		})
		return
	}
}

func preservedHasVolumeMount(preserved *corev1.PodTemplateSpec, name, mountPath string) bool {
	main, ok := findContainerByName(preserved.Spec.Containers, mainContainerName)
	if !ok {
		return false
	}
	for _, mount := range main.VolumeMounts {
		if mount.Name == name && mount.MountPath == mountPath {
			return true
		}
	}
	return false
}

func srcExtraPodSpec(src *DynamoComponentDeploymentSharedSpec) *ExtraPodSpec {
	if src == nil {
		return nil
	}
	return src.ExtraPodSpec
}

func restoreSharedHubOnlyPodTemplateMetadata(dst *metav1.ObjectMeta, preserved metav1.ObjectMeta) {
	labels := maps.Clone(dst.Labels)
	annotations := maps.Clone(dst.Annotations)
	*dst = *preserved.DeepCopy()
	dst.Labels = labels
	dst.Annotations = annotations
}

func restoreSharedHubOnlyFlatVolumeMountFields(dst, preserved *corev1.PodTemplateSpec, src *DynamoComponentDeploymentSharedSpec) {
	if src == nil || len(src.VolumeMounts) == 0 {
		return
	}
	preservedMain, ok := findContainerByName(preserved.Spec.Containers, mainContainerName)
	if !ok || len(preservedMain.VolumeMounts) == 0 {
		return
	}
	for i := range dst.Spec.Containers {
		if dst.Spec.Containers[i].Name != mainContainerName {
			continue
		}
		for j := range dst.Spec.Containers[i].VolumeMounts {
			mount := &dst.Spec.Containers[i].VolumeMounts[j]
			if !sourceFlatVolumeMountMatches(src.VolumeMounts, *mount) ||
				sourceExtraPodMainContainerVolumeMountMatches(src.ExtraPodSpec, *mount) {
				continue
			}
			if preservedMount, ok := findPreservedVolumeMount(preservedMain.VolumeMounts, *mount); ok {
				copyHubOnlyVolumeMountFields(mount, preservedMount)
			}
		}
		return
	}
}

func sourceFlatVolumeMountMatches(mounts []VolumeMount, mount corev1.VolumeMount) bool {
	for _, candidate := range mounts {
		if candidate.Name == mount.Name && candidate.MountPoint == mount.MountPath {
			return true
		}
	}
	return false
}

func sourceExtraPodMainContainerVolumeMountMatches(extraPodSpec *ExtraPodSpec, mount corev1.VolumeMount) bool {
	if extraPodSpec == nil || extraPodSpec.MainContainer == nil {
		return false
	}
	for _, candidate := range extraPodSpec.MainContainer.VolumeMounts {
		if candidate.Name == mount.Name && candidate.MountPath == mount.MountPath {
			return true
		}
	}
	return false
}

func findPreservedVolumeMount(mounts []corev1.VolumeMount, mount corev1.VolumeMount) (corev1.VolumeMount, bool) {
	for _, candidate := range mounts {
		if candidate.Name == mount.Name && candidate.MountPath == mount.MountPath {
			return *candidate.DeepCopy(), true
		}
	}
	var match *corev1.VolumeMount
	for i := range mounts {
		if mounts[i].Name != mount.Name {
			continue
		}
		if match != nil {
			return corev1.VolumeMount{}, false
		}
		match = mounts[i].DeepCopy()
	}
	if match == nil {
		return corev1.VolumeMount{}, false
	}
	return *match, true
}

func copyHubOnlyVolumeMountFields(dst *corev1.VolumeMount, preserved corev1.VolumeMount) {
	name := dst.Name
	mountPath := dst.MountPath
	*dst = *preserved.DeepCopy()
	dst.Name = name
	dst.MountPath = mountPath
}

func hasContainerNamed(containers []corev1.Container, name string) bool {
	for i := range containers {
		if containers[i].Name == name {
			return true
		}
	}
	return false
}

func experimentalIsHubOnlyShape(src *v1beta1.ExperimentalSpec) bool {
	return src != nil &&
		src.GPUMemoryService == nil &&
		src.Failover == nil &&
		src.Checkpoint == nil
}

func nilIfEmptyPodTemplate(podTemplate *corev1.PodTemplateSpec) *corev1.PodTemplateSpec {
	if podTemplate == nil || podTemplateIsZero(podTemplate) {
		return nil
	}
	return podTemplate
}

func podTemplateIsZero(podTemplate *corev1.PodTemplateSpec) bool {
	return podTemplate != nil && apiequality.Semantic.DeepEqual(*podTemplate, corev1.PodTemplateSpec{})
}

func resourceRequirementsEqual(a, b corev1.ResourceRequirements) bool {
	return apiequality.Semantic.DeepEqual(a, b)
}

func volumeMountOriginsMatchNative(origin []VolumeMount, native []corev1.VolumeMount) bool {
	if len(origin) != len(native) {
		return false
	}
	for i := range origin {
		if origin[i].Name != native[i].Name || origin[i].MountPoint != native[i].MountPath {
			return false
		}
	}
	return true
}

func restoreDedicatedFieldsFromMain(main *corev1.Container, dst *DynamoComponentDeploymentSharedSpec) {
	if main == nil {
		return
	}

	if resources := resourcesFromNative(main.Resources); resources != nil {
		dst.Resources = resources
		main.Resources = corev1.ResourceRequirements{}
	}

	if len(main.VolumeMounts) > 0 {
		restoreFlatVolumeMountsFromMain(main, dst)
	}

	if len(main.Env) > 0 {
		dst.Envs = slices.Clone(main.Env)
		main.Env = nil
	}
	if main.LivenessProbe != nil {
		dst.LivenessProbe = main.LivenessProbe.DeepCopy()
		main.LivenessProbe = nil
	}
	if main.ReadinessProbe != nil {
		dst.ReadinessProbe = main.ReadinessProbe.DeepCopy()
		main.ReadinessProbe = nil
	}
}

func restoreFlatVolumeMountsFromMain(main *corev1.Container, dst *DynamoComponentDeploymentSharedSpec) {
	if main == nil || len(main.VolumeMounts) == 0 {
		return
	}
	dst.VolumeMounts = appendMissingVolumeMounts(dst.VolumeMounts, volumeMountsFromNative(main.VolumeMounts))
	main.VolumeMounts = nil
}

// containerIsEmpty reports whether c has no user-visible fields set. Used by
// decomposePodTemplateFromHub to drop the "main" container synthesized by
// buildPodTemplateToHub when v1alpha1 had no main-container fields of its own.
// Name is expected to have been cleared by the caller.
func containerIsEmpty(c *corev1.Container) bool {
	if c == nil {
		return true
	}
	copy := *c
	copy.Name = ""
	normalizeContainerEmptySlices(&copy)
	return apiequality.Semantic.DeepEqual(copy, corev1.Container{})
}

func normalizeContainerEmptySlices(c *corev1.Container) {
	if len(c.Command) == 0 {
		c.Command = nil
	}
	if len(c.Args) == 0 {
		c.Args = nil
	}
	if len(c.Ports) == 0 {
		c.Ports = nil
	}
	if len(c.EnvFrom) == 0 {
		c.EnvFrom = nil
	}
	if len(c.Env) == 0 {
		c.Env = nil
	}
	if len(c.VolumeMounts) == 0 {
		c.VolumeMounts = nil
	}
	if len(c.VolumeDevices) == 0 {
		c.VolumeDevices = nil
	}
	if len(c.ResizePolicy) == 0 {
		c.ResizePolicy = nil
	}
	if len(c.RestartPolicyRules) == 0 {
		c.RestartPolicyRules = nil
	}
}

// appendFrontendSidecar ensures a container named
// defaultFrontendSidecarContainerName exists in the podTemplate's container
// list, synthesising one from the v1alpha1 FrontendSidecarSpec when absent.
// Callers use defaultFrontendSidecarContainerName directly for the resulting
// name reference.
func appendFrontendSidecar(podTpl *corev1.PodTemplateSpec, fs *FrontendSidecarSpec) {
	for _, ctr := range podTpl.Spec.Containers {
		if ctr.Name == defaultFrontendSidecarContainerName {
			return
		}
	}
	ctr := corev1.Container{
		Name:  defaultFrontendSidecarContainerName,
		Image: fs.Image,
		Args:  slices.Clone(fs.Args),
		Env:   slices.Clone(fs.Envs),
	}
	if fs.EnvFromSecret != nil && *fs.EnvFromSecret != "" {
		ctr.EnvFrom = append(ctr.EnvFrom, corev1.EnvFromSource{
			SecretRef: &corev1.SecretEnvSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: *fs.EnvFromSecret},
			},
		})
	}
	podTpl.Spec.Containers = append(podTpl.Spec.Containers, ctr)
}

func findContainerByName(containers []corev1.Container, name string) (corev1.Container, bool) {
	for _, ctr := range containers {
		if ctr.Name == name {
			return *ctr.DeepCopy(), true
		}
	}
	return corev1.Container{}, false
}

func frontendSidecarSpecFromContainer(ctr corev1.Container, origin *FrontendSidecarSpec) *FrontendSidecarSpec {
	out := &FrontendSidecarSpec{}
	if origin != nil {
		out = origin.DeepCopy()
	}
	out.Image = ctr.Image
	out.Args = slices.Clone(ctr.Args)
	out.Envs = slices.Clone(ctr.Env)
	if secretName, ok := frontendSidecarEnvFromSecret(ctr.EnvFrom); ok {
		out.EnvFromSecret = ptr.To(secretName)
	} else if origin != nil && origin.EnvFromSecret != nil && *origin.EnvFromSecret == "" {
		out.EnvFromSecret = ptr.To("")
	} else {
		out.EnvFromSecret = nil
	}
	return out
}

func frontendSidecarEnvFromSecret(envFrom []corev1.EnvFromSource) (string, bool) {
	if len(envFrom) != 1 || envFrom[0].Prefix != "" || envFrom[0].ConfigMapRef != nil || envFrom[0].SecretRef == nil {
		return "", false
	}
	return envFrom[0].SecretRef.Name, true
}

// podSpecIsZero reports whether a PodSpec has no fields set. Uses the
// apiserver's own apiequality.Semantic.DeepEqual against the zero value so
// that any newly-added PodSpec field (EphemeralContainers, DNSConfig,
// TopologySpreadConstraints, ResourceClaims, HostPID/IPC, SchedulingGates,
// etc.) is automatically covered without extending an allowlist.
//
// Semantic.DeepEqual is preferred over reflect.DeepEqual because it treats
// nil-vs-empty (e.g. NodeSelector: nil vs map[string]string{}) as
// equivalent, which is the exact same comparison the apiserver uses for
// generation-bump checks; this matches what we want here ("would the
// apiserver consider this PodSpec empty?").
func podSpecIsZero(p *corev1.PodSpec) bool {
	if p == nil {
		return true
	}
	return apiequality.Semantic.DeepEqual(*p, corev1.PodSpec{})
}

func extraPodSpecIsZero(eps *ExtraPodSpec) bool {
	if eps == nil {
		return true
	}
	return podSpecIsZero(eps.PodSpec) && containerIsEmpty(eps.MainContainer)
}

func extraPodSpecNeedsPreservation(eps *ExtraPodSpec) bool {
	return eps != nil && (extraPodSpecIsZero(eps) || extraPodSpecOnlyPreservesMainContainerName(eps))
}

func shouldRestorePreservedExtraPodSpec(dst, preserved *DynamoComponentDeploymentSharedSpec) bool {
	return dst != nil &&
		preserved != nil &&
		preserved.ExtraPodSpec != nil &&
		(extraPodSpecNeedsPreservation(preserved.ExtraPodSpec) ||
			alphaFrontendSidecarConflictNeedsPreservation(preserved.FrontendSidecar, preserved.ExtraPodSpec))
}

func alphaFrontendSidecarConflictNeedsPreservation(frontendSidecar *FrontendSidecarSpec, eps *ExtraPodSpec) bool {
	// Keep enough origin data for admission to preserve the v1alpha1 rule that
	// rejects frontendSidecar when extraPodSpec already declares the generated
	// sidecar container name.
	return frontendSidecar != nil && extraPodSpecHasContainer(eps, defaultFrontendSidecarContainerName)
}

func extraPodSpecHasContainer(eps *ExtraPodSpec, name string) bool {
	if eps == nil || eps.PodSpec == nil {
		return false
	}
	for _, container := range eps.PodSpec.Containers {
		if container.Name == name {
			return true
		}
	}
	return false
}

func extraPodSpecOnlyPreservesMainContainerName(eps *ExtraPodSpec) bool {
	if eps == nil || eps.MainContainer == nil || eps.MainContainer.Name == "" || !podSpecIsZero(eps.PodSpec) {
		return false
	}
	main := eps.MainContainer.DeepCopy()
	main.Name = ""
	return containerIsEmpty(main)
}

func restoreEnvFromSecretFromPreserved(dst, preserved *DynamoComponentDeploymentSharedSpec, mainContainerPresent bool) {
	if dst == nil || preserved == nil || dst.EnvFromSecret != nil || preserved.EnvFromSecret == nil {
		return
	}
	if *preserved.EnvFromSecret == "" {
		dst.EnvFromSecret = ptr.To("")
		return
	}
	if !mainContainerPresent {
		return
	}
	if dst.ExtraPodSpec == nil || dst.ExtraPodSpec.MainContainer == nil {
		return
	}
	if !envFromSecretMatches(dst.ExtraPodSpec.MainContainer.EnvFrom, *preserved.EnvFromSecret) {
		return
	}
	dst.EnvFromSecret = ptr.To(*preserved.EnvFromSecret)
	dst.ExtraPodSpec.MainContainer.EnvFrom = nil
}

func sharedHasMainContainer(src *v1beta1.DynamoComponentDeploymentSharedSpec) bool {
	return src != nil &&
		src.PodTemplate != nil &&
		hasContainerNamed(src.PodTemplate.Spec.Containers, mainContainerName)
}

func pruneEmptyExtraPodSpec(dst, restored *DynamoComponentDeploymentSharedSpec) {
	if dst == nil || dst.ExtraPodSpec == nil {
		return
	}
	if containerIsEmpty(dst.ExtraPodSpec.MainContainer) &&
		!extraPodSpecOnlyPreservesMainContainerName(restoredExtraPodSpec(restored)) {
		dst.ExtraPodSpec.MainContainer = nil
	}
	if extraPodSpecIsZero(dst.ExtraPodSpec) {
		if restored != nil && extraPodSpecNeedsPreservation(restored.ExtraPodSpec) {
			return
		}
		dst.ExtraPodSpec = nil
	}
}

func restoredExtraPodSpec(restored *DynamoComponentDeploymentSharedSpec) *ExtraPodSpec {
	if restored == nil {
		return nil
	}
	return restored.ExtraPodSpec
}

func restoreMainContainerFieldOrigins(dst, preserved *DynamoComponentDeploymentSharedSpec, mainContainerPresent bool) {
	if !mainContainerPresent ||
		dst == nil ||
		preserved == nil ||
		preserved.ExtraPodSpec == nil ||
		preserved.ExtraPodSpec.MainContainer == nil {
		return
	}
	preservedMain := preserved.ExtraPodSpec.MainContainer
	currentMain := semanticMainContainer(dst)
	preservedSemanticMain := semanticMainContainer(preserved)

	if apiequality.Semantic.DeepEqual(currentMain.Env, preservedSemanticMain.Env) &&
		(len(preserved.Envs) > 0 || len(preservedMain.Env) > 0) {
		dst.Envs = slices.Clone(preserved.Envs)
		ensureExtraPodSpecMainContainer(dst).Env = slices.Clone(preservedMain.Env)
	}
	if apiequality.Semantic.DeepEqual(currentMain.EnvFrom, preservedSemanticMain.EnvFrom) && preserved.EnvFromSecret != nil {
		dst.EnvFromSecret = ptr.To(*preserved.EnvFromSecret)
	}
	if resourceRequirementsEqual(currentMain.Resources, preservedSemanticMain.Resources) &&
		(preserved.Resources != nil || !resourceRequirementsEqual(preservedMain.Resources, corev1.ResourceRequirements{})) {
		dst.Resources = preserved.Resources.DeepCopy()
		ensureExtraPodSpecMainContainer(dst).Resources = *preservedMain.Resources.DeepCopy()
	}
	currentVolumeMounts := withoutCompilationCacheMounts(currentMain.VolumeMounts, dst.VolumeMounts)
	preservedVolumeMounts := withoutCompilationCacheMounts(preservedSemanticMain.VolumeMounts, preserved.VolumeMounts)
	if volumeMountOriginsMatchNative(volumeMountsFromNative(preservedVolumeMounts), currentVolumeMounts) &&
		(len(preserved.VolumeMounts) > 0 || len(preservedMain.VolumeMounts) > 0) {
		dst.VolumeMounts = restorePreservedVolumeMountOrigins(preserved.VolumeMounts, dst.VolumeMounts, preservedMain.VolumeMounts)
		ensureExtraPodSpecMainContainer(dst).VolumeMounts = cloneNativeVolumeMounts(preservedMain.VolumeMounts)
	}
	if apiequality.Semantic.DeepEqual(currentMain.LivenessProbe, preservedSemanticMain.LivenessProbe) &&
		(preserved.LivenessProbe != nil || preservedMain.LivenessProbe != nil) {
		dst.LivenessProbe = preserved.LivenessProbe.DeepCopy()
		ensureExtraPodSpecMainContainer(dst).LivenessProbe = preservedMain.LivenessProbe.DeepCopy()
	}
	if apiequality.Semantic.DeepEqual(currentMain.ReadinessProbe, preservedSemanticMain.ReadinessProbe) &&
		(preserved.ReadinessProbe != nil || preservedMain.ReadinessProbe != nil) {
		dst.ReadinessProbe = preserved.ReadinessProbe.DeepCopy()
		ensureExtraPodSpecMainContainer(dst).ReadinessProbe = preservedMain.ReadinessProbe.DeepCopy()
	}
}

func semanticMainContainer(src *DynamoComponentDeploymentSharedSpec) corev1.Container {
	if src == nil {
		return corev1.Container{Name: mainContainerName}
	}
	main := buildMainContainerFromDedicated(src)
	_ = mergeExtraPodSpecMainContainer(src, &main)
	main.Name = mainContainerName
	return main
}

func ensureExtraPodSpecMainContainer(dst *DynamoComponentDeploymentSharedSpec) *corev1.Container {
	if dst.ExtraPodSpec == nil {
		dst.ExtraPodSpec = &ExtraPodSpec{}
	}
	if dst.ExtraPodSpec.MainContainer == nil {
		dst.ExtraPodSpec.MainContainer = &corev1.Container{}
	}
	return dst.ExtraPodSpec.MainContainer
}

func cloneNativeVolumeMounts(in []corev1.VolumeMount) []corev1.VolumeMount {
	if len(in) == 0 {
		return nil
	}
	out := make([]corev1.VolumeMount, 0, len(in))
	for i := range in {
		out = append(out, *in[i].DeepCopy())
	}
	return out
}

func cloneNativeEnvFromSources(in []corev1.EnvFromSource) []corev1.EnvFromSource {
	if len(in) == 0 {
		return nil
	}
	out := make([]corev1.EnvFromSource, 0, len(in))
	for i := range in {
		out = append(out, *in[i].DeepCopy())
	}
	return out
}

func restorePreservedVolumeMountOrigins(preserved, current []VolumeMount, preservedMain []corev1.VolumeMount) []VolumeMount {
	out := make([]VolumeMount, 0, len(preserved)+len(current))
	for _, mount := range preserved {
		if !mount.UseAsCompilationCache {
			out = append(out, mount)
		}
	}
	for _, mount := range current {
		if !mount.UseAsCompilationCache {
			if nativeVolumeMountHasNamePath(preservedMain, mount.Name, mount.MountPoint) ||
				flatVolumeMountHasNamePath(out, mount.Name, mount.MountPoint) {
				continue
			}
			out = append(out, mount)
			continue
		}
		replaced := false
		for i := range out {
			if out[i].Name == mount.Name && out[i].MountPoint == mount.MountPoint {
				out[i].UseAsCompilationCache = true
				replaced = true
				break
			}
		}
		if !replaced {
			out = append(out, mount)
		}
	}
	return out
}

func nativeVolumeMountHasNamePath(mounts []corev1.VolumeMount, name, mountPath string) bool {
	for _, mount := range mounts {
		if mount.Name == name && mount.MountPath == mountPath {
			return true
		}
	}
	return false
}

func flatVolumeMountHasNamePath(mounts []VolumeMount, name, mountPath string) bool {
	for _, mount := range mounts {
		if mount.Name == name && mount.MountPoint == mountPath {
			return true
		}
	}
	return false
}

func withoutCompilationCacheMounts(mounts []corev1.VolumeMount, flat []VolumeMount) []corev1.VolumeMount {
	if len(mounts) == 0 || len(flat) == 0 {
		return mounts
	}
	out := make([]corev1.VolumeMount, 0, len(mounts))
	for _, mount := range mounts {
		if flatVolumeMountHasCompilationCache(flat, mount.Name, mount.MountPath) {
			continue
		}
		out = append(out, mount)
	}
	return out
}

func flatVolumeMountHasCompilationCache(mounts []VolumeMount, name, mountPath string) bool {
	for _, mount := range mounts {
		if mount.UseAsCompilationCache && mount.Name == name && mount.MountPoint == mountPath {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Resources <-> corev1.ResourceRequirements
// ---------------------------------------------------------------------------

// resourcesToNative converts v1alpha1.Resources into corev1.ResourceRequirements.
//   - "cpu", "memory" are recognised by name.
//   - A GPU value with GPUType=""  maps to "nvidia.com/gpu" (the v1alpha1
//     default); with GPUType="X" maps to key X.
//   - Custom keys are copied through.
func resourcesToNative(r *Resources) corev1.ResourceRequirements {
	out := corev1.ResourceRequirements{Claims: slices.Clone(r.Claims)}
	if r.Requests != nil {
		out.Requests = itemToResourceList(r.Requests)
	}
	if r.Limits != nil {
		out.Limits = itemToResourceList(r.Limits)
	}
	return out
}

func resourcesFromNative(r corev1.ResourceRequirements) *Resources {
	out := &Resources{
		Requests: resourceItemFromList(r.Requests),
		Limits:   resourceItemFromList(r.Limits),
		Claims:   slices.Clone(r.Claims),
	}
	if out.Requests == nil && out.Limits == nil && len(out.Claims) == 0 {
		return nil
	}
	return out
}

func itemToResourceList(item *ResourceItem) corev1.ResourceList {
	if item == nil {
		return nil
	}
	out := corev1.ResourceList{}
	if item.CPU != "" {
		if q, err := resource.ParseQuantity(item.CPU); err == nil {
			out[corev1.ResourceCPU] = q
		}
	}
	if item.Memory != "" {
		if q, err := resource.ParseQuantity(item.Memory); err == nil {
			out[corev1.ResourceMemory] = q
		}
	}
	if item.GPU != "" {
		key := item.GPUType
		if key == "" {
			key = string(defaultGPUResourceName)
		}
		if q, err := resource.ParseQuantity(item.GPU); err == nil {
			out[corev1.ResourceName(key)] = q
		}
	}
	for k, v := range item.Custom {
		if q, err := resource.ParseQuantity(v); err == nil {
			out[corev1.ResourceName(k)] = q
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func resourceItemFromList(list corev1.ResourceList) *ResourceItem {
	if len(list) == 0 {
		return nil
	}
	out := &ResourceItem{}
	for name, q := range list {
		value := q.String()
		switch name {
		case corev1.ResourceCPU:
			out.CPU = value
		case corev1.ResourceMemory:
			out.Memory = value
		case defaultGPUResourceName:
			out.GPU = value
		default:
			if out.Custom == nil {
				out.Custom = map[string]string{}
			}
			out.Custom[string(name)] = value
		}
	}
	return out
}

func volumeMountsFromNative(mounts []corev1.VolumeMount) []VolumeMount {
	if len(mounts) == 0 {
		return nil
	}
	out := make([]VolumeMount, 0, len(mounts))
	for _, mount := range mounts {
		out = append(out, VolumeMount{
			Name:       mount.Name,
			MountPoint: mount.MountPath,
		})
	}
	return out
}

func volumeMountsRoundTripThroughHub(mounts []VolumeMount) bool {
	if len(mounts) == 0 {
		return true
	}
	compilationCacheMounts := 0
	for _, mount := range mounts {
		if mount.UseAsCompilationCache {
			compilationCacheMounts++
		}
	}
	return compilationCacheMounts <= 1
}

func hasPodTemplateVolumeMounts(mounts []VolumeMount) bool {
	for _, mount := range mounts {
		if !mount.UseAsCompilationCache {
			return true
		}
	}
	return false
}

func appendMissingVolumeMounts(dst []VolumeMount, mounts []VolumeMount) []VolumeMount {
	for _, mount := range mounts {
		exists := false
		for _, existing := range dst {
			if existing.Name == mount.Name && existing.MountPoint == mount.MountPoint {
				exists = true
				break
			}
		}
		if !exists {
			dst = append(dst, mount)
		}
	}
	return dst
}

func envFromSecretMatches(envFrom []corev1.EnvFromSource, name string) bool {
	return apiequality.Semantic.DeepEqual(envFrom, []corev1.EnvFromSource{{
		SecretRef: &corev1.SecretEnvSource{
			LocalObjectReference: corev1.LocalObjectReference{Name: name},
		},
	}})
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

// mergeEnvs replicates internal/dynamo.MergeEnvs: concatenate `common` and
// `specific`, de-duplicated by Name with `specific` winning on collision.
// Duplicated here to avoid an api -> internal cycle.
func mergeEnvs(common, specific []corev1.EnvVar) []corev1.EnvVar {
	out := make([]corev1.EnvVar, 0, len(common)+len(specific))
	seen := map[string]int{}
	for _, e := range common {
		seen[e.Name] = len(out)
		out = append(out, e)
	}
	for _, e := range specific {
		if idx, ok := seen[e.Name]; ok {
			out[idx] = e
			continue
		}
		seen[e.Name] = len(out)
		out = append(out, e)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// Convert_v1alpha1_DynamoComponentDeploymentSharedSpec_To_v1beta1_DynamoComponentDeploymentSharedSpec
// converts common fields with conversion-gen and projects the v1alpha1 flat
// pod fields into the v1beta1 podTemplate representation.
func Convert_v1alpha1_DynamoComponentDeploymentSharedSpec_To_v1beta1_DynamoComponentDeploymentSharedSpec(in *DynamoComponentDeploymentSharedSpec, out *v1beta1.DynamoComponentDeploymentSharedSpec, s apiconversion.Scope) error {
	return convertSharedSpecToHub(in, out, false, s)
}

// Convert_v1beta1_DynamoComponentDeploymentSharedSpec_To_v1alpha1_DynamoComponentDeploymentSharedSpec
// converts common fields with conversion-gen and decomposes podTemplate into
// the v1alpha1 flat pod fields.
func Convert_v1beta1_DynamoComponentDeploymentSharedSpec_To_v1alpha1_DynamoComponentDeploymentSharedSpec(in *v1beta1.DynamoComponentDeploymentSharedSpec, out *DynamoComponentDeploymentSharedSpec, s apiconversion.Scope) error {
	return convertSharedSpecToAlpha(in, out, nil, s)
}

// convertSharedSpecToHub handles fields whose representation changed:
//   - legacy worker subcomponent values become first-class component types;
//   - ServiceName becomes ComponentName;
//   - SharedMemory is flattened to SharedMemorySize;
//   - compilation-cache mounts become CompilationCache;
//   - ScalingAdapter.Enabled becomes presence semantics;
//   - GMS, failover, and checkpoint move under Experimental;
//   - resources, env, probes, ExtraPodSpec, metadata, and FrontendSidecar are
//     composed into PodTemplate.
func convertSharedSpecToHub(src *DynamoComponentDeploymentSharedSpec, dst *v1beta1.DynamoComponentDeploymentSharedSpec, podTemplateOrigin bool, s apiconversion.Scope) error {
	if err := autoConvert_v1alpha1_DynamoComponentDeploymentSharedSpec_To_v1beta1_DynamoComponentDeploymentSharedSpec(src, dst, s); err != nil {
		return err
	}

	dst.ComponentType = sharedComponentTypeToHub(src)
	dst.ComponentName = src.ServiceName

	if src.SharedMemory != nil && (src.SharedMemory.Disabled || !src.SharedMemory.Size.IsZero()) {
		dst.SharedMemorySize = &resource.Quantity{}
		ConvertFromSharedMemorySpec(src.SharedMemory, dst.SharedMemorySize)
	}
	for _, vm := range src.VolumeMounts {
		if vm.UseAsCompilationCache {
			dst.CompilationCache = &v1beta1.CompilationCacheConfig{
				PVCName:   vm.Name,
				MountPath: vm.MountPoint,
			}
			break
		}
	}
	if src.ScalingAdapter != nil && src.ScalingAdapter.Enabled {
		dst.ScalingAdapter = &v1beta1.ScalingAdapter{}
		if err := Convert_v1alpha1_ScalingAdapter_To_v1beta1_ScalingAdapter(src.ScalingAdapter, dst.ScalingAdapter, s); err != nil {
			return err
		}
	} else {
		dst.ScalingAdapter = nil
	}
	if err := convertExperimentalToHub(src, dst, s); err != nil {
		return err
	}
	return buildPodTemplateToHub(src, dst, podTemplateOrigin)
}

// convertSharedSpecToAlpha reverses the cross-field projections above. The
// optional restored value carries field-origin information that cannot be
// inferred by decomposing a podTemplate alone.
func convertSharedSpecToAlpha(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst *DynamoComponentDeploymentSharedSpec, restored *DynamoComponentDeploymentSharedSpec, s apiconversion.Scope) error {
	if err := autoConvert_v1beta1_DynamoComponentDeploymentSharedSpec_To_v1alpha1_DynamoComponentDeploymentSharedSpec(src, dst, s); err != nil {
		return err
	}

	dst.ComponentType, dst.SubComponentType = sharedComponentTypeFromHub(src.ComponentType)

	dst.ServiceName = src.ComponentName
	if src.SharedMemorySize != nil {
		dst.SharedMemory = &SharedMemorySpec{}
		ConvertToSharedMemorySpec(src.SharedMemorySize, dst.SharedMemory)
	}
	convertVolumeMountsFromHub(src, dst)
	if err := convertExperimentalFromHub(src, dst, s); err != nil {
		return err
	}
	return decomposePodTemplateFromHub(src, dst, restored)
}

// Convert_v1alpha1_ServiceCheckpointConfig_To_v1beta1_ComponentCheckpointConfig
// converts the renamed checkpoint config and its nested identity and job.
func Convert_v1alpha1_ServiceCheckpointConfig_To_v1beta1_ComponentCheckpointConfig(in *ServiceCheckpointConfig, out *v1beta1.ComponentCheckpointConfig, s apiconversion.Scope) error {
	*out = v1beta1.ComponentCheckpointConfig{
		Enabled:             in.Enabled,
		Mode:                v1beta1.CheckpointMode(in.Mode),
		StartupPolicy:       v1beta1.CheckpointStartupPolicy(in.StartupPolicy),
		DeletionPolicy:      v1beta1.CheckpointDeletionPolicy(in.DeletionPolicy),
		TargetContainerName: in.TargetContainerName,
	}
	if in.CheckpointRef != nil {
		out.CheckpointRef = in.CheckpointRef
	}
	if in.Identity != nil {
		out.Identity = &v1beta1.DynamoCheckpointIdentity{}
		if err := Convert_v1alpha1_DynamoCheckpointIdentity_To_v1beta1_DynamoCheckpointIdentity(in.Identity, out.Identity, s); err != nil {
			return err
		}
	}
	if in.Job != nil {
		out.Job = &v1beta1.ComponentCheckpointJobConfig{
			GMSClientContainers: in.Job.GMSClientContainers,
			PodTemplate:         in.Job.PodTemplate,
		}
	}
	return nil
}

// Convert_v1beta1_ComponentCheckpointConfig_To_v1alpha1_ServiceCheckpointConfig
// converts the renamed checkpoint config and its nested identity and job.
func Convert_v1beta1_ComponentCheckpointConfig_To_v1alpha1_ServiceCheckpointConfig(in *v1beta1.ComponentCheckpointConfig, out *ServiceCheckpointConfig, s apiconversion.Scope) error {
	*out = ServiceCheckpointConfig{
		Enabled:             in.Enabled,
		Mode:                CheckpointMode(in.Mode),
		StartupPolicy:       CheckpointStartupPolicy(in.StartupPolicy),
		DeletionPolicy:      CheckpointDeletionPolicy(in.DeletionPolicy),
		TargetContainerName: in.TargetContainerName,
	}
	if in.CheckpointRef != nil {
		out.CheckpointRef = in.CheckpointRef
	}
	if in.Identity != nil {
		out.Identity = &DynamoCheckpointIdentity{}
		if err := Convert_v1beta1_DynamoCheckpointIdentity_To_v1alpha1_DynamoCheckpointIdentity(in.Identity, out.Identity, s); err != nil {
			return err
		}
	}
	if in.Job != nil {
		out.Job = &ServiceCheckpointJobConfig{
			GMSClientContainers: in.Job.GMSClientContainers,
			PodTemplate:         in.Job.PodTemplate,
		}
	}
	return nil
}

// Convert_v1alpha1_FailoverSpec_To_v1beta1_FailoverSpec converts an enabled
// failover config. Disabled configs are represented by absence and skipped by
// the shared-spec caller.
func Convert_v1alpha1_FailoverSpec_To_v1beta1_FailoverSpec(in *FailoverSpec, out *v1beta1.FailoverSpec, s apiconversion.Scope) error {
	if err := autoConvert_v1alpha1_FailoverSpec_To_v1beta1_FailoverSpec(in, out, s); err != nil {
		return err
	}
	out.Mode = gmsModeToV1beta1(in.Mode)
	return nil
}

// Convert_v1beta1_FailoverSpec_To_v1alpha1_FailoverSpec restores the explicit
// Enabled flag implied by presence in v1beta1.
func Convert_v1beta1_FailoverSpec_To_v1alpha1_FailoverSpec(in *v1beta1.FailoverSpec, out *FailoverSpec, s apiconversion.Scope) error {
	if err := autoConvert_v1beta1_FailoverSpec_To_v1alpha1_FailoverSpec(in, out, s); err != nil {
		return err
	}
	out.Enabled = true
	out.Mode = gmsModeFromV1beta1(in.Mode)
	return nil
}

// Convert_v1alpha1_GPUMemoryServiceSpec_To_v1beta1_GPUMemoryServiceSpec
// converts an enabled GMS config. Disabled configs are represented by absence
// and skipped by the shared-spec caller.
func Convert_v1alpha1_GPUMemoryServiceSpec_To_v1beta1_GPUMemoryServiceSpec(in *GPUMemoryServiceSpec, out *v1beta1.GPUMemoryServiceSpec, s apiconversion.Scope) error {
	if err := autoConvert_v1alpha1_GPUMemoryServiceSpec_To_v1beta1_GPUMemoryServiceSpec(in, out, s); err != nil {
		return err
	}
	out.Mode = gmsModeToV1beta1(in.Mode)
	return nil
}

// Convert_v1beta1_GPUMemoryServiceSpec_To_v1alpha1_GPUMemoryServiceSpec
// restores the explicit Enabled flag implied by presence in v1beta1.
func Convert_v1beta1_GPUMemoryServiceSpec_To_v1alpha1_GPUMemoryServiceSpec(in *v1beta1.GPUMemoryServiceSpec, out *GPUMemoryServiceSpec, s apiconversion.Scope) error {
	if err := autoConvert_v1beta1_GPUMemoryServiceSpec_To_v1alpha1_GPUMemoryServiceSpec(in, out, s); err != nil {
		return err
	}
	out.Enabled = true
	out.Mode = gmsModeFromV1beta1(in.Mode)
	return nil
}

// Convert_v1alpha1_ScalingAdapter_To_v1beta1_ScalingAdapter converts the
// v1alpha1 enabled marker to the v1beta1 presence-only marker.
func Convert_v1alpha1_ScalingAdapter_To_v1beta1_ScalingAdapter(in *ScalingAdapter, out *v1beta1.ScalingAdapter, s apiconversion.Scope) error {
	// v1beta1 represents enabled=true by the presence of ScalingAdapter and has
	// no field for Enabled. This wrapper explicitly accepts that lossy mapping;
	// conversion-gen cannot generate the public Convert function for it.
	return autoConvert_v1alpha1_ScalingAdapter_To_v1beta1_ScalingAdapter(in, out, s)
}

// Convert_v1beta1_ScalingAdapter_To_v1alpha1_ScalingAdapter restores the
// Enabled flag implied by presence in v1beta1.
func Convert_v1beta1_ScalingAdapter_To_v1alpha1_ScalingAdapter(in *v1beta1.ScalingAdapter, out *ScalingAdapter, s apiconversion.Scope) error {
	if err := autoConvert_v1beta1_ScalingAdapter_To_v1alpha1_ScalingAdapter(in, out, s); err != nil {
		return err
	}
	out.Enabled = true
	return nil
}
