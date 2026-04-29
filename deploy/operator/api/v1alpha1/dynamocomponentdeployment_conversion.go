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

// Conversion between v1alpha1 and v1beta1 DynamoComponentDeployment.
// See dynamographdeployment_conversion.go for the design rationale.

package v1alpha1

import (
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/conversion"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	annDCDHubSpec     = "nvidia.com/dcd-hub-spec"
	annDCDSpokeSpec   = "nvidia.com/dcd-spoke-spec"
	annDCDSpokeStatus = "nvidia.com/dcd-spoke-status"
	annDCDSpokeHub    = "nvidia.com/dcd-spoke-hub"
	annDCDHubOrigin   = "nvidia.com/dcd-hub-origin"
)

// ConvertTo converts this DynamoComponentDeployment (v1alpha1) into the hub
// version (v1beta1).
func (src *DynamoComponentDeployment) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoComponentDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoComponentDeployment but got %T", dstRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()
	var restored *v1beta1.DynamoComponentDeployment

	if dst.ObjectMeta.Annotations[annDCDHubOrigin] == annotationTrue {
		restored = &v1beta1.DynamoComponentDeployment{}
	}
	if raw, ok := dst.ObjectMeta.Annotations[annDCDHubSpec]; ok && raw != "" {
		var spec v1beta1.DynamoComponentDeploymentSpec
		if err := json.Unmarshal([]byte(raw), &spec); err != nil {
			return fmt.Errorf("restore DCD hub spec: %w", err)
		}
		if restored == nil {
			restored = &v1beta1.DynamoComponentDeployment{}
		}
		restored.Spec = spec
		delAnnFromObj(&dst.ObjectMeta, annDCDHubSpec)
	}
	delAnnFromObj(&dst.ObjectMeta, annDCDHubOrigin)

	save, err := convert_v1alpha1_DynamoComponentDeployment_To_v1beta1_DynamoComponentDeployment(src, dst, restored)
	if err != nil {
		return err
	}
	if restored != nil {
		scrubDCDAnnotations(&dst.ObjectMeta)
	}
	if save != nil {
		data, err := json.Marshal(save.Spec)
		if err != nil {
			return fmt.Errorf("preserve DCD spoke spec: %w", err)
		}
		setAnnOnObj(&dst.ObjectMeta, annDCDSpokeSpec, string(data))

		data, err = json.Marshal(save.Status)
		if err != nil {
			return fmt.Errorf("preserve DCD spoke status: %w", err)
		}
		setAnnOnObj(&dst.ObjectMeta, annDCDSpokeStatus, string(data))

		if err := preserveDCDSpokeHub(dst); err != nil {
			return err
		}
	}
	return nil
}

func convert_v1alpha1_DynamoComponentDeployment_To_v1beta1_DynamoComponentDeployment(src *DynamoComponentDeployment, dst *v1beta1.DynamoComponentDeployment, restored *v1beta1.DynamoComponentDeployment) (*DynamoComponentDeployment, error) {
	save := &DynamoComponentDeployment{}
	if restored == nil {
		if err := convert_v1alpha1_DynamoComponentDeploymentSpec_To_v1beta1_DynamoComponentDeploymentSpec(&src.Spec, &dst.Spec, nil, &save.Spec, newDCDCarrier(&dst.ObjectMeta)); err != nil {
			return nil, err
		}
	} else if err := convert_v1alpha1_DynamoComponentDeploymentSpec_To_v1beta1_DynamoComponentDeploymentSpec(&src.Spec, &dst.Spec, &restored.Spec, &save.Spec, newDCDCarrier(&dst.ObjectMeta)); err != nil {
		return nil, err
	}

	// v1beta1 requires DCD.spec.name (it is the +listMapKey on
	// DGD.spec.components and is enforced as Required by the schema). When a
	// v1alpha1 caller omits ServiceName, fall back to ObjectMeta.Name so the
	// converted object is schema-valid. Hub-origin conversions restore the
	// original hub shape instead.
	if dst.Spec.ComponentName == "" && restored == nil {
		dst.Spec.ComponentName = dst.ObjectMeta.Name
	}
	if err := convert_v1alpha1_DynamoComponentDeploymentStatus_To_v1beta1_DynamoComponentDeploymentStatus(&src.Status, &dst.Status, nil, &save.Status); err != nil {
		return nil, err
	}

	// Save source-only fields that dst cannot represent.
	hasSourceOnlyState := restored == nil ||
		hasSharedAlphaOnlyFields(&src.Spec.DynamoComponentDeploymentSharedSpec) ||
		len(src.Status.PodSelector) > 0 ||
		(src.Status.Service != nil &&
			serviceStatusComponentNameNeedsPreservation(src.Status.Service))
	if !hasSourceOnlyState {
		return nil, nil
	}
	return save, nil
}

func convert_v1alpha1_DynamoComponentDeploymentSpec_To_v1beta1_DynamoComponentDeploymentSpec(src *DynamoComponentDeploymentSpec, dst *v1beta1.DynamoComponentDeploymentSpec, restored *v1beta1.DynamoComponentDeploymentSpec, save *DynamoComponentDeploymentSpec, carrier *annCarrier) error {
	// Convert representable fields from src to dst.
	dst.BackendFramework = src.BackendFramework
	if err := convertSharedSpecTo(&src.DynamoComponentDeploymentSharedSpec, &dst.DynamoComponentDeploymentSharedSpec, carrier); err != nil {
		return err
	}

	// Restore target-only fields that src cannot represent.
	if restored != nil {
		if dst.FrontendSidecar == nil && restored.FrontendSidecar != nil {
			dst.FrontendSidecar = restored.FrontendSidecar
		}
		if restored.PodTemplate != nil {
			shared := &src.DynamoComponentDeploymentSharedSpec
			keepMainContainer := shared.Resources != nil ||
				len(shared.Envs) > 0 ||
				shared.EnvFromSecret != nil ||
				hasPodTemplateVolumeMounts(shared.VolumeMounts) ||
				shared.LivenessProbe != nil ||
				shared.ReadinessProbe != nil ||
				(shared.ExtraPodSpec != nil &&
					shared.ExtraPodSpec.MainContainer != nil &&
					!containerIsEmpty(shared.ExtraPodSpec.MainContainer))
			mergeDCDHubOnlyPodTemplateFields(
				&dst.PodTemplate,
				restored.PodTemplate,
				dst.FrontendSidecar,
				keepMainContainer,
				src.FrontendSidecar != nil,
			)
		}
		if dst.Experimental == nil &&
			restored.Experimental != nil &&
			restored.Experimental.GPUMemoryService == nil &&
			restored.Experimental.Failover == nil &&
			restored.Experimental.Checkpoint == nil {
			dst.Experimental = restored.Experimental
		}
	}

	// Save source-only fields that dst cannot represent.
	*save = *src
	return nil
}

func mergeDCDHubOnlyPodTemplateFields(dst **corev1.PodTemplateSpec, restored *corev1.PodTemplateSpec, frontendSidecar *string, keepMainContainer, keepFrontendSidecar bool) {
	if restored == nil {
		return
	}
	if *dst == nil {
		*dst = &corev1.PodTemplateSpec{}
	}

	// v1alpha1 can express labels and annotations, but not the rest of
	// PodTemplate metadata or the mere presence of an empty podTemplate.
	labels := maps.Clone((*dst).Labels)
	annotations := maps.Clone((*dst).Annotations)
	(*dst).ObjectMeta = restored.ObjectMeta
	(*dst).Labels = labels
	(*dst).Annotations = annotations

	// v1alpha1 separates generated containers from ExtraPodSpec, so their
	// exact positions in the v1beta1 container list are hub-only state.
	mergeDCDHubOnlyContainerPositions(&(*dst).Spec.Containers, restored.Spec.Containers, frontendSidecar, keepMainContainer, keepFrontendSidecar)
}

func mergeDCDHubOnlyContainerPositions(dst *[]corev1.Container, restored []corev1.Container, frontendSidecar *string, keepMainContainer, keepFrontendSidecar bool) {
	if len(*dst) == 0 {
		return
	}

	generated := map[string]struct{}{mainContainerName: {}}
	keepGenerated := map[string]bool{mainContainerName: keepMainContainer}
	if frontendSidecar != nil && *frontendSidecar != "" {
		generated[*frontendSidecar] = struct{}{}
		keepGenerated[*frontendSidecar] = keepFrontendSidecar
	}
	restoredGenerated := map[string]struct{}{}
	for _, container := range restored {
		if _, ok := generated[container.Name]; ok {
			restoredGenerated[container.Name] = struct{}{}
		}
	}

	nonGeneratedNames := map[string]struct{}{}
	generatedByName := map[string]corev1.Container{}
	generatedOrder := []string{}
	nonGenerated := make([]corev1.Container, 0, len(*dst))
	for _, container := range *dst {
		if _, ok := generated[container.Name]; ok {
			generatedByName[container.Name] = container
			generatedOrder = append(generatedOrder, container.Name)
			continue
		}
		nonGeneratedNames[container.Name] = struct{}{}
		nonGenerated = append(nonGenerated, container)
	}
	if len(generatedByName) == 0 {
		return
	}

	result := make([]corev1.Container, 0, len(*dst))
	placed := map[string]struct{}{}
	nonGeneratedSeen := 0
	nonGeneratedWritten := 0
	for _, container := range restored {
		if _, ok := nonGeneratedNames[container.Name]; ok {
			nonGeneratedSeen++
			continue
		}
		generatedContainer, ok := generatedByName[container.Name]
		if !ok {
			continue
		}
		for nonGeneratedWritten < nonGeneratedSeen && nonGeneratedWritten < len(nonGenerated) {
			result = append(result, nonGenerated[nonGeneratedWritten])
			nonGeneratedWritten++
		}
		result = append(result, generatedContainer)
		placed[container.Name] = struct{}{}
	}
	for nonGeneratedWritten < len(nonGenerated) {
		result = append(result, nonGenerated[nonGeneratedWritten])
		nonGeneratedWritten++
	}
	for _, name := range generatedOrder {
		if _, ok := placed[name]; ok {
			continue
		}
		if _, ok := restoredGenerated[name]; !ok && !keepGenerated[name] {
			continue
		}
		result = append(result, generatedByName[name])
	}
	if len(result) == 0 {
		result = nil
	}
	*dst = result
}

type preservedDCDHubSnapshot struct {
	Spec   v1beta1.DynamoComponentDeploymentSpec   `json:"spec"`
	Status v1beta1.DynamoComponentDeploymentStatus `json:"status"`
}

func preserveDCDSpokeHub(dst *v1beta1.DynamoComponentDeployment) error {
	data, err := json.Marshal(preservedDCDHubSnapshot{
		Spec:   dst.Spec,
		Status: dst.Status,
	})
	if err != nil {
		return fmt.Errorf("preserve DCD spoke hub snapshot: %w", err)
	}
	setAnnOnObj(&dst.ObjectMeta, annDCDSpokeHub, string(data))
	return nil
}

func dcdSpokeHubUnmodified(src *v1beta1.DynamoComponentDeployment) bool {
	raw, ok := src.ObjectMeta.Annotations[annDCDSpokeHub]
	if !ok || raw == "" {
		return false
	}
	current, err := json.Marshal(preservedDCDHubSnapshot{
		Spec:   src.Spec,
		Status: src.Status,
	})
	if err != nil {
		return false
	}
	return string(current) == raw
}

// ConvertFrom converts from the hub (v1beta1) DynamoComponentDeployment into
// this v1alpha1 instance.
func (dst *DynamoComponentDeployment) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoComponentDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoComponentDeployment but got %T", srcRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()
	var restored *DynamoComponentDeployment
	if raw, ok := dst.ObjectMeta.Annotations[annDCDSpokeSpec]; ok && raw != "" {
		var spec DynamoComponentDeploymentSpec
		if err := json.Unmarshal([]byte(raw), &spec); err != nil {
			return fmt.Errorf("restore DCD spoke spec: %w", err)
		}
		restored = &DynamoComponentDeployment{Spec: spec}
	}
	if rawStatus, ok := dst.ObjectMeta.Annotations[annDCDSpokeStatus]; ok && rawStatus != "" {
		var status DynamoComponentDeploymentStatus
		if err := json.Unmarshal([]byte(rawStatus), &status); err != nil {
			return fmt.Errorf("restore DCD spoke status: %w", err)
		}
		if restored == nil {
			restored = &DynamoComponentDeployment{}
		}
		restored.Status = status
	}

	// This hint is not a conversion fast path. It only distinguishes a
	// conversion-generated podTemplate from a live hub edit when deciding
	// whether hub-only podTemplate state must be saved.
	spokeHubUnmodified := dcdSpokeHubUnmodified(src)
	generatedPodTemplate := spokeHubUnmodified && src.ObjectMeta.Annotations[annDCDPrefix+suffixPodTemplateOrig] == "generated"

	save := &v1beta1.DynamoComponentDeployment{}
	if err := convert_v1beta1_DynamoComponentDeployment_To_v1alpha1_DynamoComponentDeployment(src, dst, restored, save); err != nil {
		return err
	}

	scrubDCDAnnotations(&dst.ObjectMeta)
	needsHubSpecPreservation := !generatedPodTemplate &&
		(save.Spec.FrontendSidecar != nil ||
			save.Spec.PodTemplate != nil ||
			save.Spec.Experimental != nil &&
				save.Spec.Experimental.GPUMemoryService == nil &&
				save.Spec.Experimental.Failover == nil &&
				save.Spec.Experimental.Checkpoint == nil)
	if needsHubSpecPreservation {
		data, err := json.Marshal(save.Spec)
		if err != nil {
			return fmt.Errorf("preserve DCD hub spec: %w", err)
		}
		setAnnOnObj(&dst.ObjectMeta, annDCDHubSpec, string(data))
	} else {
		hasInternalAnnotations := false
		for key := range src.ObjectMeta.Annotations {
			if key == annDCDHubSpec ||
				key == annDCDSpokeSpec ||
				key == annDCDSpokeStatus ||
				key == annDCDSpokeHub ||
				strings.HasPrefix(key, annDCDPrefix) {
				hasInternalAnnotations = true
				break
			}
		}
		if !hasInternalAnnotations {
			setAnnOnObj(&dst.ObjectMeta, annDCDHubOrigin, annotationTrue)
		}
	}
	return nil
}

func convert_v1beta1_DynamoComponentDeployment_To_v1alpha1_DynamoComponentDeployment(src *v1beta1.DynamoComponentDeployment, dst *DynamoComponentDeployment, restored *DynamoComponentDeployment, save *v1beta1.DynamoComponentDeployment) error {
	if restored == nil {
		if err := convert_v1beta1_DynamoComponentDeploymentSpec_To_v1alpha1_DynamoComponentDeploymentSpec(&src.Spec, &dst.Spec, nil, &save.Spec, newDCDCarrier(&dst.ObjectMeta)); err != nil {
			return err
		}
		if err := convert_v1beta1_DynamoComponentDeploymentStatus_To_v1alpha1_DynamoComponentDeploymentStatus(&src.Status, &dst.Status, nil, &save.Status); err != nil {
			return err
		}
		return nil
	}
	if err := convert_v1beta1_DynamoComponentDeploymentSpec_To_v1alpha1_DynamoComponentDeploymentSpec(&src.Spec, &dst.Spec, &restored.Spec, &save.Spec, newDCDCarrier(&dst.ObjectMeta)); err != nil {
		return err
	}

	// Restore the source-version shape when v1beta1 only carries a generated
	// projection of that shape.
	projectedComponentName := restored.Spec.ServiceName
	if projectedComponentName == "" {
		projectedComponentName = src.ObjectMeta.Name
	}
	if src.Spec.ComponentName == projectedComponentName {
		dst.Spec.ServiceName = restored.Spec.ServiceName
	}

	if err := convert_v1beta1_DynamoComponentDeploymentStatus_To_v1alpha1_DynamoComponentDeploymentStatus(&src.Status, &dst.Status, &restored.Status, &save.Status); err != nil {
		return err
	}
	return nil
}

func convert_v1beta1_DynamoComponentDeploymentSpec_To_v1alpha1_DynamoComponentDeploymentSpec(src *v1beta1.DynamoComponentDeploymentSpec, dst *DynamoComponentDeploymentSpec, restored *DynamoComponentDeploymentSpec, save *v1beta1.DynamoComponentDeploymentSpec, carrier *annCarrier) error {
	// Convert representable fields from src to dst.
	dst.BackendFramework = src.BackendFramework
	if err := convertSharedSpecFrom(&src.DynamoComponentDeploymentSharedSpec, &dst.DynamoComponentDeploymentSharedSpec, carrier); err != nil {
		return err
	}

	// Restore target-only fields that src cannot represent.
	if restored != nil {
		fillSharedAlphaOnlyFromPreserved(&dst.DynamoComponentDeploymentSharedSpec, &restored.DynamoComponentDeploymentSharedSpec)
		restoreDCDAlphaMainContainerOrigin(
			&dst.DynamoComponentDeploymentSharedSpec,
			&src.DynamoComponentDeploymentSharedSpec,
			&restored.DynamoComponentDeploymentSharedSpec,
		)
	}

	// Save source-only fields that dst cannot represent.
	*save = *src
	return nil
}

func restoreDCDAlphaMainContainerOrigin(dst *DynamoComponentDeploymentSharedSpec, src *v1beta1.DynamoComponentDeploymentSharedSpec, restored *DynamoComponentDeploymentSharedSpec) {
	if restored == nil ||
		restored.ExtraPodSpec == nil ||
		restored.ExtraPodSpec.MainContainer == nil ||
		src.PodTemplate == nil {
		return
	}
	liveMain, ok := findContainerByName(src.PodTemplate.Spec.Containers, mainContainerName)
	if !ok {
		return
	}
	projectedMain, hasProjectedMain := projectDCDAlphaMainContainer(restored)
	restoredMain := restored.ExtraPodSpec.MainContainer
	if dst.ExtraPodSpec == nil {
		dst.ExtraPodSpec = &ExtraPodSpec{}
	}
	if dst.ExtraPodSpec.MainContainer == nil {
		dst.ExtraPodSpec.MainContainer = &corev1.Container{}
	}
	main := dst.ExtraPodSpec.MainContainer
	if main.Name == "" {
		main.Name = restoredMain.Name
	}
	if len(restoredMain.Env) > 0 {
		if hasProjectedMain && apiequality.Semantic.DeepEqual(liveMain.Env, projectedMain.Env) {
			main.Env = restoredMain.Env
		} else {
			main.Env = liveMain.Env
		}
		dst.Envs = restored.Envs
	}
	if !apiequality.Semantic.DeepEqual(restoredMain.Resources, corev1.ResourceRequirements{}) {
		if hasProjectedMain && apiequality.Semantic.DeepEqual(liveMain.Resources, projectedMain.Resources) {
			main.Resources = restoredMain.Resources
		} else {
			main.Resources = liveMain.Resources
		}
		dst.Resources = restored.Resources
	}
	if len(restoredMain.VolumeMounts) > 0 {
		if hasProjectedMain && apiequality.Semantic.DeepEqual(liveMain.VolumeMounts, projectedMain.VolumeMounts) {
			main.VolumeMounts = restoredMain.VolumeMounts
		} else {
			main.VolumeMounts = liveMain.VolumeMounts
		}
		dst.VolumeMounts = restored.VolumeMounts
	}
	if restoredMain.LivenessProbe != nil {
		if hasProjectedMain && apiequality.Semantic.DeepEqual(liveMain.LivenessProbe, projectedMain.LivenessProbe) {
			main.LivenessProbe = restoredMain.LivenessProbe
		} else {
			main.LivenessProbe = liveMain.LivenessProbe
		}
		dst.LivenessProbe = restored.LivenessProbe
	}
	if restoredMain.ReadinessProbe != nil {
		if hasProjectedMain && apiequality.Semantic.DeepEqual(liveMain.ReadinessProbe, projectedMain.ReadinessProbe) {
			main.ReadinessProbe = restoredMain.ReadinessProbe
		} else {
			main.ReadinessProbe = liveMain.ReadinessProbe
		}
		dst.ReadinessProbe = restored.ReadinessProbe
	}
}

func projectDCDAlphaMainContainer(src *DynamoComponentDeploymentSharedSpec) (corev1.Container, bool) {
	var hub v1beta1.DynamoComponentDeploymentSharedSpec
	meta := &metav1.ObjectMeta{}
	if err := convertSharedSpecTo(src, &hub, newDCDCarrier(meta)); err != nil || hub.PodTemplate == nil {
		return corev1.Container{}, false
	}
	main, ok := findContainerByName(hub.PodTemplate.Spec.Containers, mainContainerName)
	return main, ok
}

func convert_v1alpha1_DynamoComponentDeploymentStatus_To_v1beta1_DynamoComponentDeploymentStatus(src *DynamoComponentDeploymentStatus, dst *v1beta1.DynamoComponentDeploymentStatus, restored *v1beta1.DynamoComponentDeploymentStatus, save *DynamoComponentDeploymentStatus) error {
	// Convert representable fields from src to dst.
	dst.ObservedGeneration = src.ObservedGeneration
	if len(src.Conditions) > 0 {
		dst.Conditions = make([]metav1.Condition, 0, len(src.Conditions))
		for _, c := range src.Conditions {
			dst.Conditions = append(dst.Conditions, *c.DeepCopy())
		}
	}
	if src.Service != nil {
		dst.Component = convertReplicaStatusTo(src.Service)
	}

	// Restore target-only fields that src cannot represent.
	_ = restored

	// Save source-only fields that dst cannot represent.
	*save = *src
	return nil
}

func convert_v1beta1_DynamoComponentDeploymentStatus_To_v1alpha1_DynamoComponentDeploymentStatus(src *v1beta1.DynamoComponentDeploymentStatus, dst *DynamoComponentDeploymentStatus, restored *DynamoComponentDeploymentStatus, save *v1beta1.DynamoComponentDeploymentStatus) error {
	// Convert representable fields from src to dst.
	dst.ObservedGeneration = src.ObservedGeneration
	if len(src.Conditions) > 0 {
		dst.Conditions = make([]metav1.Condition, 0, len(src.Conditions))
		for _, c := range src.Conditions {
			dst.Conditions = append(dst.Conditions, *c.DeepCopy())
		}
	}
	if src.Component != nil {
		dst.Service = convertReplicaStatusFrom(src.Component)
	}

	// Restore target-only fields that src cannot represent.
	if restored != nil {
		dst.PodSelector = maps.Clone(restored.PodSelector)
		if shouldRestorePreservedComponentName(dst.Service, restored.Service) {
			dst.Service.ComponentName = restored.Service.ComponentName
		}
	}

	// Save source-only fields that dst cannot represent.
	*save = *src
	return nil
}

// scrubDCDAnnotations removes any lingering "nvidia.com/dcd-*" keys that
// convertSharedSpecFrom did not consume.
func scrubDCDAnnotations(obj *metav1.ObjectMeta) {
	scrubAnnotationsByPrefix(obj, annDCDPrefix)
}
