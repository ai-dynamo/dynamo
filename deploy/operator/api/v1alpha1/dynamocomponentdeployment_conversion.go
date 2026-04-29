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
	var preservedHubSpec *v1beta1.DynamoComponentDeploymentSpec

	if raw, ok := dst.ObjectMeta.Annotations[annDCDHubSpec]; ok && raw != "" {
		if spec, ok := restoreDCDHubSpec(raw); ok {
			preservedHubSpec = &spec
			delAnnFromObj(&dst.ObjectMeta, annDCDHubSpec)
		}
	}
	hubOrigin := preservedHubSpec != nil || dst.ObjectMeta.Annotations[annDCDHubOrigin] == annotationTrue
	delAnnFromObj(&dst.ObjectMeta, annDCDHubOrigin)
	if hubOrigin {
		scrubDCDAnnotations(&dst.ObjectMeta)
	}

	var semantic v1beta1.DynamoComponentDeploymentSpec
	semantic.BackendFramework = src.Spec.BackendFramework
	carrier := newDCDCarrier(&dst.ObjectMeta)
	var preservedShared *v1beta1.DynamoComponentDeploymentSharedSpec
	if preservedHubSpec != nil {
		preservedShared = &preservedHubSpec.DynamoComponentDeploymentSharedSpec
	}
	if preservedShared != nil && preservedShared.PodTemplate != nil {
		carrier.set(suffixPodTemplateOrig, annotationTrue)
	}
	if err := convertSharedSpecTo(&src.Spec.DynamoComponentDeploymentSharedSpec,
		&semantic.DynamoComponentDeploymentSharedSpec, carrier, preservedShared); err != nil {
		return err
	}

	// v1beta1 requires DCD.spec.name (it is the +listMapKey on
	// DGD.spec.components and is enforced as Required by the schema). When a
	// v1alpha1 caller omits ServiceName -- the common case for standalone
	// DCDs -- fall back to ObjectMeta.Name so the converted object is
	// schema-valid. The v1beta1 defaulting webhook owns the same defaulting
	// at admission time.
	if semantic.ComponentName == "" && !hubOrigin {
		semantic.ComponentName = dst.ObjectMeta.Name
	}
	dst.Spec = semantic

	preserveSpoke := !hubOrigin ||
		hasSharedAlphaOnlyFields(&src.Spec.DynamoComponentDeploymentSharedSpec) ||
		dcdStatusHasAlphaOnlyFields(&src.Status)
	convertDCDStatusTo(&src.Status, &dst.Status)
	if preserveSpoke {
		preserveDCDSpoke(src, dst)
		preserveDCDSpokeHub(dst)
	}
	return nil
}

func preserveDCDSpoke(src *DynamoComponentDeployment, dst *v1beta1.DynamoComponentDeployment) {
	if data, err := marshalDCDSpokeSpec(&src.Spec); err == nil {
		if dst.ObjectMeta.Annotations == nil {
			dst.ObjectMeta.Annotations = map[string]string{}
		}
		dst.ObjectMeta.Annotations[annDCDSpokeSpec] = string(data)
	}
	if data, err := json.Marshal(src.Status); err == nil {
		if dst.ObjectMeta.Annotations == nil {
			dst.ObjectMeta.Annotations = map[string]string{}
		}
		dst.ObjectMeta.Annotations[annDCDSpokeStatus] = string(data)
	}
}

func fillDCDSpokeFromPreserved(dstSpec *DynamoComponentDeploymentSpec, dstStatus *DynamoComponentDeploymentStatus, preservedSpec *DynamoComponentDeploymentSpec, preservedStatus *DynamoComponentDeploymentStatus, objectName string) {
	if preservedSpec != nil {
		if preservedSpec.ServiceName == "" && dstSpec.ServiceName == objectName {
			dstSpec.ServiceName = ""
		}
	}
	if preservedStatus != nil && len(dstStatus.PodSelector) == 0 && len(preservedStatus.PodSelector) > 0 {
		dstStatus.PodSelector = maps.Clone(preservedStatus.PodSelector)
	}
	if preservedStatus != nil && shouldRestorePreservedComponentName(dstStatus.Service, preservedStatus.Service) {
		dstStatus.Service.ComponentName = preservedStatus.Service.ComponentName
	}
}

func dcdStatusHasAlphaOnlyFields(src *DynamoComponentDeploymentStatus) bool {
	return src != nil &&
		(len(src.PodSelector) > 0 ||
			(src.Service != nil && serviceStatusComponentNameNeedsPreservation(src.Service)))
}

type preservedDCDHubSnapshot struct {
	Spec   string                                  `json:"spec"`
	Status v1beta1.DynamoComponentDeploymentStatus `json:"status"`
}

func preserveDCDSpokeHub(dst *v1beta1.DynamoComponentDeployment) {
	spec, err := marshalDCDHubSpec(&dst.Spec)
	if err != nil {
		return
	}
	data, err := json.Marshal(preservedDCDHubSnapshot{
		Spec:   string(spec),
		Status: dst.Status,
	})
	if err == nil {
		setAnnOnObj(&dst.ObjectMeta, annDCDSpokeHub, string(data))
	}
}

func dcdSpokeHubUnmodified(src *v1beta1.DynamoComponentDeployment) bool {
	raw, ok := src.ObjectMeta.Annotations[annDCDSpokeHub]
	if !ok || raw == "" {
		return false
	}
	spec, err := marshalDCDHubSpec(&src.Spec)
	if err != nil {
		return false
	}
	current, err := json.Marshal(preservedDCDHubSnapshot{
		Spec:   string(spec),
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
	dst.Spec.BackendFramework = src.Spec.BackendFramework

	var preservedSpokeSpec *DynamoComponentDeploymentSpec
	var preservedSpokeStatus *DynamoComponentDeploymentStatus
	if raw, ok := dst.ObjectMeta.Annotations[annDCDSpokeSpec]; ok && raw != "" {
		if spec, ok := restoreDCDSpokeSpec(raw); ok {
			preservedSpokeSpec = &spec
		}
	}
	if rawStatus, ok := dst.ObjectMeta.Annotations[annDCDSpokeStatus]; ok && rawStatus != "" {
		var status DynamoComponentDeploymentStatus
		if err := json.Unmarshal([]byte(rawStatus), &status); err == nil {
			preservedSpokeStatus = &status
		}
	}
	spokeHubUnmodified := dcdSpokeHubUnmodified(src)
	generatedPodTemplate := spokeHubUnmodified && src.ObjectMeta.Annotations[annDCDPrefix+suffixPodTemplateOrig] == "generated"
	carrier := newDCDCarrier(&dst.ObjectMeta)
	var preservedShared *DynamoComponentDeploymentSharedSpec
	if preservedSpokeSpec != nil {
		preservedShared = &preservedSpokeSpec.DynamoComponentDeploymentSharedSpec
	}
	if err := convertSharedSpecFrom(&src.Spec.DynamoComponentDeploymentSharedSpec,
		&dst.Spec.DynamoComponentDeploymentSharedSpec, carrier, preservedShared); err != nil {
		return err
	}

	convertDCDStatusFrom(&src.Status, &dst.Status)
	fillDCDSpokeFromPreserved(&dst.Spec, &dst.Status, preservedSpokeSpec, preservedSpokeStatus, src.ObjectMeta.Name)
	scrubDCDAnnotations(&dst.ObjectMeta)
	if dcdNeedsHubSpecPreservation(&src.Spec, generatedPodTemplate) {
		data, err := marshalDCDHubSpec(&src.Spec)
		if err != nil {
			return fmt.Errorf("preserve DCD hub spec: %w", err)
		}
		if dst.ObjectMeta.Annotations == nil {
			dst.ObjectMeta.Annotations = map[string]string{}
		}
		dst.ObjectMeta.Annotations[annDCDHubSpec] = string(data)
	} else if !hasDCDInternalAnnotations(src.ObjectMeta.Annotations) {
		if dst.ObjectMeta.Annotations == nil {
			dst.ObjectMeta.Annotations = map[string]string{}
		}
		dst.ObjectMeta.Annotations[annDCDHubOrigin] = annotationTrue
	}
	return nil
}

func marshalDCDHubSpec(src *v1beta1.DynamoComponentDeploymentSpec) ([]byte, error) {
	return marshalPreservedSpec(*src.DeepCopy(), func(spec *v1beta1.DynamoComponentDeploymentSpec, records *[]preservedRawJSON) {
		if spec.EPPConfig != nil {
			preserveEPPPluginParameters(spec.EPPConfig.Config, "eppConfig/config", records)
		}
	})
}

func restoreDCDHubSpec(raw string) (v1beta1.DynamoComponentDeploymentSpec, bool) {
	return restorePreservedSpec(raw, func(spec *v1beta1.DynamoComponentDeploymentSpec, records []preservedRawJSON) {
		if spec.EPPConfig != nil {
			restoreEPPPluginParameters(spec.EPPConfig.Config, "eppConfig/config", records)
		}
	})
}

func marshalDCDSpokeSpec(src *DynamoComponentDeploymentSpec) ([]byte, error) {
	return marshalPreservedSpec(*src.DeepCopy(), func(spec *DynamoComponentDeploymentSpec, records *[]preservedRawJSON) {
		if spec.EPPConfig != nil {
			preserveEPPPluginParameters(spec.EPPConfig.Config, "eppConfig/config", records)
		}
	})
}

func restoreDCDSpokeSpec(raw string) (DynamoComponentDeploymentSpec, bool) {
	return restorePreservedSpec(raw, func(spec *DynamoComponentDeploymentSpec, records []preservedRawJSON) {
		if spec.EPPConfig != nil {
			restoreEPPPluginParameters(spec.EPPConfig.Config, "eppConfig/config", records)
		}
	})
}

func dcdNeedsHubSpecPreservation(src *v1beta1.DynamoComponentDeploymentSpec, generatedPodTemplate bool) bool {
	if generatedPodTemplate {
		return false
	}
	return src.FrontendSidecar != nil ||
		src.PodTemplate != nil ||
		(src.Experimental != nil &&
			src.Experimental.GPUMemoryService == nil &&
			src.Experimental.Failover == nil &&
			src.Experimental.Checkpoint == nil)
}

func hasDCDInternalAnnotations(annotations map[string]string) bool {
	for key := range annotations {
		if key == annDCDHubSpec ||
			key == annDCDSpokeSpec ||
			key == annDCDSpokeStatus ||
			key == annDCDSpokeHub ||
			strings.HasPrefix(key, annDCDPrefix) {
			return true
		}
	}
	return false
}

func convertDCDStatusTo(src *DynamoComponentDeploymentStatus, dst *v1beta1.DynamoComponentDeploymentStatus) {
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
	// PodSelector is dropped in v1beta1 (the field was never populated by the
	// controller). No annotation is needed: the round-trip invariant is on
	// v1beta1 inputs, which do not carry PodSelector.
}

func convertDCDStatusFrom(src *v1beta1.DynamoComponentDeploymentStatus, dst *DynamoComponentDeploymentStatus) {
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
}

// scrubDCDAnnotations removes any lingering "nvidia.com/dcd-*" keys that
// convertSharedSpecFrom did not consume.
func scrubDCDAnnotations(obj *metav1.ObjectMeta) {
	scrubAnnotationsByPrefix(obj, annDCDPrefix)
}
