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
	"fmt"
	"maps"
	"slices"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiconversion "k8s.io/apimachinery/pkg/conversion"
	"sigs.k8s.io/controller-runtime/pkg/conversion"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	annDCDSpec   = "nvidia.com/dcd-spec"
	annDCDStatus = "nvidia.com/dcd-status"

	preservedDCDEmptyServiceNamePath = "serviceName"
)

// IsDynamoComponentDeploymentConversionAnnotation reports whether key is owned
// by the DCD conversion layer and should be treated as conversion bookkeeping.
func IsDynamoComponentDeploymentConversionAnnotation(key string) bool {
	switch key {
	case annDCDSpec, annDCDStatus:
		return true
	default:
		return false
	}
}

// ConvertTo converts this DynamoComponentDeployment (v1alpha1) into the hub
// version (v1beta1).
func (src *DynamoComponentDeployment) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoComponentDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoComponentDeployment but got %T", dstRaw)
	}
	return Convert_v1alpha1_DynamoComponentDeployment_To_v1beta1_DynamoComponentDeployment(src, dst, nil)
}

func dcdEmptyServiceNameNeedsSave(src *DynamoComponentDeployment, dst *v1beta1.DynamoComponentDeployment) bool {
	return src != nil &&
		dst != nil &&
		src.Spec.ServiceName == "" &&
		src.ObjectMeta.Name != "" &&
		dst.Spec.ComponentName == src.ObjectMeta.Name
}

func saveDCDSpokeAnnotations(specSave *DynamoComponentDeploymentSpec, saveSpec bool, emptyServiceName bool, statusSave *DynamoComponentDeploymentStatus, dst *v1beta1.DynamoComponentDeployment) error {
	if saveSpec {
		data, err := marshalDCDSpokeSpec(specSave, emptyServiceName)
		if err != nil {
			return fmt.Errorf("preserve DCD spoke spec: %w", err)
		}
		setAnnOnObj(&dst.ObjectMeta, annDCDSpec, string(data))
	}
	if !dcdAlphaStatusSaveIsZero(statusSave) {
		if err := setJSONAnnOnObj(&dst.ObjectMeta, annDCDStatus, statusSave); err != nil {
			return err
		}
	}
	return nil
}

func restoreDCDAlphaOnlySpecFromSaved(dstSpec *DynamoComponentDeploymentSpec, emptyServiceName bool, objectName string) {
	if emptyServiceName && dstSpec.ServiceName == objectName {
		dstSpec.ServiceName = ""
	}
}

func restoreDCDAlphaOnlyStatusFromSaved(dstStatus *DynamoComponentDeploymentStatus, preservedStatus *DynamoComponentDeploymentStatus) {
	if preservedStatus != nil && len(dstStatus.PodSelector) == 0 && len(preservedStatus.PodSelector) > 0 {
		dstStatus.PodSelector = maps.Clone(preservedStatus.PodSelector)
	}
	if preservedStatus != nil && shouldRestoreSavedServiceReplicaStatus(dstStatus.Service, preservedStatus.Service) {
		dstStatus.Service.ComponentName = preservedStatus.Service.ComponentName
		dstStatus.Service.ComponentNames = slices.Clone(preservedStatus.Service.ComponentNames)
	}
}

func dcdAlphaSpecSaveIsZero(save *DynamoComponentDeploymentSpec) bool {
	return save == nil || apiequality.Semantic.DeepEqual(*save, DynamoComponentDeploymentSpec{})
}

func saveDCDAlphaOnlyStatus(src *DynamoComponentDeploymentStatus, save *DynamoComponentDeploymentStatus) {
	if src == nil || save == nil {
		return
	}
	if len(src.PodSelector) > 0 {
		save.PodSelector = maps.Clone(src.PodSelector)
	}
	if serviceStatusComponentNameNeedsPreservation(src.Service) {
		save.Service = &ServiceReplicaStatus{
			ComponentName:  src.Service.ComponentName,
			ComponentNames: slices.Clone(src.Service.ComponentNames),
		}
	}
}

func dcdAlphaStatusSaveIsZero(save *DynamoComponentDeploymentStatus) bool {
	return save == nil || apiequality.Semantic.DeepEqual(*save, DynamoComponentDeploymentStatus{})
}

// ConvertFrom converts from the hub (v1beta1) DynamoComponentDeployment into
// this v1alpha1 instance.
func (dst *DynamoComponentDeployment) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoComponentDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoComponentDeployment but got %T", srcRaw)
	}
	return Convert_v1beta1_DynamoComponentDeployment_To_v1alpha1_DynamoComponentDeployment(src, dst, nil)
}

func dcdHubSpecNeedsSave(src *v1beta1.DynamoComponentDeployment, save *v1beta1.DynamoComponentDeploymentSpec, saveHubOrigin bool) bool {
	return !dcdHubSpecSaveIsZero(save) ||
		src != nil &&
			(src.Spec.ComponentName == "" &&
				src.ObjectMeta.Name != "" ||
				saveHubOrigin &&
					src.Spec.PodTemplate != nil)
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

func marshalDCDSpokeSpec(src *DynamoComponentDeploymentSpec, emptyServiceName bool) ([]byte, error) {
	return marshalPreservedSpec(*src.DeepCopy(), func(spec *DynamoComponentDeploymentSpec, records *[]preservedRawJSON) {
		if emptyServiceName {
			*records = append(*records, preservedRawJSON{
				Path: preservedDCDEmptyServiceNamePath,
				Nil:  true,
			})
		}
		if spec.EPPConfig != nil {
			preserveEPPPluginParameters(spec.EPPConfig.Config, "eppConfig/config", records)
		}
	})
}

func restoreDCDSpokeSpec(raw string) (DynamoComponentDeploymentSpec, bool, bool) {
	emptyServiceName := false
	spec, ok := restorePreservedSpec(raw, func(spec *DynamoComponentDeploymentSpec, records []preservedRawJSON) {
		for _, record := range records {
			if record.Path == preservedDCDEmptyServiceNamePath && record.Nil {
				emptyServiceName = true
			}
		}
		if spec.EPPConfig != nil {
			restoreEPPPluginParameters(spec.EPPConfig.Config, "eppConfig/config", records)
		}
	})
	return spec, emptyServiceName, ok
}

func dcdHubSpecSaveIsZero(save *v1beta1.DynamoComponentDeploymentSpec) bool {
	return save == nil || sharedHubSpecSaveIsZero(&save.DynamoComponentDeploymentSharedSpec)
}

func hasDCDSpokeAnnotations(obj metav1.Object) bool {
	_, hasSpec := getAnnFromObj(obj, annDCDSpec)
	_, hasStatus := getAnnFromObj(obj, annDCDStatus)
	return hasSpec || hasStatus
}

func scrubDCDInternalAnnotations(obj metav1.Object) {
	for _, key := range []string{
		annDCDSpec,
		annDCDStatus,
	} {
		delAnnFromObj(obj, key)
	}
}

// Convert_v1alpha1_DynamoComponentDeployment_To_v1beta1_DynamoComponentDeployment
// converts live fields through conversion-gen, then restores hub-only fields
// and records v1alpha1-only fields in the reserved conversion annotations.
func Convert_v1alpha1_DynamoComponentDeployment_To_v1beta1_DynamoComponentDeployment(src *DynamoComponentDeployment, dst *v1beta1.DynamoComponentDeployment, s apiconversion.Scope) error {
	var preservedHubSpec *v1beta1.DynamoComponentDeploymentSpec
	if raw, ok := getAnnFromObj(&src.ObjectMeta, annDCDSpec); ok && raw != "" {
		if spec, ok := restoreDCDHubSpec(raw); ok {
			preservedHubSpec = &spec
		}
	}
	hubOrigin := preservedHubSpec != nil

	converted := v1beta1.DynamoComponentDeployment{TypeMeta: dst.TypeMeta}
	if err := autoConvert_v1alpha1_DynamoComponentDeployment_To_v1beta1_DynamoComponentDeployment(src, &converted, s); err != nil {
		return err
	}
	converted.ObjectMeta = *src.ObjectMeta.DeepCopy()

	// v1beta1 requires DCD.spec.name. When a v1alpha1-origin DCD omits
	// ServiceName, fall back to ObjectMeta.Name for schema validity. The sparse
	// spoke save below records whether the v1alpha1 field was truly empty.
	if converted.Spec.ComponentName == "" {
		converted.Spec.ComponentName = src.ObjectMeta.Name
	}
	generatedPodTemplateSave := !hubOrigin && converted.Spec.PodTemplate != nil

	var spokeSave DynamoComponentDeploymentSpec
	saveSharedAlphaOnlySpec(&src.Spec.DynamoComponentDeploymentSharedSpec, &spokeSave.DynamoComponentDeploymentSharedSpec, !hubOrigin)

	// Restore target-only fields that the live v1alpha1 source cannot represent.
	if preservedHubSpec != nil {
		preservedShared := &preservedHubSpec.DynamoComponentDeploymentSharedSpec
		if err := restoreSharedHubProjection(&src.Spec.DynamoComponentDeploymentSharedSpec, &converted.Spec.DynamoComponentDeploymentSharedSpec, preservedShared); err != nil {
			return err
		}
		if preservedHubSpec.ComponentName == "" && converted.Spec.ComponentName == src.ObjectMeta.Name {
			converted.Spec.ComponentName = ""
		}
	}
	emptyServiceNameSave := dcdEmptyServiceNameNeedsSave(src, &converted)

	var statusSave DynamoComponentDeploymentStatus
	saveDCDAlphaOnlyStatus(&src.Status, &statusSave)
	saveSpec := generatedPodTemplateSave || emptyServiceNameSave || !dcdAlphaSpecSaveIsZero(&spokeSave)

	scrubDCDInternalAnnotations(&converted.ObjectMeta)
	if saveSpec || !dcdAlphaStatusSaveIsZero(&statusSave) {
		if err := saveDCDSpokeAnnotations(&spokeSave, saveSpec, emptyServiceNameSave, &statusSave, &converted); err != nil {
			return err
		}
	}
	*dst = converted
	return nil
}

// Convert_v1beta1_DynamoComponentDeployment_To_v1alpha1_DynamoComponentDeployment
// converts live fields through conversion-gen, then restores spoke-only fields
// and records v1beta1-only fields in the reserved conversion annotations.
func Convert_v1beta1_DynamoComponentDeployment_To_v1alpha1_DynamoComponentDeployment(src *v1beta1.DynamoComponentDeployment, dst *DynamoComponentDeployment, s apiconversion.Scope) error {
	var preservedSpokeSpec *DynamoComponentDeploymentSpec
	var preservedSpokeEmptyServiceName bool
	var preservedSpokeStatus *DynamoComponentDeploymentStatus
	spokeOrigin := hasDCDSpokeAnnotations(&src.ObjectMeta)
	if raw, ok := getAnnFromObj(&src.ObjectMeta, annDCDSpec); ok && raw != "" {
		if spec, emptyServiceName, ok := restoreDCDSpokeSpec(raw); ok {
			preservedSpokeSpec = &spec
			preservedSpokeEmptyServiceName = emptyServiceName
		}
	}
	if status, ok, err := getJSONAnnFromObj[DynamoComponentDeploymentStatus](&src.ObjectMeta, annDCDStatus); err != nil {
		return err
	} else if ok {
		preservedSpokeStatus = &status
	}

	converted := DynamoComponentDeployment{TypeMeta: dst.TypeMeta}
	if err := autoConvert_v1beta1_DynamoComponentDeployment_To_v1alpha1_DynamoComponentDeployment(src, &converted, s); err != nil {
		return err
	}
	converted.ObjectMeta = *src.ObjectMeta.DeepCopy()

	var hubSave v1beta1.DynamoComponentDeploymentSpec
	if preservedSpokeSpec != nil {
		preservedShared := &preservedSpokeSpec.DynamoComponentDeploymentSharedSpec
		if err := restoreSharedAlphaProjection(&src.Spec.DynamoComponentDeploymentSharedSpec, &converted.Spec.DynamoComponentDeploymentSharedSpec, preservedShared); err != nil {
			return err
		}
	}
	if err := saveSharedHubOnlySpec(&src.Spec.DynamoComponentDeploymentSharedSpec, &converted.Spec.DynamoComponentDeploymentSharedSpec, &hubSave.DynamoComponentDeploymentSharedSpec); err != nil {
		return err
	}
	restoreDCDAlphaOnlySpecFromSaved(&converted.Spec, preservedSpokeEmptyServiceName, src.ObjectMeta.Name)
	restoreDCDAlphaOnlyStatusFromSaved(&converted.Status, preservedSpokeStatus)
	scrubDCDInternalAnnotations(&converted.ObjectMeta)

	if dcdHubSpecNeedsSave(src, &hubSave, !spokeOrigin) {
		hubSave.ComponentName = src.Spec.ComponentName
		data, err := marshalDCDHubSpec(&hubSave)
		if err != nil {
			return fmt.Errorf("preserve DCD hub spec: %w", err)
		}
		setAnnOnObj(&converted.ObjectMeta, annDCDSpec, string(data))
	}
	*dst = converted
	return nil
}

// Convert_v1alpha1_DynamoComponentDeploymentStatus_To_v1beta1_DynamoComponentDeploymentStatus
// converts the common status fields and maps Service to Component.
func Convert_v1alpha1_DynamoComponentDeploymentStatus_To_v1beta1_DynamoComponentDeploymentStatus(in *DynamoComponentDeploymentStatus, out *v1beta1.DynamoComponentDeploymentStatus, s apiconversion.Scope) error {
	if err := autoConvert_v1alpha1_DynamoComponentDeploymentStatus_To_v1beta1_DynamoComponentDeploymentStatus(in, out, s); err != nil {
		return err
	}
	if in.Service != nil {
		out.Component = &v1beta1.ComponentReplicaStatus{}
		if err := Convert_v1alpha1_ServiceReplicaStatus_To_v1beta1_ComponentReplicaStatus(in.Service, out.Component, s); err != nil {
			return err
		}
	}
	return nil
}

// Convert_v1beta1_DynamoComponentDeploymentStatus_To_v1alpha1_DynamoComponentDeploymentStatus
// converts the common status fields and maps Component to Service.
func Convert_v1beta1_DynamoComponentDeploymentStatus_To_v1alpha1_DynamoComponentDeploymentStatus(in *v1beta1.DynamoComponentDeploymentStatus, out *DynamoComponentDeploymentStatus, s apiconversion.Scope) error {
	if err := autoConvert_v1beta1_DynamoComponentDeploymentStatus_To_v1alpha1_DynamoComponentDeploymentStatus(in, out, s); err != nil {
		return err
	}
	if in.Component != nil {
		out.Service = &ServiceReplicaStatus{}
		if err := Convert_v1beta1_ComponentReplicaStatus_To_v1alpha1_ServiceReplicaStatus(in.Component, out.Service, s); err != nil {
			return err
		}
	}
	return nil
}
