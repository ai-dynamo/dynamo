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
	"sigs.k8s.io/controller-runtime/pkg/conversion"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	annDCDSpec   = "nvidia.com/dcd-spec"
	annDCDStatus = "nvidia.com/dcd-status"

	preservedDCDEmptyServiceNamePath = "serviceName"
)

type dcdConversionContext struct {
	objectName          string
	includeOriginSplits bool
}

// ConvertTo converts this DynamoComponentDeployment (v1alpha1) into the hub
// version (v1beta1).
func (src *DynamoComponentDeployment) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoComponentDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoComponentDeployment but got %T", dstRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()
	var preservedHubSpec *v1beta1.DynamoComponentDeploymentSpec

	if raw, ok := getAnnFromObj(&dst.ObjectMeta, annDCDSpec); ok && raw != "" {
		if spec, ok := restoreDCDHubSpec(raw); ok {
			preservedHubSpec = &spec
		}
	}
	hubOrigin := preservedHubSpec != nil
	scrubDCDInternalAnnotations(&dst.ObjectMeta)

	ctx := dcdConversionContext{
		objectName:          dst.ObjectMeta.Name,
		includeOriginSplits: !hubOrigin,
	}
	var spokeSave DynamoComponentDeploymentSpec
	if err := convertDCDSpecToHub(&src.Spec, &dst.Spec, preservedHubSpec, &spokeSave, ctx); err != nil {
		return err
	}
	var statusSave DynamoComponentDeploymentStatus
	saveDCDAlphaOnlyStatus(&src.Status, &statusSave)
	emptyServiceNameSave := dcdEmptyServiceNameNeedsSave(src, dst)
	generatedPodTemplateSave := !hubOrigin && dst.Spec.PodTemplate != nil
	saveSpec := generatedPodTemplateSave || emptyServiceNameSave || !dcdAlphaSpecSaveIsZero(&spokeSave)

	convertDCDStatusToHub(&src.Status, &dst.Status, nil, nil, ctx)
	if saveSpec || !dcdAlphaStatusSaveIsZero(&statusSave) {
		if err := saveDCDSpokeAnnotations(&spokeSave, saveSpec, emptyServiceNameSave, &statusSave, dst); err != nil {
			return err
		}
	}
	return nil
}

func convertDCDSpecToHub(src *DynamoComponentDeploymentSpec, dst *v1beta1.DynamoComponentDeploymentSpec, restored *v1beta1.DynamoComponentDeploymentSpec, save *DynamoComponentDeploymentSpec, ctx dcdConversionContext) error {
	if src == nil || dst == nil {
		return nil
	}

	// Convert fields represented by both versions from the live source.
	dst.BackendFramework = src.BackendFramework

	var preservedShared *v1beta1.DynamoComponentDeploymentSharedSpec
	if restored != nil {
		preservedShared = &restored.DynamoComponentDeploymentSharedSpec
	}
	var sharedSave *DynamoComponentDeploymentSharedSpec
	if save != nil {
		sharedSave = &save.DynamoComponentDeploymentSharedSpec
	}
	sharedCtx := sharedSpecConversionContext{
		includeOriginSplits: ctx.includeOriginSplits,
		podTemplateOrigin:   preservedShared != nil && preservedShared.PodTemplate != nil,
	}
	if err := convertSharedSpecToHub(&src.DynamoComponentDeploymentSharedSpec, &dst.DynamoComponentDeploymentSharedSpec, preservedShared, sharedSave, sharedCtx); err != nil {
		return err
	}

	// v1beta1 requires DCD.spec.name. When a v1alpha1-origin DCD omits
	// ServiceName, fall back to ObjectMeta.Name for schema validity. The
	// sparse spoke save records whether the v1alpha1 field was truly empty.
	if dst.ComponentName == "" {
		dst.ComponentName = ctx.objectName
	}

	// Restore target-only fields that the live source cannot represent.
	if restored != nil && restored.ComponentName == "" && dst.ComponentName == ctx.objectName {
		dst.ComponentName = ""
	}

	return nil
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
	if preservedStatus != nil && shouldRestoreSavedComponentName(dstStatus.Service, preservedStatus.Service) {
		dstStatus.Service.ComponentName = preservedStatus.Service.ComponentName
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

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()

	var preservedSpokeSpec *DynamoComponentDeploymentSpec
	var preservedSpokeEmptyServiceName bool
	var preservedSpokeStatus *DynamoComponentDeploymentStatus
	spokeOrigin := hasDCDSpokeAnnotations(&dst.ObjectMeta)
	if raw, ok := getAnnFromObj(&dst.ObjectMeta, annDCDSpec); ok && raw != "" {
		if spec, emptyServiceName, ok := restoreDCDSpokeSpec(raw); ok {
			preservedSpokeSpec = &spec
			preservedSpokeEmptyServiceName = emptyServiceName
		}
	}
	if status, ok, err := getJSONAnnFromObj[DynamoComponentDeploymentStatus](&dst.ObjectMeta, annDCDStatus); err != nil {
		return err
	} else if ok {
		preservedSpokeStatus = &status
	}

	ctx := dcdConversionContext{
		objectName: src.ObjectMeta.Name,
	}
	var hubSave v1beta1.DynamoComponentDeploymentSpec
	if err := convertDCDSpecFromHub(&src.Spec, &dst.Spec, preservedSpokeSpec, &hubSave, ctx); err != nil {
		return err
	}

	convertDCDStatusFromHub(&src.Status, &dst.Status, nil, nil, ctx)
	restoreDCDAlphaOnlySpecFromSaved(&dst.Spec, preservedSpokeEmptyServiceName, src.ObjectMeta.Name)
	restoreDCDAlphaOnlyStatusFromSaved(&dst.Status, preservedSpokeStatus)
	scrubDCDInternalAnnotations(&dst.ObjectMeta)

	if dcdHubSpecNeedsSave(src, &hubSave, !spokeOrigin) {
		hubSave.ComponentName = src.Spec.ComponentName
		data, err := marshalDCDHubSpec(&hubSave)
		if err != nil {
			return fmt.Errorf("preserve DCD hub spec: %w", err)
		}
		setAnnOnObj(&dst.ObjectMeta, annDCDSpec, string(data))
	}
	return nil
}

func dcdHubSpecNeedsSave(src *v1beta1.DynamoComponentDeployment, save *v1beta1.DynamoComponentDeploymentSpec, saveHubOrigin bool) bool {
	return !dcdHubSpecSaveIsZero(save) ||
		src != nil &&
			(src.Spec.ComponentName == "" &&
				src.ObjectMeta.Name != "" ||
				saveHubOrigin &&
					src.Spec.PodTemplate != nil)
}

//nolint:unparam // Keep the structural conversion signature aligned; this direction does not currently need ctx.
func convertDCDSpecFromHub(src *v1beta1.DynamoComponentDeploymentSpec, dst *DynamoComponentDeploymentSpec, restored *DynamoComponentDeploymentSpec, save *v1beta1.DynamoComponentDeploymentSpec, ctx dcdConversionContext) error {
	if src == nil || dst == nil {
		return nil
	}

	// Convert fields represented by both versions from the live source.
	dst.BackendFramework = src.BackendFramework

	var preservedShared *DynamoComponentDeploymentSharedSpec
	if restored != nil {
		preservedShared = &restored.DynamoComponentDeploymentSharedSpec
	}
	var sharedSave *v1beta1.DynamoComponentDeploymentSharedSpec
	if save != nil {
		sharedSave = &save.DynamoComponentDeploymentSharedSpec
	}
	sharedCtx := sharedSpecConversionContext{}
	if err := convertSharedSpecFromHub(&src.DynamoComponentDeploymentSharedSpec, &dst.DynamoComponentDeploymentSharedSpec, preservedShared, sharedSave, sharedCtx); err != nil {
		return err
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

//nolint:unparam // Keep the structural conversion signature; this leaf has no preserved fields.
func convertDCDStatusToHub(src *DynamoComponentDeploymentStatus, dst *v1beta1.DynamoComponentDeploymentStatus, restored *v1beta1.DynamoComponentDeploymentStatus, save *DynamoComponentDeploymentStatus, ctx dcdConversionContext) {
	_, _, _ = restored, save, ctx

	dst.ObservedGeneration = src.ObservedGeneration
	if len(src.Conditions) > 0 {
		dst.Conditions = make([]metav1.Condition, 0, len(src.Conditions))
		for _, c := range src.Conditions {
			dst.Conditions = append(dst.Conditions, *c.DeepCopy())
		}
	}
	if src.Service != nil {
		dst.Component = &v1beta1.ComponentReplicaStatus{}
		convertReplicaStatusToHub(src.Service, dst.Component, nil, nil, replicaStatusConversionContext{})
	}
	// PodSelector is dropped in v1beta1 (the field was never populated by the
	// controller). No annotation is needed: the round-trip invariant is on
	// v1beta1 inputs, which do not carry PodSelector.
}

//nolint:unparam // Keep the structural conversion signature; this leaf has no preserved fields.
func convertDCDStatusFromHub(src *v1beta1.DynamoComponentDeploymentStatus, dst *DynamoComponentDeploymentStatus, restored *DynamoComponentDeploymentStatus, save *v1beta1.DynamoComponentDeploymentStatus, ctx dcdConversionContext) {
	_, _, _ = restored, save, ctx

	dst.ObservedGeneration = src.ObservedGeneration
	if len(src.Conditions) > 0 {
		dst.Conditions = make([]metav1.Condition, 0, len(src.Conditions))
		for _, c := range src.Conditions {
			dst.Conditions = append(dst.Conditions, *c.DeepCopy())
		}
	}
	if src.Component != nil {
		dst.Service = &ServiceReplicaStatus{}
		convertReplicaStatusFromHub(src.Component, dst.Service, nil, nil, replicaStatusConversionContext{})
	}
}

func scrubDCDInternalAnnotations(obj metav1.Object) {
	for _, key := range []string{
		annDCDSpec,
		annDCDStatus,
	} {
		delAnnFromObj(obj, key)
	}
}
