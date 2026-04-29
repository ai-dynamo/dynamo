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
	"strings"

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

type dcdConversionContext struct {
	carrier             *annCarrier
	sourceCarrier       *annCarrier
	objectName          string
	sourceUnmodified    bool
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

	if raw, ok := getAnnFromObj(&dst.ObjectMeta, annDCDHubSpec); ok && raw != "" {
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

	ctx := dcdConversionContext{
		carrier:             newDCDCarrier(&dst.ObjectMeta),
		objectName:          dst.ObjectMeta.Name,
		includeOriginSplits: !hubOrigin,
	}
	var spokeSave DynamoComponentDeploymentSpec
	if err := convertDCDSpecToHub(&src.Spec, &dst.Spec, preservedHubSpec, &spokeSave, ctx); err != nil {
		return err
	}
	var statusSave DynamoComponentDeploymentStatus
	saveDCDAlphaOnlyStatus(&src.Status, &statusSave)
	preserveSpoke := !hubOrigin ||
		!dcdAlphaSpecSaveIsZero(&spokeSave) ||
		!dcdAlphaStatusSaveIsZero(&statusSave)

	convertDCDStatusToHub(&src.Status, &dst.Status, nil, nil, ctx)
	if preserveSpoke {
		preserveDCDSpoke(&spokeSave, &statusSave, dst)
		preserveDCDSpokeHub(dst)
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
	if preservedShared != nil && preservedShared.PodTemplate != nil {
		ctx.carrier.set(suffixPodTemplateOrig, annotationTrue)
	}
	var sharedSave *DynamoComponentDeploymentSharedSpec
	if save != nil {
		sharedSave = &save.DynamoComponentDeploymentSharedSpec
	}
	sharedCtx := sharedSpecConversionContext{
		carrier:             ctx.carrier,
		includeOriginSplits: ctx.includeOriginSplits,
	}
	if err := convertSharedSpecToHub(&src.DynamoComponentDeploymentSharedSpec, &dst.DynamoComponentDeploymentSharedSpec, preservedShared, sharedSave, sharedCtx); err != nil {
		return err
	}

	// v1beta1 requires DCD.spec.name. When a v1alpha1-origin DCD omits
	// ServiceName, fall back to ObjectMeta.Name for schema validity and leave a
	// marker so ConvertFrom can restore the omitted v1alpha1 field.
	if dst.ComponentName == "" && ctx.includeOriginSplits {
		dst.ComponentName = ctx.objectName
		ctx.carrier.set(suffixServiceName, "")
	}

	return nil
}

func preserveDCDSpoke(specSave *DynamoComponentDeploymentSpec, statusSave *DynamoComponentDeploymentStatus, dst *v1beta1.DynamoComponentDeployment) {
	if !dcdAlphaSpecSaveIsZero(specSave) {
		data, err := marshalDCDSpokeSpec(specSave)
		if err == nil {
			setAnnOnObj(&dst.ObjectMeta, annDCDSpokeSpec, string(data))
		}
	}
	if !dcdAlphaStatusSaveIsZero(statusSave) {
		setJSONAnnOnObj(&dst.ObjectMeta, annDCDSpokeStatus, statusSave)
	}
}

func fillDCDSpokeFromPreserved(dstSpec *DynamoComponentDeploymentSpec, dstStatus *DynamoComponentDeploymentStatus, preservedSpec *DynamoComponentDeploymentSpec, preservedStatus *DynamoComponentDeploymentStatus, objectName string) {
	if preservedSpec != nil && dcdPreservedSpokeSpecHasFullShape(preservedSpec) {
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

func dcdPreservedSpokeSpecHasFullShape(preservedSpec *DynamoComponentDeploymentSpec) bool {
	if preservedSpec == nil {
		return false
	}
	var sparse DynamoComponentDeploymentSpec
	saveSharedAlphaOnlySpec(&preservedSpec.DynamoComponentDeploymentSharedSpec, &sparse.DynamoComponentDeploymentSharedSpec, true)
	return !apiequality.Semantic.DeepEqual(*preservedSpec, sparse)
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
	return save == nil ||
		len(save.PodSelector) == 0 &&
			save.Service == nil
}

func preserveDCDSpokeHub(dst *v1beta1.DynamoComponentDeployment) {
	spec, err := marshalDCDHubSpec(&dst.Spec)
	if err != nil {
		return
	}
	setHubSnapshotAnn(&dst.ObjectMeta, annDCDSpokeHub, spec, dst.Status)
}

func dcdSpokeHubUnmodified(src *v1beta1.DynamoComponentDeployment) bool {
	spec, err := marshalDCDHubSpec(&src.Spec)
	if err != nil {
		return false
	}
	return hubSnapshotAnnMatches(&src.ObjectMeta, annDCDSpokeHub, spec, src.Status)
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
	var preservedSpokeStatus *DynamoComponentDeploymentStatus
	if raw, ok := getAnnFromObj(&dst.ObjectMeta, annDCDSpokeSpec); ok && raw != "" {
		if spec, ok := restoreDCDSpokeSpec(raw); ok {
			preservedSpokeSpec = &spec
		}
	}
	if status, ok := getJSONAnnFromObj[DynamoComponentDeploymentStatus](&dst.ObjectMeta, annDCDSpokeStatus); ok {
		preservedSpokeStatus = &status
	}
	_, hadSpokeHubSnapshot := getAnnFromObj(&dst.ObjectMeta, annDCDSpokeHub)
	spokeHubUnmodified := dcdSpokeHubUnmodified(src)

	ctx := dcdConversionContext{
		carrier:          newDCDCarrier(&dst.ObjectMeta),
		sourceCarrier:    newDCDCarrier(&src.ObjectMeta),
		objectName:       src.ObjectMeta.Name,
		sourceUnmodified: spokeHubUnmodified,
	}
	var hubSave v1beta1.DynamoComponentDeploymentSpec
	if err := convertDCDSpecFromHub(&src.Spec, &dst.Spec, preservedSpokeSpec, &hubSave, ctx); err != nil {
		return err
	}

	convertDCDStatusFromHub(&src.Status, &dst.Status, nil, nil, ctx)
	fillDCDSpokeFromPreserved(&dst.Spec, &dst.Status, preservedSpokeSpec, preservedSpokeStatus, src.ObjectMeta.Name)
	scrubDCDAnnotations(&dst.ObjectMeta)
	if preservedSpokeSpec != nil || preservedSpokeStatus != nil || hadSpokeHubSnapshot {
		delAnnFromObj(&dst.ObjectMeta, annDCDSpokeSpec)
		delAnnFromObj(&dst.ObjectMeta, annDCDSpokeStatus)
		delAnnFromObj(&dst.ObjectMeta, annDCDSpokeHub)
	}
	if !dcdHubSpecSaveIsZero(&hubSave) {
		data, err := marshalDCDHubSpec(&hubSave)
		if err != nil {
			return fmt.Errorf("preserve DCD hub spec: %w", err)
		}
		setAnnOnObj(&dst.ObjectMeta, annDCDHubSpec, string(data))
	} else if !hasDCDInternalAnnotations(src.ObjectMeta.Annotations) {
		setAnnOnObj(&dst.ObjectMeta, annDCDHubOrigin, annotationTrue)
	}
	return nil
}

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
	sharedCtx := sharedSpecConversionContext{
		carrier:          ctx.carrier,
		sourceCarrier:    ctx.sourceCarrier,
		sourceUnmodified: ctx.sourceUnmodified,
	}
	if err := convertSharedSpecFromHub(&src.DynamoComponentDeploymentSharedSpec, &dst.DynamoComponentDeploymentSharedSpec, preservedShared, sharedSave, sharedCtx); err != nil {
		return err
	}
	if v, ok := ctx.carrier.get(suffixServiceName); ok {
		if v == "" && src.ComponentName == ctx.objectName {
			dst.ServiceName = ""
		}
		ctx.carrier.del(suffixServiceName)
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

func dcdHubSpecSaveIsZero(save *v1beta1.DynamoComponentDeploymentSpec) bool {
	return save == nil || sharedHubSpecSaveIsZero(&save.DynamoComponentDeploymentSharedSpec)
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

// scrubDCDAnnotations removes any lingering "nvidia.com/dcd-*" keys that
// shared-spec conversion did not consume.
func scrubDCDAnnotations(obj *metav1.ObjectMeta) {
	scrubAnnotationsByPrefix(obj, annDCDPrefix)
}
