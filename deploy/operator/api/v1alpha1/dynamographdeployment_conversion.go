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

// Conversion between v1alpha1 and v1beta1 DynamoGraphDeployment.
//
// v1beta1 is the hub (see api/v1beta1/dynamographdeployment_conversion.go).
// This file implements v1alpha1 as a spoke in the hub-and-spoke model used by
// controller-runtime's conversion webhook.
//
// Round-trip fidelity
//
// For every v1beta1 input V, ConvertTo(ConvertFrom(V)) must equal V bitwise.
// Lossy-direction fields (v1alpha1 shapes with no v1beta1 equivalent, and
// v1beta1 ordering that is not representable in v1alpha1's unordered map) are
// preserved via reserved "nvidia.com/dgd-*" annotations. The annotation
// namespace is operator-owned; user-set annotations with the same prefix are
// parsed best-effort and consumed on ConvertFrom.

package v1alpha1

import (
	"fmt"
	"slices"
	"strings"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiconversion "k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/conversion"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	dgdConversionAnnotationPrefix = "nvidia.com/dgd-"
	annDGDSpec                    = dgdConversionAnnotationPrefix + "spec"
	annDGDStatus                  = dgdConversionAnnotationPrefix + "status"
)

// IsDynamoGraphDeploymentConversionAnnotation reports whether key is reserved
// for DGD conversion bookkeeping.
func IsDynamoGraphDeploymentConversionAnnotation(key string) bool {
	return strings.HasPrefix(key, dgdConversionAnnotationPrefix)
}

// ConvertTo converts this DynamoGraphDeployment (v1alpha1) into the hub
// version (v1beta1).
func (src *DynamoGraphDeployment) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeployment but got %T", dstRaw)
	}
	return Convert_v1alpha1_DynamoGraphDeployment_To_v1beta1_DynamoGraphDeployment(src, dst, nil)
}

func saveDGDSpokeAnnotations(specSave *DynamoGraphDeploymentSpec, statusSave *DynamoGraphDeploymentStatus, dst *v1beta1.DynamoGraphDeployment) error {
	if !dgdAlphaSpecSaveIsZero(specSave) {
		data, err := marshalDGDSpokeSpec(specSave)
		if err != nil {
			return fmt.Errorf("preserve DGD spoke spec: %w", err)
		}
		setAnnOnObj(&dst.ObjectMeta, annDGDSpec, string(data))
	}
	if !dgdAlphaStatusSaveIsZero(statusSave) {
		if err := setJSONAnnOnObj(&dst.ObjectMeta, annDGDStatus, statusSave); err != nil {
			return err
		}
	}
	return nil
}

func restoredDGDHubComponentsByName(restored *v1beta1.DynamoGraphDeploymentSpec) map[string]*v1beta1.DynamoComponentDeploymentSharedSpec {
	if restored == nil || len(restored.Components) == 0 {
		return nil
	}
	out := make(map[string]*v1beta1.DynamoComponentDeploymentSharedSpec, len(restored.Components))
	for i := range restored.Components {
		out[restored.Components[i].ComponentName] = &restored.Components[i]
	}
	return out
}

func dgdServiceNamesInEmissionOrder(services map[string]*DynamoComponentDeploymentSharedSpec, restored *v1beta1.DynamoGraphDeploymentSpec) []string {
	if len(services) == 0 {
		return nil
	}
	if restored == nil || !dgdComponentOrderNeedsPreservation(restored.Components) {
		return sets.List(sets.KeySet(services))
	}

	remaining := sets.KeySet(services)
	out := make([]string, 0, len(services))
	for _, comp := range restored.Components {
		name := comp.ComponentName
		if !remaining.Has(name) {
			continue
		}
		remaining.Delete(name)
		out = append(out, name)
	}
	return append(out, sets.List(remaining)...)
}

func dgdComponentOrderNeedsPreservation(components []v1beta1.DynamoComponentDeploymentSharedSpec) bool {
	for i := 1; i < len(components); i++ {
		if components[i-1].ComponentName > components[i].ComponentName {
			return true
		}
	}
	return false
}

func restoreDGDAlphaOnlySpecFromSaved(dstSpec *DynamoGraphDeploymentSpec, savedSpec *DynamoGraphDeploymentSpec) {
	if savedSpec != nil {
		if len(dstSpec.PVCs) == 0 {
			dstSpec.PVCs = slices.Clone(savedSpec.PVCs)
		}
		for name, savedComp := range savedSpec.Services {
			if savedComp != nil {
				continue
			}
			if dstSpec.Services == nil {
				dstSpec.Services = map[string]*DynamoComponentDeploymentSharedSpec{}
			}
			if _, ok := dstSpec.Services[name]; !ok {
				dstSpec.Services[name] = nil
			}
		}
		for name, dstComp := range dstSpec.Services {
			if dstComp == nil || savedSpec.Services == nil {
				continue
			}
			savedComp := savedSpec.Services[name]
			if savedComp == nil {
				continue
			}
			if dstComp.ServiceName == "" && savedComp.ServiceName != "" && savedComp.ServiceName != name {
				dstComp.ServiceName = savedComp.ServiceName
			}
		}
	}
}

func restoreDGDAlphaOnlyStatusFromSaved(dstStatus *DynamoGraphDeploymentStatus, savedStatus *DynamoGraphDeploymentStatus) {
	if savedStatus == nil {
		return
	}
	for name, dstSvc := range dstStatus.Services {
		savedSvc, ok := savedStatus.Services[name]
		if !ok {
			continue
		}
		if shouldRestoreSavedServiceReplicaStatus(&dstSvc, &savedSvc) {
			dstSvc.ComponentName = savedSvc.ComponentName
			dstSvc.ComponentNames = slices.Clone(savedSvc.ComponentNames)
		}
		dstStatus.Services[name] = dstSvc
	}
}

func dgdAlphaSpecSaveIsZero(save *DynamoGraphDeploymentSpec) bool {
	return save == nil || apiequality.Semantic.DeepEqual(*save, DynamoGraphDeploymentSpec{})
}

func saveDGDAlphaOnlyStatus(src *DynamoGraphDeploymentStatus, save *DynamoGraphDeploymentStatus) {
	if src == nil || save == nil {
		return
	}
	for name, svc := range src.Services {
		if !serviceStatusComponentNameNeedsPreservation(&svc) {
			continue
		}
		if save.Services == nil {
			save.Services = map[string]ServiceReplicaStatus{}
		}
		save.Services[name] = ServiceReplicaStatus{
			ComponentName:  svc.ComponentName,
			ComponentNames: slices.Clone(svc.ComponentNames),
		}
	}
}

func dgdAlphaStatusSaveIsZero(save *DynamoGraphDeploymentStatus) bool {
	return save == nil || apiequality.Semantic.DeepEqual(*save, DynamoGraphDeploymentStatus{})
}

func dgdHubSpecSaveIsZero(save *v1beta1.DynamoGraphDeploymentSpec) bool {
	return save == nil || apiequality.Semantic.DeepEqual(*save, v1beta1.DynamoGraphDeploymentSpec{})
}

func dgdHubComponentSaveIsZero(save *v1beta1.DynamoComponentDeploymentSharedSpec) bool {
	return sharedHubSpecSaveIsZero(save)
}

func serviceStatusComponentNameNeedsPreservation(src *ServiceReplicaStatus) bool {
	if src == nil {
		return false
	}
	if len(src.ComponentNames) == 0 {
		return src.ComponentName != ""
	}
	return src.ComponentNames[len(src.ComponentNames)-1] != src.ComponentName
}

func shouldRestoreSavedServiceReplicaStatus(dst, saved *ServiceReplicaStatus) bool {
	if !serviceStatusComponentNameNeedsPreservation(saved) || dst == nil {
		return false
	}
	if slices.Equal(dst.ComponentNames, componentNamesToHub(saved)) {
		return true
	}
	return saved.ComponentName != "" &&
		len(saved.ComponentNames) == 0 &&
		len(dst.ComponentNames) == 0
}

func componentNamesToHub(src *ServiceReplicaStatus) []string {
	if src == nil {
		return nil
	}
	if len(src.ComponentNames) > 0 {
		return src.ComponentNames
	}
	if src.ComponentName != "" {
		return []string{src.ComponentName}
	}
	return nil
}

func restoreDGDSpokeAnnotations(obj metav1.Object) (*DynamoGraphDeploymentSpec, *DynamoGraphDeploymentStatus, error) {
	var restoredSpokeSpec *DynamoGraphDeploymentSpec
	var restoredSpokeStatus *DynamoGraphDeploymentStatus
	if raw, ok := getAnnFromObj(obj, annDGDSpec); ok && raw != "" {
		if spec, ok := restoreDGDSpokeSpec(raw); ok {
			restoredSpokeSpec = &spec
		}
	}
	if status, ok, err := getJSONAnnFromObj[DynamoGraphDeploymentStatus](obj, annDGDStatus); err != nil {
		return nil, nil, err
	} else if ok {
		restoredSpokeStatus = &status
	}
	return restoredSpokeSpec, restoredSpokeStatus, nil
}

// ConvertFrom converts from the hub (v1beta1) DynamoGraphDeployment into this
// v1alpha1 instance.
func (dst *DynamoGraphDeployment) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeployment but got %T", srcRaw)
	}
	return Convert_v1beta1_DynamoGraphDeployment_To_v1alpha1_DynamoGraphDeployment(src, dst, nil)
}

func dgdHubComponentOriginSaveNeeded(src *v1beta1.DynamoComponentDeploymentSharedSpec) bool {
	return src != nil && src.PodTemplate != nil
}

func marshalDGDHubSpec(src *v1beta1.DynamoGraphDeploymentSpec) ([]byte, error) {
	return marshalPreservedSpec(*src.DeepCopy(), func(spec *v1beta1.DynamoGraphDeploymentSpec, records *[]preservedRawJSON) {
		for i := range spec.Components {
			if spec.Components[i].EPPConfig != nil {
				preserveEPPPluginParameters(spec.Components[i].EPPConfig.Config, fmt.Sprintf("components/%d/eppConfig/config", i), records)
			}
		}
	})
}

func restoreDGDHubSpec(raw string) (v1beta1.DynamoGraphDeploymentSpec, bool) {
	return restorePreservedSpec(raw, func(spec *v1beta1.DynamoGraphDeploymentSpec, records []preservedRawJSON) {
		for i := range spec.Components {
			if spec.Components[i].EPPConfig != nil {
				restoreEPPPluginParameters(spec.Components[i].EPPConfig.Config, fmt.Sprintf("components/%d/eppConfig/config", i), records)
			}
		}
	})
}

func marshalDGDSpokeSpec(src *DynamoGraphDeploymentSpec) ([]byte, error) {
	return marshalPreservedSpec(*src.DeepCopy(), func(spec *DynamoGraphDeploymentSpec, records *[]preservedRawJSON) {
		for name, svc := range spec.Services {
			if svc != nil && svc.EPPConfig != nil {
				preserveEPPPluginParameters(svc.EPPConfig.Config, fmt.Sprintf("services/%s/eppConfig/config", name), records)
			}
		}
	})
}

func restoreDGDSpokeSpec(raw string) (DynamoGraphDeploymentSpec, bool) {
	return restorePreservedSpec(raw, func(spec *DynamoGraphDeploymentSpec, records []preservedRawJSON) {
		for name, svc := range spec.Services {
			if svc != nil && svc.EPPConfig != nil {
				restoreEPPPluginParameters(svc.EPPConfig.Config, fmt.Sprintf("services/%s/eppConfig/config", name), records)
			}
		}
	})
}

func hasDGDSpokeAnnotations(obj metav1.Object) bool {
	_, hasSpec := getAnnFromObj(obj, annDGDSpec)
	_, hasStatus := getAnnFromObj(obj, annDGDStatus)
	return hasSpec || hasStatus
}

func scrubDGDInternalAnnotations(obj metav1.Object) {
	for _, key := range []string{
		annDGDSpec,
		annDGDStatus,
	} {
		delAnnFromObj(obj, key)
	}
}

// applyDGDToHubPreservation restores hub-only component leaves and collects
// v1alpha1-only values. It also preserves a prior hub component order; without
// one, alpha-first output uses deterministic name ordering.
func applyDGDToHubPreservation(src *DynamoGraphDeploymentSpec, dst *v1beta1.DynamoGraphDeploymentSpec, restored *v1beta1.DynamoGraphDeploymentSpec, save *DynamoGraphDeploymentSpec, includeOriginSplits bool) error {
	if save != nil && len(src.PVCs) > 0 {
		save.PVCs = slices.Clone(src.PVCs)
	}
	if len(src.Services) == 0 {
		return nil
	}

	convertedByName := make(map[string]v1beta1.DynamoComponentDeploymentSharedSpec, len(dst.Components))
	for i := range dst.Components {
		convertedByName[dst.Components[i].ComponentName] = dst.Components[i]
	}
	restoredByName := restoredDGDHubComponentsByName(restored)
	names := dgdServiceNamesInEmissionOrder(src.Services, restored)
	components := make([]v1beta1.DynamoComponentDeploymentSharedSpec, 0, len(names))
	for _, name := range names {
		service := src.Services[name]
		if service == nil {
			if save != nil {
				if save.Services == nil {
					save.Services = map[string]*DynamoComponentDeploymentSharedSpec{}
				}
				save.Services[name] = nil
			}
			continue
		}
		component, ok := convertedByName[name]
		if !ok {
			return fmt.Errorf("converted component %q is missing", name)
		}
		if err := restoreSharedHubProjection(service, &component, restoredByName[name]); err != nil {
			return fmt.Errorf("component %q: %w", name, err)
		}
		// In v1alpha1 DGD, the services-map key is the canonical name. A value
		// in ServiceName is legacy/redundant, so v1beta1 ComponentName must use
		// the map key to satisfy its listMapKey=name invariant. Preserve a
		// mismatched ServiceName separately so it can still round-trip.
		component.ComponentName = name
		if save != nil {
			componentSave := &DynamoComponentDeploymentSharedSpec{}
			saveSharedAlphaOnlySpec(service, componentSave, includeOriginSplits)
			if service.ServiceName != "" && service.ServiceName != name {
				componentSave.ServiceName = service.ServiceName
			}
			if !sharedAlphaSpecSaveIsZero(componentSave) {
				if save.Services == nil {
					save.Services = map[string]*DynamoComponentDeploymentSharedSpec{}
				}
				save.Services[name] = componentSave
			}
		}
		components = append(components, component)
	}
	dst.Components = components
	return nil
}

// applyDGDToAlphaPreservation restores v1alpha1-only component leaves and
// collects hub-only values, including list order and pod-template origin.
func applyDGDToAlphaPreservation(src *v1beta1.DynamoGraphDeploymentSpec, dst *DynamoGraphDeploymentSpec, restored *DynamoGraphDeploymentSpec, save *v1beta1.DynamoGraphDeploymentSpec, saveHubOrigin bool) error {
	if len(src.Components) == 0 {
		return nil
	}
	preserveOrder := dgdComponentOrderNeedsPreservation(src.Components)
	for i := range src.Components {
		component := &src.Components[i]
		service := dst.Services[component.ComponentName]
		if service == nil {
			return fmt.Errorf("converted component %q is missing", component.ComponentName)
		}
		var preservedService *DynamoComponentDeploymentSharedSpec
		if restored != nil && restored.Services != nil {
			preservedService = restored.Services[component.ComponentName]
		}
		if err := restoreSharedAlphaProjection(component, service, preservedService); err != nil {
			return fmt.Errorf("component %q: %w", component.ComponentName, err)
		}
		service.ServiceName = ""

		if save != nil {
			componentSave := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: component.ComponentName}
			if err := saveSharedHubOnlySpec(component, service, &componentSave); err != nil {
				return fmt.Errorf("component %q: %w", component.ComponentName, err)
			}
			if preserveOrder || !dgdHubComponentSaveIsZero(&componentSave) || saveHubOrigin && dgdHubComponentOriginSaveNeeded(component) {
				save.Components = append(save.Components, componentSave)
			}
		}
	}
	return nil
}

func Convert_v1alpha1_DynamoGraphDeployment_To_v1beta1_DynamoGraphDeployment(src *DynamoGraphDeployment, dst *v1beta1.DynamoGraphDeployment, s apiconversion.Scope) error {
	var restoredHubSpec *v1beta1.DynamoGraphDeploymentSpec
	if raw, ok := getAnnFromObj(&src.ObjectMeta, annDGDSpec); ok && raw != "" {
		if spec, ok := restoreDGDHubSpec(raw); ok {
			restoredHubSpec = &spec
		}
	}
	hubOrigin := restoredHubSpec != nil

	converted := v1beta1.DynamoGraphDeployment{TypeMeta: dst.TypeMeta}
	if err := autoConvert_v1alpha1_DynamoGraphDeployment_To_v1beta1_DynamoGraphDeployment(src, &converted, s); err != nil {
		return err
	}
	converted.ObjectMeta = *src.ObjectMeta.DeepCopy()

	var spokeSave DynamoGraphDeploymentSpec
	if err := applyDGDToHubPreservation(&src.Spec, &converted.Spec, restoredHubSpec, &spokeSave, !hubOrigin); err != nil {
		return err
	}
	var statusSave DynamoGraphDeploymentStatus
	saveDGDAlphaOnlyStatus(&src.Status, &statusSave)

	scrubDGDInternalAnnotations(&converted.ObjectMeta)
	if !dgdAlphaSpecSaveIsZero(&spokeSave) || !dgdAlphaStatusSaveIsZero(&statusSave) {
		if err := saveDGDSpokeAnnotations(&spokeSave, &statusSave, &converted); err != nil {
			return err
		}
	}
	*dst = converted
	return nil
}

func Convert_v1beta1_DynamoGraphDeployment_To_v1alpha1_DynamoGraphDeployment(src *v1beta1.DynamoGraphDeployment, dst *DynamoGraphDeployment, s apiconversion.Scope) error {
	spokeOrigin := hasDGDSpokeAnnotations(&src.ObjectMeta)
	restoredSpokeSpec, restoredSpokeStatus, err := restoreDGDSpokeAnnotations(&src.ObjectMeta)
	if err != nil {
		return err
	}

	converted := DynamoGraphDeployment{TypeMeta: dst.TypeMeta}
	if err := autoConvert_v1beta1_DynamoGraphDeployment_To_v1alpha1_DynamoGraphDeployment(src, &converted, s); err != nil {
		return err
	}
	converted.ObjectMeta = *src.ObjectMeta.DeepCopy()

	var hubSave v1beta1.DynamoGraphDeploymentSpec
	if err := applyDGDToAlphaPreservation(&src.Spec, &converted.Spec, restoredSpokeSpec, &hubSave, !spokeOrigin); err != nil {
		return err
	}

	restoreDGDAlphaOnlySpecFromSaved(&converted.Spec, restoredSpokeSpec)
	restoreDGDAlphaOnlyStatusFromSaved(&converted.Status, restoredSpokeStatus)
	scrubDGDInternalAnnotations(&converted.ObjectMeta)
	if !dgdHubSpecSaveIsZero(&hubSave) {
		data, err := marshalDGDHubSpec(&hubSave)
		if err != nil {
			return fmt.Errorf("preserve DGD hub spec: %w", err)
		}
		setAnnOnObj(&converted.ObjectMeta, annDGDSpec, string(data))
	}
	*dst = converted
	return nil
}

// Convert_v1alpha1_DynamoGraphDeploymentSpec_To_v1beta1_DynamoGraphDeploymentSpec
// maps the v1alpha1 services map to the v1beta1 components list. The map key is
// the canonical component name.
func Convert_v1alpha1_DynamoGraphDeploymentSpec_To_v1beta1_DynamoGraphDeploymentSpec(in *DynamoGraphDeploymentSpec, out *v1beta1.DynamoGraphDeploymentSpec, s apiconversion.Scope) error {
	if err := autoConvert_v1alpha1_DynamoGraphDeploymentSpec_To_v1beta1_DynamoGraphDeploymentSpec(in, out, s); err != nil {
		return err
	}
	out.Env = in.Envs
	if len(in.Services) == 0 {
		return nil
	}
	names := sets.List(sets.KeySet(in.Services))
	out.Components = make([]v1beta1.DynamoComponentDeploymentSharedSpec, 0, len(names))
	for _, name := range names {
		service := in.Services[name]
		if service == nil {
			continue
		}
		var component v1beta1.DynamoComponentDeploymentSharedSpec
		if err := Convert_v1alpha1_DynamoComponentDeploymentSharedSpec_To_v1beta1_DynamoComponentDeploymentSharedSpec(service, &component, s); err != nil {
			return fmt.Errorf("component %q: %w", name, err)
		}
		component.ComponentName = name
		out.Components = append(out.Components, component)
	}
	return nil
}

// Convert_v1beta1_DynamoGraphDeploymentSpec_To_v1alpha1_DynamoGraphDeploymentSpec
// maps the v1beta1 components list to the v1alpha1 services map.
func Convert_v1beta1_DynamoGraphDeploymentSpec_To_v1alpha1_DynamoGraphDeploymentSpec(in *v1beta1.DynamoGraphDeploymentSpec, out *DynamoGraphDeploymentSpec, s apiconversion.Scope) error {
	if err := autoConvert_v1beta1_DynamoGraphDeploymentSpec_To_v1alpha1_DynamoGraphDeploymentSpec(in, out, s); err != nil {
		return err
	}
	out.Envs = in.Env
	if len(in.Components) == 0 {
		return nil
	}
	out.Services = make(map[string]*DynamoComponentDeploymentSharedSpec, len(in.Components))
	for i := range in.Components {
		component := &in.Components[i]
		// The API server normally rejects duplicates because components is a
		// list-map. Conversion is also callable on in-memory objects that bypass
		// CRD validation, so reject duplicates instead of silently overwriting.
		if _, duplicate := out.Services[component.ComponentName]; duplicate {
			return fmt.Errorf("duplicate component name %q in spec.components", component.ComponentName)
		}
		service := &DynamoComponentDeploymentSharedSpec{}
		if err := Convert_v1beta1_DynamoComponentDeploymentSharedSpec_To_v1alpha1_DynamoComponentDeploymentSharedSpec(component, service, s); err != nil {
			return fmt.Errorf("component %q: %w", component.ComponentName, err)
		}
		// ServiceName is redundant in v1alpha1 DGD; the services-map key is the
		// canonical name. Saved mismatches are restored at the object level.
		service.ServiceName = ""
		out.Services[component.ComponentName] = service
	}
	return nil
}

// Convert_v1alpha1_DynamoGraphDeploymentStatus_To_v1beta1_DynamoGraphDeploymentStatus
// converts common status fields and maps the Services status map to Components.
func Convert_v1alpha1_DynamoGraphDeploymentStatus_To_v1beta1_DynamoGraphDeploymentStatus(in *DynamoGraphDeploymentStatus, out *v1beta1.DynamoGraphDeploymentStatus, s apiconversion.Scope) error {
	if err := autoConvert_v1alpha1_DynamoGraphDeploymentStatus_To_v1beta1_DynamoGraphDeploymentStatus(in, out, s); err != nil {
		return err
	}
	if len(in.Services) > 0 {
		out.Components = make(map[string]v1beta1.ComponentReplicaStatus, len(in.Services))
		for name, service := range in.Services {
			var component v1beta1.ComponentReplicaStatus
			if err := Convert_v1alpha1_ServiceReplicaStatus_To_v1beta1_ComponentReplicaStatus(&service, &component, s); err != nil {
				return err
			}
			out.Components[name] = component
		}
	}
	return nil
}

// Convert_v1beta1_DynamoGraphDeploymentStatus_To_v1alpha1_DynamoGraphDeploymentStatus
// converts common status fields and maps the Components status map to Services.
func Convert_v1beta1_DynamoGraphDeploymentStatus_To_v1alpha1_DynamoGraphDeploymentStatus(in *v1beta1.DynamoGraphDeploymentStatus, out *DynamoGraphDeploymentStatus, s apiconversion.Scope) error {
	if err := autoConvert_v1beta1_DynamoGraphDeploymentStatus_To_v1alpha1_DynamoGraphDeploymentStatus(in, out, s); err != nil {
		return err
	}
	if len(in.Components) > 0 {
		out.Services = make(map[string]ServiceReplicaStatus, len(in.Components))
		for name, component := range in.Components {
			var service ServiceReplicaStatus
			if err := Convert_v1beta1_ComponentReplicaStatus_To_v1alpha1_ServiceReplicaStatus(&component, &service, s); err != nil {
				return err
			}
			out.Services[name] = service
		}
	}
	return nil
}

// Convert_v1alpha1_ServiceCheckpointStatus_To_v1beta1_ComponentCheckpointStatus
// converts the checkpoint status renamed from service to component terminology.
func Convert_v1alpha1_ServiceCheckpointStatus_To_v1beta1_ComponentCheckpointStatus(in *ServiceCheckpointStatus, out *v1beta1.ComponentCheckpointStatus, _ apiconversion.Scope) error {
	*out = v1beta1.ComponentCheckpointStatus{
		CheckpointName: in.CheckpointName,
		CheckpointID:   in.CheckpointID,
		IdentityHash:   in.IdentityHash,
		Ready:          in.Ready,
	}
	return nil
}

// Convert_v1beta1_ComponentCheckpointStatus_To_v1alpha1_ServiceCheckpointStatus
// converts the checkpoint status back to service terminology.
func Convert_v1beta1_ComponentCheckpointStatus_To_v1alpha1_ServiceCheckpointStatus(in *v1beta1.ComponentCheckpointStatus, out *ServiceCheckpointStatus, _ apiconversion.Scope) error {
	*out = ServiceCheckpointStatus{
		CheckpointName: in.CheckpointName,
		CheckpointID:   in.CheckpointID,
		IdentityHash:   in.IdentityHash,
		Ready:          in.Ready,
	}
	return nil
}

// Convert_v1alpha1_ServiceReplicaStatus_To_v1beta1_ComponentReplicaStatus maps
// service terminology to component terminology and folds the legacy singular
// ComponentName into ComponentNames when needed.
func Convert_v1alpha1_ServiceReplicaStatus_To_v1beta1_ComponentReplicaStatus(in *ServiceReplicaStatus, out *v1beta1.ComponentReplicaStatus, _ apiconversion.Scope) error {
	*out = v1beta1.ComponentReplicaStatus{
		ComponentKind:     v1beta1.ComponentKind(in.ComponentKind),
		ComponentNames:    componentNamesToHub(in),
		RuntimeNamespace:  in.RuntimeNamespace,
		Replicas:          in.Replicas,
		UpdatedReplicas:   in.UpdatedReplicas,
		ReadyReplicas:     in.ReadyReplicas,
		AvailableReplicas: in.AvailableReplicas,
	}
	return nil
}

// Convert_v1beta1_ComponentReplicaStatus_To_v1alpha1_ServiceReplicaStatus maps
// component terminology back to service terminology. The last component name
// becomes the legacy singular ComponentName.
func Convert_v1beta1_ComponentReplicaStatus_To_v1alpha1_ServiceReplicaStatus(in *v1beta1.ComponentReplicaStatus, out *ServiceReplicaStatus, _ apiconversion.Scope) error {
	componentNames := in.ComponentNames
	*out = ServiceReplicaStatus{
		ComponentKind:     ComponentKind(in.ComponentKind),
		ComponentNames:    componentNames,
		RuntimeNamespace:  in.RuntimeNamespace,
		Replicas:          in.Replicas,
		UpdatedReplicas:   in.UpdatedReplicas,
		ReadyReplicas:     in.ReadyReplicas,
		AvailableReplicas: in.AvailableReplicas,
	}
	if len(componentNames) > 0 {
		out.ComponentName = componentNames[len(componentNames)-1]
	}
	return nil
}

// Convert_v1alpha1_RollingUpdateStatus_To_v1beta1_RollingUpdateStatus converts
// common fields and renames UpdatedServices to UpdatedComponents.
func Convert_v1alpha1_RollingUpdateStatus_To_v1beta1_RollingUpdateStatus(in *RollingUpdateStatus, out *v1beta1.RollingUpdateStatus, s apiconversion.Scope) error {
	if err := autoConvert_v1alpha1_RollingUpdateStatus_To_v1beta1_RollingUpdateStatus(in, out, s); err != nil {
		return err
	}
	out.UpdatedComponents = in.UpdatedServices
	return nil
}

// Convert_v1beta1_RollingUpdateStatus_To_v1alpha1_RollingUpdateStatus converts
// common fields and renames UpdatedComponents to UpdatedServices.
func Convert_v1beta1_RollingUpdateStatus_To_v1alpha1_RollingUpdateStatus(in *v1beta1.RollingUpdateStatus, out *RollingUpdateStatus, s apiconversion.Scope) error {
	if err := autoConvert_v1beta1_RollingUpdateStatus_To_v1alpha1_RollingUpdateStatus(in, out, s); err != nil {
		return err
	}
	out.UpdatedServices = in.UpdatedComponents
	return nil
}

// Convert_v1alpha1_SpecTopologyConstraint_To_v1beta1_SpecTopologyConstraint
// converts common fields and renames TopologyProfile to ClusterTopologyName.
func Convert_v1alpha1_SpecTopologyConstraint_To_v1beta1_SpecTopologyConstraint(in *SpecTopologyConstraint, out *v1beta1.SpecTopologyConstraint, s apiconversion.Scope) error {
	if err := autoConvert_v1alpha1_SpecTopologyConstraint_To_v1beta1_SpecTopologyConstraint(in, out, s); err != nil {
		return err
	}
	out.ClusterTopologyName = in.TopologyProfile
	return nil
}

// Convert_v1beta1_SpecTopologyConstraint_To_v1alpha1_SpecTopologyConstraint
// converts common fields and renames ClusterTopologyName to TopologyProfile.
func Convert_v1beta1_SpecTopologyConstraint_To_v1alpha1_SpecTopologyConstraint(in *v1beta1.SpecTopologyConstraint, out *SpecTopologyConstraint, s apiconversion.Scope) error {
	if err := autoConvert_v1beta1_SpecTopologyConstraint_To_v1alpha1_SpecTopologyConstraint(in, out, s); err != nil {
		return err
	}
	out.TopologyProfile = in.ClusterTopologyName
	return nil
}
