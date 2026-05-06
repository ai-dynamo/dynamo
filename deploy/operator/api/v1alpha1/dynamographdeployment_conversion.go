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

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/conversion"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	annDGDSpec   = "nvidia.com/dgd-spec"
	annDGDStatus = "nvidia.com/dgd-status"
)

type dgdConversionContext struct {
	includeOriginSplits bool
	saveHubOrigin       bool
}

// ConvertTo converts this DynamoGraphDeployment (v1alpha1) into the hub
// version (v1beta1).
func (src *DynamoGraphDeployment) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeployment but got %T", dstRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()
	var restoredHubSpec *v1beta1.DynamoGraphDeploymentSpec
	if raw, ok := getAnnFromObj(&dst.ObjectMeta, annDGDSpec); ok && raw != "" {
		if spec, ok := restoreDGDHubSpec(raw); ok {
			restoredHubSpec = &spec
		}
	}
	hubOrigin := restoredHubSpec != nil
	scrubDGDInternalAnnotations(&dst.ObjectMeta)

	ctx := dgdConversionContext{
		includeOriginSplits: !hubOrigin,
	}
	var spokeSave DynamoGraphDeploymentSpec
	if err := convertDGDSpecToHub(&src.Spec, &dst.Spec, restoredHubSpec, &spokeSave, ctx); err != nil {
		return err
	}
	var statusSave DynamoGraphDeploymentStatus
	saveDGDAlphaOnlyStatus(&src.Status, &statusSave)

	convertDGDStatusToHub(&src.Status, &dst.Status, nil, nil, ctx)
	if !dgdAlphaSpecSaveIsZero(&spokeSave) || !dgdAlphaStatusSaveIsZero(&statusSave) {
		if err := saveDGDSpokeAnnotations(&spokeSave, &statusSave, dst); err != nil {
			return err
		}
	}
	return nil
}

func convertDGDSpecToHub(src *DynamoGraphDeploymentSpec, dst *v1beta1.DynamoGraphDeploymentSpec, restored *v1beta1.DynamoGraphDeploymentSpec, save *DynamoGraphDeploymentSpec, ctx dgdConversionContext) error {
	if src == nil || dst == nil {
		return nil
	}

	// Convert fields represented by both versions from the live source.
	dst.Annotations = src.Annotations
	dst.Labels = src.Labels
	dst.BackendFramework = src.BackendFramework

	if src.Restart != nil {
		dst.Restart = &v1beta1.Restart{}
		convertRestartToHub(src.Restart, dst.Restart, nil, nil, ctx)
	}
	if src.TopologyConstraint != nil {
		dst.TopologyConstraint = &v1beta1.SpecTopologyConstraint{
			ClusterTopologyName: src.TopologyConstraint.TopologyProfile,
			PackDomain:          v1beta1.TopologyDomain(src.TopologyConstraint.PackDomain),
		}
	}
	dst.Env = src.Envs
	if save != nil && len(src.PVCs) > 0 {
		save.PVCs = slices.Clone(src.PVCs)
	}

	// Restore target-only component leaves from the preserved hub payload.
	restoredHubComponents := restoredDGDHubComponentsByName(restored)

	// Components: v1alpha1 map -> v1beta1 list. Prefer the preserved hub list
	// order when it carried non-sorted order information; otherwise sort by
	// name for deterministic alpha-first output.
	if len(src.Services) > 0 {
		names := dgdServiceNamesInEmissionOrder(src.Services, restored)
		dst.Components = make([]v1beta1.DynamoComponentDeploymentSharedSpec, 0, len(names))
		for _, name := range names {
			compSrc := src.Services[name]
			if compSrc == nil {
				if save != nil {
					if save.Services == nil {
						save.Services = map[string]*DynamoComponentDeploymentSharedSpec{}
					}
					save.Services[name] = nil
				}
				continue
			}
			restoredShared := restoredHubComponents[name]
			var compDst v1beta1.DynamoComponentDeploymentSharedSpec
			var compSave *DynamoComponentDeploymentSharedSpec
			if save != nil {
				compSave = &DynamoComponentDeploymentSharedSpec{}
			}
			sharedCtx := sharedSpecConversionContext{
				includeOriginSplits: ctx.includeOriginSplits,
				podTemplateOrigin:   restoredShared != nil && restoredShared.PodTemplate != nil,
			}
			if err := convertSharedSpecToHub(compSrc, &compDst, restoredShared, compSave, sharedCtx); err != nil {
				return fmt.Errorf("component %q: %w", name, err)
			}
			// In v1alpha1 DGD, the services-map key is the canonical name and
			// any value in compSrc.ServiceName is treated as legacy/dead by
			// the reconciler (graph.go materialises DCDs with ServiceName =
			// map key). Force the v1beta1 ComponentName to the map key so the
			// +listMapKey=name invariant (and the round-trip identity) hold, and
			// save the (now redundant) v1alpha1 ServiceName so a mismatched
			// value still round-trips.
			compDst.ComponentName = name
			if save != nil && compSrc.ServiceName != "" && compSrc.ServiceName != name {
				compSave.ServiceName = compSrc.ServiceName
			}
			if save != nil && !sharedAlphaSpecSaveIsZero(compSave) {
				if save.Services == nil {
					save.Services = map[string]*DynamoComponentDeploymentSharedSpec{}
				}
				save.Services[name] = compSave
			}
			dst.Components = append(dst.Components, compDst)
		}
	}

	return nil
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
		if shouldRestoreSavedComponentName(&dstSvc, &savedSvc) {
			dstSvc.ComponentName = savedSvc.ComponentName
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

func shouldRestoreSavedComponentName(dst, saved *ServiceReplicaStatus) bool {
	return serviceStatusComponentNameNeedsPreservation(saved) &&
		dst != nil &&
		dst.ComponentName != saved.ComponentName
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

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()

	spokeOrigin := hasDGDSpokeAnnotations(&dst.ObjectMeta)
	restoredSpokeSpec, restoredSpokeStatus, err := restoreDGDSpokeAnnotations(&dst.ObjectMeta)
	if err != nil {
		return err
	}
	scrubDGDInternalAnnotations(&dst.ObjectMeta)

	ctx := dgdConversionContext{saveHubOrigin: !spokeOrigin}
	var hubSave v1beta1.DynamoGraphDeploymentSpec
	if err := convertDGDSpecFromHub(&src.Spec, &dst.Spec, restoredSpokeSpec, &hubSave, ctx); err != nil {
		return err
	}

	convertDGDStatusFromHub(&src.Status, &dst.Status, nil, nil, ctx)
	restoreDGDAlphaOnlySpecFromSaved(&dst.Spec, restoredSpokeSpec)
	restoreDGDAlphaOnlyStatusFromSaved(&dst.Status, restoredSpokeStatus)
	if !dgdHubSpecSaveIsZero(&hubSave) {
		data, err := marshalDGDHubSpec(&hubSave)
		if err != nil {
			return fmt.Errorf("preserve DGD hub spec: %w", err)
		}
		setAnnOnObj(&dst.ObjectMeta, annDGDSpec, string(data))
	}
	return nil
}

func convertDGDSpecFromHub(src *v1beta1.DynamoGraphDeploymentSpec, dst *DynamoGraphDeploymentSpec, restored *DynamoGraphDeploymentSpec, save *v1beta1.DynamoGraphDeploymentSpec, ctx dgdConversionContext) error {
	if src == nil || dst == nil {
		return nil
	}

	// Convert fields represented by both versions from the live source.
	dst.Annotations = src.Annotations
	dst.Labels = src.Labels
	dst.BackendFramework = src.BackendFramework

	if src.Restart != nil {
		dst.Restart = &Restart{}
		convertRestartFromHub(src.Restart, dst.Restart, nil, nil, ctx)
	}
	if src.TopologyConstraint != nil {
		dst.TopologyConstraint = &SpecTopologyConstraint{
			TopologyProfile: src.TopologyConstraint.ClusterTopologyName,
			PackDomain:      TopologyDomain(src.TopologyConstraint.PackDomain),
		}
	}
	dst.Envs = src.Env

	if len(src.Components) > 0 {
		preserveComponentOrder := dgdComponentOrderNeedsPreservation(src.Components)
		dst.Services = make(map[string]*DynamoComponentDeploymentSharedSpec, len(src.Components))
		for i := range src.Components {
			compSrc := &src.Components[i]
			// v1beta1 declares +listType=map +listMapKey=name so
			// the API server normally rejects duplicates, but the
			// conversion path is also reached from in-memory unit-test
			// fixtures and other code paths that bypass CRD validation.
			// Surface duplicates here as a hard error rather than
			// silently overwriting the earlier entry on map insertion.
			if _, dup := dst.Services[compSrc.ComponentName]; dup {
				return fmt.Errorf("duplicate component name %q in spec.components", compSrc.ComponentName)
			}
			compDst := &DynamoComponentDeploymentSharedSpec{}
			var preservedShared *DynamoComponentDeploymentSharedSpec
			if restored != nil && restored.Services != nil {
				preservedShared = restored.Services[compSrc.ComponentName]
			}
			var compSave *v1beta1.DynamoComponentDeploymentSharedSpec
			if save != nil {
				compSave = &v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: compSrc.ComponentName,
				}
			}
			sharedCtx := sharedSpecConversionContext{}
			if err := convertSharedSpecFromHub(compSrc, compDst, preservedShared, compSave, sharedCtx); err != nil {
				return fmt.Errorf("component %q: %w", compSrc.ComponentName, err)
			}
			// In v1alpha1 the services-map key is the canonical name; the
			// per-entry ServiceName field is redundant. Keep it empty for
			// v1beta1-first inputs; restore saved mismatches after the full
			// spec conversion.
			compDst.ServiceName = ""
			dst.Services[compSrc.ComponentName] = compDst
			if save != nil && (preserveComponentOrder || !dgdHubComponentSaveIsZero(compSave) || ctx.saveHubOrigin && dgdHubComponentOriginSaveNeeded(compSrc)) {
				save.Components = append(save.Components, *compSave)
			}
		}
	}

	return nil
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

// convertRestartToHub / convertRestartFromHub handle the Restart struct pair,
// which is structurally identical across versions but uses version-specific types.
//
//nolint:unparam // Keep the structural conversion signature; this leaf has no preserved fields.
func convertRestartToHub(src *Restart, dst *v1beta1.Restart, restored *v1beta1.Restart, save *Restart, ctx dgdConversionContext) {
	_, _, _ = restored, save, ctx

	if src == nil || dst == nil {
		return
	}
	*dst = v1beta1.Restart{ID: src.ID}
	if src.Strategy != nil {
		dst.Strategy = &v1beta1.RestartStrategy{
			Type:  v1beta1.RestartStrategyType(src.Strategy.Type),
			Order: slices.Clone(src.Strategy.Order),
		}
	}
}

//nolint:unparam // Keep the structural conversion signature; this leaf has no preserved fields.
func convertRestartFromHub(src *v1beta1.Restart, dst *Restart, restored *Restart, save *v1beta1.Restart, ctx dgdConversionContext) {
	_, _, _ = restored, save, ctx

	if src == nil || dst == nil {
		return
	}
	*dst = Restart{ID: src.ID}
	if src.Strategy != nil {
		dst.Strategy = &RestartStrategy{
			Type:  RestartStrategyType(src.Strategy.Type),
			Order: slices.Clone(src.Strategy.Order),
		}
	}
}

// convertDGDStatusToHub / convertDGDStatusFromHub copy the status sub-struct.
// Status fields are structurally identical; version types differ so each field
// is copied explicitly.
//
//nolint:unparam // Keep the structural conversion signature; this leaf has no preserved fields.
func convertDGDStatusToHub(src *DynamoGraphDeploymentStatus, dst *v1beta1.DynamoGraphDeploymentStatus, restored *v1beta1.DynamoGraphDeploymentStatus, save *DynamoGraphDeploymentStatus, ctx dgdConversionContext) {
	_, _, _ = restored, save, ctx

	dst.ObservedGeneration = src.ObservedGeneration
	dst.State = v1beta1.DGDState(src.State)
	if len(src.Conditions) > 0 {
		dst.Conditions = make([]metav1.Condition, 0, len(src.Conditions))
		for _, c := range src.Conditions {
			dst.Conditions = append(dst.Conditions, *c.DeepCopy())
		}
	}
	if len(src.Services) > 0 {
		dst.Components = make(map[string]v1beta1.ComponentReplicaStatus, len(src.Services))
		for k, v := range src.Services {
			var component v1beta1.ComponentReplicaStatus
			convertReplicaStatusToHub(&v, &component, nil, nil, replicaStatusConversionContext{})
			dst.Components[k] = component
		}
	}
	if src.Restart != nil {
		dst.Restart = &v1beta1.RestartStatus{
			ObservedID: src.Restart.ObservedID,
			Phase:      v1beta1.RestartPhase(src.Restart.Phase),
			InProgress: slices.Clone(src.Restart.InProgress),
		}
	}
	if len(src.Checkpoints) > 0 {
		dst.Checkpoints = make(map[string]v1beta1.ComponentCheckpointStatus, len(src.Checkpoints))
		for k, v := range src.Checkpoints {
			dst.Checkpoints[k] = v1beta1.ComponentCheckpointStatus{
				CheckpointName: v.CheckpointName,
				IdentityHash:   v.IdentityHash,
				Ready:          v.Ready,
			}
		}
	}
	if src.RollingUpdate != nil {
		dst.RollingUpdate = &v1beta1.RollingUpdateStatus{
			Phase:             v1beta1.RollingUpdatePhase(src.RollingUpdate.Phase),
			StartTime:         src.RollingUpdate.StartTime.DeepCopy(),
			EndTime:           src.RollingUpdate.EndTime.DeepCopy(),
			UpdatedComponents: slices.Clone(src.RollingUpdate.UpdatedServices),
		}
	}
}

//nolint:unparam // Keep the structural conversion signature; this leaf has no preserved fields.
func convertDGDStatusFromHub(src *v1beta1.DynamoGraphDeploymentStatus, dst *DynamoGraphDeploymentStatus, restored *DynamoGraphDeploymentStatus, save *v1beta1.DynamoGraphDeploymentStatus, ctx dgdConversionContext) {
	_, _, _ = restored, save, ctx

	dst.ObservedGeneration = src.ObservedGeneration
	dst.State = DGDState(src.State)
	if len(src.Conditions) > 0 {
		dst.Conditions = make([]metav1.Condition, 0, len(src.Conditions))
		for _, c := range src.Conditions {
			dst.Conditions = append(dst.Conditions, *c.DeepCopy())
		}
	}
	if len(src.Components) > 0 {
		dst.Services = make(map[string]ServiceReplicaStatus, len(src.Components))
		for k, v := range src.Components {
			var service ServiceReplicaStatus
			convertReplicaStatusFromHub(&v, &service, nil, nil, replicaStatusConversionContext{})
			dst.Services[k] = service
		}
	}
	if src.Restart != nil {
		dst.Restart = &RestartStatus{
			ObservedID: src.Restart.ObservedID,
			Phase:      RestartPhase(src.Restart.Phase),
			InProgress: slices.Clone(src.Restart.InProgress),
		}
	}
	if len(src.Checkpoints) > 0 {
		dst.Checkpoints = make(map[string]ServiceCheckpointStatus, len(src.Checkpoints))
		for k, v := range src.Checkpoints {
			dst.Checkpoints[k] = ServiceCheckpointStatus{
				CheckpointName: v.CheckpointName,
				IdentityHash:   v.IdentityHash,
				Ready:          v.Ready,
			}
		}
	}
	if src.RollingUpdate != nil {
		dst.RollingUpdate = &RollingUpdateStatus{
			Phase:           RollingUpdatePhase(src.RollingUpdate.Phase),
			StartTime:       src.RollingUpdate.StartTime.DeepCopy(),
			EndTime:         src.RollingUpdate.EndTime.DeepCopy(),
			UpdatedServices: slices.Clone(src.RollingUpdate.UpdatedComponents),
		}
	}
}

type replicaStatusConversionContext struct{}

//nolint:unparam // Keep the structural conversion signature; this leaf has no preserved fields.
func convertReplicaStatusToHub(src *ServiceReplicaStatus, dst *v1beta1.ComponentReplicaStatus, restored *v1beta1.ComponentReplicaStatus, save *ServiceReplicaStatus, ctx replicaStatusConversionContext) {
	_, _, _ = restored, save, ctx

	if src == nil || dst == nil {
		return
	}
	*dst = v1beta1.ComponentReplicaStatus{
		ComponentKind:   v1beta1.ComponentKind(src.ComponentKind),
		ComponentNames:  slices.Clone(src.ComponentNames),
		Replicas:        src.Replicas,
		UpdatedReplicas: src.UpdatedReplicas,
	}
	if src.ReadyReplicas != nil {
		dst.ReadyReplicas = ptr.To(*src.ReadyReplicas)
	}
	if src.AvailableReplicas != nil {
		dst.AvailableReplicas = ptr.To(*src.AvailableReplicas)
	}
}

//nolint:unparam // Keep the structural conversion signature; this leaf has no preserved fields.
func convertReplicaStatusFromHub(src *v1beta1.ComponentReplicaStatus, dst *ServiceReplicaStatus, restored *ServiceReplicaStatus, save *v1beta1.ComponentReplicaStatus, ctx replicaStatusConversionContext) {
	_, _, _ = restored, save, ctx

	if src == nil || dst == nil {
		return
	}
	componentNames := slices.Clone(src.ComponentNames)

	*dst = ServiceReplicaStatus{
		ComponentKind:   ComponentKind(src.ComponentKind),
		ComponentNames:  componentNames,
		Replicas:        src.Replicas,
		UpdatedReplicas: src.UpdatedReplicas,
	}
	if len(componentNames) > 0 {
		dst.ComponentName = componentNames[len(componentNames)-1]
	}
	if src.ReadyReplicas != nil {
		dst.ReadyReplicas = ptr.To(*src.ReadyReplicas)
	}
	if src.AvailableReplicas != nil {
		dst.AvailableReplicas = ptr.To(*src.AvailableReplicas)
	}
}
