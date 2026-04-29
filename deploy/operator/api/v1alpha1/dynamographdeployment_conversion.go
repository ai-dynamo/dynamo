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
	"maps"
	"slices"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/conversion"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	annDGDNilServices = "nvidia.com/dgd-nil-services"
	annDGDHubSpec     = "nvidia.com/dgd-hub-spec"
	annDGDSpokeSpec   = "nvidia.com/dgd-spoke-spec"
	annDGDSpokeStatus = "nvidia.com/dgd-spoke-status"
	annDGDSpokeHub    = "nvidia.com/dgd-spoke-hub"
	annDGDHubOrigin   = "nvidia.com/dgd-hub-origin"
)

type dgdConversionContext struct {
	carrierForComponent func(componentName string) *annCarrier
	sourceCarrier       func(componentName string) *annCarrier
	sourceUnmodified    bool
	includeOriginSplits bool
}

func (ctx dgdConversionContext) carrier(componentName string) *annCarrier {
	if ctx.carrierForComponent == nil {
		return nil
	}
	return ctx.carrierForComponent(componentName)
}

// ConvertTo converts this DynamoGraphDeployment (v1alpha1) into the hub
// version (v1beta1).
func (src *DynamoGraphDeployment) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeployment but got %T", dstRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()
	var preservedHubSpec *v1beta1.DynamoGraphDeploymentSpec
	if raw, ok := getAnnFromObj(&dst.ObjectMeta, annDGDHubSpec); ok && raw != "" {
		if spec, ok := restoreDGDHubSpec(raw); ok {
			preservedHubSpec = &spec
			delAnnFromObj(&dst.ObjectMeta, annDGDHubSpec)
			scrubDGDInternalAnnotations(&dst.ObjectMeta)
		}
	}
	hubOrigin := preservedHubSpec != nil || dst.ObjectMeta.Annotations[annDGDHubOrigin] == annotationTrue
	delAnnFromObj(&dst.ObjectMeta, annDGDHubOrigin)
	if hubOrigin {
		scrubDGDInternalAnnotations(&dst.ObjectMeta)
	}

	// DGD.spec.pvcs has no v1beta1 equivalent -- users now declare PVC volumes
	// directly on podTemplate. Stash as an annotation.
	if len(src.Spec.PVCs) > 0 {
		setJSONAnnOnObj(&dst.ObjectMeta, annDGDPVCs, src.Spec.PVCs)
	}
	ctx := dgdConversionContext{
		includeOriginSplits: !hubOrigin,
		carrierForComponent: func(componentName string) *annCarrier {
			return newDGDComponentCarrier(&dst.ObjectMeta, componentName)
		},
	}
	var spokeSave DynamoGraphDeploymentSpec
	if err := convert_v1alpha1_DynamoGraphDeploymentSpec_To_v1beta1_DynamoGraphDeploymentSpec(&src.Spec, &dst.Spec, preservedHubSpec, &spokeSave, ctx); err != nil {
		return err
	}
	var statusSave DynamoGraphDeploymentStatus
	saveDGDAlphaOnlyStatus(&src.Status, &statusSave)
	preserveSpoke := !hubOrigin ||
		!dgdAlphaSpecSaveIsZero(&spokeSave) ||
		!dgdAlphaStatusSaveIsZero(&statusSave)

	convertDGDStatusTo(&src.Status, &dst.Status)
	if preserveSpoke {
		preserveDGDSpoke(&spokeSave, &statusSave, dst)
		preserveDGDSpokeHub(dst)
	}
	return nil
}

func convert_v1alpha1_DynamoGraphDeploymentSpec_To_v1beta1_DynamoGraphDeploymentSpec(src *DynamoGraphDeploymentSpec, dst *v1beta1.DynamoGraphDeploymentSpec, restored *v1beta1.DynamoGraphDeploymentSpec, save *DynamoGraphDeploymentSpec, ctx dgdConversionContext) error {
	if src == nil || dst == nil {
		return nil
	}

	// Convert fields represented by both versions from the live source.
	dst.Annotations = maps.Clone(src.Annotations)
	dst.Labels = maps.Clone(src.Labels)
	dst.BackendFramework = src.BackendFramework

	if src.Restart != nil {
		dst.Restart = convertRestartTo(src.Restart)
	}
	if src.TopologyConstraint != nil {
		dst.TopologyConstraint = &v1beta1.SpecTopologyConstraint{
			ClusterTopologyName: src.TopologyConstraint.TopologyProfile,
			PackDomain:          v1beta1.TopologyDomain(src.TopologyConstraint.PackDomain),
		}
	}
	if len(src.Envs) > 0 {
		dst.Env = slices.Clone(src.Envs)
	}
	if save != nil && len(src.PVCs) > 0 {
		save.PVCs = slices.Clone(src.PVCs)
	}

	// Restore target-only component leaves from the preserved hub snapshot.
	preservedHubComponents := preservedDGDHubComponentsByName(restored)

	// Components: v1alpha1 map -> v1beta1 list. Sort by name for a deterministic
	// emission order; the unordered map cannot faithfully represent the
	// v1beta1 list order, so round-trip V1 -> A1 -> V2 may reorder entries.
	if len(src.Services) > 0 {
		names := slices.Sorted(maps.Keys(src.Services))
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
			carrier := ctx.carrier(name)
			if carrier == nil {
				return fmt.Errorf("component %q: missing annotation carrier", name)
			}
			preservedShared := preservedHubComponents[name]
			if preservedShared != nil && preservedShared.PodTemplate != nil {
				carrier.set(suffixPodTemplateOrig, annotationTrue)
			}
			var compDst v1beta1.DynamoComponentDeploymentSharedSpec
			var compSave *DynamoComponentDeploymentSharedSpec
			if save != nil {
				compSave = &DynamoComponentDeploymentSharedSpec{}
			}
			sharedCtx := sharedSpecConversionContext{
				carrier:             carrier,
				includeOriginSplits: ctx.includeOriginSplits,
			}
			if err := convert_v1alpha1_DynamoComponentDeploymentSharedSpec_To_v1beta1_DynamoComponentDeploymentSharedSpec(compSrc, &compDst, preservedShared, compSave, sharedCtx); err != nil {
				return fmt.Errorf("component %q: %w", name, err)
			}
			// In v1alpha1 DGD, the services-map key is the canonical name and
			// any value in compSrc.ServiceName is treated as legacy/dead by
			// the reconciler (graph.go materialises DCDs with ServiceName =
			// map key). Force the v1beta1 ComponentName to the map key so the
			// +listMapKey=name invariant (and the round-trip identity) hold, and
			// stash the (now redundant) v1alpha1 ServiceName in an origin
			// annotation so a mismatched value still round-trips.
			compDst.ComponentName = name
			if compSrc.ServiceName != "" && compSrc.ServiceName != name {
				carrier.set(suffixServiceName, compSrc.ServiceName)
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

func preserveDGDSpoke(specSave *DynamoGraphDeploymentSpec, statusSave *DynamoGraphDeploymentStatus, dst *v1beta1.DynamoGraphDeployment) {
	if !dgdAlphaSpecSaveIsZero(specSave) {
		data, err := marshalDGDSpokeSpec(specSave)
		if err == nil {
			setAnnOnObj(&dst.ObjectMeta, annDGDSpokeSpec, string(data))
		}
	}
	if !dgdAlphaStatusSaveIsZero(statusSave) {
		setJSONAnnOnObj(&dst.ObjectMeta, annDGDSpokeStatus, statusSave)
	}
}

func preservedDGDHubComponentsByName(preserved *v1beta1.DynamoGraphDeploymentSpec) map[string]*v1beta1.DynamoComponentDeploymentSharedSpec {
	if preserved == nil || len(preserved.Components) == 0 {
		return nil
	}
	out := make(map[string]*v1beta1.DynamoComponentDeploymentSharedSpec, len(preserved.Components))
	for i := range preserved.Components {
		out[preserved.Components[i].ComponentName] = &preserved.Components[i]
	}
	return out
}

func preserveDGDSpokeHub(dst *v1beta1.DynamoGraphDeployment) {
	spec, err := marshalDGDHubSpec(&dst.Spec)
	if err != nil {
		return
	}
	setHubSnapshotAnn(&dst.ObjectMeta, annDGDSpokeHub, spec, dst.Status)
}

func dgdSpokeHubUnmodified(src *v1beta1.DynamoGraphDeployment) bool {
	spec, err := marshalDGDHubSpec(&src.Spec)
	if err != nil {
		return false
	}
	return hubSnapshotAnnMatches(&src.ObjectMeta, annDGDSpokeHub, spec, src.Status)
}

func fillDGDSpokeFromPreserved(dstSpec *DynamoGraphDeploymentSpec, dstStatus *DynamoGraphDeploymentStatus, preservedSpec *DynamoGraphDeploymentSpec, preservedStatus *DynamoGraphDeploymentStatus) {
	if preservedSpec != nil {
		if len(dstSpec.PVCs) == 0 {
			dstSpec.PVCs = slices.Clone(preservedSpec.PVCs)
		}
		for name, preservedComp := range preservedSpec.Services {
			if preservedComp != nil {
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
			if dstComp == nil || preservedSpec.Services == nil {
				continue
			}
			preservedComp := preservedSpec.Services[name]
			if preservedComp == nil {
				continue
			}
			if dstComp.ServiceName == "" && preservedComp.ServiceName != "" && preservedComp.ServiceName != name {
				dstComp.ServiceName = preservedComp.ServiceName
			}
		}
	}
	if preservedStatus != nil {
		for name, dstSvc := range dstStatus.Services {
			preservedSvc, ok := preservedStatus.Services[name]
			if !ok {
				continue
			}
			if shouldRestorePreservedComponentName(&dstSvc, &preservedSvc) {
				dstSvc.ComponentName = preservedSvc.ComponentName
			}
			dstStatus.Services[name] = dstSvc
		}
	}
}

func dgdAlphaSpecSaveIsZero(save *DynamoGraphDeploymentSpec) bool {
	return save == nil ||
		len(save.PVCs) == 0 &&
			len(save.Services) == 0
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
	return save == nil || len(save.Services) == 0
}

func dgdHubSpecSaveIsZero(save *v1beta1.DynamoGraphDeploymentSpec) bool {
	return save == nil || len(save.Components) == 0
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

func shouldRestorePreservedComponentName(dst, preserved *ServiceReplicaStatus) bool {
	return serviceStatusComponentNameNeedsPreservation(preserved) &&
		dst != nil &&
		dst.ComponentName != preserved.ComponentName
}

func decodeDGDSpokePreserved(obj metav1.Object) (*DynamoGraphDeploymentSpec, *DynamoGraphDeploymentStatus) {
	var preservedSpokeSpec *DynamoGraphDeploymentSpec
	var preservedSpokeStatus *DynamoGraphDeploymentStatus
	if raw, ok := getAnnFromObj(obj, annDGDSpokeSpec); ok && raw != "" {
		if spec, ok := restoreDGDSpokeSpec(raw); ok {
			preservedSpokeSpec = &spec
		}
	}
	if status, ok := getJSONAnnFromObj[DynamoGraphDeploymentStatus](obj, annDGDSpokeStatus); ok {
		preservedSpokeStatus = &status
	}
	return preservedSpokeSpec, preservedSpokeStatus
}

// ConvertFrom converts from the hub (v1beta1) DynamoGraphDeployment into this
// v1alpha1 instance.
func (dst *DynamoGraphDeployment) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeployment but got %T", srcRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()

	preservedSpokeSpec, preservedSpokeStatus := decodeDGDSpokePreserved(&dst.ObjectMeta)
	_, hadSpokeHubSnapshot := getAnnFromObj(&dst.ObjectMeta, annDGDSpokeHub)
	spokeHubUnmodified := dgdSpokeHubUnmodified(src)

	if _, ok := getAnnFromObj(&dst.ObjectMeta, annDGDPVCs); ok {
		if pvcs, ok := getJSONAnnFromObj[[]PVC](&dst.ObjectMeta, annDGDPVCs); ok {
			dst.Spec.PVCs = pvcs
		}
		delAnnFromObj(&dst.ObjectMeta, annDGDPVCs)
	}

	ctx := dgdConversionContext{
		sourceUnmodified: spokeHubUnmodified,
		carrierForComponent: func(componentName string) *annCarrier {
			return newDGDComponentCarrier(&dst.ObjectMeta, componentName)
		},
		sourceCarrier: func(componentName string) *annCarrier {
			return newDGDComponentCarrier(&src.ObjectMeta, componentName)
		},
	}
	var hubSave v1beta1.DynamoGraphDeploymentSpec
	if err := convert_v1beta1_DynamoGraphDeploymentSpec_To_v1alpha1_DynamoGraphDeploymentSpec(&src.Spec, &dst.Spec, preservedSpokeSpec, &hubSave, ctx); err != nil {
		return err
	}
	if _, ok := getAnnFromObj(&dst.ObjectMeta, annDGDNilServices); ok {
		if nilServices, ok := getJSONAnnFromObj[[]string](&dst.ObjectMeta, annDGDNilServices); ok {
			if dst.Spec.Services == nil {
				dst.Spec.Services = make(map[string]*DynamoComponentDeploymentSharedSpec, len(nilServices))
			}
			for _, name := range nilServices {
				if _, exists := dst.Spec.Services[name]; !exists {
					dst.Spec.Services[name] = nil
				}
			}
		}
		delAnnFromObj(&dst.ObjectMeta, annDGDNilServices)
	}

	convertDGDStatusFrom(&src.Status, &dst.Status)
	fillDGDSpokeFromPreserved(&dst.Spec, &dst.Status, preservedSpokeSpec, preservedSpokeStatus)
	scrubStaleDGDAnnotations(&dst.ObjectMeta, dst.Spec.Services)
	if preservedSpokeSpec != nil || preservedSpokeStatus != nil || hadSpokeHubSnapshot {
		delAnnFromObj(&dst.ObjectMeta, annDGDSpokeSpec)
		delAnnFromObj(&dst.ObjectMeta, annDGDSpokeStatus)
		delAnnFromObj(&dst.ObjectMeta, annDGDSpokeHub)
	}
	if !dgdHubSpecSaveIsZero(&hubSave) {
		if data, err := marshalDGDHubSpec(&hubSave); err == nil {
			setAnnOnObj(&dst.ObjectMeta, annDGDHubSpec, string(data))
		}
	} else if !hasDGDInternalAnnotations(src.ObjectMeta.Annotations) {
		setAnnOnObj(&dst.ObjectMeta, annDGDHubOrigin, annotationTrue)
	}
	return nil
}

func convert_v1beta1_DynamoGraphDeploymentSpec_To_v1alpha1_DynamoGraphDeploymentSpec(src *v1beta1.DynamoGraphDeploymentSpec, dst *DynamoGraphDeploymentSpec, restored *DynamoGraphDeploymentSpec, save *v1beta1.DynamoGraphDeploymentSpec, ctx dgdConversionContext) error {
	if src == nil || dst == nil {
		return nil
	}

	// Convert fields represented by both versions from the live source.
	dst.Annotations = maps.Clone(src.Annotations)
	dst.Labels = maps.Clone(src.Labels)
	dst.BackendFramework = src.BackendFramework

	if src.Restart != nil {
		dst.Restart = convertRestartFrom(src.Restart)
	}
	if src.TopologyConstraint != nil {
		dst.TopologyConstraint = &SpecTopologyConstraint{
			TopologyProfile: src.TopologyConstraint.ClusterTopologyName,
			PackDomain:      TopologyDomain(src.TopologyConstraint.PackDomain),
		}
	}
	if len(src.Env) > 0 {
		dst.Envs = slices.Clone(src.Env)
	}

	if len(src.Components) > 0 {
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
			carrier := ctx.carrier(compSrc.ComponentName)
			if carrier == nil {
				return fmt.Errorf("component %q: missing annotation carrier", compSrc.ComponentName)
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
			sharedCtx := sharedSpecConversionContext{
				carrier:          carrier,
				sourceUnmodified: ctx.sourceUnmodified,
			}
			if ctx.sourceCarrier != nil {
				sharedCtx.sourceCarrier = ctx.sourceCarrier(compSrc.ComponentName)
			}
			if err := convert_v1beta1_DynamoComponentDeploymentSharedSpec_To_v1alpha1_DynamoComponentDeploymentSharedSpec(compSrc, compDst, preservedShared, compSave, sharedCtx); err != nil {
				return fmt.Errorf("component %q: %w", compSrc.ComponentName, err)
			}
			// In v1alpha1 the services-map key is the canonical name; the
			// per-entry ServiceName field is redundant. Restore a non-matching
			// legacy ServiceName from the origin annotation if present;
			// otherwise leave it empty so v1beta1-first inputs round-trip
			// without spurious values.
			if v, ok := carrier.get(suffixServiceName); ok {
				compDst.ServiceName = v
				carrier.del(suffixServiceName)
			} else {
				compDst.ServiceName = ""
			}
			dst.Services[compSrc.ComponentName] = compDst
			if save != nil && !dgdHubComponentSaveIsZero(compSave) {
				save.Components = append(save.Components, *compSave)
			}
		}
	}

	return nil
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

func hasDGDInternalAnnotations(annotations map[string]string) bool {
	for key := range annotations {
		if key == annDGDPVCs ||
			key == annDGDNilServices ||
			key == annDGDHubSpec ||
			key == annDGDSpokeSpec ||
			key == annDGDSpokeStatus ||
			key == annDGDSpokeHub ||
			strings.HasPrefix(key, annDGDCompPrefix) {
			return true
		}
	}
	return false
}

func scrubDGDInternalAnnotations(obj metav1.Object) {
	for _, key := range []string{
		annDGDPVCs,
		annDGDNilServices,
		annDGDHubSpec,
		annDGDSpokeSpec,
		annDGDSpokeStatus,
		annDGDSpokeHub,
		annDGDHubOrigin,
	} {
		delAnnFromObj(obj, key)
	}
	scrubAnnotationsByPrefix(obj, annDGDCompPrefix)
}

// convertRestartTo / convertRestartFrom handle the Restart struct pair, which
// is structurally identical across versions but uses version-specific types.
func convertRestartTo(src *Restart) *v1beta1.Restart {
	out := &v1beta1.Restart{ID: src.ID}
	if src.Strategy != nil {
		out.Strategy = &v1beta1.RestartStrategy{
			Type:  v1beta1.RestartStrategyType(src.Strategy.Type),
			Order: slices.Clone(src.Strategy.Order),
		}
	}
	return out
}

func convertRestartFrom(src *v1beta1.Restart) *Restart {
	out := &Restart{ID: src.ID}
	if src.Strategy != nil {
		out.Strategy = &RestartStrategy{
			Type:  RestartStrategyType(src.Strategy.Type),
			Order: slices.Clone(src.Strategy.Order),
		}
	}
	return out
}

// convertDGDStatusTo / convertDGDStatusFrom copy the status sub-struct.
// Status fields are structurally identical; version types differ so each field
// is copied explicitly.
func convertDGDStatusTo(src *DynamoGraphDeploymentStatus, dst *v1beta1.DynamoGraphDeploymentStatus) {
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
			dst.Components[k] = *convertReplicaStatusTo(&v)
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

func convertDGDStatusFrom(src *v1beta1.DynamoGraphDeploymentStatus, dst *DynamoGraphDeploymentStatus) {
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
			dst.Services[k] = *convertReplicaStatusFrom(&v)
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

func convertReplicaStatusTo(src *ServiceReplicaStatus) *v1beta1.ComponentReplicaStatus {
	out := &v1beta1.ComponentReplicaStatus{
		ComponentKind:   v1beta1.ComponentKind(src.ComponentKind),
		ComponentNames:  slices.Clone(src.ComponentNames),
		Replicas:        src.Replicas,
		UpdatedReplicas: src.UpdatedReplicas,
	}
	if src.ReadyReplicas != nil {
		out.ReadyReplicas = ptr.To(*src.ReadyReplicas)
	}
	if src.AvailableReplicas != nil {
		out.AvailableReplicas = ptr.To(*src.AvailableReplicas)
	}
	return out
}

func convertReplicaStatusFrom(src *v1beta1.ComponentReplicaStatus) *ServiceReplicaStatus {
	componentNames := slices.Clone(src.ComponentNames)

	out := &ServiceReplicaStatus{
		ComponentKind:   ComponentKind(src.ComponentKind),
		ComponentNames:  componentNames,
		Replicas:        src.Replicas,
		UpdatedReplicas: src.UpdatedReplicas,
	}
	if len(componentNames) > 0 {
		out.ComponentName = componentNames[len(componentNames)-1]
	}
	if src.ReadyReplicas != nil {
		out.ReadyReplicas = ptr.To(*src.ReadyReplicas)
	}
	if src.AvailableReplicas != nil {
		out.AvailableReplicas = ptr.To(*src.AvailableReplicas)
	}
	return out
}

// scrubStaleDGDAnnotations removes "nvidia.com/dgd-comp-<name>-*" keys for
// any <name> that is not present in the current components map. Annotations
// scoped to active components are kept: they were either produced by the
// ConvertFrom path for the next ConvertTo to read (e.g. frontend-sidecar-ref)
// or are pass-through keys that a v1alpha1 client may rely on.
func scrubStaleDGDAnnotations(obj *metav1.ObjectMeta, components map[string]*DynamoComponentDeploymentSharedSpec) {
	anns := obj.GetAnnotations()
	if len(anns) == 0 {
		return
	}
	maps.DeleteFunc(anns, func(k, _ string) bool {
		if !strings.HasPrefix(k, annDGDCompPrefix) {
			return false
		}
		rest := strings.TrimPrefix(k, annDGDCompPrefix)
		// Key format is "<annDGDCompPrefix><name>-<suffix>", but both the
		// component name (e.g. "aa-frontend") and the suffix (e.g.
		// "frontend-sidecar-ref") may contain hyphens, so a simple
		// Index("-") split is unsafe. Keep the annotation if its remainder
		// starts with "<active-component-name>-" for any current component;
		// drop it otherwise.
		for name := range components {
			if strings.HasPrefix(rest, name+"-") {
				return false
			}
		}
		return true
	})
	if len(anns) == 0 {
		obj.SetAnnotations(nil)
	} else {
		obj.SetAnnotations(anns)
	}
}
