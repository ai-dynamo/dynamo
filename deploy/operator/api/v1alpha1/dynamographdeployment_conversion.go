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
	"encoding/json"
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

// ConvertTo converts this DynamoGraphDeployment (v1alpha1) into the hub
// version (v1beta1).
func (src *DynamoGraphDeployment) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeployment but got %T", dstRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()
	restoredHubSpec := false
	if raw, ok := getAnnFromObj(&dst.ObjectMeta, annDGDHubSpec); ok && raw != "" {
		if spec, ok := restoreDGDHubSpec(raw); ok {
			dst.Spec = spec
			restoredHubSpec = true
			delAnnFromObj(&dst.ObjectMeta, annDGDHubSpec)
			scrubDGDInternalAnnotations(&dst.ObjectMeta)
		}
	}
	hubOrigin := restoredHubSpec || dst.ObjectMeta.Annotations[annDGDHubOrigin] == annotationTrue
	delAnnFromObj(&dst.ObjectMeta, annDGDHubOrigin)

	var semanticSpec v1beta1.DynamoGraphDeploymentSpec
	semanticSpec.Annotations = maps.Clone(src.Spec.Annotations)
	semanticSpec.Labels = maps.Clone(src.Spec.Labels)
	semanticSpec.BackendFramework = src.Spec.BackendFramework

	if src.Spec.Restart != nil {
		semanticSpec.Restart = convertRestartTo(src.Spec.Restart)
	}
	if src.Spec.TopologyConstraint != nil {
		semanticSpec.TopologyConstraint = &v1beta1.SpecTopologyConstraint{
			ClusterTopologyName: src.Spec.TopologyConstraint.TopologyProfile,
			PackDomain:          v1beta1.TopologyDomain(src.Spec.TopologyConstraint.PackDomain),
		}
	}

	// DGD.spec.envs -> DGD.spec.env.
	if len(src.Spec.Envs) > 0 {
		semanticSpec.Env = slices.Clone(src.Spec.Envs)
	}

	// DGD.spec.pvcs has no v1beta1 equivalent -- users now declare PVC volumes
	// directly on podTemplate. Stash as an annotation.
	if len(src.Spec.PVCs) > 0 {
		data, err := json.Marshal(src.Spec.PVCs)
		if err != nil {
			return fmt.Errorf("marshal pvcs: %w", err)
		}
		setAnnOnObj(&dst.ObjectMeta, annDGDPVCs, string(data))
	}
	preserveSpoke := !hubOrigin || dgdHasAlphaOnlyFields(&src.Spec)

	// Components: v1alpha1 map -> v1beta1 list. Sort by name for a deterministic
	// emission order; the unordered map cannot faithfully represent the
	// v1beta1 list order, so round-trip V1 -> A1 -> V2 may reorder entries.
	// This is called out in the golden-file test fixtures.
	if len(src.Spec.Services) > 0 {
		names := slices.Sorted(maps.Keys(src.Spec.Services))
		semanticSpec.Components = make([]v1beta1.DynamoComponentDeploymentSharedSpec, 0, len(names))
		nilServices := make([]string, 0)
		for _, name := range names {
			compSrc := src.Spec.Services[name]
			if compSrc == nil {
				nilServices = append(nilServices, name)
				continue
			}
			carrier := newDGDComponentCarrier(&dst.ObjectMeta, name)
			var compDst v1beta1.DynamoComponentDeploymentSharedSpec
			if err := convertSharedSpecTo(compSrc, &compDst, carrier); err != nil {
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
			semanticSpec.Components = append(semanticSpec.Components, compDst)
		}
		if len(nilServices) > 0 {
			if data, err := json.Marshal(nilServices); err == nil {
				setAnnOnObj(&dst.ObjectMeta, annDGDNilServices, string(data))
			}
		}
	}

	if restoredHubSpec {
		overlayDGDHubSpec(&dst.Spec, &semanticSpec)
	} else {
		dst.Spec = semanticSpec
	}
	if hubOrigin {
		scrubDGDInternalAnnotations(&dst.ObjectMeta)
	}
	convertDGDStatusTo(&src.Status, &dst.Status)
	if preserveSpoke {
		preserveDGDSpoke(src, dst)
		preserveDGDSpokeHub(dst)
	}
	return nil
}

func preserveDGDSpoke(src *DynamoGraphDeployment, dst *v1beta1.DynamoGraphDeployment) {
	if data, err := marshalDGDSpokeSpec(&src.Spec); err == nil {
		setAnnOnObj(&dst.ObjectMeta, annDGDSpokeSpec, string(data))
	}
	if data, err := json.Marshal(src.Status); err == nil {
		setAnnOnObj(&dst.ObjectMeta, annDGDSpokeStatus, string(data))
	}
}

func overlayDGDHubSpec(base *v1beta1.DynamoGraphDeploymentSpec, semantic *v1beta1.DynamoGraphDeploymentSpec) {
	hubComponents := make(map[string]v1beta1.DynamoComponentDeploymentSharedSpec, len(base.Components))
	for _, comp := range base.Components {
		hubComponents[comp.ComponentName] = comp
	}

	*base = *semantic.DeepCopy()
	for i := range base.Components {
		hubComp, ok := hubComponents[base.Components[i].ComponentName]
		if !ok {
			continue
		}
		if hubComp.PodTemplate != nil {
			base.Components[i].PodTemplate = hubComp.PodTemplate.DeepCopy()
		}
		if base.Components[i].FrontendSidecar == nil && hubComp.FrontendSidecar != nil {
			base.Components[i].FrontendSidecar = ptr.To(*hubComp.FrontendSidecar)
		}
		if base.Components[i].Experimental == nil && hubComp.Experimental != nil {
			base.Components[i].Experimental = hubComp.Experimental.DeepCopy()
		}
	}
}

type preservedDGDHubSnapshot struct {
	Spec   string                              `json:"spec"`
	Status v1beta1.DynamoGraphDeploymentStatus `json:"status"`
}

func preserveDGDSpokeHub(dst *v1beta1.DynamoGraphDeployment) {
	spec, err := marshalDGDHubSpec(&dst.Spec)
	if err != nil {
		return
	}
	data, err := json.Marshal(preservedDGDHubSnapshot{
		Spec:   string(spec),
		Status: dst.Status,
	})
	if err == nil {
		setAnnOnObj(&dst.ObjectMeta, annDGDSpokeHub, string(data))
	}
}

func dgdSpokeHubUnmodified(src *v1beta1.DynamoGraphDeployment) bool {
	raw, ok := getAnnFromObj(&src.ObjectMeta, annDGDSpokeHub)
	if !ok || raw == "" {
		return false
	}
	spec, err := marshalDGDHubSpec(&src.Spec)
	if err != nil {
		return false
	}
	current, err := json.Marshal(preservedDGDHubSnapshot{
		Spec:   string(spec),
		Status: src.Status,
	})
	if err != nil {
		return false
	}
	return string(current) == raw
}

func fillDGDSpokeFromPreserved(dstSpec *DynamoGraphDeploymentSpec, dstStatus *DynamoGraphDeploymentStatus, preservedSpec *DynamoGraphDeploymentSpec, preservedStatus *DynamoGraphDeploymentStatus) {
	if preservedSpec != nil {
		if len(dstSpec.PVCs) == 0 {
			dstSpec.PVCs = slices.Clone(preservedSpec.PVCs)
		}
		for name, dstComp := range dstSpec.Services {
			if dstComp == nil || preservedSpec.Services == nil {
				continue
			}
			fillSharedAlphaOnlyFromPreserved(dstComp, preservedSpec.Services[name])
		}
	}
	if preservedStatus != nil {
		for name, dstSvc := range dstStatus.Services {
			preservedSvc, ok := preservedStatus.Services[name]
			if !ok {
				continue
			}
			if dstSvc.ComponentName == "" {
				dstSvc.ComponentName = preservedSvc.ComponentName
			}
			dstStatus.Services[name] = dstSvc
		}
	}
}

func dgdHasAlphaOnlyFields(src *DynamoGraphDeploymentSpec) bool {
	if src == nil {
		return false
	}
	if len(src.PVCs) > 0 {
		return true
	}
	for _, svc := range src.Services {
		if hasSharedAlphaOnlyFields(svc) {
			return true
		}
	}
	return false
}

func decodeDGDSpokePreserved(obj metav1.Object) (*DynamoGraphDeploymentSpec, *DynamoGraphDeploymentStatus) {
	var preservedSpokeSpec *DynamoGraphDeploymentSpec
	var preservedSpokeStatus *DynamoGraphDeploymentStatus
	if raw, ok := getAnnFromObj(obj, annDGDSpokeSpec); ok && raw != "" {
		if spec, ok := restoreDGDSpokeSpec(raw); ok {
			preservedSpokeSpec = &spec
		}
	}
	if rawStatus, ok := getAnnFromObj(obj, annDGDSpokeStatus); ok && rawStatus != "" {
		var status DynamoGraphDeploymentStatus
		if err := json.Unmarshal([]byte(rawStatus), &status); err == nil {
			preservedSpokeStatus = &status
		}
	}
	return preservedSpokeSpec, preservedSpokeStatus
}

func restoreDGDSpokeFastPath(dst *DynamoGraphDeployment, src *v1beta1.DynamoGraphDeployment, preservedSpec *DynamoGraphDeploymentSpec, preservedStatus *DynamoGraphDeploymentStatus) bool {
	// Fast path only: the fingerprint covers the hub spec/status snapshot, so
	// matching means no hub fields changed. Metadata was copied above and rides along.
	if preservedSpec == nil || !dgdSpokeHubUnmodified(src) {
		return false
	}
	dst.Spec = *preservedSpec.DeepCopy()
	if preservedStatus != nil {
		dst.Status = *preservedStatus.DeepCopy()
	} else {
		convertDGDStatusFrom(&src.Status, &dst.Status)
	}
	scrubDGDInternalAnnotations(&dst.ObjectMeta)
	return true
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
	if restoreDGDSpokeFastPath(dst, src, preservedSpokeSpec, preservedSpokeStatus) {
		return nil
	}

	dst.Spec.Annotations = maps.Clone(src.Spec.Annotations)
	dst.Spec.Labels = maps.Clone(src.Spec.Labels)
	dst.Spec.BackendFramework = src.Spec.BackendFramework

	if src.Spec.Restart != nil {
		dst.Spec.Restart = convertRestartFrom(src.Spec.Restart)
	}
	if src.Spec.TopologyConstraint != nil {
		dst.Spec.TopologyConstraint = &SpecTopologyConstraint{
			TopologyProfile: src.Spec.TopologyConstraint.ClusterTopologyName,
			PackDomain:      TopologyDomain(src.Spec.TopologyConstraint.PackDomain),
		}
	}

	if len(src.Spec.Env) > 0 {
		dst.Spec.Envs = slices.Clone(src.Spec.Env)
	}

	if v, ok := getAnnFromObj(&dst.ObjectMeta, annDGDPVCs); ok {
		var pvcs []PVC
		if err := json.Unmarshal([]byte(v), &pvcs); err == nil {
			dst.Spec.PVCs = pvcs
		}
		delAnnFromObj(&dst.ObjectMeta, annDGDPVCs)
	}

	if len(src.Spec.Components) > 0 {
		dst.Spec.Services = make(map[string]*DynamoComponentDeploymentSharedSpec, len(src.Spec.Components))
		for i := range src.Spec.Components {
			compSrc := &src.Spec.Components[i]
			// v1beta1 declares +listType=map +listMapKey=name so
			// the API server normally rejects duplicates, but the
			// conversion path is also reached from in-memory unit-test
			// fixtures and other code paths that bypass CRD validation.
			// Surface duplicates here as a hard error rather than
			// silently overwriting the earlier entry on map insertion.
			if _, dup := dst.Spec.Services[compSrc.ComponentName]; dup {
				return fmt.Errorf("duplicate component name %q in spec.components", compSrc.ComponentName)
			}
			carrier := newDGDComponentCarrier(&dst.ObjectMeta, compSrc.ComponentName)
			compDst := &DynamoComponentDeploymentSharedSpec{}
			if err := convertSharedSpecFrom(compSrc, compDst, carrier); err != nil {
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
			dst.Spec.Services[compSrc.ComponentName] = compDst
		}
	}
	if v, ok := getAnnFromObj(&dst.ObjectMeta, annDGDNilServices); ok {
		var nilServices []string
		if err := json.Unmarshal([]byte(v), &nilServices); err == nil {
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
	if preservedSpokeSpec != nil || preservedSpokeStatus != nil {
		delAnnFromObj(&dst.ObjectMeta, annDGDSpokeSpec)
		delAnnFromObj(&dst.ObjectMeta, annDGDSpokeStatus)
		delAnnFromObj(&dst.ObjectMeta, annDGDSpokeHub)
	}
	if dgdNeedsHubSpecPreservation(&src.Spec) {
		if data, err := marshalDGDHubSpec(&src.Spec); err == nil {
			setAnnOnObj(&dst.ObjectMeta, annDGDHubSpec, string(data))
		}
	} else if !hasDGDInternalAnnotations(src.ObjectMeta.Annotations) {
		setAnnOnObj(&dst.ObjectMeta, annDGDHubOrigin, annotationTrue)
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

func dgdNeedsHubSpecPreservation(src *v1beta1.DynamoGraphDeploymentSpec) bool {
	for i := range src.Components {
		if src.Components[i].PodTemplate != nil {
			return true
		}
	}
	return false
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
