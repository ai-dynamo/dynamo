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

// ConvertTo converts this DynamoGraphDeployment (v1alpha1) into the hub
// version (v1beta1).
func (src *DynamoGraphDeployment) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeployment but got %T", dstRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()

	dst.Spec.Annotations = maps.Clone(src.Spec.Annotations)
	dst.Spec.Labels = maps.Clone(src.Spec.Labels)
	dst.Spec.BackendFramework = src.Spec.BackendFramework

	if src.Spec.Restart != nil {
		dst.Spec.Restart = convertRestartTo(src.Spec.Restart)
	}
	if src.Spec.TopologyConstraint != nil {
		dst.Spec.TopologyConstraint = &v1beta1.SpecTopologyConstraint{
			TopologyProfile: src.Spec.TopologyConstraint.TopologyProfile,
			PackDomain:      v1beta1.TopologyDomain(src.Spec.TopologyConstraint.PackDomain),
		}
	}

	// DGD.spec.envs -> DGD.spec.env.
	if len(src.Spec.Envs) > 0 {
		dst.Spec.Env = slices.Clone(src.Spec.Envs)
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

	// Components: v1alpha1 map -> v1beta1 list. Sort by name for a deterministic
	// emission order; the unordered map cannot faithfully represent the
	// v1beta1 list order, so round-trip V1 -> A1 -> V2 may reorder entries.
	// This is called out in the golden-file test fixtures.
	if len(src.Spec.Services) > 0 {
		names := slices.Sorted(maps.Keys(src.Spec.Services))
		dst.Spec.Components = make([]v1beta1.DynamoComponentDeploymentSharedSpec, 0, len(names))
		for _, name := range names {
			compSrc := src.Spec.Services[name]
			if compSrc == nil {
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
			// +listMapKey invariant (and the round-trip identity) hold, and
			// stash the (now redundant) v1alpha1 ServiceName in an origin
			// annotation so a mismatched value still round-trips.
			compDst.ComponentName = name
			if compSrc.ServiceName != "" && compSrc.ServiceName != name {
				carrier.set(suffixServiceName, compSrc.ServiceName)
			}
			dst.Spec.Components = append(dst.Spec.Components, compDst)
		}
	}

	convertDGDStatusTo(&src.Status, &dst.Status)
	return nil
}

// ConvertFrom converts from the hub (v1beta1) DynamoGraphDeployment into this
// v1alpha1 instance.
func (dst *DynamoGraphDeployment) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeployment but got %T", srcRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()

	dst.Spec.Annotations = maps.Clone(src.Spec.Annotations)
	dst.Spec.Labels = maps.Clone(src.Spec.Labels)
	dst.Spec.BackendFramework = src.Spec.BackendFramework

	if src.Spec.Restart != nil {
		dst.Spec.Restart = convertRestartFrom(src.Spec.Restart)
	}
	if src.Spec.TopologyConstraint != nil {
		dst.Spec.TopologyConstraint = &SpecTopologyConstraint{
			TopologyProfile: src.Spec.TopologyConstraint.TopologyProfile,
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
			// v1beta1 declares +listType=map +listMapKey=componentName so
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

	convertDGDStatusFrom(&src.Status, &dst.Status)
	scrubStaleDGDAnnotations(&dst.ObjectMeta, dst.Spec.Services)
	return nil
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
		ComponentName:   src.ComponentName,
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
	out := &ServiceReplicaStatus{
		ComponentKind:   ComponentKind(src.ComponentKind),
		ComponentName:   src.ComponentName,
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
