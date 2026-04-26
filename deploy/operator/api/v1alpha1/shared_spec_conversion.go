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

// Shared-spec conversion helpers used by both DGD (per-service) and DCD.
//
// DynamoComponentDeploymentSharedSpec is the payload that differs most between
// v1alpha1 and v1beta1: v1alpha1 has ten per-service pod-configuration fields,
// v1beta1 replaces them with a single corev1.PodTemplateSpec plus a few
// first-class fields. The heavy lifting (podTemplate <-> flat fields,
// experimental block, origin annotations for lossy conversions) lives here so
// that the DGD and DCD conversion entry points stay thin.
//
// Round-trip fidelity
//
// Every v1beta1 field that has no v1alpha1 equivalent, and every v1alpha1
// field that has no v1beta1 equivalent, is stashed under a reserved
// "nvidia.com/dgd-..." or "nvidia.com/dcd-..." annotation so that
// ConvertTo(ConvertFrom(V)) == V for every v1beta1 input V. The annotation
// namespace is operator-managed; user-set annotations with the same prefix
// are parsed best-effort (an unparseable value is ignored rather than
// erroring) and dropped on ConvertFrom once consumed.

package v1alpha1

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/imdario/mergo"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	// mainContainerName is the well-known container name in v1beta1 podTemplate
	// that receives the operator's default merges. Duplicated from
	// v1beta1.MainContainerName so this file has no reverse dependency ordering
	// concerns at build time.
	mainContainerName = "main"

	// defaultFrontendSidecarContainerName is the v1alpha1 default container
	// name synthesised by buildPodTemplateTo when a v1alpha1 FrontendSidecarSpec
	// is converted into a podTemplate sidecar + FrontendSidecar name reference.
	defaultFrontendSidecarContainerName = "sidecar-frontend"

	// defaultGPUResourceName is the v1alpha1 default when a user sets
	// Resources.{Requests,Limits}.GPU without specifying GPUType.
	defaultGPUResourceName = corev1.ResourceName("nvidia.com/gpu")

	// DGD-scoped (DGD spec-level + per-component) annotation keys.
	annDGDPVCs       = "nvidia.com/dgd-pvcs"
	annDGDCompPrefix = "nvidia.com/dgd-comp-"

	// Shared-spec annotation key suffixes. Combined with the parent object's
	// prefix ("nvidia.com/dgd-comp-<name>-" or "nvidia.com/dcd-") to form the
	// full annotation key.
	//
	// suffixServiceName preserves the v1alpha1-only DGD-entry ServiceName
	// when it differs from the components-map key. v1beta1 fixes
	// ComponentName == listMapKey so a non-matching ServiceName on the
	// v1alpha1 side has no first-class home; we stash it here instead so
	// ConvertTo(ConvertFrom(V)) is bitwise stable.
	suffixServiceName      = "service-name"
	suffixDynamoNamespace  = "dynamo-namespace"
	suffixAutoscaling      = "autoscaling"
	suffixIngress          = "ingress"
	suffixSubComponentType = "sub-component-type"
	suffixEnvFromSecret    = "env-from-secret"
	suffixAnnotations      = "annotations"
	suffixLabels           = "labels"
	suffixSharedMemOrigin  = "shared-memory-origin"
	suffixFrontendSidecar  = "frontend-sidecar-origin"
	// suffixFrontendSidecarRef is used for v1beta1-first inputs that set
	// FrontendSidecar to a container name without a v1alpha1 origin. The
	// referenced container flows through ExtraPodSpec.PodSpec.Containers
	// as a regular user sidecar on the v1alpha1 side.
	suffixFrontendSidecarRef = "frontend-sidecar-ref"
	suffixGMSDisabled        = "gms-disabled-payload"
	suffixFailoverDisabled   = "failover-disabled-payload"
	suffixCheckpointDisabl   = "checkpoint-disabled-payload"
	suffixScalingDisabled    = "scaling-adapter-disabled"

	// DCD-scoped annotation keys (standalone DCDs carry origin data on their
	// own metadata rather than on a parent DGD).
	annDCDPrefix = "nvidia.com/dcd-"
)

// annCarrier wraps a Kubernetes object's annotation bag for a given logical
// owner (a DGD service or a standalone DCD). The prefix is baked in so that
// shared-spec helpers do not need to know whether they are serving DGD or DCD.
type annCarrier struct {
	obj    metav1.Object
	prefix string
}

// newDGDComponentCarrier returns a carrier scoped to a named component on the
// DGD object: keys are of the form "nvidia.com/dgd-comp-<name>-<suffix>".
func newDGDComponentCarrier(obj metav1.Object, componentName string) *annCarrier {
	return &annCarrier{obj: obj, prefix: annDGDCompPrefix + componentName + "-"}
}

// newDCDCarrier returns a carrier scoped to a standalone DCD: keys are of the
// form "nvidia.com/dcd-<suffix>".
func newDCDCarrier(obj metav1.Object) *annCarrier {
	return &annCarrier{obj: obj, prefix: annDCDPrefix}
}

func (c *annCarrier) key(suffix string) string { return c.prefix + suffix }

func (c *annCarrier) set(suffix, value string) {
	anns := c.obj.GetAnnotations()
	if anns == nil {
		anns = map[string]string{}
	}
	anns[c.key(suffix)] = value
	c.obj.SetAnnotations(anns)
}

func (c *annCarrier) get(suffix string) (string, bool) {
	anns := c.obj.GetAnnotations()
	if anns == nil {
		return "", false
	}
	v, ok := anns[c.key(suffix)]
	return v, ok
}

func (c *annCarrier) del(suffix string) {
	anns := c.obj.GetAnnotations()
	if anns == nil {
		return
	}
	delete(anns, c.key(suffix))
	if len(anns) == 0 {
		c.obj.SetAnnotations(nil)
	} else {
		c.obj.SetAnnotations(anns)
	}
}

// setAnnOnObj is a convenience for annotations that do not belong to a single
// carrier (e.g. DGD spec-level PVCs).
func setAnnOnObj(obj metav1.Object, key, value string) {
	anns := obj.GetAnnotations()
	if anns == nil {
		anns = map[string]string{}
	}
	anns[key] = value
	obj.SetAnnotations(anns)
}

func getAnnFromObj(obj metav1.Object, key string) (string, bool) {
	anns := obj.GetAnnotations()
	if anns == nil {
		return "", false
	}
	v, ok := anns[key]
	return v, ok
}

func delAnnFromObj(obj metav1.Object, key string) {
	anns := obj.GetAnnotations()
	if anns == nil {
		return
	}
	delete(anns, key)
	if len(anns) == 0 {
		obj.SetAnnotations(nil)
	} else {
		obj.SetAnnotations(anns)
	}
}

// scrubAnnotationsByPrefix removes every annotation whose key starts with
// prefix, collapsing the map back to nil if nothing remains. Used by the DGD
// and DCD ConvertFrom paths to clean up any reserved origin keys that were
// not consumed (e.g. v1alpha1 client left a stale entry).
func scrubAnnotationsByPrefix(obj metav1.Object, prefix string) {
	anns := obj.GetAnnotations()
	if len(anns) == 0 {
		return
	}
	maps.DeleteFunc(anns, func(k, _ string) bool {
		return strings.HasPrefix(k, prefix)
	})
	if len(anns) == 0 {
		obj.SetAnnotations(nil)
	} else {
		obj.SetAnnotations(anns)
	}
}

// ---------------------------------------------------------------------------
// Shared-spec conversion entry points
// ---------------------------------------------------------------------------

// convertSharedSpecTo converts a v1alpha1 DynamoComponentDeploymentSharedSpec
// into its v1beta1 counterpart. Lossy-direction fields are preserved on the
// carrier's annotation bag.
func convertSharedSpecTo(src *DynamoComponentDeploymentSharedSpec, dst *v1beta1.DynamoComponentDeploymentSharedSpec, c *annCarrier) error {
	if src == nil || dst == nil {
		return nil
	}

	// ComponentType: in v1alpha1 this is a free-form string; in v1beta1 it is
	// a strict enum. The conversion is a value copy; unknown strings become
	// the empty enum on the v1beta1 side and will be rejected by the CRD
	// validator, which matches the intended behaviour.
	dst.ComponentType = v1beta1.ComponentType(src.ComponentType)

	// SubComponentType has no v1beta1 equivalent -- the "prefill"/"decode"
	// values are first-class ComponentType enum values in v1beta1. Stash the
	// raw string on the carrier so round-trip restores the original wording.
	if src.SubComponentType != "" {
		c.set(suffixSubComponentType, src.SubComponentType)
	}

	// v1alpha1 ServiceName <-> v1beta1 ComponentName: the same logical
	// identifier, renamed at v1beta1 to align with the
	// `spec.components` rename. For DGD components the caller overrides
	// dst.ComponentName with the v1alpha1 services-map key (the canonical
	// source of truth on v1alpha1); for standalone DCDs the caller falls
	// back to ObjectMeta.Name when src.ServiceName is empty.
	dst.ComponentName = src.ServiceName

	// DynamoNamespace (deprecated) -> annotation.
	if src.DynamoNamespace != nil {
		c.set(suffixDynamoNamespace, *src.DynamoNamespace)
	}

	dst.GlobalDynamoNamespace = src.GlobalDynamoNamespace
	dst.Replicas = src.Replicas

	if src.Multinode != nil {
		dst.Multinode = &v1beta1.MultinodeSpec{NodeCount: src.Multinode.NodeCount}
	}

	if src.ModelRef != nil {
		dst.ModelRef = &v1beta1.ModelReference{
			Name:     src.ModelRef.Name,
			Revision: src.ModelRef.Revision,
		}
	}

	if src.TopologyConstraint != nil {
		dst.TopologyConstraint = &v1beta1.TopologyConstraint{
			PackDomain: v1beta1.TopologyDomain(src.TopologyConstraint.PackDomain),
		}
	}

	if src.EPPConfig != nil {
		dst.EPPConfig = &v1beta1.EPPConfig{
			ConfigMapRef: src.EPPConfig.ConfigMapRef.DeepCopy(),
			Config:       src.EPPConfig.Config.DeepCopy(),
		}
	}

	// Autoscaling -> annotation (deprecated, removed in v1beta1).
	if src.Autoscaling != nil {
		if data, err := json.Marshal(src.Autoscaling); err == nil {
			c.set(suffixAutoscaling, string(data))
		}
	}

	// Ingress -> annotation (removed in v1beta1).
	if src.Ingress != nil {
		if data, err := json.Marshal(src.Ingress); err == nil {
			c.set(suffixIngress, string(data))
		}
	}

	// EnvFromSecret -> podTemplate.containers[main].envFrom; the pointer string
	// itself is also preserved on the carrier so ConvertFrom can distinguish
	// "v1alpha1 set this via EnvFromSecret" from "v1beta1 set it via envFrom".
	if src.EnvFromSecret != nil {
		c.set(suffixEnvFromSecret, *src.EnvFromSecret)
	}

	// Per-service Annotations / Labels: stash verbatim; they do not merge
	// into podTemplate.metadata because they also apply to Service/Ingress
	// objects at the v1alpha1 reconcile layer.
	if len(src.Annotations) > 0 {
		if data, err := json.Marshal(src.Annotations); err == nil {
			c.set(suffixAnnotations, string(data))
		}
	}
	if len(src.Labels) > 0 {
		if data, err := json.Marshal(src.Labels); err == nil {
			c.set(suffixLabels, string(data))
		}
	}

	// sharedMemory <-> sharedMemorySize (lossy struct flatten).
	if err := convertSharedMemoryTo(src.SharedMemory, dst, c); err != nil {
		return err
	}

	// volumeMounts + useAsCompilationCache -> compilationCache (the container
	// volumeMounts themselves are emitted by buildPodTemplateTo).
	for _, vm := range src.VolumeMounts {
		if vm.UseAsCompilationCache {
			dst.CompilationCache = &v1beta1.CompilationCacheConfig{
				PVCName:   vm.Name,
				MountPath: vm.MountPoint,
			}
			break
		}
	}

	// scalingAdapter: drop Enabled bool, keep presence semantics.
	convertScalingAdapterTo(src.ScalingAdapter, dst, c)

	// experimental block: gpuMemoryService, failover, checkpoint.
	convertExperimentalTo(src, dst, c)

	// Resources + envs + probes + mainContainer -> podTemplate.containers[main].
	return buildPodTemplateTo(src, dst, c)
}

// convertSharedSpecFrom performs the inverse: v1beta1 -> v1alpha1.
func convertSharedSpecFrom(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst *DynamoComponentDeploymentSharedSpec, c *annCarrier) error {
	if src == nil || dst == nil {
		return nil
	}

	dst.ComponentType = string(src.ComponentType)
	dst.GlobalDynamoNamespace = src.GlobalDynamoNamespace
	dst.Replicas = src.Replicas

	if src.Multinode != nil {
		dst.Multinode = &MultinodeSpec{NodeCount: src.Multinode.NodeCount}
	}
	if src.ModelRef != nil {
		dst.ModelRef = &ModelReference{
			Name:     src.ModelRef.Name,
			Revision: src.ModelRef.Revision,
		}
	}
	if src.TopologyConstraint != nil {
		dst.TopologyConstraint = &TopologyConstraint{
			PackDomain: TopologyDomain(src.TopologyConstraint.PackDomain),
		}
	}
	if src.EPPConfig != nil {
		dst.EPPConfig = &EPPConfig{
			ConfigMapRef: src.EPPConfig.ConfigMapRef.DeepCopy(),
			Config:       src.EPPConfig.Config.DeepCopy(),
		}
	}

	// Restore annotation-preserved v1alpha1 fields.
	if v, ok := c.get(suffixSubComponentType); ok {
		dst.SubComponentType = v
		c.del(suffixSubComponentType)
	}
	dst.ServiceName = src.ComponentName
	if v, ok := c.get(suffixDynamoNamespace); ok {
		dst.DynamoNamespace = ptr.To(v)
		c.del(suffixDynamoNamespace)
	}
	if v, ok := c.get(suffixAutoscaling); ok {
		var as Autoscaling
		if err := json.Unmarshal([]byte(v), &as); err == nil {
			dst.Autoscaling = &as
		}
		c.del(suffixAutoscaling)
	}
	if v, ok := c.get(suffixIngress); ok {
		var ing IngressSpec
		if err := json.Unmarshal([]byte(v), &ing); err == nil {
			dst.Ingress = &ing
		}
		c.del(suffixIngress)
	}
	if v, ok := c.get(suffixAnnotations); ok {
		var m map[string]string
		if err := json.Unmarshal([]byte(v), &m); err == nil {
			dst.Annotations = m
		}
		c.del(suffixAnnotations)
	}
	if v, ok := c.get(suffixLabels); ok {
		var m map[string]string
		if err := json.Unmarshal([]byte(v), &m); err == nil {
			dst.Labels = m
		}
		c.del(suffixLabels)
	}
	if v, ok := c.get(suffixEnvFromSecret); ok {
		dst.EnvFromSecret = ptr.To(v)
		c.del(suffixEnvFromSecret)
	}

	// sharedMemorySize -> SharedMemorySpec.
	convertSharedMemoryFrom(src.SharedMemorySize, dst, c)

	// compilationCache + podTemplate volumeMounts -> VolumeMounts.
	convertVolumeMountsFrom(src, dst)

	// experimental -> GPUMemoryService, Failover, Checkpoint.
	convertExperimentalFrom(src, dst, c)

	// scalingAdapter presence -> Enabled=true; annotation -> Enabled=false payload.
	convertScalingAdapterFrom(src.ScalingAdapter, dst, c)

	// podTemplate -> mainContainer + extraPodSpec + extraPodMetadata +
	// Resources + Envs + Probes (+ FrontendSidecar).
	return decomposePodTemplate(src, dst, c)
}

// ---------------------------------------------------------------------------
// Shared-memory
// ---------------------------------------------------------------------------

func convertSharedMemoryTo(src *SharedMemorySpec, dst *v1beta1.DynamoComponentDeploymentSharedSpec, c *annCarrier) error {
	if src == nil {
		return nil
	}
	// Disabled=true -> size "0" (no origin annotation needed: convertSharedMemoryFrom
	// interprets a zero quantity without origin as Disabled=true, which is the
	// canonical meaning of sharedMemorySize=0 on the v1beta1 side).
	// Disabled=false with Size=X -> X.
	// Disabled=false with Size=zero -> empty struct, which has no v1beta1
	// equivalent; stash via annotation to preserve the explicit pointer.
	if src.Disabled {
		dst.SharedMemorySize = ptr.To(resource.MustParse("0"))
		return nil
	}
	// Not disabled, with a set size -> copy verbatim.
	if !src.Size.IsZero() {
		dst.SharedMemorySize = ptr.To(src.Size.DeepCopy())
		return nil
	}
	// Disabled=false, Size=zero -> this is an empty struct. ConvertFrom must
	// be able to reproduce "&SharedMemorySpec{}", distinct from a nil pointer.
	c.set(suffixSharedMemOrigin, "empty")
	return nil
}

func convertSharedMemoryFrom(src *resource.Quantity, dst *DynamoComponentDeploymentSharedSpec, c *annCarrier) {
	if v, ok := c.get(suffixSharedMemOrigin); ok {
		c.del(suffixSharedMemOrigin)
		if v == "empty" {
			// A bare SharedMemorySpec{} carries Size: Quantity{}, which
			// serializes to "0" because resource.Quantity is a non-pointer
			// struct (encoding/json's `omitempty` does not treat structs
			// as empty). After the etcd JSON round-trip, Size comes back
			// as a canonical zero Quantity that is NOT reflect.DeepEqual
			// to the Go zero value, which would cause every
			// kubectl apply to bump `.metadata.generation`. Emit the
			// canonical zero form directly so reapplies are idempotent.
			// See convertSharedMemoryFrom(disabled) for the twin case.
			dst.SharedMemory = &SharedMemorySpec{Size: resource.MustParse("0")}
			return
		}
	}
	if src == nil {
		return
	}
	if src.Sign() == 0 {
		// Canonical v1beta1 "size=0" <-> v1alpha1 Disabled=true. See
		// convertSharedMemoryTo for the forward direction.
		//
		// Size is carried as the DeepCopy of the incoming canonical Quantity
		// (not the Go zero value) so that every apply produces a spec that is
		// reflect.DeepEqual to what's in etcd. A bare Quantity{} and a
		// JSON-round-tripped Quantity serialize identically to "0" but differ
		// in internal state (Format, cached string), and the API server's
		// generation-bump check uses DeepEqual -- so emitting a bare
		// Quantity{} here would cause every `kubectl apply` to bump
		// `.metadata.generation` even though the spec is byte-identical.
		dst.SharedMemory = &SharedMemorySpec{Disabled: true, Size: src.DeepCopy()}
		return
	}
	dst.SharedMemory = &SharedMemorySpec{Size: src.DeepCopy()}
}

// ---------------------------------------------------------------------------
// Volume mounts and compilation cache
// ---------------------------------------------------------------------------
//
// v1alpha1 carries PVC bindings in two slots: a flat VolumeMounts list (with
// a per-entry UseAsCompilationCache flag) and ExtraPodSpec.MainContainer.
// v1beta1 consolidates volume mounts into the podTemplate main container and
// hoists the "cache" flag into a first-class CompilationCacheConfig field.
//
// Provenance rule for the round-trip:
//   - CompilationCacheConfig round-trips as a single flagged v1alpha1
//     VolumeMount (UseAsCompilationCache=true).
//   - All other container-level volumeMounts live on
//     ExtraPodSpec.MainContainer in v1alpha1 (they were set via the
//     extraPodSpec escape hatch anyway, so placing them there mirrors the
//     v1alpha1 reconcile-merge behaviour in graph.go).

// convertVolumeMountsFrom is the v1beta1 -> v1alpha1 inverse: it synthesises
// a single flagged entry in dst.VolumeMounts when the v1beta1 side declares
// a CompilationCacheConfig. Non-cache mounts on the main container are
// preserved through decomposePodTemplate's ExtraPodSpec.MainContainer copy,
// not here.
func convertVolumeMountsFrom(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst *DynamoComponentDeploymentSharedSpec) {
	if src.CompilationCache == nil {
		return
	}
	dst.VolumeMounts = []VolumeMount{{
		Name:                  src.CompilationCache.PVCName,
		MountPoint:            src.CompilationCache.MountPath,
		UseAsCompilationCache: true,
	}}
}

// ---------------------------------------------------------------------------
// Scaling adapter (Enabled flag removed in v1beta1)
// ---------------------------------------------------------------------------

func convertScalingAdapterTo(src *ScalingAdapter, dst *v1beta1.DynamoComponentDeploymentSharedSpec, c *annCarrier) {
	if src == nil {
		return
	}
	if src.Enabled {
		dst.ScalingAdapter = &v1beta1.ScalingAdapter{}
		return
	}
	// Enabled=false with a populated struct is unreachable today (no other
	// fields on v1alpha1 ScalingAdapter) but we still record presence so the
	// v1alpha1 -> v1beta1 -> v1alpha1 round-trip preserves the caller's
	// explicit "&ScalingAdapter{Enabled:false}" pointer.
	c.set(suffixScalingDisabled, "true")
}

func convertScalingAdapterFrom(src *v1beta1.ScalingAdapter, dst *DynamoComponentDeploymentSharedSpec, c *annCarrier) {
	if src != nil {
		dst.ScalingAdapter = &ScalingAdapter{Enabled: true}
		// Drop any stale v1alpha1-first "disabled" annotation: v1beta1 is
		// authoritative in this direction and an enabled v1beta1 entry
		// must not resurrect a previous Enabled:false on the next round.
		c.del(suffixScalingDisabled)
		return
	}
	if _, ok := c.get(suffixScalingDisabled); ok {
		dst.ScalingAdapter = &ScalingAdapter{Enabled: false}
		c.del(suffixScalingDisabled)
	}
}

// ---------------------------------------------------------------------------
// Experimental (gpuMemoryService, failover, checkpoint)
// ---------------------------------------------------------------------------

func convertExperimentalTo(src *DynamoComponentDeploymentSharedSpec, dst *v1beta1.DynamoComponentDeploymentSharedSpec, c *annCarrier) {
	var exp *v1beta1.ExperimentalSpec
	ensureExp := func() *v1beta1.ExperimentalSpec {
		if exp == nil {
			exp = &v1beta1.ExperimentalSpec{}
		}
		return exp
	}

	if src.GPUMemoryService != nil {
		if src.GPUMemoryService.Enabled {
			ensureExp().GPUMemoryService = &v1beta1.GPUMemoryServiceSpec{
				Mode:            v1beta1.GPUMemoryServiceMode(src.GPUMemoryService.Mode),
				DeviceClassName: src.GPUMemoryService.DeviceClassName,
			}
		} else if !gmsIsZeroPayload(src.GPUMemoryService) {
			if data, err := json.Marshal(src.GPUMemoryService); err == nil {
				c.set(suffixGMSDisabled, string(data))
			}
		} else {
			// Enabled=false with an otherwise-empty payload is semantically
			// indistinguishable from absence; still preserve the explicit
			// pointer so round-trip reproduces &GPUMemoryServiceSpec{}.
			c.set(suffixGMSDisabled, `{}`)
		}
	}

	if src.Failover != nil {
		if src.Failover.Enabled {
			ensureExp().Failover = &v1beta1.FailoverSpec{
				Mode:       v1beta1.GPUMemoryServiceMode(src.Failover.Mode),
				NumShadows: src.Failover.NumShadows,
			}
		} else if !failoverIsZeroPayload(src.Failover) {
			if data, err := json.Marshal(src.Failover); err == nil {
				c.set(suffixFailoverDisabled, string(data))
			}
		} else {
			c.set(suffixFailoverDisabled, `{}`)
		}
	}

	if src.Checkpoint != nil {
		if src.Checkpoint.Enabled {
			ensureExp().Checkpoint = checkpointToV1beta1(src.Checkpoint)
		} else if !checkpointIsZeroPayload(src.Checkpoint) {
			if data, err := json.Marshal(src.Checkpoint); err == nil {
				c.set(suffixCheckpointDisabl, string(data))
			}
		} else {
			c.set(suffixCheckpointDisabl, `{}`)
		}
	}

	dst.Experimental = exp
}

func convertExperimentalFrom(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst *DynamoComponentDeploymentSharedSpec, c *annCarrier) {
	// For each experimental sub-block, if v1beta1 declares the feature we
	// drop any stale "*-disabled-payload" annotation: v1beta1 is
	// authoritative in this direction and an enabled v1beta1 entry must
	// not resurrect a previous Enabled:false on the next round-trip.
	if src.Experimental != nil && src.Experimental.GPUMemoryService != nil {
		dst.GPUMemoryService = &GPUMemoryServiceSpec{
			Enabled:         true,
			Mode:            GPUMemoryServiceMode(src.Experimental.GPUMemoryService.Mode),
			DeviceClassName: src.Experimental.GPUMemoryService.DeviceClassName,
		}
		c.del(suffixGMSDisabled)
	} else if v, ok := c.get(suffixGMSDisabled); ok {
		var g GPUMemoryServiceSpec
		if err := json.Unmarshal([]byte(v), &g); err == nil {
			g.Enabled = false
			dst.GPUMemoryService = &g
		}
		c.del(suffixGMSDisabled)
	}

	if src.Experimental != nil && src.Experimental.Failover != nil {
		dst.Failover = &FailoverSpec{
			Enabled:    true,
			Mode:       GPUMemoryServiceMode(src.Experimental.Failover.Mode),
			NumShadows: src.Experimental.Failover.NumShadows,
		}
		c.del(suffixFailoverDisabled)
	} else if v, ok := c.get(suffixFailoverDisabled); ok {
		var f FailoverSpec
		if err := json.Unmarshal([]byte(v), &f); err == nil {
			f.Enabled = false
			dst.Failover = &f
		}
		c.del(suffixFailoverDisabled)
	}

	if src.Experimental != nil && src.Experimental.Checkpoint != nil {
		dst.Checkpoint = checkpointFromV1beta1(src.Experimental.Checkpoint, true)
		c.del(suffixCheckpointDisabl)
	} else if v, ok := c.get(suffixCheckpointDisabl); ok {
		var cp ServiceCheckpointConfig
		if err := json.Unmarshal([]byte(v), &cp); err == nil {
			cp.Enabled = false
			dst.Checkpoint = &cp
		}
		c.del(suffixCheckpointDisabl)
	}
}

func gmsIsZeroPayload(s *GPUMemoryServiceSpec) bool {
	return s.Mode == "" && s.DeviceClassName == ""
}

func failoverIsZeroPayload(s *FailoverSpec) bool {
	return s.Mode == "" && s.NumShadows == 0
}

func checkpointIsZeroPayload(s *ServiceCheckpointConfig) bool {
	return s.Mode == "" && s.CheckpointRef == nil && s.Identity == nil
}

func checkpointToV1beta1(src *ServiceCheckpointConfig) *v1beta1.ComponentCheckpointConfig {
	dst := &v1beta1.ComponentCheckpointConfig{
		Mode:          v1beta1.CheckpointMode(src.Mode),
		CheckpointRef: src.CheckpointRef,
	}
	if src.Identity != nil {
		dst.Identity = &v1beta1.DynamoCheckpointIdentity{
			Model:                src.Identity.Model,
			BackendFramework:     src.Identity.BackendFramework,
			DynamoVersion:        src.Identity.DynamoVersion,
			TensorParallelSize:   src.Identity.TensorParallelSize,
			PipelineParallelSize: src.Identity.PipelineParallelSize,
			Dtype:                src.Identity.Dtype,
			MaxModelLen:          src.Identity.MaxModelLen,
			ExtraParameters:      src.Identity.ExtraParameters,
		}
	}
	return dst
}

func checkpointFromV1beta1(src *v1beta1.ComponentCheckpointConfig, enabled bool) *ServiceCheckpointConfig {
	dst := &ServiceCheckpointConfig{
		Enabled:       enabled,
		Mode:          CheckpointMode(src.Mode),
		CheckpointRef: src.CheckpointRef,
	}
	if src.Identity != nil {
		dst.Identity = &DynamoCheckpointIdentity{
			Model:                src.Identity.Model,
			BackendFramework:     src.Identity.BackendFramework,
			DynamoVersion:        src.Identity.DynamoVersion,
			TensorParallelSize:   src.Identity.TensorParallelSize,
			PipelineParallelSize: src.Identity.PipelineParallelSize,
			Dtype:                src.Identity.Dtype,
			MaxModelLen:          src.Identity.MaxModelLen,
			ExtraParameters:      src.Identity.ExtraParameters,
		}
	}
	return dst
}

// ---------------------------------------------------------------------------
// podTemplate (the big one)
// ---------------------------------------------------------------------------

// buildPodTemplateTo composes the v1beta1 podTemplate from v1alpha1's flat
// fields (Resources, Envs, Probes, EnvFromSecret, ExtraPodSpec,
// ExtraPodMetadata, FrontendSidecar) following the same merge precedence the
// v1alpha1 controller uses at reconcile time: ExtraPodSpec.MainContainer wins
// over dedicated fields, except for env which is additive.
func buildPodTemplateTo(src *DynamoComponentDeploymentSharedSpec, dst *v1beta1.DynamoComponentDeploymentSharedSpec, c *annCarrier) error {
	hasAny := src.Resources != nil ||
		len(src.Envs) > 0 ||
		src.EnvFromSecret != nil ||
		len(src.VolumeMounts) > 0 ||
		src.LivenessProbe != nil ||
		src.ReadinessProbe != nil ||
		src.ExtraPodSpec != nil ||
		src.ExtraPodMetadata != nil ||
		src.FrontendSidecar != nil

	if !hasAny {
		return nil
	}

	podTpl := &corev1.PodTemplateSpec{}
	if src.ExtraPodMetadata != nil {
		if len(src.ExtraPodMetadata.Annotations) > 0 {
			podTpl.Annotations = maps.Clone(src.ExtraPodMetadata.Annotations)
		}
		if len(src.ExtraPodMetadata.Labels) > 0 {
			podTpl.Labels = maps.Clone(src.ExtraPodMetadata.Labels)
		}
	}
	if src.ExtraPodSpec != nil && src.ExtraPodSpec.PodSpec != nil {
		podTpl.Spec = *src.ExtraPodSpec.PodSpec.DeepCopy()
	}

	// Main container: base from dedicated fields.
	mainBase := buildMainContainerFromDedicated(src)

	// Merge ExtraPodSpec.MainContainer on top (mergo.WithOverride), except for
	// Env which is additive (dedicated first + mainContainer second).
	if src.ExtraPodSpec != nil && src.ExtraPodSpec.MainContainer != nil {
		main := src.ExtraPodSpec.MainContainer.DeepCopy()
		baseEnvs := mainBase.Env
		// Name must be "main" regardless of what MainContainer carried.
		main.Name = mainContainerName
		if err := mergo.Merge(&mainBase, *main, mergo.WithOverride); err != nil {
			return fmt.Errorf("merge main container: %w", err)
		}
		mainBase.Env = mergeEnvs(baseEnvs, main.Env)
		// StartupProbe has no dedicated v1alpha1 field; take it verbatim.
		if main.StartupProbe != nil {
			mainBase.StartupProbe = main.StartupProbe
		}
	}
	mainBase.Name = mainContainerName

	// Assemble containers: main first, then non-main user sidecars.
	containers := []corev1.Container{mainBase}
	for _, ctr := range podTpl.Spec.Containers {
		if ctr.Name != mainContainerName {
			containers = append(containers, ctr)
		}
	}
	podTpl.Spec.Containers = containers

	// FrontendSidecar: full container spec -> name reference. Append the
	// container to podTemplate.containers (as a user sidecar) if not already
	// present, and set FrontendSidecar to its name.
	if src.FrontendSidecar != nil {
		appendFrontendSidecar(podTpl, src.FrontendSidecar)
		// Stash origin so ConvertFrom can reproduce the full spec on the
		// v1alpha1 side without depending on the podTemplate sidecar.
		if data, err := json.Marshal(src.FrontendSidecar); err == nil {
			c.set(suffixFrontendSidecar, string(data))
		}
		dst.FrontendSidecar = ptr.To(defaultFrontendSidecarContainerName)
		// Drop any stale v1beta1-first ref annotation; the origin annotation
		// is authoritative in this direction.
		c.del(suffixFrontendSidecarRef)
	} else if v, ok := c.get(suffixFrontendSidecarRef); ok {
		// v1beta1-first input: the named container is already present in
		// podTpl.Spec.Containers (via ExtraPodSpec.PodSpec.Containers).
		// Restore the pointer without duplicating the container.
		dst.FrontendSidecar = ptr.To(v)
		c.del(suffixFrontendSidecarRef)
	}

	dst.PodTemplate = podTpl
	return nil
}

// buildMainContainerFromDedicated collects the v1alpha1 flat fields into a
// corev1.Container named "main".
func buildMainContainerFromDedicated(src *DynamoComponentDeploymentSharedSpec) corev1.Container {
	ctr := corev1.Container{Name: mainContainerName}
	if src.Resources != nil {
		ctr.Resources = resourcesToNative(src.Resources)
	}
	if len(src.Envs) > 0 {
		ctr.Env = slices.Clone(src.Envs)
	}
	if src.EnvFromSecret != nil && *src.EnvFromSecret != "" {
		ctr.EnvFrom = append(ctr.EnvFrom, corev1.EnvFromSource{
			SecretRef: &corev1.SecretEnvSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: *src.EnvFromSecret},
			},
		})
	}
	if src.LivenessProbe != nil {
		ctr.LivenessProbe = src.LivenessProbe.DeepCopy()
	}
	if src.ReadinessProbe != nil {
		ctr.ReadinessProbe = src.ReadinessProbe.DeepCopy()
	}
	for _, vm := range src.VolumeMounts {
		mp := vm.MountPoint
		ctr.VolumeMounts = append(ctr.VolumeMounts, corev1.VolumeMount{
			Name:      vm.Name,
			MountPath: mp,
		})
	}
	return ctr
}

// decomposePodTemplate inverts buildPodTemplateTo.
func decomposePodTemplate(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst *DynamoComponentDeploymentSharedSpec, c *annCarrier) error {
	if src.PodTemplate == nil {
		// Restore FrontendSidecar from origin annotation even if there is no
		// podTemplate (uncommon but possible for manual edits).
		if v, ok := c.get(suffixFrontendSidecar); ok {
			var fs FrontendSidecarSpec
			if err := json.Unmarshal([]byte(v), &fs); err == nil {
				dst.FrontendSidecar = &fs
			}
			c.del(suffixFrontendSidecar)
		}
		return nil
	}

	podTpl := src.PodTemplate.DeepCopy()

	// ExtraPodMetadata from podTemplate.metadata.
	if len(podTpl.Annotations) > 0 || len(podTpl.Labels) > 0 {
		dst.ExtraPodMetadata = &ExtraPodMetadata{
			Annotations: maps.Clone(podTpl.Annotations),
			Labels:      maps.Clone(podTpl.Labels),
		}
	}

	// Pick out the main container; leave everything else as podSpec sidecars.
	var main *corev1.Container
	other := make([]corev1.Container, 0, len(podTpl.Spec.Containers))
	for i := range podTpl.Spec.Containers {
		if podTpl.Spec.Containers[i].Name == mainContainerName && main == nil {
			m := podTpl.Spec.Containers[i].DeepCopy()
			main = m
			continue
		}
		other = append(other, podTpl.Spec.Containers[i])
	}

	// FrontendSidecar handling. Two cases:
	//
	//   a) v1alpha1 origin annotation present -> the v1alpha1 side had a full
	//      FrontendSidecarSpec. Restore it from the annotation and drop the
	//      corresponding container from `other` (v1alpha1 does not carry the
	//      container in extraPodSpec.containers in this case).
	//
	//   b) No v1alpha1 origin annotation (v1beta1-first input). There is no
	//      way to fit a name-only pointer into v1alpha1's schema, so we let
	//      the referenced container flow through as a regular sidecar on
	//      extraPodSpec.containers and stash the name under a reserved
	//      annotation so that ConvertTo can reassemble the v1beta1 pointer
	//      without duplicating the container.
	if src.FrontendSidecar != nil {
		if v, ok := c.get(suffixFrontendSidecar); ok {
			var fs FrontendSidecarSpec
			if err := json.Unmarshal([]byte(v), &fs); err == nil {
				dst.FrontendSidecar = &fs
			}
			c.del(suffixFrontendSidecar)
			filtered := other[:0]
			for _, ctr := range other {
				if ctr.Name != *src.FrontendSidecar {
					filtered = append(filtered, ctr)
				}
			}
			other = filtered
		} else {
			c.set(suffixFrontendSidecarRef, *src.FrontendSidecar)
		}
	}

	// Put everything non-main into ExtraPodSpec.
	podSpecCopy := podTpl.Spec.DeepCopy()
	podSpecCopy.Containers = other
	// The forward path (buildPodTemplateTo) always emits a "main" container,
	// even when v1alpha1 had no main-container fields set (e.g. only
	// FrontendSidecar triggered hasAny). Skip recording it on the v1alpha1
	// side when every field other than Name is zero-valued, so that
	// ConvertFrom does not hallucinate an empty MainContainer.
	var mainCopy *corev1.Container
	if main != nil {
		m := main.DeepCopy()
		m.Name = "" // v1alpha1 MainContainer has no Name (it is always "main").
		if !containerIsEmpty(m) {
			mainCopy = m
		}
	}
	if !podSpecIsZero(podSpecCopy) || mainCopy != nil {
		eps := &ExtraPodSpec{}
		if !podSpecIsZero(podSpecCopy) {
			eps.PodSpec = podSpecCopy
		}
		if mainCopy != nil {
			eps.MainContainer = mainCopy
		}
		dst.ExtraPodSpec = eps
	}

	return nil
}

// containerIsEmpty reports whether c has no user-visible fields set. Used by
// decomposePodTemplate to drop the "main" container synthesized by
// buildPodTemplateTo when v1alpha1 had no main-container fields of its own.
// Name is expected to have been cleared by the caller.
func containerIsEmpty(c *corev1.Container) bool {
	if c == nil {
		return true
	}
	return c.Name == "" &&
		c.Image == "" &&
		len(c.Command) == 0 &&
		len(c.Args) == 0 &&
		c.WorkingDir == "" &&
		len(c.Ports) == 0 &&
		len(c.EnvFrom) == 0 &&
		len(c.Env) == 0 &&
		len(c.Resources.Requests) == 0 &&
		len(c.Resources.Limits) == 0 &&
		len(c.Resources.Claims) == 0 &&
		len(c.ResizePolicy) == 0 &&
		c.RestartPolicy == nil &&
		len(c.VolumeMounts) == 0 &&
		len(c.VolumeDevices) == 0 &&
		c.LivenessProbe == nil &&
		c.ReadinessProbe == nil &&
		c.StartupProbe == nil &&
		c.Lifecycle == nil &&
		c.TerminationMessagePath == "" &&
		c.TerminationMessagePolicy == "" &&
		c.ImagePullPolicy == "" &&
		c.SecurityContext == nil &&
		!c.Stdin &&
		!c.StdinOnce &&
		!c.TTY
}

// appendFrontendSidecar ensures a container named
// defaultFrontendSidecarContainerName exists in the podTemplate's container
// list, synthesising one from the v1alpha1 FrontendSidecarSpec when absent.
// Callers use defaultFrontendSidecarContainerName directly for the resulting
// name reference.
func appendFrontendSidecar(podTpl *corev1.PodTemplateSpec, fs *FrontendSidecarSpec) {
	for _, ctr := range podTpl.Spec.Containers {
		if ctr.Name == defaultFrontendSidecarContainerName {
			return
		}
	}
	ctr := corev1.Container{
		Name:  defaultFrontendSidecarContainerName,
		Image: fs.Image,
		Args:  slices.Clone(fs.Args),
		Env:   slices.Clone(fs.Envs),
	}
	if fs.EnvFromSecret != nil && *fs.EnvFromSecret != "" {
		ctr.EnvFrom = append(ctr.EnvFrom, corev1.EnvFromSource{
			SecretRef: &corev1.SecretEnvSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: *fs.EnvFromSecret},
			},
		})
	}
	podTpl.Spec.Containers = append(podTpl.Spec.Containers, ctr)
}

// podSpecIsZero reports whether a PodSpec has no fields set. Uses the
// apiserver's own apiequality.Semantic.DeepEqual against the zero value so
// that any newly-added PodSpec field (EphemeralContainers, DNSConfig,
// TopologySpreadConstraints, ResourceClaims, HostPID/IPC, SchedulingGates,
// etc.) is automatically covered without extending an allowlist.
//
// Semantic.DeepEqual is preferred over reflect.DeepEqual because it treats
// nil-vs-empty (e.g. NodeSelector: nil vs map[string]string{}) as
// equivalent, which is the exact same comparison the apiserver uses for
// generation-bump checks; this matches what we want here ("would the
// apiserver consider this PodSpec empty?").
func podSpecIsZero(p *corev1.PodSpec) bool {
	if p == nil {
		return true
	}
	return apiequality.Semantic.DeepEqual(*p, corev1.PodSpec{})
}

// ---------------------------------------------------------------------------
// Resources <-> corev1.ResourceRequirements
// ---------------------------------------------------------------------------

// resourcesToNative converts v1alpha1.Resources into corev1.ResourceRequirements.
//   - "cpu", "memory" are recognised by name.
//   - A GPU value with GPUType=""  maps to "nvidia.com/gpu" (the v1alpha1
//     default); with GPUType="X" maps to key X.
//   - Custom keys are copied through.
func resourcesToNative(r *Resources) corev1.ResourceRequirements {
	out := corev1.ResourceRequirements{Claims: slices.Clone(r.Claims)}
	if r.Requests != nil {
		out.Requests = itemToResourceList(r.Requests)
	}
	if r.Limits != nil {
		out.Limits = itemToResourceList(r.Limits)
	}
	return out
}

func itemToResourceList(item *ResourceItem) corev1.ResourceList {
	if item == nil {
		return nil
	}
	out := corev1.ResourceList{}
	if item.CPU != "" {
		if q, err := resource.ParseQuantity(item.CPU); err == nil {
			out[corev1.ResourceCPU] = q
		}
	}
	if item.Memory != "" {
		if q, err := resource.ParseQuantity(item.Memory); err == nil {
			out[corev1.ResourceMemory] = q
		}
	}
	if item.GPU != "" {
		key := item.GPUType
		if key == "" {
			key = string(defaultGPUResourceName)
		}
		if q, err := resource.ParseQuantity(item.GPU); err == nil {
			out[corev1.ResourceName(key)] = q
		}
	}
	for k, v := range item.Custom {
		if q, err := resource.ParseQuantity(v); err == nil {
			out[corev1.ResourceName(k)] = q
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

// mergeEnvs replicates internal/dynamo.MergeEnvs: concatenate `common` and
// `specific`, de-duplicated by Name with `specific` winning on collision.
// Duplicated here to avoid an api -> internal cycle.
func mergeEnvs(common, specific []corev1.EnvVar) []corev1.EnvVar {
	out := make([]corev1.EnvVar, 0, len(common)+len(specific))
	seen := map[string]int{}
	for _, e := range common {
		seen[e.Name] = len(out)
		out = append(out, e)
	}
	for _, e := range specific {
		if idx, ok := seen[e.Name]; ok {
			out[idx] = e
			continue
		}
		seen[e.Name] = len(out)
		out = append(out, e)
	}
	return out
}
