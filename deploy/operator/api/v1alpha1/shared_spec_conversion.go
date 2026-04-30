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
	"reflect"
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

	annotationTrue                 = "true"
	annotationPodTemplateGenerated = "generated"

	// DGD-scoped (DGD spec-level + per-component) annotation keys.
	annDGDPVCs       = "nvidia.com/dgd-pvcs"
	annDGDCompPrefix = "nvidia.com/dgd-comp-"

	// Shared-spec annotation key suffixes. Combined with the parent object's
	// prefix ("nvidia.com/dgd-comp-<name>-" or "nvidia.com/dcd-") to form the
	// full annotation key.
	//
	// suffixServiceName preserves v1alpha1 ServiceName shapes that v1beta1's
	// required ComponentName cannot express: DGD service entries whose
	// ServiceName differs from the services-map key, and standalone DCDs where
	// ServiceName was omitted and v1beta1 used ObjectMeta.Name as the fallback.
	suffixServiceName      = "service-name"
	suffixDynamoNamespace  = "dynamo-namespace"
	suffixAutoscaling      = "autoscaling"
	suffixIngress          = "ingress"
	suffixSubComponentType = "sub-component-type"
	suffixEnvFromSecret    = "env-from-secret"
	suffixAnnotations      = "annotations"
	suffixLabels           = "labels"
	suffixSharedMemOrigin  = "shared-memory-origin"
	suffixResourcesOrigin  = "resources-origin"
	suffixVolumeMountsOrig = "volume-mounts-origin"
	suffixPodTemplateOrig  = "pod-template-origin"
	suffixPodMetadataOrig  = "pod-metadata-origin"
	suffixFrontendSidecar  = "frontend-sidecar-origin"
	// suffixFrontendSidecarRef is used for v1beta1-first inputs that set
	// FrontendSidecar to a container name without a v1alpha1 origin. The
	// referenced container flows through ExtraPodSpec.PodSpec.Containers
	// as a regular user sidecar on the v1alpha1 side.
	suffixFrontendSidecarRef = "frontend-sidecar-ref"
	suffixExperimentalOrig   = "experimental-origin"
	suffixGMSDisabled        = "gms-disabled-payload"
	suffixFailoverDisabled   = "failover-disabled-payload"
	suffixCheckpointDisabled = "checkpoint-disabled-payload"
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

func setJSONAnn(c *annCarrier, suffix string, value any) {
	data, err := json.Marshal(value)
	if err != nil {
		return
	}
	c.set(suffix, string(data))
}

func popJSONAnn[T any](c *annCarrier, suffix string) (T, bool) {
	var out T
	raw, ok := c.get(suffix)
	if !ok {
		return out, false
	}
	c.del(suffix)
	if raw == "" {
		return out, false
	}
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return out, false
	}
	return out, true
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

func setJSONAnnOnObj(obj metav1.Object, key string, value any) bool {
	data, err := json.Marshal(value)
	if err != nil {
		return false
	}
	setAnnOnObj(obj, key, string(data))
	return true
}

func getAnnFromObj(obj metav1.Object, key string) (string, bool) {
	anns := obj.GetAnnotations()
	if anns == nil {
		return "", false
	}
	v, ok := anns[key]
	return v, ok
}

func getJSONAnnFromObj[T any](obj metav1.Object, key string) (T, bool) {
	var out T
	raw, ok := getAnnFromObj(obj, key)
	if !ok || raw == "" {
		return out, false
	}
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return out, false
	}
	return out, true
}

func jsonAnnMatches(obj metav1.Object, key string, value any) bool {
	current, err := json.Marshal(value)
	if err != nil {
		return false
	}
	raw, ok := getAnnFromObj(obj, key)
	return ok && raw == string(current)
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

type sharedSpecConversionContext struct {
	carrier             *annCarrier
	includeOriginSplits bool
}

// convertSharedSpecToHub converts fields represented by both versions from src,
// restores only v1beta1-only fields from restored, and writes v1alpha1-only
// fields to save.
func convertSharedSpecToHub(src *DynamoComponentDeploymentSharedSpec, dst *v1beta1.DynamoComponentDeploymentSharedSpec, restored *v1beta1.DynamoComponentDeploymentSharedSpec, save *DynamoComponentDeploymentSharedSpec, ctx sharedSpecConversionContext) error {
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
		ctx.carrier.set(suffixSubComponentType, src.SubComponentType)
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
		ctx.carrier.set(suffixDynamoNamespace, *src.DynamoNamespace)
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
	preserveV1Alpha1OnlySharedFields(src, ctx.carrier)

	// sharedMemory <-> sharedMemorySize (lossy struct flatten).
	if err := convertSharedMemoryTo(src.SharedMemory, dst, ctx.carrier); err != nil {
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
	convertScalingAdapterTo(src.ScalingAdapter, dst, ctx.carrier)

	// experimental block: gpuMemoryService, failover, checkpoint.
	convertExperimentalTo(src, dst, ctx.carrier)

	// Resources + envs + probes + mainContainer -> podTemplate.containers[main].
	if err := buildPodTemplateTo(src, dst, ctx.carrier, ctx.includeOriginSplits); err != nil {
		return err
	}

	restoreSharedHubOnlyFields(dst, restored, src)
	if save != nil {
		saveSharedAlphaOnlySpec(src, save, ctx.includeOriginSplits)
	}
	return nil
}

func preserveV1Alpha1OnlySharedFields(src *DynamoComponentDeploymentSharedSpec, c *annCarrier) {
	// Autoscaling -> annotation (deprecated, removed in v1beta1).
	if src.Autoscaling != nil {
		setJSONAnn(c, suffixAutoscaling, src.Autoscaling)
	}
	// Ingress -> annotation (removed in v1beta1).
	if src.Ingress != nil {
		setJSONAnn(c, suffixIngress, src.Ingress)
	}
	// EnvFromSecret -> podTemplate.containers[main].envFrom; preserve the
	// pointer string so ConvertFrom can distinguish it from native envFrom.
	if src.EnvFromSecret != nil {
		c.set(suffixEnvFromSecret, *src.EnvFromSecret)
	}
	// Per-service Annotations / Labels apply beyond podTemplate.metadata.
	if len(src.Annotations) > 0 {
		setJSONAnn(c, suffixAnnotations, src.Annotations)
	}
	if len(src.Labels) > 0 {
		setJSONAnn(c, suffixLabels, src.Labels)
	}
	if extraPodMetadataNeedsPreservation(src.ExtraPodMetadata) {
		setJSONAnn(c, suffixPodMetadataOrig, src.ExtraPodMetadata)
	}
	if src.Resources != nil && !reflect.DeepEqual(src.Resources, resourcesFromNative(resourcesToNative(src.Resources))) {
		setJSONAnn(c, suffixResourcesOrigin, src.Resources)
	}
	if len(src.VolumeMounts) > 0 && !volumeMountsRoundTripThroughHub(src.VolumeMounts) {
		setJSONAnn(c, suffixVolumeMountsOrig, src.VolumeMounts)
	}
}

func fillSharedAlphaOnlyFromPreserved(dst *DynamoComponentDeploymentSharedSpec, preserved *DynamoComponentDeploymentSharedSpec) {
	if dst == nil || preserved == nil {
		return
	}
	if dst.SubComponentType == "" {
		dst.SubComponentType = preserved.SubComponentType
	}
	if dst.DynamoNamespace == nil && preserved.DynamoNamespace != nil {
		dst.DynamoNamespace = ptr.To(*preserved.DynamoNamespace)
	}
	if dst.Autoscaling == nil && preserved.Autoscaling != nil {
		cp := *preserved.Autoscaling
		dst.Autoscaling = &cp
	}
	if dst.Ingress == nil && preserved.Ingress != nil {
		cp := *preserved.Ingress
		dst.Ingress = &cp
	}
	if len(dst.Annotations) == 0 {
		dst.Annotations = maps.Clone(preserved.Annotations)
	}
	if len(dst.Labels) == 0 {
		dst.Labels = maps.Clone(preserved.Labels)
	}
	if shouldRestorePreservedSharedMemory(dst.SharedMemory, preserved.SharedMemory) {
		dst.SharedMemory = preserved.SharedMemory.DeepCopy()
	}
	if dst.EnvFromSecret == nil && preserved.EnvFromSecret != nil && *preserved.EnvFromSecret == "" {
		dst.EnvFromSecret = ptr.To("")
	}
	if dst.ExtraPodMetadata == nil && extraPodMetadataNeedsPreservation(preserved.ExtraPodMetadata) {
		dst.ExtraPodMetadata = preserved.ExtraPodMetadata.DeepCopy()
	}
	if dst.ExtraPodSpec == nil && extraPodSpecNeedsPreservation(preserved.ExtraPodSpec) {
		cp := *preserved.ExtraPodSpec.DeepCopy()
		dst.ExtraPodSpec = &cp
	}
	restoreMainContainerFieldOrigins(dst, preserved)
	if dst.ExtraPodSpec != nil && dst.ExtraPodSpec.MainContainer != nil &&
		dst.ExtraPodSpec.MainContainer.Name == "" &&
		preserved.ExtraPodSpec != nil && preserved.ExtraPodSpec.MainContainer != nil {
		dst.ExtraPodSpec.MainContainer.Name = preserved.ExtraPodSpec.MainContainer.Name
	}
	if dst.ScalingAdapter == nil && preserved.ScalingAdapter != nil && !preserved.ScalingAdapter.Enabled {
		dst.ScalingAdapter = preserved.ScalingAdapter.DeepCopy()
	}
}

func saveSharedAlphaOnlySpec(src, save *DynamoComponentDeploymentSharedSpec, includeOriginSplits bool) {
	if src == nil || save == nil {
		return
	}
	hasSave := false

	if src.SubComponentType != "" {
		save.SubComponentType = src.SubComponentType
		hasSave = true
	}
	if src.DynamoNamespace != nil {
		save.DynamoNamespace = ptr.To(*src.DynamoNamespace)
		hasSave = true
	}
	if src.Autoscaling != nil {
		save.Autoscaling = src.Autoscaling.DeepCopy()
		hasSave = true
	}
	if src.Ingress != nil {
		save.Ingress = src.Ingress.DeepCopy()
		hasSave = true
	}
	if len(src.Annotations) > 0 {
		save.Annotations = maps.Clone(src.Annotations)
		hasSave = true
	}
	if len(src.Labels) > 0 {
		save.Labels = maps.Clone(src.Labels)
		hasSave = true
	}
	if src.EnvFromSecret != nil {
		save.EnvFromSecret = ptr.To(*src.EnvFromSecret)
		hasSave = true
	}
	if resourcesNeedPreservation(src.Resources) {
		save.Resources = src.Resources.DeepCopy()
		hasSave = true
	}
	if len(src.VolumeMounts) > 0 && !volumeMountsRoundTripThroughHub(src.VolumeMounts) {
		save.VolumeMounts = slices.Clone(src.VolumeMounts)
		hasSave = true
	}
	if sharedMemoryNeedsPreservation(src.SharedMemory) {
		save.SharedMemory = src.SharedMemory.DeepCopy()
		hasSave = true
	}
	if extraPodMetadataNeedsPreservation(src.ExtraPodMetadata) {
		save.ExtraPodMetadata = src.ExtraPodMetadata.DeepCopy()
		hasSave = true
	}
	if extraPodSpecNeedsPreservation(src.ExtraPodSpec) {
		save.ExtraPodSpec = src.ExtraPodSpec.DeepCopy()
		hasSave = true
	}
	if src.ScalingAdapter != nil && !src.ScalingAdapter.Enabled {
		save.ScalingAdapter = src.ScalingAdapter.DeepCopy()
		hasSave = true
	}
	if includeOriginSplits || hasSave || sharedMainContainerNameNeedsPreservation(src) {
		saveSharedMainContainerOrigins(src, save)
	}
}

func sharedAlphaSpecSaveIsZero(save *DynamoComponentDeploymentSharedSpec) bool {
	return save == nil || apiequality.Semantic.DeepEqual(*save, DynamoComponentDeploymentSharedSpec{})
}

func sharedMainContainerNameNeedsPreservation(src *DynamoComponentDeploymentSharedSpec) bool {
	return src != nil &&
		src.ExtraPodSpec != nil &&
		src.ExtraPodSpec.MainContainer != nil &&
		src.ExtraPodSpec.MainContainer.Name != ""
}

func saveSharedMainContainerOrigins(src, save *DynamoComponentDeploymentSharedSpec) {
	if src == nil || save == nil || src.ExtraPodSpec == nil || src.ExtraPodSpec.MainContainer == nil {
		return
	}
	main := src.ExtraPodSpec.MainContainer

	if main.Name != "" {
		ensureExtraPodSpecMainContainer(save).Name = main.Name
	}
	if len(src.Envs) > 0 || len(main.Env) > 0 {
		save.Envs = slices.Clone(src.Envs)
		ensureExtraPodSpecMainContainer(save).Env = slices.Clone(main.Env)
	}
	if src.EnvFromSecret != nil || len(main.EnvFrom) > 0 {
		if src.EnvFromSecret != nil {
			save.EnvFromSecret = ptr.To(*src.EnvFromSecret)
		}
		ensureExtraPodSpecMainContainer(save).EnvFrom = slices.Clone(main.EnvFrom)
	}
	if src.Resources != nil || !resourceRequirementsEqual(main.Resources, corev1.ResourceRequirements{}) {
		save.Resources = src.Resources.DeepCopy()
		ensureExtraPodSpecMainContainer(save).Resources = *main.Resources.DeepCopy()
	}
	if len(src.VolumeMounts) > 0 || len(main.VolumeMounts) > 0 {
		save.VolumeMounts = slices.Clone(src.VolumeMounts)
		ensureExtraPodSpecMainContainer(save).VolumeMounts = cloneNativeVolumeMounts(main.VolumeMounts)
	}
	if src.LivenessProbe != nil || main.LivenessProbe != nil {
		save.LivenessProbe = src.LivenessProbe.DeepCopy()
		ensureExtraPodSpecMainContainer(save).LivenessProbe = main.LivenessProbe.DeepCopy()
	}
	if src.ReadinessProbe != nil || main.ReadinessProbe != nil {
		save.ReadinessProbe = src.ReadinessProbe.DeepCopy()
		ensureExtraPodSpecMainContainer(save).ReadinessProbe = main.ReadinessProbe.DeepCopy()
	}
}

func sharedMemoryNeedsPreservation(src *SharedMemorySpec) bool {
	if src == nil {
		return false
	}
	if src.Disabled {
		return !src.Size.IsZero()
	}
	return src.Size.IsZero()
}

func shouldRestorePreservedSharedMemory(dst, preserved *SharedMemorySpec) bool {
	if !sharedMemoryNeedsPreservation(preserved) {
		return false
	}
	if dst == nil {
		return true
	}
	if preserved.Disabled {
		return dst.Disabled && dst.Size.Sign() == 0
	}
	return false
}

func resourcesNeedPreservation(src *Resources) bool {
	return src != nil && !reflect.DeepEqual(src, resourcesFromNative(resourcesToNative(src)))
}

func extraPodMetadataNeedsPreservation(src *ExtraPodMetadata) bool {
	return src != nil && len(src.Annotations) == 0 && len(src.Labels) == 0
}

// convertSharedSpecFromHub converts fields represented by both versions from
// src, restores only v1alpha1-only fields from restored, and writes v1beta1-only
// fields to save.
func convertSharedSpecFromHub(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst *DynamoComponentDeploymentSharedSpec, restored *DynamoComponentDeploymentSharedSpec, save *v1beta1.DynamoComponentDeploymentSharedSpec, ctx sharedSpecConversionContext) error {
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
	if v, ok := ctx.carrier.get(suffixSubComponentType); ok {
		dst.SubComponentType = v
		ctx.carrier.del(suffixSubComponentType)
	}
	dst.ServiceName = src.ComponentName
	if v, ok := ctx.carrier.get(suffixDynamoNamespace); ok {
		dst.DynamoNamespace = ptr.To(v)
		ctx.carrier.del(suffixDynamoNamespace)
	}
	if as, ok := popJSONAnn[Autoscaling](ctx.carrier, suffixAutoscaling); ok {
		dst.Autoscaling = &as
	}
	if ing, ok := popJSONAnn[IngressSpec](ctx.carrier, suffixIngress); ok {
		dst.Ingress = &ing
	}
	if m, ok := popJSONAnn[map[string]string](ctx.carrier, suffixAnnotations); ok {
		dst.Annotations = m
	}
	if m, ok := popJSONAnn[map[string]string](ctx.carrier, suffixLabels); ok {
		dst.Labels = m
	}
	// sharedMemorySize -> SharedMemorySpec.
	convertSharedMemoryFrom(src.SharedMemorySize, dst, ctx.carrier)

	// compilationCache + podTemplate volumeMounts -> VolumeMounts.
	convertVolumeMountsFrom(src, dst)

	// experimental -> GPUMemoryService, Failover, Checkpoint.
	convertExperimentalFrom(src, dst, ctx.carrier)

	// scalingAdapter presence -> Enabled=true; annotation -> Enabled=false payload.
	convertScalingAdapterFrom(src.ScalingAdapter, dst, ctx.carrier)

	// podTemplate -> mainContainer + extraPodSpec + extraPodMetadata +
	// Resources + Envs + Probes (+ FrontendSidecar).
	if err := decomposePodTemplate(src, dst, ctx.carrier); err != nil {
		return err
	}

	fillSharedAlphaOnlyFromPreserved(dst, restored)
	if save != nil {
		if err := saveSharedHubOnlySpec(src, dst, save); err != nil {
			return err
		}
	}
	return nil
}

func saveSharedHubOnlySpec(src *v1beta1.DynamoComponentDeploymentSharedSpec, converted *DynamoComponentDeploymentSharedSpec, save *v1beta1.DynamoComponentDeploymentSharedSpec) error {
	if src == nil || save == nil {
		return nil
	}
	if sharedFrontendSidecarNeedsPreservation(src, converted) {
		save.FrontendSidecar = ptr.To(*src.FrontendSidecar)
	}
	if src.PodTemplate != nil {
		if err := saveSharedPodTemplateHubOnlyFields(src, converted, save); err != nil {
			return err
		}
	}
	if experimentalIsHubOnlyShape(src.Experimental) {
		save.Experimental = src.Experimental.DeepCopy()
	}
	return nil
}

func sharedFrontendSidecarNeedsPreservation(src *v1beta1.DynamoComponentDeploymentSharedSpec, converted *DynamoComponentDeploymentSharedSpec) bool {
	if src == nil || src.FrontendSidecar == nil {
		return false
	}
	if converted == nil {
		return true
	}
	return converted.FrontendSidecar == nil
}

func saveSharedPodTemplateHubOnlyFields(src *v1beta1.DynamoComponentDeploymentSharedSpec, converted *DynamoComponentDeploymentSharedSpec, save *v1beta1.DynamoComponentDeploymentSharedSpec) error {
	projected, err := buildSharedPodTemplateFromAlpha(converted, false, false)
	if err != nil {
		return err
	}
	frontendSidecar := ""
	if save.FrontendSidecar != nil {
		frontendSidecar = *save.FrontendSidecar
	}
	save.PodTemplate = sparseSharedHubOnlyPodTemplate(src.PodTemplate, projected, frontendSidecar)
	return nil
}

func sparseSharedHubOnlyPodTemplate(src, projected *corev1.PodTemplateSpec, frontendSidecar string) *corev1.PodTemplateSpec {
	if src == nil {
		return nil
	}
	meta := sharedHubOnlyPodTemplateMetadata(src)
	needsMetadataSave := !apiequality.Semantic.DeepEqual(meta, metav1.ObjectMeta{})
	needsContainerSave := sharedHubOnlyPodTemplateContainersNeedSave(src, projected, frontendSidecar)
	if !needsMetadataSave && !needsContainerSave {
		return nil
	}
	out := &corev1.PodTemplateSpec{}
	if needsMetadataSave {
		out.ObjectMeta = meta
	}
	saveSharedHubOnlyPodTemplateContainers(src, projected, out)
	return nilIfEmptyPodTemplate(out)
}

func sharedHubOnlyPodTemplateMetadata(src *corev1.PodTemplateSpec) metav1.ObjectMeta {
	meta := *src.ObjectMeta.DeepCopy()
	meta.Labels = nil
	meta.Annotations = nil
	return meta
}

func sharedHubOnlyPodTemplateContainersNeedSave(src, projected *corev1.PodTemplateSpec, frontendSidecar string) bool {
	if src == nil {
		return false
	}
	if frontendSidecar != "" && podTemplateHasContainer(src, frontendSidecar) {
		return true
	}
	if projected == nil {
		return len(src.Spec.Containers) > 0
	}
	return !apiequality.Semantic.DeepEqual(src.Spec.Containers, projected.Spec.Containers)
}

func saveSharedHubOnlyPodTemplateContainers(src, projected, save *corev1.PodTemplateSpec) {
	projectedContainers := []corev1.Container(nil)
	if projected != nil {
		projectedContainers = projected.Spec.Containers
	}
	for _, srcContainer := range src.Spec.Containers {
		savedContainer := corev1.Container{Name: srcContainer.Name}
		if projectedContainer, ok := findContainerByName(projectedContainers, srcContainer.Name); ok {
			saveSharedHubOnlyContainerFields(&savedContainer, srcContainer, projectedContainer)
		} else if !containerHasOnlyName(srcContainer) {
			savedContainer = *srcContainer.DeepCopy()
		}
		save.Spec.Containers = append(save.Spec.Containers, savedContainer)
	}
}

func saveSharedHubOnlyContainerFields(save *corev1.Container, src, projected corev1.Container) {
	if src.Name != mainContainerName {
		if !apiequality.Semantic.DeepEqual(src, projected) {
			saveSharedHubOnlyFrontendSidecarContainerFields(save, src, projected)
		}
		return
	}
	if len(src.VolumeMounts) == 0 {
		return
	}
	save.VolumeMounts = sparseSharedHubOnlyVolumeMounts(src.VolumeMounts, projected.VolumeMounts)
}

func saveSharedHubOnlyFrontendSidecarContainerFields(save *corev1.Container, src, projected corev1.Container) {
	*save = *src.DeepCopy()
	clearSharedFrontendSidecarRepresentedContainerFields(save, projected.EnvFrom)
}

func clearSharedFrontendSidecarRepresentedContainerFields(container *corev1.Container, projectedEnvFrom []corev1.EnvFromSource) {
	container.Image = ""
	container.Args = nil
	container.Env = nil
	container.EnvFrom = sparseSharedHubOnlyFrontendSidecarEnvFrom(container.EnvFrom, projectedEnvFrom)
}

func sparseSharedHubOnlyFrontendSidecarEnvFrom(src, projected []corev1.EnvFromSource) []corev1.EnvFromSource {
	if apiequality.Semantic.DeepEqual(src, projected) {
		return nil
	}
	return cloneNativeEnvFromSources(src)
}

func sparseSharedHubOnlyVolumeMounts(src, projected []corev1.VolumeMount) []corev1.VolumeMount {
	out := make([]corev1.VolumeMount, 0, len(src))
	for _, srcMount := range src {
		savedMount := corev1.VolumeMount{
			Name:      srcMount.Name,
			MountPath: srcMount.MountPath,
		}
		if projectedMount, ok := findPreservedVolumeMount(projected, srcMount); !ok ||
			!apiequality.Semantic.DeepEqual(srcMount, projectedMount) {
			savedMount = *srcMount.DeepCopy()
		}
		out = append(out, savedMount)
	}
	return out
}

func sharedHubSpecSaveIsZero(save *v1beta1.DynamoComponentDeploymentSharedSpec) bool {
	return save == nil ||
		save.FrontendSidecar == nil &&
			save.PodTemplate == nil &&
			save.Experimental == nil
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
		if !src.Size.IsZero() {
			setJSONAnn(c, suffixSharedMemOrigin, src)
		}
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
		if src != nil && src.Sign() != 0 {
			dst.SharedMemory = &SharedMemorySpec{Size: src.DeepCopy()}
			return
		}
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
		var sharedMemory SharedMemorySpec
		if err := json.Unmarshal([]byte(v), &sharedMemory); err == nil {
			dst.SharedMemory = &sharedMemory
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
	c.set(suffixScalingDisabled, annotationTrue)
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

func gmsModeToV1beta1(mode GPUMemoryServiceMode) v1beta1.GPUMemoryServiceMode {
	switch mode {
	case GMSModeIntraPod:
		return v1beta1.GMSModeIntraPod
	case GMSModeInterPod:
		return v1beta1.GMSModeInterPod
	default:
		return v1beta1.GPUMemoryServiceMode(mode)
	}
}

func gmsModeFromV1beta1(mode v1beta1.GPUMemoryServiceMode) GPUMemoryServiceMode {
	switch mode {
	case v1beta1.GMSModeIntraPod:
		return GMSModeIntraPod
	case v1beta1.GMSModeInterPod:
		return GMSModeInterPod
	default:
		return GPUMemoryServiceMode(mode)
	}
}

func checkpointModeToV1beta1(mode CheckpointMode) v1beta1.CheckpointMode {
	switch mode {
	case CheckpointModeAuto:
		return v1beta1.CheckpointModeAuto
	case CheckpointModeManual:
		return v1beta1.CheckpointModeManual
	default:
		return v1beta1.CheckpointMode(mode)
	}
}

func checkpointModeFromV1beta1(mode v1beta1.CheckpointMode) CheckpointMode {
	switch mode {
	case v1beta1.CheckpointModeAuto:
		return CheckpointModeAuto
	case v1beta1.CheckpointModeManual:
		return CheckpointModeManual
	default:
		return CheckpointMode(mode)
	}
}

func convertExperimentalTo(src *DynamoComponentDeploymentSharedSpec, dst *v1beta1.DynamoComponentDeploymentSharedSpec, c *annCarrier) {
	var exp *v1beta1.ExperimentalSpec
	ensureExp := func() *v1beta1.ExperimentalSpec {
		if exp == nil {
			exp = &v1beta1.ExperimentalSpec{}
		}
		return exp
	}
	if _, ok := c.get(suffixExperimentalOrig); ok {
		ensureExp()
		c.del(suffixExperimentalOrig)
	}

	if src.GPUMemoryService != nil {
		if src.GPUMemoryService.Enabled {
			ensureExp().GPUMemoryService = &v1beta1.GPUMemoryServiceSpec{
				Mode:            gmsModeToV1beta1(src.GPUMemoryService.Mode),
				DeviceClassName: src.GPUMemoryService.DeviceClassName,
			}
		} else if !gmsIsZeroPayload(src.GPUMemoryService) {
			setJSONAnn(c, suffixGMSDisabled, src.GPUMemoryService)
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
				Mode:       gmsModeToV1beta1(src.Failover.Mode),
				NumShadows: src.Failover.NumShadows,
			}
		} else if !failoverIsZeroPayload(src.Failover) {
			setJSONAnn(c, suffixFailoverDisabled, src.Failover)
		} else {
			c.set(suffixFailoverDisabled, `{}`)
		}
	}

	if src.Checkpoint != nil {
		if src.Checkpoint.Enabled {
			ensureExp().Checkpoint = checkpointToV1beta1(src.Checkpoint)
		} else if !checkpointIsZeroPayload(src.Checkpoint) {
			setJSONAnn(c, suffixCheckpointDisabled, src.Checkpoint)
		} else {
			c.set(suffixCheckpointDisabled, `{}`)
		}
	}

	dst.Experimental = exp
}

func convertExperimentalFrom(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst *DynamoComponentDeploymentSharedSpec, c *annCarrier) {
	// For each experimental sub-block, if v1beta1 declares the feature we
	// drop any stale "*-disabled-payload" annotation: v1beta1 is
	// authoritative in this direction and an enabled v1beta1 entry must
	// not resurrect a previous Enabled:false on the next round-trip.
	if src.Experimental != nil &&
		src.Experimental.GPUMemoryService == nil &&
		src.Experimental.Failover == nil &&
		src.Experimental.Checkpoint == nil {
		c.set(suffixExperimentalOrig, annotationTrue)
	}
	if src.Experimental != nil && src.Experimental.GPUMemoryService != nil {
		dst.GPUMemoryService = &GPUMemoryServiceSpec{
			Enabled:         true,
			Mode:            gmsModeFromV1beta1(src.Experimental.GPUMemoryService.Mode),
			DeviceClassName: src.Experimental.GPUMemoryService.DeviceClassName,
		}
		c.del(suffixGMSDisabled)
	} else if g, ok := popJSONAnn[GPUMemoryServiceSpec](c, suffixGMSDisabled); ok {
		g.Enabled = false
		dst.GPUMemoryService = &g
	}

	if src.Experimental != nil && src.Experimental.Failover != nil {
		dst.Failover = &FailoverSpec{
			Enabled:    true,
			Mode:       gmsModeFromV1beta1(src.Experimental.Failover.Mode),
			NumShadows: src.Experimental.Failover.NumShadows,
		}
		c.del(suffixFailoverDisabled)
	} else if f, ok := popJSONAnn[FailoverSpec](c, suffixFailoverDisabled); ok {
		f.Enabled = false
		dst.Failover = &f
	}

	if src.Experimental != nil && src.Experimental.Checkpoint != nil {
		dst.Checkpoint = checkpointFromV1beta1(src.Experimental.Checkpoint, true)
		c.del(suffixCheckpointDisabled)
	} else if cp, ok := popJSONAnn[ServiceCheckpointConfig](c, suffixCheckpointDisabled); ok {
		cp.Enabled = false
		dst.Checkpoint = &cp
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
		Mode: checkpointModeToV1beta1(src.Mode),
	}
	// Deep-copy CheckpointRef so the v1alpha1 and v1beta1 objects do not
	// share the same *string. A shared pointer would let a mutation on one
	// side surface on the other after conversion, which violates the
	// conversion-webhook contract.
	if src.CheckpointRef != nil {
		dst.CheckpointRef = ptr.To(*src.CheckpointRef)
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
		Enabled: enabled,
		Mode:    checkpointModeFromV1beta1(src.Mode),
	}
	// Deep-copy CheckpointRef -- see checkpointToV1beta1 for the rationale.
	if src.CheckpointRef != nil {
		dst.CheckpointRef = ptr.To(*src.CheckpointRef)
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
func buildPodTemplateTo(src *DynamoComponentDeploymentSharedSpec, dst *v1beta1.DynamoComponentDeploymentSharedSpec, c *annCarrier, includeOriginSplits bool) error {
	_, podTemplateOrigin := c.get(suffixPodTemplateOrig)
	frontendSidecarRef, hasFrontendSidecarRef := c.get(suffixFrontendSidecarRef)
	if hasFrontendSidecarRef && !hasPodTemplateContent(src, podTemplateOrigin) {
		dst.FrontendSidecar = ptr.To(frontendSidecarRef)
		c.del(suffixFrontendSidecarRef)
		return nil
	}
	podTpl, err := buildSharedPodTemplateFromAlpha(src, podTemplateOrigin, hasFrontendSidecarRef)
	if err != nil {
		return err
	}
	if podTpl == nil {
		return nil
	}
	c.del(suffixPodTemplateOrig)

	// FrontendSidecar: full container spec -> name reference; v1beta1-first
	// name references are restored from the carrier.
	setFrontendSidecarReferenceToHub(src, dst, c)

	if !podTemplateOrigin && includeOriginSplits {
		c.set(suffixPodTemplateOrig, annotationPodTemplateGenerated)
	}
	dst.PodTemplate = podTpl
	return nil
}

func hasPodTemplateContent(src *DynamoComponentDeploymentSharedSpec, podTemplateOrigin bool) bool {
	return src.Resources != nil ||
		len(src.Envs) > 0 ||
		src.EnvFromSecret != nil ||
		hasPodTemplateVolumeMounts(src.VolumeMounts) ||
		src.LivenessProbe != nil ||
		src.ReadinessProbe != nil ||
		!extraPodSpecIsZero(src.ExtraPodSpec) ||
		src.ExtraPodMetadata != nil ||
		src.FrontendSidecar != nil ||
		podTemplateOrigin
}

func shouldBuildPodTemplate(src *DynamoComponentDeploymentSharedSpec, podTemplateOrigin, hasFrontendSidecarRef bool) bool {
	return hasPodTemplateContent(src, podTemplateOrigin) || hasFrontendSidecarRef
}

func buildBasePodTemplate(src *DynamoComponentDeploymentSharedSpec) *corev1.PodTemplateSpec {
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
	return podTpl
}

func mergeExtraPodSpecMainContainer(src *DynamoComponentDeploymentSharedSpec, mainBase *corev1.Container) error {
	if src.ExtraPodSpec == nil || src.ExtraPodSpec.MainContainer == nil {
		return nil
	}
	main := src.ExtraPodSpec.MainContainer.DeepCopy()
	baseEnvs := mainBase.Env
	// Name must be "main" regardless of what MainContainer carried.
	main.Name = mainContainerName
	if err := mergo.Merge(mainBase, *main, mergo.WithOverride); err != nil {
		return fmt.Errorf("merge main container: %w", err)
	}
	mainBase.Env = mergeEnvs(baseEnvs, main.Env)
	// StartupProbe has no dedicated v1alpha1 field; take it verbatim.
	if main.StartupProbe != nil {
		mainBase.StartupProbe = main.StartupProbe
	}
	return nil
}

func buildSharedPodTemplateFromAlpha(src *DynamoComponentDeploymentSharedSpec, podTemplateOrigin, hasFrontendSidecarRef bool) (*corev1.PodTemplateSpec, error) {
	if src == nil || !shouldBuildPodTemplate(src, podTemplateOrigin, hasFrontendSidecarRef) {
		return nil, nil
	}
	podTpl := buildBasePodTemplate(src)

	// Main container: base from dedicated fields.
	mainBase := buildMainContainerFromDedicated(src)

	// Merge ExtraPodSpec.MainContainer on top, except for Env which is additive.
	if err := mergeExtraPodSpecMainContainer(src, &mainBase); err != nil {
		return nil, err
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

	if src.FrontendSidecar != nil {
		appendFrontendSidecar(podTpl, src.FrontendSidecar)
	}
	return podTpl, nil
}

func setFrontendSidecarReferenceToHub(src *DynamoComponentDeploymentSharedSpec, dst *v1beta1.DynamoComponentDeploymentSharedSpec, c *annCarrier) {
	if src.FrontendSidecar != nil {
		// Stash origin so ConvertFrom can reproduce the full spec on the
		// v1alpha1 side without depending on the podTemplate sidecar.
		setJSONAnn(c, suffixFrontendSidecar, src.FrontendSidecar)
		dst.FrontendSidecar = ptr.To(defaultFrontendSidecarContainerName)
		// Drop any stale v1beta1-first ref annotation; the origin annotation
		// is authoritative in this direction.
		c.del(suffixFrontendSidecarRef)
		return
	}
	if v, ok := c.get(suffixFrontendSidecarRef); ok {
		// v1beta1-first input: the named container is already present in
		// podTpl.Spec.Containers (via ExtraPodSpec.PodSpec.Containers).
		// Restore the pointer without duplicating the container.
		dst.FrontendSidecar = ptr.To(v)
		c.del(suffixFrontendSidecarRef)
	}
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
		decomposeMissingPodTemplate(src, dst, c)
		return nil
	}

	podTpl := src.PodTemplate.DeepCopy()
	markPodTemplateOrigin(c)

	// ExtraPodMetadata from podTemplate.metadata.
	restoreExtraPodMetadataFromPodTemplate(podTpl, dst, c)

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
	other = restoreFrontendSidecarFromPodTemplate(src, dst, c, other)

	restoreEnvFromSecretFromMain(main, dst, c)
	restoreDedicatedFieldsFromMain(main, dst, c)

	// Put everything non-main into ExtraPodSpec. The main-container fields
	// that v1alpha1 can represent directly have already been moved into their
	// dedicated fields and cleared from main, so ExtraPodSpec only carries the
	// true escape-hatch remainder.
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

func decomposeMissingPodTemplate(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst *DynamoComponentDeploymentSharedSpec, c *annCarrier) {
	if meta, ok := popJSONAnn[ExtraPodMetadata](c, suffixPodMetadataOrig); ok {
		dst.ExtraPodMetadata = &meta
	}
	// Restore FrontendSidecar from origin annotation even if there is no
	// podTemplate (uncommon but possible for manual edits).
	if fs, ok := popJSONAnn[FrontendSidecarSpec](c, suffixFrontendSidecar); ok {
		dst.FrontendSidecar = &fs
	}
	if src.FrontendSidecar != nil {
		c.set(suffixFrontendSidecarRef, *src.FrontendSidecar)
	}
}

func markPodTemplateOrigin(c *annCarrier) {
	if v, ok := c.get(suffixPodTemplateOrig); ok {
		c.del(suffixPodTemplateOrig)
		if v != annotationPodTemplateGenerated {
			c.set(suffixPodTemplateOrig, annotationTrue)
		}
		return
	}
	c.set(suffixPodTemplateOrig, annotationTrue)
}

func restoreExtraPodMetadataFromPodTemplate(podTpl *corev1.PodTemplateSpec, dst *DynamoComponentDeploymentSharedSpec, c *annCarrier) {
	hasMetadata := len(podTpl.Annotations) > 0 || len(podTpl.Labels) > 0
	if v, ok := c.get(suffixPodMetadataOrig); ok {
		var meta ExtraPodMetadata
		if err := json.Unmarshal([]byte(v), &meta); err == nil && !hasMetadata {
			dst.ExtraPodMetadata = &meta
		} else if hasMetadata {
			dst.ExtraPodMetadata = extraPodMetadataFromPodTemplate(podTpl)
		}
		c.del(suffixPodMetadataOrig)
		return
	}
	if hasMetadata {
		dst.ExtraPodMetadata = extraPodMetadataFromPodTemplate(podTpl)
	}
}

func extraPodMetadataFromPodTemplate(podTpl *corev1.PodTemplateSpec) *ExtraPodMetadata {
	return &ExtraPodMetadata{
		Annotations: maps.Clone(podTpl.Annotations),
		Labels:      maps.Clone(podTpl.Labels),
	}
}

func restoreFrontendSidecarFromPodTemplate(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst *DynamoComponentDeploymentSharedSpec, c *annCarrier, other []corev1.Container) []corev1.Container {
	if src.FrontendSidecar == nil {
		return other
	}
	sidecarName := *src.FrontendSidecar
	v, ok := c.get(suffixFrontendSidecar)
	if !ok {
		c.set(suffixFrontendSidecarRef, sidecarName)
		return other
	}
	c.del(suffixFrontendSidecar)

	var fs FrontendSidecarSpec
	if err := json.Unmarshal([]byte(v), &fs); err != nil || sidecarName != defaultFrontendSidecarContainerName {
		c.set(suffixFrontendSidecarRef, sidecarName)
		return other
	}
	ctr, ok := findContainerByName(other, sidecarName)
	if !ok {
		c.set(suffixFrontendSidecarRef, sidecarName)
		return other
	}
	dst.FrontendSidecar = frontendSidecarSpecFromContainer(ctr, &fs)
	return slices.DeleteFunc(other, func(candidate corev1.Container) bool {
		return candidate.Name == sidecarName
	})
}

func restoreSharedHubOnlyFields(dst, preserved *v1beta1.DynamoComponentDeploymentSharedSpec, src *DynamoComponentDeploymentSharedSpec) {
	if dst == nil || preserved == nil {
		return
	}
	dst.PodTemplate = restoreSharedPodTemplateHubOnlyFields(preserved, dst.PodTemplate, dst.CompilationCache, src)
	restoreSharedHubOnlyFrontendSidecar(dst, preserved)
	if dst.Experimental == nil && experimentalIsHubOnlyShape(preserved.Experimental) {
		dst.Experimental = preserved.Experimental.DeepCopy()
	}
}

func restoreSharedHubOnlyFrontendSidecar(dst, preserved *v1beta1.DynamoComponentDeploymentSharedSpec) {
	if dst.FrontendSidecar != nil || preserved.FrontendSidecar == nil {
		return
	}
	if sharedFrontendSidecarHasPreservedPodTemplateContainer(preserved) &&
		!podTemplateHasContainer(dst.PodTemplate, *preserved.FrontendSidecar) {
		return
	}
	dst.FrontendSidecar = ptr.To(*preserved.FrontendSidecar)
}

func sharedFrontendSidecarHasPreservedPodTemplateContainer(preserved *v1beta1.DynamoComponentDeploymentSharedSpec) bool {
	if preserved.PodTemplate == nil || preserved.FrontendSidecar == nil {
		return false
	}
	return podTemplateHasContainer(preserved.PodTemplate, *preserved.FrontendSidecar)
}

func podTemplateHasContainer(podTemplate *corev1.PodTemplateSpec, name string) bool {
	if podTemplate == nil {
		return false
	}
	_, ok := findContainerByName(podTemplate.Spec.Containers, name)
	return ok
}

func restoreSharedPodTemplateHubOnlyFields(preserved *v1beta1.DynamoComponentDeploymentSharedSpec, semantic *corev1.PodTemplateSpec, compilationCache *v1beta1.CompilationCacheConfig, src *DynamoComponentDeploymentSharedSpec) *corev1.PodTemplateSpec {
	if preserved == nil || preserved.PodTemplate == nil {
		if semantic == nil {
			return nil
		}
		return semantic.DeepCopy()
	}
	out := &corev1.PodTemplateSpec{}
	if semantic != nil {
		out = semantic.DeepCopy()
	}
	dropGeneratedCompilationCacheMount(out, preserved.PodTemplate, compilationCache, src)
	dropGeneratedMainContainer(out, preserved.PodTemplate, compilationCache, src)
	restoreSharedPodTemplateMissingHubOnlyContainers(out, preserved.PodTemplate)
	restoreSharedPodTemplateExistingHubOnlyContainers(out, preserved.PodTemplate, src)
	restoreSharedPodTemplateContainerOrder(out, preserved.PodTemplate)
	restoreSharedHubOnlyPodTemplateMetadata(&out.ObjectMeta, preserved.PodTemplate.ObjectMeta)
	restoreSharedHubOnlyFlatVolumeMountFields(out, preserved.PodTemplate, src)
	return nilIfEmptyPodTemplate(out)
}

func restoreSharedPodTemplateMissingHubOnlyContainers(dst, preserved *corev1.PodTemplateSpec) {
	if dst == nil || preserved == nil {
		return
	}
	for _, savedContainer := range preserved.Spec.Containers {
		if containerHasOnlyName(savedContainer) || podTemplateHasContainer(dst, savedContainer.Name) {
			continue
		}
		dst.Spec.Containers = append(dst.Spec.Containers, *savedContainer.DeepCopy())
	}
}

func restoreSharedPodTemplateExistingHubOnlyContainers(dst, preserved *corev1.PodTemplateSpec, src *DynamoComponentDeploymentSharedSpec) {
	if dst == nil || preserved == nil || src == nil || src.FrontendSidecar == nil {
		return
	}
	savedContainer, ok := findContainerByName(preserved.Spec.Containers, defaultFrontendSidecarContainerName)
	if !ok || containerHasOnlyName(savedContainer) {
		return
	}
	for i := range dst.Spec.Containers {
		if dst.Spec.Containers[i].Name != defaultFrontendSidecarContainerName {
			continue
		}
		restoreSharedHubOnlyFrontendSidecarContainerFields(&dst.Spec.Containers[i], savedContainer, src.FrontendSidecar)
		return
	}
}

func restoreSharedHubOnlyFrontendSidecarContainerFields(dst *corev1.Container, preserved corev1.Container, src *FrontendSidecarSpec) {
	saved := preserved.DeepCopy()
	saved.Name = ""
	saved.Image = ""
	saved.Args = nil
	saved.Env = nil
	if len(saved.EnvFrom) > 0 {
		envFrom, ok := restoreSharedHubOnlyFrontendSidecarEnvFrom(dst.EnvFrom, saved.EnvFrom, src)
		if !ok {
			saved.EnvFrom = nil
		} else {
			saved.EnvFrom = envFrom
		}
	}
	_ = mergo.Merge(dst, *saved, mergo.WithOverride)
}

func restoreSharedHubOnlyFrontendSidecarEnvFrom(dst, preserved []corev1.EnvFromSource, src *FrontendSidecarSpec) ([]corev1.EnvFromSource, bool) {
	if src == nil || src.EnvFromSecret == nil || *src.EnvFromSecret == "" {
		return cloneNativeEnvFromSources(preserved), true
	}
	if len(dst) != 1 || len(preserved) != 1 ||
		dst[0].SecretRef == nil || preserved[0].SecretRef == nil ||
		dst[0].SecretRef.Name != *src.EnvFromSecret ||
		preserved[0].SecretRef.Name != *src.EnvFromSecret {
		return nil, false
	}
	out := cloneNativeEnvFromSources(preserved)
	out[0].SecretRef.Name = dst[0].SecretRef.Name
	return out, true
}

func restoreSharedPodTemplateContainerOrder(dst, preserved *corev1.PodTemplateSpec) {
	if dst == nil || preserved == nil || len(preserved.Spec.Containers) < 2 || len(dst.Spec.Containers) < 2 {
		return
	}
	remaining := slices.Clone(dst.Spec.Containers)
	out := make([]corev1.Container, 0, len(dst.Spec.Containers))
	for _, savedContainer := range preserved.Spec.Containers {
		for i := range remaining {
			if remaining[i].Name != savedContainer.Name {
				continue
			}
			out = append(out, remaining[i])
			remaining = slices.Delete(remaining, i, i+1)
			break
		}
	}
	out = append(out, remaining...)
	dst.Spec.Containers = out
}

func dropGeneratedMainContainer(dst, preserved *corev1.PodTemplateSpec, compilationCache *v1beta1.CompilationCacheConfig, src *DynamoComponentDeploymentSharedSpec) {
	if hasContainerNamed(preserved.Spec.Containers, mainContainerName) {
		return
	}
	if src != nil && resourcesNeedPreservation(src.Resources) {
		return
	}
	for i := range dst.Spec.Containers {
		if dst.Spec.Containers[i].Name != mainContainerName {
			continue
		}
		if mainContainerHasOnlyGeneratedFields(dst.Spec.Containers[i], compilationCache) {
			dst.Spec.Containers = slices.Delete(dst.Spec.Containers, i, i+1)
			return
		}
	}
}

func mainContainerHasOnlyGeneratedFields(container corev1.Container, compilationCache *v1beta1.CompilationCacheConfig) bool {
	cp := container.DeepCopy()
	cp.Name = ""
	if compilationCache != nil {
		cp.VolumeMounts = slices.DeleteFunc(cp.VolumeMounts, func(mount corev1.VolumeMount) bool {
			return mount.Name == compilationCache.PVCName && mount.MountPath == compilationCache.MountPath
		})
	}
	return containerIsEmpty(cp)
}

func containerHasOnlyName(container corev1.Container) bool {
	cp := container.DeepCopy()
	cp.Name = ""
	return containerIsEmpty(cp)
}

func dropGeneratedCompilationCacheMount(dst, preserved *corev1.PodTemplateSpec, compilationCache *v1beta1.CompilationCacheConfig, src *DynamoComponentDeploymentSharedSpec) {
	if compilationCache == nil || preservedHasVolumeMount(preserved, compilationCache.PVCName, compilationCache.MountPath) ||
		sourceExtraPodMainContainerVolumeMountMatches(srcExtraPodSpec(src), corev1.VolumeMount{Name: compilationCache.PVCName, MountPath: compilationCache.MountPath}) {
		return
	}
	for i := range dst.Spec.Containers {
		if dst.Spec.Containers[i].Name != mainContainerName {
			continue
		}
		dst.Spec.Containers[i].VolumeMounts = slices.DeleteFunc(dst.Spec.Containers[i].VolumeMounts, func(mount corev1.VolumeMount) bool {
			return mount.Name == compilationCache.PVCName && mount.MountPath == compilationCache.MountPath
		})
		return
	}
}

func preservedHasVolumeMount(preserved *corev1.PodTemplateSpec, name, mountPath string) bool {
	main, ok := findContainerByName(preserved.Spec.Containers, mainContainerName)
	if !ok {
		return false
	}
	for _, mount := range main.VolumeMounts {
		if mount.Name == name && mount.MountPath == mountPath {
			return true
		}
	}
	return false
}

func srcExtraPodSpec(src *DynamoComponentDeploymentSharedSpec) *ExtraPodSpec {
	if src == nil {
		return nil
	}
	return src.ExtraPodSpec
}

func restoreSharedHubOnlyPodTemplateMetadata(dst *metav1.ObjectMeta, preserved metav1.ObjectMeta) {
	labels := maps.Clone(dst.Labels)
	annotations := maps.Clone(dst.Annotations)
	*dst = *preserved.DeepCopy()
	dst.Labels = labels
	dst.Annotations = annotations
}

func restoreSharedHubOnlyFlatVolumeMountFields(dst, preserved *corev1.PodTemplateSpec, src *DynamoComponentDeploymentSharedSpec) {
	if src == nil || len(src.VolumeMounts) == 0 {
		return
	}
	preservedMain, ok := findContainerByName(preserved.Spec.Containers, mainContainerName)
	if !ok || len(preservedMain.VolumeMounts) == 0 {
		return
	}
	for i := range dst.Spec.Containers {
		if dst.Spec.Containers[i].Name != mainContainerName {
			continue
		}
		for j := range dst.Spec.Containers[i].VolumeMounts {
			mount := &dst.Spec.Containers[i].VolumeMounts[j]
			if !sourceFlatVolumeMountMatches(src.VolumeMounts, *mount) ||
				sourceExtraPodMainContainerVolumeMountMatches(src.ExtraPodSpec, *mount) {
				continue
			}
			if preservedMount, ok := findPreservedVolumeMount(preservedMain.VolumeMounts, *mount); ok {
				copyHubOnlyVolumeMountFields(mount, preservedMount)
			}
		}
		return
	}
}

func sourceFlatVolumeMountMatches(mounts []VolumeMount, mount corev1.VolumeMount) bool {
	for _, candidate := range mounts {
		if candidate.Name == mount.Name && candidate.MountPoint == mount.MountPath {
			return true
		}
	}
	return false
}

func sourceExtraPodMainContainerVolumeMountMatches(extraPodSpec *ExtraPodSpec, mount corev1.VolumeMount) bool {
	if extraPodSpec == nil || extraPodSpec.MainContainer == nil {
		return false
	}
	for _, candidate := range extraPodSpec.MainContainer.VolumeMounts {
		if candidate.Name == mount.Name && candidate.MountPath == mount.MountPath {
			return true
		}
	}
	return false
}

func findPreservedVolumeMount(mounts []corev1.VolumeMount, mount corev1.VolumeMount) (corev1.VolumeMount, bool) {
	for _, candidate := range mounts {
		if candidate.Name == mount.Name && candidate.MountPath == mount.MountPath {
			return *candidate.DeepCopy(), true
		}
	}
	var match *corev1.VolumeMount
	for i := range mounts {
		if mounts[i].Name != mount.Name {
			continue
		}
		if match != nil {
			return corev1.VolumeMount{}, false
		}
		match = mounts[i].DeepCopy()
	}
	if match == nil {
		return corev1.VolumeMount{}, false
	}
	return *match, true
}

func copyHubOnlyVolumeMountFields(dst *corev1.VolumeMount, preserved corev1.VolumeMount) {
	name := dst.Name
	mountPath := dst.MountPath
	*dst = *preserved.DeepCopy()
	dst.Name = name
	dst.MountPath = mountPath
}

func hasContainerNamed(containers []corev1.Container, name string) bool {
	for i := range containers {
		if containers[i].Name == name {
			return true
		}
	}
	return false
}

func experimentalIsHubOnlyShape(src *v1beta1.ExperimentalSpec) bool {
	return src != nil &&
		src.GPUMemoryService == nil &&
		src.Failover == nil &&
		src.Checkpoint == nil
}

func nilIfEmptyPodTemplate(podTemplate *corev1.PodTemplateSpec) *corev1.PodTemplateSpec {
	if podTemplate == nil || apiequality.Semantic.DeepEqual(*podTemplate, corev1.PodTemplateSpec{}) {
		return nil
	}
	return podTemplate
}

func resourceRequirementsEqual(a, b corev1.ResourceRequirements) bool {
	return apiequality.Semantic.DeepEqual(a, b)
}

func volumeMountOriginsMatchNative(origin []VolumeMount, native []corev1.VolumeMount) bool {
	if len(origin) != len(native) {
		return false
	}
	for i := range origin {
		if origin[i].Name != native[i].Name || origin[i].MountPoint != native[i].MountPath {
			return false
		}
	}
	return true
}

func restoreEnvFromSecretFromMain(main *corev1.Container, dst *DynamoComponentDeploymentSharedSpec, c *annCarrier) {
	if v, ok := c.get(suffixEnvFromSecret); ok {
		if main != nil && envFromSecretMatches(main.EnvFrom, v) {
			dst.EnvFromSecret = ptr.To(v)
		}
		c.del(suffixEnvFromSecret)
	}
}

func restoreDedicatedFieldsFromMain(main *corev1.Container, dst *DynamoComponentDeploymentSharedSpec, c *annCarrier) {
	if main == nil {
		c.del(suffixResourcesOrigin)
		c.del(suffixVolumeMountsOrig)
		return
	}

	if v, ok := c.get(suffixResourcesOrigin); ok {
		var resources Resources
		if err := json.Unmarshal([]byte(v), &resources); err == nil {
			if resourceRequirementsEqual(main.Resources, resourcesToNative(&resources)) {
				dst.Resources = &resources
				main.Resources = corev1.ResourceRequirements{}
			} else if native := resourcesFromNative(main.Resources); native != nil {
				dst.Resources = native
				main.Resources = corev1.ResourceRequirements{}
			}
		}
		c.del(suffixResourcesOrigin)
	} else if resources := resourcesFromNative(main.Resources); resources != nil {
		dst.Resources = resources
		main.Resources = corev1.ResourceRequirements{}
	}

	if v, ok := c.get(suffixVolumeMountsOrig); ok {
		var volumeMounts []VolumeMount
		if err := json.Unmarshal([]byte(v), &volumeMounts); err == nil {
			if volumeMountOriginsMatchNative(volumeMounts, main.VolumeMounts) {
				dst.VolumeMounts = volumeMounts
				main.VolumeMounts = nil
			} else {
				restoreFlatVolumeMountsFromMain(main, dst)
			}
		}
		c.del(suffixVolumeMountsOrig)
	} else if len(main.VolumeMounts) > 0 {
		restoreFlatVolumeMountsFromMain(main, dst)
	}

	if len(main.Env) > 0 {
		dst.Envs = slices.Clone(main.Env)
		main.Env = nil
	}
	if dst.EnvFromSecret != nil && envFromSecretMatches(main.EnvFrom, *dst.EnvFromSecret) {
		main.EnvFrom = nil
	}
	if main.LivenessProbe != nil {
		dst.LivenessProbe = main.LivenessProbe.DeepCopy()
		main.LivenessProbe = nil
	}
	if main.ReadinessProbe != nil {
		dst.ReadinessProbe = main.ReadinessProbe.DeepCopy()
		main.ReadinessProbe = nil
	}
}

func restoreFlatVolumeMountsFromMain(main *corev1.Container, dst *DynamoComponentDeploymentSharedSpec) {
	if main == nil || len(main.VolumeMounts) == 0 {
		return
	}
	dst.VolumeMounts = appendMissingVolumeMounts(dst.VolumeMounts, volumeMountsFromNative(main.VolumeMounts))
	main.VolumeMounts = nil
}

// containerIsEmpty reports whether c has no user-visible fields set. Used by
// decomposePodTemplate to drop the "main" container synthesized by
// buildPodTemplateTo when v1alpha1 had no main-container fields of its own.
// Name is expected to have been cleared by the caller.
func containerIsEmpty(c *corev1.Container) bool {
	if c == nil {
		return true
	}
	copy := *c
	copy.Name = ""
	normalizeContainerEmptySlices(&copy)
	return apiequality.Semantic.DeepEqual(copy, corev1.Container{})
}

func normalizeContainerEmptySlices(c *corev1.Container) {
	if len(c.Command) == 0 {
		c.Command = nil
	}
	if len(c.Args) == 0 {
		c.Args = nil
	}
	if len(c.Ports) == 0 {
		c.Ports = nil
	}
	if len(c.EnvFrom) == 0 {
		c.EnvFrom = nil
	}
	if len(c.Env) == 0 {
		c.Env = nil
	}
	if len(c.VolumeMounts) == 0 {
		c.VolumeMounts = nil
	}
	if len(c.VolumeDevices) == 0 {
		c.VolumeDevices = nil
	}
	if len(c.ResizePolicy) == 0 {
		c.ResizePolicy = nil
	}
	if len(c.RestartPolicyRules) == 0 {
		c.RestartPolicyRules = nil
	}
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

func findContainerByName(containers []corev1.Container, name string) (corev1.Container, bool) {
	for _, ctr := range containers {
		if ctr.Name == name {
			return *ctr.DeepCopy(), true
		}
	}
	return corev1.Container{}, false
}

func frontendSidecarSpecFromContainer(ctr corev1.Container, origin *FrontendSidecarSpec) *FrontendSidecarSpec {
	out := &FrontendSidecarSpec{}
	if origin != nil {
		out = origin.DeepCopy()
	}
	out.Image = ctr.Image
	out.Args = slices.Clone(ctr.Args)
	out.Envs = slices.Clone(ctr.Env)
	if secretName, ok := frontendSidecarEnvFromSecret(ctr.EnvFrom); ok {
		out.EnvFromSecret = ptr.To(secretName)
	} else if origin != nil && origin.EnvFromSecret != nil && *origin.EnvFromSecret == "" {
		out.EnvFromSecret = ptr.To("")
	} else {
		out.EnvFromSecret = nil
	}
	return out
}

func frontendSidecarEnvFromSecret(envFrom []corev1.EnvFromSource) (string, bool) {
	if len(envFrom) != 1 || envFrom[0].Prefix != "" || envFrom[0].ConfigMapRef != nil || envFrom[0].SecretRef == nil {
		return "", false
	}
	return envFrom[0].SecretRef.Name, true
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

func extraPodSpecIsZero(eps *ExtraPodSpec) bool {
	if eps == nil {
		return true
	}
	return podSpecIsZero(eps.PodSpec) && containerIsEmpty(eps.MainContainer)
}

func extraPodSpecNeedsPreservation(eps *ExtraPodSpec) bool {
	return eps != nil && (extraPodSpecIsZero(eps) || extraPodSpecOnlyPreservesMainContainerName(eps))
}

func extraPodSpecOnlyPreservesMainContainerName(eps *ExtraPodSpec) bool {
	if eps == nil || eps.MainContainer == nil || eps.MainContainer.Name == "" || !podSpecIsZero(eps.PodSpec) {
		return false
	}
	main := eps.MainContainer.DeepCopy()
	main.Name = ""
	return containerIsEmpty(main)
}

func restoreMainContainerFieldOrigins(dst, preserved *DynamoComponentDeploymentSharedSpec) {
	if dst == nil || preserved == nil || preserved.ExtraPodSpec == nil || preserved.ExtraPodSpec.MainContainer == nil {
		return
	}
	preservedMain := preserved.ExtraPodSpec.MainContainer
	currentMain := semanticMainContainer(dst)
	preservedSemanticMain := semanticMainContainer(preserved)

	if apiequality.Semantic.DeepEqual(currentMain.Env, preservedSemanticMain.Env) &&
		(len(preserved.Envs) > 0 || len(preservedMain.Env) > 0) {
		dst.Envs = slices.Clone(preserved.Envs)
		ensureExtraPodSpecMainContainer(dst).Env = slices.Clone(preservedMain.Env)
	}
	if apiequality.Semantic.DeepEqual(currentMain.EnvFrom, preservedSemanticMain.EnvFrom) && preserved.EnvFromSecret != nil {
		dst.EnvFromSecret = ptr.To(*preserved.EnvFromSecret)
	}
	if dst.Resources != nil &&
		resourceRequirementsEqual(currentMain.Resources, preservedSemanticMain.Resources) &&
		(preserved.Resources != nil || !resourceRequirementsEqual(preservedMain.Resources, corev1.ResourceRequirements{})) {
		dst.Resources = preserved.Resources.DeepCopy()
		ensureExtraPodSpecMainContainer(dst).Resources = *preservedMain.Resources.DeepCopy()
	}
	currentVolumeMounts := withoutCompilationCacheMounts(currentMain.VolumeMounts, dst.VolumeMounts)
	if volumeMountOriginsMatchNative(volumeMountsFromNative(preservedSemanticMain.VolumeMounts), currentVolumeMounts) &&
		(len(preserved.VolumeMounts) > 0 || len(preservedMain.VolumeMounts) > 0) {
		dst.VolumeMounts = restorePreservedVolumeMountOrigins(preserved.VolumeMounts, dst.VolumeMounts, preservedMain.VolumeMounts)
		ensureExtraPodSpecMainContainer(dst).VolumeMounts = cloneNativeVolumeMounts(preservedMain.VolumeMounts)
	}
	if apiequality.Semantic.DeepEqual(currentMain.LivenessProbe, preservedSemanticMain.LivenessProbe) &&
		(preserved.LivenessProbe != nil || preservedMain.LivenessProbe != nil) {
		dst.LivenessProbe = preserved.LivenessProbe.DeepCopy()
		ensureExtraPodSpecMainContainer(dst).LivenessProbe = preservedMain.LivenessProbe.DeepCopy()
	}
	if apiequality.Semantic.DeepEqual(currentMain.ReadinessProbe, preservedSemanticMain.ReadinessProbe) &&
		(preserved.ReadinessProbe != nil || preservedMain.ReadinessProbe != nil) {
		dst.ReadinessProbe = preserved.ReadinessProbe.DeepCopy()
		ensureExtraPodSpecMainContainer(dst).ReadinessProbe = preservedMain.ReadinessProbe.DeepCopy()
	}
}

func semanticMainContainer(src *DynamoComponentDeploymentSharedSpec) corev1.Container {
	if src == nil {
		return corev1.Container{Name: mainContainerName}
	}
	main := buildMainContainerFromDedicated(src)
	_ = mergeExtraPodSpecMainContainer(src, &main)
	main.Name = mainContainerName
	return main
}

func ensureExtraPodSpecMainContainer(dst *DynamoComponentDeploymentSharedSpec) *corev1.Container {
	if dst.ExtraPodSpec == nil {
		dst.ExtraPodSpec = &ExtraPodSpec{}
	}
	if dst.ExtraPodSpec.MainContainer == nil {
		dst.ExtraPodSpec.MainContainer = &corev1.Container{}
	}
	return dst.ExtraPodSpec.MainContainer
}

func cloneNativeVolumeMounts(in []corev1.VolumeMount) []corev1.VolumeMount {
	if len(in) == 0 {
		return nil
	}
	out := make([]corev1.VolumeMount, 0, len(in))
	for i := range in {
		out = append(out, *in[i].DeepCopy())
	}
	return out
}

func cloneNativeEnvFromSources(in []corev1.EnvFromSource) []corev1.EnvFromSource {
	if len(in) == 0 {
		return nil
	}
	out := make([]corev1.EnvFromSource, 0, len(in))
	for i := range in {
		out = append(out, *in[i].DeepCopy())
	}
	return out
}

func restorePreservedVolumeMountOrigins(preserved, current []VolumeMount, preservedMain []corev1.VolumeMount) []VolumeMount {
	out := make([]VolumeMount, 0, len(preserved)+len(current))
	for _, mount := range preserved {
		if !mount.UseAsCompilationCache {
			out = append(out, mount)
		}
	}
	for _, mount := range current {
		if !mount.UseAsCompilationCache {
			if nativeVolumeMountHasNamePath(preservedMain, mount.Name, mount.MountPoint) ||
				flatVolumeMountHasNamePath(out, mount.Name, mount.MountPoint) {
				continue
			}
			out = append(out, mount)
			continue
		}
		replaced := false
		for i := range out {
			if out[i].Name == mount.Name && out[i].MountPoint == mount.MountPoint {
				out[i].UseAsCompilationCache = true
				replaced = true
				break
			}
		}
		if !replaced {
			out = append(out, mount)
		}
	}
	return out
}

func nativeVolumeMountHasNamePath(mounts []corev1.VolumeMount, name, mountPath string) bool {
	for _, mount := range mounts {
		if mount.Name == name && mount.MountPath == mountPath {
			return true
		}
	}
	return false
}

func flatVolumeMountHasNamePath(mounts []VolumeMount, name, mountPath string) bool {
	for _, mount := range mounts {
		if mount.Name == name && mount.MountPoint == mountPath {
			return true
		}
	}
	return false
}

func withoutCompilationCacheMounts(mounts []corev1.VolumeMount, flat []VolumeMount) []corev1.VolumeMount {
	if len(mounts) == 0 || len(flat) == 0 {
		return mounts
	}
	out := make([]corev1.VolumeMount, 0, len(mounts))
	for _, mount := range mounts {
		if flatVolumeMountHasCompilationCache(flat, mount.Name, mount.MountPath) {
			continue
		}
		out = append(out, mount)
	}
	return out
}

func flatVolumeMountHasCompilationCache(mounts []VolumeMount, name, mountPath string) bool {
	for _, mount := range mounts {
		if mount.UseAsCompilationCache && mount.Name == name && mount.MountPoint == mountPath {
			return true
		}
	}
	return false
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

func resourcesFromNative(r corev1.ResourceRequirements) *Resources {
	out := &Resources{
		Requests: resourceItemFromList(r.Requests),
		Limits:   resourceItemFromList(r.Limits),
		Claims:   slices.Clone(r.Claims),
	}
	if out.Requests == nil && out.Limits == nil && len(out.Claims) == 0 {
		return nil
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

func resourceItemFromList(list corev1.ResourceList) *ResourceItem {
	if len(list) == 0 {
		return nil
	}
	out := &ResourceItem{}
	for name, q := range list {
		value := q.String()
		switch name {
		case corev1.ResourceCPU:
			out.CPU = value
		case corev1.ResourceMemory:
			out.Memory = value
		case defaultGPUResourceName:
			out.GPU = value
		default:
			if out.Custom == nil {
				out.Custom = map[string]string{}
			}
			out.Custom[string(name)] = value
		}
	}
	return out
}

func volumeMountsFromNative(mounts []corev1.VolumeMount) []VolumeMount {
	if len(mounts) == 0 {
		return nil
	}
	out := make([]VolumeMount, 0, len(mounts))
	for _, mount := range mounts {
		out = append(out, VolumeMount{
			Name:       mount.Name,
			MountPoint: mount.MountPath,
		})
	}
	return out
}

func volumeMountsRoundTripThroughHub(mounts []VolumeMount) bool {
	if len(mounts) == 0 {
		return true
	}
	compilationCacheMounts := 0
	for _, mount := range mounts {
		if mount.UseAsCompilationCache {
			compilationCacheMounts++
		}
	}
	return compilationCacheMounts <= 1
}

func hasPodTemplateVolumeMounts(mounts []VolumeMount) bool {
	for _, mount := range mounts {
		if !mount.UseAsCompilationCache {
			return true
		}
	}
	return false
}

func appendMissingVolumeMounts(dst []VolumeMount, mounts []VolumeMount) []VolumeMount {
	for _, mount := range mounts {
		exists := false
		for _, existing := range dst {
			if existing.Name == mount.Name && existing.MountPoint == mount.MountPoint {
				exists = true
				break
			}
		}
		if !exists {
			dst = append(dst, mount)
		}
	}
	return dst
}

func envFromSecretMatches(envFrom []corev1.EnvFromSource, name string) bool {
	return apiequality.Semantic.DeepEqual(envFrom, []corev1.EnvFromSource{{
		SecretRef: &corev1.SecretEnvSource{
			LocalObjectReference: corev1.LocalObjectReference{Name: name},
		},
	}})
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
	if len(out) == 0 {
		return nil
	}
	return out
}
