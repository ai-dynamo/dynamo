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

// Conversion between v1alpha1 and v1beta1 DynamoGraphDeploymentRequest (DGDR).
//
// The two API versions have fundamentally different shapes: v1alpha1 stores SLA,
// workload, and model-cache configuration inside an opaque JSON blob
// (ProfilingConfig.Config), while v1beta1 breaks these out into typed structs.
// This file bridges the two representations.
//
// # Spec field mapping
//
// 1:1 simple mappings (value copied, path or wrapper type differs):
//
//	v1alpha1                                   v1beta1
//	──────────────────────────────────────────  ──────────────────────────────────────
//	Spec.Model              (string)           Spec.Model                    (string)
//	Spec.Backend            (string)           Spec.Backend              (BackendType)
//	Spec.AutoApply          (bool)             Spec.AutoApply                 (*bool)
//	Spec.UseMocker          (bool)             Spec.Features.Mocker.Enabled    (bool)
//	Spec.ProfilingConfig.ProfilerImage         Spec.Image                    (string)
//	Spec.DeploymentOverrides.WorkersImage      (no v1beta1 equivalent yet — TODO: overrides.dgd)
//
// JSON blob → structured fields (parsed/reconstructed on each trip):
//
//	Blob key path                              v1beta1 field
//	──────────────────────────────────────────  ──────────────────────────────────────
//	sla.ttft                                   Spec.SLA.TTFT             (*float64)
//	sla.itl                                    Spec.SLA.ITL              (*float64)
//	sla.isl                                    Spec.Workload.ISL           (*int32)
//	sla.osl                                    Spec.Workload.OSL           (*int32)
//	deployment.modelCache.pvcName              Spec.ModelCache.PVCName       (string)
//	deployment.modelCache.modelPathInPvc       Spec.ModelCache.PVCModelPath  (string)
//	deployment.modelCache.pvcMountPath         Spec.ModelCache.PVCMountPath  (string)
//
// The full JSON blob is also preserved as the annotation
// nvidia.com/dgdr-profiling-config so that unknown keys survive the round-trip.
// On ConvertFrom the blob is loaded from the annotation first, then the
// structured v1beta1 fields are written on top (structured fields win).
//
// Structural reshaping (same data, different container type):
//
//	v1alpha1                                   v1beta1
//	──────────────────────────────────────────  ──────────────────────────────────────
//	ProfilingConfig.Resources                  Overrides.ProfilingJob.Template.
//	  (*corev1.ResourceRequirements)             Spec.Containers[0].Resources
//	ProfilingConfig.Tolerations                Overrides.ProfilingJob.Template.
//	  ([]corev1.Toleration)                      Spec.Tolerations
//
// Annotation-only (no v1beta1 equivalent; round-tripped via ObjectMeta annotations):
//
//	v1alpha1 field                             Annotation key
//	──────────────────────────────────────────  ──────────────────────────────────────
//	Spec.EnableGPUDiscovery                    nvidia.com/dgdr-enable-gpu-discovery
//	Spec.ProfilingConfig.ConfigMapRef          nvidia.com/dgdr-config-map-ref
//	Spec.ProfilingConfig.OutputPVC             nvidia.com/dgdr-output-pvc
//	Spec.ProfilingConfig.Config (full blob)    nvidia.com/dgdr-profiling-config
//	Spec.DeploymentOverrides.{Name,            nvidia.com/dgdr-deployment-overrides
//	  Namespace,Labels,Annotations}
//
// Planner config (opaque blob stored verbatim under blob["planner"]):
//
//	v1beta1                                    blob key
//	──────────────────────────────────────────  ──────────────────────────────────────
//	Features.Planner (*runtime.RawExtension)   planner.*  (JSON fields written directly)
//
// v1beta1-only fields with no v1alpha1 equivalent (omitted / TODO):
//
//	Hardware.*, Workload.{Concurrency,RequestRate}, SLA.{E2ELatency},
//	Features.{KVRouter}, SearchStrategy
//
// # Status field mapping
//
// 1:1 simple mappings:
//
//	v1alpha1                                   v1beta1
//	──────────────────────────────────────────  ──────────────────────────────────────
//	Status.ObservedGeneration                  Status.ObservedGeneration
//	Status.Conditions                          Status.Conditions
//	Status.GeneratedDeployment                 Status.ProfilingResults.SelectedConfig
//	Status.Deployment.Name                     Status.DGDName
//
// State ↔ Phase (many-to-one, context-dependent):
//
//	v1alpha1 State         → v1beta1 Phase     Notes
//	───────────────────────  ─────────────────  ─────────────────────────────────
//	"" / "Pending"           Pending
//	"Profiling"              Profiling
//	"Ready"                  Ready or Deployed  Deployed if Deployment.Created
//	"Deploying"              Deploying
//	"DeploymentDeleted"      Ready              lossy
//	"Failed"                 Failed
//
//	v1beta1 Phase          → v1alpha1 State
//	───────────────────────  ─────────────────
//	Pending                  "Pending"
//	Profiling                "Profiling"
//	Ready                    "Ready"
//	Deploying                "Deploying"
//	Deployed                 "Ready"            lossy
//	Failed                   "Failed"
//
// Annotation-only status fields (no v1beta1 equivalent):
//
//	v1alpha1 field                             Annotation key
//	──────────────────────────────────────────  ──────────────────────────────────────
//	Status.Backend                             nvidia.com/dgdr-status-backend
//	Status.ProfilingResults (string ref,       nvidia.com/dgdr-profiling-results
//	  e.g. "configmap/<name>")                   (not the v1beta1 struct — see below)
//	Status.Deployment.{Namespace,State,        nvidia.com/dgdr-deployment-status
//	  Created}                                   (JSON-encoded; Name maps to DGDName)
//
// Note on ProfilingResults naming collision: the two versions both have a field
// called "ProfilingResults" but with entirely different types. v1alpha1 has a
// plain string (a configmap reference). v1beta1 has a struct with Pareto and
// SelectedConfig. The string is annotation-preserved; the struct's
// SelectedConfig maps from v1alpha1 GeneratedDeployment (see 1:1 table above).
//
// Note: v1alpha1 DeploymentStatus{Name,Namespace,State,Created} and v1beta1
// DeploymentInfoStatus{Replicas,AvailableReplicas} share no common fields —
// they track different aspects of the DGD. Only Deployment.Name ↔ DGDName
// overlaps. The rest of DeploymentStatus is round-tripped via annotation;
// DeploymentInfoStatus has no v1alpha1 source and is left empty.
//
// v1beta1-only status fields with no v1alpha1 equivalent (omitted / TODO):
//
//	Status.DeploymentInfo.{Replicas,AvailableReplicas},
//	Status.ProfilingPhase, Status.ProfilingJobName,
//	Status.ProfilingResults.Pareto

package v1alpha1

import (
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"slices"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/conversion"
)

// Annotation keys used to round-trip v1alpha1 fields that have no v1beta1 equivalent.
const (
	annDGDRConfigMapRef     = "nvidia.com/dgdr-config-map-ref"
	annDGDROutputPVC        = "nvidia.com/dgdr-output-pvc"
	annDGDREnableGPUDisc    = "nvidia.com/dgdr-enable-gpu-discovery"
	annDGDRDeployOverrides  = "nvidia.com/dgdr-deployment-overrides"
	annDGDRProfilingConfig  = "nvidia.com/dgdr-profiling-config"
	annDGDRStatusBackend    = "nvidia.com/dgdr-status-backend"
	annDGDRProfilingResults = "nvidia.com/dgdr-profiling-results"
	annDGDRDeploymentStatus = "nvidia.com/dgdr-deployment-status"
	annDGDRProfilingJobName = "nvidia.com/dgdr-profiling-job-name"
	annDGDRHubSpec          = "nvidia.com/dgdr-hub-spec"
	annDGDRHubStatus        = "nvidia.com/dgdr-hub-status"
	annDGDRSpokeSpec        = "nvidia.com/dgdr-spoke-spec"
	annDGDRSpokeStatus      = "nvidia.com/dgdr-spoke-status"
	annDGDRSpokeHubStatus   = "nvidia.com/dgdr-spoke-hub-status"
	annDGDRProfilingEmpty   = "nvidia.com/dgdr-profiling-config-empty"
)

// ConvertTo converts this DynamoGraphDeploymentRequest (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoGraphDeploymentRequest) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentRequest but got %T", dstRaw)
	}

	dst.ObjectMeta = src.ObjectMeta
	dst.Annotations = maps.Clone(src.Annotations)

	preservedHubSpec, preservedHubStatus := decodeDGDRHubPreserved(src)
	profilingJobName := src.Annotations[annDGDRProfilingJobName]
	projectedPreservedHub := projectDGDRHubPreservedToAlpha(preservedHubSpec, preservedHubStatus)
	preservedHubSpecUnmodified := preservedHubSpec != nil &&
		reflect.DeepEqual(src.Spec, projectedPreservedHub.Spec)
	preservedHubStatusPhaseUnmodified := preservedHubStatus != nil &&
		dgdrAlphaPhaseEqual(&src.Status, &projectedPreservedHub.Status)
	hubOrigin := preservedHubSpec != nil || preservedHubStatus != nil
	scrubDGDRInternalAnnotations(dst.Annotations)
	if len(dst.Annotations) == 0 {
		dst.Annotations = nil
	}

	if err := convertDGDRSpecTo(&src.Spec, &dst.Spec, dst, preservedHubSpec, preservedHubSpecUnmodified); err != nil {
		return err
	}

	convertDGDRStatusTo(&src.Status, &dst.Status, dst, preservedHubStatus, profilingJobName, preservedHubStatusPhaseUnmodified)
	if !hubOrigin {
		preserveDGDRSpoke(src, dst)
	} else {
		if dgdrAlphaDiffersFromPreservedHub(src, &projectedPreservedHub.Spec, &projectedPreservedHub.Status) {
			preserveDGDRSpoke(src, dst)
		} else {
			scrubDGDRInternalAnnotations(dst.Annotations)
		}
		if len(dst.Annotations) == 0 {
			dst.Annotations = nil
		}
	}

	return nil
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoGraphDeploymentRequest (v1alpha1).
func (dst *DynamoGraphDeploymentRequest) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentRequest but got %T", srcRaw)
	}

	dst.ObjectMeta = src.ObjectMeta
	dst.Annotations = maps.Clone(src.Annotations)

	preservedSpokeSpec, preservedSpokeStatus := decodeDGDRSpokePreserved(src)
	spokeHubStatusUnmodified := dgdrSpokeHubStatusUnmodified(src)
	spokeHubImageUnmodified := dgdrSpokeHubImageUnmodified(src, preservedSpokeSpec)
	spokeHubUnmodified := dgdrSpokeHubUnmodified(src, preservedSpokeSpec, preservedSpokeStatus)
	scrubDGDRInternalAnnotations(dst.Annotations)
	if len(dst.Annotations) == 0 {
		dst.Annotations = nil
	}

	convertDGDRSpecFrom(&src.Spec, &dst.Spec, src, preservedSpokeSpec, spokeHubImageUnmodified)
	convertDGDRStatusFrom(&src.Status, &dst.Status, src, preservedSpokeStatus, spokeHubStatusUnmodified)

	// ProfilingJobName — no v1alpha1 status field; store as annotation for round-trip
	if src.Status.ProfilingJobName != "" {
		if dst.Annotations == nil {
			dst.Annotations = make(map[string]string)
		}
		dst.Annotations[annDGDRProfilingJobName] = src.Status.ProfilingJobName
	}
	if !spokeHubUnmodified && dgdrNeedsHubPreservation(src) {
		preserveDGDRHub(src, dst)
	}

	return nil
}

// setAnnotation initialises the annotation map if needed and sets a key.
func setAnnotation(obj *v1beta1.DynamoGraphDeploymentRequest, key, value string) {
	if obj.Annotations == nil {
		obj.Annotations = make(map[string]string)
	}
	obj.Annotations[key] = value
}

func setAnnotationAlpha(obj *DynamoGraphDeploymentRequest, key, value string) {
	if obj.Annotations == nil {
		obj.Annotations = make(map[string]string)
	}
	obj.Annotations[key] = value
}

type dgdrSpokeSpecPreservation struct {
	Spec               DynamoGraphDeploymentRequestSpec `json:"spec"`
	ProfilingConfigSet bool                             `json:"profilingConfigSet,omitempty"`
	ProfilingConfigRaw []byte                           `json:"profilingConfigRaw,omitempty"`
}

func preserveDGDRSpoke(src *DynamoGraphDeploymentRequest, dst *v1beta1.DynamoGraphDeploymentRequest) {
	specPreserved := false
	envelope := dgdrSpokeSpecPreservation{Spec: src.Spec}
	if src.Spec.ProfilingConfig.Config != nil {
		envelope.ProfilingConfigSet = true
		envelope.ProfilingConfigRaw = slices.Clone(src.Spec.ProfilingConfig.Config.Raw)
		envelope.Spec.ProfilingConfig.Config = nil
	}
	if data, err := json.Marshal(envelope); err == nil {
		setAnnotation(dst, annDGDRSpokeSpec, string(data))
		specPreserved = true
	}
	if data, err := json.Marshal(src.Status); err == nil {
		setAnnotation(dst, annDGDRSpokeStatus, string(data))
	}
	preserveDGDRSpokeHubStatus(dst)
	if specPreserved && src.Spec.ProfilingConfig.Config != nil && len(src.Spec.ProfilingConfig.Config.Raw) == 0 {
		setAnnotation(dst, annDGDRProfilingEmpty, annotationTrue)
	}
}

type dgdrHubStatusFingerprint struct {
	Phase              v1beta1.DGDRPhase             `json:"phase,omitempty"`
	ProfilingPhase     v1beta1.ProfilingPhase        `json:"profilingPhase,omitempty"`
	DGDName            string                        `json:"dgdName,omitempty"`
	ProfilingJobName   string                        `json:"profilingJobName,omitempty"`
	ObservedGeneration int64                         `json:"observedGeneration,omitempty"`
	Conditions         []metav1.Condition            `json:"conditions,omitempty"`
	DeploymentInfo     *v1beta1.DeploymentInfoStatus `json:"deploymentInfo,omitempty"`
}

func preserveDGDRSpokeHubStatus(dst *v1beta1.DynamoGraphDeploymentRequest) {
	data, err := json.Marshal(fingerprintDGDRHubStatus(&dst.Status))
	if err == nil {
		setAnnotation(dst, annDGDRSpokeHubStatus, string(data))
	}
}

func fingerprintDGDRHubStatus(status *v1beta1.DynamoGraphDeploymentRequestStatus) dgdrHubStatusFingerprint {
	return dgdrHubStatusFingerprint{
		Phase:              status.Phase,
		ProfilingPhase:     status.ProfilingPhase,
		DGDName:            status.DGDName,
		ProfilingJobName:   status.ProfilingJobName,
		ObservedGeneration: status.ObservedGeneration,
		Conditions:         status.Conditions,
		DeploymentInfo:     status.DeploymentInfo,
	}
}

func preserveDGDRHub(src *v1beta1.DynamoGraphDeploymentRequest, dst *DynamoGraphDeploymentRequest) {
	if data, err := json.Marshal(src.Spec); err == nil {
		setAnnotationAlpha(dst, annDGDRHubSpec, string(data))
	}
	if data, err := json.Marshal(src.Status); err == nil {
		setAnnotationAlpha(dst, annDGDRHubStatus, string(data))
	}
}

func decodeDGDRHubPreserved(src *DynamoGraphDeploymentRequest) (*v1beta1.DynamoGraphDeploymentRequestSpec, *v1beta1.DynamoGraphDeploymentRequestStatus) {
	if src.Annotations == nil {
		return nil, nil
	}
	var spec *v1beta1.DynamoGraphDeploymentRequestSpec
	if raw, ok := src.Annotations[annDGDRHubSpec]; ok && raw != "" {
		var decoded v1beta1.DynamoGraphDeploymentRequestSpec
		if err := json.Unmarshal([]byte(raw), &decoded); err == nil {
			spec = &decoded
		}
	}
	var status *v1beta1.DynamoGraphDeploymentRequestStatus
	if raw, ok := src.Annotations[annDGDRHubStatus]; ok && raw != "" {
		var decoded v1beta1.DynamoGraphDeploymentRequestStatus
		if err := json.Unmarshal([]byte(raw), &decoded); err == nil {
			status = &decoded
		}
	}
	return spec, status
}

func decodeDGDRSpokePreserved(src *v1beta1.DynamoGraphDeploymentRequest) (*DynamoGraphDeploymentRequestSpec, *DynamoGraphDeploymentRequestStatus) {
	if src.Annotations == nil {
		return nil, nil
	}
	var spec *DynamoGraphDeploymentRequestSpec
	if raw, ok := src.Annotations[annDGDRSpokeSpec]; ok && raw != "" {
		var envelope dgdrSpokeSpecPreservation
		if err := json.Unmarshal([]byte(raw), &envelope); err == nil {
			decoded := envelope.Spec
			if envelope.ProfilingConfigSet {
				decoded.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: slices.Clone(envelope.ProfilingConfigRaw)}
			}
			spec = &decoded
		} else {
			var decoded DynamoGraphDeploymentRequestSpec
			if err := json.Unmarshal([]byte(raw), &decoded); err == nil {
				spec = &decoded
			}
		}
	}
	var status *DynamoGraphDeploymentRequestStatus
	if raw, ok := src.Annotations[annDGDRSpokeStatus]; ok && raw != "" {
		var decoded DynamoGraphDeploymentRequestStatus
		if err := json.Unmarshal([]byte(raw), &decoded); err == nil {
			status = &decoded
		}
	}
	if src.Annotations[annDGDRProfilingEmpty] == annotationTrue {
		if spec == nil {
			spec = &DynamoGraphDeploymentRequestSpec{}
		}
		spec.ProfilingConfig.Config = &apiextensionsv1.JSON{}
	}
	return spec, status
}

func dgdrSpokeHubUnmodified(src *v1beta1.DynamoGraphDeploymentRequest, preservedSpec *DynamoGraphDeploymentRequestSpec, preservedStatus *DynamoGraphDeploymentRequestStatus) bool {
	if preservedSpec == nil && preservedStatus == nil {
		return false
	}
	semantic := &v1beta1.DynamoGraphDeploymentRequest{ObjectMeta: *src.ObjectMeta.DeepCopy()}
	scrubDGDRInternalAnnotations(semantic.Annotations)
	if len(semantic.Annotations) == 0 {
		semantic.Annotations = nil
	}
	if preservedSpec != nil {
		if err := convertDGDRSpecTo(preservedSpec, &semantic.Spec, semantic, nil, false); err != nil {
			return false
		}
	}
	if preservedStatus != nil {
		convertDGDRStatusTo(preservedStatus, &semantic.Status, semantic, nil, "", false)
	}
	return reflect.DeepEqual(src.Spec, semantic.Spec) && reflect.DeepEqual(src.Status, semantic.Status)
}

func dgdrSpokeHubImageUnmodified(src *v1beta1.DynamoGraphDeploymentRequest, preservedSpec *DynamoGraphDeploymentRequestSpec) bool {
	if preservedSpec == nil {
		return false
	}
	semantic := &v1beta1.DynamoGraphDeploymentRequest{ObjectMeta: *src.ObjectMeta.DeepCopy()}
	scrubDGDRInternalAnnotations(semantic.Annotations)
	if len(semantic.Annotations) == 0 {
		semantic.Annotations = nil
	}
	if err := convertDGDRSpecTo(preservedSpec, &semantic.Spec, semantic, nil, false); err != nil {
		return false
	}
	return src.Spec.Image == semantic.Spec.Image
}

func projectDGDRHubPreservedToAlpha(preservedSpec *v1beta1.DynamoGraphDeploymentRequestSpec, preservedStatus *v1beta1.DynamoGraphDeploymentRequestStatus) DynamoGraphDeploymentRequest {
	projected := DynamoGraphDeploymentRequest{}
	if preservedSpec != nil {
		hub := &v1beta1.DynamoGraphDeploymentRequest{Spec: *preservedSpec.DeepCopy()}
		convertDGDRSpecFrom(&hub.Spec, &projected.Spec, hub, nil, false)
	}
	if preservedStatus != nil {
		hub := &v1beta1.DynamoGraphDeploymentRequest{Status: *preservedStatus.DeepCopy()}
		convertDGDRStatusFrom(&hub.Status, &projected.Status, hub, nil, false)
	}
	return projected
}

func dgdrAlphaDiffersFromPreservedHub(src *DynamoGraphDeploymentRequest, projectedSpec *DynamoGraphDeploymentRequestSpec, projectedStatus *DynamoGraphDeploymentRequestStatus) bool {
	return !dgdrAlphaOnlySpecMatchesProjectedHub(src, projectedSpec) ||
		!dgdrAlphaOnlyStatusMatchesProjectedHub(src, projectedStatus)
}

func dgdrAlphaOnlySpecMatchesProjectedHub(src *DynamoGraphDeploymentRequest, projected *DynamoGraphDeploymentRequestSpec) bool {
	if projected == nil {
		projected = &DynamoGraphDeploymentRequestSpec{}
	}
	return reflect.DeepEqual(src.Spec.EnableGPUDiscovery, projected.EnableGPUDiscovery) &&
		reflect.DeepEqual(src.Spec.ProfilingConfig.Config, projected.ProfilingConfig.Config) &&
		reflect.DeepEqual(src.Spec.ProfilingConfig.ConfigMapRef, projected.ProfilingConfig.ConfigMapRef) &&
		src.Spec.ProfilingConfig.OutputPVC == projected.ProfilingConfig.OutputPVC &&
		reflect.DeepEqual(src.Spec.ProfilingConfig.NodeSelector, projected.ProfilingConfig.NodeSelector) &&
		dgdrDeploymentOverrideWorkersImage(src.Spec.DeploymentOverrides) == dgdrDeploymentOverrideWorkersImage(projected.DeploymentOverrides) &&
		dgdrDeploymentOverrideMetadataEqual(src.Spec.DeploymentOverrides, projected.DeploymentOverrides)
}

func dgdrDeploymentOverrideWorkersImage(src *DeploymentOverridesSpec) string {
	if src == nil {
		return ""
	}
	return src.WorkersImage
}

func dgdrDeploymentOverrideMetadataEqual(a, b *DeploymentOverridesSpec) bool {
	var aName, aNamespace, bName, bNamespace string
	var aLabels, bLabels map[string]string
	var aAnnotations, bAnnotations map[string]string
	if a != nil {
		aName = a.Name
		aNamespace = a.Namespace
		aLabels = a.Labels
		aAnnotations = a.Annotations
	}
	if b != nil {
		bName = b.Name
		bNamespace = b.Namespace
		bLabels = b.Labels
		bAnnotations = b.Annotations
	}
	return aName == bName &&
		aNamespace == bNamespace &&
		reflect.DeepEqual(aLabels, bLabels) &&
		reflect.DeepEqual(aAnnotations, bAnnotations)
}

func dgdrAlphaOnlyStatusMatchesProjectedHub(src *DynamoGraphDeploymentRequest, projected *DynamoGraphDeploymentRequestStatus) bool {
	if projected == nil {
		projected = &DynamoGraphDeploymentRequestStatus{}
	}
	return src.Status.Backend == projected.Backend &&
		src.Status.ProfilingResults == projected.ProfilingResults &&
		dgdrDeploymentStatusAlphaOnlyEqual(src.Status.Deployment, projected.Deployment)
}

func dgdrDeploymentStatusAlphaOnlyEqual(a, b *DeploymentStatus) bool {
	var aNamespace, aState, bNamespace, bState string
	if a != nil {
		aNamespace = a.Namespace
		aState = string(a.State)
	}
	if b != nil {
		bNamespace = b.Namespace
		bState = string(b.State)
	}
	return aNamespace == bNamespace && aState == bState
}

func dgdrAlphaPhaseEqual(a, b *DynamoGraphDeploymentRequestStatus) bool {
	return dgdrStateToPhase(string(a.State), a.Deployment) ==
		dgdrStateToPhase(string(b.State), b.Deployment)
}

func dgdrSpokeHubStatusUnmodified(src *v1beta1.DynamoGraphDeploymentRequest) bool {
	if src.Annotations == nil {
		return false
	}
	raw, ok := src.Annotations[annDGDRSpokeHubStatus]
	if !ok || raw == "" {
		return false
	}
	current, err := json.Marshal(fingerprintDGDRHubStatus(&src.Status))
	if err != nil {
		return false
	}
	return string(current) == raw
}

func scrubDGDRInternalAnnotations(annotations map[string]string) {
	for _, key := range []string{
		annDGDRConfigMapRef,
		annDGDROutputPVC,
		annDGDREnableGPUDisc,
		annDGDRDeployOverrides,
		annDGDRProfilingConfig,
		annDGDRStatusBackend,
		annDGDRProfilingResults,
		annDGDRDeploymentStatus,
		annDGDRProfilingJobName,
		annDGDRHubSpec,
		annDGDRHubStatus,
		annDGDRSpokeSpec,
		annDGDRSpokeStatus,
		annDGDRSpokeHubStatus,
		annDGDRProfilingEmpty,
	} {
		delete(annotations, key)
	}
}

// convertDGDRSpecTo converts the v1alpha1 Spec into the v1beta1 Spec. The
// preserved hub spec is old-value context for fields v1alpha1 cannot express.
func convertDGDRSpecTo(src *DynamoGraphDeploymentRequestSpec, dst *v1beta1.DynamoGraphDeploymentRequestSpec, dstObj *v1beta1.DynamoGraphDeploymentRequest, preserved *v1beta1.DynamoGraphDeploymentRequestSpec, preservedSourceUnmodified bool) error {
	dst.Model = src.Model
	dst.AutoApply = &src.AutoApply

	if src.Backend != "" {
		dst.Backend = v1beta1.BackendType(src.Backend)
	}
	if src.DeploymentOverrides != nil && src.DeploymentOverrides.WorkersImage != "" {
		dst.Image = src.DeploymentOverrides.WorkersImage
	}
	if src.UseMocker {
		if dst.Features == nil {
			dst.Features = &v1beta1.FeaturesSpec{}
		}
		dst.Features.Mocker = &v1beta1.MockerSpec{Enabled: true}
	}
	if src.EnableGPUDiscovery != nil && *src.EnableGPUDiscovery {
		setAnnotation(dstObj, annDGDREnableGPUDisc, annotationTrue)
	}

	if src.ProfilingConfig.Config != nil && len(src.ProfilingConfig.Config.Raw) > 0 {
		setAnnotation(dstObj, annDGDRProfilingConfig, string(src.ProfilingConfig.Config.Raw))

		var blob map[string]interface{}
		if err := json.Unmarshal(src.ProfilingConfig.Config.Raw, &blob); err != nil {
			// ProfilingConfig.Config is an opaque JSON extension point on the
			// v1alpha1 side. Generic fuzzers may populate it with arbitrary
			// RawExtension bytes; preserve those bytes via the annotation but
			// only project typed v1beta1 fields when it is a JSON object in the
			// legacy shape we understand.
			blob = nil
		}
		if blob != nil {
			applySLAAndWorkloadFromBlob(blob, dst)
			applyModelCacheFromBlob(blob, dst)
			applyPlannerFromBlob(blob, dst)
		}
	}

	// ProfilerImage → Image (the profiler runs in the frontend image)
	// TODO: In a future MR, backend inference images will be managed separately via overrides.dgd.
	if src.ProfilingConfig.ProfilerImage != "" {
		dst.Image = src.ProfilingConfig.ProfilerImage
	}
	if src.ProfilingConfig.ConfigMapRef != nil {
		if data, err := json.Marshal(src.ProfilingConfig.ConfigMapRef); err == nil {
			setAnnotation(dstObj, annDGDRConfigMapRef, string(data))
		}
	}
	if src.ProfilingConfig.OutputPVC != "" {
		setAnnotation(dstObj, annDGDROutputPVC, src.ProfilingConfig.OutputPVC)
	}

	convertProfilingResourcesToOverrides(&src.ProfilingConfig, dst)
	convertDeploymentOverridesToAnnotation(src.DeploymentOverrides, dstObj)
	fillDGDRHubSpecFromPreserved(dst, preserved, preservedSourceUnmodified)

	return nil
}

// applySLAAndWorkloadFromBlob extracts SLA and Workload fields from the v1alpha1 JSON blob.
// Both are nested under blob["sla"] in the v1alpha1 schema.
func applySLAAndWorkloadFromBlob(blob map[string]interface{}, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
	slaRaw, ok := blob["sla"]
	if !ok {
		return
	}
	slaMap, ok := slaRaw.(map[string]interface{})
	if !ok {
		return
	}

	if dst.SLA == nil {
		dst.SLA = &v1beta1.SLASpec{}
	}
	if v, ok := slaMap["ttft"].(float64); ok {
		dst.SLA.TTFT = &v
	}
	if v, ok := slaMap["itl"].(float64); ok {
		dst.SLA.ITL = &v
	}
	if v, ok := slaMap["optimizationType"].(string); ok {
		ot := v1beta1.OptimizationType(v)
		if ot == v1beta1.OptimizationTypeLatency || ot == v1beta1.OptimizationTypeThroughput {
			dst.SLA.OptimizationType = &ot
		}
	}

	if v, ok := slaMap["isl"].(float64); ok {
		if dst.Workload == nil {
			dst.Workload = &v1beta1.WorkloadSpec{}
		}
		isl := int32(v)
		dst.Workload.ISL = &isl
	}
	if v, ok := slaMap["osl"].(float64); ok {
		if dst.Workload == nil {
			dst.Workload = &v1beta1.WorkloadSpec{}
		}
		osl := int32(v)
		dst.Workload.OSL = &osl
	}
}

// applyModelCacheFromBlob extracts ModelCache from blob["deployment"]["modelCache"].
func applyModelCacheFromBlob(blob map[string]interface{}, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
	deployRaw, ok := blob["deployment"]
	if !ok {
		return
	}
	deployMap, ok := deployRaw.(map[string]interface{})
	if !ok {
		return
	}
	mcRaw, ok := deployMap["modelCache"]
	if !ok {
		return
	}
	mcMap, ok := mcRaw.(map[string]interface{})
	if !ok {
		return
	}

	mc := &v1beta1.ModelCacheSpec{}
	if v, ok := mcMap["pvcName"].(string); ok {
		mc.PVCName = v
	}
	if v, ok := mcMap["modelPathInPvc"].(string); ok {
		mc.PVCModelPath = v
	}
	if v, ok := mcMap["pvcMountPath"].(string); ok {
		mc.PVCMountPath = v
	}
	dst.ModelCache = mc
}

// convertProfilingResourcesToOverrides maps ProfilingConfig Resources and Tolerations
// into the v1beta1 Overrides.ProfilingJob pod spec.
func convertProfilingResourcesToOverrides(src *ProfilingConfigSpec, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
	if src.Resources == nil && len(src.Tolerations) == 0 {
		return
	}
	if dst.Overrides == nil {
		dst.Overrides = &v1beta1.OverridesSpec{}
	}
	if dst.Overrides.ProfilingJob == nil {
		dst.Overrides.ProfilingJob = &batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{},
			},
		}
	}
	podSpec := &dst.Overrides.ProfilingJob.Template.Spec

	if src.Resources != nil {
		if len(podSpec.Containers) == 0 {
			podSpec.Containers = []corev1.Container{{}}
		}
		podSpec.Containers[0].Resources = *src.Resources
	}
	if len(src.Tolerations) > 0 {
		podSpec.Tolerations = src.Tolerations
	}
}

// convertDeploymentOverridesToAnnotation serialises the DeploymentOverrides metadata fields
// (Name, Namespace, Labels, Annotations) into an annotation for round-trip.
// WorkersImage is handled separately via dst.Image.
func convertDeploymentOverridesToAnnotation(src *DeploymentOverridesSpec, dstObj *v1beta1.DynamoGraphDeploymentRequest) {
	if src == nil {
		return
	}
	overrides := struct {
		Name        string            `json:"name,omitempty"`
		Namespace   string            `json:"namespace,omitempty"`
		Labels      map[string]string `json:"labels,omitempty"`
		Annotations map[string]string `json:"annotations,omitempty"`
	}{
		Name:        src.Name,
		Namespace:   src.Namespace,
		Labels:      src.Labels,
		Annotations: src.Annotations,
	}
	if overrides.Name == "" && overrides.Namespace == "" && len(overrides.Labels) == 0 && len(overrides.Annotations) == 0 {
		return
	}
	if data, err := json.Marshal(overrides); err == nil {
		setAnnotation(dstObj, annDGDRDeployOverrides, string(data))
	}
}

func fillDGDRHubSpecFromPreserved(dst *v1beta1.DynamoGraphDeploymentRequestSpec, preserved *v1beta1.DynamoGraphDeploymentRequestSpec, preservedSourceUnmodified bool) {
	if preserved == nil {
		return
	}
	// Preserve exact hub round-trip shape: the profiling blob can synthesize
	// empty SLA/defaulted AutoApply values that differ from the preserved nils.
	if preserved.SLA == nil && dst.SLA != nil &&
		dst.SLA.TTFT == nil && dst.SLA.ITL == nil && dst.SLA.E2ELatency == nil {
		dst.SLA = nil
	}
	if preserved.AutoApply == nil && dst.AutoApply != nil && *dst.AutoApply {
		dst.AutoApply = nil
	}
	if preserved.Hardware != nil {
		dst.Hardware = preserved.Hardware
	}
	if preserved.Features != nil && preserved.Features.Mocker != nil && !preserved.Features.Mocker.Enabled {
		if dst.Features == nil {
			dst.Features = &v1beta1.FeaturesSpec{}
		}
		if dst.Features.Mocker == nil {
			dst.Features.Mocker = preserved.Features.Mocker
		}
	}
	if preserved.SearchStrategy != "" {
		dst.SearchStrategy = preserved.SearchStrategy
	}
	if preservedSourceUnmodified && dst.ModelCache == nil && preserved.ModelCache != nil {
		dst.ModelCache = preserved.ModelCache
	}
	if preserved.SLA != nil && preserved.SLA.E2ELatency != nil {
		if dst.SLA == nil {
			dst.SLA = &v1beta1.SLASpec{}
		}
		dst.SLA.E2ELatency = preserved.SLA.E2ELatency
	}
	if preserved.Workload != nil {
		if dst.Workload == nil {
			dst.Workload = &v1beta1.WorkloadSpec{}
		}
		dst.Workload.Concurrency = preserved.Workload.Concurrency
		dst.Workload.RequestRate = preserved.Workload.RequestRate
	}
	if preserved.Overrides == nil {
		return
	}
	if dst.Overrides == nil {
		dst.Overrides = &v1beta1.OverridesSpec{}
	}
	if preserved.Overrides.DGD != nil {
		dst.Overrides.DGD = preserved.Overrides.DGD
	}
	if preserved.Overrides.ProfilingJob != nil {
		semanticJob := dst.Overrides.ProfilingJob
		dst.Overrides.ProfilingJob = preserved.Overrides.ProfilingJob.DeepCopy()
		overlayRepresentedProfilingJobFields(dst.Overrides.ProfilingJob, semanticJob)
	}
}

func overlayRepresentedProfilingJobFields(dst, semantic *batchv1.JobSpec) {
	if dst == nil {
		return
	}
	podSpec := &dst.Template.Spec
	if semantic == nil {
		podSpec.Tolerations = nil
		if len(podSpec.Containers) > 0 {
			podSpec.Containers[0].Resources.Requests = nil
			podSpec.Containers[0].Resources.Limits = nil
		}
		return
	}
	semanticPodSpec := &semantic.Template.Spec
	podSpec.Tolerations = semanticPodSpec.Tolerations
	if len(semanticPodSpec.Containers) > 0 {
		if len(podSpec.Containers) == 0 {
			podSpec.Containers = []corev1.Container{{}}
		}
		podSpec.Containers[0].Resources.Requests = semanticPodSpec.Containers[0].Resources.Requests
		podSpec.Containers[0].Resources.Limits = semanticPodSpec.Containers[0].Resources.Limits
		podSpec.Containers[0].Resources.Claims = semanticPodSpec.Containers[0].Resources.Claims
	} else if len(podSpec.Containers) > 0 {
		podSpec.Containers[0].Resources.Requests = nil
		podSpec.Containers[0].Resources.Limits = nil
		podSpec.Containers[0].Resources.Claims = nil
	}
}

// convertDGDRSpecFrom converts the v1beta1 Spec back into the v1alpha1 Spec.
func convertDGDRSpecFrom(src *v1beta1.DynamoGraphDeploymentRequestSpec, dst *DynamoGraphDeploymentRequestSpec, srcObj *v1beta1.DynamoGraphDeploymentRequest, preserved *DynamoGraphDeploymentRequestSpec, preservedSourceUnmodified bool) {
	dst.Model = src.Model
	if src.AutoApply != nil {
		dst.AutoApply = *src.AutoApply
	} else {
		dst.AutoApply = true // v1beta1 default
	}

	if src.Backend != "" {
		dst.Backend = string(src.Backend)
	}
	if src.Features != nil && src.Features.Mocker != nil {
		dst.UseMocker = src.Features.Mocker.Enabled
	}

	if srcObj.Annotations != nil {
		if v, ok := srcObj.Annotations[annDGDREnableGPUDisc]; ok && v == annotationTrue {
			trueVal := true
			dst.EnableGPUDiscovery = &trueVal
		}
	}

	// Reconstruct the JSON blob: start from the round-trip annotation (preserves unknown
	// keys), then overwrite with structured v1beta1 fields (structured fields win).
	var blob map[string]interface{}
	blobFromAnnotation := false
	if srcObj.Annotations != nil {
		if rawBlob, ok := srcObj.Annotations[annDGDRProfilingConfig]; ok && rawBlob != "" {
			dst.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: []byte(rawBlob)}
			_ = json.Unmarshal([]byte(rawBlob), &blob) // best-effort
			blobFromAnnotation = true
		}
	}
	if src.SLA != nil || src.Workload != nil {
		if blob == nil {
			blob = make(map[string]interface{})
		}
		mergeSLAWorkloadIntoBlob(src, blob, !blobFromAnnotation)
	}
	if src.ModelCache != nil {
		if blob == nil {
			blob = make(map[string]interface{})
		}
		mergeModelCacheIntoBlob(src.ModelCache, blob)
	}
	if src.Features != nil && src.Features.Planner != nil {
		if blob == nil {
			blob = make(map[string]interface{})
		}
		mergePlannerIntoBlob(src.Features.Planner, blob)
	}
	if blob != nil {
		if data, err := json.Marshal(blob); err == nil {
			dst.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: data}
		}
	}

	// Image → ProfilerImage (round-trip; see ConvertTo for rationale)
	// TODO: In a future MR, backend images will come from overrides.dgd; worker image
	//       (v1alpha1 DeploymentOverrides.WorkersImage) has no v1beta1 equivalent yet.
	if src.Image != "" {
		dst.ProfilingConfig.ProfilerImage = src.Image
	}

	restoreAnnotationFields(srcObj, dst)
	restoreProfilingJobResources(src, dst)
	fillDGDRSpokeSpecFromPreserved(dst, preserved, preservedSourceUnmodified)
}

// mergeSLAWorkloadIntoBlob writes SLA and Workload structured fields back into the JSON blob,
// overwriting any existing values for those keys.
func mergeSLAWorkloadIntoBlob(src *v1beta1.DynamoGraphDeploymentRequestSpec, blob map[string]interface{}, preserveEmpty bool) {
	slaMap, hadSLA := blob["sla"].(map[string]interface{})
	if slaMap == nil {
		slaMap = make(map[string]interface{})
	}
	wrote := false
	if src.SLA != nil {
		if src.SLA.TTFT != nil {
			slaMap["ttft"] = *src.SLA.TTFT
			wrote = true
		}
		if src.SLA.ITL != nil {
			slaMap["itl"] = *src.SLA.ITL
			wrote = true
		}
		if src.SLA.OptimizationType != nil {
			slaMap["optimizationType"] = string(*src.SLA.OptimizationType)
		}
	}
	if src.Workload != nil {
		if src.Workload.ISL != nil {
			slaMap["isl"] = float64(*src.Workload.ISL)
			wrote = true
		}
		if src.Workload.OSL != nil {
			slaMap["osl"] = float64(*src.Workload.OSL)
			wrote = true
		}
	}
	emptySLA := src.SLA != nil && src.SLA.TTFT == nil && src.SLA.ITL == nil && src.SLA.E2ELatency == nil
	if wrote || hadSLA || preserveEmpty || emptySLA {
		blob["sla"] = slaMap
	}
}

// mergeModelCacheIntoBlob writes ModelCache structured fields back into blob["deployment"]["modelCache"].
func mergeModelCacheIntoBlob(mc *v1beta1.ModelCacheSpec, blob map[string]interface{}) {
	deployMap, _ := blob["deployment"].(map[string]interface{})
	if deployMap == nil {
		deployMap = make(map[string]interface{})
	}
	mcMap := make(map[string]interface{})
	if mc.PVCName != "" {
		mcMap["pvcName"] = mc.PVCName
	}
	if mc.PVCModelPath != "" {
		mcMap["modelPathInPvc"] = mc.PVCModelPath
	}
	if mc.PVCMountPath != "" {
		mcMap["pvcMountPath"] = mc.PVCMountPath
	}
	if len(mcMap) > 0 {
		deployMap["modelCache"] = mcMap
		blob["deployment"] = deployMap
	}
}

// mergePlannerIntoBlob writes the planner RawExtension into blob["planner"].
// The RawExtension is the full PlannerConfig JSON blob (opaque to Go).
func mergePlannerIntoBlob(planner *runtime.RawExtension, blob map[string]interface{}) {
	if planner == nil || planner.Raw == nil {
		return
	}
	var plannerMap map[string]interface{}
	if err := json.Unmarshal(planner.Raw, &plannerMap); err != nil || len(plannerMap) == 0 {
		return
	}
	blob["planner"] = plannerMap
}

// applyPlannerFromBlob extracts blob["planner"] and populates v1beta1 Features.Planner.
func applyPlannerFromBlob(blob map[string]interface{}, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
	plannerRaw, ok := blob["planner"]
	if !ok {
		return
	}
	plannerMap, ok := plannerRaw.(map[string]interface{})
	if !ok || len(plannerMap) == 0 {
		return
	}
	raw, err := json.Marshal(plannerMap)
	if err != nil {
		return
	}
	if dst.Features == nil {
		dst.Features = &v1beta1.FeaturesSpec{}
	}
	dst.Features.Planner = &runtime.RawExtension{Raw: raw}
}

// restoreAnnotationFields restores v1alpha1 spec fields that were annotation-preserved
// during ConvertTo: ConfigMapRef, OutputPVC, and DeploymentOverrides.
func restoreAnnotationFields(srcObj *v1beta1.DynamoGraphDeploymentRequest, dst *DynamoGraphDeploymentRequestSpec) {
	if srcObj.Annotations == nil {
		return
	}
	if v, ok := srcObj.Annotations[annDGDRConfigMapRef]; ok && v != "" {
		var ref ConfigMapKeySelector
		if err := json.Unmarshal([]byte(v), &ref); err == nil {
			dst.ProfilingConfig.ConfigMapRef = &ref
		}
	}
	if v, ok := srcObj.Annotations[annDGDROutputPVC]; ok {
		dst.ProfilingConfig.OutputPVC = v
	}
	if v, ok := srcObj.Annotations[annDGDRDeployOverrides]; ok && v != "" {
		var overrides struct {
			Name        string            `json:"name,omitempty"`
			Namespace   string            `json:"namespace,omitempty"`
			Labels      map[string]string `json:"labels,omitempty"`
			Annotations map[string]string `json:"annotations,omitempty"`
		}
		if err := json.Unmarshal([]byte(v), &overrides); err == nil {
			if dst.DeploymentOverrides == nil {
				dst.DeploymentOverrides = &DeploymentOverridesSpec{}
			}
			dst.DeploymentOverrides.Name = overrides.Name
			dst.DeploymentOverrides.Namespace = overrides.Namespace
			dst.DeploymentOverrides.Labels = overrides.Labels
			dst.DeploymentOverrides.Annotations = overrides.Annotations
		}
	}
}

// restoreProfilingJobResources restores Resources and Tolerations from
// v1beta1 Overrides.ProfilingJob back into v1alpha1 ProfilingConfig.
func restoreProfilingJobResources(src *v1beta1.DynamoGraphDeploymentRequestSpec, dst *DynamoGraphDeploymentRequestSpec) {
	if src.Overrides == nil || src.Overrides.ProfilingJob == nil {
		return
	}
	podSpec := &src.Overrides.ProfilingJob.Template.Spec
	if len(podSpec.Containers) > 0 {
		res := podSpec.Containers[0].Resources
		if len(res.Requests) > 0 || len(res.Limits) > 0 || len(res.Claims) > 0 {
			dst.ProfilingConfig.Resources = &res
		}
	}
	if len(podSpec.Tolerations) > 0 {
		dst.ProfilingConfig.Tolerations = podSpec.Tolerations
	}
}

func fillDGDRSpokeSpecFromPreserved(dst *DynamoGraphDeploymentRequestSpec, preserved *DynamoGraphDeploymentRequestSpec, preservedSourceUnmodified bool) {
	if preserved == nil {
		return
	}
	if dst.EnableGPUDiscovery == nil {
		dst.EnableGPUDiscovery = preserved.EnableGPUDiscovery
	}
	if dst.ProfilingConfig.Config == nil && preserved.ProfilingConfig.Config != nil {
		dst.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: slices.Clone(preserved.ProfilingConfig.Config.Raw)}
	}
	if dst.ProfilingConfig.ConfigMapRef == nil && preserved.ProfilingConfig.ConfigMapRef != nil {
		cp := *preserved.ProfilingConfig.ConfigMapRef
		dst.ProfilingConfig.ConfigMapRef = &cp
	}
	if dst.ProfilingConfig.OutputPVC == "" {
		dst.ProfilingConfig.OutputPVC = preserved.ProfilingConfig.OutputPVC
	}
	if dst.ProfilingConfig.Resources == nil && preserved.ProfilingConfig.Resources != nil {
		res := preserved.ProfilingConfig.Resources.DeepCopy()
		dst.ProfilingConfig.Resources = res
	}
	if len(dst.ProfilingConfig.Tolerations) == 0 {
		dst.ProfilingConfig.Tolerations = preserved.ProfilingConfig.Tolerations
	}
	if len(dst.ProfilingConfig.NodeSelector) == 0 {
		dst.ProfilingConfig.NodeSelector = maps.Clone(preserved.ProfilingConfig.NodeSelector)
	}
	if preservedSourceUnmodified && preserved.ProfilingConfig.ProfilerImage != "" {
		dst.ProfilingConfig.ProfilerImage = preserved.ProfilingConfig.ProfilerImage
	}
	hasPreservedWorkersImage := preserved.DeploymentOverrides != nil &&
		preserved.DeploymentOverrides.WorkersImage != "" &&
		(preservedSourceUnmodified ||
			dst.ProfilingConfig.ProfilerImage == preserved.DeploymentOverrides.WorkersImage)
	clearProfilerImage := hasPreservedWorkersImage &&
		preserved.ProfilingConfig.ProfilerImage == "" &&
		dst.ProfilingConfig.ProfilerImage == preserved.DeploymentOverrides.WorkersImage
	// ProfilerImage and DeploymentOverrides.WorkersImage both collapse to
	// hub Image. If Image came from WorkersImage, do not invent ProfilerImage.
	if clearProfilerImage {
		dst.ProfilingConfig.ProfilerImage = ""
	}
	fillDGDRDeploymentOverridesFromPreserved(dst, preserved, hasPreservedWorkersImage)
}

func fillDGDRDeploymentOverridesFromPreserved(dst *DynamoGraphDeploymentRequestSpec, preserved *DynamoGraphDeploymentRequestSpec, restoreWorkersImage bool) {
	if preserved.DeploymentOverrides == nil {
		return
	}
	if dst.DeploymentOverrides == nil {
		dst.DeploymentOverrides = &DeploymentOverridesSpec{}
	}
	if restoreWorkersImage && dst.DeploymentOverrides.WorkersImage == "" {
		dst.DeploymentOverrides.WorkersImage = preserved.DeploymentOverrides.WorkersImage
	}
	if dst.DeploymentOverrides.Name == "" {
		dst.DeploymentOverrides.Name = preserved.DeploymentOverrides.Name
	}
	if dst.DeploymentOverrides.Namespace == "" {
		dst.DeploymentOverrides.Namespace = preserved.DeploymentOverrides.Namespace
	}
	if len(dst.DeploymentOverrides.Labels) == 0 {
		dst.DeploymentOverrides.Labels = preserved.DeploymentOverrides.Labels
	}
	if len(dst.DeploymentOverrides.Annotations) == 0 {
		dst.DeploymentOverrides.Annotations = preserved.DeploymentOverrides.Annotations
	}
}

// convertDGDRStatusTo converts the v1alpha1 Status into the v1beta1 Status.
func convertDGDRStatusTo(src *DynamoGraphDeploymentRequestStatus, dst *v1beta1.DynamoGraphDeploymentRequestStatus, dstObj *v1beta1.DynamoGraphDeploymentRequest, preserved *v1beta1.DynamoGraphDeploymentRequestStatus, profilingJobName string, preservedPhaseUnmodified bool) {
	dst.Phase = dgdrStateToPhase(string(src.State), src.Deployment)
	dst.ObservedGeneration = src.ObservedGeneration
	dst.Conditions = src.Conditions

	if src.Backend != "" {
		setAnnotation(dstObj, annDGDRStatusBackend, src.Backend)
	}
	if src.ProfilingResults != "" {
		setAnnotation(dstObj, annDGDRProfilingResults, src.ProfilingResults)
	}
	if src.GeneratedDeployment != nil {
		if dst.ProfilingResults == nil {
			dst.ProfilingResults = &v1beta1.ProfilingResultsStatus{}
		}
		dst.ProfilingResults.SelectedConfig = src.GeneratedDeployment
	}
	if src.Deployment != nil {
		dst.DGDName = src.Deployment.Name
		if data, err := json.Marshal(src.Deployment); err == nil {
			setAnnotation(dstObj, annDGDRDeploymentStatus, string(data))
		}
	}

	// ProfilingJobName has no v1alpha1 status field, so ConvertFrom stores it
	// as an annotation and ConvertTo passes the captured value in after scrubbing.
	if profilingJobName != "" {
		dst.ProfilingJobName = profilingJobName
	}
	fillDGDRHubStatusFromPreserved(dst, preserved, preservedPhaseUnmodified)
}

// convertDGDRStatusFrom converts the v1beta1 Status back into the v1alpha1 Status.
func convertDGDRStatusFrom(src *v1beta1.DynamoGraphDeploymentRequestStatus, dst *DynamoGraphDeploymentRequestStatus, srcObj *v1beta1.DynamoGraphDeploymentRequest, preserved *DynamoGraphDeploymentRequestStatus, preservedSourceUnmodified bool) {
	dst.State = DGDRState(dgdrPhaseToState(src.Phase))
	dst.ObservedGeneration = src.ObservedGeneration
	dst.Conditions = src.Conditions

	if srcObj.Annotations != nil {
		if v, ok := srcObj.Annotations[annDGDRStatusBackend]; ok {
			dst.Backend = v
		}
		if v, ok := srcObj.Annotations[annDGDRProfilingResults]; ok {
			dst.ProfilingResults = v
		}
	}

	if src.ProfilingResults != nil && src.ProfilingResults.SelectedConfig != nil {
		dst.GeneratedDeployment = src.ProfilingResults.SelectedConfig
	}

	if srcObj.Annotations != nil {
		if v, ok := srcObj.Annotations[annDGDRDeploymentStatus]; ok && v != "" {
			var depStatus DeploymentStatus
			if err := json.Unmarshal([]byte(v), &depStatus); err == nil {
				depStatus.Name = src.DGDName
				depStatus.Created = src.Phase == v1beta1.DGDRPhaseDeployed
				dst.Deployment = &depStatus
			}
		}
	}
	// If no annotation but we have DGDName, create a minimal deployment status.
	// Created is left false so the v1alpha1 controller does not skip re-creating the DGD.
	if dst.Deployment == nil && src.DGDName != "" {
		dst.Deployment = &DeploymentStatus{
			Name:    src.DGDName,
			Created: src.Phase == v1beta1.DGDRPhaseDeployed,
		}
	}
	fillDGDRSpokeStatusFromPreserved(dst, preserved, preservedSourceUnmodified)
}

func fillDGDRHubStatusFromPreserved(dst *v1beta1.DynamoGraphDeploymentRequestStatus, preserved *v1beta1.DynamoGraphDeploymentRequestStatus, preservedPhaseUnmodified bool) {
	if preserved == nil {
		return
	}
	if preservedPhaseUnmodified {
		dst.Phase = preserved.Phase
	}
	if dst.ProfilingPhase == "" {
		dst.ProfilingPhase = preserved.ProfilingPhase
	}
	if dst.ProfilingJobName == "" {
		dst.ProfilingJobName = preserved.ProfilingJobName
	}
	if dst.DeploymentInfo == nil {
		dst.DeploymentInfo = preserved.DeploymentInfo
	}
	if preserved.ProfilingResults != nil && len(preserved.ProfilingResults.Pareto) > 0 {
		if dst.ProfilingResults == nil {
			dst.ProfilingResults = &v1beta1.ProfilingResultsStatus{}
		}
		dst.ProfilingResults.Pareto = preserved.ProfilingResults.Pareto
	}
}

func fillDGDRSpokeStatusFromPreserved(dst *DynamoGraphDeploymentRequestStatus, preserved *DynamoGraphDeploymentRequestStatus, preservedSourceUnmodified bool) {
	if preserved == nil {
		return
	}
	if preservedSourceUnmodified && dst.State == DGDRStatePending {
		dst.State = preserved.State
	}
	if dst.Backend == "" {
		dst.Backend = preserved.Backend
	}
	if dst.ProfilingResults == "" {
		dst.ProfilingResults = preserved.ProfilingResults
	}
	if preservedSourceUnmodified && dst.GeneratedDeployment == nil {
		dst.GeneratedDeployment = preserved.GeneratedDeployment
	}
	if preserved.Deployment != nil {
		if dst.Deployment == nil {
			if preservedSourceUnmodified {
				cp := *preserved.Deployment
				dst.Deployment = &cp
			}
		} else {
			if dst.Deployment.Namespace == "" {
				dst.Deployment.Namespace = preserved.Deployment.Namespace
			}
			if dst.Deployment.State == "" {
				dst.Deployment.State = preserved.Deployment.State
			}
			if preservedSourceUnmodified && !dst.Deployment.Created {
				dst.Deployment.Created = preserved.Deployment.Created
			}
		}
	}
}

func dgdrNeedsHubPreservation(src *v1beta1.DynamoGraphDeploymentRequest) bool {
	if src.Spec.Hardware != nil || src.Spec.SearchStrategy != "" || src.Spec.ModelCache != nil {
		return true
	}
	if src.Spec.SLA != nil && src.Spec.SLA.E2ELatency != nil {
		return true
	}
	if src.Spec.Workload != nil && (src.Spec.Workload.Concurrency != nil || src.Spec.Workload.RequestRate != nil) {
		return true
	}
	if src.Spec.Overrides != nil && src.Spec.Overrides.DGD != nil {
		return true
	}
	if src.Status.ProfilingPhase != "" || src.Status.ProfilingJobName != "" || src.Status.DeploymentInfo != nil {
		return true
	}
	if !dgdrPhaseRoundTripsThroughAlpha(&src.Status) {
		return true
	}
	return src.Status.ProfilingResults != nil && len(src.Status.ProfilingResults.Pareto) > 0
}

func dgdrPhaseRoundTripsThroughAlpha(status *v1beta1.DynamoGraphDeploymentRequestStatus) bool {
	if status == nil {
		return true
	}
	var deployment *DeploymentStatus
	if status.DGDName != "" {
		deployment = &DeploymentStatus{
			Name:    status.DGDName,
			Created: status.Phase == v1beta1.DGDRPhaseDeployed,
		}
	}
	return dgdrStateToPhase(dgdrPhaseToState(status.Phase), deployment) == status.Phase
}

// dgdrStateToPhase maps v1alpha1 state strings to v1beta1 DGDRPhase.
func dgdrStateToPhase(state string, deployment *DeploymentStatus) v1beta1.DGDRPhase {
	switch state {
	case "", string(DGDRStatePending):
		return v1beta1.DGDRPhasePending
	case string(DGDRStateProfiling):
		return v1beta1.DGDRPhaseProfiling
	case string(DGDRStateReady):
		// If there is a deployment that was created, it means we are actually Deployed
		if deployment != nil && deployment.Created {
			return v1beta1.DGDRPhaseDeployed
		}
		return v1beta1.DGDRPhaseReady
	case string(DGDRStateDeploying):
		return v1beta1.DGDRPhaseDeploying
	case string(DGDRStateDeploymentDeleted):
		return v1beta1.DGDRPhaseReady
	case string(DGDRStateFailed):
		return v1beta1.DGDRPhaseFailed
	default:
		return v1beta1.DGDRPhasePending
	}
}

// dgdrPhaseToState maps v1beta1 DGDRPhase to v1alpha1 state strings.
func dgdrPhaseToState(phase v1beta1.DGDRPhase) string {
	switch phase {
	case v1beta1.DGDRPhasePending:
		return string(DGDRStatePending)
	case v1beta1.DGDRPhaseProfiling:
		return string(DGDRStateProfiling)
	case v1beta1.DGDRPhaseReady:
		return string(DGDRStateReady)
	case v1beta1.DGDRPhaseDeploying:
		return string(DGDRStateDeploying)
	case v1beta1.DGDRPhaseDeployed:
		return string(DGDRStateReady) // lossy
	case v1beta1.DGDRPhaseFailed:
		return string(DGDRStateFailed)
	default:
		return string(DGDRStatePending)
	}
}
