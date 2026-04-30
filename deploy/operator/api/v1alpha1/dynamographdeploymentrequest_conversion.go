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
//	sla.optimizationType                       Spec.SLA.OptimizationType (*OptimizationType)
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
//	ProfilingConfig.NodeSelector               Overrides.ProfilingJob.Template.
//	  (map[string]string)                        Spec.NodeSelector
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
// v1beta1-only fields with no v1alpha1 equivalent (sparse annotation-preserved):
//
//	Hardware.*, Workload.{Concurrency,RequestRate}, SLA.{E2ELatency},
//	Overrides.*, SearchStrategy
//
// Not yet implemented / omitted:
//
//	Features.KVRouter
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
// v1beta1-only status fields with no v1alpha1 equivalent
// (sparse annotation-preserved):
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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
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

	dgdrProfilingBlobSLAKey              = "sla"
	dgdrProfilingBlobTTFTKey             = "ttft"
	dgdrProfilingBlobITLKey              = "itl"
	dgdrProfilingBlobOptimizationTypeKey = "optimizationType"
	dgdrProfilingBlobISLKey              = "isl"
	dgdrProfilingBlobOSLKey              = "osl"
)

type dgdrConversionContext struct {
	hubObject        *v1beta1.DynamoGraphDeploymentRequest
	profilingJobName string
}

type dgdrHubSpecPreservation struct {
	Spec         v1beta1.DynamoGraphDeploymentRequestSpec `json:"spec"`
	AutoApplyNil bool                                     `json:"autoApplyNil,omitempty"`
}

type dgdrHubStatusPreservation struct {
	Status   v1beta1.DynamoGraphDeploymentRequestStatus `json:"status"`
	PhaseSet bool                                       `json:"phaseSet,omitempty"`
}

// ConvertTo converts this DynamoGraphDeploymentRequest (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoGraphDeploymentRequest) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentRequest but got %T", dstRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()
	dst.Annotations = maps.Clone(src.Annotations)

	restoredHubSpec, restoredHubStatus := decodeDGDRHubPreserved(src)
	profilingJobName := src.Annotations[annDGDRProfilingJobName]
	scrubDGDRInternalAnnotations(dst.Annotations)
	if len(dst.Annotations) == 0 {
		dst.Annotations = nil
	}

	ctx := dgdrConversionContext{
		hubObject:        dst,
		profilingJobName: profilingJobName,
	}
	var spokeSpecSave DynamoGraphDeploymentRequestSpec
	if err := convertDGDRSpecToHub(&src.Spec, &dst.Spec, restoredHubSpec, &spokeSpecSave, ctx); err != nil {
		return err
	}

	var spokeStatusSave DynamoGraphDeploymentRequestStatus
	convertDGDRStatusToHub(&src.Status, &dst.Status, restoredHubStatus, &spokeStatusSave, ctx)
	spokeStatusSaveNeeded := dgdrAlphaStatusNeedsSave(&src.Status)
	if !dgdrAlphaSpecSaveIsZero(&spokeSpecSave) || spokeStatusSaveNeeded {
		preserveDGDRSpoke(&spokeSpecSave, &spokeStatusSave, spokeStatusSaveNeeded, dst)
	}
	if len(dst.Annotations) == 0 {
		dst.Annotations = nil
	}

	return nil
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoGraphDeploymentRequest (v1alpha1).
func (dst *DynamoGraphDeploymentRequest) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentRequest but got %T", srcRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()
	dst.Annotations = maps.Clone(src.Annotations)

	restoredSpokeSpec, restoredSpokeStatus := decodeDGDRSpokePreserved(src)
	scrubDGDRInternalAnnotations(dst.Annotations)
	if len(dst.Annotations) == 0 {
		dst.Annotations = nil
	}

	ctx := dgdrConversionContext{
		hubObject: src,
	}
	var hubSpecSave dgdrHubSpecPreservation
	convertDGDRSpecFromHub(&src.Spec, &dst.Spec, restoredSpokeSpec, &hubSpecSave, ctx)
	var hubStatusSave dgdrHubStatusPreservation
	convertDGDRStatusFromHub(&src.Status, &dst.Status, restoredSpokeStatus, &hubStatusSave, ctx)

	// ProfilingJobName — no v1alpha1 status field; store as annotation for round-trip
	if src.Status.ProfilingJobName != "" {
		if dst.Annotations == nil {
			dst.Annotations = make(map[string]string)
		}
		dst.Annotations[annDGDRProfilingJobName] = src.Status.ProfilingJobName
	}
	if dgdrNeedsHubPreservation(&hubSpecSave, &hubStatusSave) {
		preserveDGDRHub(&hubSpecSave, &hubStatusSave, dst)
	}

	return nil
}

type dgdrSpokeSpecPreservation struct {
	Spec               DynamoGraphDeploymentRequestSpec `json:"spec"`
	ProfilingConfigSet bool                             `json:"profilingConfigSet,omitempty"`
	ProfilingConfigRaw []byte                           `json:"profilingConfigRaw,omitempty"`
}

func preserveDGDRSpoke(specSave *DynamoGraphDeploymentRequestSpec, statusSave *DynamoGraphDeploymentRequestStatus, statusSaveNeeded bool, dst *v1beta1.DynamoGraphDeploymentRequest) {
	specPreserved := false
	if !dgdrAlphaSpecSaveIsZero(specSave) {
		envelope := dgdrSpokeSpecPreservation{Spec: *specSave.DeepCopy()}
		if specSave.ProfilingConfig.Config != nil {
			envelope.ProfilingConfigSet = true
			envelope.ProfilingConfigRaw = slices.Clone(specSave.ProfilingConfig.Config.Raw)
			envelope.Spec.ProfilingConfig.Config = nil
		}
		if setJSONAnnOnObj(dst, annDGDRSpokeSpec, envelope) {
			specPreserved = true
		}
	}
	if statusSaveNeeded {
		setJSONAnnOnObj(dst, annDGDRSpokeStatus, statusSave)
	}
	if specPreserved && specSave != nil && specSave.ProfilingConfig.Config != nil && len(specSave.ProfilingConfig.Config.Raw) == 0 {
		setAnnOnObj(dst, annDGDRProfilingEmpty, annotationTrue)
	}
}

func preserveDGDRHub(specSave *dgdrHubSpecPreservation, statusSave *dgdrHubStatusPreservation, dst *DynamoGraphDeploymentRequest) {
	if !dgdrHubSpecSaveIsZero(specSave) {
		setJSONAnnOnObj(dst, annDGDRHubSpec, specSave)
	}
	if !dgdrHubStatusSaveIsZero(statusSave) {
		setJSONAnnOnObj(dst, annDGDRHubStatus, statusSave)
	}
}

func saveDGDRAlphaOnlySpec(src, save *DynamoGraphDeploymentRequestSpec) {
	if src == nil || save == nil {
		return
	}
	if src.DeploymentOverrides == nil || src.DeploymentOverrides.WorkersImage == "" {
		return
	}
	save.DeploymentOverrides = &DeploymentOverridesSpec{
		WorkersImage: src.DeploymentOverrides.WorkersImage,
	}
	save.ProfilingConfig.ProfilerImage = src.ProfilingConfig.ProfilerImage
}

func dgdrAlphaSpecSaveIsZero(save *DynamoGraphDeploymentRequestSpec) bool {
	return save == nil ||
		save.ProfilingConfig.ProfilerImage == "" &&
			(save.DeploymentOverrides == nil || save.DeploymentOverrides.WorkersImage == "")
}

func saveDGDRAlphaOnlyStatus(src, save *DynamoGraphDeploymentRequestStatus) {
	if src == nil || save == nil {
		return
	}
	if !dgdrAlphaStateRoundTripsThroughHub(src) {
		save.State = src.State
	}
}

func dgdrAlphaStatusNeedsSave(src *DynamoGraphDeploymentRequestStatus) bool {
	return !dgdrAlphaStateRoundTripsThroughHub(src)
}

func dgdrAlphaStateRoundTripsThroughHub(src *DynamoGraphDeploymentRequestStatus) bool {
	if src == nil {
		return true
	}
	phase := dgdrStateToPhase(string(src.State), src.Deployment)
	return DGDRState(dgdrPhaseToState(phase)) == src.State
}

func saveDGDRHubOnlySpec(src *v1beta1.DynamoGraphDeploymentRequestSpec, save *dgdrHubSpecPreservation) {
	if src == nil || save == nil {
		return
	}
	if src.AutoApply == nil {
		save.AutoApplyNil = true
	}
	if src.SLA == nil && src.Workload != nil {
		// v1alpha1 stores SLA and Workload together under profilingConfig.sla.
		// Preserve the hub-only "Workload without SLA" shape without copying
		// the representable Workload leaves into the save payload.
		save.Spec.Workload = &v1beta1.WorkloadSpec{}
	}
	if src.Hardware != nil {
		save.Spec.Hardware = src.Hardware
	}
	if src.SearchStrategy != "" {
		save.Spec.SearchStrategy = src.SearchStrategy
	}
	if src.SLA != nil && src.SLA.E2ELatency != nil {
		if save.Spec.SLA == nil {
			save.Spec.SLA = &v1beta1.SLASpec{}
		}
		e2e := *src.SLA.E2ELatency
		save.Spec.SLA.E2ELatency = &e2e
	}
	if src.Workload != nil && (src.Workload.Concurrency != nil || src.Workload.RequestRate != nil) {
		if save.Spec.Workload == nil {
			save.Spec.Workload = &v1beta1.WorkloadSpec{}
		}
		if src.Workload.Concurrency != nil {
			concurrency := *src.Workload.Concurrency
			save.Spec.Workload.Concurrency = &concurrency
		}
		if src.Workload.RequestRate != nil {
			requestRate := *src.Workload.RequestRate
			save.Spec.Workload.RequestRate = &requestRate
		}
	}
	if src.Workload != nil &&
		src.Workload.ISL == nil &&
		src.Workload.OSL == nil &&
		src.Workload.Concurrency == nil &&
		src.Workload.RequestRate == nil {
		save.Spec.Workload = &v1beta1.WorkloadSpec{}
	}
	if src.ModelCache != nil && src.ModelCache.PVCName == "" && src.ModelCache.PVCModelPath == "" && src.ModelCache.PVCMountPath == "" {
		save.Spec.ModelCache = &v1beta1.ModelCacheSpec{}
	}
	saveDGDRHubOnlyFeatures(src, save)
	saveDGDRHubOnlyOverrides(src, save)
}

func saveDGDRHubOnlyFeatures(src *v1beta1.DynamoGraphDeploymentRequestSpec, save *dgdrHubSpecPreservation) {
	if src.Features == nil {
		return
	}
	if src.Features.Mocker != nil && !src.Features.Mocker.Enabled {
		if save.Spec.Features == nil {
			save.Spec.Features = &v1beta1.FeaturesSpec{}
		}
		save.Spec.Features.Mocker = &v1beta1.MockerSpec{}
	}
	if src.Features.Planner != nil && len(src.Features.Planner.Raw) == 0 {
		if save.Spec.Features == nil {
			save.Spec.Features = &v1beta1.FeaturesSpec{}
		}
		save.Spec.Features.Planner = src.Features.Planner
	}
}

func saveDGDRHubOnlyOverrides(src *v1beta1.DynamoGraphDeploymentRequestSpec, save *dgdrHubSpecPreservation) {
	if src.Overrides == nil {
		return
	}
	if src.Overrides.DGD == nil && !profilingJobHasHubOnlyFields(src.Overrides.ProfilingJob) {
		return
	}
	save.Spec.Overrides = &v1beta1.OverridesSpec{}
	if src.Overrides.DGD != nil {
		save.Spec.Overrides.DGD = src.Overrides.DGD
	}
	if profilingJobHasHubOnlyFields(src.Overrides.ProfilingJob) {
		save.Spec.Overrides.ProfilingJob = saveProfilingJobHubOnlyFields(src.Overrides.ProfilingJob)
	}
}

func dgdrHubSpecSaveIsZero(save *dgdrHubSpecPreservation) bool {
	return save == nil ||
		!save.AutoApplyNil &&
			apiequality.Semantic.DeepEqual(&save.Spec, &v1beta1.DynamoGraphDeploymentRequestSpec{})
}

func saveDGDRHubOnlyStatus(src *v1beta1.DynamoGraphDeploymentRequestStatus, save *dgdrHubStatusPreservation) {
	if src == nil || save == nil {
		return
	}
	if !dgdrPhaseRoundTripsThroughAlpha(src) {
		save.PhaseSet = true
		save.Status.Phase = src.Phase
		save.Status.DGDName = src.DGDName
	}
	if src.ProfilingPhase != "" {
		save.Status.ProfilingPhase = src.ProfilingPhase
	}
	if src.ProfilingJobName != "" {
		save.Status.ProfilingJobName = src.ProfilingJobName
	}
	if src.DeploymentInfo != nil {
		save.Status.DeploymentInfo = src.DeploymentInfo
	}
	if src.ProfilingResults != nil && len(src.ProfilingResults.Pareto) > 0 {
		save.Status.ProfilingResults = &v1beta1.ProfilingResultsStatus{
			Pareto: slices.Clone(src.ProfilingResults.Pareto),
		}
	}
}

func dgdrHubStatusSaveIsZero(save *dgdrHubStatusPreservation) bool {
	return save == nil ||
		!save.PhaseSet &&
			apiequality.Semantic.DeepEqual(&save.Status, &v1beta1.DynamoGraphDeploymentRequestStatus{})
}

func decodeDGDRHubPreserved(src *DynamoGraphDeploymentRequest) (*dgdrHubSpecPreservation, *dgdrHubStatusPreservation) {
	if src.Annotations == nil {
		return nil, nil
	}
	return decodeDGDRHubSpecPreserved(src), decodeDGDRHubStatusPreserved(src)
}

func decodeDGDRHubSpecPreserved(src *DynamoGraphDeploymentRequest) *dgdrHubSpecPreservation {
	raw, ok := getAnnFromObj(src, annDGDRHubSpec)
	if !ok || raw == "" {
		return nil
	}
	var envelope dgdrHubSpecPreservation
	if err := json.Unmarshal([]byte(raw), &envelope); err == nil && !dgdrHubSpecSaveIsZero(&envelope) {
		return &envelope
	}
	var legacy v1beta1.DynamoGraphDeploymentRequestSpec
	if err := json.Unmarshal([]byte(raw), &legacy); err != nil {
		return nil
	}
	envelope.Spec = legacy
	if legacy.AutoApply != nil && !*legacy.AutoApply {
		envelope.AutoApplyNil = true
		envelope.Spec.AutoApply = nil
	}
	if dgdrHubSpecSaveIsZero(&envelope) {
		return nil
	}
	return &envelope
}

func decodeDGDRHubStatusPreserved(src *DynamoGraphDeploymentRequest) *dgdrHubStatusPreservation {
	raw, ok := getAnnFromObj(src, annDGDRHubStatus)
	if !ok || raw == "" {
		return nil
	}
	var envelope dgdrHubStatusPreservation
	if err := json.Unmarshal([]byte(raw), &envelope); err == nil && !dgdrHubStatusSaveIsZero(&envelope) {
		return &envelope
	}
	var legacy v1beta1.DynamoGraphDeploymentRequestStatus
	if err := json.Unmarshal([]byte(raw), &legacy); err != nil {
		return nil
	}
	envelope.Status = legacy
	if dgdrHubStatusSaveIsZero(&envelope) {
		return nil
	}
	return &envelope
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
	if decoded, ok := getJSONAnnFromObj[DynamoGraphDeploymentRequestStatus](src, annDGDRSpokeStatus); ok {
		status = &decoded
	}
	if src.Annotations[annDGDRProfilingEmpty] == annotationTrue {
		if spec == nil {
			spec = &DynamoGraphDeploymentRequestSpec{}
		}
		spec.ProfilingConfig.Config = &apiextensionsv1.JSON{}
	}
	return spec, status
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

// convertDGDRSpecToHub converts representable v1alpha1 fields from the live
// source, restores hub-only fields from restored target data, and saves the
// source-only fields that the hub cannot represent.
func convertDGDRSpecToHub(src *DynamoGraphDeploymentRequestSpec, dst *v1beta1.DynamoGraphDeploymentRequestSpec, restored *dgdrHubSpecPreservation, save *DynamoGraphDeploymentRequestSpec, ctx dgdrConversionContext) error {
	if src == nil || dst == nil {
		return nil
	}

	dst.Model = src.Model
	autoApply := src.AutoApply
	dst.AutoApply = &autoApply

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
	if src.EnableGPUDiscovery != nil {
		if *src.EnableGPUDiscovery {
			setAnnOnObj(ctx.hubObject, annDGDREnableGPUDisc, annotationTrue)
		} else {
			setAnnOnObj(ctx.hubObject, annDGDREnableGPUDisc, "false")
		}
	}

	if src.ProfilingConfig.Config != nil {
		convertProfilingConfigBlobToHub(src.ProfilingConfig.Config.Raw, dst, ctx.hubObject)
	}

	// ProfilerImage → Image (the profiler runs in the frontend image)
	// TODO: In a future MR, backend inference images will be managed separately via overrides.dgd.
	if src.ProfilingConfig.ProfilerImage != "" {
		dst.Image = src.ProfilingConfig.ProfilerImage
	}
	if src.ProfilingConfig.ConfigMapRef != nil {
		setJSONAnnOnObj(ctx.hubObject, annDGDRConfigMapRef, src.ProfilingConfig.ConfigMapRef)
	}
	if src.ProfilingConfig.OutputPVC != "" {
		setAnnOnObj(ctx.hubObject, annDGDROutputPVC, src.ProfilingConfig.OutputPVC)
	}

	convertProfilingResourcesToOverrides(&src.ProfilingConfig, dst)
	convertDeploymentOverridesToAnnotation(src.DeploymentOverrides, ctx.hubObject)
	restoreDGDRHubSpec(dst, restored)
	if save != nil {
		saveDGDRAlphaOnlySpec(src, save)
	}

	return nil
}

// applySLAAndWorkloadFromBlob extracts SLA and Workload fields from the v1alpha1 JSON blob.
// Both are nested under blob["sla"] in the v1alpha1 schema.
func applySLAAndWorkloadFromBlob(blob map[string]interface{}, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
	slaRaw, ok := blob[dgdrProfilingBlobSLAKey]
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
	if v, ok := slaMap[dgdrProfilingBlobTTFTKey].(float64); ok {
		dst.SLA.TTFT = &v
	}
	if v, ok := slaMap[dgdrProfilingBlobITLKey].(float64); ok {
		dst.SLA.ITL = &v
	}
	if v, ok := slaMap[dgdrProfilingBlobOptimizationTypeKey].(string); ok {
		ot := v1beta1.OptimizationType(v)
		if ot == v1beta1.OptimizationTypeLatency || ot == v1beta1.OptimizationTypeThroughput {
			dst.SLA.OptimizationType = &ot
		}
	}

	if v, ok := slaMap[dgdrProfilingBlobISLKey].(float64); ok {
		if dst.Workload == nil {
			dst.Workload = &v1beta1.WorkloadSpec{}
		}
		isl := int32(v)
		dst.Workload.ISL = &isl
	}
	if v, ok := slaMap[dgdrProfilingBlobOSLKey].(float64); ok {
		if dst.Workload == nil {
			dst.Workload = &v1beta1.WorkloadSpec{}
		}
		osl := int32(v)
		dst.Workload.OSL = &osl
	}
}

func convertProfilingConfigBlobToHub(raw []byte, dst *v1beta1.DynamoGraphDeploymentRequestSpec, dstObj *v1beta1.DynamoGraphDeploymentRequest) {
	if len(raw) == 0 {
		setAnnOnObj(dstObj, annDGDRProfilingEmpty, annotationTrue)
		return
	}

	var blob map[string]interface{}
	if err := json.Unmarshal(raw, &blob); err != nil || blob == nil {
		// ProfilingConfig.Config is an opaque JSON extension point on the
		// v1alpha1 side. Generic fuzzers may populate it with arbitrary
		// RawExtension bytes; preserve those bytes via the annotation but only
		// project typed v1beta1 fields when it is a JSON object in the legacy
		// shape we understand.
		setAnnOnObj(dstObj, annDGDRProfilingConfig, string(raw))
		return
	}

	originalKeys := len(blob)
	applySLAAndWorkloadFromBlob(blob, dst)
	applyModelCacheFromBlob(blob, dst)
	applyPlannerFromBlob(blob, dst)
	pruneDGDRKnownAlphaBlobFields(blob)
	if len(blob) == 0 && originalKeys > 0 {
		return
	}
	data, err := json.Marshal(blob)
	if err == nil {
		setAnnOnObj(dstObj, annDGDRProfilingConfig, string(data))
	}
}

func pruneDGDRKnownAlphaBlobFields(blob map[string]interface{}) {
	if slaMap, ok := blob[dgdrProfilingBlobSLAKey].(map[string]interface{}); ok {
		for _, key := range []string{
			dgdrProfilingBlobTTFTKey,
			dgdrProfilingBlobITLKey,
			dgdrProfilingBlobOptimizationTypeKey,
			dgdrProfilingBlobISLKey,
			dgdrProfilingBlobOSLKey,
		} {
			delete(slaMap, key)
		}
		if len(slaMap) == 0 {
			delete(blob, dgdrProfilingBlobSLAKey)
		}
	}
	if deployMap, ok := blob["deployment"].(map[string]interface{}); ok {
		if mcMap, ok := deployMap["modelCache"].(map[string]interface{}); ok {
			hasModelCacheProjection := false
			for _, key := range []string{"pvcName", "modelPathInPvc", "pvcMountPath"} {
				if _, ok := mcMap[key]; ok {
					hasModelCacheProjection = true
					break
				}
			}
			if hasModelCacheProjection {
				delete(deployMap, "modelCache")
			} else if len(mcMap) == 0 {
				delete(deployMap, "modelCache")
			}
		}
		if len(deployMap) == 0 {
			delete(blob, "deployment")
		}
	}
	if plannerMap, ok := blob["planner"].(map[string]interface{}); ok && len(plannerMap) > 0 {
		delete(blob, "planner")
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

// convertProfilingResourcesToOverrides maps ProfilingConfig pod fields into
// the v1beta1 Overrides.ProfilingJob pod spec.
func convertProfilingResourcesToOverrides(src *ProfilingConfigSpec, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
	if src.Resources == nil && len(src.Tolerations) == 0 && len(src.NodeSelector) == 0 {
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
	if len(src.NodeSelector) > 0 {
		podSpec.NodeSelector = maps.Clone(src.NodeSelector)
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
	setJSONAnnOnObj(dstObj, annDGDRDeployOverrides, overrides)
}

func restoreDGDRHubSpec(dst *v1beta1.DynamoGraphDeploymentRequestSpec, restored *dgdrHubSpecPreservation) {
	if restored == nil {
		return
	}
	restoreDGDRHubNilDefaultShapes(dst, restored)
	restoreDGDRHubFeatureFields(dst, restored)
	restoreDGDRHubSpecScalarFields(dst, restored)
	restoreDGDRHubModelCacheAndSLAFields(dst, restored)
	restoreDGDRHubOverrides(dst, restored)
}

func restoreDGDRHubNilDefaultShapes(dst *v1beta1.DynamoGraphDeploymentRequestSpec, restored *dgdrHubSpecPreservation) {
	if restored.Spec.SLA == nil && dst.SLA != nil &&
		dst.SLA.TTFT == nil && dst.SLA.ITL == nil && dst.SLA.E2ELatency == nil &&
		dst.SLA.OptimizationType == nil {
		dst.SLA = nil
	}
	if restored.AutoApplyNil && dst.AutoApply != nil && *dst.AutoApply {
		dst.AutoApply = nil
	}
}

func restoreDGDRHubFeatureFields(dst *v1beta1.DynamoGraphDeploymentRequestSpec, restored *dgdrHubSpecPreservation) {
	if restored.Spec.Features == nil {
		return
	}
	restoreDGDRHubMockerField(dst, restored.Spec.Features)
	restoreDGDRHubPlannerField(dst, restored.Spec.Features)
}

func restoreDGDRHubMockerField(dst *v1beta1.DynamoGraphDeploymentRequestSpec, restored *v1beta1.FeaturesSpec) {
	if restored.Mocker == nil || restored.Mocker.Enabled {
		return
	}
	if dst.Features == nil {
		dst.Features = &v1beta1.FeaturesSpec{}
	}
	if dst.Features.Mocker == nil {
		dst.Features.Mocker = restored.Mocker
	}
}

func restoreDGDRHubPlannerField(dst *v1beta1.DynamoGraphDeploymentRequestSpec, restored *v1beta1.FeaturesSpec) {
	if restored.Planner == nil || len(restored.Planner.Raw) != 0 {
		return
	}
	if dst.Features == nil {
		dst.Features = &v1beta1.FeaturesSpec{}
	}
	if dst.Features.Planner == nil {
		dst.Features.Planner = restored.Planner
	}
}

func restoreDGDRHubSpecScalarFields(dst *v1beta1.DynamoGraphDeploymentRequestSpec, restored *dgdrHubSpecPreservation) {
	if restored.Spec.Hardware != nil {
		dst.Hardware = restored.Spec.Hardware
	}
	if restored.Spec.SearchStrategy != "" {
		dst.SearchStrategy = restored.Spec.SearchStrategy
	}
}

func restoreDGDRHubModelCacheAndSLAFields(dst *v1beta1.DynamoGraphDeploymentRequestSpec, restored *dgdrHubSpecPreservation) {
	if dst.ModelCache == nil && restored.Spec.ModelCache != nil {
		dst.ModelCache = restored.Spec.ModelCache
	}
	if restored.Spec.SLA != nil && restored.Spec.SLA.E2ELatency != nil {
		if dst.SLA == nil {
			dst.SLA = &v1beta1.SLASpec{}
		}
		dst.SLA.E2ELatency = restored.Spec.SLA.E2ELatency
	}
	if restored.Spec.Workload != nil {
		if dst.Workload == nil {
			dst.Workload = &v1beta1.WorkloadSpec{}
		}
		dst.Workload.Concurrency = restored.Spec.Workload.Concurrency
		dst.Workload.RequestRate = restored.Spec.Workload.RequestRate
	}
}

func restoreDGDRHubOverrides(dst *v1beta1.DynamoGraphDeploymentRequestSpec, restored *dgdrHubSpecPreservation) {
	if restored.Spec.Overrides == nil {
		return
	}
	if dst.Overrides == nil {
		dst.Overrides = &v1beta1.OverridesSpec{}
	}
	if restored.Spec.Overrides.DGD != nil {
		dst.Overrides.DGD = restored.Spec.Overrides.DGD
	}
	if restored.Spec.Overrides.ProfilingJob != nil {
		dst.Overrides.ProfilingJob = restoreProfilingJobHubOnlyFields(dst.Overrides.ProfilingJob, restored.Spec.Overrides.ProfilingJob)
	}
}

// restoreProfilingJobHubOnlyFields starts from the live v1alpha1 projection and
// copies only v1beta1-only profiling JobSpec leaves from the restored hub spec.
func restoreProfilingJobHubOnlyFields(semantic, restored *batchv1.JobSpec) *batchv1.JobSpec {
	if restored == nil {
		if semantic == nil {
			return nil
		}
		return semantic.DeepCopy()
	}
	if semantic == nil && !profilingJobHasHubOnlyFields(restored) {
		return nil
	}

	out := &batchv1.JobSpec{}
	if semantic != nil {
		out = semantic.DeepCopy()
	}

	p := restored.DeepCopy()
	copyStructFieldsExcept(out, p, "Template")
	out.Template.ObjectMeta = *p.Template.ObjectMeta.DeepCopy()
	restoreProfilingPodSpecHubOnlyFields(&out.Template.Spec, &p.Template.Spec)
	return out
}

func restoreProfilingPodSpecHubOnlyFields(dst, restored *corev1.PodSpec) {
	p := restored.DeepCopy()
	copyStructFieldsExcept(dst, p, "Containers", "NodeSelector", "Tolerations")
	dst.Containers = restoreProfilingContainersHubOnlyFields(dst.Containers, p.Containers)
}

func restoreProfilingContainersHubOnlyFields(semantic, restored []corev1.Container) []corev1.Container {
	if len(restored) == 0 {
		if len(semantic) == 0 && restored != nil {
			return []corev1.Container{}
		}
		return slices.Clone(semantic)
	}

	var firstSemantic *corev1.Container
	if len(semantic) > 0 {
		firstSemantic = &semantic[0]
	}

	out := make([]corev1.Container, 0, len(restored))
	first := restoreProfilingFirstContainerHubOnlyFields(firstSemantic, &restored[0])
	if firstSemantic != nil || profilingFirstContainerHasHubOnlyFields(&restored[0]) || len(restored) > 1 {
		out = append(out, first)
	}
	for i := 1; i < len(restored); i++ {
		out = append(out, *restored[i].DeepCopy())
	}
	return out
}

func restoreProfilingFirstContainerHubOnlyFields(semantic, restored *corev1.Container) corev1.Container {
	out := corev1.Container{}
	if semantic != nil {
		out = *semantic.DeepCopy()
	}
	// The first container's Resources is represented by v1alpha1
	// ProfilingConfig.Resources; every other container field is hub-only here.
	p := restored.DeepCopy()
	copyStructFieldsExcept(&out, p, "Resources")
	return out
}

func saveProfilingJobHubOnlyFields(job *batchv1.JobSpec) *batchv1.JobSpec {
	if job == nil || !profilingJobHasHubOnlyFields(job) {
		return nil
	}
	out := job.DeepCopy()
	podSpec := &out.Template.Spec
	podSpec.NodeSelector = nil
	podSpec.Tolerations = nil
	if len(podSpec.Containers) > 0 {
		podSpec.Containers[0].Resources = corev1.ResourceRequirements{}
		if len(podSpec.Containers) == 1 && apiequality.Semantic.DeepEqual(podSpec.Containers[0], corev1.Container{}) {
			podSpec.Containers = nil
		}
	}
	return out
}

func copyStructFieldsExcept(dst, src any, except ...string) {
	dstValue := reflect.ValueOf(dst)
	srcValue := reflect.ValueOf(src)
	if dstValue.Kind() != reflect.Ptr || srcValue.Kind() != reflect.Ptr ||
		dstValue.IsNil() || srcValue.IsNil() {
		panic("copyStructFieldsExcept expects non-nil pointers")
	}

	dstStruct := dstValue.Elem()
	srcStruct := srcValue.Elem()
	if dstStruct.Kind() != reflect.Struct || srcStruct.Kind() != reflect.Struct ||
		dstStruct.Type() != srcStruct.Type() {
		panic("copyStructFieldsExcept expects pointers to the same struct type")
	}

	skip := map[string]struct{}{}
	for _, name := range except {
		skip[name] = struct{}{}
	}
	for i := 0; i < dstStruct.NumField(); i++ {
		field := dstStruct.Type().Field(i)
		if _, ok := skip[field.Name]; ok {
			continue
		}
		if dstStruct.Field(i).CanSet() {
			dstStruct.Field(i).Set(srcStruct.Field(i))
		}
	}
}

func profilingJobHasHubOnlyFields(job *batchv1.JobSpec) bool {
	if job == nil {
		return false
	}
	cp := job.DeepCopy()
	podSpec := &cp.Template.Spec
	podSpec.NodeSelector = nil
	podSpec.Tolerations = nil
	hasEmptyContainersShape := podSpec.Containers != nil && len(podSpec.Containers) == 0
	if len(podSpec.Containers) > 0 {
		podSpec.Containers[0].Resources = corev1.ResourceRequirements{}
		if len(podSpec.Containers) == 1 && apiequality.Semantic.DeepEqual(podSpec.Containers[0], corev1.Container{}) {
			podSpec.Containers = nil
		}
	}
	return hasEmptyContainersShape || !apiequality.Semantic.DeepEqual(*cp, batchv1.JobSpec{})
}

func profilingFirstContainerHasHubOnlyFields(container *corev1.Container) bool {
	if container == nil {
		return false
	}
	cp := container.DeepCopy()
	cp.Resources = corev1.ResourceRequirements{}
	return !apiequality.Semantic.DeepEqual(*cp, corev1.Container{})
}

// convertDGDRSpecFromHub converts representable hub fields from the live source,
// restores v1alpha1-only fields from restored target data, and saves source-only
// hub fields.
func convertDGDRSpecFromHub(src *v1beta1.DynamoGraphDeploymentRequestSpec, dst *DynamoGraphDeploymentRequestSpec, restored *DynamoGraphDeploymentRequestSpec, save *dgdrHubSpecPreservation, ctx dgdrConversionContext) {
	if src == nil || dst == nil {
		return
	}

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

	if ctx.hubObject.Annotations != nil {
		if v, ok := ctx.hubObject.Annotations[annDGDREnableGPUDisc]; ok {
			enabled := v == annotationTrue
			dst.EnableGPUDiscovery = &enabled
		}
	}

	// Reconstruct the JSON blob: start from the round-trip annotation (preserves unknown
	// keys), then overwrite with structured v1beta1 fields (structured fields win).
	var blob map[string]interface{}
	blobFromAnnotation := false
	if ctx.hubObject.Annotations != nil {
		if rawBlob, ok := ctx.hubObject.Annotations[annDGDRProfilingConfig]; ok && rawBlob != "" {
			if err := json.Unmarshal([]byte(rawBlob), &blob); err != nil || blob == nil {
				dst.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: []byte(rawBlob)}
			} else {
				pruneKnownProfilingBlobFields(src, blob)
				blobFromAnnotation = true
			}
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
		if len(blob) == 0 && !blobFromAnnotation {
			dst.ProfilingConfig.Config = nil
		} else if data, err := json.Marshal(blob); err == nil {
			dst.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: data}
		}
	}
	if ctx.hubObject.Annotations[annDGDRProfilingEmpty] == annotationTrue {
		dst.ProfilingConfig.Config = &apiextensionsv1.JSON{}
	}

	// Image → ProfilerImage (round-trip; see ConvertTo for rationale)
	// TODO: In a future MR, backend images will come from overrides.dgd; worker image
	//       (v1alpha1 DeploymentOverrides.WorkersImage) has no v1beta1 equivalent yet.
	if src.Image != "" {
		dst.ProfilingConfig.ProfilerImage = src.Image
	}

	restoreDGDRSpokeAnnotationFields(ctx.hubObject, dst)
	restoreProfilingJobResources(src, dst)
	restoreDGDRSpokeSpec(dst, restored, src.Image)
	if save != nil {
		saveDGDRHubOnlySpec(src, save)
	}
}

func pruneKnownProfilingBlobFields(src *v1beta1.DynamoGraphDeploymentRequestSpec, blob map[string]interface{}) {
	pruneSLAWorkloadBlobFields(src, blob)
	pruneModelCacheBlobFields(src, blob)
	if src.Features == nil || src.Features.Planner == nil || len(src.Features.Planner.Raw) == 0 {
		delete(blob, "planner")
	}
}

func pruneSLAWorkloadBlobFields(src *v1beta1.DynamoGraphDeploymentRequestSpec, blob map[string]interface{}) {
	slaMap, ok := blob[dgdrProfilingBlobSLAKey].(map[string]interface{})
	if !ok {
		return
	}
	deleted := false
	if src.SLA == nil || src.SLA.TTFT == nil {
		_, had := slaMap[dgdrProfilingBlobTTFTKey]
		delete(slaMap, dgdrProfilingBlobTTFTKey)
		deleted = deleted || had
	}
	if src.SLA == nil || src.SLA.ITL == nil {
		_, had := slaMap[dgdrProfilingBlobITLKey]
		delete(slaMap, dgdrProfilingBlobITLKey)
		deleted = deleted || had
	}
	if src.SLA == nil || src.SLA.OptimizationType == nil {
		_, had := slaMap[dgdrProfilingBlobOptimizationTypeKey]
		delete(slaMap, dgdrProfilingBlobOptimizationTypeKey)
		deleted = deleted || had
	}
	if src.Workload == nil || src.Workload.ISL == nil {
		_, had := slaMap[dgdrProfilingBlobISLKey]
		delete(slaMap, dgdrProfilingBlobISLKey)
		deleted = deleted || had
	}
	if src.Workload == nil || src.Workload.OSL == nil {
		_, had := slaMap[dgdrProfilingBlobOSLKey]
		delete(slaMap, dgdrProfilingBlobOSLKey)
		deleted = deleted || had
	}
	if deleted && len(slaMap) == 0 {
		delete(blob, dgdrProfilingBlobSLAKey)
	}
}

func pruneModelCacheBlobFields(src *v1beta1.DynamoGraphDeploymentRequestSpec, blob map[string]interface{}) {
	deployMap, ok := blob["deployment"].(map[string]interface{})
	if !ok {
		return
	}
	deleted := false
	if src.ModelCache == nil ||
		(src.ModelCache.PVCName == "" &&
			src.ModelCache.PVCModelPath == "" &&
			src.ModelCache.PVCMountPath == "") {
		if _, ok := deployMap["modelCache"]; ok {
			delete(deployMap, "modelCache")
			deleted = true
		}
	}
	if deleted && len(deployMap) == 0 {
		delete(blob, "deployment")
	}
}

// mergeSLAWorkloadIntoBlob writes SLA and Workload structured fields back into the JSON blob,
// overwriting any existing values for those keys.
func mergeSLAWorkloadIntoBlob(src *v1beta1.DynamoGraphDeploymentRequestSpec, blob map[string]interface{}, preserveEmpty bool) {
	slaMap, hadSLA := blob[dgdrProfilingBlobSLAKey].(map[string]interface{})
	if slaMap == nil {
		slaMap = make(map[string]interface{})
	}
	wrote := false
	if src.SLA != nil {
		if src.SLA.TTFT != nil {
			slaMap[dgdrProfilingBlobTTFTKey] = *src.SLA.TTFT
			wrote = true
		}
		if src.SLA.ITL != nil {
			slaMap[dgdrProfilingBlobITLKey] = *src.SLA.ITL
			wrote = true
		}
		if src.SLA.OptimizationType != nil {
			slaMap[dgdrProfilingBlobOptimizationTypeKey] = string(*src.SLA.OptimizationType)
			wrote = true
		}
	}
	if src.Workload != nil {
		if src.Workload.ISL != nil {
			slaMap[dgdrProfilingBlobISLKey] = float64(*src.Workload.ISL)
			wrote = true
		}
		if src.Workload.OSL != nil {
			slaMap[dgdrProfilingBlobOSLKey] = float64(*src.Workload.OSL)
			wrote = true
		}
	}
	emptySLA := src.SLA != nil &&
		src.SLA.TTFT == nil &&
		src.SLA.ITL == nil &&
		src.SLA.E2ELatency == nil &&
		src.SLA.OptimizationType == nil
	if wrote || hadSLA || preserveEmpty || emptySLA {
		blob[dgdrProfilingBlobSLAKey] = slaMap
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

// restoreDGDRSpokeAnnotationFields restores v1alpha1 spec fields that were annotation-preserved
// during ConvertTo: ConfigMapRef, OutputPVC, and DeploymentOverrides.
func restoreDGDRSpokeAnnotationFields(srcObj *v1beta1.DynamoGraphDeploymentRequest, dst *DynamoGraphDeploymentRequestSpec) {
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

// restoreProfilingJobResources restores representable profiling pod fields from
// v1beta1 Overrides.ProfilingJob back into v1alpha1 ProfilingConfig.
func restoreProfilingJobResources(src *v1beta1.DynamoGraphDeploymentRequestSpec, dst *DynamoGraphDeploymentRequestSpec) {
	if src.Overrides == nil || src.Overrides.ProfilingJob == nil {
		return
	}
	podSpec := &src.Overrides.ProfilingJob.Template.Spec
	if len(podSpec.Containers) > 0 {
		res := podSpec.Containers[0].Resources
		dst.ProfilingConfig.Resources = &res
	}
	if len(podSpec.Tolerations) > 0 {
		dst.ProfilingConfig.Tolerations = podSpec.Tolerations
	}
	if len(podSpec.NodeSelector) > 0 {
		dst.ProfilingConfig.NodeSelector = maps.Clone(podSpec.NodeSelector)
	}
}

func restoreDGDRSpokeSpec(dst *DynamoGraphDeploymentRequestSpec, restored *DynamoGraphDeploymentRequestSpec, hubImage string) {
	if restored == nil {
		return
	}
	hasRestoredWorkersImage := restored.DeploymentOverrides != nil &&
		restored.DeploymentOverrides.WorkersImage != "" &&
		dgdrSpokeSpecImageProjection(restored) == hubImage
	clearProfilerImage := hasRestoredWorkersImage &&
		restored.ProfilingConfig.ProfilerImage == "" &&
		dst.ProfilingConfig.ProfilerImage == restored.DeploymentOverrides.WorkersImage
	// ProfilerImage and DeploymentOverrides.WorkersImage both collapse to
	// hub Image. If Image came from WorkersImage, do not invent ProfilerImage.
	if clearProfilerImage {
		dst.ProfilingConfig.ProfilerImage = ""
	}
	if !hasRestoredWorkersImage {
		return
	}
	if dst.DeploymentOverrides == nil {
		dst.DeploymentOverrides = &DeploymentOverridesSpec{}
	}
	if dst.DeploymentOverrides.WorkersImage == "" {
		dst.DeploymentOverrides.WorkersImage = restored.DeploymentOverrides.WorkersImage
	}
}

func dgdrSpokeSpecImageProjection(src *DynamoGraphDeploymentRequestSpec) string {
	if src == nil {
		return ""
	}
	if src.ProfilingConfig.ProfilerImage != "" {
		return src.ProfilingConfig.ProfilerImage
	}
	if src.DeploymentOverrides != nil {
		return src.DeploymentOverrides.WorkersImage
	}
	return ""
}

// convertDGDRStatusToHub converts represented status fields from live v1alpha1
// state, restores hub-only status leaves from restored target data, and saves
// source-only v1alpha1 leaves.
func convertDGDRStatusToHub(src *DynamoGraphDeploymentRequestStatus, dst *v1beta1.DynamoGraphDeploymentRequestStatus, restored *dgdrHubStatusPreservation, save *DynamoGraphDeploymentRequestStatus, ctx dgdrConversionContext) {
	if src == nil || dst == nil {
		return
	}

	dst.Phase = dgdrStateToPhase(string(src.State), src.Deployment)
	dst.ObservedGeneration = src.ObservedGeneration
	dst.Conditions = slices.Clone(src.Conditions)

	if src.Backend != "" {
		setAnnOnObj(ctx.hubObject, annDGDRStatusBackend, src.Backend)
	}
	if src.ProfilingResults != "" {
		setAnnOnObj(ctx.hubObject, annDGDRProfilingResults, src.ProfilingResults)
	}
	if src.GeneratedDeployment != nil {
		if dst.ProfilingResults == nil {
			dst.ProfilingResults = &v1beta1.ProfilingResultsStatus{}
		}
		dst.ProfilingResults.SelectedConfig = src.GeneratedDeployment.DeepCopy()
	}
	if src.Deployment != nil {
		dst.DGDName = src.Deployment.Name
		if dgdrDeploymentStatusNeedsAnnotation(src.Deployment, dst.Phase) {
			setJSONAnnOnObj(ctx.hubObject, annDGDRDeploymentStatus, src.Deployment)
		}
	}

	// ProfilingJobName has no v1alpha1 status field, so ConvertFrom stores it
	// as an annotation and ConvertTo passes the captured value in after scrubbing.
	if ctx.profilingJobName != "" {
		dst.ProfilingJobName = ctx.profilingJobName
	}
	restoreDGDRHubStatus(dst, restored, dgdrAlphaStatusNeedsSave(src))
	if save != nil {
		saveDGDRAlphaOnlyStatus(src, save)
	}
}

// convertDGDRStatusFromHub converts represented status fields from the live hub
// source, restores v1alpha1-only status leaves from restored target data, and
// saves hub-only status leaves.
func convertDGDRStatusFromHub(src *v1beta1.DynamoGraphDeploymentRequestStatus, dst *DynamoGraphDeploymentRequestStatus, restored *DynamoGraphDeploymentRequestStatus, save *dgdrHubStatusPreservation, ctx dgdrConversionContext) {
	if src == nil || dst == nil {
		return
	}

	dst.State = DGDRState(dgdrPhaseToState(src.Phase))
	dst.ObservedGeneration = src.ObservedGeneration
	dst.Conditions = slices.Clone(src.Conditions)

	if ctx.hubObject.Annotations != nil {
		if v, ok := ctx.hubObject.Annotations[annDGDRStatusBackend]; ok {
			dst.Backend = v
		}
		if v, ok := ctx.hubObject.Annotations[annDGDRProfilingResults]; ok {
			dst.ProfilingResults = v
		}
	}

	if src.ProfilingResults != nil && src.ProfilingResults.SelectedConfig != nil {
		dst.GeneratedDeployment = src.ProfilingResults.SelectedConfig.DeepCopy()
	}

	if ctx.hubObject.Annotations != nil {
		if v, ok := ctx.hubObject.Annotations[annDGDRDeploymentStatus]; ok && v != "" {
			var depStatus DeploymentStatus
			if err := json.Unmarshal([]byte(v), &depStatus); err == nil {
				annotatedName := depStatus.Name
				depStatus.Name = src.DGDName
				createdForCurrentDeployment := depStatus.Created && (annotatedName == "" || annotatedName == src.DGDName)
				depStatus.Created = createdForCurrentDeployment || src.Phase == v1beta1.DGDRPhaseDeployed
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
	restoreDGDRSpokeStatus(dst, restored, src.Phase)
	if save != nil {
		saveDGDRHubOnlyStatus(src, save)
	}
}

func dgdrDeploymentStatusNeedsAnnotation(src *DeploymentStatus, phase v1beta1.DGDRPhase) bool {
	return src != nil &&
		(src.Namespace != "" ||
			src.State != "" ||
			src.Created && phase != v1beta1.DGDRPhaseDeployed)
}

func restoreDGDRHubStatus(dst *v1beta1.DynamoGraphDeploymentRequestStatus, restored *dgdrHubStatusPreservation, sourceStateNeedsSave bool) {
	if restored == nil {
		return
	}
	if restored.PhaseSet && !sourceStateNeedsSave && dgdrHubPhaseCompatibleWithSaved(dst.Phase, &restored.Status) {
		dst.Phase = restored.Status.Phase
	}
	if dst.ProfilingPhase == "" {
		dst.ProfilingPhase = restored.Status.ProfilingPhase
	}
	if dst.ProfilingJobName == "" {
		dst.ProfilingJobName = restored.Status.ProfilingJobName
	}
	if dst.DeploymentInfo == nil {
		dst.DeploymentInfo = restored.Status.DeploymentInfo
	}
	if restored.Status.ProfilingResults != nil && len(restored.Status.ProfilingResults.Pareto) > 0 {
		if dst.ProfilingResults == nil {
			dst.ProfilingResults = &v1beta1.ProfilingResultsStatus{}
		}
		dst.ProfilingResults.Pareto = restored.Status.ProfilingResults.DeepCopy().Pareto
	}
}

func dgdrHubPhaseCompatibleWithSaved(current v1beta1.DGDRPhase, saved *v1beta1.DynamoGraphDeploymentRequestStatus) bool {
	if saved == nil {
		return false
	}
	if current == saved.Phase {
		return true
	}
	if saved.Phase == "" && current == v1beta1.DGDRPhasePending {
		return true
	}
	if saved.Phase == v1beta1.DGDRPhaseDeployed && saved.DGDName != "" {
		return current == v1beta1.DGDRPhaseDeployed
	}
	return dgdrPhaseToState(current) == dgdrPhaseToState(saved.Phase)
}

func restoreDGDRSpokeStatus(dst *DynamoGraphDeploymentRequestStatus, restored *DynamoGraphDeploymentRequestStatus, hubPhase v1beta1.DGDRPhase) {
	if restored == nil {
		return
	}
	if dgdrStateToPhase(string(restored.State), dst.Deployment) == hubPhase {
		dst.State = restored.State
	}
}

func dgdrNeedsHubPreservation(specSave *dgdrHubSpecPreservation, statusSave *dgdrHubStatusPreservation) bool {
	if !dgdrHubSpecSaveIsZero(specSave) {
		return true
	}
	if !dgdrHubStatusSaveIsZero(statusSave) {
		return true
	}
	return false
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
