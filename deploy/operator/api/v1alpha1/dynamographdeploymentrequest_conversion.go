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
//	Spec.AutoApply          (bool)             Spec.AutoApply                  (bool)
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
// v1beta1-only fields with no v1alpha1 equivalent (omitted / TODO):
//
//	Hardware.*, Workload.{Concurrency,RequestRate}, SLA.{E2ELatency,OptimizationType},
//	Features.{Planner.*,KVRouter}, SearchStrategy
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

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
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
)

// ConvertTo converts this DynamoGraphDeploymentRequest (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoGraphDeploymentRequest) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentRequest but got %T", dstRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	if err := convertDGDRSpecTo(&src.Spec, &dst.Spec, dst); err != nil {
		return err
	}

	convertDGDRStatusTo(&src.Status, &dst.Status, dst)

	return nil
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoGraphDeploymentRequest (v1alpha1).
func (dst *DynamoGraphDeploymentRequest) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentRequest but got %T", srcRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	convertDGDRSpecFrom(&src.Spec, &dst.Spec, src)
	convertDGDRStatusFrom(&src.Status, &dst.Status, src)

	// ProfilingJobName — no v1alpha1 status field; store as annotation for round-trip
	if src.Status.ProfilingJobName != "" {
		if dst.Annotations == nil {
			dst.Annotations = make(map[string]string)
		}
		dst.Annotations[annDGDRProfilingJobName] = src.Status.ProfilingJobName
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

// convertDGDRSpecTo converts the v1alpha1 Spec into the v1beta1 Spec.
func convertDGDRSpecTo(src *DynamoGraphDeploymentRequestSpec, dst *v1beta1.DynamoGraphDeploymentRequestSpec, dstObj *v1beta1.DynamoGraphDeploymentRequest) error {
	dst.Model = src.Model
	dst.AutoApply = src.AutoApply

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
		setAnnotation(dstObj, annDGDREnableGPUDisc, "true")
	}

	if src.ProfilingConfig.Config != nil && src.ProfilingConfig.Config.Raw != nil {
		var blob map[string]interface{}
		if err := json.Unmarshal(src.ProfilingConfig.Config.Raw, &blob); err != nil {
			return fmt.Errorf("failed to parse ProfilingConfig.Config: %w", err)
		}
		applySLAAndWorkloadFromBlob(blob, dst)
		applyModelCacheFromBlob(blob, dst)
		setAnnotation(dstObj, annDGDRProfilingConfig, string(src.ProfilingConfig.Config.Raw))
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

// convertDGDRSpecFrom converts the v1beta1 Spec back into the v1alpha1 Spec.
func convertDGDRSpecFrom(src *v1beta1.DynamoGraphDeploymentRequestSpec, dst *DynamoGraphDeploymentRequestSpec, srcObj *v1beta1.DynamoGraphDeploymentRequest) {
	dst.Model = src.Model
	dst.AutoApply = src.AutoApply

	if src.Backend != "" {
		dst.Backend = string(src.Backend)
	}
	if src.Features != nil && src.Features.Mocker != nil {
		dst.UseMocker = src.Features.Mocker.Enabled
	}

	if srcObj.Annotations != nil {
		if v, ok := srcObj.Annotations[annDGDREnableGPUDisc]; ok && v == "true" {
			trueVal := true
			dst.EnableGPUDiscovery = &trueVal
		}
	}

	// Reconstruct the JSON blob: start from the round-trip annotation (preserves unknown
	// keys), then overwrite with structured v1beta1 fields (structured fields win).
	var blob map[string]interface{}
	if srcObj.Annotations != nil {
		if rawBlob, ok := srcObj.Annotations[annDGDRProfilingConfig]; ok && rawBlob != "" {
			_ = json.Unmarshal([]byte(rawBlob), &blob) // best-effort
		}
	}
	if src.SLA != nil || src.Workload != nil {
		if blob == nil {
			blob = make(map[string]interface{})
		}
		mergeSLAWorkloadIntoBlob(src, blob)
	}
	if src.ModelCache != nil {
		if blob == nil {
			blob = make(map[string]interface{})
		}
		mergeModelCacheIntoBlob(src.ModelCache, blob)
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
}

// mergeSLAWorkloadIntoBlob writes SLA and Workload structured fields back into the JSON blob,
// overwriting any existing values for those keys.
func mergeSLAWorkloadIntoBlob(src *v1beta1.DynamoGraphDeploymentRequestSpec, blob map[string]interface{}) {
	slaMap, _ := blob["sla"].(map[string]interface{})
	if slaMap == nil {
		slaMap = make(map[string]interface{})
	}
	if src.SLA != nil {
		if src.SLA.TTFT != nil {
			slaMap["ttft"] = *src.SLA.TTFT
		}
		if src.SLA.ITL != nil {
			slaMap["itl"] = *src.SLA.ITL
		}
	}
	if src.Workload != nil {
		if src.Workload.ISL != nil {
			slaMap["isl"] = float64(*src.Workload.ISL)
		}
		if src.Workload.OSL != nil {
			slaMap["osl"] = float64(*src.Workload.OSL)
		}
	}
	blob["sla"] = slaMap
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
		if len(res.Requests) > 0 || len(res.Limits) > 0 {
			dst.ProfilingConfig.Resources = &res
		}
	}
	if len(podSpec.Tolerations) > 0 {
		dst.ProfilingConfig.Tolerations = podSpec.Tolerations
	}
}

// convertDGDRStatusTo converts the v1alpha1 Status into the v1beta1 Status.
func convertDGDRStatusTo(src *DynamoGraphDeploymentRequestStatus, dst *v1beta1.DynamoGraphDeploymentRequestStatus, dstObj *v1beta1.DynamoGraphDeploymentRequest) {
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

	// ProfilingJobName — read from annotation (stored during ConvertFrom for round-trip)
	if ann, ok := dstObj.Annotations[annDGDRProfilingJobName]; ok && ann != "" {
		dst.ProfilingJobName = ann
	}
}

// convertDGDRStatusFrom converts the v1beta1 Status back into the v1alpha1 Status.
func convertDGDRStatusFrom(src *v1beta1.DynamoGraphDeploymentRequestStatus, dst *DynamoGraphDeploymentRequestStatus, srcObj *v1beta1.DynamoGraphDeploymentRequest) {
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
				dst.Deployment = &depStatus
			}
		}
	}
	// If no annotation but we have DGDName, create a minimal deployment status.
	// Created is left false so the v1alpha1 controller does not skip re-creating the DGD.
	if dst.Deployment == nil && src.DGDName != "" {
		dst.Deployment = &DeploymentStatus{
			Name:    src.DGDName,
			Created: false,
		}
	}
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
