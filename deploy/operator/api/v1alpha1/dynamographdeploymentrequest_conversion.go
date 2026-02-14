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
//	Spec.Model              (string)           Spec.Model.ModelName          (string)
//	Spec.Backend            (string)           Spec.Backend.Backend      (BackendType)
//	Spec.AutoApply          (bool)             Spec.AutoApply                  (bool)
//	Spec.UseMocker          (bool)             Spec.Features.Mocker.Enabled    (bool)
//	Spec.DeploymentOverrides.WorkersImage      Spec.Backend.DynamoImage      (string)
//
// JSON blob → structured fields (parsed/reconstructed on each trip):
//
//	Blob key path                              v1beta1 field
//	──────────────────────────────────────────  ──────────────────────────────────────
//	sla.ttft                                   Spec.SLA.TTFT             (*float64)
//	sla.itl                                    Spec.SLA.ITL              (*float64)
//	sla.isl                                    Spec.Workload.ISL           (*int32)
//	sla.osl                                    Spec.Workload.OSL           (*int32)
//	deployment.modelCache.pvcName              Spec.Model.ModelCache.PVCName
//	deployment.modelCache.modelPathInPvc       Spec.Model.ModelCache.ModelPathInPVC
//	deployment.modelCache.pvcMountPath         Spec.Model.ModelCache.PVCMountPath
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
//	Spec.EnableGpuDiscovery                    nvidia.com/dgdr-enable-gpu-discovery
//	Spec.ProfilingConfig.ProfilerImage         nvidia.com/dgdr-profiler-image
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
	annDGDRProfilerImage    = "nvidia.com/dgdr-profiler-image"
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

	// ObjectMeta
	dst.ObjectMeta = src.ObjectMeta

	// Spec
	convertDGDRSpecTo(&src.Spec, &dst.Spec, dst)

	// Status
	convertDGDRStatusTo(&src.Status, &dst.Status, dst)

	return nil
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoGraphDeploymentRequest (v1alpha1).
func (dst *DynamoGraphDeploymentRequest) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentRequest but got %T", srcRaw)
	}

	// ObjectMeta
	dst.ObjectMeta = src.ObjectMeta

	// Spec
	convertDGDRSpecFrom(&src.Spec, &dst.Spec, src)

	// Status
	convertDGDRStatusFrom(&src.Status, &dst.Status, src)

	return nil
}

// setAnnotation is a helper that initialises the annotation map if needed and sets a key.
func setAnnotation(obj *v1beta1.DynamoGraphDeploymentRequest, key, value string) {
	if obj.Annotations == nil {
		obj.Annotations = make(map[string]string)
	}
	obj.Annotations[key] = value
}

// convertDGDRSpecTo converts the v1alpha1 Spec into the v1beta1 Spec.
func convertDGDRSpecTo(src *DynamoGraphDeploymentRequestSpec, dst *v1beta1.DynamoGraphDeploymentRequestSpec, dstObj *v1beta1.DynamoGraphDeploymentRequest) {
	// --- Simple fields ---
	dst.Model.ModelName = src.Model
	dst.AutoApply = src.AutoApply

	// Backend
	if src.Backend != "" {
		dst.Backend = &v1beta1.BackendSpec{
			Backend: v1beta1.BackendType(src.Backend),
		}
	}

	// WorkersImage → Backend.DynamoImage
	if src.DeploymentOverrides != nil && src.DeploymentOverrides.WorkersImage != "" {
		if dst.Backend == nil {
			dst.Backend = &v1beta1.BackendSpec{}
		}
		dst.Backend.DynamoImage = src.DeploymentOverrides.WorkersImage
	}

	// UseMocker → Features.Mocker.Enabled
	if src.UseMocker {
		if dst.Features == nil {
			dst.Features = &v1beta1.FeaturesSpec{}
		}
		dst.Features.Mocker = &v1beta1.MockerSpec{Enabled: true}
	}

	// EnableGpuDiscovery — no v1beta1 field; store as annotation
	if src.EnableGpuDiscovery {
		setAnnotation(dstObj, annDGDREnableGPUDisc, "true")
	}

	// --- Parse the JSON blob to extract structured fields ---
	if src.ProfilingConfig.Config != nil && src.ProfilingConfig.Config.Raw != nil {
		var blob map[string]interface{}
		if err := json.Unmarshal(src.ProfilingConfig.Config.Raw, &blob); err == nil {
			// SLA fields from blob.sla
			if slaRaw, ok := blob["sla"]; ok {
				if slaMap, ok := slaRaw.(map[string]interface{}); ok {
					if dst.SLA == nil {
						dst.SLA = &v1beta1.SLASpec{}
					}
					if v, ok := slaMap["ttft"].(float64); ok {
						dst.SLA.TTFT = &v
					}
					if v, ok := slaMap["itl"].(float64); ok {
						dst.SLA.ITL = &v
					}
				}
			}

			// Workload fields from blob.sla (ISL, OSL are under sla in the blob)
			if slaRaw, ok := blob["sla"]; ok {
				if slaMap, ok := slaRaw.(map[string]interface{}); ok {
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
			}

			// ModelCache from blob.deployment.modelCache
			if deployRaw, ok := blob["deployment"]; ok {
				if deployMap, ok := deployRaw.(map[string]interface{}); ok {
					if mcRaw, ok := deployMap["modelCache"]; ok {
						if mcMap, ok := mcRaw.(map[string]interface{}); ok {
							mc := &v1beta1.ModelCacheSpec{}
							if v, ok := mcMap["pvcName"].(string); ok {
								mc.PVCName = v
							}
							if v, ok := mcMap["modelPathInPvc"].(string); ok {
								mc.ModelPathInPVC = v
							}
							if v, ok := mcMap["pvcMountPath"].(string); ok {
								mc.PVCMountPath = v
							}
							dst.Model.ModelCache = mc
						}
					}
				}
			}
		}

		// Preserve the full JSON blob as annotation for round-trip
		setAnnotation(dstObj, annDGDRProfilingConfig, string(src.ProfilingConfig.Config.Raw))
	}

	// ProfilerImage — no v1beta1 field; store as annotation
	if src.ProfilingConfig.ProfilerImage != "" {
		setAnnotation(dstObj, annDGDRProfilerImage, src.ProfilingConfig.ProfilerImage)
	}

	// ConfigMapRef — no v1beta1 field; store as annotation
	if src.ProfilingConfig.ConfigMapRef != nil {
		if data, err := json.Marshal(src.ProfilingConfig.ConfigMapRef); err == nil {
			setAnnotation(dstObj, annDGDRConfigMapRef, string(data))
		}
	}

	// OutputPVC — no v1beta1 field; store as annotation
	if src.ProfilingConfig.OutputPVC != "" {
		setAnnotation(dstObj, annDGDROutputPVC, src.ProfilingConfig.OutputPVC)
	}

	// Resources, Tolerations → Overrides.ProfilingJob
	if src.ProfilingConfig.Resources != nil || len(src.ProfilingConfig.Tolerations) > 0 {
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

		if src.ProfilingConfig.Resources != nil {
			// Ensure at least one container exists
			if len(podSpec.Containers) == 0 {
				podSpec.Containers = []corev1.Container{{}}
			}
			podSpec.Containers[0].Resources = *src.ProfilingConfig.Resources
		}

		if len(src.ProfilingConfig.Tolerations) > 0 {
			podSpec.Tolerations = src.ProfilingConfig.Tolerations
		}
	}

	// DeploymentOverrides (Name, Namespace, Labels, Annotations) — no v1beta1 equivalent; store as annotation
	if src.DeploymentOverrides != nil {
		overrides := struct {
			Name        string            `json:"name,omitempty"`
			Namespace   string            `json:"namespace,omitempty"`
			Labels      map[string]string `json:"labels,omitempty"`
			Annotations map[string]string `json:"annotations,omitempty"`
		}{
			Name:        src.DeploymentOverrides.Name,
			Namespace:   src.DeploymentOverrides.Namespace,
			Labels:      src.DeploymentOverrides.Labels,
			Annotations: src.DeploymentOverrides.Annotations,
		}
		// Only store if there's meaningful data (beyond WorkersImage which is handled above)
		if overrides.Name != "" || overrides.Namespace != "" || len(overrides.Labels) > 0 || len(overrides.Annotations) > 0 {
			if data, err := json.Marshal(overrides); err == nil {
				setAnnotation(dstObj, annDGDRDeployOverrides, string(data))
			}
		}
	}
}

// convertDGDRSpecFrom converts the v1beta1 Spec back into the v1alpha1 Spec.
func convertDGDRSpecFrom(src *v1beta1.DynamoGraphDeploymentRequestSpec, dst *DynamoGraphDeploymentRequestSpec, srcObj *v1beta1.DynamoGraphDeploymentRequest) {
	// --- Simple fields ---
	dst.Model = src.Model.ModelName
	dst.AutoApply = src.AutoApply

	// Backend
	if src.Backend != nil {
		dst.Backend = string(src.Backend.Backend)
	}

	// UseMocker
	if src.Features != nil && src.Features.Mocker != nil {
		dst.UseMocker = src.Features.Mocker.Enabled
	}

	// EnableGpuDiscovery from annotation
	if srcObj.Annotations != nil {
		if v, ok := srcObj.Annotations[annDGDREnableGPUDisc]; ok && v == "true" {
			dst.EnableGpuDiscovery = true
		}
	}

	// --- Reconstruct the JSON blob ---
	// Start from the round-trip annotation if available, otherwise build from structured fields.
	var blob map[string]interface{}
	if srcObj.Annotations != nil {
		if rawBlob, ok := srcObj.Annotations[annDGDRProfilingConfig]; ok && rawBlob != "" {
			_ = json.Unmarshal([]byte(rawBlob), &blob) // best-effort
		}
	}

	// Override blob values from structured fields (they take precedence)
	if src.SLA != nil || src.Workload != nil {
		if blob == nil {
			blob = make(map[string]interface{})
		}
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

	// ModelCache into blob.deployment.modelCache
	if src.Model.ModelCache != nil {
		if blob == nil {
			blob = make(map[string]interface{})
		}
		deployMap, _ := blob["deployment"].(map[string]interface{})
		if deployMap == nil {
			deployMap = make(map[string]interface{})
		}
		mcMap := make(map[string]interface{})
		if src.Model.ModelCache.PVCName != "" {
			mcMap["pvcName"] = src.Model.ModelCache.PVCName
		}
		if src.Model.ModelCache.ModelPathInPVC != "" {
			mcMap["modelPathInPvc"] = src.Model.ModelCache.ModelPathInPVC
		}
		if src.Model.ModelCache.PVCMountPath != "" {
			mcMap["pvcMountPath"] = src.Model.ModelCache.PVCMountPath
		}
		if len(mcMap) > 0 {
			deployMap["modelCache"] = mcMap
			blob["deployment"] = deployMap
		}
	}

	// Marshal the blob
	if blob != nil {
		if data, err := json.Marshal(blob); err == nil {
			dst.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: data}
		}
	}

	// ProfilerImage from annotation
	if srcObj.Annotations != nil {
		if v, ok := srcObj.Annotations[annDGDRProfilerImage]; ok {
			dst.ProfilingConfig.ProfilerImage = v
		}
	}

	// ConfigMapRef from annotation
	if srcObj.Annotations != nil {
		if v, ok := srcObj.Annotations[annDGDRConfigMapRef]; ok && v != "" {
			var ref ConfigMapKeySelector
			if err := json.Unmarshal([]byte(v), &ref); err == nil {
				dst.ProfilingConfig.ConfigMapRef = &ref
			}
		}
	}

	// OutputPVC from annotation
	if srcObj.Annotations != nil {
		if v, ok := srcObj.Annotations[annDGDROutputPVC]; ok {
			dst.ProfilingConfig.OutputPVC = v
		}
	}

	// Resources, Tolerations from Overrides.ProfilingJob
	if src.Overrides != nil && src.Overrides.ProfilingJob != nil {
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

	// DeploymentOverrides from annotation + Backend.DynamoImage
	if srcObj.Annotations != nil {
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
	if src.Backend != nil && src.Backend.DynamoImage != "" {
		if dst.DeploymentOverrides == nil {
			dst.DeploymentOverrides = &DeploymentOverridesSpec{}
		}
		dst.DeploymentOverrides.WorkersImage = src.Backend.DynamoImage
	}
}

// convertDGDRStatusTo converts the v1alpha1 Status into the v1beta1 Status.
func convertDGDRStatusTo(src *DynamoGraphDeploymentRequestStatus, dst *v1beta1.DynamoGraphDeploymentRequestStatus, dstObj *v1beta1.DynamoGraphDeploymentRequest) {
	// State → Phase
	dst.Phase = dgdrStateToPhase(src.State, src.Deployment)
	dst.ObservedGeneration = src.ObservedGeneration
	dst.Conditions = src.Conditions

	// Backend — no v1beta1 status equivalent; store as annotation
	if src.Backend != "" {
		setAnnotation(dstObj, annDGDRStatusBackend, src.Backend)
	}

	// ProfilingResults (string ref) — store as annotation
	if src.ProfilingResults != "" {
		setAnnotation(dstObj, annDGDRProfilingResults, src.ProfilingResults)
	}

	// GeneratedDeployment → ProfilingResults.SelectedConfig
	if src.GeneratedDeployment != nil {
		if dst.ProfilingResults == nil {
			dst.ProfilingResults = &v1beta1.ProfilingResultsStatus{}
		}
		dst.ProfilingResults.SelectedConfig = src.GeneratedDeployment
	}

	// Deployment → DGDName + DeploymentInfo + annotation for full struct
	if src.Deployment != nil {
		dst.DGDName = src.Deployment.Name
		// Store the full deployment status as annotation for round-trip
		if data, err := json.Marshal(src.Deployment); err == nil {
			setAnnotation(dstObj, annDGDRDeploymentStatus, string(data))
		}
	}
}

// convertDGDRStatusFrom converts the v1beta1 Status back into the v1alpha1 Status.
func convertDGDRStatusFrom(src *v1beta1.DynamoGraphDeploymentRequestStatus, dst *DynamoGraphDeploymentRequestStatus, srcObj *v1beta1.DynamoGraphDeploymentRequest) {
	// Phase → State
	dst.State = dgdrPhaseToState(src.Phase)
	dst.ObservedGeneration = src.ObservedGeneration
	dst.Conditions = src.Conditions

	// Backend from annotation
	if srcObj.Annotations != nil {
		if v, ok := srcObj.Annotations[annDGDRStatusBackend]; ok {
			dst.Backend = v
		}
	}

	// ProfilingResults from annotation
	if srcObj.Annotations != nil {
		if v, ok := srcObj.Annotations[annDGDRProfilingResults]; ok {
			dst.ProfilingResults = v
		}
	}

	// ProfilingResults.SelectedConfig → GeneratedDeployment
	if src.ProfilingResults != nil && src.ProfilingResults.SelectedConfig != nil {
		dst.GeneratedDeployment = src.ProfilingResults.SelectedConfig
	}

	// Deployment status — restore from annotation if available, otherwise reconstruct minimally
	if srcObj.Annotations != nil {
		if v, ok := srcObj.Annotations[annDGDRDeploymentStatus]; ok && v != "" {
			var depStatus DeploymentStatus
			if err := json.Unmarshal([]byte(v), &depStatus); err == nil {
				dst.Deployment = &depStatus
			}
		}
	}
	// If no annotation, but we have DGDName, create a minimal deployment status
	if dst.Deployment == nil && src.DGDName != "" {
		dst.Deployment = &DeploymentStatus{
			Name:    src.DGDName,
			Created: true,
		}
	}
}

// dgdrStateToPhase maps v1alpha1 state strings to v1beta1 DGDRPhase.
func dgdrStateToPhase(state string, deployment *DeploymentStatus) v1beta1.DGDRPhase {
	switch state {
	case "", "Pending":
		return v1beta1.DGDRPhasePending
	case "Profiling":
		return v1beta1.DGDRPhaseProfiling
	case "Ready":
		// If there is a deployment that was created, it means we are actually Deployed
		if deployment != nil && deployment.Created {
			return v1beta1.DGDRPhaseDeployed
		}
		return v1beta1.DGDRPhaseReady
	case "Deploying":
		return v1beta1.DGDRPhaseDeploying
	case "DeploymentDeleted":
		return v1beta1.DGDRPhaseReady
	case "Failed":
		return v1beta1.DGDRPhaseFailed
	default:
		return v1beta1.DGDRPhasePending
	}
}

// dgdrPhaseToState maps v1beta1 DGDRPhase to v1alpha1 state strings.
func dgdrPhaseToState(phase v1beta1.DGDRPhase) string {
	switch phase {
	case v1beta1.DGDRPhasePending:
		return "Pending"
	case v1beta1.DGDRPhaseProfiling:
		return "Profiling"
	case v1beta1.DGDRPhaseReady:
		return "Ready"
	case v1beta1.DGDRPhaseDeploying:
		return "Deploying"
	case v1beta1.DGDRPhaseDeployed:
		return "Ready"
	case v1beta1.DGDRPhaseFailed:
		return "Failed"
	default:
		return "Pending"
	}
}
