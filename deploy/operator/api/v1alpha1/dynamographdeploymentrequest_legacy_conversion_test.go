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

package v1alpha1

import (
	"encoding/json"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// legacyDGDRConvertToHubForTest is the pre-structural DGDR converter kept as a
// compatibility oracle while we still write legacy annotations for downgrade.
func legacyDGDRConvertToHubForTest(src *DynamoGraphDeploymentRequest) (*v1beta1.DynamoGraphDeploymentRequest, error) {
	dst := &v1beta1.DynamoGraphDeploymentRequest{}
	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()

	dst.Spec.Model = src.Spec.Model
	dst.Spec.AutoApply = &src.Spec.AutoApply
	if src.Spec.Backend != "" {
		dst.Spec.Backend = v1beta1.BackendType(src.Spec.Backend)
	}
	if src.Spec.DeploymentOverrides != nil && src.Spec.DeploymentOverrides.WorkersImage != "" {
		dst.Spec.Image = src.Spec.DeploymentOverrides.WorkersImage
	}
	if src.Spec.UseMocker {
		if dst.Spec.Features == nil {
			dst.Spec.Features = &v1beta1.FeaturesSpec{}
		}
		dst.Spec.Features.Mocker = &v1beta1.MockerSpec{Enabled: true}
	}
	if src.Spec.EnableGPUDiscovery != nil && *src.Spec.EnableGPUDiscovery {
		setAnnotation(dst, legacyAnnDGDREnableGPUDisc, annotationTrue)
	}
	if src.Spec.ProfilingConfig.Config != nil && src.Spec.ProfilingConfig.Config.Raw != nil {
		var blob map[string]any
		if err := json.Unmarshal(src.Spec.ProfilingConfig.Config.Raw, &blob); err != nil {
			return nil, err
		}
		legacyDGDRApplySLAAndWorkloadFromBlob(blob, &dst.Spec)
		legacyDGDRApplyModelCacheFromBlob(blob, &dst.Spec)
		legacyDGDRApplyPlannerFromBlob(blob, &dst.Spec)
		setAnnotation(dst, legacyAnnDGDRProfilingConfig, string(src.Spec.ProfilingConfig.Config.Raw))
	}
	if src.Spec.ProfilingConfig.ProfilerImage != "" {
		dst.Spec.Image = src.Spec.ProfilingConfig.ProfilerImage
	}
	if src.Spec.ProfilingConfig.ConfigMapRef != nil {
		if data, err := json.Marshal(src.Spec.ProfilingConfig.ConfigMapRef); err == nil {
			setAnnotation(dst, legacyAnnDGDRConfigMapRef, string(data))
		}
	}
	if src.Spec.ProfilingConfig.OutputPVC != "" {
		setAnnotation(dst, legacyAnnDGDROutputPVC, src.Spec.ProfilingConfig.OutputPVC)
	}
	legacyDGDRConvertProfilingResourcesToOverrides(&src.Spec.ProfilingConfig, &dst.Spec)
	saveDGDRLegacyDeploymentOverridesAnnotation(src.Spec.DeploymentOverrides, dst)

	dst.Status.Phase = dgdrStateToPhase(string(src.Status.State), src.Status.Deployment)
	dst.Status.ObservedGeneration = src.Status.ObservedGeneration
	dst.Status.Conditions = src.Status.Conditions
	if src.Status.Backend != "" {
		setAnnotation(dst, legacyAnnDGDRStatusBackend, src.Status.Backend)
	}
	if src.Status.ProfilingResults != "" {
		setAnnotation(dst, legacyAnnDGDRProfilingResults, src.Status.ProfilingResults)
	}
	if src.Status.GeneratedDeployment != nil {
		dst.Status.ProfilingResults = &v1beta1.ProfilingResultsStatus{SelectedConfig: src.Status.GeneratedDeployment}
	}
	if src.Status.Deployment != nil {
		dst.Status.DGDName = src.Status.Deployment.Name
		payload := dgdrDeploymentStatusAnnotation{
			DeploymentStatus: *src.Status.Deployment,
			RequestState:     src.Status.State,
		}
		if data, err := json.Marshal(payload); err == nil {
			setAnnotation(dst, legacyAnnDGDRDeploymentStatus, string(data))
		}
	}
	if ann, ok := dst.Annotations[legacyAnnDGDRProfilingJobName]; ok && ann != "" {
		dst.Status.ProfilingJobName = ann
	}
	return dst, nil
}

// legacyDGDRConvertFromHubForTest is the pre-structural hub-to-spoke converter
// kept as a downgrade-read oracle for objects written by the new converter.
func legacyDGDRConvertFromHubForTest(src *v1beta1.DynamoGraphDeploymentRequest) *DynamoGraphDeploymentRequest {
	dst := &DynamoGraphDeploymentRequest{}
	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()

	legacyDGDRConvertHubSpecForTest(src, &dst.Spec)
	legacyDGDRConvertHubStatusForTest(src, &dst.Status, &dst.ObjectMeta)
	return dst
}

func legacyDGDRConvertHubSpecForTest(src *v1beta1.DynamoGraphDeploymentRequest, dst *DynamoGraphDeploymentRequestSpec) {
	dst.Model = src.Spec.Model
	if src.Spec.AutoApply != nil {
		dst.AutoApply = *src.Spec.AutoApply
	} else {
		dst.AutoApply = true
	}
	if src.Spec.Backend != "" {
		dst.Backend = string(src.Spec.Backend)
	}
	if src.Spec.Features != nil && src.Spec.Features.Mocker != nil {
		dst.UseMocker = src.Spec.Features.Mocker.Enabled
	}
	if raw, ok := getAnnFromObj(&src.ObjectMeta, legacyAnnDGDREnableGPUDisc); ok && raw == annotationTrue {
		v := true
		dst.EnableGPUDiscovery = &v
	}

	var blob map[string]any
	if raw, ok := getAnnFromObj(&src.ObjectMeta, legacyAnnDGDRProfilingConfig); ok && raw != "" {
		_ = json.Unmarshal([]byte(raw), &blob)
	}
	if src.Spec.SLA != nil || src.Spec.Workload != nil {
		if blob == nil {
			blob = map[string]any{}
		}
		legacyDGDRMergeSLAWorkloadIntoBlob(&src.Spec, blob)
	}
	if src.Spec.ModelCache != nil {
		if blob == nil {
			blob = map[string]any{}
		}
		legacyDGDRMergeModelCacheIntoBlob(src.Spec.ModelCache, blob)
	}
	if src.Spec.Features != nil && src.Spec.Features.Planner != nil {
		if blob == nil {
			blob = map[string]any{}
		}
		legacyDGDRMergePlannerIntoBlob(src.Spec.Features.Planner, blob)
	}
	if blob != nil {
		if data, err := json.Marshal(blob); err == nil {
			dst.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: data}
		}
	}
	if src.Spec.Image != "" {
		dst.ProfilingConfig.ProfilerImage = src.Spec.Image
	}
	legacyDGDRRestoreAnnotationFields(src, dst)
	legacyDGDRRestoreProfilingJobResources(&src.Spec, dst)
}

func legacyDGDRConvertHubStatusForTest(src *v1beta1.DynamoGraphDeploymentRequest, dst *DynamoGraphDeploymentRequestStatus, metadata *metav1.ObjectMeta) {
	dst.State = DGDRState(dgdrPhaseToState(src.Status.Phase))
	dst.ObservedGeneration = src.Status.ObservedGeneration
	dst.Conditions = src.Status.Conditions
	if v, ok := getAnnFromObj(&src.ObjectMeta, legacyAnnDGDRStatusBackend); ok {
		dst.Backend = v
	}
	if v, ok := getAnnFromObj(&src.ObjectMeta, legacyAnnDGDRProfilingResults); ok {
		dst.ProfilingResults = v
	}
	if src.Status.ProfilingResults != nil && src.Status.ProfilingResults.SelectedConfig != nil {
		dst.GeneratedDeployment = src.Status.ProfilingResults.SelectedConfig
	}
	if raw, ok := getAnnFromObj(&src.ObjectMeta, legacyAnnDGDRDeploymentStatus); ok && raw != "" {
		if depStatus, requestState, ok := restoreDGDRDeploymentStatus(raw, &src.Status); ok {
			dst.Deployment = &depStatus
			if requestState != "" {
				dst.State = requestState
			}
		}
	}
	if dst.Deployment == nil && src.Status.DGDName != "" {
		dst.Deployment = &DeploymentStatus{Name: src.Status.DGDName}
	}
	if src.Status.ProfilingJobName != "" {
		setAnnOnObj(metadata, legacyAnnDGDRProfilingJobName, src.Status.ProfilingJobName)
	}
}

func legacyDGDRConvertProfilingResourcesToOverrides(src *ProfilingConfigSpec, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
	if src.Resources == nil && len(src.Tolerations) == 0 {
		return
	}
	if dst.Overrides == nil {
		dst.Overrides = &v1beta1.OverridesSpec{}
	}
	if dst.Overrides.ProfilingJob == nil {
		dst.Overrides.ProfilingJob = &batchv1.JobSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{}}}
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

func legacyDGDRRestoreAnnotationFields(src *v1beta1.DynamoGraphDeploymentRequest, dst *DynamoGraphDeploymentRequestSpec) {
	if raw, ok := getAnnFromObj(&src.ObjectMeta, legacyAnnDGDRConfigMapRef); ok && raw != "" {
		var ref ConfigMapKeySelector
		if err := json.Unmarshal([]byte(raw), &ref); err == nil {
			dst.ProfilingConfig.ConfigMapRef = &ref
		}
	}
	if raw, ok := getAnnFromObj(&src.ObjectMeta, legacyAnnDGDROutputPVC); ok {
		dst.ProfilingConfig.OutputPVC = raw
	}
	if raw, ok := getAnnFromObj(&src.ObjectMeta, legacyAnnDGDRDeployOverrides); ok && raw != "" {
		var overrides struct {
			Name        string            `json:"name,omitempty"`
			Namespace   string            `json:"namespace,omitempty"`
			Labels      map[string]string `json:"labels,omitempty"`
			Annotations map[string]string `json:"annotations,omitempty"`
		}
		if err := json.Unmarshal([]byte(raw), &overrides); err == nil {
			dst.DeploymentOverrides = &DeploymentOverridesSpec{
				Name:        overrides.Name,
				Namespace:   overrides.Namespace,
				Labels:      overrides.Labels,
				Annotations: overrides.Annotations,
			}
		}
	}
}

func legacyDGDRRestoreProfilingJobResources(src *v1beta1.DynamoGraphDeploymentRequestSpec, dst *DynamoGraphDeploymentRequestSpec) {
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

func legacyDGDRApplySLAAndWorkloadFromBlob(blob map[string]any, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
	slaRaw, ok := blob["sla"]
	if !ok {
		return
	}
	slaMap, ok := slaRaw.(map[string]any)
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

func legacyDGDRApplyModelCacheFromBlob(blob map[string]any, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
	deployRaw, ok := blob["deployment"]
	if !ok {
		return
	}
	deployMap, ok := deployRaw.(map[string]any)
	if !ok {
		return
	}
	mcRaw, ok := deployMap["modelCache"]
	if !ok {
		return
	}
	mcMap, ok := mcRaw.(map[string]any)
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

func legacyDGDRApplyPlannerFromBlob(blob map[string]any, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
	plannerRaw, ok := blob["planner"]
	if !ok {
		return
	}
	plannerMap, ok := plannerRaw.(map[string]any)
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

func legacyDGDRMergeSLAWorkloadIntoBlob(src *v1beta1.DynamoGraphDeploymentRequestSpec, blob map[string]any) {
	slaMap, _ := blob["sla"].(map[string]any)
	if slaMap == nil {
		slaMap = make(map[string]any)
	}
	if src.SLA != nil {
		if src.SLA.TTFT != nil {
			slaMap["ttft"] = *src.SLA.TTFT
		}
		if src.SLA.ITL != nil {
			slaMap["itl"] = *src.SLA.ITL
		}
		if src.SLA.OptimizationType != nil {
			slaMap["optimizationType"] = string(*src.SLA.OptimizationType)
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

func legacyDGDRMergeModelCacheIntoBlob(mc *v1beta1.ModelCacheSpec, blob map[string]any) {
	deployMap, _ := blob["deployment"].(map[string]any)
	if deployMap == nil {
		deployMap = make(map[string]any)
	}
	mcMap := make(map[string]any)
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

func legacyDGDRMergePlannerIntoBlob(planner *runtime.RawExtension, blob map[string]any) {
	if planner == nil || planner.Raw == nil {
		return
	}
	var plannerMap map[string]any
	if err := json.Unmarshal(planner.Raw, &plannerMap); err != nil || len(plannerMap) == 0 {
		return
	}
	blob["planner"] = plannerMap
}
