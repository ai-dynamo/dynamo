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
// v1beta1 is the hub. DGDR conversion predates the structural DGD/DCD cleanup,
// so this file reads legacy Dynamo 1.0/1.1 annotations and still writes them for
// downgrade compatibility. New preservation uses sparse nvidia.com/dgdr-spec and
// nvidia.com/dgdr-status annotations.
//
// Live source fields are authoritative. Preservation annotations are old-value
// caches only for fields the live source version cannot represent.
//
// The main spec projections are:
//
//   - v1alpha1 ProfilingConfig.Config JSON keys under sla, deployment.modelCache,
//     and planner map to v1beta1 SLA, Workload, ModelCache, and Features.Planner.
//   - v1alpha1 ProfilingConfig.Resources, Tolerations, and NodeSelector map to
//     v1beta1 Overrides.ProfilingJob pod fields.
//   - v1alpha1-only spec fields are saved sparsely in annDGDRSpec and, for
//     downgrade compatibility, in legacyAnnDGDR* annotations.
//   - v1beta1-only spec fields such as Hardware, Workload Concurrency/RequestRate,
//     SLA E2ELatency, Overrides.DGD, hub-only ProfilingJob leaves, disabled Mocker,
//     and SearchStrategy are saved sparsely in annDGDRSpec.
//
// Status follows the same rules: common fields are converted from live source,
// alpha-only status is saved in annDGDRStatus and legacy annotations, and
// hub-only status such as ProfilingPhase, ProfilingJobName, Pareto results,
// DeploymentInfo, and the Deployed phase is saved in annDGDRStatus.

package v1alpha1

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/conversion"
)

const (
	annDGDRSpec   = "nvidia.com/dgdr-spec"
	annDGDRStatus = "nvidia.com/dgdr-status"

	// TODO(sttts): remove after 1.2 when downgrade does not matter anymore.
	legacyAnnDGDRConfigMapRef     = "nvidia.com/dgdr-config-map-ref"
	legacyAnnDGDROutputPVC        = "nvidia.com/dgdr-output-pvc"
	legacyAnnDGDREnableGPUDisc    = "nvidia.com/dgdr-enable-gpu-discovery"
	legacyAnnDGDRDeployOverrides  = "nvidia.com/dgdr-deployment-overrides"
	legacyAnnDGDRProfilingConfig  = "nvidia.com/dgdr-profiling-config"
	legacyAnnDGDRStatusBackend    = "nvidia.com/dgdr-status-backend"
	legacyAnnDGDRProfilingResults = "nvidia.com/dgdr-profiling-results"
	legacyAnnDGDRDeploymentStatus = "nvidia.com/dgdr-deployment-status"
	legacyAnnDGDRProfilingJobName = "nvidia.com/dgdr-profiling-job-name"
)

// dgdrDeploymentStatusAnnotation preserves alpha deployment status with its source request state.
type dgdrDeploymentStatusAnnotation struct {
	DeploymentStatus
	RequestState DGDRState `json:"requestState,omitempty"`
}

type dgdrProfilingConfigBlob = map[string]any

// ConvertTo converts this DynamoGraphDeploymentRequest (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoGraphDeploymentRequest) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentRequest but got %T", dstRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()

	restoredHubSpec, restoredHubStatus := restoreDGDRHubAnnotations(&dst.ObjectMeta)
	scrubDGDRInternalAnnotations(&dst.ObjectMeta)

	var spokeSpecSave DynamoGraphDeploymentRequestSpec
	if err := ConvertFromDynamoGraphDeploymentRequestSpec(&src.Spec, &dst.Spec, restoredHubSpec, &spokeSpecSave); err != nil {
		return err
	}

	var spokeStatusSave DynamoGraphDeploymentRequestStatus
	ConvertFromDynamoGraphDeploymentRequestStatus(&src.Status, &dst.Status, restoredHubStatus, &spokeStatusSave)
	if err := saveDGDRSpokeAnnotations(&src.Spec, &spokeSpecSave, &src.Status, &spokeStatusSave, dst); err != nil {
		return err
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

	restoredSpokeSpec, restoredSpokeStatus := restoreDGDRSpokeAnnotations(&dst.ObjectMeta)
	scrubDGDRInternalAnnotations(&dst.ObjectMeta)

	var hubSpecSave v1beta1.DynamoGraphDeploymentRequestSpec
	ConvertToDynamoGraphDeploymentRequestSpec(&src.Spec, &dst.Spec, restoredSpokeSpec, &hubSpecSave)

	var hubStatusSave v1beta1.DynamoGraphDeploymentRequestStatus
	ConvertToDynamoGraphDeploymentRequestStatus(&src.Status, &dst.Status, restoredSpokeStatus, &hubStatusSave)
	if err := saveDGDRHubAnnotations(&hubSpecSave, &src.Status, &hubStatusSave, dst); err != nil {
		return err
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

func restoreDGDRHubAnnotations(obj metav1.Object) (*v1beta1.DynamoGraphDeploymentRequestSpec, *v1beta1.DynamoGraphDeploymentRequestStatus) {
	var restoredSpec *v1beta1.DynamoGraphDeploymentRequestSpec
	var restoredStatus *v1beta1.DynamoGraphDeploymentRequestStatus
	if raw, ok := getAnnFromObj(obj, annDGDRSpec); ok && raw != "" {
		if spec, ok := restoreDGDRHubSpec(raw); ok {
			restoredSpec = &spec
		}
	}
	if raw, ok := getAnnFromObj(obj, annDGDRStatus); ok && raw != "" {
		if status, ok := restoreDGDRHubStatus(raw); ok {
			restoredStatus = &status
		}
	}
	if raw, ok := getAnnFromObj(obj, legacyAnnDGDRProfilingJobName); ok && raw != "" {
		if restoredStatus == nil {
			restoredStatus = &v1beta1.DynamoGraphDeploymentRequestStatus{}
		}
		restoredStatus.ProfilingJobName = raw
	}
	return restoredSpec, restoredStatus
}

func restoreDGDRSpokeAnnotations(obj metav1.Object) (*DynamoGraphDeploymentRequestSpec, *DynamoGraphDeploymentRequestStatus) {
	var restoredSpec *DynamoGraphDeploymentRequestSpec
	var restoredStatus *DynamoGraphDeploymentRequestStatus
	if raw, ok := getAnnFromObj(obj, annDGDRSpec); ok && raw != "" {
		if spec, ok := restoreDGDRSpokeSpec(raw); ok {
			restoredSpec = &spec
		}
	}
	if restoredSpec == nil {
		if spec := restoreDGDRLegacySpokeSpec(obj); !dgdrAlphaSpecSaveIsZero(spec) {
			restoredSpec = spec
		}
	}
	if raw, ok := getAnnFromObj(obj, annDGDRStatus); ok && raw != "" {
		if status, ok := restoreDGDRSpokeStatus(raw); ok {
			restoredStatus = &status
		}
	}
	if restoredStatus == nil {
		status := restoreDGDRLegacySpokeStatus(obj)
		if !dgdrAlphaStatusSaveIsZero(status) {
			restoredStatus = status
		}
	}
	return restoredSpec, restoredStatus
}

func scrubDGDRInternalAnnotations(obj metav1.Object) {
	for _, key := range []string{
		annDGDRSpec,
		annDGDRStatus,
		legacyAnnDGDRConfigMapRef,
		legacyAnnDGDROutputPVC,
		legacyAnnDGDREnableGPUDisc,
		legacyAnnDGDRDeployOverrides,
		legacyAnnDGDRProfilingConfig,
		legacyAnnDGDRStatusBackend,
		legacyAnnDGDRProfilingResults,
		legacyAnnDGDRDeploymentStatus,
		legacyAnnDGDRProfilingJobName,
	} {
		delAnnFromObj(obj, key)
	}
}

func saveDGDRSpokeAnnotations(srcSpec *DynamoGraphDeploymentRequestSpec, specSave *DynamoGraphDeploymentRequestSpec, srcStatus *DynamoGraphDeploymentRequestStatus, statusSave *DynamoGraphDeploymentRequestStatus, dst *v1beta1.DynamoGraphDeploymentRequest) error {
	if !dgdrAlphaSpecSaveIsZero(specSave) {
		data, err := marshalDGDRSpokeSpec(specSave)
		if err != nil {
			return fmt.Errorf("preserve DGDR spoke spec: %w", err)
		}
		setAnnOnObj(&dst.ObjectMeta, annDGDRSpec, string(data))
	}
	if !dgdrAlphaStatusSaveIsZero(statusSave) {
		if err := setJSONAnnOnObj(&dst.ObjectMeta, annDGDRStatus, statusSave); err != nil {
			return err
		}
	}
	saveDGDRLegacySpokeAnnotations(srcSpec, srcStatus, dst)
	return nil
}

func saveDGDRHubAnnotations(specSave *v1beta1.DynamoGraphDeploymentRequestSpec, srcStatus *v1beta1.DynamoGraphDeploymentRequestStatus, statusSave *v1beta1.DynamoGraphDeploymentRequestStatus, dst *DynamoGraphDeploymentRequest) error {
	if !dgdrHubSpecSaveIsZero(specSave) {
		data, err := marshalDGDRHubSpec(specSave)
		if err != nil {
			return fmt.Errorf("preserve DGDR hub spec: %w", err)
		}
		setAnnOnObj(&dst.ObjectMeta, annDGDRSpec, string(data))
	}
	if !dgdrHubStatusSaveIsZero(statusSave) {
		if err := setJSONAnnOnObj(&dst.ObjectMeta, annDGDRStatus, statusSave); err != nil {
			return err
		}
	}
	if srcStatus != nil && srcStatus.ProfilingJobName != "" {
		setAnnOnObj(&dst.ObjectMeta, legacyAnnDGDRProfilingJobName, srcStatus.ProfilingJobName)
	}
	return nil
}

func dgdrAlphaSpecSaveIsZero(save *DynamoGraphDeploymentRequestSpec) bool {
	if save == nil {
		return true
	}
	if save.ProfilingConfig.Config != nil {
		return false
	}
	return apiequality.Semantic.DeepEqual(*save, DynamoGraphDeploymentRequestSpec{})
}

func dgdrAlphaStatusSaveIsZero(save *DynamoGraphDeploymentRequestStatus) bool {
	return save == nil || apiequality.Semantic.DeepEqual(*save, DynamoGraphDeploymentRequestStatus{})
}

func dgdrHubSpecSaveIsZero(save *v1beta1.DynamoGraphDeploymentRequestSpec) bool {
	return save == nil || apiequality.Semantic.DeepEqual(*save, v1beta1.DynamoGraphDeploymentRequestSpec{})
}

func dgdrHubStatusSaveIsZero(save *v1beta1.DynamoGraphDeploymentRequestStatus) bool {
	return save == nil || apiequality.Semantic.DeepEqual(*save, v1beta1.DynamoGraphDeploymentRequestStatus{})
}

func marshalDGDRHubSpec(src *v1beta1.DynamoGraphDeploymentRequestSpec) ([]byte, error) {
	return marshalPreservedSpec(*src.DeepCopy(), nil)
}

func restoreDGDRHubSpec(raw string) (v1beta1.DynamoGraphDeploymentRequestSpec, bool) {
	return restorePreservedSpec[v1beta1.DynamoGraphDeploymentRequestSpec](raw, nil)
}

func marshalDGDRSpokeSpec(src *DynamoGraphDeploymentRequestSpec) ([]byte, error) {
	return marshalPreservedSpec(*src.DeepCopy(), nil)
}

func restoreDGDRSpokeSpec(raw string) (DynamoGraphDeploymentRequestSpec, bool) {
	return restorePreservedSpec[DynamoGraphDeploymentRequestSpec](raw, nil)
}

func restoreDGDRHubStatus(raw string) (v1beta1.DynamoGraphDeploymentRequestStatus, bool) {
	var status v1beta1.DynamoGraphDeploymentRequestStatus
	if err := json.Unmarshal([]byte(raw), &status); err != nil {
		return v1beta1.DynamoGraphDeploymentRequestStatus{}, false
	}
	return status, true
}

func restoreDGDRSpokeStatus(raw string) (DynamoGraphDeploymentRequestStatus, bool) {
	var status DynamoGraphDeploymentRequestStatus
	if err := json.Unmarshal([]byte(raw), &status); err != nil {
		return DynamoGraphDeploymentRequestStatus{}, false
	}
	return status, true
}

// ConvertFromDynamoGraphDeploymentRequestSpec converts the DGDR spec from
// v1alpha1 to v1beta1.
func ConvertFromDynamoGraphDeploymentRequestSpec(src *DynamoGraphDeploymentRequestSpec, dst *v1beta1.DynamoGraphDeploymentRequestSpec, restored *v1beta1.DynamoGraphDeploymentRequestSpec, save *DynamoGraphDeploymentRequestSpec) error {
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

	if src.ProfilingConfig.Config != nil && src.ProfilingConfig.Config.Raw != nil {
		var blob dgdrProfilingConfigBlob
		if err := json.Unmarshal(src.ProfilingConfig.Config.Raw, &blob); err != nil {
			return fmt.Errorf("failed to parse ProfilingConfig.Config: %w", err)
		}
		projectSLAAndWorkloadFromProfilingConfigBlob(blob, dst)
		projectModelCacheFromProfilingConfigBlob(blob, dst)
		projectPlannerFromProfilingConfigBlob(blob, dst)
		saveDGDRAlphaOnlyProfilingConfig(blob, save)
	}

	// ProfilerImage → Image (the profiler runs in the frontend image)
	// TODO: In a future MR, backend inference images will be managed separately via overrides.dgd.
	if src.ProfilingConfig.ProfilerImage != "" {
		dst.Image = src.ProfilingConfig.ProfilerImage
	}

	projectProfilingConfigToProfilingJob(&src.ProfilingConfig, dst)
	saveDGDRAlphaOnlySpec(src, save)
	restoreDGDRHubOnlySpec(restored, dst)

	return nil
}

func saveDGDRAlphaOnlySpec(src *DynamoGraphDeploymentRequestSpec, save *DynamoGraphDeploymentRequestSpec) {
	if src == nil || save == nil {
		return
	}
	if src.EnableGPUDiscovery != nil {
		v := *src.EnableGPUDiscovery
		save.EnableGPUDiscovery = &v
	}
	if src.ProfilingConfig.ConfigMapRef != nil {
		ref := *src.ProfilingConfig.ConfigMapRef
		save.ProfilingConfig.ConfigMapRef = &ref
	}
	save.ProfilingConfig.OutputPVC = src.ProfilingConfig.OutputPVC
	if src.DeploymentOverrides != nil {
		overrides := &DeploymentOverridesSpec{
			Name:         src.DeploymentOverrides.Name,
			Namespace:    src.DeploymentOverrides.Namespace,
			Labels:       maps.Clone(src.DeploymentOverrides.Labels),
			Annotations:  maps.Clone(src.DeploymentOverrides.Annotations),
			WorkersImage: src.DeploymentOverrides.WorkersImage,
		}
		if !apiequality.Semantic.DeepEqual(*overrides, DeploymentOverridesSpec{}) {
			save.DeploymentOverrides = overrides
		}
	}
}

func saveDGDRAlphaOnlyProfilingConfig(blob dgdrProfilingConfigBlob, save *DynamoGraphDeploymentRequestSpec) {
	if save == nil || blob == nil {
		return
	}
	if len(blob) == 0 {
		save.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: []byte(`{}`)}
		return
	}
	remainder := stripDGDRTypedProfilingConfig(blob)
	if len(remainder) == 0 {
		return
	}
	data, err := json.Marshal(remainder)
	if err != nil {
		return
	}
	save.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: data}
}

func restoreDGDRHubOnlySpec(restored *v1beta1.DynamoGraphDeploymentRequestSpec, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
	if restored == nil || dst == nil {
		return
	}
	if restored.Hardware != nil {
		dst.Hardware = restored.Hardware.DeepCopy()
	}
	if restored.Workload != nil {
		if dst.Workload == nil {
			dst.Workload = &v1beta1.WorkloadSpec{}
		}
		if restored.Workload.Concurrency != nil {
			v := *restored.Workload.Concurrency
			dst.Workload.Concurrency = &v
		}
		if restored.Workload.RequestRate != nil {
			v := *restored.Workload.RequestRate
			dst.Workload.RequestRate = &v
		}
	}
	if restored.SLA != nil {
		if dst.SLA == nil {
			dst.SLA = &v1beta1.SLASpec{}
		}
		if restored.SLA.E2ELatency != nil {
			v := *restored.SLA.E2ELatency
			dst.SLA.E2ELatency = &v
		}
	}
	if restored.Overrides != nil {
		restoreDGDRHubOnlyOverrides(restored.Overrides, dst)
	}
	if restored.Features != nil && restored.Features.Mocker != nil && !restored.Features.Mocker.Enabled {
		if dst.Features == nil {
			dst.Features = &v1beta1.FeaturesSpec{}
		}
		if dst.Features.Mocker == nil {
			dst.Features.Mocker = &v1beta1.MockerSpec{Enabled: false}
		}
	}
	if restored.SearchStrategy != "" {
		dst.SearchStrategy = restored.SearchStrategy
	}
}

// projectSLAAndWorkloadFromProfilingConfigBlob extracts SLA and Workload fields from the v1alpha1 JSON blob.
// Both are nested under blob["sla"] in the v1alpha1 schema.
func projectSLAAndWorkloadFromProfilingConfigBlob(blob map[string]interface{}, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
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

// projectModelCacheFromProfilingConfigBlob extracts ModelCache from blob["deployment"]["modelCache"].
func projectModelCacheFromProfilingConfigBlob(blob map[string]interface{}, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
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

// projectProfilingConfigToProfilingJob maps ProfilingConfig pod-level fields into
// the v1beta1 Overrides.ProfilingJob pod spec.
func projectProfilingConfigToProfilingJob(src *ProfilingConfigSpec, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
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

func saveDGDRLegacySpokeAnnotations(srcSpec *DynamoGraphDeploymentRequestSpec, srcStatus *DynamoGraphDeploymentRequestStatus, dst *v1beta1.DynamoGraphDeploymentRequest) {
	if srcSpec != nil {
		if srcSpec.EnableGPUDiscovery != nil && *srcSpec.EnableGPUDiscovery {
			setAnnotation(dst, legacyAnnDGDREnableGPUDisc, annotationTrue)
		}
		if srcSpec.ProfilingConfig.Config != nil && srcSpec.ProfilingConfig.Config.Raw != nil {
			setAnnotation(dst, legacyAnnDGDRProfilingConfig, string(srcSpec.ProfilingConfig.Config.Raw))
		}
		if srcSpec.ProfilingConfig.ConfigMapRef != nil {
			if data, err := json.Marshal(srcSpec.ProfilingConfig.ConfigMapRef); err == nil {
				setAnnotation(dst, legacyAnnDGDRConfigMapRef, string(data))
			}
		}
		if srcSpec.ProfilingConfig.OutputPVC != "" {
			setAnnotation(dst, legacyAnnDGDROutputPVC, srcSpec.ProfilingConfig.OutputPVC)
		}
		saveDGDRLegacyDeploymentOverridesAnnotation(srcSpec.DeploymentOverrides, dst)
	}
	if srcStatus != nil {
		if srcStatus.Backend != "" {
			setAnnotation(dst, legacyAnnDGDRStatusBackend, srcStatus.Backend)
		}
		if srcStatus.ProfilingResults != "" {
			setAnnotation(dst, legacyAnnDGDRProfilingResults, srcStatus.ProfilingResults)
		}
		if srcStatus.Deployment != nil {
			payload := dgdrDeploymentStatusAnnotation{
				DeploymentStatus: *srcStatus.Deployment.DeepCopy(),
				RequestState:     srcStatus.State,
			}
			if data, err := json.Marshal(payload); err == nil {
				setAnnotation(dst, legacyAnnDGDRDeploymentStatus, string(data))
			}
		}
	}
}

func saveDGDRLegacyDeploymentOverridesAnnotation(src *DeploymentOverridesSpec, dstObj *v1beta1.DynamoGraphDeploymentRequest) {
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
		setAnnotation(dstObj, legacyAnnDGDRDeployOverrides, string(data))
	}
}

// ConvertToDynamoGraphDeploymentRequestSpec converts the DGDR spec from
// v1beta1 to v1alpha1.
func ConvertToDynamoGraphDeploymentRequestSpec(src *v1beta1.DynamoGraphDeploymentRequestSpec, dst *DynamoGraphDeploymentRequestSpec, restored *DynamoGraphDeploymentRequestSpec, save *v1beta1.DynamoGraphDeploymentRequestSpec) {
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

	// Reconstruct the JSON blob from alpha-only saved remainder, then write
	// live v1beta1 structured fields on top. The saved remainder has typed
	// paths stripped, so stale annotations cannot resurrect old live values.
	var blob dgdrProfilingConfigBlob
	if restored != nil && restored.ProfilingConfig.Config != nil && restored.ProfilingConfig.Config.Raw != nil {
		_ = json.Unmarshal(restored.ProfilingConfig.Config.Raw, &blob)
		blob = stripDGDRTypedProfilingConfig(blob)
	}
	if src.SLA != nil || src.Workload != nil {
		next := blob
		if next == nil {
			next = make(dgdrProfilingConfigBlob)
		}
		if projectSLAAndWorkloadToProfilingConfigBlob(src, next) {
			blob = next
		}
	}
	if src.ModelCache != nil {
		next := blob
		if next == nil {
			next = make(dgdrProfilingConfigBlob)
		}
		if projectModelCacheToProfilingConfigBlob(src.ModelCache, next) {
			blob = next
		}
	}
	if src.Features != nil && src.Features.Planner != nil {
		next := blob
		if next == nil {
			next = make(dgdrProfilingConfigBlob)
		}
		if projectPlannerToProfilingConfigBlob(src.Features.Planner, next) {
			blob = next
		}
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

	restoreDGDRAlphaOnlySpec(restored, dst)
	projectProfilingJobToProfilingConfig(src, dst)
	saveDGDRHubOnlySpec(src, save)
}

func restoreDGDRAlphaOnlySpec(restored *DynamoGraphDeploymentRequestSpec, dst *DynamoGraphDeploymentRequestSpec) {
	if restored == nil || dst == nil {
		return
	}
	if restored.EnableGPUDiscovery != nil {
		v := *restored.EnableGPUDiscovery
		dst.EnableGPUDiscovery = &v
	}
	if restored.ProfilingConfig.ConfigMapRef != nil {
		ref := *restored.ProfilingConfig.ConfigMapRef
		dst.ProfilingConfig.ConfigMapRef = &ref
	}
	dst.ProfilingConfig.OutputPVC = restored.ProfilingConfig.OutputPVC
	if restored.DeploymentOverrides != nil {
		dst.DeploymentOverrides = &DeploymentOverridesSpec{
			Name:         restored.DeploymentOverrides.Name,
			Namespace:    restored.DeploymentOverrides.Namespace,
			Labels:       maps.Clone(restored.DeploymentOverrides.Labels),
			Annotations:  maps.Clone(restored.DeploymentOverrides.Annotations),
			WorkersImage: restored.DeploymentOverrides.WorkersImage,
		}
	}
}

func saveDGDRHubOnlySpec(src *v1beta1.DynamoGraphDeploymentRequestSpec, save *v1beta1.DynamoGraphDeploymentRequestSpec) {
	if src == nil || save == nil {
		return
	}
	if src.Hardware != nil {
		save.Hardware = src.Hardware.DeepCopy()
	}
	if src.Workload != nil && (src.Workload.Concurrency != nil || src.Workload.RequestRate != nil) {
		workload := &v1beta1.WorkloadSpec{}
		if src.Workload.Concurrency != nil {
			v := *src.Workload.Concurrency
			workload.Concurrency = &v
		}
		if src.Workload.RequestRate != nil {
			v := *src.Workload.RequestRate
			workload.RequestRate = &v
		}
		save.Workload = workload
	}
	if src.SLA != nil && src.SLA.E2ELatency != nil {
		v := *src.SLA.E2ELatency
		save.SLA = &v1beta1.SLASpec{E2ELatency: &v}
	}
	if src.Overrides != nil {
		saveDGDRHubOnlyOverrides(src.Overrides, save)
	}
	if src.Features != nil && src.Features.Mocker != nil && !src.Features.Mocker.Enabled {
		save.Features = &v1beta1.FeaturesSpec{Mocker: &v1beta1.MockerSpec{Enabled: false}}
	}
	save.SearchStrategy = src.SearchStrategy
}

func stripDGDRTypedProfilingConfig(blob dgdrProfilingConfigBlob) dgdrProfilingConfigBlob {
	if blob == nil {
		return nil
	}
	out := maps.Clone(blob)
	if len(out) == 0 {
		return out
	}

	// The profiling config blob is opaque except for leaves that conversion can
	// project into v1beta1 typed fields. Keep unprojectable values in the sparse
	// remainder so an alpha->hub->alpha round trip does not drop user JSON.
	if slaMap, ok := cloneStringAnyMap(out["sla"]); ok {
		stripped := stripDGDRProjectedSLAKeys(slaMap)
		if stripped && len(slaMap) == 0 {
			delete(out, "sla")
		} else {
			out["sla"] = slaMap
		}
	}
	if deploymentMap, ok := cloneStringAnyMap(out["deployment"]); ok {
		stripped := false
		if modelCacheMap, ok := cloneStringAnyMap(deploymentMap["modelCache"]); ok {
			stripped = stripDGDRProjectedModelCacheKeys(modelCacheMap)
			if stripped && len(modelCacheMap) == 0 {
				delete(deploymentMap, "modelCache")
			} else {
				deploymentMap["modelCache"] = modelCacheMap
			}
		}
		if stripped && len(deploymentMap) == 0 {
			delete(out, "deployment")
		} else {
			out["deployment"] = deploymentMap
		}
	}
	if dgdrPlannerBlobProjects(out["planner"]) {
		delete(out, "planner")
	}
	if len(out) == 0 {
		return dgdrProfilingConfigBlob{}
	}
	return out
}

func stripDGDRProjectedSLAKeys(slaMap map[string]any) bool {
	stripped := false
	for _, key := range []string{"ttft", "itl", "isl", "osl"} {
		if _, ok := slaMap[key].(float64); ok {
			delete(slaMap, key)
			stripped = true
		}
	}
	if v, ok := slaMap["optimizationType"].(string); ok {
		ot := v1beta1.OptimizationType(v)
		if ot == v1beta1.OptimizationTypeLatency || ot == v1beta1.OptimizationTypeThroughput {
			delete(slaMap, "optimizationType")
			stripped = true
		}
	}
	return stripped
}

func stripDGDRProjectedModelCacheKeys(modelCacheMap map[string]any) bool {
	stripped := false
	for _, key := range []string{"pvcName", "modelPathInPvc", "pvcMountPath"} {
		// Empty strings are not reconstructed by projectModelCacheToProfilingConfigBlob,
		// so they must remain in the opaque blob remainder.
		if v, ok := modelCacheMap[key].(string); ok && v != "" {
			delete(modelCacheMap, key)
			stripped = true
		}
	}
	return stripped
}

func dgdrPlannerBlobProjects(v any) bool {
	plannerMap, ok := v.(map[string]any)
	return ok && len(plannerMap) > 0
}

func cloneStringAnyMap(v any) (map[string]any, bool) {
	m, ok := v.(map[string]any)
	if !ok {
		return nil, false
	}
	return maps.Clone(m), true
}

func saveDGDRHubOnlyOverrides(src *v1beta1.OverridesSpec, save *v1beta1.DynamoGraphDeploymentRequestSpec) {
	if src == nil || save == nil {
		return
	}
	var overrides v1beta1.OverridesSpec
	if src.DGD != nil {
		overrides.DGD = src.DGD.DeepCopy()
	}
	if profilingJob := dgdrHubOnlyProfilingJob(src.ProfilingJob); profilingJob != nil {
		overrides.ProfilingJob = profilingJob
	}
	if !apiequality.Semantic.DeepEqual(overrides, v1beta1.OverridesSpec{}) {
		save.Overrides = &overrides
	}
}

func restoreDGDRHubOnlyOverrides(restored *v1beta1.OverridesSpec, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
	if restored == nil || dst == nil {
		return
	}
	if dst.Overrides == nil {
		dst.Overrides = &v1beta1.OverridesSpec{}
	}
	if restored.DGD != nil {
		dst.Overrides.DGD = restored.DGD.DeepCopy()
	}
	if restored.ProfilingJob != nil {
		if dst.Overrides.ProfilingJob == nil {
			dst.Overrides.ProfilingJob = restored.ProfilingJob.DeepCopy()
		} else {
			mergeDGDRHubOnlyProfilingJob(dst.Overrides.ProfilingJob, restored.ProfilingJob)
		}
	}
	if apiequality.Semantic.DeepEqual(*dst.Overrides, v1beta1.OverridesSpec{}) {
		dst.Overrides = nil
	}
}

func dgdrHubOnlyProfilingJob(src *batchv1.JobSpec) *batchv1.JobSpec {
	if src == nil {
		return nil
	}
	save := src.DeepCopy()
	if len(save.Template.Spec.Containers) > 0 {
		save.Template.Spec.Containers[0].Resources = corev1.ResourceRequirements{}
		if len(save.Template.Spec.Containers) == 1 && apiequality.Semantic.DeepEqual(save.Template.Spec.Containers[0], corev1.Container{}) {
			save.Template.Spec.Containers = slices.Delete(save.Template.Spec.Containers, 0, 1)
		}
	}
	save.Template.Spec.Tolerations = nil
	save.Template.Spec.NodeSelector = nil
	if apiequality.Semantic.DeepEqual(*save, batchv1.JobSpec{}) {
		return nil
	}
	return save
}

func mergeDGDRHubOnlyProfilingJob(dst *batchv1.JobSpec, restored *batchv1.JobSpec) {
	if dst == nil || restored == nil {
		return
	}
	resources, hasResources := dgdrFirstContainerResources(dst)
	tolerations := slices.Clone(dst.Template.Spec.Tolerations)
	nodeSelector := maps.Clone(dst.Template.Spec.NodeSelector)

	*dst = *restored.DeepCopy()
	if hasResources {
		if len(dst.Template.Spec.Containers) == 0 {
			dst.Template.Spec.Containers = []corev1.Container{{}}
		}
		dst.Template.Spec.Containers[0].Resources = resources
	}
	if len(tolerations) > 0 {
		dst.Template.Spec.Tolerations = tolerations
	}
	if len(nodeSelector) > 0 {
		dst.Template.Spec.NodeSelector = nodeSelector
	}
}

func dgdrFirstContainerResources(job *batchv1.JobSpec) (corev1.ResourceRequirements, bool) {
	if job == nil || len(job.Template.Spec.Containers) == 0 {
		return corev1.ResourceRequirements{}, false
	}
	res := job.Template.Spec.Containers[0].Resources
	return res, !apiequality.Semantic.DeepEqual(res, corev1.ResourceRequirements{})
}

// projectSLAAndWorkloadToProfilingConfigBlob writes SLA and Workload structured fields back into the JSON blob.
// It overwrites existing typed values for those keys and reports whether it wrote or retained SLA data.
func projectSLAAndWorkloadToProfilingConfigBlob(src *v1beta1.DynamoGraphDeploymentRequestSpec, blob map[string]interface{}) bool {
	slaMap, hadSLA := blob["sla"].(map[string]interface{})
	if slaMap == nil {
		slaMap = make(map[string]interface{})
	}
	wroteSLA := false
	if src.SLA != nil {
		if src.SLA.TTFT != nil {
			slaMap["ttft"] = *src.SLA.TTFT
			wroteSLA = true
		}
		if src.SLA.ITL != nil {
			slaMap["itl"] = *src.SLA.ITL
			wroteSLA = true
		}
		if src.SLA.OptimizationType != nil {
			slaMap["optimizationType"] = string(*src.SLA.OptimizationType)
			wroteSLA = true
		}
	}
	if src.Workload != nil {
		if src.Workload.ISL != nil {
			slaMap["isl"] = float64(*src.Workload.ISL)
			wroteSLA = true
		}
		if src.Workload.OSL != nil {
			slaMap["osl"] = float64(*src.Workload.OSL)
			wroteSLA = true
		}
	}
	if wroteSLA || hadSLA {
		blob["sla"] = slaMap
	}
	return wroteSLA || hadSLA
}

// projectModelCacheToProfilingConfigBlob writes ModelCache structured fields back into blob["deployment"]["modelCache"].
func projectModelCacheToProfilingConfigBlob(mc *v1beta1.ModelCacheSpec, blob map[string]interface{}) bool {
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
		return true
	}
	return false
}

// projectPlannerToProfilingConfigBlob writes the planner RawExtension into blob["planner"].
// The RawExtension is the full PlannerConfig JSON blob (opaque to Go).
func projectPlannerToProfilingConfigBlob(planner *runtime.RawExtension, blob map[string]interface{}) bool {
	if planner == nil || planner.Raw == nil {
		return false
	}
	var plannerMap map[string]interface{}
	if err := json.Unmarshal(planner.Raw, &plannerMap); err != nil || len(plannerMap) == 0 {
		return false
	}
	blob["planner"] = plannerMap
	return true
}

// projectPlannerFromProfilingConfigBlob extracts blob["planner"] and populates v1beta1 Features.Planner.
func projectPlannerFromProfilingConfigBlob(blob map[string]interface{}, dst *v1beta1.DynamoGraphDeploymentRequestSpec) {
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

func restoreDGDRLegacySpokeSpec(obj metav1.Object) *DynamoGraphDeploymentRequestSpec {
	restored := &DynamoGraphDeploymentRequestSpec{}
	if raw, ok := getAnnFromObj(obj, legacyAnnDGDRProfilingConfig); ok && raw != "" {
		var blob dgdrProfilingConfigBlob
		if err := json.Unmarshal([]byte(raw), &blob); err == nil {
			blob = stripDGDRTypedProfilingConfig(blob)
			if blob != nil {
				if data, err := json.Marshal(blob); err == nil {
					restored.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: data}
				}
			}
		}
	}
	if raw, ok := getAnnFromObj(obj, legacyAnnDGDREnableGPUDisc); ok && raw == annotationTrue {
		v := true
		restored.EnableGPUDiscovery = &v
	}
	if v, ok := getAnnFromObj(obj, legacyAnnDGDRConfigMapRef); ok && v != "" {
		var ref ConfigMapKeySelector
		if err := json.Unmarshal([]byte(v), &ref); err == nil {
			restored.ProfilingConfig.ConfigMapRef = &ref
		}
	}
	if v, ok := getAnnFromObj(obj, legacyAnnDGDROutputPVC); ok {
		restored.ProfilingConfig.OutputPVC = v
	}
	if v, ok := getAnnFromObj(obj, legacyAnnDGDRDeployOverrides); ok && v != "" {
		var overrides struct {
			Name        string            `json:"name,omitempty"`
			Namespace   string            `json:"namespace,omitempty"`
			Labels      map[string]string `json:"labels,omitempty"`
			Annotations map[string]string `json:"annotations,omitempty"`
		}
		if err := json.Unmarshal([]byte(v), &overrides); err == nil {
			restored.DeploymentOverrides = &DeploymentOverridesSpec{
				Name:        overrides.Name,
				Namespace:   overrides.Namespace,
				Labels:      overrides.Labels,
				Annotations: overrides.Annotations,
			}
		}
	}
	return restored
}

func restoreDGDRLegacySpokeStatus(obj metav1.Object) *DynamoGraphDeploymentRequestStatus {
	restored := &DynamoGraphDeploymentRequestStatus{}
	if v, ok := getAnnFromObj(obj, legacyAnnDGDRStatusBackend); ok {
		restored.Backend = v
	}
	if v, ok := getAnnFromObj(obj, legacyAnnDGDRProfilingResults); ok {
		restored.ProfilingResults = v
	}
	if raw, ok := getAnnFromObj(obj, legacyAnnDGDRDeploymentStatus); ok && raw != "" {
		deployment, state, ok := restoreDGDRLegacyDeploymentStatus(raw)
		if ok {
			restored.Deployment = &deployment
			restored.State = state
		}
	}
	return restored
}

func restoreDGDRLegacyDeploymentStatus(raw string) (DeploymentStatus, DGDRState, bool) {
	var payload dgdrDeploymentStatusAnnotation
	if err := json.Unmarshal([]byte(raw), &payload); err == nil && payload.Name != "" {
		if payload.RequestState == "" || isValidDGDRRequestState(payload.RequestState) {
			return payload.DeploymentStatus, payload.RequestState, true
		}
		return DeploymentStatus{}, "", false
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &obj); err == nil {
		if _, hasRequestState := obj["requestState"]; hasRequestState {
			return DeploymentStatus{}, "", false
		}
	}
	var legacy DeploymentStatus
	if err := json.Unmarshal([]byte(raw), &legacy); err != nil || legacy.Name == "" {
		return DeploymentStatus{}, "", false
	}
	return legacy, "", true
}

// projectProfilingJobToProfilingConfig maps alpha-representable profiling pod
// fields from v1beta1 Overrides.ProfilingJob back into v1alpha1 ProfilingConfig.
func projectProfilingJobToProfilingConfig(src *v1beta1.DynamoGraphDeploymentRequestSpec, dst *DynamoGraphDeploymentRequestSpec) {
	if src.Overrides == nil || src.Overrides.ProfilingJob == nil {
		return
	}
	podSpec := &src.Overrides.ProfilingJob.Template.Spec
	if len(podSpec.Containers) > 0 {
		res := podSpec.Containers[0].Resources
		if !apiequality.Semantic.DeepEqual(res, corev1.ResourceRequirements{}) {
			dst.ProfilingConfig.Resources = &res
		}
	}
	if len(podSpec.Tolerations) > 0 {
		dst.ProfilingConfig.Tolerations = podSpec.Tolerations
	}
	if len(podSpec.NodeSelector) > 0 {
		dst.ProfilingConfig.NodeSelector = maps.Clone(podSpec.NodeSelector)
	}
}

// ConvertFromDynamoGraphDeploymentRequestStatus converts the DGDR status from
// v1alpha1 to v1beta1.
func ConvertFromDynamoGraphDeploymentRequestStatus(src *DynamoGraphDeploymentRequestStatus, dst *v1beta1.DynamoGraphDeploymentRequestStatus, restored *v1beta1.DynamoGraphDeploymentRequestStatus, save *DynamoGraphDeploymentRequestStatus) {
	dst.Phase = dgdrStateToPhase(string(src.State), src.Deployment)
	dst.ObservedGeneration = src.ObservedGeneration
	dst.Conditions = slices.Clone(src.Conditions)
	if src.GeneratedDeployment != nil {
		if dst.ProfilingResults == nil {
			dst.ProfilingResults = &v1beta1.ProfilingResultsStatus{}
		}
		dst.ProfilingResults.SelectedConfig = src.GeneratedDeployment.DeepCopy()
	}
	if src.Deployment != nil {
		dst.DGDName = src.Deployment.Name
	}

	saveDGDRAlphaOnlyStatus(src, save)
	restoreDGDRHubOnlyStatus(restored, dst, src)
}

// ConvertToDynamoGraphDeploymentRequestStatus converts the DGDR status from
// v1beta1 to v1alpha1.
func ConvertToDynamoGraphDeploymentRequestStatus(src *v1beta1.DynamoGraphDeploymentRequestStatus, dst *DynamoGraphDeploymentRequestStatus, restored *DynamoGraphDeploymentRequestStatus, save *v1beta1.DynamoGraphDeploymentRequestStatus) {
	dst.State = DGDRState(dgdrPhaseToState(src.Phase))
	dst.ObservedGeneration = src.ObservedGeneration
	dst.Conditions = slices.Clone(src.Conditions)

	if src.ProfilingResults != nil && src.ProfilingResults.SelectedConfig != nil {
		dst.GeneratedDeployment = src.ProfilingResults.SelectedConfig.DeepCopy()
	}

	// If no annotation but we have DGDName, create a minimal deployment status.
	// Created is left false so the v1alpha1 controller does not skip re-creating the DGD.
	if dst.Deployment == nil && src.DGDName != "" {
		dst.Deployment = &DeploymentStatus{
			Name:    src.DGDName,
			Created: false,
		}
	}
	restoreDGDRAlphaOnlyStatus(restored, dst, src)
	saveDGDRHubOnlyStatus(src, dst, save)
}

func saveDGDRAlphaOnlyStatus(src *DynamoGraphDeploymentRequestStatus, save *DynamoGraphDeploymentRequestStatus) {
	if src == nil || save == nil {
		return
	}
	save.Backend = src.Backend
	save.ProfilingResults = src.ProfilingResults
	if !dgdrAlphaStateHasHubPhase(src.State) {
		save.State = src.State
	}
	if dgdrAlphaDeploymentNeedsSave(src.State, src.Deployment) {
		save.State = src.State
		save.Deployment = src.Deployment.DeepCopy()
	}
}

func dgdrAlphaDeploymentNeedsSave(state DGDRState, deployment *DeploymentStatus) bool {
	if deployment == nil {
		return false
	}
	if deployment.Namespace != "" || deployment.State != "" || deployment.Created {
		return true
	}
	return !dgdrAlphaStateHasHubPhase(state)
}

func restoreDGDRAlphaOnlyStatus(restored *DynamoGraphDeploymentRequestStatus, dst *DynamoGraphDeploymentRequestStatus, src *v1beta1.DynamoGraphDeploymentRequestStatus) {
	if restored == nil || dst == nil {
		return
	}
	dst.Backend = restored.Backend
	dst.ProfilingResults = restored.ProfilingResults
	if restored.State != "" &&
		dgdrAlphaStatusMatchesHubPhase(restored.State, restored.Deployment, src.Phase) &&
		dgdrAlphaRestoredStateMatchesLiveDGD(restored.State, restored.Deployment, src.DGDName) {
		dst.State = restored.State
	}
	if restored.Deployment != nil &&
		restored.Deployment.Name == src.DGDName &&
		(restored.State == "" || dgdrAlphaStatusMatchesHubPhase(restored.State, restored.Deployment, src.Phase)) {
		dst.Deployment = restored.Deployment.DeepCopy()
	}
}

func saveDGDRHubOnlyStatus(src *v1beta1.DynamoGraphDeploymentRequestStatus, dst *DynamoGraphDeploymentRequestStatus, save *v1beta1.DynamoGraphDeploymentRequestStatus) {
	if src == nil || save == nil {
		return
	}
	if src.Phase == v1beta1.DGDRPhaseDeployed && dgdrStateToPhase(string(dst.State), dst.Deployment) != src.Phase {
		save.Phase = src.Phase
		save.DGDName = src.DGDName
	}
	if src.Phase == v1beta1.DGDRPhaseProfiling {
		save.ProfilingPhase = src.ProfilingPhase
		save.ProfilingJobName = src.ProfilingJobName
	}
	if src.ProfilingResults != nil && len(src.ProfilingResults.Pareto) > 0 {
		save.ProfilingResults = src.ProfilingResults.DeepCopy()
		save.ProfilingResults.SelectedConfig = nil
	}
	if src.DeploymentInfo != nil {
		save.DGDName = src.DGDName
		save.DeploymentInfo = src.DeploymentInfo.DeepCopy()
	}
}

func restoreDGDRHubOnlyStatus(restored *v1beta1.DynamoGraphDeploymentRequestStatus, dst *v1beta1.DynamoGraphDeploymentRequestStatus, src *DynamoGraphDeploymentRequestStatus) {
	if restored == nil || dst == nil {
		return
	}
	if restored.Phase == v1beta1.DGDRPhaseDeployed &&
		dst.Phase == v1beta1.DGDRPhaseReady &&
		src != nil &&
		dgdrAlphaStatusMatchesHubPhase(src.State, src.Deployment, restored.Phase) &&
		restored.DGDName == dst.DGDName {
		dst.Phase = v1beta1.DGDRPhaseDeployed
	}
	if dst.Phase == v1beta1.DGDRPhaseProfiling {
		dst.ProfilingPhase = restored.ProfilingPhase
		dst.ProfilingJobName = restored.ProfilingJobName
	}
	if restored.ProfilingResults != nil && len(restored.ProfilingResults.Pareto) > 0 {
		if dst.ProfilingResults == nil {
			dst.ProfilingResults = &v1beta1.ProfilingResultsStatus{}
		}
		dst.ProfilingResults.Pareto = restored.ProfilingResults.DeepCopy().Pareto
	}
	if restored.DeploymentInfo != nil && restored.DGDName == dst.DGDName {
		dst.DeploymentInfo = restored.DeploymentInfo.DeepCopy()
	}
}

func dgdrAlphaStateHasHubPhase(state DGDRState) bool {
	return state == "" ||
		state == DGDRStatePending ||
		state == DGDRStateProfiling ||
		state == DGDRStateReady ||
		state == DGDRStateDeploying ||
		state == DGDRStateFailed
}

func dgdrAlphaStatusMatchesHubPhase(state DGDRState, deployment *DeploymentStatus, phase v1beta1.DGDRPhase) bool {
	alphaPhase := dgdrStateToPhase(string(state), deployment)
	if alphaPhase == phase {
		return true
	}
	return phase == v1beta1.DGDRPhaseDeployed && state == DGDRStateReady
}

func dgdrAlphaRestoredStateMatchesLiveDGD(state DGDRState, deployment *DeploymentStatus, dgdName string) bool {
	if state != DGDRStateDeploymentDeleted {
		return true
	}

	// DeploymentDeleted is an alpha-only state for the specific DGD recorded in status.deployment.
	// If the live hub status now points at another DGD, the preserved state is stale.
	if deployment == nil {
		return dgdName == ""
	}
	return deployment.Name == dgdName
}

// restoreDGDRDeploymentStatus ignores stale deployment-status overlays after hub-side status edits.
func restoreDGDRDeploymentStatus(raw string, src *v1beta1.DynamoGraphDeploymentRequestStatus) (DeploymentStatus, DGDRState, bool) {
	var payload dgdrDeploymentStatusAnnotation
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		return DeploymentStatus{}, "", false
	}
	if !isValidDGDRRequestState(payload.RequestState) {
		return DeploymentStatus{}, "", false
	}
	if payload.Name != src.DGDName {
		return DeploymentStatus{}, "", false
	}
	if !dgdrAlphaStatusMatchesHubPhase(payload.RequestState, &payload.DeploymentStatus, src.Phase) {
		return DeploymentStatus{}, "", false
	}
	return payload.DeploymentStatus, payload.RequestState, true
}

func isValidDGDRRequestState(state DGDRState) bool {
	switch state {
	case DGDRStateInitializing,
		DGDRStatePending,
		DGDRStateProfiling,
		DGDRStateDeploying,
		DGDRStateReady,
		DGDRStateDeploymentDeleted,
		DGDRStateFailed:
		return true
	default:
		return false
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
