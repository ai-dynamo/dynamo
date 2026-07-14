/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Read-only compatibility for objects written by Dynamo 1.0/1.1. Do not emit
// these keys; current preservation uses annDGDRSpec and annDGDRStatus.
const (
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

type dgdrDeploymentStatusAnnotation struct {
	DeploymentStatus
	RequestState DGDRState `json:"requestState,omitempty"`
}

func restoreDGDRLegacyHubStatus(obj metav1.Object, structural *v1beta1.DynamoGraphDeploymentRequestStatus) *v1beta1.DynamoGraphDeploymentRequestStatus {
	if structural != nil {
		return structural
	}
	raw, ok := getAnnFromObj(obj, legacyAnnDGDRProfilingJobName)
	if !ok || raw == "" {
		return nil
	}
	return &v1beta1.DynamoGraphDeploymentRequestStatus{ProfilingJobName: raw}
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

func scrubDGDRLegacyAnnotations(obj metav1.Object) {
	for _, key := range legacyDGDRAnnotationKeys() {
		delAnnFromObj(obj, key)
	}
}

func legacyDGDRAnnotationKeys() []string {
	return []string{
		legacyAnnDGDRConfigMapRef,
		legacyAnnDGDROutputPVC,
		legacyAnnDGDREnableGPUDisc,
		legacyAnnDGDRDeployOverrides,
		legacyAnnDGDRProfilingConfig,
		legacyAnnDGDRStatusBackend,
		legacyAnnDGDRProfilingResults,
		legacyAnnDGDRDeploymentStatus,
		legacyAnnDGDRProfilingJobName,
	}
}
