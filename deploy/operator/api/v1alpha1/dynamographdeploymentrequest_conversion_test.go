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
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
)

// newV1alpha1DGDR builds a fully-populated v1alpha1 DGDR for use in tests.
func newV1alpha1DGDR() *DynamoGraphDeploymentRequest {
	profilingBlob := map[string]interface{}{
		"sla": map[string]interface{}{
			"ttft": float64(500),
			"itl":  float64(20),
			"isl":  float64(2048),
			"osl":  float64(512),
		},
		"deployment": map[string]interface{}{
			"modelCache": map[string]interface{}{
				"pvcName":        "model-pvc",
				"modelPathInPvc": "llama-3",
				"pvcMountPath":   "/data/model",
			},
		},
		"extra_key": "preserved",
	}
	blobRaw, _ := json.Marshal(profilingBlob)

	trueVal := true
	return &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgdr",
			Namespace: "default",
		},
		Spec: DynamoGraphDeploymentRequestSpec{
			Model:     "meta-llama/Llama-3.1-8B",
			Backend:   "vllm",
			AutoApply: true,
			UseMocker: true,
			ProfilingConfig: ProfilingConfigSpec{
				ProfilerImage: "nvcr.io/nvidia/dynamo:latest",
				OutputPVC:     "output-pvc",
				Config:        &apiextensionsv1.JSON{Raw: blobRaw},
				ConfigMapRef:  &ConfigMapKeySelector{Name: "base-config", Key: "disagg.yaml"},
			},
			EnableGPUDiscovery: &trueVal,
			DeploymentOverrides: &DeploymentOverridesSpec{
				Name:      "my-dgd",
				Namespace: "prod",
				Labels:    map[string]string{"team": "ml"},
			},
		},
		Status: DynamoGraphDeploymentRequestStatus{
			State:              DGDRStateProfiling,
			Backend:            "vllm",
			ObservedGeneration: 3,
			ProfilingResults:   "configmap/profiling-cm",
			Deployment: &DeploymentStatus{
				Name:      "my-dgd",
				Namespace: "prod",
				State:     "initializing",
				Created:   true,
			},
		},
	}
}

// newV1beta1DGDR builds a fully-populated v1beta1 DGDR for use in tests.
func newV1beta1DGDR() *v1beta1.DynamoGraphDeploymentRequest {
	ttft := float64(300)
	itl := float64(15)
	isl := int32(1024)
	osl := int32(256)

	rawDGD, _ := json.Marshal(map[string]interface{}{"apiVersion": "nvidia.com/v1alpha1", "kind": "DynamoGraphDeployment"})

	return &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "hub-dgdr",
			Namespace: "default",
		},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:     "Qwen/Qwen3-32B",
			Backend:   v1beta1.BackendTypeVllm,
			AutoApply: false,
			Image:     "nvcr.io/nvidia/dynamo:0.3.2",
			SLA: &v1beta1.SLASpec{
				TTFT: &ttft,
				ITL:  &itl,
			},
			Workload: &v1beta1.WorkloadSpec{
				ISL: &isl,
				OSL: &osl,
			},
			ModelCache: &v1beta1.ModelCacheSpec{
				PVCName:      "qwen-pvc",
				PVCModelPath: "qwen3-32b",
				PVCMountPath: "/models",
			},
			Features: &v1beta1.FeaturesSpec{
				Mocker: &v1beta1.MockerSpec{Enabled: true},
			},
		},
		Status: v1beta1.DynamoGraphDeploymentRequestStatus{
			Phase:              v1beta1.DGDRPhaseDeployed,
			ObservedGeneration: 2,
			DGDName:            "hub-dgd",
			ProfilingJobName:   "profiling-job-1",
			ProfilingResults: &v1beta1.ProfilingResultsStatus{
				SelectedConfig: &runtime.RawExtension{Raw: rawDGD},
			},
		},
	}
}

// TestConvertTo_SpecFields verifies that key v1alpha1 spec fields land in the correct v1beta1 locations.
func TestConvertTo_SpecFields(t *testing.T) {
	src := newV1alpha1DGDR()
	dst := &v1beta1.DynamoGraphDeploymentRequest{}

	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	// Simple 1:1 fields
	if dst.Spec.Model != src.Spec.Model {
		t.Errorf("Model: got %q, want %q", dst.Spec.Model, src.Spec.Model)
	}
	if string(dst.Spec.Backend) != src.Spec.Backend {
		t.Errorf("Backend: got %q, want %q", dst.Spec.Backend, src.Spec.Backend)
	}
	if dst.Spec.AutoApply != src.Spec.AutoApply {
		t.Errorf("AutoApply: got %v, want %v", dst.Spec.AutoApply, src.Spec.AutoApply)
	}

	// ProfilerImage → Image
	if dst.Spec.Image != src.Spec.ProfilingConfig.ProfilerImage {
		t.Errorf("Image: got %q, want %q", dst.Spec.Image, src.Spec.ProfilingConfig.ProfilerImage)
	}

	// UseMocker → Features.Mocker.Enabled
	if dst.Spec.Features == nil || dst.Spec.Features.Mocker == nil {
		t.Fatal("Features.Mocker is nil")
	}
	if !dst.Spec.Features.Mocker.Enabled {
		t.Error("Features.Mocker.Enabled: got false, want true")
	}

	// SLA from JSON blob
	if dst.Spec.SLA == nil {
		t.Fatal("SLA is nil")
	}
	if dst.Spec.SLA.TTFT == nil || *dst.Spec.SLA.TTFT != 500 {
		t.Errorf("SLA.TTFT: got %v, want 500", dst.Spec.SLA.TTFT)
	}
	if dst.Spec.SLA.ITL == nil || *dst.Spec.SLA.ITL != 20 {
		t.Errorf("SLA.ITL: got %v, want 20", dst.Spec.SLA.ITL)
	}

	// Workload from JSON blob
	if dst.Spec.Workload == nil {
		t.Fatal("Workload is nil")
	}
	if dst.Spec.Workload.ISL == nil || *dst.Spec.Workload.ISL != 2048 {
		t.Errorf("Workload.ISL: got %v, want 2048", dst.Spec.Workload.ISL)
	}
	if dst.Spec.Workload.OSL == nil || *dst.Spec.Workload.OSL != 512 {
		t.Errorf("Workload.OSL: got %v, want 512", dst.Spec.Workload.OSL)
	}

	// ModelCache from JSON blob
	if dst.Spec.ModelCache == nil {
		t.Fatal("ModelCache is nil")
	}
	if dst.Spec.ModelCache.PVCName != "model-pvc" {
		t.Errorf("ModelCache.PVCName: got %q, want %q", dst.Spec.ModelCache.PVCName, "model-pvc")
	}
	if dst.Spec.ModelCache.PVCModelPath != "llama-3" {
		t.Errorf("ModelCache.PVCModelPath: got %q, want %q", dst.Spec.ModelCache.PVCModelPath, "llama-3")
	}
	if dst.Spec.ModelCache.PVCMountPath != "/data/model" {
		t.Errorf("ModelCache.PVCMountPath: got %q, want %q", dst.Spec.ModelCache.PVCMountPath, "/data/model")
	}

	// EnableGPUDiscovery → annotation
	if dst.Annotations[annDGDREnableGPUDisc] != "true" {
		t.Errorf("annDGDREnableGPUDisc annotation: got %q, want %q", dst.Annotations[annDGDREnableGPUDisc], "true")
	}

	// OutputPVC → annotation
	if dst.Annotations[annDGDROutputPVC] != "output-pvc" {
		t.Errorf("annDGDROutputPVC annotation: got %q, want %q", dst.Annotations[annDGDROutputPVC], "output-pvc")
	}

	// DeploymentOverrides → annotation
	if dst.Annotations[annDGDRDeployOverrides] == "" {
		t.Error("annDGDRDeployOverrides annotation is empty")
	}
}

// TestConvertTo_StatusFields verifies that key v1alpha1 status fields land in the correct v1beta1 locations.
func TestConvertTo_StatusFields(t *testing.T) {
	src := newV1alpha1DGDR()
	dst := &v1beta1.DynamoGraphDeploymentRequest{}

	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	// Profiling state → Profiling phase
	if dst.Status.Phase != v1beta1.DGDRPhaseProfiling {
		t.Errorf("Status.Phase: got %q, want %q", dst.Status.Phase, v1beta1.DGDRPhaseProfiling)
	}
	if dst.Status.ObservedGeneration != 3 {
		t.Errorf("Status.ObservedGeneration: got %d, want 3", dst.Status.ObservedGeneration)
	}

	// Deployment.Name → DGDName
	if dst.Status.DGDName != "my-dgd" {
		t.Errorf("Status.DGDName: got %q, want %q", dst.Status.DGDName, "my-dgd")
	}

	// Backend → annotation
	if dst.Annotations[annDGDRStatusBackend] != "vllm" {
		t.Errorf("annDGDRStatusBackend annotation: got %q, want %q", dst.Annotations[annDGDRStatusBackend], "vllm")
	}

	// ProfilingResults → annotation
	if dst.Annotations[annDGDRProfilingResults] != "configmap/profiling-cm" {
		t.Errorf("annDGDRProfilingResults annotation: got %q, want %q", dst.Annotations[annDGDRProfilingResults], "configmap/profiling-cm")
	}
}

// TestAlpha1RoundTrip verifies v1alpha1 → v1beta1 → v1alpha1 preserves all round-tripped fields.
func TestAlpha1RoundTrip(t *testing.T) {
	original := newV1alpha1DGDR()

	// Step 1: v1alpha1 → v1beta1
	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := original.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	// Step 2: v1beta1 → v1alpha1
	restored := &DynamoGraphDeploymentRequest{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	// --- Spec checks ---
	if restored.Spec.Model != original.Spec.Model {
		t.Errorf("Spec.Model: got %q, want %q", restored.Spec.Model, original.Spec.Model)
	}
	if restored.Spec.Backend != original.Spec.Backend {
		t.Errorf("Spec.Backend: got %q, want %q", restored.Spec.Backend, original.Spec.Backend)
	}
	if restored.Spec.AutoApply != original.Spec.AutoApply {
		t.Errorf("Spec.AutoApply: got %v, want %v", restored.Spec.AutoApply, original.Spec.AutoApply)
	}
	if restored.Spec.UseMocker != original.Spec.UseMocker {
		t.Errorf("Spec.UseMocker: got %v, want %v", restored.Spec.UseMocker, original.Spec.UseMocker)
	}
	if restored.Spec.ProfilingConfig.ProfilerImage != original.Spec.ProfilingConfig.ProfilerImage {
		t.Errorf("ProfilingConfig.ProfilerImage: got %q, want %q", restored.Spec.ProfilingConfig.ProfilerImage, original.Spec.ProfilingConfig.ProfilerImage)
	}
	if restored.Spec.ProfilingConfig.OutputPVC != original.Spec.ProfilingConfig.OutputPVC {
		t.Errorf("ProfilingConfig.OutputPVC: got %q, want %q", restored.Spec.ProfilingConfig.OutputPVC, original.Spec.ProfilingConfig.OutputPVC)
	}

	// ConfigMapRef round-trip
	if restored.Spec.ProfilingConfig.ConfigMapRef == nil {
		t.Fatal("ProfilingConfig.ConfigMapRef is nil after round-trip")
	}
	if restored.Spec.ProfilingConfig.ConfigMapRef.Name != original.Spec.ProfilingConfig.ConfigMapRef.Name {
		t.Errorf("ConfigMapRef.Name: got %q, want %q", restored.Spec.ProfilingConfig.ConfigMapRef.Name, original.Spec.ProfilingConfig.ConfigMapRef.Name)
	}

	// EnableGPUDiscovery round-trip
	if restored.Spec.EnableGPUDiscovery == nil || !*restored.Spec.EnableGPUDiscovery {
		t.Error("Spec.EnableGPUDiscovery: expected true after round-trip")
	}

	// DeploymentOverrides round-trip
	if restored.Spec.DeploymentOverrides == nil {
		t.Fatal("Spec.DeploymentOverrides is nil after round-trip")
	}
	if restored.Spec.DeploymentOverrides.Name != original.Spec.DeploymentOverrides.Name {
		t.Errorf("DeploymentOverrides.Name: got %q, want %q", restored.Spec.DeploymentOverrides.Name, original.Spec.DeploymentOverrides.Name)
	}

	// JSON blob round-trip: SLA fields re-emerge in ProfilingConfig.Config
	if restored.Spec.ProfilingConfig.Config == nil {
		t.Fatal("ProfilingConfig.Config is nil after round-trip")
	}
	var blob map[string]interface{}
	if err := json.Unmarshal(restored.Spec.ProfilingConfig.Config.Raw, &blob); err != nil {
		t.Fatalf("failed to unmarshal restored ProfilingConfig.Config: %v", err)
	}
	slaMap, _ := blob["sla"].(map[string]interface{})
	if slaMap == nil {
		t.Fatal("sla key missing in restored JSON blob")
	}
	if slaMap["ttft"] != float64(500) {
		t.Errorf("blob sla.ttft: got %v, want 500", slaMap["ttft"])
	}
	if slaMap["isl"] != float64(2048) {
		t.Errorf("blob sla.isl: got %v, want 2048", slaMap["isl"])
	}
	// Verify unknown keys are preserved via the annotation round-trip
	if blob["extra_key"] != "preserved" {
		t.Errorf("extra_key: got %v, want %q", blob["extra_key"], "preserved")
	}

	// --- Status checks ---
	if restored.Status.State != original.Status.State {
		t.Errorf("Status.State: got %q, want %q", restored.Status.State, original.Status.State)
	}
	if restored.Status.ObservedGeneration != original.Status.ObservedGeneration {
		t.Errorf("Status.ObservedGeneration: got %d, want %d", restored.Status.ObservedGeneration, original.Status.ObservedGeneration)
	}
	if restored.Status.Backend != original.Status.Backend {
		t.Errorf("Status.Backend: got %q, want %q", restored.Status.Backend, original.Status.Backend)
	}
	if restored.Status.ProfilingResults != original.Status.ProfilingResults {
		t.Errorf("Status.ProfilingResults: got %q, want %q", restored.Status.ProfilingResults, original.Status.ProfilingResults)
	}
	if restored.Status.Deployment == nil {
		t.Fatal("Status.Deployment is nil after round-trip")
	}
	if restored.Status.Deployment.Name != original.Status.Deployment.Name {
		t.Errorf("Status.Deployment.Name: got %q, want %q", restored.Status.Deployment.Name, original.Status.Deployment.Name)
	}
	if restored.Status.Deployment.Created != original.Status.Deployment.Created {
		t.Errorf("Status.Deployment.Created: got %v, want %v", restored.Status.Deployment.Created, original.Status.Deployment.Created)
	}
}

// TestHubRoundTrip verifies v1beta1 → v1alpha1 → v1beta1 preserves all round-tripped fields.
func TestHubRoundTrip(t *testing.T) {
	original := newV1beta1DGDR()

	// Step 1: v1beta1 → v1alpha1
	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	// Step 2: v1alpha1 → v1beta1
	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	// --- Spec checks ---
	if restored.Spec.Model != original.Spec.Model {
		t.Errorf("Spec.Model: got %q, want %q", restored.Spec.Model, original.Spec.Model)
	}
	if restored.Spec.Backend != original.Spec.Backend {
		t.Errorf("Spec.Backend: got %q, want %q", restored.Spec.Backend, original.Spec.Backend)
	}
	if restored.Spec.AutoApply != original.Spec.AutoApply {
		t.Errorf("Spec.AutoApply: got %v, want %v", restored.Spec.AutoApply, original.Spec.AutoApply)
	}
	if restored.Spec.Image != original.Spec.Image {
		t.Errorf("Spec.Image: got %q, want %q", restored.Spec.Image, original.Spec.Image)
	}

	// UseMocker round-trip via Features.Mocker.Enabled
	if restored.Spec.Features == nil || restored.Spec.Features.Mocker == nil {
		t.Fatal("Spec.Features.Mocker is nil after round-trip")
	}
	if restored.Spec.Features.Mocker.Enabled != original.Spec.Features.Mocker.Enabled {
		t.Errorf("Features.Mocker.Enabled: got %v, want %v", restored.Spec.Features.Mocker.Enabled, original.Spec.Features.Mocker.Enabled)
	}

	// SLA round-trip via JSON blob
	if restored.Spec.SLA == nil {
		t.Fatal("Spec.SLA is nil after round-trip")
	}
	if restored.Spec.SLA.TTFT == nil || *restored.Spec.SLA.TTFT != *original.Spec.SLA.TTFT {
		t.Errorf("SLA.TTFT: got %v, want %v", restored.Spec.SLA.TTFT, original.Spec.SLA.TTFT)
	}
	if restored.Spec.SLA.ITL == nil || *restored.Spec.SLA.ITL != *original.Spec.SLA.ITL {
		t.Errorf("SLA.ITL: got %v, want %v", restored.Spec.SLA.ITL, original.Spec.SLA.ITL)
	}

	// Workload round-trip via JSON blob
	if restored.Spec.Workload == nil {
		t.Fatal("Spec.Workload is nil after round-trip")
	}
	if restored.Spec.Workload.ISL == nil || *restored.Spec.Workload.ISL != *original.Spec.Workload.ISL {
		t.Errorf("Workload.ISL: got %v, want %v", restored.Spec.Workload.ISL, original.Spec.Workload.ISL)
	}
	if restored.Spec.Workload.OSL == nil || *restored.Spec.Workload.OSL != *original.Spec.Workload.OSL {
		t.Errorf("Workload.OSL: got %v, want %v", restored.Spec.Workload.OSL, original.Spec.Workload.OSL)
	}

	// ModelCache round-trip via JSON blob
	if restored.Spec.ModelCache == nil {
		t.Fatal("Spec.ModelCache is nil after round-trip")
	}
	if restored.Spec.ModelCache.PVCName != original.Spec.ModelCache.PVCName {
		t.Errorf("ModelCache.PVCName: got %q, want %q", restored.Spec.ModelCache.PVCName, original.Spec.ModelCache.PVCName)
	}
	if restored.Spec.ModelCache.PVCModelPath != original.Spec.ModelCache.PVCModelPath {
		t.Errorf("ModelCache.PVCModelPath: got %q, want %q", restored.Spec.ModelCache.PVCModelPath, original.Spec.ModelCache.PVCModelPath)
	}
	if restored.Spec.ModelCache.PVCMountPath != original.Spec.ModelCache.PVCMountPath {
		t.Errorf("ModelCache.PVCMountPath: got %q, want %q", restored.Spec.ModelCache.PVCMountPath, original.Spec.ModelCache.PVCMountPath)
	}

	// --- Status checks ---
	// Deployed → Ready (lossy: v1alpha1 has no "Deployed" state; maps to "Ready")
	// then on the way back Ready→Ready
	if restored.Status.Phase != v1beta1.DGDRPhaseReady {
		t.Errorf("Status.Phase: got %q, want %q (Deployed→Ready is lossy)", restored.Status.Phase, v1beta1.DGDRPhaseReady)
	}
	if restored.Status.ObservedGeneration != original.Status.ObservedGeneration {
		t.Errorf("Status.ObservedGeneration: got %d, want %d", restored.Status.ObservedGeneration, original.Status.ObservedGeneration)
	}
	// DGDName round-trip
	if restored.Status.DGDName != original.Status.DGDName {
		t.Errorf("Status.DGDName: got %q, want %q", restored.Status.DGDName, original.Status.DGDName)
	}
	// ProfilingJobName round-trip via annotation
	if restored.Status.ProfilingJobName != original.Status.ProfilingJobName {
		t.Errorf("Status.ProfilingJobName: got %q, want %q", restored.Status.ProfilingJobName, original.Status.ProfilingJobName)
	}
	// GeneratedDeployment round-trip via ProfilingResults.SelectedConfig
	if restored.Status.ProfilingResults == nil || restored.Status.ProfilingResults.SelectedConfig == nil {
		t.Fatal("Status.ProfilingResults.SelectedConfig is nil after round-trip")
	}
}

// TestConvertTo_InvalidProfilingConfigJSON verifies that malformed JSON in ProfilingConfig.Config
// returns an error rather than silently producing an incomplete conversion.
func TestConvertTo_InvalidProfilingConfigJSON(t *testing.T) {
	src := newV1alpha1DGDR()
	src.Spec.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: []byte(`{not valid json`)}

	dst := &v1beta1.DynamoGraphDeploymentRequest{}
	err := src.ConvertTo(dst)
	if err == nil {
		t.Fatal("ConvertTo() expected error for invalid JSON, got nil")
	}
}
