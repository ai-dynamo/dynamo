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

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
)

const (
	dgdrHubSaveRepresentedFieldsMsg = "hub save included represented fields: %#v"
	dgdrAliasLabelKey               = "source"
	dgdrAliasBefore                 = "before"
)

func decodeDGDRHubSpecSaveForTest(t *testing.T, raw string) dgdrHubSpecSave {
	t.Helper()
	var saved dgdrHubSpecSave
	if err := json.Unmarshal([]byte(raw), &saved); err != nil {
		t.Fatalf("unmarshal %s: %v", annDGDRHubSpec, err)
	}
	return saved
}

// newV1alpha1DGDR builds a fully-populated v1alpha1 DGDR for use in tests.
func newV1alpha1DGDR() *DynamoGraphDeploymentRequest {
	profilingBlob := map[string]interface{}{
		"sla": map[string]interface{}{
			"ttft":             float64(500),
			"itl":              float64(20),
			"isl":              float64(2048),
			"osl":              float64(512),
			"optimizationType": "latency",
		},
		"deployment": map[string]interface{}{
			"modelCache": map[string]interface{}{
				"pvcName":        testModelPVCName,
				"modelPathInPvc": "llama-3",
				"pvcMountPath":   "/data/model",
			},
		},
		"planner": map[string]interface{}{
			"enable_load_scaling": false,
		},
		"extra_key": "saved",
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
	rawPlanner, _ := json.Marshal(map[string]interface{}{"enable_load_scaling": false})
	autoApplyFalse := false

	return &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "hub-dgdr",
			Namespace: "default",
		},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:     "Qwen/Qwen3-32B",
			Backend:   v1beta1.BackendTypeVllm,
			AutoApply: &autoApplyFalse,
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
				Mocker:  &v1beta1.MockerSpec{Enabled: true},
				Planner: &runtime.RawExtension{Raw: rawPlanner},
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
	if dst.Spec.AutoApply == nil || *dst.Spec.AutoApply != src.Spec.AutoApply {
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
	if dst.Spec.SLA.OptimizationType == nil || *dst.Spec.SLA.OptimizationType != v1beta1.OptimizationTypeLatency {
		t.Errorf("SLA.OptimizationType: got %v, want %q", dst.Spec.SLA.OptimizationType, v1beta1.OptimizationTypeLatency)
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
	if dst.Spec.ModelCache.PVCName != testModelPVCName {
		t.Errorf("ModelCache.PVCName: got %q, want %q", dst.Spec.ModelCache.PVCName, testModelPVCName)
	}
	if dst.Spec.ModelCache.PVCModelPath != "llama-3" {
		t.Errorf("ModelCache.PVCModelPath: got %q, want %q", dst.Spec.ModelCache.PVCModelPath, "llama-3")
	}
	if dst.Spec.ModelCache.PVCMountPath != "/data/model" {
		t.Errorf("ModelCache.PVCMountPath: got %q, want %q", dst.Spec.ModelCache.PVCMountPath, "/data/model")
	}

	// EnableGPUDiscovery → annotation
	if dst.Annotations[annDGDREnableGPUDisc] != annotationTrue {
		t.Errorf("annDGDREnableGPUDisc annotation: got %q, want %q", dst.Annotations[annDGDREnableGPUDisc], annotationTrue)
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

// TestAlpha1RoundTrip verifies v1alpha1 → v1beta1 → v1alpha1 keeps all round-tripped fields.
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
	// ProfilingConfig.Config (raw JSON blob) is verified separately below.
	if diff := cmp.Diff(original.Spec, restored.Spec, cmpopts.IgnoreFields(ProfilingConfigSpec{}, "Config")); diff != "" {
		t.Errorf("Spec mismatch after round-trip (-want +got):\n%s", diff)
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
	// Verify unknown keys are saved via the annotation round-trip.
	if blob["extra_key"] != "saved" {
		t.Errorf("extra_key: got %v, want %q", blob["extra_key"], "saved")
	}
	// Planner round-trip via applyPlannerFromBlob / mergePlannerIntoBlob
	plannerMap, _ := blob["planner"].(map[string]interface{})
	if plannerMap == nil {
		t.Fatal("planner key missing in restored JSON blob")
	}
	if plannerMap["enable_load_scaling"] != false {
		t.Errorf("planner.enable_load_scaling: got %v, want false", plannerMap["enable_load_scaling"])
	}

	// --- Status checks ---
	if diff := cmp.Diff(original.Status, restored.Status); diff != "" {
		t.Errorf("Status mismatch after round-trip (-want +got):\n%s", diff)
	}
}

// TestHubRoundTrip verifies v1beta1 → v1alpha1 → v1beta1 keeps all round-tripped fields.
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
	if diff := cmp.Diff(original.Spec, restored.Spec); diff != "" {
		t.Errorf("Spec mismatch after round-trip (-want +got):\n%s", diff)
	}

	// --- Status checks ---
	if diff := cmp.Diff(original.Status, restored.Status); diff != "" {
		t.Errorf("Status mismatch after round-trip (-want +got):\n%s", diff)
	}
	// GeneratedDeployment round-trip via ProfilingResults.SelectedConfig
	if restored.Status.ProfilingResults == nil || restored.Status.ProfilingResults.SelectedConfig == nil {
		t.Fatal("Status.ProfilingResults.SelectedConfig is nil after round-trip")
	}
}

func TestDGDR_IntermediateHubEditsWinOverSavedSpoke(t *testing.T) {
	const (
		editedModel = "edited-model"
		editedImage = "edited-image"
	)

	original := newV1alpha1DGDR()
	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := original.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	hub.Spec.Model = editedModel
	hub.Spec.Image = editedImage
	hub.Status.Phase = v1beta1.DGDRPhaseFailed

	restored := &DynamoGraphDeploymentRequest{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if restored.Spec.Model != editedModel {
		t.Fatalf("model = %q, want %s", restored.Spec.Model, editedModel)
	}
	if restored.Spec.ProfilingConfig.ProfilerImage != editedImage {
		t.Fatalf("profiler image = %q, want %s", restored.Spec.ProfilingConfig.ProfilerImage, editedImage)
	}
	if restored.Status.State != DGDRStateFailed {
		t.Fatalf("state = %q, want %q", restored.Status.State, DGDRStateFailed)
	}
}

func TestDGDR_IntermediateHubStatusWinsOverSavedSpokeStatus(t *testing.T) {
	original := newV1alpha1DGDR()
	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := original.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	hub.Status.Phase = v1beta1.DGDRPhasePending
	hub.Status.DGDName = "edited-dgd"

	restored := &DynamoGraphDeploymentRequest{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if restored.Status.State != DGDRStatePending {
		t.Fatalf("state = %q, want %q", restored.Status.State, DGDRStatePending)
	}
	if restored.Status.Deployment == nil {
		t.Fatalf("deployment status is nil")
	}
	if restored.Status.Deployment.Name != "edited-dgd" {
		t.Fatalf("deployment name = %q, want edited-dgd", restored.Status.Deployment.Name)
	}
	if restored.Status.Deployment.Created {
		t.Fatalf("deployment created = true, want false")
	}
	if restored.Status.Deployment.Namespace != original.Status.Deployment.Namespace {
		t.Fatalf("deployment namespace = %q, want %q", restored.Status.Deployment.Namespace, original.Status.Deployment.Namespace)
	}
}

func TestDGDR_IntermediateHubOnlyEditsAreSavedWithSpokeSnapshot(t *testing.T) {
	original := newV1alpha1DGDR()
	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := original.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	hub.Spec.SearchStrategy = v1beta1.SearchStrategyThorough

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if _, ok := spoke.Annotations[annDGDRHubSpec]; !ok {
		t.Fatalf("expected current hub-only edit to be saved in %q", annDGDRHubSpec)
	}

	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if restored.Spec.SearchStrategy != v1beta1.SearchStrategyThorough {
		t.Fatalf("searchStrategy = %q, want %q", restored.Spec.SearchStrategy, v1beta1.SearchStrategyThorough)
	}
}

func TestDGDR_IntermediateSpokeEditsWinOverSavedHub(t *testing.T) {
	original := newV1beta1DGDR()
	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	spoke.Spec.Model = "edited-model"
	spoke.Spec.ProfilingConfig.ProfilerImage = "edited-image"
	spoke.Spec.UseMocker = false
	spoke.Status.State = DGDRStateFailed

	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if restored.Spec.Model != "edited-model" {
		t.Fatalf("model = %q, want edited-model", restored.Spec.Model)
	}
	if restored.Spec.Image != "edited-image" {
		t.Fatalf("image = %q, want edited-image", restored.Spec.Image)
	}
	if restored.Spec.Features != nil && restored.Spec.Features.Mocker != nil && restored.Spec.Features.Mocker.Enabled {
		t.Fatalf("mocker enabled = true, want false")
	}
	if restored.Status.Phase != v1beta1.DGDRPhaseFailed {
		t.Fatalf("phase = %q, want %q", restored.Status.Phase, v1beta1.DGDRPhaseFailed)
	}
}

func TestDGDR_IntermediateProfilingJobAnnotationWinsOverSavedHub(t *testing.T) {
	original := newV1beta1DGDR()
	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	spoke.Annotations[annDGDRProfilingJobName] = "edited-profiling-job"

	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if restored.Status.ProfilingJobName != "edited-profiling-job" {
		t.Fatalf("profilingJobName = %q, want edited-profiling-job", restored.Status.ProfilingJobName)
	}
}

func TestDGDR_IntermediateSpokeAlphaOnlyStatusEditsSurviveSavedHub(t *testing.T) {
	original := newV1beta1DGDR()
	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	spoke.Status.Backend = "edited-backend"
	spoke.Status.ProfilingResults = "configmap/edited-results"
	if spoke.Status.Deployment == nil {
		spoke.Status.Deployment = &DeploymentStatus{}
	}
	spoke.Status.Deployment.Namespace = "edited-namespace"
	spoke.Status.Deployment.State = DGDStateFailed

	restoredHub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restoredHub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	restoredSpoke := &DynamoGraphDeploymentRequest{}
	if err := restoredSpoke.ConvertFrom(restoredHub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if restoredSpoke.Status.Backend != "edited-backend" {
		t.Fatalf("backend = %q, want edited-backend", restoredSpoke.Status.Backend)
	}
	if restoredSpoke.Status.ProfilingResults != "configmap/edited-results" {
		t.Fatalf("profilingResults = %q, want configmap/edited-results", restoredSpoke.Status.ProfilingResults)
	}
	if restoredSpoke.Status.Deployment == nil {
		t.Fatalf("deployment status is nil")
	}
	if restoredSpoke.Status.Deployment.Namespace != "edited-namespace" {
		t.Fatalf("deployment namespace = %q, want edited-namespace", restoredSpoke.Status.Deployment.Namespace)
	}
	if restoredSpoke.Status.Deployment.State != DGDStateFailed {
		t.Fatalf("deployment state = %q, want %q", restoredSpoke.Status.Deployment.State, DGDStateFailed)
	}
}

func TestDGDR_IntermediateSpokeAlphaOnlyEditsSurviveSavedHub(t *testing.T) {
	original := newV1beta1DGDR()
	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	spoke.Spec.ProfilingConfig.ConfigMapRef = &ConfigMapKeySelector{Name: "edited-config", Key: "profile.yaml"}
	spoke.Spec.ProfilingConfig.OutputPVC = "edited-output-pvc"

	restoredHub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restoredHub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	restoredSpoke := &DynamoGraphDeploymentRequest{}
	if err := restoredSpoke.ConvertFrom(restoredHub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if restoredSpoke.Spec.ProfilingConfig.ConfigMapRef == nil {
		t.Fatalf("configMapRef is nil")
	}
	if got := restoredSpoke.Spec.ProfilingConfig.ConfigMapRef.Name; got != "edited-config" {
		t.Fatalf("configMapRef.name = %q, want edited-config", got)
	}
	if got := restoredSpoke.Spec.ProfilingConfig.OutputPVC; got != "edited-output-pvc" {
		t.Fatalf("outputPVC = %q, want edited-output-pvc", got)
	}
}

func TestDGDR_ExplicitFalseEnableGPUDiscoveryRoundTrips(t *testing.T) {
	enableGPUDiscovery := false
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "explicit-false-gpu-discovery"},
		Spec: DynamoGraphDeploymentRequestSpec{
			Model:              "llama",
			Backend:            "vllm",
			EnableGPUDiscovery: &enableGPUDiscovery,
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Spec.EnableGPUDiscovery == nil {
		t.Fatalf("EnableGPUDiscovery = nil, want explicit false")
	}
	if *spoke.Spec.EnableGPUDiscovery {
		t.Fatalf("EnableGPUDiscovery = true, want false")
	}
}

func TestDGDR_IntermediateSpokeProfilingConfigModelCacheEditWins(t *testing.T) {
	original := newV1beta1DGDR()
	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	raw, err := json.Marshal(map[string]interface{}{
		"deployment": map[string]interface{}{
			"modelCache": map[string]interface{}{
				"pvcName":        "edited-pvc",
				"modelPathInPvc": "edited-path",
				"pvcMountPath":   "/edited",
			},
		},
	})
	if err != nil {
		t.Fatalf("marshal profiling config: %v", err)
	}
	spoke.Spec.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: raw}

	restoredHub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restoredHub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if restoredHub.Spec.ModelCache == nil {
		t.Fatalf("ModelCache is nil")
	}
	if got := restoredHub.Spec.ModelCache.PVCName; got != "edited-pvc" {
		t.Fatalf("ModelCache.PVCName = %q, want edited-pvc", got)
	}
	if got := restoredHub.Spec.ModelCache.PVCModelPath; got != "edited-path" {
		t.Fatalf("ModelCache.PVCModelPath = %q, want edited-path", got)
	}
	if got := restoredHub.Spec.ModelCache.PVCMountPath; got != "/edited" {
		t.Fatalf("ModelCache.PVCMountPath = %q, want /edited", got)
	}
}

func TestDGDR_ProfilingNodeSelectorRoundTrip(t *testing.T) {
	original := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "node-selector"},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "llama",
			Backend: v1beta1.BackendTypeVllm,
			Overrides: &v1beta1.OverridesSpec{
				ProfilingJob: &batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers:   []corev1.Container{},
							NodeSelector: map[string]string{"kubernetes.io/arch": "arm64"},
						},
					},
				},
			},
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if diff := cmp.Diff(original.Spec.Overrides.ProfilingJob.Template.Spec.NodeSelector, spoke.Spec.ProfilingConfig.NodeSelector); diff != "" {
		t.Fatalf("nodeSelector mismatch after ConvertFrom (-want +got):\n%s", diff)
	}

	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if diff := cmp.Diff(original.Spec, restored.Spec); diff != "" {
		t.Fatalf("spec mismatch after round-trip (-want +got):\n%s", diff)
	}
}

func TestDGDR_StaleProfilingConfigAnnotationDoesNotRestoreClearedHubFields(t *testing.T) {
	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "stale-blob",
			Annotations: map[string]string{
				annDGDRProfilingConfig: `{"sla":{"ttft":500,"isl":2048,"optimizationType":"throughput"},"planner":{"enabled":true},"deployment":{"modelCache":{"pvcName":"old-pvc"}},"extra":"saved"}`,
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "llama",
			Backend: v1beta1.BackendTypeVllm,
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Spec.ProfilingConfig.Config == nil {
		t.Fatalf("ProfilingConfig.Config is nil, want unknown blob data saved")
	}
	var blob map[string]interface{}
	if err := json.Unmarshal(spoke.Spec.ProfilingConfig.Config.Raw, &blob); err != nil {
		t.Fatalf("unmarshal profiling config: %v", err)
	}
	if _, ok := blob["sla"]; ok {
		t.Fatalf("stale sla key survived in profiling config: %v", blob["sla"])
	}
	if _, ok := blob["planner"]; ok {
		t.Fatalf("stale planner key survived in profiling config: %v", blob["planner"])
	}
	deployMap, _ := blob["deployment"].(map[string]interface{})
	if _, ok := deployMap["modelCache"]; ok {
		t.Fatalf("stale modelCache key survived in profiling config: %v", deployMap["modelCache"])
	}
	if blob["extra"] != "saved" {
		t.Fatalf("extra key = %v, want saved", blob["extra"])
	}

	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if restored.Spec.SLA != nil {
		t.Fatalf("SLA = %#v, want nil", restored.Spec.SLA)
	}
	if restored.Spec.Workload != nil {
		t.Fatalf("Workload = %#v, want nil", restored.Spec.Workload)
	}
	if restored.Spec.ModelCache != nil {
		t.Fatalf("ModelCache = %#v, want nil", restored.Spec.ModelCache)
	}
	if restored.Spec.Features != nil && restored.Spec.Features.Planner != nil {
		t.Fatalf("Planner = %#v, want nil", restored.Spec.Features.Planner)
	}
}

func TestDGDR_OptimizationTypeOnlyRoundTrip(t *testing.T) {
	opt := v1beta1.OptimizationTypeThroughput
	original := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "optimization-only"},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "llama",
			Backend: v1beta1.BackendTypeVllm,
			SLA: &v1beta1.SLASpec{
				OptimizationType: &opt,
			},
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Spec.ProfilingConfig.Config == nil {
		t.Fatalf("ProfilingConfig.Config is nil, want optimizationType in profiling blob")
	}
	var blob map[string]interface{}
	if err := json.Unmarshal(spoke.Spec.ProfilingConfig.Config.Raw, &blob); err != nil {
		t.Fatalf("unmarshal profiling config: %v", err)
	}
	slaMap, _ := blob[dgdrProfilingBlobSLAKey].(map[string]interface{})
	if slaMap == nil {
		t.Fatalf("sla key missing in profiling config: %v", blob)
	}
	if got := slaMap[dgdrProfilingBlobOptimizationTypeKey]; got != string(v1beta1.OptimizationTypeThroughput) {
		t.Fatalf("sla.optimizationType = %v, want %q", got, v1beta1.OptimizationTypeThroughput)
	}

	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if diff := cmp.Diff(original.Spec, restored.Spec); diff != "" {
		t.Fatalf("spec mismatch after round-trip (-want +got):\n%s", diff)
	}
}

func TestDGDR_OptimizationTypeWinsOverSavedNilSLA(t *testing.T) {
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "optimization-stale-saved"},
		Spec: DynamoGraphDeploymentRequestSpec{
			Model:     "llama",
			Backend:   "vllm",
			AutoApply: true,
			ProfilingConfig: ProfilingConfigSpec{
				Config: &apiextensionsv1.JSON{Raw: []byte(`{"sla":{"optimizationType":"throughput"}}`)},
			},
		},
	}
	setJSONAnnOnObj(src, annDGDRHubSpec, v1beta1.DynamoGraphDeploymentRequestSpec{
		Model:   "llama",
		Backend: v1beta1.BackendTypeVllm,
	})

	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if restored.Spec.SLA == nil || restored.Spec.SLA.OptimizationType == nil {
		t.Fatalf("SLA.OptimizationType = nil, want throughput")
	}
	if got := *restored.Spec.SLA.OptimizationType; got != v1beta1.OptimizationTypeThroughput {
		t.Fatalf("SLA.OptimizationType = %q, want %q", got, v1beta1.OptimizationTypeThroughput)
	}
}

func TestDGDR_ConversionDoesNotAliasSourceObjects(t *testing.T) {
	rawDGD := []byte(`{"apiVersion":"nvidia.com/v1alpha1","kind":"DynamoGraphDeployment"}`)
	expectedRawDGD := string(rawDGD)
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "alias-check",
			Labels: map[string]string{dgdrAliasLabelKey: dgdrAliasBefore},
		},
		Spec: DynamoGraphDeploymentRequestSpec{
			Model:     "llama",
			Backend:   "vllm",
			AutoApply: true,
		},
		Status: DynamoGraphDeploymentRequestStatus{
			Conditions: []metav1.Condition{{
				Type:    "Ready",
				Status:  metav1.ConditionTrue,
				Message: dgdrAliasBefore,
			}},
			GeneratedDeployment: &runtime.RawExtension{Raw: rawDGD},
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	src.Labels[dgdrAliasLabelKey] = "after"
	src.Spec.AutoApply = false
	src.Status.Conditions[0].Message = "after"
	src.Status.GeneratedDeployment.Raw[0] = '['
	if hub.Labels[dgdrAliasLabelKey] != dgdrAliasBefore {
		t.Fatalf("hub label aliases source label map: %q", hub.Labels[dgdrAliasLabelKey])
	}
	if hub.Spec.AutoApply == nil || !*hub.Spec.AutoApply {
		t.Fatalf("hub AutoApply aliases source bool: %v", hub.Spec.AutoApply)
	}
	if hub.Status.Conditions[0].Message != dgdrAliasBefore {
		t.Fatalf("hub condition aliases source condition: %q", hub.Status.Conditions[0].Message)
	}
	if string(hub.Status.ProfilingResults.SelectedConfig.Raw) != expectedRawDGD {
		t.Fatalf("hub selectedConfig aliases source raw extension: %s", string(hub.Status.ProfilingResults.SelectedConfig.Raw))
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	hub.Labels[dgdrAliasLabelKey] = "hub-after"
	*hub.Spec.AutoApply = false
	hub.Status.Conditions[0].Message = "hub-after"
	hub.Status.ProfilingResults.SelectedConfig.Raw[0] = '['
	if spoke.Labels[dgdrAliasLabelKey] != dgdrAliasBefore {
		t.Fatalf("spoke label aliases hub label map: %q", spoke.Labels[dgdrAliasLabelKey])
	}
	if !spoke.Spec.AutoApply {
		t.Fatalf("spoke AutoApply aliases hub bool")
	}
	if spoke.Status.Conditions[0].Message != dgdrAliasBefore {
		t.Fatalf("spoke condition aliases hub condition: %q", spoke.Status.Conditions[0].Message)
	}
	if string(spoke.Status.GeneratedDeployment.Raw) != expectedRawDGD {
		t.Fatalf("spoke generatedDeployment aliases hub raw extension: %s", string(spoke.Status.GeneratedDeployment.Raw))
	}
}

func TestDGDR_ProfilingResourcesClaimsOnlyRoundTrip(t *testing.T) {
	original := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "claims-only"},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "llama",
			Backend: v1beta1.BackendTypeVllm,
			Overrides: &v1beta1.OverridesSpec{
				ProfilingJob: &batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{
								Name: "profiler",
								Resources: corev1.ResourceRequirements{
									Claims: []corev1.ResourceClaim{{
										Name:    "gpu-claim",
										Request: "gpu",
									}},
								},
							}},
						},
					},
				},
			},
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Spec.ProfilingConfig.Resources == nil {
		t.Fatalf("ProfilingConfig.Resources is nil")
	}
	if diff := cmp.Diff(original.Spec.Overrides.ProfilingJob.Template.Spec.Containers[0].Resources.Claims, spoke.Spec.ProfilingConfig.Resources.Claims); diff != "" {
		t.Fatalf("claims mismatch after ConvertFrom (-want +got):\n%s", diff)
	}

	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if diff := cmp.Diff(original.Spec, restored.Spec); diff != "" {
		t.Fatalf("spec mismatch after round-trip (-want +got):\n%s", diff)
	}
}

func TestDGDR_ProfilingJobHubOnlyFieldsRoundTrip(t *testing.T) {
	parallelism := int32(2)
	original := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "profiling-job-hub-only"},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "llama",
			Backend: v1beta1.BackendTypeVllm,
			Overrides: &v1beta1.OverridesSpec{
				ProfilingJob: &batchv1.JobSpec{
					Parallelism: &parallelism,
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"job": "profiling"},
						},
						Spec: corev1.PodSpec{
							RestartPolicy: corev1.RestartPolicyNever,
							Containers: []corev1.Container{
								{
									Name:  "profiler",
									Image: "profiler:v1",
									Resources: corev1.ResourceRequirements{
										Claims: []corev1.ResourceClaim{{
											Name:    "gpu-claim",
											Request: "gpu",
										}},
									},
								},
								{
									Name:  "helper",
									Image: "helper:v1",
								},
							},
						},
					},
				},
			},
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if diff := cmp.Diff(original.Spec, restored.Spec); diff != "" {
		t.Fatalf("spec mismatch after round-trip (-want +got):\n%s", diff)
	}
}

func TestDGDR_LiveProfilingFieldsOverrideSavedJob(t *testing.T) {
	original := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "profiling-job-live-wins"},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "llama",
			Backend: v1beta1.BackendTypeVllm,
			Overrides: &v1beta1.OverridesSpec{
				ProfilingJob: &batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							NodeSelector: map[string]string{"old": "selector"},
							Tolerations: []corev1.Toleration{{
								Key:      "old",
								Operator: corev1.TolerationOpExists,
							}},
							Containers: []corev1.Container{{
								Name:  "profiler",
								Image: "profiler:v1",
								Resources: corev1.ResourceRequirements{
									Claims: []corev1.ResourceClaim{{
										Name:    "old-claim",
										Request: "gpu",
									}},
								},
							}},
						},
					},
				},
			},
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	spoke.Spec.ProfilingConfig.NodeSelector = map[string]string{"new": "selector"}
	spoke.Spec.ProfilingConfig.Tolerations = []corev1.Toleration{{
		Key:      "new",
		Operator: corev1.TolerationOpExists,
	}}
	spoke.Spec.ProfilingConfig.Resources = &corev1.ResourceRequirements{
		Claims: []corev1.ResourceClaim{{
			Name:    "new-claim",
			Request: "gpu",
		}},
	}

	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	gotJob := restored.Spec.Overrides.ProfilingJob
	if gotJob == nil || len(gotJob.Template.Spec.Containers) == 0 {
		t.Fatalf("ProfilingJob container missing after ConvertTo")
	}
	if got := gotJob.Template.Spec.Containers[0].Name; got != "profiler" {
		t.Fatalf("container name = %q, want profiler", got)
	}
	if got := gotJob.Template.Spec.Containers[0].Image; got != "profiler:v1" {
		t.Fatalf("container image = %q, want profiler:v1", got)
	}
	if diff := cmp.Diff(spoke.Spec.ProfilingConfig.Resources, &gotJob.Template.Spec.Containers[0].Resources); diff != "" {
		t.Fatalf("resources mismatch after live edit (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(spoke.Spec.ProfilingConfig.NodeSelector, gotJob.Template.Spec.NodeSelector); diff != "" {
		t.Fatalf("nodeSelector mismatch after live edit (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(spoke.Spec.ProfilingConfig.Tolerations, gotJob.Template.Spec.Tolerations); diff != "" {
		t.Fatalf("tolerations mismatch after live edit (-want +got):\n%s", diff)
	}
}

func TestDGDR_ClearedSelectedConfigDoesNotRestoreSavedGeneratedDeployment(t *testing.T) {
	original := newV1alpha1DGDR()
	original.Status.GeneratedDeployment = &runtime.RawExtension{Raw: []byte(`{"apiVersion":"nvidia.com/v1alpha1","kind":"DynamoGraphDeployment"}`)}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := original.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	hub.Status.ProfilingResults = nil

	restored := &DynamoGraphDeploymentRequest{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if restored.Status.GeneratedDeployment != nil {
		t.Fatalf("GeneratedDeployment = %#v, want nil after hub selectedConfig clear", restored.Status.GeneratedDeployment)
	}
}

func TestDGDR_SparseSpokeSpecOmitsRepresentableProfilingFields(t *testing.T) {
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "alpha-profiling"},
		Spec: DynamoGraphDeploymentRequestSpec{
			Model:   "llama",
			Backend: "vllm",
			ProfilingConfig: ProfilingConfigSpec{
				ProfilerImage: "profiler:v1",
				Resources:     &corev1.ResourceRequirements{},
				Tolerations: []corev1.Toleration{{
					Key:      "gpu",
					Operator: corev1.TolerationOpExists,
				}},
				NodeSelector: map[string]string{"kubernetes.io/arch": "arm64"},
			},
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if _, ok := hub.Annotations[annDGDRSpokeSpec]; ok {
		t.Fatalf("%s should be omitted when only represented profiling fields need round-trip: %v", annDGDRSpokeSpec, hub.Annotations)
	}

	restored := &DynamoGraphDeploymentRequest{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if diff := cmp.Diff(src.Spec.ProfilingConfig.Resources, restored.Spec.ProfilingConfig.Resources); diff != "" {
		t.Fatalf("resources mismatch after round-trip (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(src.Spec.ProfilingConfig.Tolerations, restored.Spec.ProfilingConfig.Tolerations); diff != "" {
		t.Fatalf("tolerations mismatch after round-trip (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(src.Spec.ProfilingConfig.NodeSelector, restored.Spec.ProfilingConfig.NodeSelector); diff != "" {
		t.Fatalf("nodeSelector mismatch after round-trip (-want +got):\n%s", diff)
	}
}

func TestDGDR_SparseHubSpecOmitsRepresentableFields(t *testing.T) {
	autoApply := true
	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "hub-sparse"},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:          "llama",
			Backend:        v1beta1.BackendTypeVllm,
			Image:          "profiler:v1",
			AutoApply:      &autoApply,
			SearchStrategy: v1beta1.SearchStrategyThorough,
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	raw := spoke.Annotations[annDGDRHubSpec]
	if raw == "" {
		t.Fatalf("expected %s to save hub-only searchStrategy", annDGDRHubSpec)
	}
	saved := decodeDGDRHubSpecSaveForTest(t, raw)
	if saved.Spec.Model != "" || saved.Spec.Backend != "" || saved.Spec.Image != "" {
		t.Fatalf(dgdrHubSaveRepresentedFieldsMsg, saved.Spec)
	}
	if saved.Spec.SearchStrategy != v1beta1.SearchStrategyThorough {
		t.Fatalf("saved searchStrategy = %q, want %q", saved.Spec.SearchStrategy, v1beta1.SearchStrategyThorough)
	}
}

func TestDGDR_SparseHubSpecNilAutoApplyOmitsRepresentableFields(t *testing.T) {
	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "hub-sparse-nil-autoapply"},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:     "llama",
			Backend:   v1beta1.BackendTypeVllm,
			Image:     "profiler:v1",
			AutoApply: nil,
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	raw := spoke.Annotations[annDGDRHubSpec]
	if raw == "" {
		t.Fatalf("expected %s to save nil AutoApply shape", annDGDRHubSpec)
	}
	saved := decodeDGDRHubSpecSaveForTest(t, raw)
	if saved.Spec.Model != "" || saved.Spec.Backend != "" || saved.Spec.Image != "" {
		t.Fatalf(dgdrHubSaveRepresentedFieldsMsg, saved.Spec)
	}
	if !saved.AutoApplyNil {
		t.Fatalf("AutoApplyNil = false, want true")
	}

	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if restored.Spec.AutoApply != nil {
		t.Fatalf("AutoApply = %v, want nil", *restored.Spec.AutoApply)
	}
}

func TestDGDR_HubNilAutoApplyWithOtherHubSavesRoundTrips(t *testing.T) {
	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "nil-autoapply-with-mocker"},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "llama",
			Backend: v1beta1.BackendTypeVllm,
			Features: &v1beta1.FeaturesSpec{
				Mocker: &v1beta1.MockerSpec{},
			},
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if restored.Spec.AutoApply != nil {
		t.Fatalf("AutoApply = %v, want nil", *restored.Spec.AutoApply)
	}
}

func TestDGDR_SparseHubSpecWorkloadWithoutSLAOmitsRepresentableFields(t *testing.T) {
	isl := int32(2048)
	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "hub-sparse-workload-no-sla"},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "llama",
			Backend: v1beta1.BackendTypeVllm,
			Image:   "profiler:v1",
			Workload: &v1beta1.WorkloadSpec{
				ISL: &isl,
			},
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	raw := spoke.Annotations[annDGDRHubSpec]
	if raw == "" {
		t.Fatalf("expected %s to save workload-without-sla shape", annDGDRHubSpec)
	}
	saved := decodeDGDRHubSpecSaveForTest(t, raw)
	if saved.Spec.Model != "" || saved.Spec.Backend != "" || saved.Spec.Image != "" {
		t.Fatalf(dgdrHubSaveRepresentedFieldsMsg, saved.Spec)
	}
	if saved.Spec.Workload == nil || saved.Spec.Workload.ISL != nil {
		t.Fatalf("saved Workload = %#v, want sparse shape marker without represented ISL", saved.Spec.Workload)
	}

	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if restored.Spec.SLA != nil {
		t.Fatalf("SLA = %#v, want nil", restored.Spec.SLA)
	}
	if restored.Spec.Workload == nil || restored.Spec.Workload.ISL == nil || *restored.Spec.Workload.ISL != isl {
		t.Fatalf("Workload = %#v, want ISL %d", restored.Spec.Workload, isl)
	}
}

func TestDGDR_HubKnownProfilingBlobDoesNotLeakAnnotation(t *testing.T) {
	ttft := 0.1
	itl := 0.2
	e2e := 0.3
	optimizationType := v1beta1.OptimizationTypeLatency
	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "known-profiling"},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:          "llama",
			Backend:        v1beta1.BackendTypeVllm,
			SearchStrategy: v1beta1.SearchStrategyThorough,
			SLA: &v1beta1.SLASpec{
				TTFT:             &ttft,
				ITL:              &itl,
				E2ELatency:       &e2e,
				OptimizationType: &optimizationType,
			},
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if _, ok := restored.Annotations[annDGDRProfilingConfig]; ok {
		t.Fatalf("known profiling fields leaked into %s: %v", annDGDRProfilingConfig, restored.Annotations)
	}
}

func TestDGDR_HubDGDNameDoesNotCreateDeploymentAnnotation(t *testing.T) {
	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "synthetic-deployment"},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "llama",
			Backend: v1beta1.BackendTypeVllm,
		},
		Status: v1beta1.DynamoGraphDeploymentRequestStatus{
			DGDName: "generated-dgd",
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if _, ok := restored.Annotations[annDGDRDeploymentStatus]; ok {
		t.Fatalf("synthetic deployment leaked into %s: %v", annDGDRDeploymentStatus, restored.Annotations)
	}
}

func TestDGDR_HubDeployedPhaseRespectsIntermediateSpokeCreatedEdit(t *testing.T) {
	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "deployed-with-dgd"},
		Status: v1beta1.DynamoGraphDeploymentRequestStatus{
			Phase:   v1beta1.DGDRPhaseDeployed,
			DGDName: "created-dgd",
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Status.Deployment == nil {
		t.Fatalf("Deployment = nil, want created deployment")
	}
	spoke.Status.Deployment.Created = false

	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if restored.Status.Phase != v1beta1.DGDRPhaseReady {
		t.Fatalf("Phase = %q, want %q", restored.Status.Phase, v1beta1.DGDRPhaseReady)
	}
}

func TestDGDR_EmptyStateRoundTrips(t *testing.T) {
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "empty-state"},
		Status: DynamoGraphDeploymentRequestStatus{
			State: "",
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Status.State != "" {
		t.Fatalf("State = %q, want empty", spoke.Status.State)
	}
}

func TestDGDR_SpokeStatusSaveDoesNotWriteLegacyStatusSnapshot(t *testing.T) {
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "no-legacy-status-snapshot"},
		Status: DynamoGraphDeploymentRequestStatus{
			State: DGDRStateDeploymentDeleted,
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if _, ok := hub.Annotations[annDGDRSpokeStatus]; !ok {
		t.Fatalf("expected %s to save lossy state", annDGDRSpokeStatus)
	}
	if _, ok := hub.Annotations[annDGDRLegacyStatusSnapshot]; ok {
		t.Fatalf("unexpected legacy status snapshot annotation %s: %v", annDGDRLegacyStatusSnapshot, hub.Annotations)
	}
}

func TestDGDR_OldFullSpokeSpecDoesNotRestoreClearedProfilingFields(t *testing.T) {
	oldSpokeSpec := DynamoGraphDeploymentRequestSpec{
		Model:   "llama",
		Backend: "vllm",
		ProfilingConfig: ProfilingConfigSpec{
			Resources: &corev1.ResourceRequirements{},
			Tolerations: []corev1.Toleration{{
				Key:      "old",
				Operator: corev1.TolerationOpExists,
			}},
			NodeSelector: map[string]string{"old": "selector"},
		},
	}
	envelope, err := json.Marshal(dgdrSpokeSpecSave{Spec: oldSpokeSpec})
	if err != nil {
		t.Fatalf("marshal old spoke spec: %v", err)
	}
	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "cleared-profiling",
			Annotations: map[string]string{
				annDGDRSpokeSpec: string(envelope),
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "llama",
			Backend: v1beta1.BackendTypeVllm,
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Spec.ProfilingConfig.Resources != nil {
		t.Fatalf("resources restored from stale full annotation: %#v", spoke.Spec.ProfilingConfig.Resources)
	}
	if len(spoke.Spec.ProfilingConfig.Tolerations) != 0 {
		t.Fatalf("tolerations restored from stale full annotation: %#v", spoke.Spec.ProfilingConfig.Tolerations)
	}
	if len(spoke.Spec.ProfilingConfig.NodeSelector) != 0 {
		t.Fatalf("nodeSelector restored from stale full annotation: %#v", spoke.Spec.ProfilingConfig.NodeSelector)
	}
}

// TestConvertTo_InvalidProfilingConfigJSON verifies that malformed or arbitrary
// RawExtension bytes in ProfilingConfig.Config are saved rather than
// rejected. The v1alpha1 field is an opaque extension point; typed v1beta1
// fields are projected only when the payload is a legacy JSON object we
// understand.
func TestConvertTo_InvalidProfilingConfigJSON(t *testing.T) {
	src := newV1alpha1DGDR()
	src.Spec.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: []byte(`{not valid json`)}

	dst := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo() unexpected error = %v", err)
	}
	if got := dst.Annotations[annDGDRProfilingConfig]; got != `{not valid json` {
		t.Fatalf("profiling config annotation = %q, want raw payload", got)
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(dst); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Spec.ProfilingConfig.Config == nil || string(spoke.Spec.ProfilingConfig.Config.Raw) != `{not valid json` {
		t.Fatalf("ProfilingConfig.Config raw = %v, want saved invalid payload", spoke.Spec.ProfilingConfig.Config)
	}
}

func TestConvertTo_EmptyProfilingConfigRawDoesNotError(t *testing.T) {
	src := newV1alpha1DGDR()
	src.Spec.ProfilingConfig.Config = &apiextensionsv1.JSON{Raw: []byte{}}

	dst := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo() unexpected error = %v", err)
	}
	if _, ok := dst.Annotations[annDGDRProfilingConfig]; ok {
		t.Fatalf("empty profiling config raw should not emit annotation, got %v", dst.Annotations)
	}
}
