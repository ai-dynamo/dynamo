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
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestBugDGDR_HubClearsAlphaOnlySpecAnnotationsDropsSavedSpokeSpec(t *testing.T) {
	enableGPUDiscovery := true
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "alpha-only-spec"},
		Spec: DynamoGraphDeploymentRequestSpec{
			Model:              "llama",
			Backend:            "vllm",
			AutoApply:          true,
			EnableGPUDiscovery: &enableGPUDiscovery,
			ProfilingConfig: ProfilingConfigSpec{
				ConfigMapRef: &ConfigMapKeySelector{Name: "old-config", Key: "profile.yaml"},
				OutputPVC:    "old-output",
			},
			DeploymentOverrides: &DeploymentOverridesSpec{
				Name:        "old-dgd",
				Namespace:   "old-ns",
				Labels:      map[string]string{"old": "label"},
				Annotations: map[string]string{"old": "annotation"},
			},
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	for _, key := range []string{
		annDGDREnableGPUDisc,
		annDGDRConfigMapRef,
		annDGDROutputPVC,
		annDGDRDeployOverrides,
	} {
		delete(hub.Annotations, key)
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Spec.EnableGPUDiscovery != nil {
		t.Fatalf("EnableGPUDiscovery restored from stale snapshot: %v", *spoke.Spec.EnableGPUDiscovery)
	}
	if spoke.Spec.ProfilingConfig.ConfigMapRef != nil {
		t.Fatalf("ConfigMapRef restored from stale snapshot: %#v", spoke.Spec.ProfilingConfig.ConfigMapRef)
	}
	if spoke.Spec.ProfilingConfig.OutputPVC != "" {
		t.Fatalf("OutputPVC restored from stale snapshot: %q", spoke.Spec.ProfilingConfig.OutputPVC)
	}
	if spoke.Spec.DeploymentOverrides != nil {
		t.Fatalf("DeploymentOverrides restored from stale snapshot: %#v", spoke.Spec.DeploymentOverrides)
	}
}

func TestBugDGDR_HubClearsAlphaOnlyStatusAnnotationsDropsSavedSpokeStatus(t *testing.T) {
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "alpha-only-status"},
		Status: DynamoGraphDeploymentRequestStatus{
			State:            DGDRStateProfiling,
			Backend:          "old-backend",
			ProfilingResults: "configmap/old-results",
			Deployment: &DeploymentStatus{
				Namespace: "old-ns",
				State:     "old-state",
			},
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	for _, key := range []string{
		annDGDRStatusBackend,
		annDGDRProfilingResults,
		annDGDRDeploymentStatus,
	} {
		delete(hub.Annotations, key)
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Status.Backend != "" {
		t.Fatalf("Backend restored from stale snapshot: %q", spoke.Status.Backend)
	}
	if spoke.Status.ProfilingResults != "" {
		t.Fatalf("ProfilingResults restored from stale snapshot: %q", spoke.Status.ProfilingResults)
	}
	if spoke.Status.Deployment != nil {
		t.Fatalf("Deployment restored from stale snapshot: %#v", spoke.Status.Deployment)
	}
}

func TestBugDGDR_SparseHubSpecSaveOmitsRepresentableContextFields(t *testing.T) {
	autoApply := true
	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "sparse-context"},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:          "llama",
			Backend:        v1beta1.BackendTypeVllm,
			Image:          "profiler:v1",
			AutoApply:      &autoApply,
			SearchStrategy: v1beta1.SearchStrategyThorough,
			ModelCache: &v1beta1.ModelCacheSpec{
				PVCName:      "model-pvc",
				PVCModelPath: "llama",
				PVCMountPath: "/models",
			},
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
	if saved.Spec.SearchStrategy != v1beta1.SearchStrategyThorough {
		t.Fatalf("saved searchStrategy = %q, want %q", saved.Spec.SearchStrategy, v1beta1.SearchStrategyThorough)
	}
	if saved.Spec.AutoApply != nil {
		t.Fatalf("hub save included representable AutoApply: %v", *saved.Spec.AutoApply)
	}
	if saved.Spec.ModelCache != nil {
		t.Fatalf("hub save included representable ModelCache: %#v", saved.Spec.ModelCache)
	}
}

func TestBugDGDR_HubEmptyModelCacheShapeRoundTrips(t *testing.T) {
	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "empty-model-cache"},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:      "llama",
			Backend:    v1beta1.BackendTypeVllm,
			ModelCache: &v1beta1.ModelCacheSpec{},
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
	if restored.Spec.ModelCache == nil {
		t.Fatalf("ModelCache = nil, want empty object shape saved")
	}
}

func TestBugDGDR_SpokeLossyStateSurvivesUnrelatedHubStatusChange(t *testing.T) {
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "deployment-deleted-state"},
		Status: DynamoGraphDeploymentRequestStatus{
			State:              DGDRStateDeploymentDeleted,
			ObservedGeneration: 1,
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	hub.Status.ObservedGeneration = 2

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Status.State != DGDRStateDeploymentDeleted {
		t.Fatalf("State = %q, want %q", spoke.Status.State, DGDRStateDeploymentDeleted)
	}
	if spoke.Status.ObservedGeneration != 2 {
		t.Fatalf("ObservedGeneration = %d, want 2", spoke.Status.ObservedGeneration)
	}
}

func TestBugDGDR_HubEmptyPhaseRoundTrips(t *testing.T) {
	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "empty-phase"},
		Status: v1beta1.DynamoGraphDeploymentRequestStatus{
			Phase: "",
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
	if restored.Status.Phase != "" {
		t.Fatalf("Phase = %q, want empty", restored.Status.Phase)
	}
}
