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
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDGDRReadsLegacyAnnotationsWrittenByOldConverter(t *testing.T) {
	legacyBlob, err := json.Marshal(map[string]any{"extra_key": "preserved"})
	if err != nil {
		t.Fatalf("marshal profiling config: %v", err)
	}
	legacyOverrides, err := json.Marshal(map[string]any{
		"name": testDGDName, "namespace": "prod", "labels": map[string]string{"team": "ml"},
	})
	if err != nil {
		t.Fatalf("marshal deployment overrides: %v", err)
	}
	legacyDeployment, err := json.Marshal(dgdrDeploymentStatusAnnotation{
		DeploymentStatus: DeploymentStatus{Name: testDGDName, Namespace: "prod", State: "initializing", Created: true},
		RequestState:     DGDRStateProfiling,
	})
	if err != nil {
		t.Fatalf("marshal deployment status: %v", err)
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "legacy",
			Annotations: map[string]string{
				legacyAnnDGDRConfigMapRef:     `{"name":"base-config","key":"disagg.yaml"}`,
				legacyAnnDGDROutputPVC:        "output-pvc",
				legacyAnnDGDREnableGPUDisc:    annotationTrue,
				legacyAnnDGDRDeployOverrides:  string(legacyOverrides),
				legacyAnnDGDRProfilingConfig:  string(legacyBlob),
				legacyAnnDGDRStatusBackend:    "vllm",
				legacyAnnDGDRProfilingResults: "configmap/profiling-cm",
				legacyAnnDGDRDeploymentStatus: string(legacyDeployment),
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "meta-llama/Llama-3.1-8B",
			Backend: v1beta1.BackendTypeVllm,
			Image:   "nvcr.io/nvidia/dynamo:latest",
		},
		Status: v1beta1.DynamoGraphDeploymentRequestStatus{
			Phase:              v1beta1.DGDRPhaseProfiling,
			ObservedGeneration: 3,
			DGDName:            testDGDName,
		},
	}

	restored := &DynamoGraphDeploymentRequest{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	if got := restored.Spec.ProfilingConfig.ConfigMapRef; got == nil || got.Name != "base-config" || got.Key != "disagg.yaml" {
		t.Fatalf("ConfigMapRef after legacy read = %#v", got)
	}
	if got := restored.Spec.ProfilingConfig.OutputPVC; got != "output-pvc" {
		t.Fatalf("OutputPVC after legacy read = %q", got)
	}
	if restored.Spec.EnableGPUDiscovery == nil || !*restored.Spec.EnableGPUDiscovery {
		t.Fatal("EnableGPUDiscovery was not restored from legacy annotations")
	}
	if got := restored.Spec.DeploymentOverrides; got == nil || got.Name != testDGDName || got.Namespace != "prod" || got.Labels["team"] != "ml" {
		t.Fatalf("DeploymentOverrides after legacy read = %#v", got)
	}
	assertProfilingConfigBlobHas(t, restored.Spec.ProfilingConfig.Config, map[string]any{
		"extra_key": "preserved",
	})
	if restored.Status.Backend != "vllm" || restored.Status.ProfilingResults != "configmap/profiling-cm" {
		t.Fatalf("status after legacy read = %#v", restored.Status)
	}
	if got := restored.Status.Deployment; got == nil || got.Name != testDGDName || got.Namespace != "prod" || !got.Created {
		t.Fatalf("Deployment status after legacy read = %#v", got)
	}
}

func TestDGDRDoesNotWriteLegacyAnnotations(t *testing.T) {
	original := newV1alpha1DGDR()
	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := original.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	for _, key := range legacyDGDRAnnotationKeys() {
		if _, ok := hub.Annotations[key]; ok {
			t.Fatalf("legacy annotation %q was written", key)
		}
	}
	if hub.Annotations[annDGDRSpec] == "" || hub.Annotations[annDGDRStatus] == "" {
		t.Fatalf("structural annotations were not written: %#v", hub.Annotations)
	}
}

func TestDGDRReadsLegacyHubProfilingJobName(t *testing.T) {
	spoke := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{legacyAnnDGDRProfilingJobName: "legacy-job"},
		},
		Status: DynamoGraphDeploymentRequestStatus{State: DGDRStateProfiling},
	}
	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if hub.Status.ProfilingJobName != "legacy-job" {
		t.Fatalf("ProfilingJobName = %q, want legacy-job", hub.Status.ProfilingJobName)
	}
	if _, ok := hub.Annotations[legacyAnnDGDRProfilingJobName]; ok {
		t.Fatal("legacy profiling-job annotation was re-emitted")
	}
}

func TestDGDRStructuralStatusTakesPrecedenceOverLegacyProfilingJobName(t *testing.T) {
	structuralStatus, err := json.Marshal(v1beta1.DynamoGraphDeploymentRequestStatus{
		ProfilingJobName: "structural-job",
	})
	if err != nil {
		t.Fatalf("marshal structural status: %v", err)
	}
	metadata := &metav1.ObjectMeta{Annotations: map[string]string{
		annDGDRStatus:                 string(structuralStatus),
		legacyAnnDGDRProfilingJobName: "stale-legacy-job",
	}}

	_, restoredStatus := restoreDGDRHubAnnotations(metadata)
	if restoredStatus == nil {
		t.Fatal("structural status was not restored")
	}
	if got := restoredStatus.ProfilingJobName; got != "structural-job" {
		t.Fatalf("ProfilingJobName = %q, want structural-job", got)
	}
}

func TestDGDRLegacyProfilingConfigDoesNotOverrideLiveHubFields(t *testing.T) {
	legacyBlob, err := json.Marshal(map[string]any{
		"sla": map[string]any{
			"ttft": 100,
			"itl":  10,
			"isl":  200,
			"osl":  20,
		},
		"deployment": map[string]any{
			"modelCache": map[string]any{
				"pvcName": "stale-pvc",
			},
		},
		"planner":   map[string]any{"stale": true},
		"extra_key": "keep",
	})
	if err != nil {
		t.Fatalf("marshal legacy blob: %v", err)
	}
	hub := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "edited",
			Annotations: map[string]string{
				legacyAnnDGDRProfilingConfig: string(legacyBlob),
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model: "model",
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	if restored.Spec.SLA != nil {
		t.Fatalf("stale legacy SLA restored into live hub spec: %#v", restored.Spec.SLA)
	}
	if restored.Spec.Workload != nil {
		t.Fatalf("stale legacy workload restored into live hub spec: %#v", restored.Spec.Workload)
	}
	if restored.Spec.ModelCache != nil {
		t.Fatalf("stale legacy model cache restored into live hub spec: %#v", restored.Spec.ModelCache)
	}
	if restored.Spec.Features != nil && restored.Spec.Features.Planner != nil {
		t.Fatalf("stale legacy planner restored into live hub spec: %#v", restored.Spec.Features.Planner)
	}
	assertProfilingConfigBlobHas(t, spoke.Spec.ProfilingConfig.Config, map[string]any{
		"extra_key": "keep",
	})
}
