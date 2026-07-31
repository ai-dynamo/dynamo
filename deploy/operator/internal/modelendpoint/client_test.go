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

package modelendpoint

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
)

func TestLoadLoRA(t *testing.T) {
	// Create test servers for different scenarios
	successServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify HTTP method
		if r.Method != http.MethodPost {
			t.Errorf("expected POST method, got %s", r.Method)
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		// Verify Content-Type header
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer successServer.Close()

	failingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify HTTP method even for failing requests
		if r.Method != http.MethodPost {
			t.Errorf("expected POST method, got %s", r.Method)
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer failingServer.Close()

	tests := []struct {
		name               string
		modelType          string
		sourceURI          string
		candidates         []Candidate
		expectError        bool
		errorContains      string
		expectedCount      int
		expectedReadyCount int
	}{
		{
			name:               "non-lora model - skips loading",
			modelType:          "base",
			candidates:         []Candidate{{Address: "http://10.0.1.5:9090", PodName: "pod-1"}},
			expectError:        false,
			expectedCount:      1,
			expectedReadyCount: 0,
		},
		{
			name:               "empty candidates",
			modelType:          "base",
			candidates:         []Candidate{},
			expectError:        false,
			expectedCount:      0,
			expectedReadyCount: 0,
		},
		{
			name:          "lora with nil source",
			modelType:     "lora",
			sourceURI:     "",
			candidates:    []Candidate{{Address: "http://10.0.1.5:9090", PodName: "pod-1"}},
			expectError:   true,
			errorContains: "source URI is required",
		},
		{
			name:      "lora with valid source - all success",
			modelType: "lora",
			sourceURI: "s3://bucket/model",
			candidates: []Candidate{
				{Address: successServer.URL, PodName: "pod-1"},
				{Address: successServer.URL, PodName: "pod-2"},
			},
			expectError:        false,
			expectedCount:      2,
			expectedReadyCount: 2,
		},
		{
			name:      "lora with valid source - partial failure",
			modelType: "lora",
			sourceURI: "s3://bucket/model",
			candidates: []Candidate{
				{Address: successServer.URL, PodName: "pod-1"},
				{Address: failingServer.URL, PodName: "pod-2"},
			},
			expectError:        true, // workerpool returns error on any failure
			expectedCount:      2,
			expectedReadyCount: 1,
		},
		{
			name:      "lora with huggingface source",
			modelType: "lora",
			sourceURI: "hf://org/model@v1.0",
			candidates: []Candidate{
				{Address: successServer.URL, PodName: "pod-1"},
			},
			expectError:        false,
			expectedCount:      1,
			expectedReadyCount: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient()
			ctx := context.Background()

			var source *v1alpha1.ModelSource
			if tt.sourceURI != "" {
				source = &v1alpha1.ModelSource{URI: tt.sourceURI}
			}

			model := &v1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-model",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoModelSpec{
					ModelName: "test-model",
					ModelType: tt.modelType,
					Source:    source,
				},
			}

			endpoints, err := client.LoadLoRA(ctx, tt.candidates, model)

			// Check error expectation
			if tt.expectError && tt.errorContains != "" {
				// For validation errors (like missing source URI), we return early
				if err == nil {
					t.Error("expected error but got none")
				} else if !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("expected error to contain %q, got %v", tt.errorContains, err)
				}
				return
			}

			// For partial failures, we expect an error but still get endpoints
			if tt.expectError && err == nil {
				t.Error("expected error for partial failure but got none")
			}

			if !tt.expectError && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Verify endpoint count
			if len(endpoints) != tt.expectedCount {
				t.Errorf("expected %d endpoints, got %d", tt.expectedCount, len(endpoints))
			}

			// Count ready endpoints
			readyCount := 0
			for _, ep := range endpoints {
				if ep.Ready {
					readyCount++
				}
			}

			if readyCount != tt.expectedReadyCount {
				t.Errorf("expected %d ready endpoints, got %d", tt.expectedReadyCount, readyCount)
			}
		})
	}
}

func TestLoadLoRAAllowsLegacyUnavailableVLLMPrefill(t *testing.T) {
	var capableLoads atomic.Int32
	capableServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/loras":
			capableLoads.Add(1)
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer capableServer.Close()

	var incapableLoads atomic.Int32
	incapableServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		incapableLoads.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("LoRA management not available: no 'load_lora' handler is registered"))
	}))
	defer incapableServer.Close()

	model := &v1alpha1.DynamoModel{
		ObjectMeta: metav1.ObjectMeta{Name: "test-lora", Namespace: "default"},
		Spec: v1alpha1.DynamoModelSpec{
			ModelName: "test-lora",
			ModelType: "lora",
			Source:    &v1alpha1.ModelSource{URI: "s3://bucket/model"},
		},
	}
	candidates := []Candidate{
		{Address: capableServer.URL, PodName: "vllm-decode"},
		{
			Address:                        incapableServer.URL,
			PodName:                        "vllm-prefill",
			KubernetesReady:                true,
			AllowLoRAManagementUnavailable: true,
			LoRAFallbackGroup:              "graph:a",
		},
	}

	endpoints, err := NewClient().LoadLoRA(context.Background(), candidates, model)
	if err == nil {
		t.Fatal("all-legacy vLLM prefill must remain not ready without an adapter-prefill card")
	}
	if len(endpoints) != 2 {
		t.Fatalf("expected both physical endpoints in status, got %d: %#v", len(endpoints), endpoints)
	}
	if !endpoints[0].Ready || endpoints[1].Ready {
		t.Fatalf("expected decode ready and legacy-only prefill not ready: %#v", endpoints)
	}
	if got := capableLoads.Load(); got != 1 {
		t.Errorf("expected one load request to capable worker, got %d", got)
	}
	if got := incapableLoads.Load(); got != 1 {
		t.Errorf("expected one compatibility-probed request to vLLM prefill, got %d", got)
	}
}

func TestLoadLoRAAllowsLegacyFallbackAlongsideCapableVLLMPrefill(t *testing.T) {
	newPrefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer newPrefill.Close()
	oldPrefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("Endpoint 'load_lora' not found in local registry"))
	}))
	defer oldPrefill.Close()
	decode := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer decode.Close()

	model := &v1alpha1.DynamoModel{
		Spec: v1alpha1.DynamoModelSpec{
			ModelName: "test-lora",
			ModelType: "lora",
			Source:    &v1alpha1.ModelSource{URI: "s3://bucket/model"},
		},
	}
	endpoints, err := NewClient().LoadLoRA(context.Background(), []Candidate{
		{Address: decode.URL, PodName: "decode"},
		{
			Address:                        newPrefill.URL,
			PodName:                        "new-prefill",
			KubernetesReady:                true,
			AllowLoRAManagementUnavailable: true,
			LoRAFallbackGroup:              "graph:a",
		},
		{
			Address:                        oldPrefill.URL,
			PodName:                        "old-prefill",
			KubernetesReady:                true,
			AllowLoRAManagementUnavailable: true,
			LoRAFallbackGroup:              "graph:a",
		},
	}, model)

	if err != nil {
		t.Fatalf("a capable prefill card should permit the legacy rolling fallback: %v", err)
	}
	if !endpoints[0].Ready || !endpoints[1].Ready {
		t.Fatalf("expected the direct-serving endpoints to be ready: %#v", endpoints)
	}
	if endpoints[2].Ready || !endpoints[2].LoRAFallbackCovered {
		t.Fatalf("expected legacy prefill to remain not ready but be fallback-covered: %#v", endpoints)
	}
}

func TestLoadLoRADoesNotCoverLegacyPrefillFromAnotherDeployment(t *testing.T) {
	capable := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer capable.Close()
	legacy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("Endpoint 'load_lora' not found in local registry"))
	}))
	defer legacy.Close()

	model := &v1alpha1.DynamoModel{Spec: v1alpha1.DynamoModelSpec{
		ModelName: "test-lora",
		ModelType: "lora",
		Source:    &v1alpha1.ModelSource{URI: "s3://bucket/model"},
	}}
	endpoints, err := NewClient().LoadLoRA(context.Background(), []Candidate{
		{
			Address:                        capable.URL,
			PodName:                        "prefill-a",
			KubernetesReady:                true,
			AllowLoRAManagementUnavailable: true,
			LoRAFallbackGroup:              "graph:a",
		},
		{
			Address:                        legacy.URL,
			PodName:                        "prefill-b",
			KubernetesReady:                true,
			AllowLoRAManagementUnavailable: true,
			LoRAFallbackGroup:              "graph:b",
		},
	}, model)

	if err == nil {
		t.Fatalf("capable prefill in graph A must not cover legacy graph B: %#v", endpoints)
	}
	if len(endpoints) != 2 || !endpoints[0].Ready || endpoints[1].Ready {
		t.Fatalf("expected graph A ready and legacy graph B not ready, got %#v", endpoints)
	}
}

func TestLoadLoRAIncludesCapablePrefill(t *testing.T) {
	var loads atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/loras":
			loads.Add(1)
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	model := &v1alpha1.DynamoModel{
		ObjectMeta: metav1.ObjectMeta{Name: "test-lora", Namespace: "default"},
		Spec: v1alpha1.DynamoModelSpec{
			ModelName: "test-lora",
			ModelType: "lora",
			Source:    &v1alpha1.ModelSource{URI: "s3://bucket/model"},
		},
	}
	endpoints, err := NewClient().LoadLoRA(context.Background(), []Candidate{
		{
			Address:                        server.URL,
			PodName:                        "vllm-prefill",
			KubernetesReady:                true,
			AllowLoRAManagementUnavailable: true,
		},
	}, model)
	if err != nil {
		t.Fatalf("capable prefill worker must remain eligible: %v", err)
	}
	if len(endpoints) != 1 || !endpoints[0].Ready {
		t.Fatalf("expected capable prefill endpoint to be ready, got %#v", endpoints)
	}
	if got := loads.Load(); got != 1 {
		t.Errorf("expected one load request to capable prefill worker, got %d", got)
	}
}

func TestLoadLoRALegacyFallbackRequiresKubernetesReadiness(t *testing.T) {
	var loads atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		loads.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("Endpoint 'load_lora' not found in local registry"))
	}))
	defer server.Close()

	model := &v1alpha1.DynamoModel{
		ObjectMeta: metav1.ObjectMeta{Name: "test-lora", Namespace: "default"},
		Spec: v1alpha1.DynamoModelSpec{
			ModelName: "test-lora",
			ModelType: "lora",
			Source:    &v1alpha1.ModelSource{URI: "s3://bucket/model"},
		},
	}
	endpoints, err := NewClient().LoadLoRA(context.Background(), []Candidate{
		{
			Address:                        server.URL,
			PodName:                        "vllm-prefill",
			KubernetesReady:                false,
			AllowLoRAManagementUnavailable: true,
		},
	}, model)
	if err == nil {
		t.Fatalf("expected a not-ready error, got endpoints=%#v", endpoints)
	}
	if len(endpoints) != 1 || endpoints[0].Ready {
		t.Fatalf("expected the physical prefill endpoint to remain not ready, got %#v", endpoints)
	}
	if got := loads.Load(); got != 1 {
		t.Errorf("expected one compatibility-probed request to vLLM prefill, got %d", got)
	}
}

func TestLoadLoRAManagementUnavailableIsNotAllowedForDecode(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("LoRA management not available: no 'load_lora' handler is registered"))
	}))
	defer server.Close()

	model := &v1alpha1.DynamoModel{
		Spec: v1alpha1.DynamoModelSpec{
			ModelName: "test-lora",
			ModelType: "lora",
			Source:    &v1alpha1.ModelSource{URI: "s3://bucket/model"},
		},
	}
	endpoints, err := NewClient().LoadLoRA(context.Background(), []Candidate{
		{Address: server.URL, PodName: "vllm-decode", KubernetesReady: true},
	}, model)

	if err == nil {
		t.Fatalf("decode must not use the prefill compatibility fallback: %#v", endpoints)
	}
	if len(endpoints) != 1 || endpoints[0].Ready {
		t.Fatalf("expected decode endpoint to remain not ready, got %#v", endpoints)
	}
}

func TestLoadLoRAGenericPrefillFailureIsNotSwallowed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("adapter download failed"))
	}))
	defer server.Close()

	model := &v1alpha1.DynamoModel{
		Spec: v1alpha1.DynamoModelSpec{
			ModelName: "test-lora",
			ModelType: "lora",
			Source:    &v1alpha1.ModelSource{URI: "s3://bucket/model"},
		},
	}
	endpoints, err := NewClient().LoadLoRA(context.Background(), []Candidate{
		{
			Address:                        server.URL,
			PodName:                        "vllm-prefill",
			KubernetesReady:                true,
			AllowLoRAManagementUnavailable: true,
		},
	}, model)

	if err == nil {
		t.Fatalf("generic prefill failures must not be swallowed: %#v", endpoints)
	}
	if len(endpoints) != 1 || endpoints[0].Ready {
		t.Fatalf("expected failed prefill endpoint to remain not ready, got %#v", endpoints)
	}
}

func TestUnloadLoRA(t *testing.T) {
	successServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify HTTP method
		if r.Method != http.MethodDelete {
			t.Errorf("expected DELETE method, got %s", r.Method)
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		// Verify URL path contains model name
		if !strings.Contains(r.URL.Path, "/loras/") {
			t.Errorf("expected URL path to contain /loras/, got %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer successServer.Close()

	failingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify HTTP method even for failing requests
		if r.Method != http.MethodDelete {
			t.Errorf("expected DELETE method, got %s", r.Method)
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer failingServer.Close()

	var notApplicableUnloads atomic.Int32
	unsupportedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		notApplicableUnloads.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("LoRA management not available: no 'unload_lora' handler is registered"))
	}))
	defer unsupportedServer.Close()

	tests := []struct {
		name        string
		candidates  []Candidate
		modelName   string
		expectError bool
	}{
		{
			name:        "empty candidates",
			candidates:  []Candidate{},
			modelName:   "test-model",
			expectError: false,
		},
		{
			name: "single endpoint success",
			candidates: []Candidate{
				{Address: successServer.URL, PodName: "pod-1"},
			},
			modelName:   "test-model",
			expectError: false,
		},
		{
			name: "multiple endpoints success",
			candidates: []Candidate{
				{Address: successServer.URL, PodName: "pod-1"},
				{Address: successServer.URL, PodName: "pod-2"},
			},
			modelName:   "test-model",
			expectError: false,
		},
		{
			name: "partial failure",
			candidates: []Candidate{
				{Address: successServer.URL, PodName: "pod-1"},
				{Address: failingServer.URL, PodName: "pod-2"},
			},
			modelName:   "test-model",
			expectError: true, // workerpool returns error on any failure
		},
		{
			name: "legacy unavailable worker uses cleanup fallback",
			candidates: []Candidate{
				{Address: successServer.URL, PodName: "decode-0"},
				{
					Address:                        unsupportedServer.URL,
					PodName:                        "prefill-0",
					AllowLoRAManagementUnavailable: true,
				},
			},
			modelName:   "test-model",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient()
			ctx := context.Background()

			err := client.UnloadLoRA(ctx, tt.candidates, tt.modelName)

			if tt.expectError && err == nil {
				t.Error("expected error but got none")
			} else if !tt.expectError && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
	if got := notApplicableUnloads.Load(); got != 1 {
		t.Errorf("expected one compatibility-probed unload request to vLLM prefill, got %d", got)
	}
}
