/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
}
