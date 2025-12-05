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
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestLoadLoRA_URLConstruction(t *testing.T) {
	tests := []struct {
		name            string
		baseAddress     string
		expectedURLPath string
	}{
		{
			name:            "address without trailing slash",
			baseAddress:     "http://10.0.1.5:9090",
			expectedURLPath: "/engine/v1/load_lora",
		},
		{
			name:            "address with trailing slash",
			baseAddress:     "http://10.0.1.5:9090/",
			expectedURLPath: "/engine/v1/load_lora",
		},
		{
			name:            "address with path",
			baseAddress:     "http://10.0.1.5:9090/api",
			expectedURLPath: "/api/engine/v1/load_lora",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a test server that captures the request
			var capturedPath string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				capturedPath = r.URL.Path
				w.WriteHeader(http.StatusOK)
			}))
			defer server.Close()

			client := NewClient()
			ctx := context.Background()

			// Call loadLoRA with test server URL
			_ = client.loadLoRA(ctx, server.URL+tt.baseAddress[len("http://10.0.1.5:9090"):], "test-model", "s3://bucket/model")

			if capturedPath != tt.expectedURLPath {
				t.Errorf("expected URL path %s, got %s", tt.expectedURLPath, capturedPath)
			}
		})
	}
}

func TestLoadLoRA_RequestBody(t *testing.T) {
	tests := []struct {
		name              string
		modelName         string
		sourceURI         string
		expectedLoraName  string
		expectedSourceURI string
	}{
		{
			name:              "basic lora load",
			modelName:         "my-lora",
			sourceURI:         "s3://bucket/model",
			expectedLoraName:  "my-lora",
			expectedSourceURI: "s3://bucket/model",
		},
		{
			name:              "huggingface lora",
			modelName:         "hf-lora",
			sourceURI:         "hf://org/model@v1.0",
			expectedLoraName:  "hf-lora",
			expectedSourceURI: "hf://org/model@v1.0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a test server that captures the request body
			var capturedBody map[string]interface{}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &capturedBody)

				if r.Header.Get("Content-Type") != "application/json" {
					t.Errorf("expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
				}

				w.WriteHeader(http.StatusOK)
			}))
			defer server.Close()

			client := NewClient()
			ctx := context.Background()

			// Call loadLoRA
			err := client.loadLoRA(ctx, server.URL, tt.modelName, tt.sourceURI)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Verify request body
			if capturedBody["lora_name"] != tt.expectedLoraName {
				t.Errorf("expected lora_name %s, got %v", tt.expectedLoraName, capturedBody["lora_name"])
			}

			source, ok := capturedBody["source"].(map[string]interface{})
			if !ok {
				t.Fatal("expected source to be a map")
			}

			if source["uri"] != tt.expectedSourceURI {
				t.Errorf("expected source URI %s, got %v", tt.expectedSourceURI, source["uri"])
			}
		})
	}
}

func TestLoadLoRA_ResponseHandling(t *testing.T) {
	tests := []struct {
		name          string
		statusCode    int
		responseBody  string
		expectError   bool
		errorContains string
	}{
		{
			name:        "success - 200 OK",
			statusCode:  http.StatusOK,
			expectError: false,
		},
		{
			name:        "success - 201 Created",
			statusCode:  http.StatusCreated,
			expectError: false,
		},
		{
			name:          "failure - 400 Bad Request",
			statusCode:    http.StatusBadRequest,
			responseBody:  "Invalid LoRA",
			expectError:   true,
			errorContains: "400",
		},
		{
			name:          "failure - 404 Not Found",
			statusCode:    http.StatusNotFound,
			responseBody:  "Endpoint not found",
			expectError:   true,
			errorContains: "404",
		},
		{
			name:          "failure - 500 Internal Server Error",
			statusCode:    http.StatusInternalServerError,
			responseBody:  "Server error",
			expectError:   true,
			errorContains: "500",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				if tt.responseBody != "" {
					_, _ = w.Write([]byte(tt.responseBody))
				}
			}))
			defer server.Close()

			client := NewClient()
			ctx := context.Background()

			err := client.loadLoRA(ctx, server.URL, "test-model", "s3://bucket/model")

			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				} else if tt.errorContains != "" && !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("expected error to contain %q, got %v", tt.errorContains, err)
				}
			} else {
				if err != nil {
					t.Errorf("expected no error but got: %v", err)
				}
			}
		})
	}
}

func TestUnloadLoRA_RequestBody(t *testing.T) {
	tests := []struct {
		name             string
		modelName        string
		expectedLoraName string
	}{
		{
			name:             "simple model name",
			modelName:        "my-lora",
			expectedLoraName: "my-lora",
		},
		{
			name:             "model name with special chars",
			modelName:        "my-lora-v1.0",
			expectedLoraName: "my-lora-v1.0",
		},
		{
			name:             "model name with slashes",
			modelName:        "org/model",
			expectedLoraName: "org/model",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a test server that captures the request
			var capturedPath string
			var capturedMethod string
			var capturedBody map[string]interface{}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				capturedPath = r.URL.Path
				capturedMethod = r.Method
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &capturedBody)

				if r.Header.Get("Content-Type") != "application/json" {
					t.Errorf("expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
				}

				w.WriteHeader(http.StatusOK)
			}))
			defer server.Close()

			client := NewClient()
			ctx := context.Background()

			// Call unloadLoRA
			err := client.unloadLoRA(ctx, server.URL, tt.modelName)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Verify HTTP method is POST
			if capturedMethod != "POST" {
				t.Errorf("expected POST method, got %s", capturedMethod)
			}

			// Verify URL path is correct
			expectedPath := "/engine/v1/unload_lora"
			if capturedPath != expectedPath {
				t.Errorf("expected URL path %s, got %s", expectedPath, capturedPath)
			}

			// Verify request body contains correct lora_name
			if capturedBody["lora_name"] != tt.expectedLoraName {
				t.Errorf("expected lora_name %s in body, got %v", tt.expectedLoraName, capturedBody["lora_name"])
			}
		})
	}
}

func TestUnloadLoRA_ResponseHandling(t *testing.T) {
	tests := []struct {
		name          string
		statusCode    int
		responseBody  string
		expectError   bool
		errorContains string
	}{
		{
			name:        "success - 200 OK",
			statusCode:  http.StatusOK,
			expectError: false,
		},
		{
			name:        "success - 204 No Content",
			statusCode:  http.StatusNoContent,
			expectError: false,
		},
		{
			name:          "failure - 404 Not Found",
			statusCode:    http.StatusNotFound,
			responseBody:  "LoRA not found",
			expectError:   true,
			errorContains: "404",
		},
		{
			name:          "failure - 500 Internal Server Error",
			statusCode:    http.StatusInternalServerError,
			responseBody:  "Failed to unload",
			expectError:   true,
			errorContains: "500",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				if tt.responseBody != "" {
					_, _ = w.Write([]byte(tt.responseBody))
				}
			}))
			defer server.Close()

			client := NewClient()
			ctx := context.Background()

			err := client.unloadLoRA(ctx, server.URL, "test-model")

			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				} else if tt.errorContains != "" && !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("expected error to contain %q, got %v", tt.errorContains, err)
				}
			} else {
				if err != nil {
					t.Errorf("expected no error but got: %v", err)
				}
			}
		})
	}
}
