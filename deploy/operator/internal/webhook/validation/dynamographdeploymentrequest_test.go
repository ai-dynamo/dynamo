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

package validation

import (
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDynamoGraphDeploymentRequestValidator_Validate(t *testing.T) {
	validConfig := `{"engine": {"backend": "vllm"}, "deployment": {"model": "test-model"}}`
	configWithDifferentBackend := `{"engine": {"backend": "sglang"}}`
	configWithDifferentModel := `{"deployment": {"model": "different-model"}}`
	invalidYAML := `{invalid yaml`

	tests := []struct {
		name            string
		request         *nvidiacomv1alpha1.DynamoGraphDeploymentRequest
		isClusterWide   bool
		wantErr         bool
		errMsg          string
		wantWarnings    bool
		expectedWarning string
		errContains     bool
	}{
		{
			name: "valid request",
			request: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgdr",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "profiler:latest",
						Config: &apiextensionsv1.JSON{
							Raw: []byte(validConfig),
						},
					},
				},
			},
			isClusterWide: true,
			wantErr:       false,
		},
		{
			name: "missing profiler image",
			request: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgdr",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "",
						Config: &apiextensionsv1.JSON{
							Raw: []byte(validConfig),
						},
					},
				},
			},
			isClusterWide: true,
			wantErr:       true,
			errMsg:        "spec.profilingConfig.profilerImage is required",
		},
		{
			name: "missing profiling config",
			request: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgdr",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "profiler:latest",
						Config:        nil,
					},
				},
			},
			isClusterWide: true,
			wantErr:       true,
			errMsg:        "spec.profilingConfig.config is required and must not be empty",
		},
		{
			name: "empty profiling config",
			request: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgdr",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "profiler:latest",
						Config: &apiextensionsv1.JSON{
							Raw: []byte{},
						},
					},
				},
			},
			isClusterWide: true,
			wantErr:       true,
			errMsg:        "spec.profilingConfig.config is required and must not be empty",
		},
		{
			name: "enableGpuDiscovery true for cluster-wide operator",
			request: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgdr",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:              "llama-3-8b",
					Backend:            "vllm",
					EnableGpuDiscovery: true,
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "profiler:latest",
						Config: &apiextensionsv1.JSON{
							Raw: []byte(validConfig),
						},
					},
				},
			},
			isClusterWide: true,
			wantErr:       false,
		},
		{
			name: "enableGpuDiscovery true for namespace-restricted operator",
			request: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgdr",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:              "llama-3-8b",
					Backend:            "vllm",
					EnableGpuDiscovery: true,
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "profiler:latest",
						Config: &apiextensionsv1.JSON{
							Raw: []byte(validConfig),
						},
					},
				},
			},
			isClusterWide: false,
			wantErr:       true,
			errMsg:        "spec.enableGpuDiscovery can only be set to true for cluster-wide operators. Namespace-restricted operators cannot access cluster nodes for GPU discovery. Please set enableGpuDiscovery to false and provide hardware configuration (hardware.min_num_gpus_per_engine, hardware.max_num_gpus_per_engine, hardware.num_gpus_per_node) in spec.profilingConfig.config",
		},
		{
			name: "enableGpuDiscovery false for namespace-restricted operator",
			request: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgdr",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:              "llama-3-8b",
					Backend:            "vllm",
					EnableGpuDiscovery: false,
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "profiler:latest",
						Config: &apiextensionsv1.JSON{
							Raw: []byte(validConfig),
						},
					},
				},
			},
			isClusterWide: false,
			wantErr:       false,
		},
		{
			name: "invalid config YAML",
			request: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgdr",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "profiler:latest",
						Config: &apiextensionsv1.JSON{
							Raw: []byte(invalidYAML),
						},
					},
				},
			},
			isClusterWide: true,
			wantErr:       true,
			errMsg:        "failed to parse spec.profilingConfig.config: error converting YAML to JSON: yaml: line 1: did not find expected ',' or '}'",
		},
		{
			name: "warning for different backend in config",
			request: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgdr",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "profiler:latest",
						Config: &apiextensionsv1.JSON{
							Raw: []byte(configWithDifferentBackend),
						},
					},
				},
			},
			isClusterWide:   true,
			wantErr:         false,
			wantWarnings:    true,
			expectedWarning: "spec.profilingConfig.config.engine.backend (sglang) will be overwritten by spec.backend (vllm)",
		},
		{
			name: "warning for different model in config",
			request: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgdr",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "profiler:latest",
						Config: &apiextensionsv1.JSON{
							Raw: []byte(configWithDifferentModel),
						},
					},
				},
			},
			isClusterWide:   true,
			wantErr:         false,
			wantWarnings:    true,
			expectedWarning: "spec.profilingConfig.config.deployment.model (different-model) will be overwritten by spec.model (llama-3-8b)",
		},
		{
			name: "multiple errors (missing profiler image, missing config, and enableGpuDiscovery for namespace-restricted)",
			request: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgdr",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:              "llama-3-8b",
					Backend:            "vllm",
					EnableGpuDiscovery: true,
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "",
						Config:        nil,
					},
				},
			},
			isClusterWide: false,
			wantErr:       true,
			errMsg:        "spec.profilingConfig.profilerImage is required\nspec.profilingConfig.config is required and must not be empty\nspec.enableGpuDiscovery can only be set to true for cluster-wide operators",
			errContains:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoGraphDeploymentRequestValidator(tt.request, tt.isClusterWide)
			warnings, err := validator.Validate()

			if (err != nil) != tt.wantErr {
				t.Errorf("DynamoGraphDeploymentRequestValidator.Validate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				if tt.errContains {
					// For multiple errors, check that all expected error messages are present
					errStr := err.Error()
					for _, expectedMsg := range strings.Split(tt.errMsg, "\n") {
						if !strings.Contains(errStr, expectedMsg) {
							t.Errorf("DynamoGraphDeploymentRequestValidator.Validate() error message = %v, want to contain %v", errStr, expectedMsg)
						}
					}
				} else {
					if err.Error() != tt.errMsg {
						t.Errorf("DynamoGraphDeploymentRequestValidator.Validate() error message = %v, want %v", err.Error(), tt.errMsg)
					}
				}
			}

			if tt.wantWarnings && len(warnings) == 0 {
				t.Errorf("DynamoGraphDeploymentRequestValidator.Validate() expected warnings but got none")
			}

			if tt.wantWarnings && len(warnings) > 0 && warnings[0] != tt.expectedWarning {
				t.Errorf("DynamoGraphDeploymentRequestValidator.Validate() warning = %v, want %v", warnings[0], tt.expectedWarning)
			}
		})
	}
}

func TestDynamoGraphDeploymentRequestValidator_ValidateUpdate(t *testing.T) {
	validConfig := `{"engine": {"backend": "vllm"}}`

	tests := []struct {
		name         string
		oldRequest   *nvidiacomv1alpha1.DynamoGraphDeploymentRequest
		newRequest   *nvidiacomv1alpha1.DynamoGraphDeploymentRequest
		wantErr      bool
		wantWarnings bool
	}{
		{
			name: "no changes",
			oldRequest: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "profiler:latest",
						Config: &apiextensionsv1.JSON{
							Raw: []byte(validConfig),
						},
					},
				},
			},
			newRequest: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "profiler:latest",
						Config: &apiextensionsv1.JSON{
							Raw: []byte(validConfig),
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "changing model name is allowed",
			oldRequest: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "profiler:latest",
						Config: &apiextensionsv1.JSON{
							Raw: []byte(validConfig),
						},
					},
				},
			},
			newRequest: &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-70b",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "profiler:latest",
						Config: &apiextensionsv1.JSON{
							Raw: []byte(validConfig),
						},
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoGraphDeploymentRequestValidator(tt.newRequest, true)
			warnings, err := validator.ValidateUpdate(tt.oldRequest)

			if (err != nil) != tt.wantErr {
				t.Errorf("DynamoGraphDeploymentRequestValidator.ValidateUpdate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantWarnings && len(warnings) == 0 {
				t.Errorf("DynamoGraphDeploymentRequestValidator.ValidateUpdate() expected warnings but got none")
			}
		})
	}
}
