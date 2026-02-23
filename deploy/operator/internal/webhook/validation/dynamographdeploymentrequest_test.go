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
	validConfigWithHardware := `{"engine": {"backend": "vllm"}, "deployment": {"model": "test-model"}, "hardware": {"numGpusPerNode": 8, "gpuModel": "H100-SXM5-80GB", "gpuVramMib": 81920}}`
	minimalConfig := `{"sla": {"ttft": 200.0}}`
	configWithDifferentBackend := `{"engine": {"backend": "sglang"}}`
	configWithDifferentModel := `{"deployment": {"model": "different-model"}}`
	invalidYAML := `{invalid yaml`

	// errMsg: if non-empty, an error is expected and each newline-separated substring must appear in it.
	// expectedWarning: if non-empty, at least one warning must contain this substring.
	tests := []struct {
		name                string
		request             *nvidiacomv1alpha1.DynamoGraphDeploymentRequest
		isClusterWide       bool
		gpuDiscoveryEnabled bool
		errMsg              string
		expectedWarning     string
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
			errMsg:        "spec.profilingConfig.config is required and must not be empty",
		},
		{
			name: "namespace-scoped operator with manual hardware config (should pass)",
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
							Raw: []byte(validConfigWithHardware),
						},
					},
				},
			},
			isClusterWide:       false,
			gpuDiscoveryEnabled: false,
		},
		{
			name: "namespace-scoped operator with GPU discovery enabled (should pass without manual config)",
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
							Raw: []byte(minimalConfig),
						},
					},
				},
			},
			isClusterWide:       false,
			gpuDiscoveryEnabled: true,
		},
		{
			name: "namespace-scoped operator with GPU discovery disabled and no hardware config (should error)",
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
							Raw: []byte(minimalConfig),
						},
					},
				},
			},
			isClusterWide:       false,
			gpuDiscoveryEnabled: false,
			errMsg:              "GPU hardware configuration required: GPU discovery is disabled",
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
			errMsg:        "failed to parse spec.profilingConfig.config",
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
			expectedWarning: "spec.profilingConfig.config.deployment.model (different-model) will be overwritten by spec.model (llama-3-8b)",
		},
		{
			name: "multiple errors (missing profiler image and missing config)",
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
						Config:        nil,
					},
				},
			},
			isClusterWide: false,
			errMsg:        "spec.profilingConfig.profilerImage is required\nspec.profilingConfig.config is required and must not be empty",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoGraphDeploymentRequestValidator(tt.request, tt.isClusterWide, tt.gpuDiscoveryEnabled)
			warnings, err := validator.Validate()

			wantErr := tt.errMsg != ""
			if (err != nil) != wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, wantErr)
				return
			}
			if wantErr {
				for _, msg := range strings.Split(tt.errMsg, "\n") {
					if !strings.Contains(err.Error(), msg) {
						t.Errorf("Validate() error %q does not contain %q", err.Error(), msg)
					}
				}
			}

			wantWarning := tt.expectedWarning != ""
			if wantWarning && len(warnings) == 0 {
				t.Errorf("Validate() expected warning %q but got none", tt.expectedWarning)
			}
			if wantWarning && len(warnings) > 0 && !strings.Contains(warnings[0], tt.expectedWarning) {
				t.Errorf("Validate() warning %q does not contain %q", warnings[0], tt.expectedWarning)
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
			validator := NewDynamoGraphDeploymentRequestValidator(tt.newRequest, true, true)
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
