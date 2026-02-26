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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDynamoGraphDeploymentRequestValidator_Validate(t *testing.T) {
	vram := int32(81920)
	gpuCount := int32(8)

	// errMsg: if non-empty, an error is expected and each newline-separated substring must appear in it.
	tests := []struct {
		name                string
		request             *nvidiacomv1beta1.DynamoGraphDeploymentRequest
		isClusterWide       bool
		gpuDiscoveryEnabled bool
		errMsg              string
	}{
		{
			name: "valid request",
			request: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
			},
			isClusterWide: true,
		},
		{
			name: "missing image",
			request: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "",
				},
			},
			isClusterWide: true,
			errMsg:        "spec.image is required",
		},
		{
			name: "thorough + auto is invalid",
			request: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:          "llama-3-8b",
					Image:          "profiler:latest",
					Backend:        nvidiacomv1beta1.BackendTypeAuto,
					SearchStrategy: nvidiacomv1beta1.SearchStrategyThorough,
				},
			},
			isClusterWide: true,
			errMsg:        `spec.searchStrategy "thorough" is incompatible with spec.backend "auto"`,
		},
		{
			name: "rapid + auto is valid (default combination)",
			request: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:          "llama-3-8b",
					Image:          "profiler:latest",
					Backend:        nvidiacomv1beta1.BackendTypeAuto,
					SearchStrategy: nvidiacomv1beta1.SearchStrategyRapid,
				},
			},
			isClusterWide: true,
		},
		{
			name: "thorough + vllm is valid",
			request: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:          "llama-3-8b",
					Image:          "profiler:latest",
					Backend:        nvidiacomv1beta1.BackendTypeVllm,
					SearchStrategy: nvidiacomv1beta1.SearchStrategyThorough,
				},
			},
			isClusterWide: true,
		},
		{
			name: "thorough + trtllm is valid",
			request: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:          "llama-3-8b",
					Image:          "profiler:latest",
					Backend:        nvidiacomv1beta1.BackendTypeTrtllm,
					SearchStrategy: nvidiacomv1beta1.SearchStrategyThorough,
				},
			},
			isClusterWide: true,
		},
		{
			name: "thorough + sglang is valid",
			request: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:          "llama-3-8b",
					Image:          "profiler:latest",
					Backend:        nvidiacomv1beta1.BackendTypeSglang,
					SearchStrategy: nvidiacomv1beta1.SearchStrategyThorough,
				},
			},
			isClusterWide: true,
		},
		{
			name: "namespace-scoped operator with manual hardware config (should pass)",
			request: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
					Hardware: &nvidiacomv1beta1.HardwareSpec{
						GPUSKU:         "H100-SXM5-80GB",
						VRAMMB:         &vram,
						NumGPUsPerNode: &gpuCount,
					},
				},
			},
			isClusterWide:       false,
			gpuDiscoveryEnabled: false,
		},
		{
			name: "namespace-scoped operator with GPU discovery enabled (should pass without manual config)",
			request: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
			},
			isClusterWide:       false,
			gpuDiscoveryEnabled: true,
		},
		{
			name: "namespace-scoped operator with GPU discovery disabled and no hardware config (should error)",
			request: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
			},
			isClusterWide:       false,
			gpuDiscoveryEnabled: false,
			errMsg:              "GPU hardware configuration required: GPU discovery is disabled",
		},
		{
			name: "multiple errors (missing image and thorough+auto)",
			request: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:          "llama-3-8b",
					Backend:        nvidiacomv1beta1.BackendTypeAuto,
					SearchStrategy: nvidiacomv1beta1.SearchStrategyThorough,
					Image:          "",
				},
			},
			isClusterWide: true,
			errMsg:        "spec.image is required\nspec.searchStrategy",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoGraphDeploymentRequestValidator(tt.request, tt.isClusterWide, tt.gpuDiscoveryEnabled)
			_, err := validator.Validate()

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
		})
	}
}

func TestDynamoGraphDeploymentRequestValidator_ValidateUpdate(t *testing.T) {
	tests := []struct {
		name         string
		oldRequest   *nvidiacomv1beta1.DynamoGraphDeploymentRequest
		newRequest   *nvidiacomv1beta1.DynamoGraphDeploymentRequest
		wantErr      bool
		errMsg       string
		wantWarnings bool
	}{
		{
			name: "no changes",
			oldRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
			},
			newRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
			},
			wantErr: false,
		},
		{
			name: "changing model name is allowed when not in immutable phase",
			oldRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
			},
			newRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-70b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
			},
			wantErr: false,
		},
		{
			name: "spec change rejected during Profiling phase",
			oldRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
				Status: nvidiacomv1beta1.DynamoGraphDeploymentRequestStatus{
					Phase: nvidiacomv1beta1.DGDRPhaseProfiling,
				},
			},
			newRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-70b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
			},
			wantErr: true,
			errMsg:  "spec updates are forbidden while the resource is in phase",
		},
		{
			name: "spec change rejected during Deploying phase",
			oldRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
				Status: nvidiacomv1beta1.DynamoGraphDeploymentRequestStatus{
					Phase: nvidiacomv1beta1.DGDRPhaseDeploying,
				},
			},
			newRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-70b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
			},
			wantErr: true,
			errMsg:  "spec updates are forbidden while the resource is in phase",
		},
		{
			name: "spec change rejected during Deployed phase",
			oldRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
				Status: nvidiacomv1beta1.DynamoGraphDeploymentRequestStatus{
					Phase: nvidiacomv1beta1.DGDRPhaseDeployed,
				},
			},
			newRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-70b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
			},
			wantErr: true,
			errMsg:  "spec updates are forbidden while the resource is in phase",
		},
		{
			name: "no spec change during immutable phase is allowed",
			oldRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
				Status: nvidiacomv1beta1.DynamoGraphDeploymentRequestStatus{
					Phase: nvidiacomv1beta1.DGDRPhaseProfiling,
				},
			},
			newRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
			},
			wantErr: false,
		},
		{
			name: "spec change allowed during Failed phase",
			oldRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-8b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
				},
				Status: nvidiacomv1beta1.DynamoGraphDeploymentRequestStatus{
					Phase: nvidiacomv1beta1.DGDRPhaseFailed,
				},
			},
			newRequest: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "llama-3-70b",
					Backend: nvidiacomv1beta1.BackendTypeVllm,
					Image:   "profiler:latest",
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
				t.Errorf("ValidateUpdate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr && tt.errMsg != "" {
				if !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("ValidateUpdate() error %q does not contain %q", err.Error(), tt.errMsg)
				}
			}

			if tt.wantWarnings && len(warnings) == 0 {
				t.Errorf("ValidateUpdate() expected warnings but got none")
			}
			if !tt.wantWarnings && len(warnings) > 0 {
				t.Errorf("ValidateUpdate() unexpected warnings: %v", warnings)
			}
		})
	}
}
