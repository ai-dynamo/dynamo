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

package controller

import (
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"k8s.io/utils/ptr"
)

func TestApplySLADefaults(t *testing.T) {
	tests := []struct {
		name                     string
		spec                     nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec
		expectedOptimizationType nvidiacomv1beta1.OptimizationType
		expectedTTFT             *float64
		expectedITL              *float64
	}{
		{
			name: "nil SLA defaults to optimizationType=throughput",
			spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model: "test-model",
			},
			expectedOptimizationType: nvidiacomv1beta1.OptimizationTypeThroughput,
			expectedTTFT:             nil,
			expectedITL:              nil,
		},
		{
			name: "empty SLA struct defaults to optimizationType=throughput",
			spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model: "test-model",
				SLA:   &nvidiacomv1beta1.SLASpec{},
			},
			expectedOptimizationType: nvidiacomv1beta1.OptimizationTypeThroughput,
			expectedTTFT:             nil,
			expectedITL:              nil,
		},
		{
			name: "user-provided TTFT+ITL preserved, no optimizationType injected",
			spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model: "test-model",
				SLA: &nvidiacomv1beta1.SLASpec{
					TTFT: ptr.To(500.0),
					ITL:  ptr.To(10.0),
				},
			},
			expectedOptimizationType: "",
			expectedTTFT:             ptr.To(500.0),
			expectedITL:              ptr.To(10.0),
		},
		{
			name: "user-provided TTFT only preserved, no defaults injected",
			spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model: "test-model",
				SLA: &nvidiacomv1beta1.SLASpec{
					TTFT: ptr.To(500.0),
				},
			},
			expectedOptimizationType: "",
			expectedTTFT:             ptr.To(500.0),
			expectedITL:              nil,
		},
		{
			name: "user-provided ITL only preserved, no defaults injected",
			spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model: "test-model",
				SLA: &nvidiacomv1beta1.SLASpec{
					ITL: ptr.To(10.0),
				},
			},
			expectedOptimizationType: "",
			expectedTTFT:             nil,
			expectedITL:              ptr.To(10.0),
		},
		{
			name: "optimizationType=latency preserved as-is",
			spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model: "test-model",
				SLA: &nvidiacomv1beta1.SLASpec{
					OptimizationType: nvidiacomv1beta1.OptimizationTypeLatency,
				},
			},
			expectedOptimizationType: nvidiacomv1beta1.OptimizationTypeLatency,
			expectedTTFT:             nil,
			expectedITL:              nil,
		},
		{
			name: "e2eLatency preserved, no optimizationType injected",
			spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model: "test-model",
				SLA: &nvidiacomv1beta1.SLASpec{
					E2ELatency: ptr.To(5000.0),
				},
			},
			expectedOptimizationType: "",
			expectedTTFT:             nil,
			expectedITL:              nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec := tt.spec // copy
			applySLADefaults(&spec)

			if spec.SLA == nil {
				t.Fatal("applySLADefaults() SLA is nil after defaults applied")
			}

			if spec.SLA.OptimizationType != tt.expectedOptimizationType {
				t.Errorf("applySLADefaults() OptimizationType = %q, want %q",
					spec.SLA.OptimizationType, tt.expectedOptimizationType)
			}

			if tt.expectedTTFT == nil {
				if spec.SLA.TTFT != nil {
					t.Errorf("applySLADefaults() TTFT = %v, want nil", *spec.SLA.TTFT)
				}
			} else {
				if spec.SLA.TTFT == nil {
					t.Errorf("applySLADefaults() TTFT is nil, want %v", *tt.expectedTTFT)
				} else if *spec.SLA.TTFT != *tt.expectedTTFT {
					t.Errorf("applySLADefaults() TTFT = %v, want %v", *spec.SLA.TTFT, *tt.expectedTTFT)
				}
			}

			if tt.expectedITL == nil {
				if spec.SLA.ITL != nil {
					t.Errorf("applySLADefaults() ITL = %v, want nil", *spec.SLA.ITL)
				}
			} else {
				if spec.SLA.ITL == nil {
					t.Errorf("applySLADefaults() ITL is nil, want %v", *tt.expectedITL)
				} else if *spec.SLA.ITL != *tt.expectedITL {
					t.Errorf("applySLADefaults() ITL = %v, want %v", *spec.SLA.ITL, *tt.expectedITL)
				}
			}
		})
	}
}
