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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestComputeDGDName(t *testing.T) {
	tests := []struct {
		name     string
		dgdr     *nvidiacomv1beta1.DynamoGraphDeploymentRequest
		expected string
	}{
		{
			name: "no overrides — uses DGDR name with -dgd suffix",
			dgdr: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "my-dgdr"},
			},
			expected: "my-dgdr-dgd",
		},
		{
			name: "overrides.dgd is nil — uses DGDR name with -dgd suffix",
			dgdr: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "my-dgdr"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Overrides: &nvidiacomv1beta1.OverridesSpec{DGD: nil},
				},
			},
			expected: "my-dgdr-dgd",
		},
		{
			name: "overrides.dgd has explicit metadata.name — uses override name",
			dgdr: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "my-dgdr"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Overrides: &nvidiacomv1beta1.OverridesSpec{
						DGD: &runtime.RawExtension{
							Raw: []byte(`{"apiVersion":"nvidia.com/v1alpha1","kind":"DynamoGraphDeployment","metadata":{"name":"explicit-name"}}`),
						},
					},
				},
			},
			expected: "explicit-name",
		},
		{
			name: "overrides.dgd has no metadata.name — falls back to DGDR name with -dgd suffix",
			dgdr: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "my-dgdr"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Overrides: &nvidiacomv1beta1.OverridesSpec{
						DGD: &runtime.RawExtension{
							Raw: []byte(`{"apiVersion":"nvidia.com/v1alpha1","kind":"DynamoGraphDeployment","metadata":{}}`),
						},
					},
				},
			},
			expected: "my-dgdr-dgd",
		},
		{
			name: "two DGDRs with identical specs produce different names",
			dgdr: &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "tc-2-8-parallel-beta"},
			},
			expected: "tc-2-8-parallel-beta-dgd",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := computeDGDName(tt.dgdr)
			if got != tt.expected {
				t.Errorf("computeDGDName() = %q, want %q", got, tt.expected)
			}
		})
	}
}
