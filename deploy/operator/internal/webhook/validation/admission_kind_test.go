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

package validation

import (
	"context"
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

func TestValidateAdmissionKind(t *testing.T) {
	tests := []struct {
		name        string
		expected    schema.GroupVersionKind
		got         schema.GroupVersionKind
		wantErrText string
	}{
		{
			name:     "allows converted DGD",
			expected: nvidiacomv1alpha1.DynamoGraphDeploymentGVK,
			got:      nvidiacomv1alpha1.DynamoGraphDeploymentGVK,
		},
		{
			name:     "allows converted DCD",
			expected: nvidiacomv1alpha1.DynamoComponentDeploymentGVK,
			got:      nvidiacomv1alpha1.DynamoComponentDeploymentGVK,
		},
		{
			name:     "allows supported DGDR",
			expected: nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK,
			got:      nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK,
		},
		{
			name:        "rejects unconverted DGD",
			expected:    nvidiacomv1alpha1.DynamoGraphDeploymentGVK,
			got:         nvidiacomv1beta1.DynamoGraphDeploymentGVK,
			wantErrText: "admission requires nvidia.com/v1alpha1, Kind=DynamoGraphDeployment, got nvidia.com/v1beta1, Kind=DynamoGraphDeployment",
		},
		{
			name:        "rejects unconverted DCD",
			expected:    nvidiacomv1alpha1.DynamoComponentDeploymentGVK,
			got:         nvidiacomv1beta1.DynamoComponentDeploymentGVK,
			wantErrText: "admission requires nvidia.com/v1alpha1, Kind=DynamoComponentDeployment, got nvidia.com/v1beta1, Kind=DynamoComponentDeployment",
		},
		{
			name:        "rejects unsupported DGDR version",
			expected:    nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK,
			got:         nvidiacomv1alpha1.DynamoGraphDeploymentRequestGVK,
			wantErrText: "admission requires nvidia.com/v1beta1, Kind=DynamoGraphDeploymentRequest, got nvidia.com/v1alpha1, Kind=DynamoGraphDeploymentRequest",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := admission.NewContextWithRequest(context.Background(), admission.Request{
				AdmissionRequest: admissionv1.AdmissionRequest{
					Kind: metav1.GroupVersionKind{
						Group:   tt.got.Group,
						Version: tt.got.Version,
						Kind:    tt.got.Kind,
					},
				},
			})

			err := validateAdmissionKind(ctx, tt.expected)
			if tt.wantErrText == "" {
				if err != nil {
					t.Fatalf("validateAdmissionKind() error = %v", err)
				}
				return
			}
			if err == nil {
				t.Fatal("validateAdmissionKind() expected error but got nil")
			}
			if !strings.Contains(err.Error(), tt.wantErrText) {
				t.Fatalf("validateAdmissionKind() error = %q, want to contain %q", err.Error(), tt.wantErrText)
			}
		})
	}
}

func TestValidateAdmissionKindRejectsMissingAdmissionRequest(t *testing.T) {
	err := validateAdmissionKind(context.Background(), nvidiacomv1alpha1.DynamoGraphDeploymentGVK)
	if err == nil {
		t.Fatal("validateAdmissionKind() expected error but got nil")
	}
	if !strings.Contains(err.Error(), "admission request missing from context") {
		t.Fatalf("validateAdmissionKind() error = %q, want missing request error", err.Error())
	}
}
