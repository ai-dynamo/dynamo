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

package webhook

import (
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

func requestWithKinds(kind string, requestKind string) admission.Request {
	req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
		Kind: metav1.GroupVersionKind{Group: "nvidia.com", Version: kind, Kind: "DynamoGraphDeploymentRequest"},
	}}
	if requestKind != "" {
		req.RequestKind = &metav1.GroupVersionKind{Group: "nvidia.com", Version: requestKind, Kind: "DynamoGraphDeploymentRequest"}
	}
	return req
}

func TestRequestedVersion(t *testing.T) {
	tests := []struct {
		name string
		req  admission.Request
		want string
	}{
		{"no conversion uses Kind", requestWithKinds("v1beta1", ""), "v1beta1"},
		{"RequestKind wins after conversion", requestWithKinds("v1beta1", "v1beta2"), "v1beta2"},
		{"RequestKind wins over a matching Kind", requestWithKinds("v1beta1", "v1alpha1"), "v1alpha1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := RequestedVersion(tt.req); got != tt.want {
				t.Fatalf("RequestedVersion() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestRequestIsNative(t *testing.T) {
	tests := []struct {
		name    string
		req     admission.Request
		carrier bool
		want    bool
	}{
		{"native v1beta2 create", requestWithKinds("v1beta1", "v1beta2"), true, true},
		{"native v1beta1 create", requestWithKinds("v1beta1", ""), false, true},
		{"native v1alpha1 create", requestWithKinds("v1beta1", "v1alpha1"), false, true},
		{"carrier written through v1beta1", requestWithKinds("v1beta1", ""), true, false},
		{"carrier written through v1alpha1", requestWithKinds("v1beta1", "v1alpha1"), true, false},
		{"v1beta1 payload written through v1beta2", requestWithKinds("v1beta1", "v1beta2"), false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := RequestIsNative(tt.req, tt.carrier); got != tt.want {
				t.Fatalf("RequestIsNative() = %v, want %v", got, tt.want)
			}
		})
	}
}
