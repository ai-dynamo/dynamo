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

package defaulting

import (
	"context"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

func TestDGDRDefaulter_defaultImageFor(t *testing.T) {
	tests := []struct {
		name            string
		operatorVersion string
		expectedImage   string
	}{
		{
			name:            "known version produces default image",
			operatorVersion: "1.0.0",
			expectedImage:   "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.0",
		},
		{
			name:            "pre-release version is valid",
			operatorVersion: "1.0.0-rc1",
			expectedImage:   "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.0-rc1",
		},
		{
			name:            "unknown operator version cannot be defaulted",
			operatorVersion: "unknown",
			expectedImage:   "",
		},
		{
			name:            "empty operator version cannot be defaulted",
			operatorVersion: "",
			expectedImage:   "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := NewDGDRDefaulter(tt.operatorVersion)
			got := d.defaultImageFor()
			if got != tt.expectedImage {
				t.Errorf("defaultImageFor() = %q, want %q", got, tt.expectedImage)
			}
		})
	}
}

func makeAdmissionCtx(op admissionv1.Operation) context.Context {
	req := admission.Request{
		AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: op,
		},
	}
	return admission.NewContextWithRequest(context.Background(), req)
}

func TestDGDRDefaulter_Default(t *testing.T) {
	tests := []struct {
		name          string
		version       string
		operation     admissionv1.Operation
		initialImage  string
		expectedImage string
	}{
		{
			name:          "CREATE with empty image defaults to operator version",
			version:       "1.0.0",
			operation:     admissionv1.Create,
			initialImage:  "",
			expectedImage: "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.0",
		},
		{
			name:          "CREATE with preset image is not overwritten",
			version:       "1.0.0",
			operation:     admissionv1.Create,
			initialImage:  "nvcr.io/nvidia/ai-dynamo/my-image:custom",
			expectedImage: "nvcr.io/nvidia/ai-dynamo/my-image:custom",
		},
		{
			name:          "CREATE with unknown operator version leaves image empty",
			version:       "unknown",
			operation:     admissionv1.Create,
			initialImage:  "",
			expectedImage: "",
		},
		{
			name:          "UPDATE does not default image",
			version:       "1.0.0",
			operation:     admissionv1.Update,
			initialImage:  "",
			expectedImage: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := NewDGDRDefaulter(tt.version)
			dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec:       nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{Image: tt.initialImage},
			}
			ctx := makeAdmissionCtx(tt.operation)
			if err := d.Default(ctx, dgdr); err != nil {
				t.Fatalf("Default() unexpected error: %v", err)
			}
			if dgdr.Spec.Image != tt.expectedImage {
				t.Errorf("after Default(): spec.image = %q, want %q", dgdr.Spec.Image, tt.expectedImage)
			}
		})
	}
}
