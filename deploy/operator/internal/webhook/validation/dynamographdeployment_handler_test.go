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
	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	k8sptr "k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

func TestDynamoGraphDeploymentV1Alpha1Handler_ValidateCreate(t *testing.T) {
	dgd := newAlphaDGDForCompatibilityValidation()
	dgd.Spec.Services["worker"].Replicas = k8sptr.To(int32(-1))

	handler := &dynamoGraphDeploymentV1Alpha1Handler{
		handler: NewDynamoGraphDeploymentHandler(nil, "", false),
	}
	_, err := handler.ValidateCreate(
		dgdAdmissionContext(admissionv1.Create, nvidiacomv1alpha1.DynamoGraphDeploymentGVK),
		dgd,
	)
	if err == nil {
		t.Fatal("ValidateCreate() error = nil, want converted v1beta1 validation error")
	}
	if !strings.Contains(err.Error(), "spec.components[worker].replicas must be non-negative") {
		t.Fatalf("ValidateCreate() error = %q, want v1beta1 component validation error", err)
	}
}

func dgdAdmissionContext(op admissionv1.Operation, gvk schema.GroupVersionKind) context.Context {
	return admission.NewContextWithRequest(context.Background(), admission.Request{
		AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: op,
			Kind: metav1.GroupVersionKind{
				Group:   gvk.Group,
				Version: gvk.Version,
				Kind:    gvk.Kind,
			},
		},
	})
}
