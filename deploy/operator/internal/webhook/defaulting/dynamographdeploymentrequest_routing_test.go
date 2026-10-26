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

package defaulting

import (
	"bytes"
	"context"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	admissionv1 "k8s.io/api/admission/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// ctxForRequest builds the admission context of a request the API server converted to
// v1beta1. requestVersion is the version the client used; empty means no conversion.
func ctxForRequest(op admissionv1.Operation, requestVersion string) context.Context {
	gvk := nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK
	req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
		Operation: op,
		Kind:      metav1.GroupVersionKind{Group: gvk.Group, Version: gvk.Version, Kind: gvk.Kind},
	}}
	if requestVersion != "" {
		req.RequestKind = &metav1.GroupVersionKind{Group: gvk.Group, Version: requestVersion, Kind: gvk.Kind}
	}
	return admission.NewContextWithRequest(context.Background(), req)
}

const modelOnlyCarrier = `{"model":{"name":"Qwen/Qwen3-32B"}}`

func carrierRequest() *nvidiacomv1beta1.DynamoGraphDeploymentRequest {
	return &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "r", Namespace: "default"},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			V1Beta2: &apiextensionsv1.JSON{Raw: []byte(modelOnlyCarrier)},
		},
	}
}

func TestDGDRDefaulter_RoutesByRequestedVersion(t *testing.T) {
	d := NewDGDRDefaulter("1.6.0", "").WithV1beta2(DGDRV1beta2Options{})

	t.Run("RequestKind decides, not the converted Kind", func(t *testing.T) {
		dgdr := carrierRequest()
		if err := d.Default(ctxForRequest(admissionv1.Create, "v1beta2"), dgdr); err != nil {
			t.Fatalf("Default() error = %v", err)
		}
		if dgdr.Spec.Image != "" {
			t.Fatalf("v1beta1 image defaulting touched a v1beta2 request: %q", dgdr.Spec.Image)
		}
		if !bytes.Contains(dgdr.Spec.V1Beta2.Raw, []byte(`"backends"`)) {
			t.Fatalf("native v1beta2 defaults were not applied: %s", dgdr.Spec.V1Beta2.Raw)
		}
	})

	for _, version := range []string{"", "v1beta1", "v1alpha1"} {
		t.Run("a foreign carrier is never mutated, requested as "+version, func(t *testing.T) {
			for _, op := range []admissionv1.Operation{admissionv1.Create, admissionv1.Update} {
				dgdr := carrierRequest()
				before := append([]byte(nil), dgdr.Spec.V1Beta2.Raw...)
				if err := d.Default(ctxForRequest(op, version), dgdr); err != nil {
					t.Fatalf("Default() error = %v", err)
				}
				if !bytes.Equal(dgdr.Spec.V1Beta2.Raw, before) {
					t.Fatalf("%s: carrier changed from %s to %s", op, before, dgdr.Spec.V1Beta2.Raw)
				}
				if dgdr.Spec.Image != "" {
					t.Fatalf("%s: native v1beta1 defaults reached a foreign carrier", op)
				}
			}
		})
	}

	t.Run("a v1beta1 payload written through v1beta2 is left for validation", func(t *testing.T) {
		dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{Model: "m"}}
		if err := d.Default(ctxForRequest(admissionv1.Create, "v1beta2"), dgdr); err != nil {
			t.Fatalf("Default() error = %v", err)
		}
		if dgdr.Spec.Image != "" {
			t.Fatalf("a foreign v1beta1 payload was defaulted: image %q", dgdr.Spec.Image)
		}
	})

	t.Run("a native v1beta1 request keeps its defaults", func(t *testing.T) {
		dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{Model: "m"}}
		if err := d.Default(ctxForRequest(admissionv1.Create, ""), dgdr); err != nil {
			t.Fatalf("Default() error = %v", err)
		}
		if dgdr.Spec.Image == "" {
			t.Fatal("native v1beta1 image defaulting stopped working")
		}
	})
}
