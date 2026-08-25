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

package v1beta2

import (
	"encoding/json"
	"reflect"
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestV1Beta1RoundTripUsesEnvelope(t *testing.T) {
	t.Parallel()

	autoApply := false
	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "legacy-search",
			Namespace:   "default",
			Labels:      map[string]string{"owner": "test"},
			Annotations: map[string]string{"example.com/value": "preserved"},
		},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:          "Qwen/Qwen3-32B",
			Backend:        v1beta1.BackendTypeVllm,
			Image:          "nvcr.io/nvidia/dynamo:latest",
			SearchStrategy: v1beta1.SearchStrategyThorough,
			AutoApply:      &autoApply,
		},
		Status: v1beta1.DynamoGraphDeploymentRequestStatus{
			Phase:              v1beta1.DGDRPhaseProfiling,
			ProfilingJobName:   "legacy-search-profiler",
			ObservedGeneration: 7,
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Spec.V1Beta1 == nil || spoke.Status.V1Beta1 == nil {
		t.Fatal("ConvertFrom() did not create complete v1beta1 envelopes")
	}
	if hasNativeV1Beta2Spec(spoke.Spec) || hasNativeV1Beta2Status(spoke.Status) {
		t.Fatal("converted object contains native v1beta2 fields alongside envelopes")
	}

	dst := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	assertDGDRPayloadEqual(t, src, dst)

	dst.Labels["owner"] = "changed"
	if src.Labels["owner"] != "test" {
		t.Fatal("conversion aliased source metadata")
	}
}

func TestV1Beta2RoundTripUsesEnvelope(t *testing.T) {
	t.Parallel()

	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "replay-search", Namespace: "default"},
		Spec: DynamoGraphDeploymentRequestSpec{
			ModelRef:  &ModelReference{Name: "Qwen/Qwen3-32B"},
			Backends:  []Backend{BackendVLLM},
			Image:     "nvcr.io/nvidia/dynamo:latest",
			Hardware:  &HardwareSpec{},
			Workload:  &WorkloadSpec{},
			Objective: &ObjectiveSpec{},
			Search:    &SearchSpec{},
		},
		Status: DynamoGraphDeploymentRequestStatus{
			ObservedGeneration: 3,
			ActiveRunRef:       &corev1.LocalObjectReference{Name: "replay-search-3"},
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if hub.Spec.V1Beta2 == nil || hub.Status.V1Beta2 == nil {
		t.Fatal("ConvertTo() did not create complete v1beta2 envelopes")
	}
	if hasNativeV1Beta1Spec(hub.Spec) || hasNativeV1Beta1Status(hub.Status) {
		t.Fatal("converted object contains native v1beta1 fields alongside envelopes")
	}

	dst := &DynamoGraphDeploymentRequest{}
	if err := dst.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if !reflect.DeepEqual(src.Spec, dst.Spec) {
		t.Fatalf("spec changed across round trip\nwant: %#v\n got: %#v", src.Spec, dst.Spec)
	}
	if !reflect.DeepEqual(src.Status, dst.Status) {
		t.Fatalf("status changed across round trip\nwant: %#v\n got: %#v", src.Status, dst.Status)
	}
}

func TestEnvelopeJSONShape(t *testing.T) {
	t.Parallel()

	spoke := &DynamoGraphDeploymentRequest{
		Spec: DynamoGraphDeploymentRequestSpec{
			V1Beta1: &apiextensionsv1.JSON{Raw: []byte(`{"model":"legacy","future":{"nested":[1,true,"value"]}}`)},
		},
	}

	raw, err := json.Marshal(spoke)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var object map[string]any
	if err := json.Unmarshal(raw, &object); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	spec := object["spec"].(map[string]any)
	envelope := spec["v1beta1"].(map[string]any)
	if envelope["model"] != "legacy" {
		t.Fatalf("spec.v1beta1.model = %v, want legacy", envelope["model"])
	}
	nested := envelope["future"].(map[string]any)["nested"].([]any)
	if !reflect.DeepEqual(nested, []any{float64(1), true, "value"}) {
		t.Fatalf("spec.v1beta1.future.nested = %#v", nested)
	}
	if _, found := envelope["raw"]; found {
		t.Fatal("spec.v1beta1 contains an unexpected raw wrapper")
	}
}

func TestConversionRejectsMixedRepresentations(t *testing.T) {
	t.Parallel()

	t.Run("v1beta2 spec", func(t *testing.T) {
		src := &DynamoGraphDeploymentRequest{
			Spec: DynamoGraphDeploymentRequestSpec{
				ModelRef: &ModelReference{Name: "Qwen/Qwen3-32B"},
				V1Beta1:  &apiextensionsv1.JSON{Raw: []byte(`{"model":"legacy"}`)},
			},
		}
		if err := src.ConvertTo(&v1beta1.DynamoGraphDeploymentRequest{}); err == nil {
			t.Fatal("ConvertTo() accepted native spec fields alongside a v1beta1 envelope")
		}
	})

	t.Run("v1beta2 status", func(t *testing.T) {
		src := &DynamoGraphDeploymentRequest{
			Status: DynamoGraphDeploymentRequestStatus{
				ObservedGeneration: 1,
				V1Beta1:            &apiextensionsv1.JSON{Raw: []byte(`{}`)},
			},
		}
		if err := src.ConvertTo(&v1beta1.DynamoGraphDeploymentRequest{}); err == nil {
			t.Fatal("ConvertTo() accepted native status fields alongside a v1beta1 envelope")
		}
	})

	t.Run("v1beta1 spec", func(t *testing.T) {
		src := &v1beta1.DynamoGraphDeploymentRequest{
			Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
				Model:   "legacy",
				V1Beta2: &apiextensionsv1.JSON{Raw: []byte(`{"modelRef":{"name":"Qwen/Qwen3-32B"}}`)},
			},
		}
		if err := (&DynamoGraphDeploymentRequest{}).ConvertFrom(src); err == nil {
			t.Fatal("ConvertFrom() accepted native spec fields alongside a v1beta2 envelope")
		}
	})

	t.Run("v1beta1 status", func(t *testing.T) {
		src := &v1beta1.DynamoGraphDeploymentRequest{
			Status: v1beta1.DynamoGraphDeploymentRequestStatus{
				Phase:   v1beta1.DGDRPhasePending,
				V1Beta2: &apiextensionsv1.JSON{Raw: []byte(`{}`)},
			},
		}
		if err := (&DynamoGraphDeploymentRequest{}).ConvertFrom(src); err == nil {
			t.Fatal("ConvertFrom() accepted native status fields alongside a v1beta2 envelope")
		}
	})
}

func TestConversionRejectsEmptyEnvelope(t *testing.T) {
	t.Parallel()

	src := &v1beta1.DynamoGraphDeploymentRequest{
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			V1Beta2: &apiextensionsv1.JSON{},
		},
	}
	if err := (&DynamoGraphDeploymentRequest{}).ConvertFrom(src); err == nil {
		t.Fatal("ConvertFrom() accepted an empty v1beta2 envelope")
	}
}

func TestConversionRejectsMalformedEnvelope(t *testing.T) {
	t.Parallel()

	src := &v1beta1.DynamoGraphDeploymentRequest{
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			V1Beta2: &apiextensionsv1.JSON{Raw: []byte(`{"modelRef":`)},
		},
	}
	if err := (&DynamoGraphDeploymentRequest{}).ConvertFrom(src); err == nil {
		t.Fatal("ConvertFrom() accepted malformed JSON in a v1beta2 envelope")
	}
}

func assertDGDRPayloadEqual(t *testing.T, want, got *v1beta1.DynamoGraphDeploymentRequest) {
	t.Helper()
	if !reflect.DeepEqual(want.ObjectMeta, got.ObjectMeta) {
		t.Fatalf("metadata changed across round trip\nwant: %#v\n got: %#v", want.ObjectMeta, got.ObjectMeta)
	}
	if !reflect.DeepEqual(want.Spec, got.Spec) {
		t.Fatalf("spec changed across round trip\nwant: %#v\n got: %#v", want.Spec, got.Spec)
	}
	if !reflect.DeepEqual(want.Status, got.Status) {
		t.Fatalf("status changed across round trip\nwant: %#v\n got: %#v", want.Status, got.Status)
	}
}
