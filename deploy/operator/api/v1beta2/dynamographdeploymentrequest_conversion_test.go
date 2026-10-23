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
	"k8s.io/apimachinery/pkg/runtime"
)

func TestV1Beta1RoundTripUsesEnvelope(t *testing.T) {
	t.Parallel()

	t.Log("Create a native v1beta1 request with metadata, spec, and status values.")
	autoApply := false
	src := &v1beta1.DynamoGraphDeploymentRequest{
		Name:        "legacy-search",
		Namespace:   "default",
		Labels:      map[string]string{"owner": "test"},
		Annotations: map[string]string{"example.com/value": "preserved"},
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

	t.Log("Convert the native v1beta1 request to a v1beta2 envelope representation.")
	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Spec.V1Beta1 == nil || spoke.Status.V1Beta1 == nil {
		t.Fatal("ConvertFrom() did not create complete v1beta1 envelopes")
	}
	if hasNativeV1Beta2Spec(spoke.Spec.DynamoGraphDeploymentRequestNativeSpec) ||
		hasNativeV1Beta2Status(spoke.Status.DynamoGraphDeploymentRequestNativeStatus) {
		t.Fatal("converted object contains native v1beta2 fields alongside envelopes")
	}

	t.Log("Convert the envelope representation back to the native v1beta1 request.")
	dst := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	assertDGDRPayloadEqual(t, src, dst)

	t.Log("Mutate converted metadata and verify it does not alias the source.")
	dst.Labels["owner"] = "changed"
	if src.Labels["owner"] != "test" {
		t.Fatal("conversion aliased source metadata")
	}
}

func TestV1Beta2RoundTripUsesEnvelope(t *testing.T) {
	t.Parallel()

	t.Log("Create a native v1beta2 request with spec and status values.")
	src := &DynamoGraphDeploymentRequest{
		Name: "replay-search", Namespace: "default",
		Spec: DynamoGraphDeploymentRequestSpec{
			NativeSearchInput: NativeSearchInput{
				Model:    &ModelSpec{Name: "Qwen/Qwen3-32B"},
				Backends: []Backend{BackendVLLM, BackendSGLang},
				Hardware: &HardwareSpec{
					GPU: &GPUHardwareSpec{
						SKU:    new("H100-SXM5-80GB"),
						Budget: new(int32(8)),
					},
				},
				Workload: &WorkloadSpec{
					Static: &StaticWorkloadSpec{
						ISL:             new(int32(4096)),
						OSL:             new(int32(1024)),
						NumRequestRatio: new(2.0),
						RequestRate:     new(12.5),
					},
				},
				Objective: &ObjectiveSpec{
					Mode:    ObjectiveModePareto,
					Metrics: []ObjectiveMetric{"ThroughputPerGPU", "TTFT"},
					SLA:     &ObjectiveSLASpec{TTFTMs: new(2000.0), ITLMs: new(30.0)},
				},
				AISimulate: &AISimulateSpec{
					Image:           "nvcr.io/nvidia/aisimulate:latest",
					VersionOverride: "1.0.0",
					Parameters:      &runtime.RawExtension{Raw: []byte(`{"search_method":"grid"}`)},
				},
				Recommendation: &RecommendationSpec{MaxCandidates: 3},
			},
		},
		Status: DynamoGraphDeploymentRequestStatus{
			ObservedGeneration: 3,
			ActiveRunRef:       &corev1.LocalObjectReference{Name: "replay-search-3"},
		},
	}

	t.Log("Convert the native v1beta2 request to a v1beta1 envelope representation.")
	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if hub.Spec.V1Beta2 == nil || hub.Status.V1Beta2 == nil {
		t.Fatal("ConvertTo() did not create complete v1beta2 envelopes")
	}
	if hasNativeV1Beta1Spec(hub.Spec.DynamoGraphDeploymentRequestNativeSpec) ||
		hasNativeV1Beta1Status(hub.Status.DynamoGraphDeploymentRequestNativeStatus) {
		t.Fatal("converted object contains native v1beta1 fields alongside envelopes")
	}

	t.Log("Convert the envelope representation back and compare the native payload.")
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

func TestEmptyStatusInheritsNativeV1Beta2Representation(t *testing.T) {
	t.Parallel()

	t.Log("Convert stored native v1beta2 spec with status omitted by the status subresource strategy.")
	hub := &v1beta1.DynamoGraphDeploymentRequest{
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			V1Beta2: &apiextensionsv1.JSON{Raw: []byte(`{"model":{"name":"Qwen/Qwen3-32B"}}`)},
		},
	}
	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	t.Log("Verify empty status stays native instead of becoming a v1beta1 envelope.")
	if spoke.Spec.Model == nil || spoke.Spec.Model.Name != "Qwen/Qwen3-32B" {
		t.Fatalf("native spec.model = %#v, want Qwen/Qwen3-32B", spoke.Spec.Model)
	}
	if spoke.Status.V1Beta1 != nil {
		t.Fatalf("status.v1beta1 = %s, want nil", spoke.Status.V1Beta1.Raw)
	}
	if !reflect.DeepEqual(spoke.Status, DynamoGraphDeploymentRequestStatus{}) {
		t.Fatalf("status = %#v, want empty native v1beta2 status", spoke.Status)
	}
}

func TestEmptyStatusInheritsV1Beta1CarrierRepresentation(t *testing.T) {
	t.Parallel()

	t.Log("Convert a v1beta2 view of native v1beta1 spec with empty status.")
	spoke := &DynamoGraphDeploymentRequest{
		Spec: DynamoGraphDeploymentRequestSpec{
			V1Beta1: &apiextensionsv1.JSON{Raw: []byte(`{"model":"legacy-model"}`)},
		},
	}
	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	t.Log("Verify empty status stays native in the v1beta1 storage representation.")
	if hub.Spec.Model != "legacy-model" {
		t.Fatalf("spec.model = %q, want legacy-model", hub.Spec.Model)
	}
	if hub.Status.V1Beta2 != nil {
		t.Fatalf("status.v1beta2 = %s, want nil", hub.Status.V1Beta2.Raw)
	}
	if !reflect.DeepEqual(hub.Status, v1beta1.DynamoGraphDeploymentRequestStatus{}) {
		t.Fatalf("status = %#v, want empty native v1beta1 status", hub.Status)
	}
}

func TestEnvelopeJSONShape(t *testing.T) {
	t.Parallel()

	t.Log("Create a v1beta2 object containing nested unknown JSON in its v1beta1 envelope.")
	spoke := &DynamoGraphDeploymentRequest{
		Spec: DynamoGraphDeploymentRequestSpec{
			V1Beta1: &apiextensionsv1.JSON{Raw: []byte(`{"model":"legacy","future":{"nested":[1,true,"value"]}}`)},
		},
	}

	t.Log("Marshal the public object and decode its wire representation.")
	raw, err := json.Marshal(spoke)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var object map[string]any
	if err := json.Unmarshal(raw, &object); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	t.Log("Verify the envelope is a direct JSON object with nested values and no raw wrapper.")
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

	tests := []struct {
		name  string
		spoke *DynamoGraphDeploymentRequest
		hub   *v1beta1.DynamoGraphDeploymentRequest
	}{
		{
			name: "v1beta2 spec",
			spoke: &DynamoGraphDeploymentRequest{
				Spec: DynamoGraphDeploymentRequestSpec{
					NativeSearchInput: NativeSearchInput{
						Model: &ModelSpec{Name: "Qwen/Qwen3-32B"},
					},
					V1Beta1: &apiextensionsv1.JSON{Raw: []byte(`{"model":"legacy"}`)},
				},
			},
		},
		{
			name: "v1beta2 status",
			spoke: &DynamoGraphDeploymentRequest{
				Status: DynamoGraphDeploymentRequestStatus{
					ObservedGeneration: 1,
					V1Beta1:            &apiextensionsv1.JSON{Raw: []byte(`{}`)},
				},
			},
		},
		{
			name: "v1beta1 spec",
			hub: &v1beta1.DynamoGraphDeploymentRequest{
				Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
					Model:   "legacy",
					V1Beta2: &apiextensionsv1.JSON{Raw: []byte(`{"modelRef":{"name":"Qwen/Qwen3-32B"}}`)},
				},
			},
		},
		{
			name: "v1beta1 status",
			hub: &v1beta1.DynamoGraphDeploymentRequest{
				Status: v1beta1.DynamoGraphDeploymentRequestStatus{
					Phase:   v1beta1.DGDRPhasePending,
					V1Beta2: &apiextensionsv1.JSON{Raw: []byte(`{}`)},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Convert an object containing both native fields and a foreign-version envelope.")
			var err error
			if test.spoke != nil {
				err = test.spoke.ConvertTo(&v1beta1.DynamoGraphDeploymentRequest{})
			} else {
				err = (&DynamoGraphDeploymentRequest{}).ConvertFrom(test.hub)
			}
			if err == nil {
				t.Fatal("conversion accepted mixed native and envelope representations")
			}
		})
	}
}

func TestNativeV1Beta2SpecFields(t *testing.T) {
	t.Parallel()

	t.Log("Verify the shared input contains exactly the AISimulate-native fields.")
	nativeType := reflect.TypeFor[NativeSearchInput]()
	gotNative := make([]string, nativeType.NumField())
	for i := range nativeType.NumField() {
		gotNative[i] = nativeType.Field(i).Name
	}
	wantNative := []string{
		"Model",
		"Backends",
		"Hardware",
		"Workload",
		"Objective",
		"AISimulate",
		"Recommendation",
	}
	if !reflect.DeepEqual(gotNative, wantNative) {
		t.Fatalf("native search input fields = %v, want %v", gotNative, wantNative)
	}

	t.Log("Verify Request separates native input from conversion state.")
	nativeSpecType := reflect.TypeFor[DynamoGraphDeploymentRequestNativeSpec]()
	if nativeSpecType.NumField() != 1 || nativeSpecType.Field(0).Name != "NativeSearchInput" {
		t.Fatalf("native request spec does not exclusively embed native search input: %v", nativeSpecType)
	}
	requestType := reflect.TypeFor[DynamoGraphDeploymentRequestSpec]()
	gotRequest := make([]string, requestType.NumField())
	for i := range requestType.NumField() {
		gotRequest[i] = requestType.Field(i).Name
	}
	wantRequest := []string{
		"DynamoGraphDeploymentRequestNativeSpec",
		"V1Beta1",
	}
	if !reflect.DeepEqual(gotRequest, wantRequest) {
		t.Fatalf("request spec fields = %v, want %v", gotRequest, wantRequest)
	}
	runType := reflect.TypeFor[DynamoGraphDeploymentRunSpec]()
	if runType.NumField() != 1 || runType.Field(0).Name != "NativeSearchInput" {
		t.Fatalf("run spec exposes request-only fields: %v", runType)
	}

	t.Log("Verify every native field is recognized and an empty native payload is not.")
	tests := map[string]DynamoGraphDeploymentRequestNativeSpec{
		"model":          {NativeSearchInput: NativeSearchInput{Model: &ModelSpec{}}},
		"backends":       {NativeSearchInput: NativeSearchInput{Backends: []Backend{BackendVLLM}}},
		"hardware":       {NativeSearchInput: NativeSearchInput{Hardware: &HardwareSpec{}}},
		"workload":       {NativeSearchInput: NativeSearchInput{Workload: &WorkloadSpec{}}},
		"objective":      {NativeSearchInput: NativeSearchInput{Objective: &ObjectiveSpec{}}},
		"aisimulate":     {NativeSearchInput: NativeSearchInput{AISimulate: &AISimulateSpec{}}},
		"recommendation": {NativeSearchInput: NativeSearchInput{Recommendation: &RecommendationSpec{}}},
	}
	for name, spec := range tests {
		t.Run(name, func(t *testing.T) {
			if !hasNativeV1Beta2Spec(spec) {
				t.Fatalf("%s field was not recognized as native v1beta2 data", name)
			}
		})
	}
	if hasNativeV1Beta2Spec(DynamoGraphDeploymentRequestNativeSpec{}) {
		t.Fatal("empty spec was recognized as native v1beta2 data")
	}
}

func TestConversionRejectsEmptyEnvelope(t *testing.T) {
	t.Parallel()

	t.Log("Create a v1beta1 hub object containing an empty v1beta2 envelope.")
	src := &v1beta1.DynamoGraphDeploymentRequest{
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			V1Beta2: &apiextensionsv1.JSON{},
		},
	}

	t.Log("Convert the object and require the empty envelope to be rejected.")
	if err := (&DynamoGraphDeploymentRequest{}).ConvertFrom(src); err == nil {
		t.Fatal("ConvertFrom() accepted an empty v1beta2 envelope")
	}
}

func TestConversionRejectsMalformedEnvelope(t *testing.T) {
	t.Parallel()

	t.Log("Create a v1beta1 hub object containing malformed JSON in its v1beta2 envelope.")
	src := &v1beta1.DynamoGraphDeploymentRequest{
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			V1Beta2: &apiextensionsv1.JSON{Raw: []byte(`{"modelRef":`)},
		},
	}

	t.Log("Convert the object and require the malformed envelope to be rejected.")
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
