/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation_test

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	webhookdefaulting "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/defaulting"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

const v2Model = "Qwen/Qwen3-0.6B"

// v2Spec returns a v1beta2 Request spec as a client sends it, before the CRD defaults.
func v2Spec(mutate func(spec map[string]any)) map[string]any {
	spec := map[string]any{"model": map[string]any{"name": v2Model}}
	if mutate != nil {
		mutate(spec)
	}
	return spec
}

// v2DGDR wraps spec in a v1beta2 Request. It is unstructured on purpose: an omitted field,
// an explicit empty object and a null are different requests, and typed Go cannot tell them
// apart.
func v2DGDR(spec map[string]any, labels map[string]any) *unstructured.Unstructured {
	metadata := map[string]any{"name": "test-v1beta2-dgdr", "namespace": "default"}
	if labels != nil {
		metadata["labels"] = labels
	}
	return &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "nvidia.com/v1beta2",
		"kind":       "DynamoGraphDeploymentRequest",
		"metadata":   metadata,
		"spec":       spec,
	}}
}

// mustJSONObject decodes a native AISimulate parameters document.
func mustJSONObject(t *testing.T, document string) map[string]any {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal([]byte(document), &value); err != nil {
		t.Fatalf("decode %q: %v", document, err)
	}
	return value
}

func v2Field(t *testing.T, obj *unstructured.Unstructured, path ...string) (any, bool) {
	t.Helper()
	value, found, err := unstructured.NestedFieldNoCopy(obj.Object, path...)
	if err != nil {
		t.Fatalf("read %v: %v", path, err)
	}
	return value, found
}

func v2Number(t *testing.T, obj *unstructured.Unstructured, path ...string) float64 {
	t.Helper()
	value, found := v2Field(t, obj, path...)
	if !found {
		t.Fatalf("%v is not set", path)
	}
	switch n := value.(type) {
	case int64:
		return float64(n)
	case float64:
		return n
	default:
		t.Fatalf("%v = %#v, want a number", path, value)
		return 0
	}
}

func v2String(t *testing.T, obj *unstructured.Unstructured, path ...string) string {
	t.Helper()
	value, found := v2Field(t, obj, path...)
	if !found {
		t.Fatalf("%v is not set", path)
	}
	text, ok := value.(string)
	if !ok {
		t.Fatalf("%v = %#v, want a string", path, value)
	}
	return text
}

func v2Absent(t *testing.T, obj *unstructured.Unstructured, path ...string) {
	t.Helper()
	if value, found := v2Field(t, obj, path...); found {
		t.Fatalf("%v = %#v, want it absent", path, value)
	}
}

func v2Empty(t *testing.T, obj *unstructured.Unstructured, path ...string) {
	t.Helper()
	value, found := v2Field(t, obj, path...)
	if !found || !reflect.DeepEqual(value, map[string]any{}) {
		t.Fatalf("%v = %#v (found=%v), want an empty object", path, value, found)
	}
}

// v2ExpectSchemaDefaults checks what the CRD supplies when workload and sla are omitted.
func v2ExpectSchemaDefaults(t *testing.T, obj *unstructured.Unstructured) {
	t.Helper()
	if got := v2Number(t, obj, "spec", "workload", "static", "isl"); got != 4000 {
		t.Fatalf("workload.static.isl = %v, want 4000", got)
	}
	if got := v2Number(t, obj, "spec", "workload", "static", "osl"); got != 1000 {
		t.Fatalf("workload.static.osl = %v, want 1000", got)
	}
	ratio, found := v2Field(t, obj, "spec", "workload", "static", "kvLoadRatio")
	list, ok := ratio.([]any)
	if !found || !ok || len(list) != 2 {
		t.Fatalf("workload.static.kvLoadRatio = %#v, want [0, 1]", ratio)
	}
	if got := v2Number(t, obj, "spec", "objective", "sla", "ttftMs"); got != 2000 {
		t.Fatalf("objective.sla.ttftMs = %v, want 2000", got)
	}
	if got := v2Number(t, obj, "spec", "objective", "sla", "itlMs"); got != 30 {
		t.Fatalf("objective.sla.itlMs = %v, want 30", got)
	}
	if got := v2String(t, obj, "spec", "objective", "mode"); got != "Optimize" {
		t.Fatalf("objective.mode = %q, want Optimize", got)
	}
}

// v2ExpectAdmissionDefaults checks what admission supplies from the environment.
func v2ExpectAdmissionDefaults(t *testing.T, obj *unstructured.Unstructured) {
	t.Helper()
	if got := v2String(t, obj, "spec", "objective", "metric"); got != "ThroughputPerGPU" {
		t.Fatalf("objective.metric = %q, want ThroughputPerGPU", got)
	}
	if got := v2String(t, obj, "spec", "aisimulate", "image"); got != admissionAISimulateImage {
		t.Fatalf("aisimulate.image = %q, want %q", got, admissionAISimulateImage)
	}
	if got := v2String(t, obj, "spec", "hardware", "gpu", "sku"); got != "h200_sxm" {
		t.Fatalf("hardware.gpu.sku = %q, want h200_sxm", got)
	}
	if got := v2Number(t, obj, "spec", "hardware", "gpu", "budget"); got != 8 {
		t.Fatalf("hardware.gpu.budget = %v, want 8", got)
	}
}

func TestDynamoGraphDeploymentRequestV1beta2Admission(t *testing.T) {
	discovered := webhookdefaulting.DiscoveredHardware{SKU: "h200_sxm", GPUs: 8}
	nativeWorkload := `{"workload":{"isl":512,"osl":128,"concurrency":8,"request_count":100}}`

	tests := []struct {
		name       string
		request    *unstructured.Unstructured
		oldRequest *unstructured.Unstructured
		hardware   webhookdefaulting.DiscoveredHardware
		hardwareEr error
		wantErrs   []string
		check      func(t *testing.T, result *unstructured.Unstructured)
	}{
		// CREATE: the schema owns the unconditional defaults, admission the environment ones.
		{
			name:     "model-only request gets schema and admission defaults",
			request:  v2DGDR(v2Spec(nil), nil),
			hardware: discovered,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				v2ExpectSchemaDefaults(t, result)
				v2ExpectAdmissionDefaults(t, result)
			},
		},
		{
			name: "explicit hardware is kept and discovery is not consulted",
			request: v2DGDR(v2Spec(func(spec map[string]any) {
				spec["hardware"] = map[string]any{"gpu": map[string]any{"sku": "h100_sxm", "budget": int64(4)}}
			}), nil),
			hardwareEr: errors.New("discovery must not run"),
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if got := v2String(t, result, "spec", "hardware", "gpu", "sku"); got != "h100_sxm" {
					t.Fatalf("hardware.gpu.sku = %q, want h100_sxm", got)
				}
				if got := v2Number(t, result, "spec", "hardware", "gpu", "budget"); got != 4 {
					t.Fatalf("hardware.gpu.budget = %v, want 4", got)
				}
			},
		},
		{
			name:       "unresolved hardware is reported on the typed fields",
			request:    v2DGDR(v2Spec(nil), nil),
			hardwareEr: errors.New("no GPUs found"),
			wantErrs: []string{
				"spec.hardware.gpu.sku: Required value: effective hardware is unresolved: set hardware.gpu.sku, or enable GPU discovery",
				"spec.hardware.gpu.budget: Required value: effective hardware is unresolved: set hardware.gpu.budget, or enable GPU discovery",
			},
		},
		{
			name: "explicit empty workload is an opt-out and needs native parameters",
			request: v2DGDR(v2Spec(func(spec map[string]any) {
				spec["workload"] = map[string]any{}
			}), nil),
			hardware: discovered,
			wantErrs: []string{
				"spec.workload.static: Required value: effective workload is empty: set workload.static or aisimulate.parameters.workload",
			},
		},
		{
			name: "explicit empty workload stays empty when native parameters supply it",
			request: v2DGDR(v2Spec(func(spec map[string]any) {
				spec["workload"] = map[string]any{}
				spec["aisimulate"] = map[string]any{"parameters": mustJSONObject(t, nativeWorkload)}
			}), nil),
			hardware: discovered,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				v2Empty(t, result, "spec", "workload")
				v2Absent(t, result, "spec", "workload", "static")
				if got := v2Number(t, result, "spec", "objective", "sla", "ttftMs"); got != 2000 {
					t.Fatalf("objective.sla.ttftMs = %v, want the schema default 2000", got)
				}
				v2ExpectAdmissionDefaults(t, result)
			},
		},
		{
			name: "explicit empty sla is an opt-out and is not refilled",
			request: v2DGDR(v2Spec(func(spec map[string]any) {
				spec["objective"] = map[string]any{"sla": map[string]any{}}
			}), nil),
			hardware: discovered,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				v2Empty(t, result, "spec", "objective", "sla")
				if got := v2Number(t, result, "spec", "workload", "static", "isl"); got != 4000 {
					t.Fatalf("workload.static.isl = %v, want the schema default 4000", got)
				}
			},
		},
		{
			name: "native parameters win over the typed spec without rewriting it",
			request: v2DGDR(v2Spec(func(spec map[string]any) {
				spec["aisimulate"] = map[string]any{"parameters": mustJSONObject(t, `{"workload":{"isl":256}}`)}
			}), nil),
			hardware: discovered,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if got := v2Number(t, result, "spec", "workload", "static", "isl"); got != 4000 {
					t.Fatalf("typed workload.static.isl = %v, want 4000 (the native value applies only to the run)", got)
				}
				if got := v2Number(t, result, "spec", "aisimulate", "parameters", "workload", "isl"); got != 256 {
					t.Fatalf("parameters.workload.isl = %v, want 256", got)
				}
			},
		},
		{
			name: "duplicate backends are rejected",
			request: v2DGDR(v2Spec(func(spec map[string]any) {
				spec["backends"] = []any{"vllm", "vllm"}
			}), nil),
			hardware: discovered,
			wantErrs: []string{`spec.backends[1]: Duplicate value: "vllm"`},
		},

		// UPDATE: defaults run again and are idempotent; any spec change is fully validated.
		{
			name:       "update without a spec change keeps the defaulted spec and generation",
			oldRequest: v2DGDR(v2Spec(nil), nil),
			request:    v2DGDR(v2Spec(nil), map[string]any{"updated": "true"}),
			hardware:   discovered,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if result.GetGeneration() != 1 {
					t.Fatalf("generation = %d, want 1: defaulting must not change an unchanged spec", result.GetGeneration())
				}
				v2ExpectSchemaDefaults(t, result)
				v2ExpectAdmissionDefaults(t, result)
			},
		},
		{
			name:       "update that changes the spec is accepted and bumps the generation",
			oldRequest: v2DGDR(v2Spec(nil), nil),
			request: v2DGDR(v2Spec(func(spec map[string]any) {
				spec["model"] = map[string]any{"name": alternateAdmissionModel}
			}), nil),
			hardware: discovered,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if result.GetGeneration() != 2 {
					t.Fatalf("generation = %d, want 2", result.GetGeneration())
				}
				if got := v2String(t, result, "spec", "model", "name"); got != alternateAdmissionModel {
					t.Fatalf("model.name = %q, want %q", got, alternateAdmissionModel)
				}
			},
		},
		{
			name:       "update to an invalid spec is rejected",
			oldRequest: v2DGDR(v2Spec(nil), nil),
			request: v2DGDR(v2Spec(func(spec map[string]any) {
				spec["backends"] = []any{"sglang", "sglang"}
			}), nil),
			hardware: discovered,
			wantErrs: []string{`spec.backends[1]: Duplicate value: "sglang"`},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			admissionHardware.set(tt.hardware, tt.hardwareEr)
			t.Cleanup(func() { admissionHardware.set(webhookdefaulting.DiscoveredHardware{}, nil) })

			test := admissionTestCase{
				object:            tt.request,
				gates:             features.Gates{GPUDiscovery: true},
				wantWebhookErrors: tt.wantErrs,
			}
			// A nil *Unstructured in the runtime.Object field is not a nil interface, so only
			// set the old object for update cases.
			if tt.oldRequest != nil {
				test.oldObject = tt.oldRequest
			}
			result := runAdmissionTest(t, test)
			if tt.check != nil && len(tt.wantErrs) == 0 {
				tt.check(t, result)
			}
		})
	}
}
