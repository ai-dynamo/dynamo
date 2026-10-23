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

package aisimulate

import (
	"encoding/json"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func mustParse(t *testing.T, raw string) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// modelOnlyUnderlay is the underlay of a defaulted model-only Request: fixed-concurrency workload,
// scalar target and hardware. It has no SLA bounds and no KV-load range.
func modelOnlyUnderlay(t *testing.T) map[string]any {
	t.Helper()
	return mustParse(t, `{"goal":{"target":"throughput"},`+
		`"search_space":{"backend":["vllm","sglang","trtllm"],"gpu_budget":8,"hardware_sku":"h200_sxm","model_name":"meta-llama/Llama-3.1-8B-Instruct"},`+
		`"workload":{"concurrency":10,"isl":1024,"osl":128}}`)
}

func setNative(t *testing.T, input map[string]any, value any, path ...string) map[string]any {
	t.Helper()
	if err := unstructured.SetNestedField(input, value, path...); err != nil {
		t.Fatal(err)
	}
	return input
}

func slaUnderlay(t *testing.T) map[string]any {
	t.Helper()
	input := modelOnlyUnderlay(t)
	setNative(t, input, 2000.0, "goal", "sla", "ttft_ms")
	return setNative(t, input, 30.0, "goal", "sla", "itl_ms")
}

func kvRangeUnderlay(t *testing.T) map[string]any {
	t.Helper()
	return mustParse(t, `{"goal":{"target":"throughput"},`+
		`"search_space":{"backend":["vllm"],"gpu_budget":8,"hardware_sku":"h200_sxm","model_name":"m"},`+
		`"workload":{"isl":4000,"osl":1000,"kv_load_ratio":[0,1]}}`)
}

func buildInput(t *testing.T, underlay map[string]any, parameters string) map[string]any {
	t.Helper()
	input, err := BuildInput(underlay, []byte(parameters))
	if err != nil {
		t.Fatalf("BuildInput: %v", err)
	}
	return input
}

func encodeInput(t *testing.T, input map[string]any) string {
	t.Helper()
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func nestedValue(t *testing.T, input map[string]any, path ...string) any {
	t.Helper()
	value, _, err := unstructured.NestedFieldNoCopy(input, path...)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func TestBuildInputModelOnly(t *testing.T) {
	want := `{"goal":{"target":"throughput"},` +
		`"search_space":{"backend":["vllm","sglang","trtllm"],"gpu_budget":8,"hardware_sku":"h200_sxm","model_name":"meta-llama/Llama-3.1-8B-Instruct"},` +
		`"workload":{"concurrency":10,"isl":1024,"osl":128,"request_count":1000}}`
	if got := encodeInput(t, buildInput(t, modelOnlyUnderlay(t), "")); got != want {
		t.Fatalf("effective input\n got: %s\nwant: %s", got, want)
	}
}

func TestBuildInputIsDeterministicAndDoesNotMutateItsArguments(t *testing.T) {
	const parameters = `{"sweep":{"max_trials":100},"workload":{"request_count":5}}`
	underlay := modelOnlyUnderlay(t)
	before := encodeInput(t, underlay)
	first, second := encodeInput(t, buildInput(t, underlay, parameters)), encodeInput(t, buildInput(t, underlay, parameters))
	if first != second {
		t.Fatalf("not deterministic:\n%s\n%s", first, second)
	}
	if after := encodeInput(t, underlay); after != before {
		t.Fatalf("the underlay was mutated:\n%s\n%s", before, after)
	}
}

func TestBuildInputExplicitRequest(t *testing.T) {
	underlay := mustParse(t, `{"goal":{"sla":{"itl_ms":30,"ttft_ms":2000},"target":"throughput"},`+
		`"search_space":{"backend":["vllm","sglang"],"gpu_budget":8,"hardware_sku":"h200_sxm","model_name":"meta-llama/Llama-3.1-8B-Instruct"},`+
		`"workload":{"isl":4000,"osl":1000,"request_rate":10}}`)
	want := `{"goal":{"sla":{"itl_ms":30,"ttft_ms":2000},"strict_sla":true,"target":"throughput"},` +
		`"search_space":{"backend":["vllm","sglang"],"gpu_budget":8,"hardware_sku":"h200_sxm","model_name":"meta-llama/Llama-3.1-8B-Instruct"},` +
		`"sweep":{"max_trials":100},` +
		`"workload":{"isl":4000,"osl":1000,"request_count":1000,"request_rate":10}}`
	got := buildInput(t, underlay, `{"workload":{"request_count":1000},"sweep":{"max_trials":100}}`)
	if encoded := encodeInput(t, got); encoded != want {
		t.Fatalf("effective input\n got: %s\nwant: %s", encoded, want)
	}
}

func TestMergePatchSemantics(t *testing.T) {
	cases := []struct{ name, target, patch, want string }{
		{"scalar replaces", `{"a":1}`, `{"a":2}`, `{"a":2}`},
		{"object merges recursively", `{"a":{"b":1,"c":2}}`, `{"a":{"b":3}}`, `{"a":{"b":3,"c":2}}`},
		{"null deletes", `{"a":1,"b":2}`, `{"a":null}`, `{"b":2}`},
		{"array replaces", `{"a":[1,2,3]}`, `{"a":[4]}`, `{"a":[4]}`},
		{"object replaces scalar and drops nulls", `{"a":1}`, `{"a":{"b":null,"c":1}}`, `{"a":{"c":1}}`},
		{"null on absent key is a no-op", `{"a":1}`, `{"z":null}`, `{"a":1}`},
		{"empty patch", `{"a":1}`, `{}`, `{"a":1}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var target, patch any
			if err := json.Unmarshal([]byte(tc.target), &target); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(tc.patch), &patch); err != nil {
				t.Fatal(err)
			}
			got, _ := json.Marshal(mergePatch(target, patch))
			if string(got) != tc.want {
				t.Fatalf("got %s want %s", got, tc.want)
			}
		})
	}
	t.Run("nil patch map", func(t *testing.T) {
		target := map[string]any{"a": 1.0}
		if got := mergePatch(target, map[string]any(nil)); got == nil {
			t.Fatal("a nil patch must keep the target")
		}
	})
}

func TestNativeValuesWinOverTheUnderlay(t *testing.T) {
	t.Run("a native null deletes an underlay field", func(t *testing.T) {
		input := buildInput(t, modelOnlyUnderlay(t), `{"workload":{"isl":null}}`)
		if nestedValue(t, input, "workload", "isl") != nil {
			t.Fatalf("isl survived a native null: %s", encodeInput(t, input))
		}
	})
	t.Run("a native value overrides an underlay field", func(t *testing.T) {
		input := buildInput(t, modelOnlyUnderlay(t), `{"workload":{"isl":8000},"search_space":{"gpu_budget":16}}`)
		if nestedValue(t, input, "workload", "isl") != 8000.0 || nestedValue(t, input, "search_space", "gpu_budget") != 16.0 {
			t.Fatalf("native values did not win: %s", encodeInput(t, input))
		}
	})
	t.Run("a native target overrides the underlay target", func(t *testing.T) {
		input := buildInput(t, modelOnlyUnderlay(t), `{"goal":{"target":"ttft"}}`)
		if nestedValue(t, input, "goal", "target") != "ttft" {
			t.Fatalf("the underlay target beat the native one: %s", encodeInput(t, input))
		}
	})
	t.Run("a native SLA bound overrides only the bound it names", func(t *testing.T) {
		input := buildInput(t, slaUnderlay(t), `{"goal":{"sla":{"ttft_ms":500}}}`)
		if nestedValue(t, input, "goal", "sla", "ttft_ms") != 500.0 || nestedValue(t, input, "goal", "sla", "itl_ms") != 30.0 {
			t.Fatalf("unexpected SLA: %s", encodeInput(t, input))
		}
	})
}

func TestRequestCountDefault(t *testing.T) {
	withRatio := func() map[string]any { return setNative(t, modelOnlyUnderlay(t), 5.0, "workload", "num_request_ratio") }
	t.Run("derived from the underlay", func(t *testing.T) {
		if nestedValue(t, buildInput(t, modelOnlyUnderlay(t), ""), "workload", "request_count") != 1000.0 {
			t.Fatal("request_count not derived")
		}
	})
	t.Run("a native ratio deletes the derived count itself", func(t *testing.T) {
		input := buildInput(t, modelOnlyUnderlay(t), `{"workload":{"num_request_ratio":10,"request_count":null}}`)
		if nestedValue(t, input, "workload", "request_count") != nil || nestedValue(t, input, "workload", "num_request_ratio") != 10.0 {
			t.Fatalf("unexpected workload: %s", encodeInput(t, input))
		}
	})
	t.Run("skipped when the underlay carries a ratio", func(t *testing.T) {
		if nestedValue(t, buildInput(t, withRatio(), ""), "workload", "request_count") != nil {
			t.Fatal("request_count defaulted next to a num_request_ratio")
		}
	})
	t.Run("native null deletes the default", func(t *testing.T) {
		input := buildInput(t, modelOnlyUnderlay(t), `{"workload":{"request_count":null}}`)
		if nestedValue(t, input, "workload", "request_count") != nil {
			t.Fatal("request_count re-added after a native null")
		}
	})
	t.Run("deleting the underlay ratio natively does not derive a count", func(t *testing.T) {
		input := buildInput(t, withRatio(), `{"workload":{"num_request_ratio":null}}`)
		if nestedValue(t, input, "workload", "request_count") != nil {
			t.Fatalf("a count was derived after the native overlay: %s", encodeInput(t, input))
		}
	})
	t.Run("not added without a workload", func(t *testing.T) {
		underlay := modelOnlyUnderlay(t)
		delete(underlay, "workload")
		if nestedValue(t, buildInput(t, underlay, ""), "workload") != nil {
			t.Fatal("a workload appeared from nothing")
		}
	})
}

func TestMaxTrialsDefault(t *testing.T) {
	t.Run("added for a KV-load range", func(t *testing.T) {
		if nestedValue(t, buildInput(t, kvRangeUnderlay(t), ""), "sweep", "max_trials") != 320.0 {
			t.Fatal("no default max_trials for a range search")
		}
	})
	t.Run("not derived for the model-only fixed concurrency", func(t *testing.T) {
		if input := buildInput(t, modelOnlyUnderlay(t), ""); nestedValue(t, input, "sweep") != nil {
			t.Fatalf("unexpected sweep: %s", encodeInput(t, input))
		}
	})
	t.Run("not added for a pinned load", func(t *testing.T) {
		underlay := setNative(t, kvRangeUnderlay(t), 0.5, "workload", "kv_load_ratio")
		input := buildInput(t, underlay, "")
		if nestedValue(t, input, "sweep") != nil {
			t.Fatalf("unexpected sweep: %s", encodeInput(t, input))
		}
		if nestedValue(t, input, "workload", "kv_load_ratio") != 0.5 {
			t.Fatal("a one-value ratio should stay a scalar")
		}
	})
	t.Run("native value or null wins", func(t *testing.T) {
		if nestedValue(t, buildInput(t, kvRangeUnderlay(t), `{"sweep":{"max_trials":7}}`), "sweep", "max_trials") != 7.0 {
			t.Fatal("native max_trials ignored")
		}
		if nestedValue(t, buildInput(t, kvRangeUnderlay(t), `{"sweep":{"max_trials":null}}`), "sweep", "max_trials") != nil {
			t.Fatal("max_trials re-added after a native null")
		}
	})
}

func TestStrictSLADerivation(t *testing.T) {
	t.Run("enabled when the underlay has a bound", func(t *testing.T) {
		if nestedValue(t, buildInput(t, slaUnderlay(t), ""), "goal", "strict_sla") != true {
			t.Fatal("strict_sla not derived")
		}
	})
	t.Run("model-only derives nothing", func(t *testing.T) {
		input := buildInput(t, modelOnlyUnderlay(t), "")
		if nestedValue(t, input, "goal", "strict_sla") != nil || nestedValue(t, input, "goal", "sla") != nil {
			t.Fatalf("model-only input has SLA keys: %s", encodeInput(t, input))
		}
	})
	t.Run("a native bound alone does not derive it", func(t *testing.T) {
		input := buildInput(t, modelOnlyUnderlay(t), `{"goal":{"sla":{"ttft_ms":500}}}`)
		if nestedValue(t, input, "goal", "sla", "ttft_ms") != 500.0 || nestedValue(t, input, "goal", "strict_sla") != nil {
			t.Fatalf("unexpected goal: %s", encodeInput(t, input))
		}
	})
	t.Run("native value is kept", func(t *testing.T) {
		if nestedValue(t, buildInput(t, slaUnderlay(t), `{"goal":{"strict_sla":false}}`), "goal", "strict_sla") != false {
			t.Fatal("native strict_sla overridden")
		}
	})
	t.Run("is not revisited after the native overlay", func(t *testing.T) {
		input := buildInput(t, slaUnderlay(t), `{"goal":{"sla":{"ttft_ms":null,"itl_ms":null}}}`)
		if nestedValue(t, input, "goal", "strict_sla") != true {
			t.Fatalf("the derived strict_sla changed after native bounds were deleted: %s", encodeInput(t, input))
		}
	})
}

func TestParametersApplyOnceAfterDerivedDefaults(t *testing.T) {
	t.Run("sweep null removes the complete derived sweep", func(t *testing.T) {
		input := buildInput(t, kvRangeUnderlay(t), `{"sweep":null}`)
		if nestedValue(t, input, "sweep") != nil {
			t.Fatalf("sweep survived a native null: %s", encodeInput(t, input))
		}
	})
	t.Run("max_trials null removes only max_trials", func(t *testing.T) {
		input := buildInput(t, kvRangeUnderlay(t), `{"sweep":{"max_trials":null,"seed":3}}`)
		if nestedValue(t, input, "sweep", "max_trials") != nil || nestedValue(t, input, "sweep", "seed") != 3.0 {
			t.Fatalf("unexpected sweep: %s", encodeInput(t, input))
		}
	})
	t.Run("nothing is derived from values only parameters introduce", func(t *testing.T) {
		input := buildInput(t, modelOnlyUnderlay(t), `{"workload":{"kv_load_ratio":[0,1]},"goal":{"sla":{"ttft_ms":500}}}`)
		if nestedValue(t, input, "sweep") != nil || nestedValue(t, input, "goal", "strict_sla") != nil {
			t.Fatalf("a field was derived from native-only values: %s", encodeInput(t, input))
		}
	})
}

func TestBuildInputParametersMustBeAnObject(t *testing.T) {
	for _, parameters := range []string{`[1]`, `"x"`, `3`, `{`} {
		if _, err := BuildInput(modelOnlyUnderlay(t), []byte(parameters)); err == nil {
			t.Errorf("BuildInput accepted parameters %s", parameters)
		}
	}
	for _, parameters := range []string{``, `null`, `{}`} {
		if _, err := BuildInput(modelOnlyUnderlay(t), []byte(parameters)); err != nil {
			t.Errorf("BuildInput rejected parameters %q: %v", parameters, err)
		}
	}
	if input, err := BuildInput(nil, nil); err != nil || input == nil {
		t.Errorf("BuildInput(nil, nil) = %v, %v; want an empty input", input, err)
	}
}

func TestDecodeInputReturnsTheStoredObjectUnchanged(t *testing.T) {
	input, err := DecodeInput([]byte(`{"search_space":{"gpu_budget":4},"workload":{"osl":128}}`))
	if err != nil {
		t.Fatalf("DecodeInput: %v", err)
	}
	workload, _ := input["workload"].(map[string]any)
	if _, found := workload["isl"]; found || workload["osl"] != float64(128) {
		t.Fatalf("workload = %v, want exactly the stored keys", workload)
	}
}

func TestDecodeInputRejectsAMissingOrMalformedInput(t *testing.T) {
	for name, raw := range map[string]string{
		"empty":         ``,
		"empty object":  `{}`,
		"not an object": `[1]`,
		"not json":      `{`,
	} {
		if _, err := DecodeInput([]byte(raw)); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
}

func TestHasPath(t *testing.T) {
	parameters := map[string]any{
		"goal":     map[string]any{"target": nil, "sla": map[string]any{"ttft_ms": 1.0}},
		"workload": map[string]any{"isl": 8.0},
	}
	for path, want := range map[string]bool{
		"goal.target":         true, // a null still names the key
		"goal.sla":            true,
		"goal.sla.ttft_ms":    true,
		"goal.sla.itl_ms":     false,
		"workload.isl":        true,
		"workload.osl":        false,
		"sweep.max_trials":    false,
		"goal.target.deeper":  false,
		"workload.isl.deeper": false,
	} {
		if got := has(parameters, strings.Split(path, ".")...); got != want {
			t.Errorf("has(%s) = %v, want %v", path, got, want)
		}
	}
	if has(nil, "goal") {
		t.Error("nothing is named in nil parameters")
	}
}
