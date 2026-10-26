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

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

func ptrTo[T any](v T) *T { return &v }

// defaultedSpec is a model-only Request as it looks after the CRD's structural defaults and
// admission: the fixed-concurrency workload, the scalar metric and discovered hardware. It
// has no SLA bounds and no KV-load range; withSLA and withKVRange add them.
func defaultedSpec() *v1beta2.DynamoGraphDeploymentRequestSpec {
	metric := v1beta2.ObjectiveMetric("Throughput")
	return &v1beta2.DynamoGraphDeploymentRequestSpec{
		Model:    &v1beta2.ModelSpec{Name: "meta-llama/Llama-3.1-8B-Instruct"},
		Backends: []v1beta2.Backend{v1beta2.BackendVLLM, v1beta2.BackendSGLang, v1beta2.BackendTRTLLM},
		Hardware: &v1beta2.HardwareSpec{GPU: &v1beta2.GPUHardwareSpec{SKU: ptrTo("h200_sxm"), Budget: ptrTo(int32(8))}},
		Workload: &v1beta2.WorkloadSpec{Static: &v1beta2.StaticWorkloadSpec{
			ISL: ptrTo(int32(1024)), OSL: ptrTo(int32(128)), Concurrency: ptrTo(int32(10)),
		}},
		Objective:      &v1beta2.ObjectiveSpec{Mode: v1beta2.ObjectiveModeOptimize, Metric: &metric},
		Recommendation: &v1beta2.RecommendationSpec{MaxCandidates: 1},
	}
}

// withSLA adds the latency bounds the earlier defaults used to supply.
func withSLA(spec *v1beta2.DynamoGraphDeploymentRequestSpec) *v1beta2.DynamoGraphDeploymentRequestSpec {
	spec.Objective.SLA = &v1beta2.ObjectiveSLASpec{TTFTMs: ptrTo(2000.0), ITLMs: ptrTo(30.0)}
	return spec
}

// withKVRange replaces the fixed concurrency with a KV-load range search.
func withKVRange(spec *v1beta2.DynamoGraphDeploymentRequestSpec) *v1beta2.DynamoGraphDeploymentRequestSpec {
	spec.Workload.Static = &v1beta2.StaticWorkloadSpec{
		ISL: ptrTo(int32(4000)), OSL: ptrTo(int32(1000)), KVLoadRatio: v1beta2.KVLoadRatio{0, 1},
	}
	return spec
}

func withInputOverrides(spec *v1beta2.DynamoGraphDeploymentRequestSpec, overrides string) *v1beta2.DynamoGraphDeploymentRequestSpec {
	spec.AISimulate = &v1beta2.AISimulateSpec{InputOverrides: &runtime.RawExtension{Raw: []byte(overrides)}}
	return spec
}

func lower(t *testing.T, spec *v1beta2.DynamoGraphDeploymentRequestSpec) map[string]any {
	t.Helper()
	input, err := Lower(&spec.NativeSearchInput)
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	return input
}

func encode(t *testing.T, input map[string]any) string {
	t.Helper()
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func underlay(t *testing.T, spec *v1beta2.DynamoGraphDeploymentRequestSpec) map[string]any {
	t.Helper()
	input, err := Underlay(&spec.NativeSearchInput)
	if err != nil {
		t.Fatalf("Underlay: %v", err)
	}
	return input
}

func nested(t *testing.T, input map[string]any, path ...string) any {
	t.Helper()
	value, _, err := unstructured.NestedFieldNoCopy(input, path...)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func TestLowerModelOnlyRequest(t *testing.T) {
	want := `{"goal":{"target":"throughput"},` +
		`"search_space":{"backend":["vllm","sglang","trtllm"],"gpu_budget":8,"hardware_sku":"h200_sxm","model_name":"meta-llama/Llama-3.1-8B-Instruct"},` +
		`"workload":{"concurrency":10,"isl":1024,"osl":128,"request_count":1000}}`
	if got := encode(t, lower(t, defaultedSpec())); got != want {
		t.Fatalf("effective input\n got: %s\nwant: %s", got, want)
	}
}

func TestLowerIsDeterministicAndDoesNotMutateTheSpec(t *testing.T) {
	const overrides = `{"sweep":{"max_trials":100},"workload":{"request_count":5}}`
	spec := withInputOverrides(defaultedSpec(), overrides)
	first, second := encode(t, lower(t, spec)), encode(t, lower(t, spec))
	if first != second {
		t.Fatalf("not deterministic:\n%s\n%s", first, second)
	}
	if string(spec.AISimulate.InputOverrides.Raw) != overrides {
		t.Fatalf("overrides mutated: %s", spec.AISimulate.InputOverrides.Raw)
	}
}

func TestLowerExplicitRequest(t *testing.T) {
	metric := v1beta2.ObjectiveMetric("Throughput")
	spec := withSLA(defaultedSpec())
	spec.Backends = []v1beta2.Backend{v1beta2.BackendVLLM, v1beta2.BackendSGLang}
	spec.Workload.Static = &v1beta2.StaticWorkloadSpec{ISL: ptrTo(int32(4000)), OSL: ptrTo(int32(1000)), RequestRate: ptrTo(10.0)}
	spec.Objective.Metric = &metric
	spec = withInputOverrides(spec, `{"workload":{"request_count":1000},"sweep":{"max_trials":100}}`)

	want := `{"goal":{"sla":{"itl_ms":30,"ttft_ms":2000},"strict_sla":true,"target":"throughput"},` +
		`"search_space":{"backend":["vllm","sglang"],"gpu_budget":8,"hardware_sku":"h200_sxm","model_name":"meta-llama/Llama-3.1-8B-Instruct"},` +
		`"sweep":{"max_trials":100},` +
		`"workload":{"isl":4000,"osl":1000,"request_count":1000,"request_rate":10}}`
	if got := encode(t, lower(t, spec)); got != want {
		t.Fatalf("effective input\n got: %s\nwant: %s", got, want)
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

func TestNativeValuesWinOverTheTypedUnderlay(t *testing.T) {
	t.Run("a native null deletes a typed field", func(t *testing.T) {
		input := lower(t, withInputOverrides(defaultedSpec(), `{"workload":{"isl":null}}`))
		if nested(t, input, "workload", "isl") != nil {
			t.Fatalf("isl survived a native null: %s", encode(t, input))
		}
	})
	t.Run("a native value overrides a typed field", func(t *testing.T) {
		input := lower(t, withInputOverrides(defaultedSpec(), `{"workload":{"isl":8000},"search_space":{"gpu_budget":16}}`))
		if nested(t, input, "workload", "isl") != 8000.0 || nested(t, input, "search_space", "gpu_budget") != 16.0 {
			t.Fatalf("native values did not win: %s", encode(t, input))
		}
	})
	t.Run("a native target overrides the typed metric", func(t *testing.T) {
		input := lower(t, withInputOverrides(defaultedSpec(), `{"goal":{"target":"ttft"}}`))
		if nested(t, input, "goal", "target") != "ttft" {
			t.Fatalf("typed metric beat the native target: %s", encode(t, input))
		}
	})
	t.Run("a native SLA bound overrides only the bound it names", func(t *testing.T) {
		input := lower(t, withInputOverrides(withSLA(defaultedSpec()), `{"goal":{"sla":{"ttft_ms":500}}}`))
		if nested(t, input, "goal", "sla", "ttft_ms") != 500.0 || nested(t, input, "goal", "sla", "itl_ms") != 30.0 {
			t.Fatalf("unexpected SLA: %s", encode(t, input))
		}
	})
}

func TestRequestCountDefault(t *testing.T) {
	t.Run("derived from the typed underlay", func(t *testing.T) {
		if nested(t, lower(t, defaultedSpec()), "workload", "request_count") != 1000.0 {
			t.Fatal("request_count not derived")
		}
	})
	t.Run("a native ratio deletes the derived count itself", func(t *testing.T) {
		input := lower(t, withInputOverrides(defaultedSpec(), `{"workload":{"num_request_ratio":10,"request_count":null}}`))
		if nested(t, input, "workload", "request_count") != nil || nested(t, input, "workload", "num_request_ratio") != 10.0 {
			t.Fatalf("unexpected workload: %s", encode(t, input))
		}
	})
	t.Run("skipped when typed numRequestRatio is set", func(t *testing.T) {
		spec := defaultedSpec()
		spec.Workload.Static.NumRequestRatio = ptrTo(5.0)
		if nested(t, lower(t, spec), "workload", "request_count") != nil {
			t.Fatal("request_count defaulted next to typed numRequestRatio")
		}
	})
	t.Run("native null deletes the default", func(t *testing.T) {
		input := lower(t, withInputOverrides(defaultedSpec(), `{"workload":{"request_count":null}}`))
		if nested(t, input, "workload", "request_count") != nil {
			t.Fatal("request_count re-added after a native null")
		}
	})
	t.Run("native deleting the typed ratio does not derive a count", func(t *testing.T) {
		spec := defaultedSpec()
		spec.Workload.Static.NumRequestRatio = ptrTo(5.0)
		input := lower(t, withInputOverrides(spec, `{"workload":{"num_request_ratio":null}}`))
		if nested(t, input, "workload", "request_count") != nil {
			t.Fatalf("a count was derived after the native overlay: %s", encode(t, input))
		}
	})
	t.Run("not added without a workload", func(t *testing.T) {
		spec := defaultedSpec()
		spec.Workload = &v1beta2.WorkloadSpec{}
		if nested(t, lower(t, spec), "workload") != nil {
			t.Fatal("a workload appeared from nothing")
		}
	})
}

func TestMaxTrialsDefault(t *testing.T) {
	t.Run("added for a KV-load range", func(t *testing.T) {
		if nested(t, lower(t, withKVRange(defaultedSpec())), "sweep", "max_trials") != 320.0 {
			t.Fatal("no default max_trials for a range search")
		}
	})
	t.Run("not derived for the model-only fixed concurrency", func(t *testing.T) {
		if input := lower(t, defaultedSpec()); nested(t, input, "sweep") != nil {
			t.Fatalf("unexpected sweep: %s", encode(t, input))
		}
	})
	t.Run("not added for a pinned load", func(t *testing.T) {
		spec := withKVRange(defaultedSpec())
		spec.Workload.Static.KVLoadRatio = v1beta2.KVLoadRatio{0.5}
		input := lower(t, spec)
		if nested(t, input, "sweep") != nil {
			t.Fatalf("unexpected sweep: %s", encode(t, input))
		}
		if nested(t, input, "workload", "kv_load_ratio") != 0.5 {
			t.Fatal("a one-value ratio should be a scalar")
		}
	})
	t.Run("native value or null wins", func(t *testing.T) {
		if nested(t, lower(t, withInputOverrides(withKVRange(defaultedSpec()), `{"sweep":{"max_trials":7}}`)), "sweep", "max_trials") != 7.0 {
			t.Fatal("native max_trials ignored")
		}
		if nested(t, lower(t, withInputOverrides(withKVRange(defaultedSpec()), `{"sweep":{"max_trials":null}}`)), "sweep", "max_trials") != nil {
			t.Fatal("max_trials re-added after a native null")
		}
	})
}

func TestStrictSLADerivation(t *testing.T) {
	t.Run("enabled when the typed SLA has a bound", func(t *testing.T) {
		if nested(t, lower(t, withSLA(defaultedSpec())), "goal", "strict_sla") != true {
			t.Fatal("strict_sla not derived")
		}
	})
	t.Run("model-only derives nothing", func(t *testing.T) {
		input := lower(t, defaultedSpec())
		if nested(t, input, "goal", "strict_sla") != nil || nested(t, input, "goal", "sla") != nil {
			t.Fatalf("model-only input has SLA keys: %s", encode(t, input))
		}
	})
	t.Run("an empty SLA derives nothing", func(t *testing.T) {
		spec := defaultedSpec()
		spec.Objective.SLA = &v1beta2.ObjectiveSLASpec{}
		input := lower(t, spec)
		if nested(t, input, "goal", "strict_sla") != nil || nested(t, input, "goal", "sla") != nil {
			t.Fatalf("an empty SLA produced SLA keys: %s", encode(t, input))
		}
	})
	t.Run("a native bound alone does not derive it", func(t *testing.T) {
		input := lower(t, withInputOverrides(defaultedSpec(), `{"goal":{"sla":{"ttft_ms":500}}}`))
		if nested(t, input, "goal", "sla", "ttft_ms") != 500.0 || nested(t, input, "goal", "strict_sla") != nil {
			t.Fatalf("unexpected goal: %s", encode(t, input))
		}
	})
	t.Run("native value is kept", func(t *testing.T) {
		if nested(t, lower(t, withInputOverrides(withSLA(defaultedSpec()), `{"goal":{"strict_sla":false}}`)), "goal", "strict_sla") != false {
			t.Fatal("native strict_sla overridden")
		}
	})
	t.Run("is not revisited after the native overlay", func(t *testing.T) {
		input := lower(t, withInputOverrides(withSLA(defaultedSpec()), `{"goal":{"sla":{"ttft_ms":null,"itl_ms":null}}}`))
		if nested(t, input, "goal", "strict_sla") != true {
			t.Fatalf("the derived strict_sla changed after native bounds were deleted: %s", encode(t, input))
		}
	})
}

func TestInputOverridesApplyOnceAfterDerivedDefaults(t *testing.T) {
	t.Run("sweep null removes the complete derived sweep", func(t *testing.T) {
		input := lower(t, withInputOverrides(withKVRange(defaultedSpec()), `{"sweep":null}`))
		if nested(t, input, "sweep") != nil {
			t.Fatalf("sweep survived a native null: %s", encode(t, input))
		}
	})
	t.Run("max_trials null removes only max_trials", func(t *testing.T) {
		input := lower(t, withInputOverrides(withKVRange(defaultedSpec()), `{"sweep":{"max_trials":null,"seed":3}}`))
		if nested(t, input, "sweep", "max_trials") != nil || nested(t, input, "sweep", "seed") != 3.0 {
			t.Fatalf("unexpected sweep: %s", encode(t, input))
		}
	})
	t.Run("a native max_trials replaces the derived value", func(t *testing.T) {
		input := lower(t, withInputOverrides(withKVRange(defaultedSpec()), `{"sweep":{"max_trials":7}}`))
		if nested(t, input, "sweep", "max_trials") != 7.0 {
			t.Fatalf("unexpected sweep: %s", encode(t, input))
		}
	})
	t.Run("nothing is derived from values only overrides introduce", func(t *testing.T) {
		input := lower(t, withInputOverrides(defaultedSpec(), `{"workload":{"kv_load_ratio":[0,1]},"goal":{"sla":{"ttft_ms":500}}}`))
		if nested(t, input, "sweep") != nil || nested(t, input, "goal", "strict_sla") != nil {
			t.Fatalf("a field was derived from native-only values: %s", encode(t, input))
		}
	})
}

func TestUnderlayParetoObjective(t *testing.T) {
	spec := defaultedSpec()
	spec.Objective.Mode = v1beta2.ObjectiveModePareto
	spec.Objective.Metric = nil
	spec.Objective.Metrics = []v1beta2.ObjectiveMetric{"ThroughputPerGPU", "ThroughputPerUser", "Unknown"}
	input := underlay(t, spec)
	if nested(t, input, "goal", "target") != TargetPareto {
		t.Fatalf("not a pareto goal: %v", input["goal"])
	}
	objectives, _, _ := unstructured.NestedSlice(input, "goal", "pareto_objectives")
	got, _ := json.Marshal(objectives)
	if string(got) != `["throughput_per_gpu","throughput_per_user","unknown"]` {
		t.Fatalf("pareto_objectives = %s", got)
	}
}

func TestUnderlayOmitsUnsetTypedFields(t *testing.T) {
	spec := &v1beta2.DynamoGraphDeploymentRequestSpec{Model: &v1beta2.ModelSpec{Name: "m"}}
	if got := encode(t, underlay(t, spec)); got != `{"search_space":{"model_name":"m"}}` {
		t.Fatalf("got %s", got)
	}
	spec.Hardware = &v1beta2.HardwareSpec{GPU: &v1beta2.GPUHardwareSpec{SKU: ptrTo("")}}
	if got := encode(t, underlay(t, spec)); got != `{"search_space":{"model_name":"m"}}` {
		t.Fatalf("an empty SKU must not lower: %s", got)
	}
}

func TestUnderlayKVLoadRatioShapes(t *testing.T) {
	spec := defaultedSpec()
	spec.Workload.Static.KVLoadRatio = v1beta2.KVLoadRatio{0.25}
	if nested(t, underlay(t, spec), "workload", "kv_load_ratio") != 0.25 {
		t.Fatal("a single ratio lowers to a scalar")
	}
	spec.Workload.Static.KVLoadRatio = v1beta2.KVLoadRatio{0.1, 0.2, 0.3}
	if ratios, _ := nested(t, underlay(t, spec), "workload", "kv_load_ratio").([]any); len(ratios) != 3 {
		t.Fatal("longer lists lower unchanged so validation can report them")
	}
}

func TestInputOverridesMustBeAnObject(t *testing.T) {
	for _, overrides := range []string{`[1]`, `"x"`, `3`, `{`} {
		spec := withInputOverrides(defaultedSpec(), overrides)
		if _, err := Lower(&spec.NativeSearchInput); err == nil {
			t.Errorf("Lower accepted overrides %s", overrides)
		}
	}
	for _, overrides := range []string{``, `null`, `{}`} {
		spec := withInputOverrides(defaultedSpec(), overrides)
		if _, err := Lower(&spec.NativeSearchInput); err != nil {
			t.Errorf("Lower rejected overrides %q: %v", overrides, err)
		}
	}
}

func TestHasPath(t *testing.T) {
	overrides := map[string]any{
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
		if got := has(overrides, strings.Split(path, ".")...); got != want {
			t.Errorf("has(%s) = %v, want %v", path, got, want)
		}
	}
	if has(nil, "goal") {
		t.Error("nothing is named in nil overrides")
	}
}

func TestNativeMetricNames(t *testing.T) {
	for metric, want := range map[v1beta2.ObjectiveMetric]string{
		"Throughput":        "throughput",
		"ThroughputPerGPU":  "throughput_per_gpu",
		"ThroughputPerUser": "throughput_per_user",
		"TTFT":              "ttft",
		"E2ELatency":        "e2e_latency",
		"Goodput":           "goodput",
		"GoodputPerGPU":     "goodput_per_gpu",
	} {
		if got := nativeMetric(metric); got != want {
			t.Errorf("nativeMetric(%s) = %q, want %q", metric, got, want)
		}
	}
}
