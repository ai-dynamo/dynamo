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

// defaultedSpec is a model-only Request after structural and admission defaults:
// fixed-concurrency traffic, the scalar objective with latency bounds, and discovered hardware.
func defaultedSpec() *v1beta2.DynamoGraphDeploymentRequestSpec {
	metric := v1beta2.ObjectiveMetric("ThroughputPerGPU")
	return &v1beta2.DynamoGraphDeploymentRequestSpec{
		Model:    &v1beta2.ModelSpec{Name: "meta-llama/Llama-3.1-8B-Instruct"},
		Backends: []v1beta2.Backend{v1beta2.BackendVLLM, v1beta2.BackendSGLang, v1beta2.BackendTRTLLM},
		Hardware: &v1beta2.HardwareSpec{GPU: &v1beta2.GPUHardwareSpec{SKU: new("h200_sxm"), Budget: new(int32(8))}},
		Workload: &v1beta2.WorkloadSpec{Static: &v1beta2.StaticWorkloadSpec{
			InputTokens:  new(int32(1024)),
			OutputTokens: new(int32(128)),
			Load:         &v1beta2.StaticLoadSpec{Concurrency: new(int32(10))},
		}},
		Objective: &v1beta2.ObjectiveSpec{
			Mode:   v1beta2.ObjectiveModeOptimize,
			Metric: &metric,
			SLA:    &v1beta2.ObjectiveSLASpec{TTFTMs: new(2000.0), ITLMs: new(30.0)},
		},
		Recommendation: &v1beta2.RecommendationSpec{MaxCandidates: 1},
	}
}

// withSLA restores the model-only latency defaults after a test changes them.
func withSLA(spec *v1beta2.DynamoGraphDeploymentRequestSpec) *v1beta2.DynamoGraphDeploymentRequestSpec {
	spec.Objective.SLA = &v1beta2.ObjectiveSLASpec{TTFTMs: new(2000.0), ITLMs: new(30.0)}
	return spec
}

// withoutSLA models a caller that explicitly opted out with objective.sla: {}.
func withoutSLA(spec *v1beta2.DynamoGraphDeploymentRequestSpec) *v1beta2.DynamoGraphDeploymentRequestSpec {
	spec.Objective.SLA = &v1beta2.ObjectiveSLASpec{}
	return spec
}

// withKVRange replaces the fixed concurrency with a KV-load range search.
func withKVRange(spec *v1beta2.DynamoGraphDeploymentRequestSpec) *v1beta2.DynamoGraphDeploymentRequestSpec {
	spec.Workload.Static = &v1beta2.StaticWorkloadSpec{
		InputTokens:  new(int32(4000)),
		OutputTokens: new(int32(1000)),
		Load: &v1beta2.StaticLoadSpec{KVCapacityFraction: &v1beta2.KVCapacityFractionSpec{
			Range: &v1beta2.KVCapacityFractionRange{Min: 0.1, Max: 1},
		}},
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
	want := `{"engine":{"backend":{"choices":["vllm","sglang","trtllm"]},"hardware":"h200_sxm","model":"meta-llama/Llama-3.1-8B-Instruct","workers":{"aggregated":{},"decode":{},"prefill":{}}},` +
		`"evaluation":{"sla":{"itl_ms":30,"ttft_ms":2000}},` +
		`"optimization":{"constraints":{"max_candidate_gpus":8},"strict_sla":true,"target":"throughput_per_gpu"},` +
		`"traffic":{"load":{"concurrency":10,"type":"concurrency"},"source":{"input_tokens":1024,"output_tokens":128,"type":"synthetic"},"stop":{"requests":1000}}}`
	if got := encode(t, lower(t, defaultedSpec())); got != want {
		t.Fatalf("effective input\n got: %s\nwant: %s", got, want)
	}
}

func TestLowerIsDeterministicAndDoesNotMutateTheSpec(t *testing.T) {
	const overrides = `{"optimizer":{"max_trials":100},"traffic":{"stop":{"requests":5}}}`
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
	spec.Workload.Static = &v1beta2.StaticWorkloadSpec{
		InputTokens:  new(int32(4000)),
		OutputTokens: new(int32(1000)),
		Load:         &v1beta2.StaticLoadSpec{RequestsPerSecond: new(10.0)},
	}
	spec.Objective.Metric = &metric
	spec = withInputOverrides(spec, `{"traffic":{"stop":{"requests":1000}},"optimizer":{"max_trials":100}}`)

	want := `{"engine":{"backend":{"choices":["vllm","sglang"]},"hardware":"h200_sxm","model":"meta-llama/Llama-3.1-8B-Instruct","workers":{"aggregated":{},"decode":{},"prefill":{}}},` +
		`"evaluation":{"sla":{"itl_ms":30,"ttft_ms":2000}},` +
		`"optimization":{"constraints":{"max_candidate_gpus":8},"strict_sla":true,"target":"throughput"},` +
		`"optimizer":{"max_trials":100},` +
		`"traffic":{"load":{"requests_per_second":10,"type":"constant_rate"},"source":{"input_tokens":4000,"output_tokens":1000,"type":"synthetic"},"stop":{"requests":1000}}}`
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

func TestInputOverridesWinOverTheTypedUnderlay(t *testing.T) {
	t.Run("a null deletes a typed field", func(t *testing.T) {
		input := lower(t, withInputOverrides(defaultedSpec(), `{"traffic":{"source":{"input_tokens":null}}}`))
		if nested(t, input, "traffic", "source", "input_tokens") != nil {
			t.Fatalf("inputTokens survived an override null: %s", encode(t, input))
		}
	})
	t.Run("a value overrides a typed field", func(t *testing.T) {
		input := lower(t, withInputOverrides(defaultedSpec(), `{"traffic":{"source":{"input_tokens":8000}},"optimization":{"constraints":{"max_candidate_gpus":16}}}`))
		if nested(t, input, "traffic", "source", "input_tokens") != 8000.0 || nested(t, input, "optimization", "constraints", "max_candidate_gpus") != 16.0 {
			t.Fatalf("input overrides did not win: %s", encode(t, input))
		}
	})
	t.Run("a target overrides the typed metric", func(t *testing.T) {
		input := lower(t, withInputOverrides(defaultedSpec(), `{"optimization":{"target":"ttft"}}`))
		if nested(t, input, "optimization", "target") != "ttft" {
			t.Fatalf("typed metric beat the override target: %s", encode(t, input))
		}
	})
	t.Run("an SLA bound overrides only the bound it names", func(t *testing.T) {
		input := lower(t, withInputOverrides(withSLA(defaultedSpec()), `{"evaluation":{"sla":{"ttft_ms":500}}}`))
		if nested(t, input, "evaluation", "sla", "ttft_ms") != 500.0 || nested(t, input, "evaluation", "sla", "itl_ms") != 30.0 {
			t.Fatalf("unexpected SLA: %s", encode(t, input))
		}
	})
}

func TestRequestCountDefault(t *testing.T) {
	t.Run("derived from the typed underlay", func(t *testing.T) {
		if nested(t, lower(t, defaultedSpec()), "traffic", "stop", "requests") != 1000.0 {
			t.Fatal("traffic.stop.requests not derived")
		}
	})
	t.Run("an override ratio deletes the derived count itself", func(t *testing.T) {
		input := lower(t, withInputOverrides(defaultedSpec(), `{"traffic":{"stop":{"requests_per_load_unit":10,"requests":null}}}`))
		if nested(t, input, "traffic", "stop", "requests") != nil || nested(t, input, "traffic", "stop", "requests_per_load_unit") != 10.0 {
			t.Fatalf("unexpected workload: %s", encode(t, input))
		}
	})
	t.Run("an override null deletes the default", func(t *testing.T) {
		input := lower(t, withInputOverrides(defaultedSpec(), `{"traffic":{"stop":{"requests":null}}}`))
		if nested(t, input, "traffic", "stop", "requests") != nil {
			t.Fatal("request count re-added after an override null")
		}
	})
	t.Run("not added without a workload", func(t *testing.T) {
		spec := defaultedSpec()
		spec.Workload = &v1beta2.WorkloadSpec{}
		if nested(t, lower(t, spec), "traffic") != nil {
			t.Fatal("a workload appeared from nothing")
		}
	})
}

func TestMaxTrialsDefault(t *testing.T) {
	t.Run("added for a KV-load range", func(t *testing.T) {
		if nested(t, lower(t, withKVRange(defaultedSpec())), "optimizer", "max_trials") != 320.0 {
			t.Fatal("no default max_trials for a range search")
		}
	})
	t.Run("not derived for the model-only fixed concurrency", func(t *testing.T) {
		if input := lower(t, defaultedSpec()); nested(t, input, "optimizer") != nil {
			t.Fatalf("unexpected optimizer: %s", encode(t, input))
		}
	})
	t.Run("not added for a pinned load", func(t *testing.T) {
		spec := withKVRange(defaultedSpec())
		spec.Workload.Static.Load.KVCapacityFraction = &v1beta2.KVCapacityFractionSpec{Value: new(0.5)}
		input := lower(t, spec)
		if nested(t, input, "optimizer") != nil {
			t.Fatalf("unexpected optimizer: %s", encode(t, input))
		}
		if nested(t, input, "traffic", "load", "fraction") != 0.5 {
			t.Fatal("a one-value ratio should be a scalar")
		}
	})
	t.Run("override value or null wins", func(t *testing.T) {
		if nested(t, lower(t, withInputOverrides(withKVRange(defaultedSpec()), `{"optimizer":{"max_trials":7}}`)), "optimizer", "max_trials") != 7.0 {
			t.Fatal("override max_trials ignored")
		}
		if nested(t, lower(t, withInputOverrides(withKVRange(defaultedSpec()), `{"optimizer":{"max_trials":null}}`)), "optimizer", "max_trials") != nil {
			t.Fatal("max_trials re-added after an override null")
		}
	})
}

func TestStrictSLADerivation(t *testing.T) {
	t.Run("enabled when the typed SLA has a bound", func(t *testing.T) {
		if nested(t, lower(t, withSLA(defaultedSpec())), "optimization", "strict_sla") != true {
			t.Fatal("strict_sla not derived")
		}
	})
	t.Run("an absent SLA derives nothing", func(t *testing.T) {
		spec := defaultedSpec()
		spec.Objective.SLA = nil
		input := lower(t, spec)
		if nested(t, input, "optimization", "strict_sla") != nil || nested(t, input, "evaluation", "sla") != nil {
			t.Fatalf("model-only input has SLA keys: %s", encode(t, input))
		}
	})
	t.Run("an empty SLA derives nothing", func(t *testing.T) {
		input := lower(t, withoutSLA(defaultedSpec()))
		if nested(t, input, "optimization", "strict_sla") != nil || nested(t, input, "evaluation", "sla") != nil {
			t.Fatalf("an empty SLA produced SLA keys: %s", encode(t, input))
		}
	})
	t.Run("an override bound alone does not derive it", func(t *testing.T) {
		input := lower(t, withInputOverrides(withoutSLA(defaultedSpec()), `{"evaluation":{"sla":{"ttft_ms":500}}}`))
		if nested(t, input, "evaluation", "sla", "ttft_ms") != 500.0 || nested(t, input, "optimization", "strict_sla") != nil {
			t.Fatalf("unexpected optimization input: %s", encode(t, input))
		}
	})
	t.Run("override value is kept", func(t *testing.T) {
		if nested(t, lower(t, withInputOverrides(withSLA(defaultedSpec()), `{"optimization":{"strict_sla":false}}`)), "optimization", "strict_sla") != false {
			t.Fatal("override strict_sla was ignored")
		}
	})
	t.Run("is not revisited after the input override", func(t *testing.T) {
		input := lower(t, withInputOverrides(withSLA(defaultedSpec()), `{"evaluation":{"sla":{"ttft_ms":null,"itl_ms":null}}}`))
		if nested(t, input, "optimization", "strict_sla") != true {
			t.Fatalf("the derived strict_sla changed after override bounds were deleted: %s", encode(t, input))
		}
	})
}

func TestInputOverridesApplyOnceAfterDerivedDefaults(t *testing.T) {
	t.Run("optimizer null removes the complete derived optimizer", func(t *testing.T) {
		input := lower(t, withInputOverrides(withKVRange(defaultedSpec()), `{"optimizer":null}`))
		if nested(t, input, "optimizer") != nil {
			t.Fatalf("optimizer survived an override null: %s", encode(t, input))
		}
	})
	t.Run("max_trials null removes only max_trials", func(t *testing.T) {
		input := lower(t, withInputOverrides(withKVRange(defaultedSpec()), `{"optimizer":{"max_trials":null,"seed":3}}`))
		if nested(t, input, "optimizer", "max_trials") != nil || nested(t, input, "optimizer", "seed") != 3.0 {
			t.Fatalf("unexpected optimizer: %s", encode(t, input))
		}
	})
	t.Run("an override max_trials replaces the derived value", func(t *testing.T) {
		input := lower(t, withInputOverrides(withKVRange(defaultedSpec()), `{"optimizer":{"max_trials":7}}`))
		if nested(t, input, "optimizer", "max_trials") != 7.0 {
			t.Fatalf("unexpected optimizer: %s", encode(t, input))
		}
	})
	t.Run("nothing is derived from values only overrides introduce", func(t *testing.T) {
		input := lower(t, withInputOverrides(withoutSLA(defaultedSpec()), `{"traffic":{"load":{"fraction":{"range":{"min":0.1,"max":1}}}},"evaluation":{"sla":{"ttft_ms":500}}}`))
		if nested(t, input, "optimizer") != nil || nested(t, input, "optimization", "strict_sla") != nil {
			t.Fatalf("a field was derived from override-only values: %s", encode(t, input))
		}
	})
}

func TestUnderlayParetoObjective(t *testing.T) {
	spec := defaultedSpec()
	spec.Objective.Mode = v1beta2.ObjectiveModePareto
	spec.Objective.Metric = nil
	spec.Objective.Metrics = []v1beta2.ObjectiveMetric{"ThroughputPerGPU", "ThroughputPerUser"}
	input := underlay(t, spec)
	if nested(t, input, "optimization", "target") != TargetPareto {
		t.Fatalf("not a Pareto optimization: %v", input["optimization"])
	}

	spec.Objective.Metrics = []v1beta2.ObjectiveMetric{"Throughput", "TTFT"}
	if _, err := Underlay(&spec.NativeSearchInput); err == nil {
		t.Fatal("custom Pareto objectives must fail until the public AISimulate input exposes them")
	}
}

func TestUnderlayOmitsUnsetTypedFields(t *testing.T) {
	spec := &v1beta2.DynamoGraphDeploymentRequestSpec{Model: &v1beta2.ModelSpec{Name: "m"}}
	if got := encode(t, underlay(t, spec)); got != `{"engine":{"model":"m"}}` {
		t.Fatalf("got %s", got)
	}
	spec.Hardware = &v1beta2.HardwareSpec{GPU: &v1beta2.GPUHardwareSpec{SKU: new("")}}
	if got := encode(t, underlay(t, spec)); got != `{"engine":{"model":"m"}}` {
		t.Fatalf("an empty SKU must not lower: %s", got)
	}
}

func TestUnderlayKVCapacityFractionShapes(t *testing.T) {
	spec := defaultedSpec()
	spec.Workload.Static.Load = &v1beta2.StaticLoadSpec{
		KVCapacityFraction: &v1beta2.KVCapacityFractionSpec{Value: new(0.25)},
	}
	if nested(t, underlay(t, spec), "traffic", "load", "fraction") != 0.25 {
		t.Fatal("a fixed fraction must lower to a scalar")
	}
	spec.Workload.Static.Load.KVCapacityFraction = &v1beta2.KVCapacityFractionSpec{
		Range: &v1beta2.KVCapacityFractionRange{Min: 0.1, Max: 0.9},
	}
	if got := nested(t, underlay(t, spec), "traffic", "load", "fraction", "range", "min"); got != 0.1 {
		t.Fatalf("range min = %v, want 0.1", got)
	}
	if got := nested(t, underlay(t, spec), "traffic", "load", "fraction", "range", "max"); got != 0.9 {
		t.Fatalf("range max = %v, want 0.9", got)
	}
}

func TestLowerWithUnderlayKeepsInputOverridesAuthoritative(t *testing.T) {
	spec := withInputOverrides(defaultedSpec(), `{"dgdr_run":{"name":"custom","runtime_image":null}}`)
	input, err := LowerWithUnderlay(&spec.NativeSearchInput, map[string]any{"dgdr_run": map[string]any{
		"name": "default", "runtime_image": "planner:1.2.3", "max_candidates": int64(1),
	}})
	if err != nil {
		t.Fatalf("LowerWithUnderlay: %v", err)
	}
	if nested(t, input, "dgdr_run", "name") != "custom" {
		t.Fatalf("input override did not replace the execution name: %s", encode(t, input))
	}
	if nested(t, input, "dgdr_run", "runtime_image") != nil {
		t.Fatalf("input override did not delete the runtime image: %s", encode(t, input))
	}
	if nested(t, input, "dgdr_run", "max_candidates") != float64(1) {
		t.Fatalf("static execution underlay was lost: %s", encode(t, input))
	}
}

func TestLowerIncludesPublicEngineRoleUnderlay(t *testing.T) {
	workers, found, err := unstructured.NestedMap(lower(t, defaultedSpec()), engineWorkers...)
	if err != nil || !found {
		t.Fatalf("engine.workers: found=%v err=%v", found, err)
	}
	for _, role := range []string{"aggregated", "prefill", "decode"} {
		worker, ok := workers[role].(map[string]any)
		if !ok || len(worker) != 0 {
			t.Errorf("engine.workers.%s = %#v, want an empty object", role, workers[role])
		}
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
		"optimization": map[string]any{"target": nil},
		"evaluation":   map[string]any{"sla": map[string]any{"ttft_ms": 1.0}},
		"traffic":      map[string]any{"source": map[string]any{"input_tokens": 8.0}},
	}
	for path, want := range map[string]bool{
		"optimization.target":             true, // a null still names the key
		"evaluation.sla":                  true,
		"evaluation.sla.ttft_ms":          true,
		"evaluation.sla.itl_ms":           false,
		"traffic.source.input_tokens":     true,
		"traffic.source.output_tokens":    false,
		"optimizer.max_trials":            false,
		"optimization.target.deeper":      false,
		"traffic.source.input_tokens.foo": false,
	} {
		if got := has(overrides, strings.Split(path, ".")...); got != want {
			t.Errorf("has(%s) = %v, want %v", path, got, want)
		}
	}
	if has(nil, "optimization") {
		t.Error("nothing is named in nil overrides")
	}
}

func TestPublicMetricNames(t *testing.T) {
	for metric, want := range map[v1beta2.ObjectiveMetric]string{
		"Throughput":        "throughput",
		"ThroughputPerGPU":  "throughput_per_gpu",
		"ThroughputPerUser": "throughput_per_user",
		"TTFT":              "ttft",
		"E2ELatency":        "e2e_latency",
		"Goodput":           "goodput",
		"GoodputPerGPU":     "goodput_per_gpu",
	} {
		if got := publicMetric(metric); got != want {
			t.Errorf("publicMetric(%s) = %q, want %q", metric, got, want)
		}
	}
}
