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
// admission: the complete workload, the SLA bounds, the metric, and discovered hardware.
func defaultedSpec() *v1beta2.DynamoGraphDeploymentRequestSpec {
	metric := v1beta2.ObjectiveMetric("ThroughputPerGPU")
	return &v1beta2.DynamoGraphDeploymentRequestSpec{
		Model:    &v1beta2.ModelSpec{Name: "meta-llama/Llama-3.1-8B-Instruct"},
		Backends: []v1beta2.Backend{v1beta2.BackendVLLM, v1beta2.BackendSGLang, v1beta2.BackendTRTLLM},
		Hardware: &v1beta2.HardwareSpec{GPU: &v1beta2.GPUHardwareSpec{SKU: ptrTo("h200_sxm"), Budget: ptrTo(int32(8))}},
		Workload: &v1beta2.WorkloadSpec{Static: &v1beta2.StaticWorkloadSpec{
			ISL: ptrTo(int32(4000)), OSL: ptrTo(int32(1000)), KVLoadRatio: v1beta2.KVLoadRatio{0, 1},
		}},
		Objective: &v1beta2.ObjectiveSpec{
			Mode:   v1beta2.ObjectiveModeOptimize,
			Metric: &metric,
			SLA:    &v1beta2.ObjectiveSLASpec{TTFTMs: ptrTo(2000.0), ITLMs: ptrTo(30.0)},
		},
		Recommendation: &v1beta2.RecommendationSpec{MaxCandidates: 1},
	}
}

func withParameters(spec *v1beta2.DynamoGraphDeploymentRequestSpec, parameters string) *v1beta2.DynamoGraphDeploymentRequestSpec {
	spec.AISimulate = &v1beta2.AISimulateSpec{Parameters: &runtime.RawExtension{Raw: []byte(parameters)}}
	return spec
}

func lower(t *testing.T, spec *v1beta2.DynamoGraphDeploymentRequestSpec) map[string]any {
	t.Helper()
	input, err := Lower(spec)
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	return input
}

func encode(t *testing.T, input map[string]any) string {
	t.Helper()
	raw, err := JSON(input)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
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
	want := `{"goal":{"sla":{"itl_ms":30,"ttft_ms":2000},"strict_sla":true,"target":"throughput_per_gpu"},` +
		`"search_space":{"backend":["vllm","sglang","trtllm"],"gpu_budget":8,"hardware_sku":"h200_sxm","model_name":"meta-llama/Llama-3.1-8B-Instruct"},` +
		`"sweep":{"max_trials":320},` +
		`"workload":{"isl":4000,"kv_load_ratio":[0,1],"osl":1000,"request_count":1000}}`
	if got := encode(t, lower(t, defaultedSpec())); got != want {
		t.Fatalf("effective input\n got: %s\nwant: %s", got, want)
	}
}

func TestLowerIsDeterministicAndDoesNotMutateTheSpec(t *testing.T) {
	const parameters = `{"sweep":{"max_trials":100},"workload":{"request_count":5}}`
	spec := withParameters(defaultedSpec(), parameters)
	first, second := encode(t, lower(t, spec)), encode(t, lower(t, spec))
	if first != second {
		t.Fatalf("not deterministic:\n%s\n%s", first, second)
	}
	if string(spec.AISimulate.Parameters.Raw) != parameters {
		t.Fatalf("parameters mutated: %s", spec.AISimulate.Parameters.Raw)
	}
}

func TestLowerExplicitRequest(t *testing.T) {
	metric := v1beta2.ObjectiveMetric("Throughput")
	spec := defaultedSpec()
	spec.Backends = []v1beta2.Backend{v1beta2.BackendVLLM, v1beta2.BackendSGLang}
	spec.Workload.Static = &v1beta2.StaticWorkloadSpec{ISL: ptrTo(int32(4000)), OSL: ptrTo(int32(1000)), RequestRate: ptrTo(10.0)}
	spec.Objective.Metric = &metric
	spec = withParameters(spec, `{"workload":{"request_count":1000},"sweep":{"max_trials":100}}`)

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
		input := lower(t, withParameters(defaultedSpec(), `{"workload":{"isl":null}}`))
		if nested(t, input, "workload", "isl") != nil {
			t.Fatalf("isl survived a native null: %s", encode(t, input))
		}
	})
	t.Run("a native value overrides a typed field", func(t *testing.T) {
		input := lower(t, withParameters(defaultedSpec(), `{"workload":{"isl":8000},"search_space":{"gpu_budget":16}}`))
		if nested(t, input, "workload", "isl") != 8000.0 || nested(t, input, "search_space", "gpu_budget") != 16.0 {
			t.Fatalf("native values did not win: %s", encode(t, input))
		}
	})
	t.Run("a native target overrides the typed metric", func(t *testing.T) {
		input := lower(t, withParameters(defaultedSpec(), `{"goal":{"target":"ttft"}}`))
		if nested(t, input, "goal", "target") != "ttft" {
			t.Fatalf("typed metric beat the native target: %s", encode(t, input))
		}
	})
	t.Run("a native SLA bound overrides only the bound it names", func(t *testing.T) {
		input := lower(t, withParameters(defaultedSpec(), `{"goal":{"sla":{"ttft_ms":500}}}`))
		if nested(t, input, "goal", "sla", "ttft_ms") != 500.0 || nested(t, input, "goal", "sla", "itl_ms") != 30.0 {
			t.Fatalf("unexpected SLA: %s", encode(t, input))
		}
	})
}

func TestRequestCountDefault(t *testing.T) {
	t.Run("skipped when native selects num_request_ratio", func(t *testing.T) {
		input := lower(t, withParameters(defaultedSpec(), `{"workload":{"num_request_ratio":10}}`))
		if nested(t, input, "workload", "request_count") != nil {
			t.Fatalf("request_count defaulted next to num_request_ratio: %s", encode(t, input))
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
		input := lower(t, withParameters(defaultedSpec(), `{"workload":{"request_count":null}}`))
		if nested(t, input, "workload", "request_count") != nil {
			t.Fatal("request_count re-added after a native null")
		}
	})
	t.Run("native deleting the ratio restores the default", func(t *testing.T) {
		spec := defaultedSpec()
		spec.Workload.Static.NumRequestRatio = ptrTo(5.0)
		input := lower(t, withParameters(spec, `{"workload":{"num_request_ratio":null}}`))
		if nested(t, input, "workload", "request_count") != 1000.0 {
			t.Fatalf("default not restored: %s", encode(t, input))
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
		if nested(t, lower(t, defaultedSpec()), "sweep", "max_trials") != 320.0 {
			t.Fatal("no default max_trials for a range search")
		}
	})
	t.Run("not added for a pinned load", func(t *testing.T) {
		spec := defaultedSpec()
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
		if nested(t, lower(t, withParameters(defaultedSpec(), `{"sweep":{"max_trials":7}}`)), "sweep", "max_trials") != 7.0 {
			t.Fatal("native max_trials ignored")
		}
		if nested(t, lower(t, withParameters(defaultedSpec(), `{"sweep":{"max_trials":null}}`)), "sweep", "max_trials") != nil {
			t.Fatal("max_trials re-added after a native null")
		}
	})
}

func TestStrictSLADerivation(t *testing.T) {
	t.Run("enabled when the SLA has a bound", func(t *testing.T) {
		if nested(t, lower(t, defaultedSpec()), "goal", "strict_sla") != true {
			t.Fatal("strict_sla not derived")
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
	t.Run("a native bound under an empty SLA still derives it", func(t *testing.T) {
		spec := withParameters(defaultedSpec(), `{"goal":{"sla":{"ttft_ms":500}}}`)
		spec.Objective.SLA = &v1beta2.ObjectiveSLASpec{}
		if nested(t, lower(t, spec), "goal", "strict_sla") != true {
			t.Fatal("strict_sla not derived from the native bound")
		}
	})
	t.Run("native value is kept", func(t *testing.T) {
		if nested(t, lower(t, withParameters(defaultedSpec(), `{"goal":{"strict_sla":false}}`)), "goal", "strict_sla") != false {
			t.Fatal("native strict_sla overridden")
		}
	})
	t.Run("follows the merged SLA", func(t *testing.T) {
		input := lower(t, withParameters(defaultedSpec(), `{"goal":{"sla":{"ttft_ms":null,"itl_ms":null}}}`))
		if nested(t, input, "goal", "strict_sla") != nil {
			t.Fatalf("strict_sla derived with no bounds left: %s", encode(t, input))
		}
	})
}

func TestUnderlayParetoObjective(t *testing.T) {
	spec := defaultedSpec()
	spec.Objective.Mode = v1beta2.ObjectiveModePareto
	spec.Objective.Metric = nil
	spec.Objective.Metrics = []v1beta2.ObjectiveMetric{"ThroughputPerGPU", "ThroughputPerUser", "Unknown"}
	input := Underlay(spec)
	if nested(t, input, "goal", "target") != TargetPareto {
		t.Fatalf("not a pareto goal: %v", input["goal"])
	}
	objectives, _, _ := unstructured.NestedSlice(input, "goal", "pareto_objectives")
	got, _ := json.Marshal(objectives)
	if string(got) != `["throughput_per_gpu","throughput_per_user","Unknown"]` {
		t.Fatalf("pareto_objectives = %s", got)
	}
}

func TestUnderlayOmitsUnsetTypedFields(t *testing.T) {
	spec := &v1beta2.DynamoGraphDeploymentRequestSpec{Model: &v1beta2.ModelSpec{Name: "m"}}
	if got := encode(t, Underlay(spec)); got != `{"search_space":{"model_name":"m"}}` {
		t.Fatalf("got %s", got)
	}
	spec.Hardware = &v1beta2.HardwareSpec{GPU: &v1beta2.GPUHardwareSpec{SKU: ptrTo("")}}
	if got := encode(t, Underlay(spec)); got != `{"search_space":{"model_name":"m"}}` {
		t.Fatalf("an empty SKU must not lower: %s", got)
	}
}

func TestUnderlayKVLoadRatioShapes(t *testing.T) {
	spec := defaultedSpec()
	spec.Workload.Static.KVLoadRatio = v1beta2.KVLoadRatio{0.25}
	if nested(t, Underlay(spec), "workload", "kv_load_ratio") != 0.25 {
		t.Fatal("a single ratio lowers to a scalar")
	}
	spec.Workload.Static.KVLoadRatio = v1beta2.KVLoadRatio{0.1, 0.2, 0.3}
	if ratios, _ := nested(t, Underlay(spec), "workload", "kv_load_ratio").([]any); len(ratios) != 3 {
		t.Fatal("longer lists lower unchanged so validation can report them")
	}
}

func TestParametersMustBeAnObject(t *testing.T) {
	for _, parameters := range []string{`[1]`, `"x"`, `3`, `{`} {
		if _, err := Lower(withParameters(defaultedSpec(), parameters)); err == nil {
			t.Errorf("Lower accepted parameters %s", parameters)
		}
	}
	for _, parameters := range []string{``, `null`, `{}`} {
		if _, err := Lower(withParameters(defaultedSpec(), parameters)); err != nil {
			t.Errorf("Lower rejected parameters %q: %v", parameters, err)
		}
	}
}

func TestHas(t *testing.T) {
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
		if got := Has(parameters, strings.Split(path, ".")...); got != want {
			t.Errorf("Has(%s) = %v, want %v", path, got, want)
		}
	}
	if Has(nil, "goal") {
		t.Error("nothing is named in nil parameters")
	}
}
