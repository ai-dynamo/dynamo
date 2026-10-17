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

package dgdrinput

import (
	"encoding/json"
	"strings"
	"testing"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func ptr[T any](v T) *T { return &v }

// defaulted returns a model-only Request as it looks after the CRD's structural defaults
// and the admission defaults for hardware and the metric.
func defaulted() *v1beta2.DynamoGraphDeploymentRequestSpec {
	metric := v1beta2.ObjectiveMetric("ThroughputPerGPU")
	return &v1beta2.DynamoGraphDeploymentRequestSpec{
		Model:    v1beta2.ModelSpec{Name: "meta-llama/Llama-3.1-8B-Instruct"},
		Backends: []v1beta2.Backend{v1beta2.BackendVLLM, v1beta2.BackendSGLang, v1beta2.BackendTRTLLM},
		Hardware: &v1beta2.HardwareSpec{GPU: &v1beta2.GPUHardwareSpec{SKU: ptr("h200_sxm"), Budget: ptr(int32(8))}},
		Workload: &v1beta2.WorkloadSpec{Static: &v1beta2.StaticWorkloadSpec{
			ISL: ptr(int32(4000)), OSL: ptr(int32(1000)), KVLoadRatio: v1beta2.KVLoadRatio{0, 1},
		}},
		Objective: &v1beta2.ObjectiveSpec{
			Mode:   v1beta2.ObjectiveModeOptimize,
			Metric: &metric,
			SLA:    &v1beta2.ObjectiveSLASpec{TTFTMs: ptr(2000.0), ITLMs: ptr(30.0)},
		},
		Recommendation: &v1beta2.RecommendationSpec{MaxCandidates: 1},
	}
}

func withParams(spec *v1beta2.DynamoGraphDeploymentRequestSpec, params string) *v1beta2.DynamoGraphDeploymentRequestSpec {
	spec.AISimulate = &v1beta2.AISimulateSpec{Parameters: &runtime.RawExtension{Raw: []byte(params)}}
	return spec
}

func build(t *testing.T, spec *v1beta2.DynamoGraphDeploymentRequestSpec) map[string]any {
	t.Helper()
	input, err := Build(spec, DefaultOptions())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	return input
}

func mustJSON(t *testing.T, input map[string]any) string {
	t.Helper()
	raw, err := JSON(input)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func details(errs field.ErrorList) string {
	var out []string
	for _, err := range errs {
		out = append(out, err.Field+": "+err.Detail)
	}
	return strings.Join(out, "\n")
}

func requireError(t *testing.T, errs field.ErrorList, substr string) {
	t.Helper()
	if !strings.Contains(details(errs), substr) {
		t.Fatalf("want an error containing %q, got:\n%s", substr, details(errs))
	}
}

func requireNoError(t *testing.T, errs field.ErrorList) {
	t.Helper()
	if len(errs) > 0 {
		t.Fatalf("unexpected errors:\n%s", details(errs))
	}
}

func TestBuildModelOnlyRequest(t *testing.T) {
	input := build(t, defaulted())
	want := `{"goal":{"sla":{"itl_ms":30,"ttft_ms":2000},"strict_sla":true,"target":"throughput_per_gpu"},` +
		`"search_space":{"backend":["vllm","sglang","trtllm"],"gpu_budget":8,"hardware_sku":"h200_sxm","model_name":"meta-llama/Llama-3.1-8B-Instruct"},` +
		`"sweep":{"max_trials":320},` +
		`"workload":{"isl":4000,"kv_load_ratio":[0,1],"osl":1000,"request_count":1000}}`
	if got := mustJSON(t, input); got != want {
		t.Fatalf("effective input\n got: %s\nwant: %s", got, want)
	}
	requireNoError(t, Validate(input))
}

func TestBuildIsDeterministicAndDoesNotMutateSpec(t *testing.T) {
	const params = `{"sweep":{"max_trials":100},"workload":{"request_count":5}}`
	spec := withParams(defaulted(), params)
	first, second := mustJSON(t, build(t, spec)), mustJSON(t, build(t, spec))
	if first != second {
		t.Fatalf("not deterministic:\n%s\n%s", first, second)
	}
	if string(spec.AISimulate.Parameters.Raw) != params {
		t.Fatalf("parameters mutated: %s", spec.AISimulate.Parameters.Raw)
	}
}

func TestBuildExplicitRequestFromDocs(t *testing.T) {
	metric := v1beta2.ObjectiveMetric("Throughput")
	spec := defaulted()
	spec.Backends = []v1beta2.Backend{v1beta2.BackendVLLM, v1beta2.BackendSGLang}
	spec.Workload.Static = &v1beta2.StaticWorkloadSpec{ISL: ptr(int32(4000)), OSL: ptr(int32(1000)), RequestRate: ptr(10.0)}
	spec.Objective.Metric = &metric
	spec = withParams(spec, `{"workload":{"request_count":1000},"sweep":{"max_trials":100}}`)

	input := build(t, spec)
	want := `{"goal":{"sla":{"itl_ms":30,"ttft_ms":2000},"strict_sla":true,"target":"throughput"},` +
		`"search_space":{"backend":["vllm","sglang"],"gpu_budget":8,"hardware_sku":"h200_sxm","model_name":"meta-llama/Llama-3.1-8B-Instruct"},` +
		`"sweep":{"max_trials":100},` +
		`"workload":{"isl":4000,"osl":1000,"request_count":1000,"request_rate":10}}`
	if got := mustJSON(t, input); got != want {
		t.Fatalf("effective input\n got: %s\nwant: %s", got, want)
	}
	requireNoError(t, Validate(input))
}

func TestMergePatchSemantics(t *testing.T) {
	cases := []struct{ name, target, patch, want string }{
		{"scalar replaces", `{"a":1}`, `{"a":2}`, `{"a":2}`},
		{"object merges recursively", `{"a":{"b":1,"c":2}}`, `{"a":{"b":3}}`, `{"a":{"b":3,"c":2}}`},
		{"null deletes", `{"a":1,"b":2}`, `{"a":null}`, `{"b":2}`},
		{"array replaces", `{"a":[1,2,3]}`, `{"a":[4]}`, `{"a":[4]}`},
		{"object replaces scalar and drops nulls", `{"a":1}`, `{"a":{"b":null,"c":1}}`, `{"a":{"c":1}}`},
		{"null on absent key is a no-op", `{"a":1}`, `{"z":null}`, `{"a":1}`},
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
}

func TestNativeNullDeletesTypedField(t *testing.T) {
	input := build(t, withParams(defaulted(), `{"workload":{"isl":null}}`))
	requireError(t, Validate(input), "requires isl")
}

func TestExplicitNativeValueOverridesTypedField(t *testing.T) {
	input := build(t, withParams(defaulted(), `{"workload":{"isl":8000},"search_space":{"gpu_budget":16}}`))
	workload := asMap(input["workload"])
	if workload["isl"] != 8000.0 || asMap(input["search_space"])["gpu_budget"] != 16.0 {
		t.Fatalf("native values did not win: %s", mustJSON(t, input))
	}
}

func TestRequestCountDefault(t *testing.T) {
	t.Run("skipped when native selects num_request_ratio", func(t *testing.T) {
		input := build(t, withParams(defaulted(), `{"workload":{"num_request_ratio":10}}`))
		if _, has := asMap(input["workload"])["request_count"]; has {
			t.Fatalf("request_count defaulted next to num_request_ratio: %s", mustJSON(t, input))
		}
		requireNoError(t, Validate(input))
	})
	t.Run("skipped when typed numRequestRatio is set", func(t *testing.T) {
		spec := defaulted()
		spec.Workload.Static.NumRequestRatio = ptr(5.0)
		if _, has := asMap(build(t, spec)["workload"])["request_count"]; has {
			t.Fatal("request_count defaulted next to typed numRequestRatio")
		}
	})
	t.Run("native null deletes the default", func(t *testing.T) {
		input := build(t, withParams(defaulted(), `{"workload":{"request_count":null}}`))
		if _, has := asMap(input["workload"])["request_count"]; has {
			t.Fatal("request_count re-added after a native null")
		}
		requireError(t, Validate(input), "request_count or num_request_ratio")
	})
	t.Run("native deleting the ratio restores the default", func(t *testing.T) {
		spec := defaulted()
		spec.Workload.Static.NumRequestRatio = ptr(5.0)
		input := build(t, withParams(spec, `{"workload":{"num_request_ratio":null}}`))
		if asMap(input["workload"])["request_count"] != 1000.0 {
			t.Fatalf("default not restored: %s", mustJSON(t, input))
		}
	})
}

func TestMaxTrialsDefault(t *testing.T) {
	t.Run("added for a KV-load range", func(t *testing.T) {
		if asMap(build(t, defaulted())["sweep"])["max_trials"] != 320.0 {
			t.Fatal("no default max_trials for a range search")
		}
	})
	t.Run("not added for a pinned load", func(t *testing.T) {
		spec := defaulted()
		spec.Workload.Static.KVLoadRatio = v1beta2.KVLoadRatio{0.5}
		input := build(t, spec)
		if _, has := input["sweep"]; has {
			t.Fatalf("unexpected sweep: %s", mustJSON(t, input))
		}
		if asMap(input["workload"])["kv_load_ratio"] != 0.5 {
			t.Fatal("a one-value ratio should be a scalar")
		}
		requireNoError(t, Validate(input))
	})
	t.Run("native value or null wins", func(t *testing.T) {
		explicit := build(t, withParams(defaulted(), `{"sweep":{"max_trials":7}}`))
		if asMap(explicit["sweep"])["max_trials"] != 7.0 {
			t.Fatal("native max_trials ignored")
		}
		deleted := build(t, withParams(defaulted(), `{"sweep":{"max_trials":null}}`))
		if _, has := asMap(deleted["sweep"])["max_trials"]; has {
			t.Fatal("max_trials re-added after a native null")
		}
		requireError(t, Validate(deleted), "requires a Pareto objective or sweep.max_trials")
	})
	t.Run("disabled by a zero option", func(t *testing.T) {
		input, err := Build(defaulted(), Options{RequestCount: 1000})
		if err != nil {
			t.Fatal(err)
		}
		requireError(t, Validate(input), "sweep.max_trials")
	})
}

func TestStrictSLADerivation(t *testing.T) {
	t.Run("enabled when the SLA has a bound", func(t *testing.T) {
		if asMap(build(t, defaulted())["goal"])["strict_sla"] != true {
			t.Fatal("strict_sla not derived")
		}
	})
	t.Run("disabled for an empty SLA", func(t *testing.T) {
		spec := defaulted()
		spec.Objective.SLA = &v1beta2.ObjectiveSLASpec{}
		goal := asMap(build(t, spec)["goal"])
		if _, has := goal["strict_sla"]; has {
			t.Fatalf("strict_sla derived from an empty SLA: %v", goal)
		}
		if _, has := goal["sla"]; has {
			t.Fatalf("empty SLA should not produce goal.sla: %v", goal)
		}
	})
	t.Run("native value is kept", func(t *testing.T) {
		goal := asMap(build(t, withParams(defaulted(), `{"goal":{"strict_sla":false}}`))["goal"])
		if goal["strict_sla"] != false {
			t.Fatalf("native strict_sla overridden: %v", goal)
		}
	})
	t.Run("follows the merged SLA", func(t *testing.T) {
		input := build(t, withParams(defaulted(), `{"goal":{"sla":{"ttft_ms":null,"itl_ms":null}}}`))
		if _, has := asMap(input["goal"])["strict_sla"]; has {
			t.Fatalf("strict_sla derived with no bounds left: %s", mustJSON(t, input))
		}
	})
}

func TestParetoObjective(t *testing.T) {
	spec := defaulted()
	spec.Objective.Mode = v1beta2.ObjectiveModePareto
	spec.Objective.Metric = nil
	spec.Objective.Metrics = []v1beta2.ObjectiveMetric{"ThroughputPerGPU", "ThroughputPerUser"}
	input := build(t, spec)
	goal := asMap(input["goal"])
	if got := mustJSON(t, map[string]any{"o": goal["pareto_objectives"]}); goal["target"] != "pareto" ||
		got != `{"o":["throughput_per_gpu","throughput_per_user"]}` {
		t.Fatalf("unexpected pareto goal: %v", goal)
	}
	requireNoError(t, Validate(input))

	t.Run("duplicates rejected", func(t *testing.T) {
		spec := withParams(defaulted(), `{"goal":{"target":"pareto","pareto_objectives":["ttft","ttft"]}}`)
		spec.Objective.Metric = nil
		requireError(t, Validate(build(t, spec)), "distinct")
	})
	t.Run("goodput needs an SLA bound", func(t *testing.T) {
		spec := withParams(defaulted(), `{"goal":{"target":"pareto","pareto_objectives":["goodput","ttft"]}}`)
		spec.Objective.Metric = nil
		spec.Objective.SLA = &v1beta2.ObjectiveSLASpec{}
		requireError(t, Validate(build(t, spec)), "require at least one SLA bound")
	})
	t.Run("pareto_objectives needs the pareto target", func(t *testing.T) {
		spec := withParams(defaulted(), `{"goal":{"pareto_objectives":["ttft","e2e_latency"]}}`)
		requireError(t, Validate(build(t, spec)), "only valid when goal.target is pareto")
	})
}

func TestGoalValidation(t *testing.T) {
	cases := []struct{ name, params, want string }{
		{"min_gpus unsupported", `{"goal":{"target":"min_gpus"}}`, "not supported by the Dynamo integration"},
		{"unknown target", `{"goal":{"target":"latency"}}`, "unknown goal.target"},
		{"e2e excludes ttft", `{"goal":{"sla":{"e2e_ms":100}}}`, "e2e_ms cannot be combined"},
		{"strict_sla needs a bound", `{"goal":{"strict_sla":true,"sla":{"ttft_ms":null,"itl_ms":null}}}`, "strict_sla requires at least one SLA bound"},
		{"non-boolean strict_sla", `{"goal":{"strict_sla":"yes"}}`, "must be a boolean"},
		{"negative bound", `{"goal":{"sla":{"ttft_ms":-1}}}`, "positive number"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			requireError(t, Validate(build(t, withParams(defaulted(), tc.params))), tc.want)
		})
	}
	t.Run("target deleted", func(t *testing.T) {
		spec := withParams(defaulted(), `{"goal":{"target":null}}`)
		spec.Objective.Metric = nil
		requireError(t, Validate(build(t, spec)), "goal.target is missing")
	})
	t.Run("goodput without SLA", func(t *testing.T) {
		spec := defaulted()
		spec.Objective.Metric = ptr(v1beta2.ObjectiveMetric("Goodput"))
		spec.Objective.SLA = &v1beta2.ObjectiveSLASpec{}
		requireError(t, Validate(build(t, spec)), "require at least one SLA bound")
	})
}

func TestWorkloadValidation(t *testing.T) {
	t.Run("opt-out without native workload", func(t *testing.T) {
		spec := defaulted()
		spec.Workload.Static = &v1beta2.StaticWorkloadSpec{}
		requireError(t, Validate(build(t, spec)), "effective workload is empty")
	})
	t.Run("opt-out completed natively", func(t *testing.T) {
		spec := withParams(defaulted(), `{"workload":{"isl":512,"osl":64,"concurrency":8}}`)
		spec.Workload.Static = &v1beta2.StaticWorkloadSpec{}
		input := build(t, spec)
		requireNoError(t, Validate(input))
		if _, has := asMap(input["workload"])["kv_load_ratio"]; has {
			t.Fatal("opt-out still carries the typed kv_load_ratio")
		}
	})
	t.Run("two load modes", func(t *testing.T) {
		requireError(t, Validate(build(t, withParams(defaulted(), `{"workload":{"concurrency":4}}`))), "exactly one of request_rate")
	})
	t.Run("native switches the load mode by deleting the typed one", func(t *testing.T) {
		requireNoError(t, Validate(build(t, withParams(defaulted(), `{"workload":{"kv_load_ratio":null,"concurrency":4}}`))))
	})
	t.Run("bad values", func(t *testing.T) {
		cases := map[string]string{
			`{"workload":{"isl":0}}`:                         "isl must be a positive integer",
			`{"workload":{"osl":1.5}}`:                       "osl must be a positive integer",
			`{"workload":{"num_request_ratio":-2}}`:          "num_request_ratio must be a positive number",
			`{"workload":{"kv_load_ratio":[0.9,0.1]}}`:       "needs min < max",
			`{"workload":{"kv_load_ratio":[0,0.5,1]}}`:       "exactly [min, max]",
			`{"workload":{"kv_load_ratio":-1}}`:              "non-negative",
			`{"workload":{"trace_path":"/tmp/trace.jsonl"}}`: "trace workloads are not supported",
		}
		for params, want := range cases {
			requireError(t, Validate(build(t, withParams(defaulted(), params))), want)
		}
	})
}

func TestSearchSpaceValidation(t *testing.T) {
	t.Run("unresolved hardware", func(t *testing.T) {
		spec := defaulted()
		spec.Hardware = nil
		errs := Validate(build(t, spec))
		requireError(t, errs, "spec.hardware.gpu.sku: effective hardware is unresolved")
		requireError(t, errs, "spec.hardware.gpu.budget: effective hardware is unresolved")
	})
	t.Run("partial hardware", func(t *testing.T) {
		spec := defaulted()
		spec.Hardware.GPU.Budget = nil
		errs := Validate(build(t, spec))
		if len(errs) != 1 || errs[0].Field != "spec.hardware.gpu.budget" {
			t.Fatalf("want only the budget unresolved, got:\n%s", details(errs))
		}
	})
	t.Run("native hardware completes the typed one", func(t *testing.T) {
		spec := withParams(defaulted(), `{"search_space":{"hardware_sku":"h100_sxm","gpu_budget":4}}`)
		spec.Hardware = nil
		requireNoError(t, Validate(build(t, spec)))
	})
	t.Run("unknown backend", func(t *testing.T) {
		spec := withParams(defaulted(), `{"search_space":{"backend":["tgi"]}}`)
		requireError(t, Validate(build(t, spec)), "must be one of vllm, sglang, trtllm")
	})
}

func TestParametersMustBeAnObject(t *testing.T) {
	for _, params := range []string{`[1]`, `"x"`, `3`, `{`} {
		if _, err := Build(withParams(defaulted(), params), DefaultOptions()); err == nil {
			t.Errorf("Build accepted parameters %s", params)
		}
	}
	for _, params := range []string{``, `null`, `{}`} {
		if _, err := Build(withParams(defaulted(), params), DefaultOptions()); err != nil {
			t.Errorf("Build rejected parameters %q: %v", params, err)
		}
	}
}

func TestMetricDefault(t *testing.T) {
	t.Run("optimize without a metric", func(t *testing.T) {
		spec := defaulted()
		spec.Objective.Metric = nil
		if metric, ok := MetricDefault(spec); !ok || metric != "ThroughputPerGPU" {
			t.Fatalf("got %q %v", metric, ok)
		}
	})
	t.Run("an explicit metric is kept", func(t *testing.T) {
		if _, ok := MetricDefault(defaulted()); ok {
			t.Fatal("defaulted over an explicit metric")
		}
	})
	t.Run("pareto has no scalar metric", func(t *testing.T) {
		spec := defaulted()
		spec.Objective.Mode, spec.Objective.Metric = v1beta2.ObjectiveModePareto, nil
		if _, ok := MetricDefault(spec); ok {
			t.Fatal("defaulted a metric in Pareto mode")
		}
	})
	t.Run("the native input owns the target", func(t *testing.T) {
		for _, params := range []string{`{"goal":{"target":"ttft"}}`, `{"goal":{"target":null}}`} {
			spec := withParams(defaulted(), params)
			spec.Objective.Metric = nil
			if _, ok := MetricDefault(spec); ok {
				t.Fatalf("defaulted a metric next to %s", params)
			}
		}
	})
	t.Run("a nil objective", func(t *testing.T) {
		spec := defaulted()
		spec.Objective = nil
		if _, ok := MetricDefault(spec); ok {
			t.Fatal("defaulted without an objective")
		}
	})
}

func TestConflicts(t *testing.T) {
	t.Run("agreeing values", func(t *testing.T) {
		spec := withParams(defaulted(), `{"goal":{"target":"throughput_per_gpu","sla":{"ttft_ms":2000}}}`)
		requireNoError(t, Conflicts(spec))
	})
	t.Run("typed metric against native target", func(t *testing.T) {
		requireError(t, Conflicts(withParams(defaulted(), `{"goal":{"target":"ttft"}}`)),
			"conflicts with spec.aisimulate.parameters.goal.target")
	})
	t.Run("a defaulted metric cannot conflict", func(t *testing.T) {
		spec := withParams(defaulted(), `{"goal":{"target":"ttft"}}`)
		spec.Objective.Metric = nil
		requireNoError(t, Conflicts(spec))
	})
	t.Run("default SLA does not conflict with a native bound", func(t *testing.T) {
		spec := withParams(defaulted(), `{"goal":{"sla":{"ttft_ms":500}}}`)
		requireNoError(t, Conflicts(spec))
		sla := asMap(asMap(build(t, spec)["goal"])["sla"])
		if sla["ttft_ms"] != 500.0 || sla["itl_ms"] != 30.0 {
			t.Fatalf("native bound should win and keep the typed itl: %v", sla)
		}
	})
	t.Run("explicit SLA against native bound", func(t *testing.T) {
		spec := withParams(defaulted(), `{"goal":{"sla":{"ttft_ms":800}}}`)
		spec.Objective.SLA = &v1beta2.ObjectiveSLASpec{TTFTMs: ptr(500.0)}
		requireError(t, Conflicts(spec), "conflicts with spec.aisimulate.parameters.goal.sla.ttft_ms")
	})
	t.Run("explicit SLA opt-out against native bounds", func(t *testing.T) {
		spec := withParams(defaulted(), `{"goal":{"sla":{"ttft_ms":800}}}`)
		spec.Objective.SLA = &v1beta2.ObjectiveSLASpec{}
		requireError(t, Conflicts(spec), "disables SLA bounds")
	})
	t.Run("typed Pareto against a scalar native target", func(t *testing.T) {
		spec := withParams(defaulted(), `{"goal":{"target":"throughput"}}`)
		spec.Objective.Mode, spec.Objective.Metric = v1beta2.ObjectiveModePareto, nil
		spec.Objective.Metrics = []v1beta2.ObjectiveMetric{"TTFT", "Throughput"}
		requireError(t, Conflicts(spec), "spec.objective.mode")
	})
	t.Run("typed Pareto metrics against native objectives", func(t *testing.T) {
		spec := withParams(defaulted(), `{"goal":{"target":"pareto","pareto_objectives":["ttft","e2e_latency"]}}`)
		spec.Objective.Mode, spec.Objective.Metric = v1beta2.ObjectiveModePareto, nil
		spec.Objective.Metrics = []v1beta2.ObjectiveMetric{"TTFT", "Throughput"}
		requireError(t, Conflicts(spec), "pareto_objectives")
	})
	t.Run("same Pareto metrics in another order", func(t *testing.T) {
		spec := withParams(defaulted(), `{"goal":{"target":"pareto","pareto_objectives":["throughput","ttft"]}}`)
		spec.Objective.Mode, spec.Objective.Metric = v1beta2.ObjectiveModePareto, nil
		spec.Objective.Metrics = []v1beta2.ObjectiveMetric{"TTFT", "Throughput"}
		requireNoError(t, Conflicts(spec))
	})
	t.Run("no parameters", func(t *testing.T) {
		requireNoError(t, Conflicts(defaulted()))
	})
}

func TestHardware(t *testing.T) {
	t.Run("typed", func(t *testing.T) {
		sku, hasBudget, err := Hardware(defaulted())
		if err != nil || sku != "h200_sxm" || !hasBudget {
			t.Fatalf("got %q %v %v", sku, hasBudget, err)
		}
	})
	t.Run("unresolved", func(t *testing.T) {
		spec := defaulted()
		spec.Hardware = nil
		sku, hasBudget, err := Hardware(spec)
		if err != nil || sku != "" || hasBudget {
			t.Fatalf("got %q %v %v", sku, hasBudget, err)
		}
	})
	t.Run("native completes typed", func(t *testing.T) {
		spec := withParams(defaulted(), `{"search_space":{"hardware_sku":"b200_sxm"}}`)
		spec.Hardware.GPU.SKU = nil
		sku, hasBudget, err := Hardware(spec)
		if err != nil || sku != "b200_sxm" || !hasBudget {
			t.Fatalf("got %q %v %v", sku, hasBudget, err)
		}
	})
	t.Run("native null deletes typed", func(t *testing.T) {
		sku, hasBudget, _ := Hardware(withParams(defaulted(), `{"search_space":{"gpu_budget":null}}`))
		if sku != "h200_sxm" || hasBudget {
			t.Fatalf("got %q %v", sku, hasBudget)
		}
	})
}
