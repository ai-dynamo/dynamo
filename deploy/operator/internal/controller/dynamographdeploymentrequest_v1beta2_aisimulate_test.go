/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package controller

import (
	"encoding/json"
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

func dgdrLoweringPtr[T any](v T) *T { return &v }

// dgdrLoweringSpec is a model-only Request as it looks after the CRD's structural defaults and
// admission: the fixed-concurrency workload, the scalar metric and discovered hardware.
func dgdrLoweringSpec() *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	metric := nvidiacomv1beta2.ObjectiveMetric("Throughput")
	return &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{NativeSearchInput: nvidiacomv1beta2.NativeSearchInput{
		Model: &nvidiacomv1beta2.ModelSpec{Name: "meta-llama/Llama-3.1-8B-Instruct"},
		Backends: []nvidiacomv1beta2.Backend{
			nvidiacomv1beta2.BackendVLLM, nvidiacomv1beta2.BackendSGLang, nvidiacomv1beta2.BackendTRTLLM,
		},
		Hardware: &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{
			SKU: dgdrLoweringPtr("h200_sxm"), Budget: dgdrLoweringPtr(int32(8)),
		}},
		Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{
			ISL: dgdrLoweringPtr(int32(1024)), OSL: dgdrLoweringPtr(int32(128)), Concurrency: dgdrLoweringPtr(int32(10)),
		}},
		Objective:      &nvidiacomv1beta2.ObjectiveSpec{Mode: nvidiacomv1beta2.ObjectiveModeOptimize, Metric: &metric},
		Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 1},
	}}
}

func dgdrLoweringWithSLA(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	spec.Objective.SLA = &nvidiacomv1beta2.ObjectiveSLASpec{TTFTMs: dgdrLoweringPtr(2000.0), ITLMs: dgdrLoweringPtr(30.0)}
	return spec
}

func dgdrLoweringWithParameters(
	spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
	parameters string,
) *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{Parameters: &runtime.RawExtension{Raw: []byte(parameters)}}
	return spec
}

func dgdrLowered(t *testing.T, spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) map[string]any {
	t.Helper()
	input, err := lowerDGDRV1beta2Input(spec)
	if err != nil {
		t.Fatalf("lowerDGDRV1beta2Input: %v", err)
	}
	return input
}

func dgdrLoweringEncode(t *testing.T, input map[string]any) string {
	t.Helper()
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func dgdrLoweringUnderlay(t *testing.T, spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) map[string]any {
	t.Helper()
	input, err := dgdrV1beta2Underlay(spec)
	if err != nil {
		t.Fatalf("dgdrV1beta2Underlay: %v", err)
	}
	return input
}

func dgdrLoweringNested(t *testing.T, input map[string]any, path ...string) any {
	t.Helper()
	value, _, err := unstructured.NestedFieldNoCopy(input, path...)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func TestDGDRV1beta2LoweringModelOnlyRequest(t *testing.T) {
	want := `{"goal":{"target":"throughput"},` +
		`"search_space":{"backend":["vllm","sglang","trtllm"],"gpu_budget":8,"hardware_sku":"h200_sxm","model_name":"meta-llama/Llama-3.1-8B-Instruct"},` +
		`"workload":{"concurrency":10,"isl":1024,"osl":128,"request_count":1000}}`
	if got := dgdrLoweringEncode(t, dgdrLowered(t, dgdrLoweringSpec())); got != want {
		t.Fatalf("effective input\n got: %s\nwant: %s", got, want)
	}
}

func TestDGDRV1beta2LoweringIsDeterministicAndDoesNotMutateTheSpec(t *testing.T) {
	const parameters = `{"sweep":{"max_trials":100},"workload":{"request_count":5}}`
	spec := dgdrLoweringWithParameters(dgdrLoweringSpec(), parameters)
	first, second := dgdrLoweringEncode(t, dgdrLowered(t, spec)), dgdrLoweringEncode(t, dgdrLowered(t, spec))
	if first != second {
		t.Fatalf("not deterministic:\n%s\n%s", first, second)
	}
	if string(spec.AISimulate.Parameters.Raw) != parameters {
		t.Fatalf("parameters mutated: %s", spec.AISimulate.Parameters.Raw)
	}
}

func TestDGDRV1beta2LoweringExplicitRequest(t *testing.T) {
	metric := nvidiacomv1beta2.ObjectiveMetric("Throughput")
	spec := dgdrLoweringWithSLA(dgdrLoweringSpec())
	spec.Backends = []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM, nvidiacomv1beta2.BackendSGLang}
	spec.Workload.Static = &nvidiacomv1beta2.StaticWorkloadSpec{
		ISL: dgdrLoweringPtr(int32(4000)), OSL: dgdrLoweringPtr(int32(1000)), RequestRate: dgdrLoweringPtr(10.0),
	}
	spec.Objective.Metric = &metric
	spec = dgdrLoweringWithParameters(spec, `{"workload":{"request_count":1000},"sweep":{"max_trials":100}}`)

	want := `{"goal":{"sla":{"itl_ms":30,"ttft_ms":2000},"strict_sla":true,"target":"throughput"},` +
		`"search_space":{"backend":["vllm","sglang"],"gpu_budget":8,"hardware_sku":"h200_sxm","model_name":"meta-llama/Llama-3.1-8B-Instruct"},` +
		`"sweep":{"max_trials":100},` +
		`"workload":{"isl":4000,"osl":1000,"request_count":1000,"request_rate":10}}`
	if got := dgdrLoweringEncode(t, dgdrLowered(t, spec)); got != want {
		t.Fatalf("effective input\n got: %s\nwant: %s", got, want)
	}
}

func TestDGDRV1beta2LoweringParametersWinOverTheTypedUnderlay(t *testing.T) {
	t.Run("a native null deletes a typed field", func(t *testing.T) {
		input := dgdrLowered(t, dgdrLoweringWithParameters(dgdrLoweringSpec(), `{"workload":{"isl":null}}`))
		if dgdrLoweringNested(t, input, "workload", "isl") != nil {
			t.Fatalf("isl survived a native null: %s", dgdrLoweringEncode(t, input))
		}
	})
	t.Run("a native value overrides a typed field", func(t *testing.T) {
		input := dgdrLowered(t, dgdrLoweringWithParameters(dgdrLoweringSpec(),
			`{"workload":{"isl":8000},"search_space":{"gpu_budget":16}}`))
		if dgdrLoweringNested(t, input, "workload", "isl") != 8000.0 ||
			dgdrLoweringNested(t, input, "search_space", "gpu_budget") != 16.0 {
			t.Fatalf("native values did not win: %s", dgdrLoweringEncode(t, input))
		}
	})
	t.Run("a native target overrides the typed metric", func(t *testing.T) {
		input := dgdrLowered(t, dgdrLoweringWithParameters(dgdrLoweringSpec(), `{"goal":{"target":"ttft"}}`))
		if dgdrLoweringNested(t, input, "goal", "target") != "ttft" {
			t.Fatalf("typed metric beat the native target: %s", dgdrLoweringEncode(t, input))
		}
	})
}

func TestDGDRV1beta2LoweringUnderlayParetoObjective(t *testing.T) {
	spec := dgdrLoweringSpec()
	spec.Objective.Mode = nvidiacomv1beta2.ObjectiveModePareto
	spec.Objective.Metric = nil
	spec.Objective.Metrics = []nvidiacomv1beta2.ObjectiveMetric{"ThroughputPerGPU", "ThroughputPerUser", "Unknown"}
	input := dgdrLoweringUnderlay(t, spec)
	if dgdrLoweringNested(t, input, "goal", "target") != "pareto" {
		t.Fatalf("not a pareto goal: %v", input["goal"])
	}
	objectives, _, _ := unstructured.NestedSlice(input, "goal", "pareto_objectives")
	got, _ := json.Marshal(objectives)
	if string(got) != `["throughput_per_gpu","throughput_per_user","unknown"]` {
		t.Fatalf("pareto_objectives = %s", got)
	}
}

func TestDGDRV1beta2LoweringUnderlayOmitsUnsetTypedFields(t *testing.T) {
	spec := &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
		NativeSearchInput: nvidiacomv1beta2.NativeSearchInput{Model: &nvidiacomv1beta2.ModelSpec{Name: "m"}},
	}
	if got := dgdrLoweringEncode(t, dgdrLoweringUnderlay(t, spec)); got != `{"search_space":{"model_name":"m"}}` {
		t.Fatalf("got %s", got)
	}
	spec.Hardware = &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: dgdrLoweringPtr("")}}
	if got := dgdrLoweringEncode(t, dgdrLoweringUnderlay(t, spec)); got != `{"search_space":{"model_name":"m"}}` {
		t.Fatalf("an empty SKU must not lower: %s", got)
	}
}

func TestDGDRV1beta2LoweringUnderlayKVLoadRatioShapes(t *testing.T) {
	spec := dgdrLoweringSpec()
	spec.Workload.Static.KVLoadRatio = nvidiacomv1beta2.KVLoadRatio{0.25}
	if dgdrLoweringNested(t, dgdrLoweringUnderlay(t, spec), "workload", "kv_load_ratio") != 0.25 {
		t.Fatal("a single ratio lowers to a scalar")
	}
	spec.Workload.Static.KVLoadRatio = nvidiacomv1beta2.KVLoadRatio{0.1, 0.2, 0.3}
	if ratios, _ := dgdrLoweringNested(t, dgdrLoweringUnderlay(t, spec), "workload", "kv_load_ratio").([]any); len(ratios) != 3 {
		t.Fatal("longer lists lower unchanged so validation can report them")
	}
}

func TestDGDRV1beta2LoweringParametersMustBeAnObject(t *testing.T) {
	for _, parameters := range []string{`[1]`, `"x"`, `3`, `{`} {
		if _, err := lowerDGDRV1beta2Input(dgdrLoweringWithParameters(dgdrLoweringSpec(), parameters)); err == nil {
			t.Errorf("lowering accepted parameters %s", parameters)
		}
	}
	for _, parameters := range []string{``, `null`, `{}`} {
		if _, err := lowerDGDRV1beta2Input(dgdrLoweringWithParameters(dgdrLoweringSpec(), parameters)); err != nil {
			t.Errorf("lowering rejected parameters %q: %v", parameters, err)
		}
	}
}

func TestDGDRV1beta2LoweringNativeMetricNames(t *testing.T) {
	for metric, want := range map[nvidiacomv1beta2.ObjectiveMetric]string{
		"Throughput":        "throughput",
		"ThroughputPerGPU":  "throughput_per_gpu",
		"ThroughputPerUser": "throughput_per_user",
		"TTFT":              "ttft",
		"E2ELatency":        "e2e_latency",
		"Goodput":           "goodput",
		"GoodputPerGPU":     "goodput_per_gpu",
	} {
		if got := dgdrNativeMetric(metric); got != want {
			t.Errorf("dgdrNativeMetric(%s) = %q, want %q", metric, got, want)
		}
	}
}
