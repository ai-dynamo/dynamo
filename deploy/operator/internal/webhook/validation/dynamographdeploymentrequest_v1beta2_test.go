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

package validation

import (
	"encoding/json"
	"strings"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func v1b2Ptr[T any](v T) *T { return &v }

// v1b2DefaultedSpec is a Request as it looks once the schema and admission have defaulted it.
func v1b2DefaultedSpec() *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	metric := nvidiacomv1beta2.ObjectiveMetric("ThroughputPerGPU")
	return &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
		Model:    &nvidiacomv1beta2.ModelSpec{Name: "meta-llama/Llama-3.1-8B-Instruct"},
		Backends: []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM, nvidiacomv1beta2.BackendSGLang, nvidiacomv1beta2.BackendTRTLLM},
		Hardware: &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: v1b2Ptr("h200_sxm"), Budget: v1b2Ptr(int32(8))}},
		Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{
			ISL: v1b2Ptr(int32(4000)), OSL: v1b2Ptr(int32(1000)), KVLoadRatio: nvidiacomv1beta2.KVLoadRatio{0, 1},
		}},
		Objective: &nvidiacomv1beta2.ObjectiveSpec{
			Mode:   nvidiacomv1beta2.ObjectiveModeOptimize,
			Metric: &metric,
			SLA:    &nvidiacomv1beta2.ObjectiveSLASpec{TTFTMs: v1b2Ptr(2000.0), ITLMs: v1b2Ptr(30.0)},
		},
		AISimulate:     &nvidiacomv1beta2.AISimulateSpec{Image: "registry.example.com/aisimulate:1.2.0"},
		Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 1},
	}
}

func v1b2WithParameters(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec, parameters string) *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	if spec.AISimulate == nil {
		spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{Image: "registry.example.com/aisimulate:1.2.0"}
	}
	spec.AISimulate.Parameters = &runtime.RawExtension{Raw: []byte(parameters)}
	return spec
}

func v1b2Details(errs field.ErrorList) string {
	var out []string
	for _, err := range errs {
		out = append(out, err.Field+": "+err.Detail)
	}
	return strings.Join(out, "\n")
}

func v1b2RequireError(t *testing.T, errs field.ErrorList, substr string) {
	t.Helper()
	if !strings.Contains(v1b2Details(errs), substr) {
		t.Fatalf("want an error containing %q, got:\n%s", substr, v1b2Details(errs))
	}
}

func v1b2RequireNoError(t *testing.T, errs field.ErrorList) {
	t.Helper()
	if len(errs) > 0 {
		t.Fatalf("unexpected errors:\n%s", v1b2Details(errs))
	}
}

func TestValidateV1beta2AcceptsADefaultedRequest(t *testing.T) {
	v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(v1b2DefaultedSpec()))
}

func TestValidateV1beta2TypedStage(t *testing.T) {
	t.Run("missing model", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.Model = nil
		errs := validateDynamoGraphDeploymentRequestV1beta2(spec)
		v1b2RequireError(t, errs, "spec.model: model is required")
	})
	t.Run("blank model", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.Model.Name = "  "
		errs := validateDynamoGraphDeploymentRequestV1beta2(spec)
		v1b2RequireError(t, errs, "spec.model.name: must not be blank")
	})
	t.Run("duplicate backend", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.Backends = []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM, nvidiacomv1beta2.BackendVLLM}
		v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "spec.backends[1]")
	})
	t.Run("kv_load_ratio range order", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.Workload.Static.KVLoadRatio = nvidiacomv1beta2.KVLoadRatio{0.9, 0.1}
		v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "spec.workload.static.kvLoadRatio: a range needs min < max")
	})
	t.Run("an empty workload has nothing to validate", func(t *testing.T) {
		spec := v1b2WithParameters(v1b2DefaultedSpec(), `{"workload":{"isl":512,"osl":64,"concurrency":8}}`)
		spec.Workload = &nvidiacomv1beta2.WorkloadSpec{}
		v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(spec))
	})
	t.Run("an invalid typed value is not excused by the parameters", func(t *testing.T) {
		spec := v1b2WithParameters(v1b2DefaultedSpec(), `{"workload":{"kv_load_ratio":0.5}}`)
		spec.Workload.Static.KVLoadRatio = nvidiacomv1beta2.KVLoadRatio{0.9, 0.1}
		v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "spec.workload.static.kvLoadRatio")
	})
	t.Run("stages aggregate", func(t *testing.T) {
		spec := v1b2WithParameters(v1b2DefaultedSpec(), `{"workload":{"isl":0}}`)
		spec.Model.Name = ""
		errs := validateDynamoGraphDeploymentRequestV1beta2(spec)
		v1b2RequireError(t, errs, "spec.model.name")
		v1b2RequireError(t, errs, "spec.aisimulate.parameters.workload.isl: isl must be a positive integer")
	})
}

func TestValidateV1beta2AISimulateImage(t *testing.T) {
	t.Run("no AISimulate section", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.AISimulate = nil
		v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "spec.aisimulate.image: no AISimulate image")
	})
	t.Run("parameters without an image", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{}
		v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "spec.aisimulate.image")
	})
	t.Run("custom tag needs a version", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.AISimulate.Image = "registry.example.com/aisimulate:custom"
		v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "spec.aisimulate.versionOverride: is required when spec.aisimulate.image has no parseable version tag")
	})
	t.Run("custom tag with a version", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.AISimulate.Image = "registry.example.com/aisimulate:custom"
		spec.AISimulate.VersionOverride = "1.4.0"
		v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(spec))
	})
}

func TestValidateV1beta2MalformedParameters(t *testing.T) {
	errs := validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), `[1]`))
	v1b2RequireError(t, errs, "spec.aisimulate.parameters: spec.aisimulate.parameters: must be a JSON object")
}

func TestEffectiveInputGoal(t *testing.T) {
	cases := []struct{ name, parameters, want string }{
		{"min_gpus unsupported", `{"goal":{"target":"min_gpus"}}`, "not supported by the Dynamo integration"},
		{"unknown target", `{"goal":{"target":"latency"}}`, "unknown goal.target"},
		{"e2e excludes ttft", `{"goal":{"sla":{"e2e_ms":100}}}`, "e2e_ms cannot be combined"},
		{"strict_sla needs a bound", `{"goal":{"strict_sla":true,"sla":{"ttft_ms":null,"itl_ms":null}}}`, "strict_sla requires at least one SLA bound"},
		{"non-boolean strict_sla", `{"goal":{"strict_sla":"yes"}}`, "must be a boolean"},
		{"negative bound", `{"goal":{"sla":{"ttft_ms":-1}}}`, "positive number"},
		{"min_goodput_rps", `{"goal":{"min_goodput_rps":3}}`, "min_goodput_rps"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), tc.parameters)), tc.want)
		})
	}

	t.Run("a native target replaces the typed metric", func(t *testing.T) {
		// The typed metric stays valid; the override decides what runs.
		v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), `{"goal":{"target":"ttft"}}`)))
	})
	t.Run("target deleted", func(t *testing.T) {
		spec := v1b2WithParameters(v1b2DefaultedSpec(), `{"goal":{"target":null}}`)
		errs := validateDynamoGraphDeploymentRequestV1beta2(spec)
		v1b2RequireError(t, errs, "spec.aisimulate.parameters.goal.target: effective goal.target is missing")
	})
	t.Run("goodput without an SLA", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.Objective.Metric = v1b2Ptr(nvidiacomv1beta2.ObjectiveMetric("Goodput"))
		spec.Objective.SLA = &nvidiacomv1beta2.ObjectiveSLASpec{}
		errs := validateDynamoGraphDeploymentRequestV1beta2(spec)
		v1b2RequireError(t, errs, "spec.objective.sla: [goodput] require at least one SLA bound")
	})
	t.Run("an empty typed SLA is valid for a throughput metric", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.Objective.SLA = &nvidiacomv1beta2.ObjectiveSLASpec{}
		v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(spec))
	})
	t.Run("a native SLA satisfies goodput", func(t *testing.T) {
		spec := v1b2WithParameters(v1b2DefaultedSpec(), `{"goal":{"sla":{"ttft_ms":500}}}`)
		spec.Objective.Metric = v1b2Ptr(nvidiacomv1beta2.ObjectiveMetric("Goodput"))
		spec.Objective.SLA = &nvidiacomv1beta2.ObjectiveSLASpec{}
		v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(spec))
	})
	t.Run("a typed bound is overridden by the native bound", func(t *testing.T) {
		// Native parameters always win; the typed 2000 ms is not a conflict.
		v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), `{"goal":{"sla":{"ttft_ms":500}}}`)))
	})
	t.Run("a range needs max_trials", func(t *testing.T) {
		errs := validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), `{"sweep":{"max_trials":null}}`))
		v1b2RequireError(t, errs, "spec.aisimulate.parameters.sweep.max_trials: a kv_load_ratio range requires a Pareto objective or sweep.max_trials")
	})
}

func TestEffectiveInputPareto(t *testing.T) {
	pareto := func(parameters string) *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
		spec := v1b2WithParameters(v1b2DefaultedSpec(), parameters)
		spec.Objective.Mode = nvidiacomv1beta2.ObjectiveModePareto
		spec.Objective.Metric = nil
		spec.Objective.Metrics = []nvidiacomv1beta2.ObjectiveMetric{"ThroughputPerGPU", "ThroughputPerUser"}
		return spec
	}
	t.Run("typed metrics are valid", func(t *testing.T) {
		v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(pareto(`{}`)))
	})
	t.Run("a native overlay with duplicates", func(t *testing.T) {
		errs := validateDynamoGraphDeploymentRequestV1beta2(pareto(`{"goal":{"pareto_objectives":["ttft","ttft"]}}`))
		v1b2RequireError(t, errs, "spec.aisimulate.parameters.goal.pareto_objectives: Pareto metrics must be distinct")
	})
	t.Run("a native overlay with one metric", func(t *testing.T) {
		v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(pareto(`{"goal":{"pareto_objectives":["ttft"]}}`)), "at least two metrics")
	})
	t.Run("non-scalar metric", func(t *testing.T) {
		v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(pareto(`{"goal":{"pareto_objectives":["ttft","pareto"]}}`)), "must be scalar targets")
	})
	t.Run("goodput needs a bound", func(t *testing.T) {
		spec := pareto(`{"goal":{"pareto_objectives":["goodput","ttft"]}}`)
		spec.Objective.SLA = &nvidiacomv1beta2.ObjectiveSLASpec{}
		v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "require at least one SLA bound")
	})
	t.Run("pareto_objectives needs the pareto target", func(t *testing.T) {
		spec := v1b2WithParameters(v1b2DefaultedSpec(), `{"goal":{"pareto_objectives":["ttft","e2e_latency"]}}`)
		v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "only valid when goal.target is pareto")
	})
	t.Run("a range search needs no max_trials", func(t *testing.T) {
		v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(pareto(`{"sweep":{"max_trials":null}}`)))
	})
}

func TestEffectiveInputWorkload(t *testing.T) {
	t.Run("an explicit empty workload without parameters", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.Workload = &nvidiacomv1beta2.WorkloadSpec{}
		errs := validateDynamoGraphDeploymentRequestV1beta2(spec)
		v1b2RequireError(t, errs, "spec.workload.static: effective workload is empty")
	})
	t.Run("an explicit empty workload completed natively", func(t *testing.T) {
		spec := v1b2WithParameters(v1b2DefaultedSpec(), `{"workload":{"isl":4000,"osl":1000,"concurrency":8}}`)
		spec.Workload = &nvidiacomv1beta2.WorkloadSpec{}
		v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(spec))
	})
	t.Run("two load modes", func(t *testing.T) {
		errs := validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), `{"workload":{"concurrency":4}}`))
		v1b2RequireError(t, errs, "spec.aisimulate.parameters.workload: effective workload needs exactly one of request_rate")
	})
	t.Run("native switches the load mode by deleting the typed one", func(t *testing.T) {
		v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), `{"workload":{"kv_load_ratio":null,"concurrency":4}}`)))
	})
	t.Run("a deleted field is reported on the parameter", func(t *testing.T) {
		errs := validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), `{"workload":{"isl":null}}`))
		v1b2RequireError(t, errs, "spec.aisimulate.parameters.workload: effective workload requires isl")
	})
	t.Run("a missing typed field is reported on the typed field's section", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.Workload.Static.OSL = nil
		v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "spec.workload.static: effective workload requires osl")
	})
	t.Run("a missing request count", func(t *testing.T) {
		errs := validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), `{"workload":{"request_count":null}}`))
		v1b2RequireError(t, errs, "request_count or num_request_ratio")
	})
	t.Run("bad values", func(t *testing.T) {
		cases := map[string]string{
			`{"workload":{"isl":0}}`:                         "isl must be a positive integer",
			`{"workload":{"osl":1.5}}`:                       "osl must be a positive integer",
			`{"workload":{"isl":"big"}}`:                     "isl must be a positive integer",
			`{"workload":{"num_request_ratio":-2}}`:          "num_request_ratio must be a positive number",
			`{"workload":{"trace_path":"/tmp/trace.jsonl"}}`: "trace workloads are not supported",
			`{"workload":{"trace_paths":["a"]}}`:             "remove workload.trace_paths",
		}
		for parameters, want := range cases {
			v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), parameters)), want)
		}
	})
}

func TestEffectiveInputKVLoadRatio(t *testing.T) {
	cases := []struct{ name, parameters, want string }{
		{"range order", `{"workload":{"kv_load_ratio":[0.9,0.1]}}`, "needs min < max"},
		{"range length", `{"workload":{"kv_load_ratio":[0,0.5,1]}}`, "exactly [min, max]"},
		{"scalar below zero", `{"workload":{"kv_load_ratio":-1}}`, "between 0 and 1"},
		{"scalar above one", `{"workload":{"kv_load_ratio":1.5}}`, "between 0 and 1"},
		{"range upper bound above one", `{"workload":{"kv_load_ratio":[0.2,1.5]}}`, "between 0 and 1"},
		{"range lower bound below zero", `{"workload":{"kv_load_ratio":[-0.1,0.5]}}`, "between 0 and 1"},
		{"non-numeric range entry", `{"workload":{"kv_load_ratio":[0,"x"]}}`, "must be numbers"},
		{"wrong type", `{"workload":{"kv_load_ratio":"all"}}`, "number or a [min, max] range"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), tc.parameters))
			v1b2RequireError(t, errs, "spec.aisimulate.parameters.workload.kv_load_ratio")
			v1b2RequireError(t, errs, tc.want)
		})
	}
	for _, parameters := range []string{
		`{"workload":{"kv_load_ratio":1}}`,
		`{"workload":{"kv_load_ratio":0}}`,
		`{"workload":{"kv_load_ratio":[0,1]}}`,
		`{"workload":{"kv_load_ratio":[0.25,0.75]}}`,
	} {
		v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), parameters)))
	}
}

func TestEffectiveInputSearchSpace(t *testing.T) {
	t.Run("unresolved hardware", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.Hardware = nil
		errs := validateDynamoGraphDeploymentRequestV1beta2(spec)
		v1b2RequireError(t, errs, "spec.hardware.gpu.sku: effective hardware is unresolved")
		v1b2RequireError(t, errs, "spec.hardware.gpu.budget: effective hardware is unresolved")
	})
	t.Run("partial hardware", func(t *testing.T) {
		spec := v1b2DefaultedSpec()
		spec.Hardware.GPU.Budget = nil
		errs := validateDynamoGraphDeploymentRequestV1beta2(spec)
		if len(errs) != 1 || errs[0].Field != "spec.hardware.gpu.budget" {
			t.Fatalf("want only the budget unresolved, got:\n%s", v1b2Details(errs))
		}
	})
	t.Run("native hardware completes the typed one", func(t *testing.T) {
		spec := v1b2WithParameters(v1b2DefaultedSpec(), `{"search_space":{"hardware_sku":"h100_sxm","gpu_budget":4}}`)
		spec.Hardware = nil
		v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(spec))
	})
	t.Run("a native budget below one", func(t *testing.T) {
		errs := validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), `{"search_space":{"gpu_budget":0}}`))
		v1b2RequireError(t, errs, "spec.aisimulate.parameters.search_space.gpu_budget: effective search_space.gpu_budget must be an integer >= 1")
	})
	t.Run("a fractional native budget", func(t *testing.T) {
		v1b2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), `{"search_space":{"gpu_budget":2.5}}`)), "integer")
	})
	t.Run("unknown backend", func(t *testing.T) {
		errs := validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), `{"search_space":{"backend":["tgi"]}}`))
		v1b2RequireError(t, errs, "must be one of vllm, sglang, trtllm")
	})
	t.Run("backends deleted", func(t *testing.T) {
		errs := validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), `{"search_space":{"backend":null}}`))
		v1b2RequireError(t, errs, "spec.aisimulate.parameters.search_space.backend: effective search_space.backend must list at least one backend")
	})
	t.Run("a native blank model name", func(t *testing.T) {
		errs := validateDynamoGraphDeploymentRequestV1beta2(v1b2WithParameters(v1b2DefaultedSpec(), `{"search_space":{"model_name":""}}`))
		v1b2RequireError(t, errs, "spec.aisimulate.parameters.search_space.model_name: effective search_space.model_name is empty")
	})
}

func TestEffectiveInputPathAttribution(t *testing.T) {
	e := effectiveInput{parameters: map[string]any{"workload": map[string]any{"isl": nil}}}
	for path, want := range map[string]string{
		"workload.isl":            "spec.aisimulate.parameters.workload.isl", // the parameters name it
		"workload.osl":            "spec.workload.static.osl",                // only the typed field produces it
		"workload.request_count":  "spec.aisimulate.parameters.workload.request_count",
		"sweep.max_trials":        "spec.aisimulate.parameters.sweep.max_trials",
		"goal.sla.ttft_ms":        "spec.objective.sla.ttftMs",
		"search_space.gpu_budget": "spec.hardware.gpu.budget",
	} {
		if got := e.at(strings.Split(path, ".")).String(); got != want {
			t.Errorf("at(%s) = %s, want %s", path, got, want)
		}
	}
}

func v1b2EnvelopeSpec(t *testing.T, raw string) *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec {
	t.Helper()
	return &nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{V1Beta2: &apiextensionsv1.JSON{Raw: []byte(raw)}}
}

func TestValidateEnvelopeV1beta2(t *testing.T) {
	path := field.NewPath("spec")

	good, err := json.Marshal(v1b2DefaultedSpec())
	if err != nil {
		t.Fatal(err)
	}
	v1b2RequireNoError(t, validateDynamoGraphDeploymentRequestEnvelopeV1beta2(v1b2EnvelopeSpec(t, string(good)), path))

	bad := v1b2DefaultedSpec()
	bad.Model.Name = ""
	raw, _ := json.Marshal(bad)
	v1b2RequireError(t, validateDynamoGraphDeploymentRequestEnvelopeV1beta2(v1b2EnvelopeSpec(t, string(raw)), path), "model")

	v1b2RequireError(t, validateDynamoGraphDeploymentRequestEnvelopeV1beta2(v1b2EnvelopeSpec(t, `{"model": 3}`), path), "valid v1beta2 spec")
}
