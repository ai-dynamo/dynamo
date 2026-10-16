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
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/aisimulate"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

var (
	knownBackends = map[string]bool{"vllm": true, "sglang": true, "trtllm": true}

	// scalarTargets are the goal.target values Dynamo accepts. min_gpus exists in
	// AISimulate but is not supported by the Dynamo integration.
	scalarTargets = map[string]bool{
		"throughput":          true,
		"throughput_per_gpu":  true,
		"throughput_per_user": true,
		"ttft":                true,
		"e2e_latency":         true,
		"goodput":             true,
		"goodput_per_gpu":     true,
	}

	slaTargets = map[string]bool{"goodput": true, "goodput_per_gpu": true}

	// typedPaths maps a native input path to the typed field that produces it. A native
	// path without an entry has no typed counterpart.
	typedPaths = map[string]*field.Path{
		"search_space.model_name":    field.NewPath("spec", "model", "name"),
		"search_space.backend":       field.NewPath("spec", "backends"),
		"search_space.hardware_sku":  field.NewPath("spec", "hardware", "gpu", "sku"),
		"search_space.gpu_budget":    field.NewPath("spec", "hardware", "gpu", "budget"),
		"workload":                   field.NewPath("spec", "workload", "static"),
		"workload.isl":               field.NewPath("spec", "workload", "static", "isl"),
		"workload.osl":               field.NewPath("spec", "workload", "static", "osl"),
		"workload.num_request_ratio": field.NewPath("spec", "workload", "static", "numRequestRatio"),
		"workload.request_rate":      field.NewPath("spec", "workload", "static", "requestRate"),
		"workload.concurrency":       field.NewPath("spec", "workload", "static", "concurrency"),
		"workload.kv_load_ratio":     field.NewPath("spec", "workload", "static", "kvLoadRatio"),
		"goal":                       field.NewPath("spec", "objective"),
		"goal.target":                field.NewPath("spec", "objective", "metric"),
		"goal.pareto_objectives":     field.NewPath("spec", "objective", "metrics"),
		"goal.sla":                   field.NewPath("spec", "objective", "sla"),
		"goal.sla.ttft_ms":           field.NewPath("spec", "objective", "sla", "ttftMs"),
		"goal.sla.itl_ms":            field.NewPath("spec", "objective", "sla", "itlMs"),
		"goal.sla.e2e_ms":            field.NewPath("spec", "objective", "sla", "e2eMs"),
	}
)

// effectiveInput is a lowered AISimulate input together with the spec.aisimulate.parameters
// that were merged into it.
type effectiveInput struct {
	input      map[string]any
	parameters map[string]any
}

// get returns the value at the native path, or nil when it is absent.
func (e effectiveInput) get(path []string) any {
	value, _, _ := unstructured.NestedFieldNoCopy(e.input, path...)
	return value
}

// at returns the field to blame for the native path: the parameter when the parameters
// name the key, otherwise the typed field that produced the value, otherwise the parameter
// the caller would have to add.
func (e effectiveInput) at(path []string) *field.Path {
	if !aisimulate.Has(e.parameters, path...) {
		if typed, ok := typedPaths[strings.Join(path, ".")]; ok {
			return typed
		}
	}
	return parametersPath.Child(path[0], path[1:]...)
}

// validateEffectiveAISimulateInput checks the complete effective input the way AISimulate's
// own validators would, so a Request that cannot run is rejected at admission. It mirrors
// the cross-field rules the schema cannot express: the merged workload, goal and search
// space. It does not check unknown native keys; that is the job of the version-bound
// schema. parameters is spec.aisimulate.parameters, used only to place errors.
func validateEffectiveAISimulateInput(input, parameters map[string]any) field.ErrorList {
	e := effectiveInput{input: input, parameters: parameters}
	var errs field.ErrorList
	errs = append(errs, e.validateSearchSpace()...)
	errs = append(errs, e.validateWorkload()...)
	errs = append(errs, e.validateGoal()...)
	return errs
}

func (e effectiveInput) validateSearchSpace() field.ErrorList {
	var errs field.ErrorList
	if name, _, _ := unstructured.NestedString(e.input, aisimulate.SearchSpaceModelName...); strings.TrimSpace(name) == "" {
		errs = append(errs, field.Required(e.at(aisimulate.SearchSpaceModelName), "effective search_space.model_name is empty"))
	}

	backends, _, _ := unstructured.NestedSlice(e.input, aisimulate.SearchSpaceBackend...)
	if len(backends) == 0 {
		errs = append(errs, field.Required(e.at(aisimulate.SearchSpaceBackend),
			"effective search_space.backend must list at least one backend"))
	}
	for _, entry := range backends {
		if name, _ := entry.(string); !knownBackends[name] {
			errs = append(errs, field.Invalid(e.at(aisimulate.SearchSpaceBackend), entry, "must be one of vllm, sglang, trtllm"))
		}
	}

	if sku, _, _ := unstructured.NestedString(e.input, aisimulate.SearchSpaceHardwareSKU...); strings.TrimSpace(sku) == "" {
		errs = append(errs, field.Required(e.at(aisimulate.SearchSpaceHardwareSKU),
			"effective hardware is unresolved: set hardware.gpu.sku, or enable GPU discovery"))
	}
	switch raw := e.get(aisimulate.SearchSpaceGPUBudget); {
	case raw == nil:
		errs = append(errs, field.Required(e.at(aisimulate.SearchSpaceGPUBudget),
			"effective hardware is unresolved: set hardware.gpu.budget, or enable GPU discovery"))
	default:
		if budget, ok := raw.(float64); !ok || !isInteger(budget) || budget < 1 {
			errs = append(errs, field.Invalid(e.at(aisimulate.SearchSpaceGPUBudget), raw,
				"effective search_space.gpu_budget must be an integer >= 1"))
		}
	}
	return errs
}

func (e effectiveInput) validateWorkload() field.ErrorList {
	workload := e.at(aisimulate.Workload)
	if _, found, _ := unstructured.NestedMap(e.input, aisimulate.Workload...); !found {
		return field.ErrorList{field.Required(workload,
			"effective workload is empty: set workload.static or aisimulate.parameters.workload")}
	}
	var errs field.ErrorList
	for _, path := range [][]string{aisimulate.WorkloadTracePath, aisimulate.WorkloadTracePaths} {
		if e.get(path) != nil {
			errs = append(errs, field.Forbidden(e.at(path), "trace workloads are not supported; remove workload."+path[1]))
		}
	}

	var loads []string
	for _, path := range [][]string{aisimulate.WorkloadRequestRate, aisimulate.WorkloadConcurrency, aisimulate.WorkloadKVLoadRatio} {
		if e.get(path) != nil {
			loads = append(loads, path[1])
		}
	}
	if len(loads) != 1 {
		errs = append(errs, field.Invalid(workload, loads, fmt.Sprintf(
			"effective workload needs exactly one of request_rate, concurrency, or kv_load_ratio, got %d", len(loads))))
	}

	var missing []string
	for _, path := range [][]string{aisimulate.WorkloadISL, aisimulate.WorkloadOSL} {
		if e.get(path) == nil {
			missing = append(missing, path[1])
		}
	}
	if e.get(aisimulate.WorkloadRequestCount) == nil && e.get(aisimulate.WorkloadNumRequestRatio) == nil {
		missing = append(missing, "request_count or num_request_ratio")
	}
	if len(missing) > 0 {
		errs = append(errs, field.Required(workload, "effective workload requires "+strings.Join(missing, ", ")))
	}

	for _, path := range [][]string{aisimulate.WorkloadISL, aisimulate.WorkloadOSL, aisimulate.WorkloadConcurrency, aisimulate.WorkloadRequestCount} {
		if raw := e.get(path); raw != nil {
			if value, ok := raw.(float64); !ok || !isInteger(value) || value < 1 {
				errs = append(errs, field.Invalid(e.at(path), raw, path[1]+" must be a positive integer"))
			}
		}
	}
	for _, path := range [][]string{aisimulate.WorkloadRequestRate, aisimulate.WorkloadNumRequestRatio} {
		if raw := e.get(path); raw != nil {
			if value, ok := raw.(float64); !ok || !(value > 0) || math.IsInf(value, 0) {
				errs = append(errs, field.Invalid(e.at(path), raw, path[1]+" must be a positive number"))
			}
		}
	}
	errs = append(errs, e.validateKVLoadRatio()...)
	return errs
}

// validateKVLoadRatio checks the scalar form and both entries of the [min, max] form of
// the effective kv_load_ratio. The schema bounds the typed field to [0, 1]; this applies the
// same closed interval after spec.aisimulate.parameters has overridden it.
func (e effectiveInput) validateKVLoadRatio() field.ErrorList {
	path := e.at(aisimulate.WorkloadKVLoadRatio)
	switch typed := e.get(aisimulate.WorkloadKVLoadRatio).(type) {
	case nil:
		return nil
	case float64:
		return validateRatios([]float64{typed}, typed, path)
	case []any:
		if len(typed) != 2 {
			return field.ErrorList{field.Invalid(path, typed, "kv_load_ratio range must contain exactly [min, max]")}
		}
		ratios := make([]float64, 0, 2)
		for _, entry := range typed {
			ratio, ok := entry.(float64)
			if !ok {
				return field.ErrorList{field.Invalid(path, typed, "kv_load_ratio values must be numbers")}
			}
			ratios = append(ratios, ratio)
		}
		if errs := validateRatios(ratios, typed, path); len(errs) > 0 {
			return errs
		}
		if ratios[0] >= ratios[1] {
			return field.ErrorList{field.Invalid(path, typed, "kv_load_ratio range needs min < max")}
		}
		return nil
	default:
		return field.ErrorList{field.Invalid(path, typed, "kv_load_ratio must be a number or a [min, max] range")}
	}
}

// validateRatios requires every ratio to be finite and within [0, 1]: a ratio is a fraction
// of the candidate's estimated KV-token capacity, and 1 is the full modeled capacity.
func validateRatios(ratios []float64, value any, path *field.Path) field.ErrorList {
	for _, ratio := range ratios {
		if math.IsNaN(ratio) || math.IsInf(ratio, 0) || ratio < 0 || ratio > 1 {
			return field.ErrorList{field.Invalid(path, value, "kv_load_ratio values must be finite and between 0 and 1")}
		}
	}
	return nil
}

func (e effectiveInput) validateGoal() field.ErrorList {
	target, _, _ := unstructured.NestedString(e.input, aisimulate.GoalTarget...)
	if target == "" {
		return field.ErrorList{field.Required(e.at(aisimulate.GoalTarget),
			"effective goal.target is missing: set objective.metric (admission defaults it to ThroughputPerGPU) or aisimulate.parameters.goal.target")}
	}
	if target == "min_gpus" {
		return field.ErrorList{field.Forbidden(e.at(aisimulate.GoalTarget), "goal.target min_gpus is not supported by the Dynamo integration")}
	}

	var errs field.ErrorList
	pareto := target == aisimulate.TargetPareto
	objectives := e.at(aisimulate.GoalParetoObjectives)
	effective := map[string]bool{target: true}
	switch {
	case !pareto && !scalarTargets[target]:
		return field.ErrorList{field.Invalid(e.at(aisimulate.GoalTarget), target, "unknown goal.target")}
	case !pareto && e.get(aisimulate.GoalParetoObjectives) != nil:
		errs = append(errs, field.Invalid(objectives, e.get(aisimulate.GoalParetoObjectives),
			"goal.pareto_objectives is only valid when goal.target is pareto"))
	case pareto:
		effective = map[string]bool{}
		entries, _, _ := unstructured.NestedSlice(e.input, aisimulate.GoalParetoObjectives...)
		if len(entries) < 2 {
			errs = append(errs, field.Invalid(objectives, entries, "a Pareto objective needs at least two metrics"))
			break
		}
		for _, entry := range entries {
			name, _ := entry.(string)
			switch {
			case !scalarTargets[name]:
				errs = append(errs, field.Invalid(objectives, entry, "Pareto metrics must be scalar targets"))
			case effective[name]:
				errs = append(errs, field.Invalid(objectives, entry, "Pareto metrics must be distinct"))
			default:
				effective[name] = true
			}
		}
	}

	errs = append(errs, e.validateSLA()...)
	bounded := aisimulate.HasSLABound(e.input)
	var needing []string
	for name := range effective {
		if slaTargets[name] {
			needing = append(needing, name)
		}
	}
	sort.Strings(needing)
	if len(needing) > 0 && !bounded {
		errs = append(errs, field.Required(e.at(aisimulate.GoalSLA), fmt.Sprintf("%v require at least one SLA bound", needing)))
	}
	switch strict := e.get(aisimulate.GoalStrictSLA).(type) {
	case nil:
	case bool:
		if strict && !bounded {
			errs = append(errs, field.Invalid(e.at(aisimulate.GoalStrictSLA), strict, "goal.strict_sla requires at least one SLA bound"))
		}
	default:
		errs = append(errs, field.Invalid(e.at(aisimulate.GoalStrictSLA), strict, "goal.strict_sla must be a boolean"))
	}
	if e.get(aisimulate.GoalMinGoodputRPS) != nil {
		errs = append(errs, field.Forbidden(e.at(aisimulate.GoalMinGoodputRPS),
			"goal.min_goodput_rps is only supported with min_gpus, which Dynamo does not support"))
	}

	if _, isRange := e.get(aisimulate.WorkloadKVLoadRatio).([]any); isRange && !pareto && e.get(aisimulate.SweepMaxTrials) == nil {
		errs = append(errs, field.Required(e.at(aisimulate.SweepMaxTrials),
			"a kv_load_ratio range requires a Pareto objective or sweep.max_trials"))
	}
	return errs
}

func (e effectiveInput) validateSLA() field.ErrorList {
	var errs field.ErrorList
	for _, path := range [][]string{aisimulate.GoalSLATTFTMs, aisimulate.GoalSLAITLMs, aisimulate.GoalSLAE2EMs} {
		if raw := e.get(path); raw != nil {
			if value, ok := raw.(float64); !ok || !(value > 0) || math.IsInf(value, 0) {
				errs = append(errs, field.Invalid(e.at(path), raw, path[2]+" must be a positive number"))
			}
		}
	}
	if e.get(aisimulate.GoalSLAE2EMs) != nil && (e.get(aisimulate.GoalSLATTFTMs) != nil || e.get(aisimulate.GoalSLAITLMs) != nil) {
		errs = append(errs, field.Invalid(e.at(aisimulate.GoalSLA), e.get(aisimulate.GoalSLA), "e2e_ms cannot be combined with ttft_ms or itl_ms"))
	}
	return errs
}

func isInteger(value float64) bool {
	return value == math.Trunc(value) && math.Abs(value) < 1<<53
}
