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
	"fmt"
	"math"
	"sort"
	"strings"

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
)

// Validate checks the complete effective input the way AISimulate's own validators
// would, so a Request that cannot run is rejected at admission. It mirrors the
// cross-field rules the CRD cannot express: the merged workload, goal and search space.
// It does not check unknown native keys; that is the job of the version-bound schema.
func Validate(input map[string]any) field.ErrorList {
	var errs field.ErrorList
	spec := field.NewPath("spec")
	errs = append(errs, validateSearchSpace(asMap(input["search_space"]), spec)...)
	errs = append(errs, validateWorkload(asMap(input["workload"]), spec.Child("workload"))...)
	errs = append(errs, validateGoal(asMap(input["goal"]), asMap(input["workload"]), asMap(input["sweep"]), spec.Child("objective"))...)
	return errs
}

func validateSearchSpace(searchSpace map[string]any, spec *field.Path) field.ErrorList {
	var errs field.ErrorList
	if name, _ := searchSpace["model_name"].(string); strings.TrimSpace(name) == "" {
		errs = append(errs, field.Required(spec.Child("model", "name"), "effective search_space.model_name is empty"))
	}
	switch backends, ok := searchSpace["backend"].([]any); {
	case !ok || len(backends) == 0:
		errs = append(errs, field.Required(spec.Child("backends"), "effective search_space.backend must list at least one backend"))
	default:
		for _, entry := range backends {
			if name, _ := entry.(string); !knownBackends[name] {
				errs = append(errs, field.Invalid(spec.Child("backends"), entry, "must be one of vllm, sglang, trtllm"))
			}
		}
	}
	hardware := spec.Child("hardware", "gpu")
	if sku, _ := searchSpace["hardware_sku"].(string); strings.TrimSpace(sku) == "" {
		errs = append(errs, field.Required(hardware.Child("sku"),
			"effective hardware is unresolved: set hardware.gpu.sku or enable GPU discovery"))
	}
	budget, ok := number(searchSpace["gpu_budget"])
	switch {
	case searchSpace["gpu_budget"] == nil:
		errs = append(errs, field.Required(hardware.Child("budget"),
			"effective hardware is unresolved: set hardware.gpu.budget or enable GPU discovery"))
	case !ok || !isInteger(budget) || budget < 1:
		errs = append(errs, field.Invalid(hardware.Child("budget"), searchSpace["gpu_budget"], "effective search_space.gpu_budget must be an integer >= 1"))
	}
	return errs
}

func validateWorkload(workload map[string]any, path *field.Path) field.ErrorList {
	if workload == nil {
		return field.ErrorList{field.Required(path,
			"effective workload is empty: set workload.static or aisimulate.parameters.workload")}
	}
	var errs field.ErrorList
	for _, key := range []string{"trace_path", "trace_paths"} {
		if workload[key] != nil {
			errs = append(errs, field.Forbidden(path, "trace workloads are not supported; remove workload."+key))
		}
	}

	var loads []string
	for _, key := range []string{"request_rate", "concurrency", "kv_load_ratio"} {
		if workload[key] != nil {
			loads = append(loads, key)
		}
	}
	if len(loads) != 1 {
		errs = append(errs, field.Invalid(path, loads, fmt.Sprintf(
			"effective workload needs exactly one of request_rate, concurrency, or kv_load_ratio, got %d", len(loads))))
	}

	var missing []string
	for _, key := range []string{"isl", "osl"} {
		if workload[key] == nil {
			missing = append(missing, key)
		}
	}
	if workload["request_count"] == nil && workload["num_request_ratio"] == nil {
		missing = append(missing, "request_count or num_request_ratio")
	}
	if len(missing) > 0 {
		errs = append(errs, field.Required(path, "effective workload requires "+strings.Join(missing, ", ")))
	}

	for _, key := range []string{"isl", "osl", "concurrency", "request_count"} {
		if workload[key] == nil {
			continue
		}
		if value, ok := number(workload[key]); !ok || !isInteger(value) || value < 1 {
			errs = append(errs, field.Invalid(path, workload[key], key+" must be a positive integer"))
		}
	}
	for _, key := range []string{"request_rate", "num_request_ratio"} {
		if workload[key] == nil {
			continue
		}
		if value, ok := number(workload[key]); !ok || !(value > 0) || math.IsInf(value, 0) {
			errs = append(errs, field.Invalid(path, workload[key], key+" must be a positive number"))
		}
	}
	errs = append(errs, validateKVLoadRatio(workload["kv_load_ratio"], path)...)
	return errs
}

func validateKVLoadRatio(value any, path *field.Path) field.ErrorList {
	switch typed := value.(type) {
	case nil:
		return nil
	case float64:
		return validateRatios([]float64{typed}, value, path)
	case []any:
		if len(typed) != 2 {
			return field.ErrorList{field.Invalid(path, value, "kv_load_ratio range must contain exactly [min, max]")}
		}
		ratios := make([]float64, 0, 2)
		for _, entry := range typed {
			ratio, ok := number(entry)
			if !ok {
				return field.ErrorList{field.Invalid(path, value, "kv_load_ratio values must be numbers")}
			}
			ratios = append(ratios, ratio)
		}
		if errs := validateRatios(ratios, value, path); len(errs) > 0 {
			return errs
		}
		if ratios[0] >= ratios[1] {
			return field.ErrorList{field.Invalid(path, value, "kv_load_ratio range needs min < max")}
		}
		return nil
	default:
		return field.ErrorList{field.Invalid(path, value, "kv_load_ratio must be a number or a [min, max] range")}
	}
}

func validateRatios(ratios []float64, value any, path *field.Path) field.ErrorList {
	for _, ratio := range ratios {
		if math.IsNaN(ratio) || math.IsInf(ratio, 0) || ratio < 0 {
			return field.ErrorList{field.Invalid(path, value, "kv_load_ratio values must be finite and non-negative")}
		}
	}
	return nil
}

func validateGoal(goal, workload, sweep map[string]any, path *field.Path) field.ErrorList {
	var errs field.ErrorList
	target, _ := goal["target"].(string)
	if target == "" {
		return field.ErrorList{field.Required(path.Child("metric"),
			"effective goal.target is missing: set objective.metric (admission defaults it to ThroughputPerGPU) or aisimulate.parameters.goal.target")}
	}
	if target == "min_gpus" {
		return field.ErrorList{field.Forbidden(path, "goal.target min_gpus is not supported by the Dynamo integration")}
	}

	pareto := target == nativePareto
	effective := map[string]bool{target: true}
	switch {
	case !pareto && !scalarTargets[target]:
		return field.ErrorList{field.Invalid(path.Child("metric"), target, "unknown goal.target")}
	case !pareto && goal["pareto_objectives"] != nil:
		errs = append(errs, field.Invalid(path.Child("metrics"), goal["pareto_objectives"], "goal.pareto_objectives is only valid when goal.target is pareto"))
	case pareto:
		effective = map[string]bool{}
		objectives, ok := goal["pareto_objectives"].([]any)
		if !ok || len(objectives) < 2 {
			errs = append(errs, field.Invalid(path.Child("metrics"), goal["pareto_objectives"], "a Pareto objective needs at least two metrics"))
			break
		}
		for _, entry := range objectives {
			name, _ := entry.(string)
			switch {
			case !scalarTargets[name]:
				errs = append(errs, field.Invalid(path.Child("metrics"), entry, "Pareto metrics must be scalar targets"))
			case effective[name]:
				errs = append(errs, field.Invalid(path.Child("metrics"), entry, "Pareto metrics must be distinct"))
			default:
				effective[name] = true
			}
		}
	}

	sla := asMap(goal["sla"])
	errs = append(errs, validateSLA(sla, path.Child("sla"))...)
	bounded := hasBound(sla)

	var needing []string
	for name := range effective {
		if slaTargets[name] {
			needing = append(needing, name)
		}
	}
	sort.Strings(needing)
	if len(needing) > 0 && !bounded {
		errs = append(errs, field.Required(path.Child("sla"), fmt.Sprintf("%v require at least one SLA bound", needing)))
	}
	switch strict := goal["strict_sla"].(type) {
	case nil:
	case bool:
		if strict && !bounded {
			errs = append(errs, field.Invalid(path.Child("sla"), nil, "goal.strict_sla requires at least one SLA bound"))
		}
	default:
		errs = append(errs, field.Invalid(path, goal["strict_sla"], "goal.strict_sla must be a boolean"))
	}
	if goal["min_goodput_rps"] != nil {
		errs = append(errs, field.Forbidden(path, "goal.min_goodput_rps is only supported with min_gpus, which Dynamo does not support"))
	}

	if _, isRange := workload["kv_load_ratio"].([]any); isRange && !pareto && sweep["max_trials"] == nil {
		errs = append(errs, field.Required(path, "a kv_load_ratio range requires a Pareto objective or sweep.max_trials"))
	}
	return errs
}

func validateSLA(sla map[string]any, path *field.Path) field.ErrorList {
	var errs field.ErrorList
	for _, key := range []string{"ttft_ms", "itl_ms", "e2e_ms"} {
		if sla[key] == nil {
			continue
		}
		if value, ok := number(sla[key]); !ok || !(value > 0) || math.IsInf(value, 0) {
			errs = append(errs, field.Invalid(path, sla[key], key+" must be a positive number"))
		}
	}
	if sla["e2e_ms"] != nil && (sla["ttft_ms"] != nil || sla["itl_ms"] != nil) {
		errs = append(errs, field.Invalid(path, sla, "e2e_ms cannot be combined with ttft_ms or itl_ms"))
	}
	return errs
}

func number(value any) (float64, bool) {
	f, ok := value.(float64)
	return f, ok
}

func isInteger(value float64) bool {
	return value == math.Trunc(value) && math.Abs(value) < 1<<53
}
