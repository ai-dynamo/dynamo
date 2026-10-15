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
	"fmt"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
)

// Defaults that are not expressible in the CRD schema.
const (
	// DefaultRequestCount is workload.request_count when the effective input does not
	// choose num_request_ratio.
	DefaultRequestCount int64 = 1000

	// DefaultMaxTrials bounds a KV-load range search that has no explicit sweep.max_trials.
	// It matches the trial budget of AISimulate's unified CLI (optimizer.max_trials).
	DefaultMaxTrials int64 = 320

	// DefaultTTFTMs and DefaultITLMs mirror the kubebuilder default of
	// objective.sla. They let the conflict check tell a value the user wrote from one the
	// schema filled in. Keep them in step with the marker on ObjectiveSpec.SLA.
	DefaultTTFTMs float64 = 2000
	DefaultITLMs  float64 = 30

	// DefaultMetric is the scalar target admission supplies in Optimize mode.
	DefaultMetric = v1beta2.ObjectiveMetric("ThroughputPerGPU")
)

// Options tunes the derived defaults. A zero field disables that default.
type Options struct {
	RequestCount int64
	MaxTrials    int64
}

// DefaultOptions returns the production defaults.
func DefaultOptions() Options {
	return Options{RequestCount: DefaultRequestCount, MaxTrials: DefaultMaxTrials}
}

// Build returns the effective native input: the typed underlay, merged with
// spec.aisimulate.parameters, plus the derived defaults. The only error is malformed
// parameters (not a JSON object); semantic problems are reported by Validate.
func Build(spec *v1beta2.DynamoGraphDeploymentRequestSpec, opts Options) (map[string]any, error) {
	underlay, err := normalize(Underlay(spec))
	if err != nil {
		return nil, err
	}
	patch, err := Parameters(spec)
	if err != nil {
		return nil, err
	}
	merged, _ := mergePatch(underlay, patch).(map[string]any)
	if merged == nil {
		merged = map[string]any{}
	}
	applyDefaults(merged, patch, opts)
	return merged, nil
}

// JSON returns the canonical encoding of an effective input. Object keys are sorted, so
// equal inputs produce equal bytes.
func JSON(input map[string]any) ([]byte, error) {
	return json.Marshal(input)
}

// Parameters returns spec.aisimulate.parameters as a JSON object, or nil when absent.
func Parameters(spec *v1beta2.DynamoGraphDeploymentRequestSpec) (map[string]any, error) {
	if spec.AISimulate == nil || spec.AISimulate.Parameters == nil {
		return nil, nil
	}
	raw := spec.AISimulate.Parameters.Raw
	if len(raw) == 0 && spec.AISimulate.Parameters.Object != nil {
		var err error
		if raw, err = json.Marshal(spec.AISimulate.Parameters.Object); err != nil {
			return nil, fmt.Errorf("spec.aisimulate.parameters: %w", err)
		}
	}
	if len(raw) == 0 {
		return nil, nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, fmt.Errorf("spec.aisimulate.parameters: %w", err)
	}
	switch typed := value.(type) {
	case nil:
		return nil, nil
	case map[string]any:
		return typed, nil
	default:
		return nil, fmt.Errorf("spec.aisimulate.parameters: must be a JSON object, got %T", value)
	}
}

// NativeSetsTarget reports whether spec.aisimulate.parameters mentions goal.target, even
// to delete it. Admission then leaves objective.metric unset: the native input owns the
// scalar target, and a defaulted typed metric would conflict with it.
func NativeSetsTarget(spec *v1beta2.DynamoGraphDeploymentRequestSpec) bool {
	patch, err := Parameters(spec)
	return err == nil && mentions(patch, "goal", "target")
}

// MetricDefault returns the metric admission should set: ThroughputPerGPU for an Optimize
// objective without a metric whose target the native input does not choose.
func MetricDefault(spec *v1beta2.DynamoGraphDeploymentRequestSpec) (v1beta2.ObjectiveMetric, bool) {
	objective := spec.Objective
	if objective == nil {
		return "", false
	}
	if objective.Mode != "" && objective.Mode != v1beta2.ObjectiveModeOptimize {
		return "", false
	}
	if objective.Metric != nil || NativeSetsTarget(spec) {
		return "", false
	}
	return DefaultMetric, true
}

// Hardware reports the effective hardware: the SKU, or "" when unresolved, and whether a
// GPU budget is set. Typed fields and spec.aisimulate.parameters.search_space both count,
// so admission discovers only what neither provides.
func Hardware(spec *v1beta2.DynamoGraphDeploymentRequestSpec) (sku string, hasBudget bool, err error) {
	input, err := Build(spec, Options{})
	if err != nil {
		return "", false, err
	}
	searchSpace := asMap(input["search_space"])
	sku, _ = searchSpace["hardware_sku"].(string)
	return sku, searchSpace["gpu_budget"] != nil, nil
}

// Underlay translates the typed spec into native input. Typed fields that are unset
// produce no key, so the native defaults and the parameters decide them.
func Underlay(spec *v1beta2.DynamoGraphDeploymentRequestSpec) map[string]any {
	root := map[string]any{}

	searchSpace := map[string]any{"model_name": spec.Model.Name}
	if len(spec.Backends) > 0 {
		backends := make([]any, len(spec.Backends))
		for i, backend := range spec.Backends {
			backends[i] = string(backend)
		}
		searchSpace["backend"] = backends
	}
	if spec.Hardware != nil && spec.Hardware.GPU != nil {
		if sku := spec.Hardware.GPU.SKU; sku != nil && *sku != "" {
			searchSpace["hardware_sku"] = *sku
		}
		if budget := spec.Hardware.GPU.Budget; budget != nil {
			searchSpace["gpu_budget"] = int64(*budget)
		}
	}
	root["search_space"] = searchSpace

	if spec.Workload != nil && spec.Workload.Static != nil {
		if workload := workloadUnderlay(spec.Workload.Static); len(workload) > 0 {
			root["workload"] = workload
		}
	}
	if goal := goalUnderlay(spec.Objective); len(goal) > 0 {
		root["goal"] = goal
	}
	return root
}

func workloadUnderlay(static *v1beta2.StaticWorkloadSpec) map[string]any {
	workload := map[string]any{}
	if static.ISL != nil {
		workload["isl"] = int64(*static.ISL)
	}
	if static.OSL != nil {
		workload["osl"] = int64(*static.OSL)
	}
	if static.NumRequestRatio != nil {
		workload["num_request_ratio"] = *static.NumRequestRatio
	}
	if static.RequestRate != nil {
		workload["request_rate"] = *static.RequestRate
	}
	if static.Concurrency != nil {
		workload["concurrency"] = int64(*static.Concurrency)
	}
	switch len(static.KVLoadRatio) {
	case 0:
	case 1:
		workload["kv_load_ratio"] = static.KVLoadRatio[0]
	default:
		ratios := make([]any, len(static.KVLoadRatio))
		for i, ratio := range static.KVLoadRatio {
			ratios[i] = ratio
		}
		workload["kv_load_ratio"] = ratios
	}
	return workload
}

func goalUnderlay(objective *v1beta2.ObjectiveSpec) map[string]any {
	goal := map[string]any{}
	if objective == nil {
		return goal
	}
	if objective.Mode == v1beta2.ObjectiveModePareto {
		goal["target"] = nativePareto
		if len(objective.Metrics) > 0 {
			goal["pareto_objectives"] = nativeMetrics(objective.Metrics)
		}
	} else if objective.Metric != nil {
		goal["target"] = nativeMetric(*objective.Metric)
	}
	if sla := slaUnderlay(objective.SLA); len(sla) > 0 {
		goal["sla"] = sla
	}
	return goal
}

func slaUnderlay(sla *v1beta2.ObjectiveSLASpec) map[string]any {
	out := map[string]any{}
	if sla == nil {
		return out
	}
	if sla.TTFTMs != nil {
		out["ttft_ms"] = *sla.TTFTMs
	}
	if sla.ITLMs != nil {
		out["itl_ms"] = *sla.ITLMs
	}
	if sla.E2EMs != nil {
		out["e2e_ms"] = *sla.E2EMs
	}
	return out
}

const nativePareto = "pareto"

var nativeMetricNames = map[v1beta2.ObjectiveMetric]string{
	"Throughput":        "throughput",
	"ThroughputPerGPU":  "throughput_per_gpu",
	"ThroughputPerUser": "throughput_per_user",
	"TTFT":              "ttft",
	"E2ELatency":        "e2e_latency",
	"Goodput":           "goodput",
	"GoodputPerGPU":     "goodput_per_gpu",
}

// nativeMetric maps an API metric to AISimulate's goal.target. An unknown metric is
// passed through so Validate reports it instead of silently dropping it.
func nativeMetric(metric v1beta2.ObjectiveMetric) string {
	if name, ok := nativeMetricNames[metric]; ok {
		return name
	}
	return string(metric)
}

func nativeMetrics(metrics []v1beta2.ObjectiveMetric) []any {
	out := make([]any, len(metrics))
	for i, metric := range metrics {
		out[i] = nativeMetric(metric)
	}
	return out
}

// applyDefaults adds the derived defaults after the merge. A default is skipped when the
// parameters mention its key, including with null: an explicit native value overrides or
// deletes the default.
func applyDefaults(merged, patch map[string]any, opts Options) {
	if workload := asMap(merged["workload"]); workload != nil {
		_, hasCount := workload["request_count"]
		_, hasRatio := workload["num_request_ratio"]
		if opts.RequestCount > 0 && !hasCount && !hasRatio && !mentions(patch, "workload", "request_count") {
			workload["request_count"] = float64(opts.RequestCount)
		}
		if _, isRange := workload["kv_load_ratio"].([]any); isRange && opts.MaxTrials > 0 {
			sweep := asMap(merged["sweep"])
			if sweep == nil {
				sweep = map[string]any{}
			}
			if _, has := sweep["max_trials"]; !has && !mentions(patch, "sweep", "max_trials") {
				sweep["max_trials"] = float64(opts.MaxTrials)
				merged["sweep"] = sweep
			}
		}
	}
	if goal := asMap(merged["goal"]); goal != nil && !mentions(patch, "goal", "strict_sla") {
		if _, has := goal["strict_sla"]; !has && hasBound(asMap(goal["sla"])) {
			goal["strict_sla"] = true
		}
	}
}

// normalize round-trips through JSON so every number is a float64, the type the
// parameters decode to. Merge and validation then see one representation.
func normalize(value map[string]any) (map[string]any, error) {
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// mergePatch applies patch to target with RFC 7386 semantics. It returns a new value for
// the merged parts and does not retain the patch's maps for null members.
func mergePatch(target, patch any) any {
	patchMap, ok := patch.(map[string]any)
	if !ok {
		return patch
	}
	targetMap, ok := target.(map[string]any)
	if !ok {
		targetMap = map[string]any{}
	}
	for key, value := range patchMap {
		if value == nil {
			delete(targetMap, key)
			continue
		}
		targetMap[key] = mergePatch(targetMap[key], value)
	}
	return targetMap
}

// mentions reports whether patch contains the key path, even with a null value.
func mentions(patch map[string]any, path ...string) bool {
	current := patch
	for i, key := range path {
		value, ok := current[key]
		if !ok {
			return false
		}
		if i == len(path)-1 {
			return true
		}
		if current = asMap(value); current == nil {
			return false
		}
	}
	return false
}

func asMap(value any) map[string]any {
	m, _ := value.(map[string]any)
	return m
}

func hasBound(sla map[string]any) bool {
	for _, key := range []string{"ttft_ms", "itl_ms", "e2e_ms"} {
		if sla[key] != nil {
			return true
		}
	}
	return false
}
