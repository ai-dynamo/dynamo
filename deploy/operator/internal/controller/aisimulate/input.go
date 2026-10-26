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
	"fmt"
	"strings"
	"unicode"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// Lower returns the effective native input: the typed underlay and derived defaults, merged
// with spec.aisimulate.inputOverrides. The spec must already carry its structural defaults;
// lowering repeats none of them. The only error is malformed inputOverrides (not a JSON
// object); semantic problems are the caller's to validate.
func Lower(spec *v1beta2.NativeSearchInput) (map[string]any, error) {
	overrides, err := InputOverrides(spec)
	if err != nil {
		return nil, err
	}
	underlay, err := Underlay(spec)
	if err != nil {
		return nil, err
	}
	if underlay, err = normalize(underlay); err != nil {
		return nil, err
	}
	if err := applyDefaults(underlay); err != nil {
		return nil, err
	}
	input, _ := mergePatch(underlay, overrides).(map[string]any)
	if input == nil {
		input = map[string]any{}
	}
	return input, nil
}

// InputOverrides returns spec.aisimulate.inputOverrides as a JSON object, or nil when absent.
func InputOverrides(spec *v1beta2.NativeSearchInput) (map[string]any, error) {
	if spec.AISimulate == nil || spec.AISimulate.InputOverrides == nil {
		return nil, nil
	}
	raw := spec.AISimulate.InputOverrides.Raw
	if len(raw) == 0 && spec.AISimulate.InputOverrides.Object != nil {
		var err error
		if raw, err = json.Marshal(spec.AISimulate.InputOverrides.Object); err != nil {
			return nil, fmt.Errorf("spec.aisimulate.inputOverrides: %w", err)
		}
	}
	if len(raw) == 0 {
		return nil, nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, fmt.Errorf("spec.aisimulate.inputOverrides: %w", err)
	}
	switch typed := value.(type) {
	case nil:
		return nil, nil
	case map[string]any:
		return typed, nil
	default:
		return nil, fmt.Errorf("spec.aisimulate.inputOverrides: must be a JSON object, got %T", value)
	}
}

// Underlay translates the typed spec into native input. Typed fields that are unset
// produce no key, so the native defaults and input overrides decide them.
func Underlay(spec *v1beta2.NativeSearchInput) (map[string]any, error) {
	b := builder{root: map[string]any{}}

	if spec.Model != nil {
		b.set(spec.Model.Name, searchSpaceModelName)
	}
	if len(spec.Backends) > 0 {
		backends := make([]string, len(spec.Backends))
		for i, backend := range spec.Backends {
			backends[i] = string(backend)
		}
		b.setStrings(backends, searchSpaceBackend)
	}
	if spec.Hardware != nil && spec.Hardware.GPU != nil {
		if sku := spec.Hardware.GPU.SKU; sku != nil && *sku != "" {
			b.set(*sku, searchSpaceHardwareSKU)
		}
		if budget := spec.Hardware.GPU.Budget; budget != nil {
			b.set(int64(*budget), searchSpaceGPUBudget)
		}
	}

	if spec.Workload != nil && spec.Workload.Static != nil {
		static := spec.Workload.Static
		if static.ISL != nil {
			b.set(int64(*static.ISL), workloadISL)
		}
		if static.OSL != nil {
			b.set(int64(*static.OSL), workloadOSL)
		}
		if static.NumRequestRatio != nil {
			b.set(*static.NumRequestRatio, workloadNumRequestRatio)
		}
		if static.RequestRate != nil {
			b.set(*static.RequestRate, workloadRequestRate)
		}
		if static.Concurrency != nil {
			b.set(int64(*static.Concurrency), workloadConcurrency)
		}
		switch len(static.KVLoadRatio) {
		case 0:
		case 1:
			b.set(static.KVLoadRatio[0], workloadKVLoadRatio)
		default:
			ratios := make([]any, len(static.KVLoadRatio))
			for i, ratio := range static.KVLoadRatio {
				ratios[i] = ratio
			}
			b.set(ratios, workloadKVLoadRatio)
		}
	}

	if objective := spec.Objective; objective != nil {
		if objective.Mode == v1beta2.ObjectiveModePareto {
			b.set(TargetPareto, goalTarget)
			if len(objective.Metrics) > 0 {
				b.setStrings(nativeMetrics(objective.Metrics), goalParetoObjectives)
			}
		} else if objective.Metric != nil {
			b.set(nativeMetric(*objective.Metric), goalTarget)
		}
		if sla := objective.SLA; sla != nil {
			if sla.TTFTMs != nil {
				b.set(*sla.TTFTMs, goalSLATTFTMs)
			}
			if sla.ITLMs != nil {
				b.set(*sla.ITLMs, goalSLAITLMs)
			}
			if sla.E2EMs != nil {
				b.set(*sla.E2EMs, goalSLAE2EMs)
			}
		}
	}
	return b.root, b.err
}

// builder writes native input values, remembering the first failure so the translation reads
// as a straight list of assignments. Every path is written under a map the caller created
// and every value is a JSON scalar or a list of them, so a failure means a bad path.
type builder struct {
	root map[string]any
	err  error
}

func (b *builder) set(value any, path []string) {
	if b.err != nil {
		return
	}
	if err := unstructured.SetNestedField(b.root, value, path...); err != nil {
		b.err = fmt.Errorf("set %s: %w", strings.Join(path, "."), err)
	}
}

func (b *builder) setStrings(values []string, path []string) {
	if b.err != nil {
		return
	}
	if err := unstructured.SetNestedStringSlice(b.root, values, path...); err != nil {
		b.err = fmt.Errorf("set %s: %w", strings.Join(path, "."), err)
	}
}

// nativeMetric maps an API metric to AISimulate's goal.target by converting its CamelCase
// name to snake_case: ThroughputPerGPU becomes throughput_per_gpu and E2ELatency becomes
// e2e_latency. A run of capitals stays together, a digit does not start a word, and the last
// capital of a run starts the next word when a lowercase letter follows it.
func nativeMetric(metric v1beta2.ObjectiveMetric) string {
	name := []rune(string(metric))
	var out []rune
	for i, r := range name {
		if i > 0 && unicode.IsUpper(r) {
			prev := name[i-1]
			nextLower := i+1 < len(name) && unicode.IsLower(name[i+1])
			if unicode.IsLower(prev) || (unicode.IsUpper(prev) && nextLower) {
				out = append(out, '_')
			}
		}
		out = append(out, unicode.ToLower(r))
	}
	return string(out)
}

func nativeMetrics(metrics []v1beta2.ObjectiveMetric) []string {
	out := make([]string, len(metrics))
	for i, metric := range metrics {
		out[i] = nativeMetric(metric)
	}
	return out
}

// normalize round-trips through JSON so every number is a float64, the type the
// input overrides decode to. Merge and validation then see one representation.
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
