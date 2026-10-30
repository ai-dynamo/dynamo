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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// Lower returns the effective public recommendation input: the typed underlay and derived defaults, merged
// with spec.aisimulate.inputOverrides. The spec must already carry its structural defaults;
// lowering repeats none of them.
func Lower(spec *v1beta2.NativeSearchInput) (map[string]any, error) {
	return LowerWithUnderlay(spec, nil)
}

// LowerWithUnderlay returns the effective public recommendation input with a controller-owned
// static underlay beneath the typed API fields and derived defaults. InputOverrides remains the
// final merge patch, so it may override or delete fields from either underlay.
func LowerWithUnderlay(spec *v1beta2.NativeSearchInput, staticUnderlay map[string]any) (map[string]any, error) {
	overrides, err := InputOverrides(spec)
	if err != nil {
		return nil, err
	}
	typedUnderlay, err := Underlay(spec)
	if err != nil {
		return nil, err
	}
	staticUnderlay, err = normalize(staticUnderlay)
	if err != nil {
		return nil, err
	}
	underlay := publicUnderlay()
	underlay, _ = mergePatch(underlay, staticUnderlay).(map[string]any)
	underlay, _ = mergePatch(underlay, typedUnderlay).(map[string]any)
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
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return nil, fmt.Errorf("spec.aisimulate.inputOverrides: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return nil, fmt.Errorf("spec.aisimulate.inputOverrides: trailing JSON value")
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

// Underlay translates the typed spec into the public CoreRecommendationConfig input consumed by
// `aisimulate recommend`. Typed fields that are unset produce no key, so AISimulate defaults and
// input overrides decide them.
func Underlay(spec *v1beta2.NativeSearchInput) (map[string]any, error) {
	b := builder{root: map[string]any{}}

	if spec.Model != nil {
		b.set(spec.Model.Name, engineModel)
	}
	if len(spec.Backends) > 0 {
		backends := make([]string, len(spec.Backends))
		for i, backend := range spec.Backends {
			backends[i] = string(backend)
		}
		b.setStrings(backends, engineBackendChoices)
	}
	if spec.Hardware != nil && spec.Hardware.GPU != nil {
		if sku := spec.Hardware.GPU.SKU; sku != nil && *sku != "" {
			b.set(*sku, engineHardware)
		}
		if budget := spec.Hardware.GPU.Budget; budget != nil {
			b.set(int64(*budget), optimizationHardwareBudget)
		}
	}

	if spec.Workload != nil && spec.Workload.Static != nil {
		static := spec.Workload.Static
		if hasStaticWorkloadFields(static) {
			b.set("synthetic", trafficSourceType)
		}
		if static.InputTokens != nil {
			b.set(int64(*static.InputTokens), trafficSourceInputTokens)
		}
		if static.OutputTokens != nil {
			b.set(int64(*static.OutputTokens), trafficSourceOutputTokens)
		}
		if load := static.Load; load != nil {
			switch {
			case load.RequestsPerSecond != nil:
				b.set("constant_rate", trafficLoadType)
				b.set(*load.RequestsPerSecond, trafficLoadRequestsPerSecond)
			case load.Concurrency != nil:
				b.set("concurrency", trafficLoadType)
				b.set(int64(*load.Concurrency), trafficLoadConcurrency)
			case load.KVCapacityFraction != nil:
				b.set("kv_capacity_fraction", trafficLoadType)
				fraction := load.KVCapacityFraction
				switch {
				case fraction.Value != nil:
					b.set(*fraction.Value, trafficLoadFraction)
				case fraction.Range != nil:
					b.set(map[string]any{"range": map[string]any{
						"min": fraction.Range.Min,
						"max": fraction.Range.Max,
					}}, trafficLoadFraction)
				}
			}
		}
	}

	if objective := spec.Objective; objective != nil {
		if objective.Metric != nil {
			target, err := publicMetric(*objective.Metric)
			if err != nil {
				b.err = err
			} else {
				b.set(target, optimizationTarget)
			}
		}
		if sla := objective.SLA; sla != nil {
			if sla.TTFTMs != nil {
				b.set(*sla.TTFTMs, evaluationSLATTFTMs)
			}
			if sla.ITLMs != nil {
				b.set(*sla.ITLMs, evaluationSLAITLMs)
			}
			if sla.E2EMs != nil {
				b.set(*sla.E2EMs, evaluationSLAE2EMs)
			}
		}
	}
	return b.root, b.err
}

func hasStaticWorkloadFields(static *v1beta2.StaticWorkloadSpec) bool {
	return static.InputTokens != nil || static.OutputTokens != nil || static.Load != nil
}

// builder writes public input values, remembering the first failure so the translation reads
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

// publicMetric maps each public API metric to its explicit AISimulate optimization target.
func publicMetric(metric v1beta2.ObjectiveMetric) (string, error) {
	switch metric {
	case v1beta2.ObjectiveMetric("Throughput"):
		return "throughput", nil
	case v1beta2.ObjectiveMetric("ThroughputPerGPU"):
		return "throughput_per_gpu", nil
	case v1beta2.ObjectiveMetric("ThroughputPerUser"):
		return "throughput_per_user", nil
	case v1beta2.ObjectiveMetric("TTFT"):
		return "ttft", nil
	case v1beta2.ObjectiveMetric("E2ELatency"):
		return "e2e_latency", nil
	case v1beta2.ObjectiveMetric("Goodput"):
		return "goodput", nil
	case v1beta2.ObjectiveMetric("GoodputPerGPU"):
		return "goodput_per_gpu", nil
	default:
		return "", fmt.Errorf("spec.objective.metric: unsupported value %q", metric)
	}
}

// normalize round-trips operator-owned maps through JSON so their Go-specific numeric types
// become JSON-compatible values before merging them with the user-owned input overrides.
func normalize(value map[string]any) (map[string]any, error) {
	if value == nil {
		return nil, nil
	}
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
