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
	"fmt"
	"strings"
	"unicode"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/aisimulate"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// Where each API field lands in the native AISimulate input, in the snake_case spelling of the
// upstream contract. The mapping is the Request reconciler's: the aisimulate package knows only the
// native document.
var (
	dgdrNativeModelName   = []string{"search_space", "model_name"}
	dgdrNativeBackend     = []string{"search_space", "backend"}
	dgdrNativeHardwareSKU = []string{"search_space", "hardware_sku"}
	dgdrNativeGPUBudget   = []string{"search_space", "gpu_budget"}

	dgdrNativeISL             = []string{"workload", "isl"}
	dgdrNativeOSL             = []string{"workload", "osl"}
	dgdrNativeNumRequestRatio = []string{"workload", "num_request_ratio"}
	dgdrNativeRequestRate     = []string{"workload", "request_rate"}
	dgdrNativeConcurrency     = []string{"workload", "concurrency"}
	dgdrNativeKVLoadRatio     = []string{"workload", "kv_load_ratio"}

	dgdrNativeGoalTarget           = []string{"goal", "target"}
	dgdrNativeGoalParetoObjectives = []string{"goal", "pareto_objectives"}
	dgdrNativeGoalSLATTFTMs        = []string{"goal", "sla", "ttft_ms"}
	dgdrNativeGoalSLAITLMs         = []string{"goal", "sla", "itl_ms"}
	dgdrNativeGoalSLAE2EMs         = []string{"goal", "sla", "e2e_ms"}
)

// lowerDGDRV1beta2Input returns the effective native input of a Request: the typed underlay, the
// derived defaults, and spec.aisimulate.parameters applied last. The spec must already carry its
// structural defaults; lowering repeats none of them. The only error is malformed parameters (not a
// JSON object); semantic problems are the caller's to validate.
func lowerDGDRV1beta2Input(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) (map[string]any, error) {
	underlay, err := dgdrV1beta2Underlay(spec)
	if err != nil {
		return nil, err
	}
	parameters, err := dgdrV1beta2ParametersJSON(spec)
	if err != nil {
		return nil, err
	}
	input, err := aisimulate.BuildInput(underlay, parameters)
	if err != nil {
		return nil, fmt.Errorf("build the AISimulate input: %w", err)
	}
	return input, nil
}

// dgdrV1beta2ParametersJSON returns spec.aisimulate.parameters as raw JSON, or nil when absent.
func dgdrV1beta2ParametersJSON(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) ([]byte, error) {
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
	return raw, nil
}

// dgdrV1beta2Underlay translates the typed spec into native input. Typed fields that are unset
// produce no key, so the native defaults and the parameters decide them.
func dgdrV1beta2Underlay(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) (map[string]any, error) {
	b := dgdrNativeBuilder{root: map[string]any{}}

	if spec.Model != nil {
		b.set(spec.Model.Name, dgdrNativeModelName)
	}
	if len(spec.Backends) > 0 {
		backends := make([]string, len(spec.Backends))
		for i, backend := range spec.Backends {
			backends[i] = string(backend)
		}
		b.setStrings(backends, dgdrNativeBackend)
	}
	if spec.Hardware != nil && spec.Hardware.GPU != nil {
		if sku := spec.Hardware.GPU.SKU; sku != nil && *sku != "" {
			b.set(*sku, dgdrNativeHardwareSKU)
		}
		if budget := spec.Hardware.GPU.Budget; budget != nil {
			b.set(int64(*budget), dgdrNativeGPUBudget)
		}
	}

	if spec.Workload != nil && spec.Workload.Static != nil {
		static := spec.Workload.Static
		if static.ISL != nil {
			b.set(int64(*static.ISL), dgdrNativeISL)
		}
		if static.OSL != nil {
			b.set(int64(*static.OSL), dgdrNativeOSL)
		}
		if static.NumRequestRatio != nil {
			b.set(*static.NumRequestRatio, dgdrNativeNumRequestRatio)
		}
		if static.RequestRate != nil {
			b.set(*static.RequestRate, dgdrNativeRequestRate)
		}
		if static.Concurrency != nil {
			b.set(int64(*static.Concurrency), dgdrNativeConcurrency)
		}
		switch len(static.KVLoadRatio) {
		case 0:
		case 1:
			b.set(static.KVLoadRatio[0], dgdrNativeKVLoadRatio)
		default:
			ratios := make([]any, len(static.KVLoadRatio))
			for i, ratio := range static.KVLoadRatio {
				ratios[i] = ratio
			}
			b.set(ratios, dgdrNativeKVLoadRatio)
		}
	}

	if objective := spec.Objective; objective != nil {
		if objective.Mode == nvidiacomv1beta2.ObjectiveModePareto {
			b.set(aisimulate.TargetPareto, dgdrNativeGoalTarget)
			if len(objective.Metrics) > 0 {
				b.setStrings(dgdrNativeMetrics(objective.Metrics), dgdrNativeGoalParetoObjectives)
			}
		} else if objective.Metric != nil {
			b.set(dgdrNativeMetric(*objective.Metric), dgdrNativeGoalTarget)
		}
		if sla := objective.SLA; sla != nil {
			if sla.TTFTMs != nil {
				b.set(*sla.TTFTMs, dgdrNativeGoalSLATTFTMs)
			}
			if sla.ITLMs != nil {
				b.set(*sla.ITLMs, dgdrNativeGoalSLAITLMs)
			}
			if sla.E2EMs != nil {
				b.set(*sla.E2EMs, dgdrNativeGoalSLAE2EMs)
			}
		}
	}
	return b.root, b.err
}

// dgdrNativeBuilder writes native input values, remembering the first failure so the translation
// reads as a straight list of assignments. Every path is written under a map the caller created and
// every value is a JSON scalar or a list of them, so a failure means a bad path.
type dgdrNativeBuilder struct {
	root map[string]any
	err  error
}

func (b *dgdrNativeBuilder) set(value any, path []string) {
	if b.err != nil {
		return
	}
	if err := unstructured.SetNestedField(b.root, value, path...); err != nil {
		b.err = fmt.Errorf("set %s: %w", strings.Join(path, "."), err)
	}
}

func (b *dgdrNativeBuilder) setStrings(values []string, path []string) {
	if b.err != nil {
		return
	}
	if err := unstructured.SetNestedStringSlice(b.root, values, path...); err != nil {
		b.err = fmt.Errorf("set %s: %w", strings.Join(path, "."), err)
	}
}

// dgdrNativeMetric maps an API metric to AISimulate's goal.target by converting its CamelCase
// name to snake_case: ThroughputPerGPU becomes throughput_per_gpu and E2ELatency becomes
// e2e_latency. A run of capitals stays together, a digit does not start a word, and the last
// capital of a run starts the next word when a lowercase letter follows it.
func dgdrNativeMetric(metric nvidiacomv1beta2.ObjectiveMetric) string {
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

func dgdrNativeMetrics(metrics []nvidiacomv1beta2.ObjectiveMetric) []string {
	out := make([]string, len(metrics))
	for i, metric := range metrics {
		out[i] = dgdrNativeMetric(metric)
	}
	return out
}
