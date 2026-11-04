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

import "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

// The derived defaults. No API field carries them: each depends on the typed underlay, so
// no single typed field can supply it.
const (
	// DefaultRequestCount is traffic.stop.requests for every native static workload.
	// Alternative stop policies belong entirely to inputOverrides, which must delete
	// this value when replacing it.
	DefaultRequestCount int64 = 1000

	// DefaultMaxTrials bounds every search that has no explicit optimizer.max_trials.
	// It matches the trial budget of AISimulate's unified CLI while making the
	// effective input persisted in Run status self-contained.
	DefaultMaxTrials int64 = 320
)

// publicUnderlay supplies the role objects required by AISimulate's default
// aggregated-and-disaggregated engine search. Empty workers retain the upstream
// parallelism, scheduler, KV-cache and timing defaults.
func publicUnderlay() map[string]any {
	return map[string]any{
		"engine": map[string]any{
			"workers": map[string]any{
				"aggregated": map[string]any{},
				"prefill":    map[string]any{},
				"decode":     map[string]any{},
			},
		},
	}
}

// applyDefaults adds the derived defaults to the typed underlay, before the public-input
// input overrides are applied. spec.aisimulate.inputOverrides is the final overlay: it overrides or
// deletes any derived value, and nothing is derived in response to what it introduces.
// An override that needs a matching change elsewhere (deleting requests when adding
// requests_per_load_unit, or strict_sla next to a new SLA bound) must make that change itself.
func applyDefaults(underlay map[string]any) error {
	b := builder{root: underlay}

	if has(underlay, trafficPath...) {
		if !has(underlay, trafficStopRequests...) {
			b.set(float64(DefaultRequestCount), trafficStopRequests)
		}
	}
	if !has(underlay, optimizerMaxTrials...) {
		b.set(float64(DefaultMaxTrials), optimizerMaxTrials)
	}

	if hasSLABound(underlay) && !has(underlay, optimizationStrictSLA...) {
		b.set(true, optimizationStrictSLA)
	}
	return b.err
}

// hasSLABound reports whether the input bounds at least one latency.
func hasSLABound(input map[string]any) bool {
	for _, path := range [][]string{evaluationSLATTFTMs, evaluationSLAITLMs, evaluationSLAE2EMs} {
		if bound, _, _ := unstructured.NestedFieldNoCopy(input, path...); bound != nil {
			return true
		}
	}
	return false
}
