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
	"slices"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

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

// publicUnderlay supplies the empty role objects AISimulate needs for the engine mode that
// spec.aisim.inputOverrides selects, the only source of engine.mode. Empty workers retain the
// upstream parallelism, scheduler, KV-cache and timing defaults. AISimulate reads engine.mode,
// or its choices, as a set of modes and rejects a role that a single-mode set does not run, so
// {aggregated} gets only the aggregated role and {disaggregated} only prefill and decode. Any
// other set (no mode, several choices, afd) keeps all three roles for AISimulate's default
// aggregated-and-disaggregated search. Overrides are applied after this, so a role the caller
// names is kept in every mode.
func publicUnderlay(overrides map[string]any) map[string]any {
	modes, _, _ := unstructured.NestedStringSlice(overrides, engineModeChoices...)
	if mode, found, _ := unstructured.NestedString(overrides, engineMode...); found {
		modes = []string{mode}
	}
	slices.Sort(modes)
	modes = slices.Compact(modes)

	workers := map[string]any{
		"aggregated": map[string]any{},
		"prefill":    map[string]any{},
		"decode":     map[string]any{},
	}
	if len(modes) == 1 {
		switch modes[0] {
		case "aggregated":
			workers = map[string]any{"aggregated": map[string]any{}}
		case "disaggregated":
			workers = map[string]any{"prefill": map[string]any{}, "decode": map[string]any{}}
		}
	}
	return map[string]any{"engine": map[string]any{"workers": workers}}
}

// applyDefaults adds the derived defaults to the typed underlay, before the public-input
// input overrides are applied. spec.aisim.inputOverrides is the final overlay: it overrides or
// deletes any derived value, and nothing is derived in response to what it introduces.
// An override that needs a matching change elsewhere (deleting requests when adding
// requests_per_load_unit, or strict_sla next to a new SLA bound) must make that change itself.
func applyDefaults(underlay map[string]any) error {
	b := builder{root: underlay}

	if has(underlay, trafficPath...) {
		// A trace accepts only max_virtual_time_seconds; a request count would be rejected.
		sourceType, _, _ := unstructured.NestedString(underlay, trafficSourceType...)
		if sourceType != "trace" && !has(underlay, trafficStopRequests...) {
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
