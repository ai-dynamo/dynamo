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
	// DefaultRequestCount is workload.request_count when the typed underlay does not
	// choose num_request_ratio.
	DefaultRequestCount int64 = 1000

	// DefaultMaxTrials bounds a KV-load range search that has no explicit sweep.max_trials.
	// It matches the trial budget of AISimulate's unified CLI (optimizer.max_trials).
	DefaultMaxTrials int64 = 320
)

// applyDefaults adds the derived defaults to the typed underlay, before the native
// parameters are applied. spec.aisimulate.parameters is the final overlay: it overrides or
// deletes any derived value, and nothing is derived in response to what it introduces.
// A native change that needs a matching change elsewhere (a request_count next to
// num_request_ratio, a strict_sla next to a new SLA bound) must make it itself.
func applyDefaults(underlay map[string]any) error {
	b := builder{root: underlay}

	if has(underlay, workloadPath...) {
		if !has(underlay, workloadRequestCount...) && !has(underlay, workloadNumRequestRatio...) {
			b.set(float64(DefaultRequestCount), workloadRequestCount)
		}
		kvLoadRatio, _, _ := unstructured.NestedFieldNoCopy(underlay, workloadKVLoadRatio...)
		if _, isRange := kvLoadRatio.([]any); isRange && !has(underlay, sweepMaxTrials...) {
			b.set(float64(DefaultMaxTrials), sweepMaxTrials)
		}
	}

	if hasSLABound(underlay) && !has(underlay, goalStrictSLA...) {
		b.set(true, goalStrictSLA)
	}
	return b.err
}

// hasSLABound reports whether the input bounds at least one latency.
func hasSLABound(input map[string]any) bool {
	for _, path := range [][]string{goalSLATTFTMs, goalSLAITLMs, goalSLAE2EMs} {
		if bound, _, _ := unstructured.NestedFieldNoCopy(input, path...); bound != nil {
			return true
		}
	}
	return false
}
