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

// The derived defaults. No API field carries them: each depends on the merged input, so
// the typed spec and the native parameters cannot supply it independently.
const (
	// DefaultRequestCount is workload.request_count when the effective input does not
	// choose num_request_ratio.
	DefaultRequestCount int64 = 1000

	// DefaultMaxTrials bounds a KV-load range search that has no explicit sweep.max_trials.
	// It matches the trial budget of AISimulate's unified CLI (optimizer.max_trials).
	DefaultMaxTrials int64 = 320
)

// applyDefaults adds the derived defaults to the merged input. A default is skipped when
// the parameters name its key, including with null: the caller's native value overrides or
// deletes the default.
func applyDefaults(input, parameters map[string]any) error {
	b := builder{root: input}

	if has(input, Workload...) {
		if !has(input, WorkloadRequestCount...) && !has(input, WorkloadNumRequestRatio...) &&
			!has(parameters, WorkloadRequestCount...) {
			b.set(float64(DefaultRequestCount), WorkloadRequestCount)
		}
		kvLoadRatio, _, _ := unstructured.NestedFieldNoCopy(input, WorkloadKVLoadRatio...)
		if _, isRange := kvLoadRatio.([]any); isRange &&
			!has(input, SweepMaxTrials...) && !has(parameters, SweepMaxTrials...) {
			b.set(float64(DefaultMaxTrials), SweepMaxTrials)
		}
	}

	if hasSLABound(input) && !has(input, GoalStrictSLA...) && !has(parameters, GoalStrictSLA...) {
		b.set(true, GoalStrictSLA)
	}
	return b.err
}

// hasSLABound reports whether the effective input bounds at least one latency.
func hasSLABound(input map[string]any) bool {
	for _, path := range [][]string{GoalSLATTFTMs, GoalSLAITLMs, GoalSLAE2EMs} {
		if bound, _, _ := unstructured.NestedFieldNoCopy(input, path...); bound != nil {
			return true
		}
	}
	return false
}
