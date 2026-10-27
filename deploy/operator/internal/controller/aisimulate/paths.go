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

// Native input paths, in the snake_case spelling of the upstream AISimulate contract. They
// address the same native key when the underlay is built and when derived defaults are read,
// and are passed to the unstructured helpers, which must not modify them.
var (
	searchSpaceModelName   = []string{"search_space", "model_name"}
	searchSpaceBackend     = []string{"search_space", "backend"}
	searchSpaceHardwareSKU = []string{"search_space", "hardware_sku"}
	searchSpaceGPUBudget   = []string{"search_space", "gpu_budget"}

	workloadPath            = []string{"workload"}
	workloadISL             = []string{"workload", "isl"}
	workloadOSL             = []string{"workload", "osl"}
	workloadNumRequestRatio = []string{"workload", "num_request_ratio"}
	workloadRequestRate     = []string{"workload", "request_rate"}
	workloadConcurrency     = []string{"workload", "concurrency"}
	workloadKVLoadRatio     = []string{"workload", "kv_load_ratio"}
	workloadRequestCount    = []string{"workload", "request_count"}

	goalTarget           = []string{"goal", "target"}
	goalParetoObjectives = []string{"goal", "pareto_objectives"}
	goalSLATTFTMs        = []string{"goal", "sla", "ttft_ms"}
	goalSLAITLMs         = []string{"goal", "sla", "itl_ms"}
	goalSLAE2EMs         = []string{"goal", "sla", "e2e_ms"}
	goalStrictSLA        = []string{"goal", "strict_sla"}

	sweepMaxTrials = []string{"sweep", "max_trials"}
)

// TargetPareto is the goal.target of a multi-objective search.
const TargetPareto = "pareto"

// has reports whether obj contains the path, even with a null value. Applied to
// spec.aisimulate.parameters it tells whether the caller named a key, including to delete
// it.
func has(obj map[string]any, path ...string) bool {
	_, found, _ := unstructured.NestedFieldNoCopy(obj, path...)
	return found
}
