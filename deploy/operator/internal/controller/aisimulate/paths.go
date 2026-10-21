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

// Native input paths, in the snake_case spelling of the upstream AISimulate contract.
// They are read-only: callers pass them to the unstructured helpers and must not modify them.
var (
	SearchSpaceModelName   = []string{"search_space", "model_name"}
	SearchSpaceBackend     = []string{"search_space", "backend"}
	SearchSpaceHardwareSKU = []string{"search_space", "hardware_sku"}
	SearchSpaceGPUBudget   = []string{"search_space", "gpu_budget"}

	Workload                = []string{"workload"}
	WorkloadISL             = []string{"workload", "isl"}
	WorkloadOSL             = []string{"workload", "osl"}
	WorkloadNumRequestRatio = []string{"workload", "num_request_ratio"}
	WorkloadRequestRate     = []string{"workload", "request_rate"}
	WorkloadConcurrency     = []string{"workload", "concurrency"}
	WorkloadKVLoadRatio     = []string{"workload", "kv_load_ratio"}
	WorkloadRequestCount    = []string{"workload", "request_count"}
	WorkloadTracePath       = []string{"workload", "trace_path"}
	WorkloadTracePaths      = []string{"workload", "trace_paths"}

	Goal                 = []string{"goal"}
	GoalTarget           = []string{"goal", "target"}
	GoalParetoObjectives = []string{"goal", "pareto_objectives"}
	GoalSLA              = []string{"goal", "sla"}
	GoalSLATTFTMs        = []string{"goal", "sla", "ttft_ms"}
	GoalSLAITLMs         = []string{"goal", "sla", "itl_ms"}
	GoalSLAE2EMs         = []string{"goal", "sla", "e2e_ms"}
	GoalStrictSLA        = []string{"goal", "strict_sla"}
	GoalMinGoodputRPS    = []string{"goal", "min_goodput_rps"}

	SweepMaxTrials = []string{"sweep", "max_trials"}
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
