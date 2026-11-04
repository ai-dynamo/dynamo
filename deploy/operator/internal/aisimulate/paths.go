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

// Public recommendation input paths, in the snake_case spelling of the upstream AISimulate
// contract. They address the same key when the underlay is built and when derived defaults are read,
// and are passed to the unstructured helpers, which must not modify them.
var (
	engineModel          = []string{"engine", "model"}
	engineBackendChoices = []string{"engine", "backend", "choices"}
	engineHardware       = []string{"engine", "hardware"}
	engineWorkers        = []string{"engine", "workers"}

	optimizationTarget         = []string{"optimization", "target"}
	optimizationHardwareBudget = []string{"optimization", "constraints", "max_candidate_gpus"}
	optimizationStrictSLA      = []string{"optimization", "strict_sla"}
	evaluationSLATTFTMs        = []string{"evaluation", "sla", "ttft_ms"}
	evaluationSLAITLMs         = []string{"evaluation", "sla", "itl_ms"}
	evaluationSLAE2EMs         = []string{"evaluation", "sla", "e2e_ms"}
	optimizerMaxTrials         = []string{"optimizer", "max_trials"}

	trafficPath                  = []string{"traffic"}
	trafficSource                = []string{"traffic", "source"}
	trafficSourceType            = []string{"traffic", "source", "type"}
	trafficSourceInputTokens     = []string{"traffic", "source", "input_tokens"}
	trafficSourceOutputTokens    = []string{"traffic", "source", "output_tokens"}
	trafficLoadType              = []string{"traffic", "load", "type"}
	trafficLoadConcurrency       = []string{"traffic", "load", "concurrency"}
	trafficLoadRequestsPerSecond = []string{"traffic", "load", "requests_per_second"}
	trafficLoadFraction          = []string{"traffic", "load", "fraction"}
	trafficLoadSpeedup           = []string{"traffic", "load", "speedup"}
	trafficStopRequests          = []string{"traffic", "stop", "requests"}
)

// TargetPareto is optimization.target for a multi-objective search.
const TargetPareto = "pareto"

// has reports whether obj contains the path, even with a null value. Applied to
// spec.aisimulate.inputOverrides it tells whether the caller named a key, including to delete
// it.
func has(obj map[string]any, path ...string) bool {
	_, found, _ := unstructured.NestedFieldNoCopy(obj, path...)
	return found
}
