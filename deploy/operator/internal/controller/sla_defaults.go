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
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// applySLADefaults ensures that the SLA section of the DGDR spec has
// a valid SLA mode when the user has not provided explicit targets.
//
// When SLA is nil or empty (no fields set), the operator defaults to
// optimizationType=throughput. This lets users deploy models with zero
// SLA configuration and get a throughput-optimized deployment.
//
// When the user provides explicit TTFT, ITL, e2eLatency, or
// optimizationType, their values are preserved unchanged.
//
// The defaults are applied in-memory before the spec is marshalled to JSON
// and passed to the profiler — the persisted spec on the API server is not modified.
func applySLADefaults(spec *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec) {
	if spec.SLA == nil {
		spec.SLA = &nvidiacomv1beta1.SLASpec{
			OptimizationType: nvidiacomv1beta1.OptimizationTypeThroughput,
		}
		return
	}

	// If the user explicitly set any SLA field, respect it as-is.
	if spec.SLA.OptimizationType != "" || spec.SLA.E2ELatency != nil ||
		spec.SLA.TTFT != nil || spec.SLA.ITL != nil {
		return
	}

	// Empty SLA struct (all fields nil/zero) — default to throughput.
	spec.SLA.OptimizationType = nvidiacomv1beta1.OptimizationTypeThroughput
}
