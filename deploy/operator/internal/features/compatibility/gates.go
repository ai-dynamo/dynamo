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

package compatibility

import (
	semver "github.com/Masterminds/semver/v3"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
)

// Compatibility feature gates.

var (
	// VLLMMultiprocessing gates the use of vLLM native multiprocessing (mp)
	// instead of Ray for multi-node deployments. Enabled for DGDs originally
	// created by operator >= 1.0.0.
	VLLMMultiprocessing = OriginGate{
		Name:             "VLLMMultiprocessing",
		MinOriginVersion: *semver.MustParse("1.0.0"),
	}

	// Canary health check compatibility policy:
	//
	//	default: enabled for resources created by operator >= 1.4.0 when runtime >= 1.4.0
	//	upgrade: operator-only upgrades preserve the legacy default for existing resources
	//	scale: new replicas keep the existing workload template and do not opt in
	//	opt-in: set DYN_HEALTH_CHECK_ENABLED=true explicitly
	//	rollout: explicit opt-in intentionally rolls the affected workload
	//	ordering: none; mixed workers with and without canary checks are supported
	//	conditions:
	//	  - type: FeatureGatePending
	//	    when: the runtime gate passes but the origin gate preserves the legacy default

	// CanaryHealthChecksOrigin gates the canary health-check default on the
	// operator version that created the resource.
	CanaryHealthChecksOrigin = OriginGate{
		Name:             "CanaryHealthChecks",
		MinOriginVersion: *semver.MustParse("1.4.0"),
	}

	// CanaryHealthChecksRuntime gates canary health checks on runtime support.
	CanaryHealthChecksRuntime = RuntimeGate{
		Name:              "CanaryHealthChecks",
		MinRuntimeVersion: runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
	}
)
