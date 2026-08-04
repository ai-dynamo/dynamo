/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtime

import "github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"

// Add each new gate to RuntimeProfile and ProfileForVersion so hashing tracks
// every runtime-dependent rendering decision.
var (
	// CanaryHealthChecks gates the canary health-check rendering defaults
	// introduced for Dynamo runtime 1.4.0.
	CanaryHealthChecks = Gate{
		Name:              "CanaryHealthChecks",
		MinRuntimeVersion: runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
	}
)
