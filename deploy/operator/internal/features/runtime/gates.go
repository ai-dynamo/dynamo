/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtime

import "github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"

// A new gate must not change PodSpec rendering for a previously released runtime.
// Its minimum runtime version must be the release introducing the gate or newer.
var (
	// CanaryHealthChecks gates the canary health-check rendering defaults
	// introduced for Dynamo runtime 1.5.0.
	CanaryHealthChecks = Gate{
		Name:              "CanaryHealthChecks",
		MinRuntimeVersion: runtimeversion.Version{Major: 1, Minor: 5, Patch: 0},
	}
)
