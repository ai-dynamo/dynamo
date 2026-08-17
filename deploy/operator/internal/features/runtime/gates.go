/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtime

import "github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"

var (
	// CanaryHealthChecks gates the canary health-check rendering defaults.
	// Runtime 1.5.0 is the first version whose resolved runtime version is
	// included in the worker hash, so enabling the feature cannot silently
	// change an existing worker generation.
	CanaryHealthChecks = Gate{
		Name:              "CanaryHealthChecks",
		MinRuntimeVersion: runtimeversion.Version{Major: 1, Minor: 5, Patch: 0},
	}
)
