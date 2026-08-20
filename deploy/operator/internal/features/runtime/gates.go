/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtime

import "github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"

var (
	// CanaryHealthChecks gates the canary health-check rendering defaults
	// introduced for Dynamo runtime 1.4.0.
	CanaryHealthChecks = Gate{
		Name:              "CanaryHealthChecks",
		MinRuntimeVersion: runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
	}

	// NativeRustEPP gates the native Rust EPP requirement: runtime versions at
	// or above this threshold must not carry the legacy Go EPP's eppConfig,
	// and versions below it still require it. Introduced for Dynamo runtime
	// 1.5.0, the first release shipping the native Rust EPP image.
	NativeRustEPP = Gate{
		Name:              "NativeRustEPP",
		MinRuntimeVersion: runtimeversion.Version{Major: 1, Minor: 5, Patch: 0},
	}
)
