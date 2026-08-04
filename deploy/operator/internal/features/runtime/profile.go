/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtime

import "github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"

// RuntimeProfile contains the effective runtime-version-gated features.
type RuntimeProfile struct {
	CanaryHealthChecks bool `json:"canaryHealthChecks,omitempty"`
}

// ProfileForVersion evaluates every runtime feature gate for a version.
func ProfileForVersion(version *runtimeversion.Version) RuntimeProfile {
	return RuntimeProfile{
		CanaryHealthChecks: CanaryHealthChecks.Enabled(version),
	}
}

// IsEmpty reports whether no runtime features are enabled.
func (p RuntimeProfile) IsEmpty() bool {
	return p == RuntimeProfile{}
}
