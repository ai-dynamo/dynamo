/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtime

import "github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"

// firstVersionedRenderingProfile is the adoption boundary for profile-aware
// rendering. Individual runtime capabilities may have earlier gate versions,
// but older runtimes retain their existing operator rendering behavior.
var firstVersionedRenderingProfile = runtimeversion.Version{Major: 1, Minor: 5, Patch: 0}

// RuntimeProfile is the complete set of runtime-dependent rendering decisions
// for a component. Renderers and the v2 worker hash consume the same profile so
// the hash changes exactly when runtime-gated rendering changes.
type RuntimeProfile struct {
	CanaryHealthChecks bool `json:"canaryHealthChecks,omitempty"`
}

// ProfileForVersion constructs the effective rendering profile for a resolved
// runtime version. Runtime 1.4 and older retain the empty legacy profile so
// introducing profile-aware rendering does not change their established v2
// worker hashes.
func ProfileForVersion(version *runtimeversion.Version) RuntimeProfile {
	if version == nil || version.Compare(firstVersionedRenderingProfile) < 0 {
		return RuntimeProfile{}
	}

	return RuntimeProfile{
		CanaryHealthChecks: CanaryHealthChecks.Enabled(version),
	}
}

// IsEmpty reports whether the legacy rendering profile is in effect.
func (p RuntimeProfile) IsEmpty() bool {
	return p == RuntimeProfile{}
}
