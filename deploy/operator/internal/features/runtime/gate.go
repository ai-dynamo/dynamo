/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtime

import "github.com/ai-dynamo/dynamo/deploy/operator/internal/versionoverride"

// Gate controls runtime-version-dependent operator behavior.
type Gate struct {
	Name              string
	MinRuntimeVersion versionoverride.Version
}

// Enabled reports whether a known runtime version meets the feature threshold.
func (g Gate) Enabled(version *versionoverride.Version) bool {
	if version == nil {
		return false
	}

	return version.Compare(g.MinRuntimeVersion) >= 0
}
