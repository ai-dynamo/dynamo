/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtime

import "github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"

// Gate controls a feature's rendered defaults by Dynamo runtime version.
type Gate struct {
	Name              string
	MinRuntimeVersion runtimeversion.Version
}

// Enabled reports whether a known runtime version meets the feature threshold.
func (g Gate) Enabled(version *runtimeversion.Version) bool {
	if version == nil {
		return false
	}

	return version.Compare(g.MinRuntimeVersion) >= 0
}
