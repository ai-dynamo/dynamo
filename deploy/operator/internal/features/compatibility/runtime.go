/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package compatibility

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// RuntimeGate represents a feature gated on the resolved Dynamo runtime
// compatibility version.
type RuntimeGate struct {
	Name              string                 // Human-readable feature name (for logging)
	MinRuntimeVersion runtimeversion.Version // Minimum runtime version required
}

// Enabled returns true if runtimeVersion is known and meets or exceeds the
// gate's minimum runtime version.
func (fg RuntimeGate) Enabled(runtimeVersion *runtimeversion.Version) bool {
	logger := log.Log.WithName("compatibility").WithValues("feature", fg.Name)
	if runtimeVersion == nil {
		logger.V(1).Info("Runtime version unknown, feature disabled")
		return false
	}

	enabled := runtimeVersion.AtLeast(fg.MinRuntimeVersion)

	logger.V(1).Info("Runtime feature gate evaluated",
		"runtimeVersion", runtimeVersion.String(),
		"threshold", fg.MinRuntimeVersion,
		"enabled", enabled)

	return enabled
}
