/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import "strings"

// SelectedWorkload is the immutable result of resolving exactly one DGD for
// LPX scheduling and runtime materialization.
// Its methods require a successfully resolved, non-nil workload with projections.
type SelectedWorkload struct {
	modelProjections []*ModelProjection
	digest           WorkloadDigest
	engineReplicas   int32
}

// ModelProjections returns a read-only view of the selected projection list.
// Callers must not modify the slice or its projections.
func (w *SelectedWorkload) ModelProjections() []*ModelProjection {
	return w.modelProjections
}

// Digest returns the selected workload digest.
func (w *SelectedWorkload) Digest() WorkloadDigest {
	return w.digest
}

// BuildFamily returns the selected build family.
func (w *SelectedWorkload) BuildFamily() BuildFamily {
	return w.modelProjections[0].configuredBuild.Family
}

// Pipeline returns the selected pipeline.
func (w *SelectedWorkload) Pipeline() Pipeline {
	return w.modelProjections[0].pipeline
}

// LPXComponentName returns the serving component name from the final projection.
func (w *SelectedWorkload) LPXComponentName() string {
	return w.modelProjections[len(w.modelProjections)-1].stage
}

// CyborgTemplateName derives the hybrid engine's GPU-role template name.
// A non-hybrid workload has no Cyborg template.
func (w *SelectedWorkload) CyborgTemplateName() string {
	if w.Pipeline() != PipelineLPX {
		return ""
	}
	return strings.ToLower(w.LPXComponentName()) + "-engine-gpu"
}
