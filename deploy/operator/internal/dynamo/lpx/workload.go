/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"math"
)

// Workload is the immutable result of resolving one LPX component group
// for scheduling and runtime materialization.
// Its methods require a successfully resolved, non-nil workload with projections.
type Workload struct {
	modelProjections     []*ModelProjection
	digest               WorkloadDigest
	scalingGroupReplicas int32
}

// ModelProjections returns a read-only view of the workload's model projections.
// Callers must not modify the slice or its projections.
func (w *Workload) ModelProjections() []*ModelProjection {
	return w.modelProjections
}

// Digest returns the workload's digest.
func (w *Workload) Digest() WorkloadDigest {
	return w.digest
}

// Pipeline returns the workload's runtime pipeline.
func (w *Workload) Pipeline() Pipeline {
	return w.modelProjections[0].pipeline
}

// ComponentNames returns this workload's members in canonical runtime order.
func (w *Workload) ComponentNames() []string {
	// Draft fanout can contribute several consecutive models from one component.
	var names []string
	for _, projection := range w.modelProjections {
		if len(names) == 0 || names[len(names)-1] != projection.stage {
			names = append(names, projection.stage)
		}
	}
	return names
}

// ServingComponentName returns the conductor owner, placed last by ResolveWorkload.
func (w *Workload) ServingComponentName() string {
	return w.modelProjections[len(w.modelProjections)-1].stage
}

// MinimumCyborgReplicas returns one complete client group for a resolved hybrid workload.
func (w *Workload) MinimumCyborgReplicas() (int32, error) {
	build := w.modelProjections[0].configuredBuild
	replicas := int64(build.ioFPGACount) * int64(build.ioFanoutFactor)
	if replicas > math.MaxInt32 {
		return 0, fmt.Errorf("minimum Cyborg replicas %d exceeds the PodClique replica limit %d", replicas, math.MaxInt32)
	}
	return int32(replicas), nil
}

// ValidateCyborgReplicas ensures that the specified Cyborg replicas are in a valid configuration.
func (w *Workload) ValidateCyborgReplicas(replicas int32) error {
	return validateCyborgReplicas(w.modelProjections[0].configuredBuild, replicas)
}
