/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dynamo

// Elastic-EP autoscale decision logic (Phase 6). This file is deliberately pure -- no
// Kubernetes client, no HTTP -- because the arithmetic here is the part most likely to be
// subtly wrong, and the fault case is dangerous to get wrong. The reconciler that owns the
// settle window, the HTTP call, and status lives in the controller package.

// EPCapacity mirrors the read-only get_ep_capacity engine response (Phase 5, #13179).
type EPCapacity struct {
	Status              string      `json:"status"`
	DataParallelSize    int         `json:"data_parallel_size"`
	TensorParallelSize  int         `json:"tensor_parallel_size"`
	DataParallelBackend string      `json:"data_parallel_backend"`
	TotalGPUs           *float64    `json:"total_gpus"`
	AvailableGPUs       *float64    `json:"available_gpus"`
	UsedGPUs            *float64    `json:"used_gpus"`
	Nodes               []EPRayNode `json:"nodes"`
}

// EPRayNode is one node in the Ray cluster as the engine sees it. node_ip is the address the
// raylet registered, which for elastic EP is the pod IP (the launch passes
// --node-ip-address="$POD_IP"), so it can be intersected with this deployment's pod IPs.
type EPRayNode struct {
	NodeIP    string  `json:"node_ip"`
	TotalGPUs float64 `json:"total_gpus"`
}

// EPScaleAction is what the reconciler should do with the engine's data-parallel size.
type EPScaleAction int

const (
	// EPScaleNone: the engine already runs at the right size; do nothing.
	EPScaleNone EPScaleAction = iota
	// EPScaleGrow: more owned capacity has joined than the engine uses; grow into it.
	EPScaleGrow
	// EPScaleShrink: the desired follower count fell; shrink to it.
	EPScaleShrink
	// EPScaleFault: owned capacity dropped below what the engine is running on -- a pod
	// holding live ranks died or drained. This is NOT a scale-down signal; scaling to match
	// would shrink around dead ranks. The reconciler must route this to recovery, not scale.
	EPScaleFault
)

func (a EPScaleAction) String() string {
	switch a {
	case EPScaleGrow:
		return "grow"
	case EPScaleShrink:
		return "shrink"
	case EPScaleFault:
		return "fault"
	default:
		return "none"
	}
}

// EPScaleDecision is the outcome of DecideEPScale.
type EPScaleDecision struct {
	Action EPScaleAction
	// Target is the data-parallel size to request. For Fault and None it echoes the current
	// size (nothing is sent).
	Target int
}

// OwnedRayNodeCount is the generation filter: it counts only the Ray nodes this deployment's
// CURRENT generation owns, by intersecting what the engine reports with the set of pod IPs
// the operator knows belong to the live generation. During validation a pod from a superseded
// generation was still alive in the Ray cluster; counting it would let the reconciler grow the
// engine onto a rank it does not own and cannot manage. Anything Ray reports that is not in
// ownedPodIPs is noise to report, not capacity to use.
//
// Under one pod per node this count is also the data-parallel target directly: one node, one
// rank, so the arithmetic is a division that always comes out even and the reconciler never
// reasons about partially filled pods.
func OwnedRayNodeCount(cap EPCapacity, ownedPodIPs map[string]struct{}) int {
	owned := 0
	for _, n := range cap.Nodes {
		if _, ok := ownedPodIPs[n.NodeIP]; ok {
			owned++
		}
	}
	return owned
}

// DecideEPScale converges the engine's data-parallel size on the owned capacity that has
// joined, bounded by desire. The three numbers must not be confused:
//
//   - current:  the size the engine is running at right now (capacity.DataParallelSize).
//   - observed: owned Ray nodes that have joined (OwnedRayNodeCount) -- current-generation only.
//   - desired:  the target data-parallel size implied by the scaling adapter's follower count
//     (1 leader + desiredFollowers), owned by whoever drives the adapter.
//
// Rules, in order, straight from the design:
//   - Shrink is driven ONLY by desire falling. If desired < current, shrink to desired.
//   - Otherwise, observation falling below current is a FAULT (a live rank's pod died), never a
//     scale-down -- scaling to match would shrink around dead ranks. Route to recovery.
//   - Otherwise growth is driven by observation and bounded by desire: grow to min(observed, desired).
//
// The lower bound is one rank; a caller must never pass desired < 1. Deriving the target from
// observation alone -- the obvious formulation -- gets growth right and loss dangerously wrong,
// which is why shrink and fault are separated here rather than collapsed into "match observed".
func DecideEPScale(current, observed, desired int) EPScaleDecision {
	if desired < 1 {
		desired = 1
	}

	// Desire fell: shrink unconditionally to it (Phase 7). This is the only scale-down trigger.
	if desired < current {
		return EPScaleDecision{Action: EPScaleShrink, Target: desired}
	}

	// Not shrinking, yet fewer owned ranks are alive than the engine runs on: a rank died.
	if observed < current {
		return EPScaleDecision{Action: EPScaleFault, Target: current}
	}

	// Grow into joined capacity, but never past what the driver asked for.
	target := observed
	if desired < target {
		target = desired
	}
	if target > current {
		return EPScaleDecision{Action: EPScaleGrow, Target: target}
	}

	return EPScaleDecision{Action: EPScaleNone, Target: current}
}
