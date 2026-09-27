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

import (
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	schedulergrovev1alpha1 "github.com/ai-dynamo/grove/scheduler/api/core/v1alpha1"
)

// placementScoreLowerBound and placementScoreUpperBound are the bounds the DGD
// status field enforces. Grove's PodGang status does not constrain the value it
// reports, so the operator must not copy an out-of-range score into a bounded
// status field: doing so would fail every subsequent status update and wedge the
// DGD. Out-of-range reports are therefore treated as unreported.
const (
	placementScoreLowerBound = 0.0
	placementScoreUpperBound = 1.0
)

// AggregatePlacementScore derives the DGD-level placement score from the
// scheduler PodGangs observed for one PodCliqueSet generation.
//
// The returned score is the minimum across the scored PodGangs, so one poorly
// placed serving unit cannot be averaged away by well-placed siblings. The
// contract is documented on nvidiacomv1beta1.PlacementStatus.
//
// gangs must be the PodGangs belonging to the generation being observed; the
// caller owns that selection. An empty slice is supported and means the
// scheduler has not published anything for that generation.
//
// The returned state is never Unsupported: that value describes a workload
// pathway with no placement score source at all, which is a caller decision
// rather than an aggregation outcome. A nil score is returned with every state
// other than Reported and Partial.
func AggregatePlacementScore(
	gangs []schedulergrovev1alpha1.PodGang,
) (*float64, nvidiacomv1beta1.PlacementScoreState) {
	var scored int
	var lowest float64

	// Fold the usable reports together, counting how many were usable.
	for i := range gangs {
		score := gangs[i].Status.PlacementScore
		if score == nil || *score < placementScoreLowerBound || *score > placementScoreUpperBound {
			continue
		}
		if scored == 0 || *score < lowest {
			lowest = *score
		}
		scored++
	}

	// No usable score yet is indeterminate, not a permanent absence.
	if scored == 0 {
		return nil, nvidiacomv1beta1.PlacementScoreStateUnknown
	}

	// Some gangs reporting and others not is a partial observation of the graph.
	state := nvidiacomv1beta1.PlacementScoreStateReported
	if scored < len(gangs) {
		state = nvidiacomv1beta1.PlacementScoreStatePartial
	}

	return &lowest, state
}
