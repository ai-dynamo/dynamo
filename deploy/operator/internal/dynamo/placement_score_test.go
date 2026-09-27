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
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	schedulergrovev1alpha1 "github.com/ai-dynamo/grove/scheduler/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestAggregatePlacementScore(t *testing.T) {
	const (
		gangA = "pcs-0-anchor"
		gangB = "pcs-0-scaleout"
	)

	testCases := []struct {
		name      string
		gangs     []schedulergrovev1alpha1.PodGang
		wantScore *float64
		wantState nvidiacomv1beta1.PlacementScoreState
	}{
		{
			name:      "no PodGangs observed yet is indeterminate",
			gangs:     nil,
			wantScore: nil,
			wantState: nvidiacomv1beta1.PlacementScoreStateUnknown,
		},
		{
			name:      "a single scored PodGang is a complete observation",
			gangs:     []schedulergrovev1alpha1.PodGang{placementGang(gangA, ptr.To(0.8))},
			wantScore: ptr.To(0.8),
			wantState: nvidiacomv1beta1.PlacementScoreStateReported,
		},
		{
			name: "the worst placement wins across fully scored PodGangs",
			gangs: []schedulergrovev1alpha1.PodGang{
				placementGang(gangA, ptr.To(0.95)),
				placementGang(gangB, ptr.To(0.35)),
			},
			wantScore: ptr.To(0.35),
			wantState: nvidiacomv1beta1.PlacementScoreStateReported,
		},
		{
			name: "an unscored PodGang makes the observation partial",
			gangs: []schedulergrovev1alpha1.PodGang{
				placementGang(gangA, ptr.To(0.6)),
				placementGang(gangB, nil),
			},
			wantScore: ptr.To(0.6),
			wantState: nvidiacomv1beta1.PlacementScoreStatePartial,
		},
		{
			name: "no scored PodGang is indeterminate even when PodGangs exist",
			gangs: []schedulergrovev1alpha1.PodGang{
				placementGang(gangA, nil),
				placementGang(gangB, nil),
			},
			wantScore: nil,
			wantState: nvidiacomv1beta1.PlacementScoreStateUnknown,
		},
		{
			name:      "the lower bound is a usable score",
			gangs:     []schedulergrovev1alpha1.PodGang{placementGang(gangA, ptr.To(0.0))},
			wantScore: ptr.To(0.0),
			wantState: nvidiacomv1beta1.PlacementScoreStateReported,
		},
		{
			name:      "the upper bound is a usable score",
			gangs:     []schedulergrovev1alpha1.PodGang{placementGang(gangA, ptr.To(1.0))},
			wantScore: ptr.To(1.0),
			wantState: nvidiacomv1beta1.PlacementScoreStateReported,
		},
		{
			// The DGD status field is bounded to [0, 1] while Grove's PodGang
			// status is not, so an out-of-range report must not be copied through:
			// it would fail every subsequent status write.
			name:      "a score above the upper bound is not copied into the bounded field",
			gangs:     []schedulergrovev1alpha1.PodGang{placementGang(gangA, ptr.To(1.5))},
			wantScore: nil,
			wantState: nvidiacomv1beta1.PlacementScoreStateUnknown,
		},
		{
			name:      "a negative score is not copied into the bounded field",
			gangs:     []schedulergrovev1alpha1.PodGang{placementGang(gangA, ptr.To(-0.25))},
			wantScore: nil,
			wantState: nvidiacomv1beta1.PlacementScoreStateUnknown,
		},
		{
			name: "an out-of-range report alongside a valid one is a partial observation",
			gangs: []schedulergrovev1alpha1.PodGang{
				placementGang(gangA, ptr.To(0.7)),
				placementGang(gangB, ptr.To(2.0)),
			},
			wantScore: ptr.To(0.7),
			wantState: nvidiacomv1beta1.PlacementScoreStatePartial,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			score, state := AggregatePlacementScore(tc.gangs)

			require.Equal(t, tc.wantState, state, "unexpected placement score state")

			// A score is reported only when the state claims one is available.
			if tc.wantScore == nil {
				require.Nil(t, score, "score must be cleared when the state is not Reported or Partial")
				return
			}
			require.NotNil(t, score, "score must be reported")
			require.InDelta(t, *tc.wantScore, *score, 0.0, "unexpected aggregated score")
		})
	}
}

// placementGang builds a PodGang with the given reported placement score. A nil
// score models a scheduler that has not published a score for this PodGang yet.
func placementGang(name string, score *float64) schedulergrovev1alpha1.PodGang {
	return schedulergrovev1alpha1.PodGang{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status: schedulergrovev1alpha1.PodGangStatus{
			PlacementScore: score,
		},
	}
}
