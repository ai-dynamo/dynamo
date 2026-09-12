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

import "testing"

func TestDecideEPScale(t *testing.T) {
	tests := []struct {
		name                       string
		current, observed, desired int
		wantAction                 EPScaleAction
		wantTarget                 int
	}{
		{
			name:    "converged: nothing joined beyond current, desire matches",
			current: 3, observed: 3, desired: 3, wantAction: EPScaleNone, wantTarget: 3,
		},
		{
			name:    "grow: four owned nodes joined, desire allows it -> one call to 4",
			current: 1, observed: 4, desired: 4, wantAction: EPScaleGrow, wantTarget: 4,
		},
		{
			name:    "grow bounded by desire: five joined but only four desired -> grow to 4",
			current: 1, observed: 5, desired: 4, wantAction: EPScaleGrow, wantTarget: 4,
		},
		{
			name:    "grow partial: two of four desired have joined -> use what's available, grow to 2",
			current: 1, observed: 2, desired: 4, wantAction: EPScaleGrow, wantTarget: 2,
		},
		{
			name:    "no grow: desire raised but nothing new has joined yet",
			current: 2, observed: 2, desired: 5, wantAction: EPScaleNone, wantTarget: 2,
		},
		{
			name:    "shrink: desired fell below current -> shrink to desired, ignoring observed",
			current: 4, observed: 4, desired: 2, wantAction: EPScaleShrink, wantTarget: 2,
		},
		{
			name:    "shrink to the one-rank floor",
			current: 3, observed: 3, desired: 1, wantAction: EPScaleShrink, wantTarget: 1,
		},
		{
			name:    "shrink clamps a below-floor desire up to one",
			current: 3, observed: 3, desired: 0, wantAction: EPScaleShrink, wantTarget: 1,
		},
		{
			name:    "FAULT: a live rank's pod died (observed < current) and we are not shrinking -> do not scale",
			current: 4, observed: 2, desired: 4, wantAction: EPScaleFault, wantTarget: 4,
		},
		{
			name:    "FAULT even when idle desire is higher: never scale down to match lost capacity",
			current: 3, observed: 1, desired: 8, wantAction: EPScaleFault, wantTarget: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DecideEPScale(tt.current, tt.observed, tt.desired)
			if got.Action != tt.wantAction || got.Target != tt.wantTarget {
				t.Errorf("DecideEPScale(current=%d, observed=%d, desired=%d) = {%s, %d}, want {%s, %d}",
					tt.current, tt.observed, tt.desired, got.Action, got.Target, tt.wantAction, tt.wantTarget)
			}
		})
	}
}

func TestOwnedRayNodeCount_GenerationFilter(t *testing.T) {
	cap := EPCapacity{
		Nodes: []EPRayNode{
			{NodeIP: "10.0.0.1", TotalGPUs: 4}, // leader (owned)
			{NodeIP: "10.0.0.2", TotalGPUs: 4}, // current-gen follower (owned)
			{NodeIP: "10.9.9.9", TotalGPUs: 4}, // superseded-generation pod still alive in Ray (NOISE)
		},
	}
	owned := map[string]struct{}{
		"10.0.0.1": {},
		"10.0.0.2": {},
		"10.0.0.3": {}, // a desired pod not yet joined -- present in ownership, absent from Ray
	}

	t.Log("only Ray nodes whose IP is an owned current-generation pod count; the superseded pod is excluded")
	if got := OwnedRayNodeCount(cap, owned); got != 2 {
		t.Errorf("OwnedRayNodeCount = %d, want 2 (leader + one joined follower; superseded 10.9.9.9 excluded)", got)
	}

	t.Log("counting every live Ray node instead would over-count and grow the engine onto a rank it does not own")
	if len(cap.Nodes) == 2 {
		t.Fatal("test fixture should include the superseded node to prove it is filtered")
	}
}

func TestOwnedRayNodeCount_Empty(t *testing.T) {
	if got := OwnedRayNodeCount(EPCapacity{}, map[string]struct{}{"10.0.0.1": {}}); got != 0 {
		t.Errorf("no Ray nodes -> owned 0, got %d", got)
	}
}
