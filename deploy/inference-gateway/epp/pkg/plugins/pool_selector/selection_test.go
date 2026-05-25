/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package pool_selector

import "testing"

// grid3x2 models 3 ISL bands × 2 TTFT bands over a 3-pool fleet (TP1/TP2/TP4).
// Size axis [0,3000) → 3 bands of 1000. Latency axis [1000,5000) → 2 bands of
// 2000, so column 0 = tight TTFT [1000,3000) and column 1 = loose [3000,5000).
// Semantics encoded: tighter TTFT and/or longer ISL prefer a bigger pool.
func grid3x2() *GridSelectionStrategy {
	return &GridSelectionStrategy{
		SizeMin: 0, SizeMax: 3000, SizeResolution: 3,
		LatencyMinMs: 1000, LatencyMaxMs: 5000, LatencyResolution: 2,
		Mapping: [][]int{
			//  tight  loose
			{1, 0}, // ISL [0,1000):    tight→1, loose→0
			{2, 1}, // ISL [1000,2000): tight→2, loose→1
			{2, 2}, // ISL [2000,3000): →2
		},
	}
}

func TestValidate(t *testing.T) {
	if err := grid3x2().Validate(3); err != nil {
		t.Fatalf("valid grid rejected: %v", err)
	}
	bad := grid3x2()
	bad.Mapping[2][1] = 9 // pool out of range
	if err := bad.Validate(3); err == nil {
		t.Fatal("expected out-of-range pool index to fail validation")
	}
	dims := grid3x2()
	dims.SizeResolution = 4 // mapping still has 3 rows
	if err := dims.Validate(3); err == nil {
		t.Fatal("expected row-count mismatch to fail validation")
	}
}

func TestSelectPool_GridLookup(t *testing.T) {
	g := grid3x2()
	cases := []struct {
		name      string
		size      float64
		latencyMs float64 // TTFT target
		want      int
	}{
		{"short loose", 500, 4000, 0},   // band0, col1
		{"medium tight", 1500, 1500, 2}, // band1, col0
		{"long", 2500, 4000, 2},         // band2
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := g.SelectPool(c.size, c.latencyMs, -1); got != c.want {
				t.Fatalf("SelectPool(%v,%v)=%d, want %d", c.size, c.latencyMs, got, c.want)
			}
		})
	}
}

func TestSelectPool_ClampsOutOfRange(t *testing.T) {
	g := grid3x2()
	// size below min clamps to band 0; above max clamps to last band. (col1=loose for 4000)
	if got := g.SelectPool(-100, 4000, -1); got != 0 {
		t.Fatalf("below-min size: got pool %d, want 0", got)
	}
	if got := g.SelectPool(99999, 4000, -1); got != 2 {
		t.Fatalf("above-max size: got pool %d, want 2", got)
	}
	// latency below min clamps to col0 (tight); above max clamps to col1 (loose). size 1500 = band1.
	if got := g.SelectPool(1500, 10, -1); got != 2 {
		t.Fatalf("below-min latency: got pool %d, want 2 (tight col)", got)
	}
	if got := g.SelectPool(1500, 99999, -1); got != 1 {
		t.Fatalf("above-max latency: got pool %d, want 1 (loose col)", got)
	}
}

func TestSelectPool_DefaultLatencyUsesMiddle(t *testing.T) {
	g := grid3x2()
	// latency < 0 (absent sentinel) → middle of [1000,5000] = 3000 →
	// clampIndex((3000-1000)/2000)=1 (loose col). size 1500 = band1 → mapping[1][1] = 1.
	if got := g.SelectPool(1500, -1, -1); got != 1 {
		t.Fatalf("absent latency: got pool %d, want 1 (middle→loose column)", got)
	}
}

// TestSelectPool_ExplicitZeroIsStrictest verifies an explicit 0 target is treated
// as a real (strictest) target, not "unspecified": it clamps to the fastest
// bucket — parity with the Python Global Router, which defaults only on None.
func TestSelectPool_ExplicitZeroIsStrictest(t *testing.T) {
	g := grid3x2()
	// ttft 0 → clampIndex((0-1000)/2000) = 0 (tight col). size 1500 = band1 → mapping[1][0] = 2.
	if got := g.SelectPool(1500, 0, -1); got != 2 {
		t.Fatalf("explicit zero: got pool %d, want 2 (tight column / fastest bucket)", got)
	}
}

func TestSelectPool_PriorityOverride(t *testing.T) {
	g := grid3x2()
	g.PriorityOverrides = []PriorityOverride{
		{MinPriority: 10, MaxPriority: 20, TargetPool: 0}, // high-priority → fastest pool
	}
	// Grid would pick pool 2 (long ISL), but priority 15 overrides to pool 0.
	if got := g.SelectPool(2500, 4000, 15); got != 0 {
		t.Fatalf("priority override: got pool %d, want 0", got)
	}
	// Priority outside any range → grid result.
	if got := g.SelectPool(2500, 4000, 5); got != 2 {
		t.Fatalf("non-matching priority: got pool %d, want 2", got)
	}
	// priority < 0 → grid result even with overrides present.
	if got := g.SelectPool(2500, 4000, -1); got != 2 {
		t.Fatalf("no priority: got pool %d, want 2", got)
	}
}
