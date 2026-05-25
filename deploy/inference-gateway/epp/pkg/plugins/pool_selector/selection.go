/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package pool_selector implements cross-pool (ISL/SLA-aware) selection for the
// Dynamo EPP. It is the Go counterpart of the standalone Global Router's
// selection logic (components/src/dynamo/global_router/pool_selection.py): a
// precomputed 2D grid maps a (size, latency-target) pair to a pool index, so the
// gateway can route across heterogeneous pools (e.g. TP1/TP2/TP4) before the
// within-pool KV/load scorer picks an endpoint.
//
// Both the EPP (this package) and the Global Router consume the *same*
// profiler-derived grid; keeping the lookup identical here preserves parity
// between the standalone and gateway routing paths.
package pool_selector

import "fmt"

// PriorityOverride routes a request to TargetPool when its priority falls within
// [MinPriority, MaxPriority] (inclusive), taking precedence over the grid.
type PriorityOverride struct {
	MinPriority int `json:"minPriority"`
	MaxPriority int `json:"maxPriority"`
	TargetPool  int `json:"targetPool"`
}

// GridSelectionStrategy maps a (size, latency-target) pair to a pool index via a
// precomputed 2D grid. It generalizes the Global Router's
// PrefillPoolSelectionStrategy (size=ISL, latency=TTFT) and
// DecodePoolSelectionStrategy (size=context length, latency=ITL): both are the
// same shape — two clamped axes indexing into Mapping[sizeIdx][latencyIdx].
type GridSelectionStrategy struct {
	// Size axis: ISL for prefill, context length for decode.
	SizeMin        float64 `json:"sizeMin"`
	SizeMax        float64 `json:"sizeMax"`
	SizeResolution int     `json:"sizeResolution"`

	// Latency axis (ms): TTFT target for prefill, ITL target for decode.
	LatencyMinMs      float64 `json:"latencyMinMs"`
	LatencyMaxMs      float64 `json:"latencyMaxMs"`
	LatencyResolution int     `json:"latencyResolution"`

	// Mapping[sizeIdx][latencyIdx] = pool index. Dimensions must be
	// SizeResolution rows by LatencyResolution columns.
	Mapping [][]int `json:"mapping"`

	// PriorityOverrides take precedence over the grid (first match wins).
	PriorityOverrides []PriorityOverride `json:"priorityOverrides,omitempty"`
}

// Validate checks the grid is well-formed (positive resolutions, finite ranges,
// and Mapping dimensions matching the resolutions with in-range pool indices).
func (g *GridSelectionStrategy) Validate(numPools int) error {
	if g.SizeResolution <= 0 || g.LatencyResolution <= 0 {
		return fmt.Errorf("resolutions must be positive (size=%d, latency=%d)", g.SizeResolution, g.LatencyResolution)
	}
	if g.SizeMax <= g.SizeMin || g.LatencyMaxMs <= g.LatencyMinMs {
		return fmt.Errorf("max must exceed min (size %v..%v, latency %v..%v)", g.SizeMin, g.SizeMax, g.LatencyMinMs, g.LatencyMaxMs)
	}
	if len(g.Mapping) != g.SizeResolution {
		return fmt.Errorf("mapping has %d rows, want %d (sizeResolution)", len(g.Mapping), g.SizeResolution)
	}
	for i, row := range g.Mapping {
		if len(row) != g.LatencyResolution {
			return fmt.Errorf("mapping row %d has %d cols, want %d (latencyResolution)", i, len(row), g.LatencyResolution)
		}
		for j, pool := range row {
			if pool < 0 || pool >= numPools {
				return fmt.Errorf("mapping[%d][%d]=%d out of range [0,%d)", i, j, pool, numPools)
			}
		}
	}
	for k, o := range g.PriorityOverrides {
		if o.MinPriority > o.MaxPriority {
			return fmt.Errorf("priorityOverride[%d]: minPriority %d > maxPriority %d", k, o.MinPriority, o.MaxPriority)
		}
		if o.TargetPool < 0 || o.TargetPool >= numPools {
			return fmt.Errorf("priorityOverride[%d]: targetPool %d out of range [0,%d)", k, o.TargetPool, numPools)
		}
	}
	return nil
}

func (g *GridSelectionStrategy) sizeStep() float64 {
	return (g.SizeMax - g.SizeMin) / float64(g.SizeResolution)
}

func (g *GridSelectionStrategy) latencyStep() float64 {
	return (g.LatencyMaxMs - g.LatencyMinMs) / float64(g.LatencyResolution)
}

// SelectPool returns the pool index for the given size and latency target.
//
//   - latencyTargetMs <= 0 means "unspecified" → the middle of the configured
//     range is used (matching the Global Router's None handling).
//   - priority < 0 means "no priority" → grid result is used as-is.
//
// Indices are clamped into the grid, so out-of-range sizes/latencies map to the
// nearest edge cell rather than erroring.
func (g *GridSelectionStrategy) SelectPool(size float64, latencyTargetMs float64, priority int) int {
	if latencyTargetMs <= 0 {
		latencyTargetMs = (g.LatencyMinMs + g.LatencyMaxMs) / 2
	}
	sizeIdx := clampIndex((size-g.SizeMin)/g.sizeStep(), g.SizeResolution)
	latIdx := clampIndex((latencyTargetMs-g.LatencyMinMs)/g.latencyStep(), g.LatencyResolution)
	pool := g.Mapping[sizeIdx][latIdx]
	return applyPriorityOverrides(pool, priority, g.PriorityOverrides)
}

// clampIndex truncates value to an int and clamps it to [0, resolution-1],
// mirroring PrefillPoolSelectionStrategy._clamp_index.
func clampIndex(value float64, resolution int) int {
	i := int(value)
	if i < 0 {
		return 0
	}
	if i > resolution-1 {
		return resolution - 1
	}
	return i
}

// applyPriorityOverrides returns the first override's TargetPool whose range
// contains priority, else base. priority < 0 or no overrides → base.
func applyPriorityOverrides(base, priority int, overrides []PriorityOverride) int {
	if priority < 0 || len(overrides) == 0 {
		return base
	}
	for _, r := range overrides {
		if r.MinPriority <= priority && priority <= r.MaxPriority {
			return r.TargetPool
		}
	}
	return base
}
