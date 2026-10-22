// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dependency-free arithmetic for the adaptive policy. Everything here is a pure function.

/// Coefficient of variation of `loads`, clamped to `0..=1`. Returns 0 for fewer than two
/// workers or when all loads are zero, so an idle or single-worker pool never looks skewed.
pub fn imbalance(loads: &[f64]) -> f64 {
    if loads.len() < 2 {
        return 0.0;
    }
    let n = loads.len() as f64;
    let mean = loads.iter().sum::<f64>() / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let var = loads.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (var.sqrt() / mean).clamp(0.0, 1.0)
}

/// Exponential moving average: `alpha` is the weight of the previous value.
pub fn smooth(previous: f64, raw: f64, alpha: f64) -> f64 {
    alpha * previous + (1.0 - alpha) * raw
}

/// Logistic curve centred on `midpoint` with steepness `slope`, output in `0..1`.
pub fn sigmoid(x: f64, midpoint: f64, slope: f64) -> f64 {
    1.0 / (1.0 + (-slope * (x - midpoint)).exp())
}

/// The three router weights derived from one `split` value and the baselines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightSet {
    pub overlap_score_credit: f64,
    pub prefill_load_scale: f64,
    pub decode_active_request_weight: f64,
}

/// Affinity goes down with `split`, both load weights go up by the same factor `1 + gain × split`.
pub fn weights_for(base: WeightSet, split: f64, gain: f64) -> WeightSet {
    let up = 1.0 + gain * split;
    WeightSet {
        overlap_score_credit: base.overlap_score_credit * (1.0 - split),
        prefill_load_scale: base.prefill_load_scale * up,
        decode_active_request_weight: base.decode_active_request_weight * up,
    }
}

/// The default selector's device-cache and load terms, lower is better: prefill work left after
/// the cache credit, plus memory in use, plus a per-request penalty.
///
/// `request_blocks` and `prefill_backlog_blocks` are token counts divided by the block size, as the
/// default selector computes them. As there, the prefill term is zero when the request does not track
/// prefill tokens (`track_prefill == false`).
pub fn cost_for(
    w: WeightSet,
    request_blocks: f64,
    prefill_backlog_blocks: f64,
    track_prefill: bool,
    cached_blocks: f64,
    decode_blocks: f64,
    active_requests: f64,
) -> f64 {
    let prefill = if track_prefill {
        let prefill_left = (prefill_backlog_blocks + request_blocks
            - w.overlap_score_credit * cached_blocks)
            .max(0.0);
        w.prefill_load_scale * prefill_left
    } else {
        0.0
    };
    prefill + decode_blocks + w.decode_active_request_weight * active_requests
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: WeightSet = WeightSet {
        overlap_score_credit: 1.0,
        prefill_load_scale: 1.0,
        decode_active_request_weight: 0.5,
    };

    #[test]
    fn imbalance_is_zero_for_even_or_degenerate_pools() {
        assert_eq!(imbalance(&[5.0, 5.0, 5.0, 5.0]), 0.0);
        assert_eq!(imbalance(&[7.0]), 0.0);
        assert_eq!(imbalance(&[0.0, 0.0]), 0.0);
        assert_eq!(imbalance(&[]), 0.0);
    }

    #[test]
    fn imbalance_matches_the_design_example() {
        // [30, 4, 2, 5]: mean 10.25, population stddev ≈ 11.45, ratio ≈ 1.12, clamped to 1.
        assert_eq!(imbalance(&[30.0, 4.0, 2.0, 5.0]), 1.0);
        // mild skew stays well below 1
        let mild = imbalance(&[12.0, 4.0, 3.0, 5.0]);
        assert!(mild > 0.5 && mild < 0.7, "got {mild}");
    }

    #[test]
    fn sigmoid_has_midpoint_and_saturates() {
        assert!((sigmoid(0.5, 0.5, 10.0) - 0.5).abs() < 1e-9);
        assert!(sigmoid(0.1, 0.5, 10.0) < 0.02);
        assert!(sigmoid(0.9, 0.5, 10.0) > 0.98);
    }

    #[test]
    fn weights_at_split_zero_equal_baseline() {
        assert_eq!(weights_for(BASE, 0.0, 4.0), BASE);
    }

    #[test]
    fn weights_move_opposite_ways_with_split() {
        let w = weights_for(BASE, 0.98, 4.0);
        assert!((w.overlap_score_credit - 0.02).abs() < 1e-9);
        assert!((w.prefill_load_scale - 4.92).abs() < 1e-9);
        assert!((w.decode_active_request_weight - 2.46).abs() < 1e-9);
    }

    #[test]
    fn cost_reproduces_the_worked_example() {
        // No prefill backlog and prefill tracking on, so only the request's own blocks count.
        // baseline: W1 (60 cached, 30 active) = 4 + 15 = 19 ; W4 (20 cached, 5 active) = 44 + 2.5 = 46.5
        assert!((cost_for(BASE, 64.0, 0.0, true, 60.0, 0.0, 30.0) - 19.0).abs() < 1e-9);
        assert!((cost_for(BASE, 64.0, 0.0, true, 20.0, 0.0, 5.0) - 46.5).abs() < 1e-9);
        // hot pool: with split 0.98 W4 becomes cheaper than W1
        let hot = weights_for(BASE, 0.98, 4.0);
        let w1 = cost_for(hot, 64.0, 0.0, true, 60.0, 0.0, 30.0);
        let w4 = cost_for(hot, 64.0, 0.0, true, 20.0, 0.0, 5.0);
        assert!(w4 < w1, "w4={w4} w1={w1}");
    }

    #[test]
    fn cost_never_rewards_cache_beyond_the_request() {
        // 70 cached blocks for a 64-block request must not go negative
        assert_eq!(cost_for(BASE, 64.0, 0.0, true, 70.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn prefill_backlog_adds_to_the_uncached_blocks_before_scaling() {
        // 64 request blocks, 60 cached, 10 blocks already queued for prefill: 1 × (10 + 64 − 60) = 14.
        assert!((cost_for(BASE, 64.0, 10.0, true, 60.0, 0.0, 0.0) - 14.0).abs() < 1e-9);
        // The cache credit can absorb the backlog too, but the term never goes negative.
        assert_eq!(cost_for(BASE, 64.0, 10.0, true, 80.0, 0.0, 0.0), 0.0);
        // The scale multiplies the backlog as well: 3 × (10 + 64 − 60) = 42.
        let scaled = WeightSet {
            prefill_load_scale: 3.0,
            ..BASE
        };
        assert!((cost_for(scaled, 64.0, 10.0, true, 60.0, 0.0, 0.0) - 42.0).abs() < 1e-9);
    }

    #[test]
    fn untracked_prefill_drops_the_whole_prefill_term() {
        // Only decode blocks (7) and the active-request penalty (0.5 × 4 = 2) remain.
        assert!((cost_for(BASE, 64.0, 10.0, false, 0.0, 7.0, 4.0) - 9.0).abs() < 1e-9);
    }
}
