// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Non-prefix ("blended") KV reuse plans for the mocker.
//!
//! A [`SemanticReusePlan`] describes, per request, which token ranges of a
//! previously-seen donor sequence can seed the recipient's KV instead of
//! being prefilled from scratch. Plans are *inputs* to the simulator: the
//! mocker prices reuse (copy bandwidth, positional repair, recompute halos);
//! it never judges whether reuse is semantically safe. Plan construction is
//! provider territory.
//!
//! Plans are carried out-of-band in [`SemanticSimConfig::plans`], keyed by
//! the request UUID, so `DirectRequest` and existing trace formats stay
//! untouched.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// How a reused segment is materialized, in increasing cost order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentMode {
    /// Donor KV is copied into recipient-owned slots (copy bandwidth cost).
    Copied,
    /// Copied and RoPE-repositioned (copy + per-token repair cost).
    Repaired,
    /// Boundary/halo region deliberately recomputed (normal prefill cost).
    Recomputed,
}

/// One reused span. Ranges are half-open `[start, end)` token positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReuseSegment {
    /// Token positions in the recipient prompt.
    pub recipient_range: (usize, usize),
    /// Token positions in the donor sequence.
    pub donor_range: (usize, usize),
    pub mode: SegmentMode,
}

impl ReuseSegment {
    fn len(&self) -> usize {
        self.recipient_range
            .1
            .saturating_sub(self.recipient_range.0)
    }
}

/// A structured materialization plan for one request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticReusePlan {
    pub donor_uuid: Uuid,
    /// Reserved for cross-worker donors. v1 simulation only resolves donors
    /// resident on the same worker; plans with a foreign donor fall back.
    #[serde(default)]
    pub donor_worker: Option<u64>,
    pub segments: Vec<ReuseSegment>,
    /// Staleness gate: plans built against an older provider generation
    /// than the engine's current one fall back to cold prefill.
    #[serde(default)]
    pub provider_generation: u64,
}

/// Why a plan was not applied. Requests always fall back to normal
/// (exact-prefix + cold) prefill; fallback is accounting, not failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackReason {
    Disabled,
    DonorUnknown,
    DonorEvicted,
    GenerationStale,
    InvalidPlan,
    BelowMinSegment,
    BelowMinSaved,
    ChunkedUnsupported,
}

/// Per-request semantic admission outcome, surfaced into per-request records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SemanticOutcome {
    Accepted {
        copied_tokens: usize,
        repaired_tokens: usize,
        recomputed_halo_tokens: usize,
    },
    Fallback {
        reason: FallbackReason,
    },
}

/// Simulation config for blended-prefill pricing. Additive and disabled by
/// default: when `enabled` is false (or the whole config is absent) the
/// scheduler path is provably unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSimConfig {
    #[serde(default)]
    pub enabled: bool,
    /// KV bytes per token (all layers, K+V). Used for copy-time pricing.
    pub kv_bytes_per_token: f64,
    /// Effective same-worker (device-to-device) copy bandwidth.
    pub intra_worker_copy_gbps: f64,
    /// Per-token RoPE repositioning cost for `Repaired` segments.
    pub rope_repair_us_per_token: f64,
    /// Fixed per-accepted-plan resolution overhead.
    pub plan_overhead_ms: f64,
    /// Segments shorter than this (after exact-prefix trim) are dropped.
    pub min_segment_tokens: usize,
    /// Plans saving fewer total tokens than this fall back entirely.
    pub min_total_saved_tokens: usize,
    /// Engine-side provider generation; plans from older generations fall back.
    #[serde(default)]
    pub provider_generation: u64,
    /// Out-of-band plan map keyed by request UUID. Not serialized; the
    /// driver populates it programmatically.
    #[serde(skip, default)]
    pub plans: Arc<HashMap<Uuid, SemanticReusePlan>>,
}

/// A plan after exact-prefix trimming and admission gating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPlan {
    pub copied_tokens: usize,
    pub repaired_tokens: usize,
    pub recomputed_halo_tokens: usize,
    /// Donor tokens that must still be resident: max donor_range.1 over
    /// kept Copied/Repaired segments.
    pub donor_pin_len: usize,
}

impl ResolvedPlan {
    pub fn reused_tokens(&self) -> usize {
        self.copied_tokens + self.repaired_tokens
    }
}

/// Trim a plan against the recipient's exact-prefix match and apply gates.
///
/// Exact prefix always wins: any part of a segment covered by the exact
/// match is dropped (the radix cache already serves it). Segments must be
/// sorted, non-overlapping on the recipient side, in bounds on both sides,
/// and donor/recipient spans of equal length.
pub fn trim_and_gate(
    plan: &SemanticReusePlan,
    exact_prefix_len: usize,
    prompt_len: usize,
    donor_len: usize,
    cfg: &SemanticSimConfig,
) -> Result<ResolvedPlan, FallbackReason> {
    if plan.provider_generation < cfg.provider_generation {
        return Err(FallbackReason::GenerationStale);
    }

    let mut prev_end = 0usize;
    let mut copied = 0usize;
    let mut repaired = 0usize;
    let mut halo = 0usize;
    let mut pin_len = 0usize;

    for seg in &plan.segments {
        let (rs, re) = seg.recipient_range;
        let (ds, de) = seg.donor_range;
        if rs >= re || re > prompt_len || rs < prev_end {
            return Err(FallbackReason::InvalidPlan);
        }
        if re - rs != de - ds || de > donor_len {
            return Err(FallbackReason::InvalidPlan);
        }
        prev_end = re;

        if seg.mode == SegmentMode::Recomputed {
            // Halos are priced as normal compute; track for accounting only.
            halo += seg.len();
            continue;
        }

        // Clip away the part served by the exact prefix.
        let clipped_rs = rs.max(exact_prefix_len);
        if clipped_rs >= re {
            continue; // fully inside the exact prefix
        }
        let clipped_ds = ds + (clipped_rs - rs);
        let kept = re - clipped_rs;
        if kept < cfg.min_segment_tokens {
            continue; // too small to be worth a copy after trimming
        }
        match seg.mode {
            SegmentMode::Copied => copied += kept,
            SegmentMode::Repaired => repaired += kept,
            SegmentMode::Recomputed => unreachable!(),
        }
        pin_len = pin_len.max(clipped_ds + kept);
    }

    if copied + repaired == 0 {
        return Err(FallbackReason::BelowMinSegment);
    }
    if copied + repaired < cfg.min_total_saved_tokens {
        return Err(FallbackReason::BelowMinSaved);
    }

    Ok(ResolvedPlan {
        copied_tokens: copied,
        repaired_tokens: repaired,
        recomputed_halo_tokens: halo,
        donor_pin_len: pin_len,
    })
}

/// Price the blended portion of a prefill batch.
///
/// Copy overlaps compute (`max`); RoPE repair serializes before logits; a
/// fixed per-plan resolution overhead is added on top.
pub fn blended_extra_ms(
    compute_ms: f64,
    copied_tokens: usize,
    repaired_tokens: usize,
    accepted_plans: usize,
    cfg: &SemanticSimConfig,
) -> f64 {
    let moved_bytes = (copied_tokens + repaired_tokens) as f64 * cfg.kv_bytes_per_token;
    let copy_ms = if cfg.intra_worker_copy_gbps > 0.0 {
        moved_bytes / (cfg.intra_worker_copy_gbps * 1e6)
    } else {
        0.0
    };
    let repair_ms = repaired_tokens as f64 * cfg.rope_repair_us_per_token / 1000.0;
    let overhead_ms = accepted_plans as f64 * cfg.plan_overhead_ms;
    (copy_ms - compute_ms).max(0.0) + repair_ms + overhead_ms
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(min_segment: usize, min_saved: usize) -> SemanticSimConfig {
        SemanticSimConfig {
            enabled: true,
            kv_bytes_per_token: 114_688.0,
            intra_worker_copy_gbps: 1500.0,
            rope_repair_us_per_token: 0.5,
            plan_overhead_ms: 0.2,
            min_segment_tokens: min_segment,
            min_total_saved_tokens: min_saved,
            provider_generation: 0,
            plans: Arc::default(),
        }
    }

    fn plan(segments: Vec<ReuseSegment>) -> SemanticReusePlan {
        SemanticReusePlan {
            donor_uuid: Uuid::nil(),
            donor_worker: None,
            segments,
            provider_generation: 0,
        }
    }

    fn seg(r: (usize, usize), d: (usize, usize), mode: SegmentMode) -> ReuseSegment {
        ReuseSegment {
            recipient_range: r,
            donor_range: d,
            mode,
        }
    }

    #[test]
    fn exact_prefix_wins_and_segments_clip() {
        let p = plan(vec![
            seg((0, 100), (0, 100), SegmentMode::Copied),
            seg((150, 250), (300, 400), SegmentMode::Repaired),
        ]);
        // Exact prefix covers [0, 120): first segment fully absorbed.
        let r = trim_and_gate(&p, 120, 400, 500, &cfg(16, 32)).unwrap();
        assert_eq!(r.copied_tokens, 0);
        assert_eq!(r.repaired_tokens, 100);
        assert_eq!(r.donor_pin_len, 400);
    }

    #[test]
    fn partial_clip_shifts_donor_side() {
        let p = plan(vec![seg((100, 200), (50, 150), SegmentMode::Copied)]);
        let r = trim_and_gate(&p, 130, 300, 200, &cfg(16, 32)).unwrap();
        assert_eq!(r.copied_tokens, 70);
        // Donor start shifts by the same 30 tokens: pin to 80 + 70 = 150.
        assert_eq!(r.donor_pin_len, 150);
    }

    #[test]
    fn gates_fire() {
        let p = plan(vec![seg((100, 110), (0, 10), SegmentMode::Copied)]);
        assert_eq!(
            trim_and_gate(&p, 0, 200, 100, &cfg(16, 32)),
            Err(FallbackReason::BelowMinSegment)
        );
        let p = plan(vec![seg((100, 120), (0, 20), SegmentMode::Copied)]);
        assert_eq!(
            trim_and_gate(&p, 0, 200, 100, &cfg(16, 64)),
            Err(FallbackReason::BelowMinSaved)
        );
    }

    #[test]
    fn invalid_plans_rejected() {
        // Overlapping recipient ranges.
        let p = plan(vec![
            seg((0, 100), (0, 100), SegmentMode::Copied),
            seg((50, 150), (200, 300), SegmentMode::Copied),
        ]);
        assert_eq!(
            trim_and_gate(&p, 0, 200, 400, &cfg(1, 1)),
            Err(FallbackReason::InvalidPlan)
        );
        // Length mismatch between donor and recipient spans.
        let p = plan(vec![seg((0, 100), (0, 50), SegmentMode::Copied)]);
        assert_eq!(
            trim_and_gate(&p, 0, 200, 400, &cfg(1, 1)),
            Err(FallbackReason::InvalidPlan)
        );
        // Donor range out of bounds.
        let p = plan(vec![seg((0, 100), (350, 450), SegmentMode::Copied)]);
        assert_eq!(
            trim_and_gate(&p, 0, 200, 400, &cfg(1, 1)),
            Err(FallbackReason::InvalidPlan)
        );
    }

    #[test]
    fn stale_generation_falls_back() {
        let mut c = cfg(1, 1);
        c.provider_generation = 3;
        let p = plan(vec![seg((0, 100), (0, 100), SegmentMode::Copied)]);
        assert_eq!(
            trim_and_gate(&p, 0, 200, 400, &c),
            Err(FallbackReason::GenerationStale)
        );
    }

    #[test]
    fn halos_counted_but_not_saved() {
        let p = plan(vec![
            seg((100, 116), (0, 16), SegmentMode::Recomputed),
            seg((116, 300), (16, 200), SegmentMode::Copied),
            seg((300, 316), (200, 216), SegmentMode::Recomputed),
        ]);
        let r = trim_and_gate(&p, 0, 400, 300, &cfg(16, 32)).unwrap();
        assert_eq!(r.copied_tokens, 184);
        assert_eq!(r.recomputed_halo_tokens, 32);
    }

    #[test]
    fn copy_overlaps_compute() {
        let c = cfg(16, 32);
        // Small copy fully hidden by compute.
        let extra = blended_extra_ms(10.0, 1000, 0, 1, &c);
        assert!((extra - c.plan_overhead_ms).abs() < 1e-9);
        // Huge copy dominates compute.
        let extra = blended_extra_ms(0.1, 1_000_000, 0, 1, &c);
        assert!(extra > 50.0);
    }
}
