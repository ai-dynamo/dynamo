// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use ordered_float::OrderedFloat;

use super::config::RouterQueuePolicy;
use super::types::SchedulingRequest;
/// Pluggable scheduling policy that determines queue ordering.
/// Monomorphized for zero-cost inlining on the hot comparison path.
///
/// Higher key = higher priority (natural max-heap ordering).
pub trait SchedulingPolicy: Send + Sync + 'static {
    /// Priority key stored in each queue entry.
    type Key: Ord + Eq + Clone + Send + 'static;

    /// Compute priority key at enqueue time.
    fn enqueue_key(&self, arrival_offset: Duration, request: &SchedulingRequest) -> Self::Key;

    /// Recompute priority key during update(). Default: return old key unchanged.
    fn rekey(&self, _now: Duration, old_key: &Self::Key, _req: &SchedulingRequest) -> Self::Key {
        old_key.clone()
    }

    /// When true, queue rebuilds heap via rekey() on each update() call.
    /// When false (default), rekey path is compiled out entirely.
    const DYNAMIC: bool = false;
}

/// FCFS with priority bumps: key = priority_jump - arrival_offset.
/// Earlier arrival or higher priority_jump produces a higher key, scheduled first.
///
/// Optimizes for tail TTFT — no request waits longer than necessary,
/// since ordering is purely by (adjusted) arrival time.
pub struct FcfsPolicy;

impl SchedulingPolicy for FcfsPolicy {
    type Key = OrderedFloat<f64>;

    fn enqueue_key(&self, arrival_offset: Duration, request: &SchedulingRequest) -> Self::Key {
        OrderedFloat(request.priority_jump.max(0.0) - arrival_offset.as_secs_f64())
    }
}

/// Weighted Shortest Processing Time (Smith's rule):
/// key = (1 + priority_jump) / new_tokens, where new_tokens estimates the
/// actual prefill cost by subtracting the p75 KV cache overlap from ISL.
/// We use p75 rather than median because the downstream selector routes to
/// the best-overlap worker, so the realized overlap skews above the median.
///
/// TODO: the p75 heuristic is a rough proxy for post-routing overlap;
/// a tighter estimate could condition on worker load or use the selector's
/// own scoring function directly.
///
/// Optimizes for average TTFT — minimizes total weighted completion time
/// (Smith 1956). Short or high-priority requests are scheduled before
/// long low-priority ones, reducing mean latency across the batch.
pub struct WsptPolicy {
    pub block_size: usize,
}

impl SchedulingPolicy for WsptPolicy {
    type Key = OrderedFloat<f64>;

    fn enqueue_key(&self, _arrival_offset: Duration, request: &SchedulingRequest) -> Self::Key {
        let weight = 1.0 + request.priority_jump.max(0.0);
        let mut overlaps: Vec<u32> = request.overlaps.scores.values().copied().collect();
        let p75_overlap = if overlaps.is_empty() {
            0
        } else {
            let idx = (overlaps.len() - 1) * 3 / 4;
            *overlaps.select_nth_unstable(idx).1
        } as usize;
        let cached_tokens = p75_overlap * self.block_size;
        let new_tokens = request.isl_tokens.saturating_sub(cached_tokens).max(1);
        OrderedFloat(weight / new_tokens as f64)
    }
}

/// Runtime-dispatched scheduling policy selected via configuration.
/// Delegates to the concrete policy variant; the branch is fully predictable
/// since the variant is fixed at queue construction time.
pub enum RouterSchedulingPolicy {
    Fcfs(FcfsPolicy),
    Wspt(WsptPolicy),
}

impl RouterSchedulingPolicy {
    pub fn new(kind: RouterQueuePolicy, block_size: usize) -> Self {
        match kind {
            RouterQueuePolicy::Fcfs => Self::Fcfs(FcfsPolicy),
            RouterQueuePolicy::Wspt => Self::Wspt(WsptPolicy { block_size }),
        }
    }
}

impl SchedulingPolicy for RouterSchedulingPolicy {
    type Key = OrderedFloat<f64>;

    fn enqueue_key(&self, arrival_offset: Duration, request: &SchedulingRequest) -> Self::Key {
        match self {
            Self::Fcfs(p) => p.enqueue_key(arrival_offset, request),
            Self::Wspt(p) => p.enqueue_key(arrival_offset, request),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rustc_hash::FxHashMap;

    use super::*;
    use crate::protocols::{OverlapScores, WorkerWithDpRank};

    fn request_with(
        isl_tokens: usize,
        priority_jump: f64,
        overlaps: OverlapScores,
    ) -> SchedulingRequest {
        SchedulingRequest {
            maybe_request_id: None,
            token_seq: None,
            isl_tokens,
            overlaps,
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump,
            expected_output_tokens: None,
            allowed_worker_ids: None,
            resp_tx: None,
        }
    }

    fn overlaps_from(scores: &[(u64, u32)]) -> OverlapScores {
        let mut map = FxHashMap::default();
        for &(worker_id, score) in scores {
            map.insert(WorkerWithDpRank::new(worker_id, 0), score);
        }
        OverlapScores {
            scores: map,
            frequencies: vec![],
            tree_sizes: FxHashMap::default(),
        }
    }

    // ---- FCFS policy tests ----

    #[test]
    fn fcfs_earlier_arrival_scheduled_first() {
        let policy = FcfsPolicy;
        let req = request_with(512, 0.0, OverlapScores::default());
        let early = policy.enqueue_key(Duration::from_secs(1), &req);
        let late = policy.enqueue_key(Duration::from_secs(10), &req);
        assert!(early > late, "earlier arrival should have higher key");
    }

    #[test]
    fn fcfs_priority_jump_promotes() {
        let policy = FcfsPolicy;
        // Both arrive at the same wall-clock offset (10s), but one has priority.
        let normal = request_with(512, 0.0, OverlapScores::default());
        let boosted = request_with(512, 100.0, OverlapScores::default());
        let t = Duration::from_secs(10);
        let key_normal = policy.enqueue_key(t, &normal);
        let key_boosted = policy.enqueue_key(t, &boosted);
        assert!(
            key_boosted > key_normal,
            "priority_jump should produce a higher key"
        );
    }

    #[test]
    fn fcfs_priority_jump_beats_earlier_arrival() {
        let policy = FcfsPolicy;
        // Request A arrived at t=0 with no priority.
        // Request B arrived at t=5 with priority_jump=50s.
        // B should be scheduled first despite arriving later.
        let a = request_with(512, 0.0, OverlapScores::default());
        let b = request_with(512, 50.0, OverlapScores::default());
        let key_a = policy.enqueue_key(Duration::from_secs(0), &a);
        let key_b = policy.enqueue_key(Duration::from_secs(5), &b);
        assert!(key_b > key_a);
    }

    // ---- WSPT policy tests ----

    #[test]
    fn wspt_shorter_request_scheduled_first() {
        let policy = WsptPolicy { block_size: 16 };
        let short = request_with(100, 0.0, OverlapScores::default());
        let long = request_with(1000, 0.0, OverlapScores::default());
        let t = Duration::ZERO;
        assert!(
            policy.enqueue_key(t, &short) > policy.enqueue_key(t, &long),
            "shorter request should have higher key"
        );
    }

    #[test]
    fn wspt_overlap_reduces_effective_cost() {
        let policy = WsptPolicy { block_size: 16 };
        // Both 1024 ISL tokens, but one has 60 blocks cached (960 tokens).
        let no_cache = request_with(1024, 0.0, OverlapScores::default());
        let cached = request_with(1024, 0.0, overlaps_from(&[(0, 60)]));
        let t = Duration::ZERO;
        let key_no_cache = policy.enqueue_key(t, &no_cache);
        let key_cached = policy.enqueue_key(t, &cached);
        assert!(
            key_cached > key_no_cache,
            "request with overlap should have higher key (fewer new tokens)"
        );
    }

    #[test]
    fn wspt_priority_promotes() {
        let policy = WsptPolicy { block_size: 16 };
        let normal = request_with(512, 0.0, OverlapScores::default());
        let boosted = request_with(512, 5.0, OverlapScores::default());
        let t = Duration::ZERO;
        assert!(
            policy.enqueue_key(t, &boosted) > policy.enqueue_key(t, &normal),
            "priority_jump should increase key"
        );
    }

    #[test]
    fn wspt_uses_p75_overlap() {
        let policy = WsptPolicy { block_size: 16 };
        // 4 workers with overlaps [10, 20, 50, 60]. p75 = 50 (index 3 of 4).
        // new_tokens = 1024 - 50*16 = 224
        let req = request_with(
            1024,
            0.0,
            overlaps_from(&[(0, 10), (1, 20), (2, 50), (3, 60)]),
        );
        let key = policy.enqueue_key(Duration::ZERO, &req);
        let expected = OrderedFloat(1.0 / 224.0);
        assert_eq!(key, expected);
    }

    #[test]
    fn wspt_no_overlap_falls_back_to_isl() {
        let policy = WsptPolicy { block_size: 16 };
        let req = request_with(512, 0.0, OverlapScores::default());
        let key = policy.enqueue_key(Duration::ZERO, &req);
        let expected = OrderedFloat(1.0 / 512.0);
        assert_eq!(key, expected);
    }

    #[test]
    fn wspt_full_overlap_clamps_to_one() {
        let policy = WsptPolicy { block_size: 16 };
        // 512 tokens, 64 blocks cached = 1024 cached tokens > ISL → saturating_sub → 0 → max(1)
        let req = request_with(512, 0.0, overlaps_from(&[(0, 64)]));
        let key = policy.enqueue_key(Duration::ZERO, &req);
        let expected = OrderedFloat(1.0 / 1.0);
        assert_eq!(key, expected);
    }
}
