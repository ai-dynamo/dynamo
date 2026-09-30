// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker signals shared by the ported routing policies.

use dynamo_kv_router::plugins::worker_selection::{WorkerCacheInput, WorkerSelectionContext};
use dynamo_kv_router::protocols::WorkerWithDpRank;

/// Device-resident prefix overlap in blocks. Hosts that report no per-tier matches supply only
/// the accounting estimate, which the default scorer also treats as device overlap.
pub(crate) fn device_overlap_blocks(cache: WorkerCacheInput<'_>) -> f64 {
    if cache.has_tier_matches() {
        cache.device_overlap_blocks()
    } else {
        cache.accounting_cache_estimate().0
    }
}

/// Prompt tokens a worker would still prefill after its device-resident prefix.
pub(crate) fn uncached_prompt_tokens(
    context: &WorkerSelectionContext<'_>,
    cache: WorkerCacheInput<'_>,
) -> usize {
    let cached = device_overlap_blocks(cache) * f64::from(context.block_size());
    context
        .prompt_tokens()
        .saturating_sub(cached.max(0.0) as usize)
}

/// Among `rows`, return the one that follows `rotation` in worker order. Tie-breaking by worker
/// identity keeps selection independent of the host's unspecified row order.
pub(crate) fn rotate_by_worker(
    rows: &mut [usize],
    worker: impl Fn(usize) -> WorkerWithDpRank,
    rotation: usize,
) -> Option<usize> {
    rows.sort_unstable_by_key(|&row| worker(row));
    (!rows.is_empty()).then(|| rows[rotation % rows.len()])
}

fn mix(mut value: u64) -> u64 {
    // SplitMix64 finalizer.
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Rendezvous (highest-random-weight) hash of `key` onto `worker`. Ranking workers by this weight
/// maps a key consistently: adding or removing a worker moves only the keys it wins or held.
pub(crate) fn rendezvous(key: u64, worker: WorkerWithDpRank, seed: u64) -> u64 {
    mix(key ^ mix(worker.worker_id ^ mix(u64::from(worker.dp_rank) ^ seed)))
}
