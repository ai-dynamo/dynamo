// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::AtomicU64;

/// Monotonic counters for the affinity path, shared with the host so it can
/// export them. Every field is cumulative since process start; a host that
/// exports Prometheus counters reports the delta since its last scrape.
///
/// These make the degraded path visible: `failures` and `timeouts` separate
/// why affinity was lost, and `skipped` counts requests routed without
/// consulting storage at all while the breaker was open.
#[derive(Debug, Default)]
pub struct AffinityStats {
    /// Lookups that returned a live binding.
    pub hits: AtomicU64,
    /// Lookups that found no live binding, so the session was placed fresh.
    pub misses: AtomicU64,
    /// Conditional writes that lost to another writer, after the retry bound.
    pub conflicts: AtomicU64,
    /// Storage errors. The request still routed on the best candidate.
    pub failures: AtomicU64,
    /// Requests whose affinity work ran out of its deadline.
    pub timeouts: AtomicU64,
    /// Requests routed without touching storage while the breaker was open.
    pub skipped: AtomicU64,
    /// Records that no reader could decode and that were replaced.
    pub repairs: AtomicU64,
}

/// Counters for a read-through cache in front of a remote store.
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Reads answered from memory, with no call to the backing store.
    pub hits: AtomicU64,
    /// Reads that had to call the backing store.
    pub misses: AtomicU64,
    /// Entries dropped because a write conflicted or failed, or because the
    /// cache was full.
    pub invalidations: AtomicU64,
}
