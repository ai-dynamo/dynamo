// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::AtomicU64;

/// Monotonic counters for the affinity path, shared with the host so it can
/// export them. Every field is cumulative since process start; a host that
/// exports Prometheus counters reports the delta since its last scrape.
///
/// These make the degraded path visible: `failures` and `timeouts` separate
/// why affinity was lost, and `skipped` counts requests routed without
/// consulting storage at all while the breaker was open. The write counters
/// add up to the writes sent to the store, by why each was needed.
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
    /// Writes for a session with no live pin: a new or expired session.
    pub creates: AtomicU64,
    /// Writes that moved a pinned session to another pool.
    pub moves: AtomicU64,
    /// Writes that only pushed back the expiry of a pin past half its idle
    /// window. A pin with more of its window left is not rewritten.
    pub refreshes: AtomicU64,
    /// Writes replacing a record that no reader could decode.
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
