// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw atomic counters and gauges for a single block pool type.
//!
//! All increment/decrement methods use `Ordering::Relaxed` for zero overhead on the hot path.
//! The [`MetricsAggregator`] reads these atomics at scrape time and builds Prometheus protos.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Raw atomic metrics for a single block pool (one per `BlockManager<T>`).
///
/// Counters are monotonically increasing `AtomicU64`.
/// Gauges are bidirectional `AtomicI64`.
pub struct BlockPoolMetrics {
    type_label: String,

    // Counters (monotonic)
    allocations: AtomicU64,
    allocations_from_reset: AtomicU64,
    evictions: AtomicU64,
    registrations: AtomicU64,
    duplicate_blocks: AtomicU64,
    registration_dedup: AtomicU64,
    stagings: AtomicU64,
    match_hashes_requested: AtomicU64,
    match_blocks_returned: AtomicU64,
    scan_hashes_requested: AtomicU64,
    scan_blocks_returned: AtomicU64,

    // Gauges (bidirectional)
    inflight_mutable: AtomicI64,
    inflight_immutable: AtomicI64,
    reset_pool_size: AtomicI64,
    inactive_pool_size: AtomicI64,
}

impl BlockPoolMetrics {
    /// Create a new `BlockPoolMetrics` with the given type label (e.g. `"G1"`).
    pub fn new(type_label: String) -> Self {
        Self {
            type_label,
            allocations: AtomicU64::new(0),
            allocations_from_reset: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            registrations: AtomicU64::new(0),
            duplicate_blocks: AtomicU64::new(0),
            registration_dedup: AtomicU64::new(0),
            stagings: AtomicU64::new(0),
            match_hashes_requested: AtomicU64::new(0),
            match_blocks_returned: AtomicU64::new(0),
            scan_hashes_requested: AtomicU64::new(0),
            scan_blocks_returned: AtomicU64::new(0),
            inflight_mutable: AtomicI64::new(0),
            inflight_immutable: AtomicI64::new(0),
            reset_pool_size: AtomicI64::new(0),
            inactive_pool_size: AtomicI64::new(0),
        }
    }

    /// The pool type label (e.g. `"G1"`, `"G2"`).
    #[inline(always)]
    pub fn type_label(&self) -> &str {
        &self.type_label
    }

    // ---- Counter increments ----

    #[inline(always)]
    pub fn inc_allocations(&self, n: u64) {
        self.allocations.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_allocations_from_reset(&self, n: u64) {
        self.allocations_from_reset.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_evictions(&self, n: u64) {
        self.evictions.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_registrations(&self) {
        self.registrations.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_duplicate_blocks(&self) {
        self.duplicate_blocks.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_registration_dedup(&self) {
        self.registration_dedup.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_stagings(&self) {
        self.stagings.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_match_hashes_requested(&self, n: u64) {
        self.match_hashes_requested.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_match_blocks_returned(&self, n: u64) {
        self.match_blocks_returned.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_scan_hashes_requested(&self, n: u64) {
        self.scan_hashes_requested.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_scan_blocks_returned(&self, n: u64) {
        self.scan_blocks_returned.fetch_add(n, Ordering::Relaxed);
    }

    // ---- Gauge operations ----

    #[inline(always)]
    pub fn inc_inflight_mutable(&self) {
        self.inflight_mutable.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dec_inflight_mutable(&self) {
        self.inflight_mutable.fetch_sub(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_inflight_immutable(&self) {
        self.inflight_immutable.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dec_inflight_immutable(&self) {
        self.inflight_immutable.fetch_sub(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn set_reset_pool_size(&self, size: i64) {
        self.reset_pool_size.store(size, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_reset_pool_size(&self) {
        self.reset_pool_size.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dec_reset_pool_size(&self) {
        self.reset_pool_size.fetch_sub(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn set_inactive_pool_size(&self, size: i64) {
        self.inactive_pool_size.store(size, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_inactive_pool_size(&self) {
        self.inactive_pool_size.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dec_inactive_pool_size(&self) {
        self.inactive_pool_size.fetch_sub(1, Ordering::Relaxed);
    }

    // ---- Snapshot for stats collector ----

    /// Take a point-in-time snapshot of all metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            allocations: self.allocations.load(Ordering::Relaxed),
            allocations_from_reset: self.allocations_from_reset.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            registrations: self.registrations.load(Ordering::Relaxed),
            duplicate_blocks: self.duplicate_blocks.load(Ordering::Relaxed),
            registration_dedup: self.registration_dedup.load(Ordering::Relaxed),
            stagings: self.stagings.load(Ordering::Relaxed),
            match_hashes_requested: self.match_hashes_requested.load(Ordering::Relaxed),
            match_blocks_returned: self.match_blocks_returned.load(Ordering::Relaxed),
            scan_hashes_requested: self.scan_hashes_requested.load(Ordering::Relaxed),
            scan_blocks_returned: self.scan_blocks_returned.load(Ordering::Relaxed),
            inflight_mutable: self.inflight_mutable.load(Ordering::Relaxed),
            inflight_immutable: self.inflight_immutable.load(Ordering::Relaxed),
            reset_pool_size: self.reset_pool_size.load(Ordering::Relaxed),
            inactive_pool_size: self.inactive_pool_size.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of all atomic metrics, used by the stats collector and prometheus collector.
#[derive(Debug, Clone, Copy)]
pub struct MetricsSnapshot {
    pub allocations: u64,
    pub allocations_from_reset: u64,
    pub evictions: u64,
    pub registrations: u64,
    pub duplicate_blocks: u64,
    pub registration_dedup: u64,
    pub stagings: u64,
    pub match_hashes_requested: u64,
    pub match_blocks_returned: u64,
    pub scan_hashes_requested: u64,
    pub scan_blocks_returned: u64,
    pub inflight_mutable: i64,
    pub inflight_immutable: i64,
    pub reset_pool_size: i64,
    pub inactive_pool_size: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_increments() {
        let m = BlockPoolMetrics::new("G1".to_string());

        m.inc_allocations(5);
        m.inc_allocations(3);
        m.inc_evictions(2);
        m.inc_registrations();
        m.inc_duplicate_blocks();
        m.inc_registration_dedup();
        m.inc_stagings();

        let snap = m.snapshot();
        assert_eq!(snap.allocations, 8);
        assert_eq!(snap.evictions, 2);
        assert_eq!(snap.registrations, 1);
        assert_eq!(snap.duplicate_blocks, 1);
        assert_eq!(snap.registration_dedup, 1);
        assert_eq!(snap.stagings, 1);
    }

    #[test]
    fn test_gauge_bidirectional() {
        let m = BlockPoolMetrics::new("G2".to_string());

        m.inc_inflight_mutable();
        m.inc_inflight_mutable();
        m.dec_inflight_mutable();

        m.inc_inflight_immutable();
        m.inc_inflight_immutable();
        m.inc_inflight_immutable();
        m.dec_inflight_immutable();

        let snap = m.snapshot();
        assert_eq!(snap.inflight_mutable, 1);
        assert_eq!(snap.inflight_immutable, 2);
    }

    #[test]
    fn test_pool_size_gauges() {
        let m = BlockPoolMetrics::new("G1".to_string());

        m.set_reset_pool_size(100);
        m.set_inactive_pool_size(50);

        let snap = m.snapshot();
        assert_eq!(snap.reset_pool_size, 100);
        assert_eq!(snap.inactive_pool_size, 50);

        m.set_reset_pool_size(80);
        let snap = m.snapshot();
        assert_eq!(snap.reset_pool_size, 80);

        // Test inc/dec for reset pool size
        m.inc_reset_pool_size();
        m.inc_reset_pool_size();
        m.dec_reset_pool_size();
        let snap = m.snapshot();
        assert_eq!(snap.reset_pool_size, 81);

        // Test inc/dec for inactive pool size
        m.inc_inactive_pool_size();
        m.inc_inactive_pool_size();
        m.inc_inactive_pool_size();
        m.dec_inactive_pool_size();
        let snap = m.snapshot();
        assert_eq!(snap.inactive_pool_size, 52);
    }

    #[test]
    fn test_type_label() {
        let m = BlockPoolMetrics::new("MyPool".to_string());
        assert_eq!(m.type_label(), "MyPool");
    }

    #[test]
    fn test_match_scan_counters() {
        let m = BlockPoolMetrics::new("G1".to_string());

        m.inc_match_hashes_requested(10);
        m.inc_match_blocks_returned(7);
        m.inc_scan_hashes_requested(20);
        m.inc_scan_blocks_returned(15);

        let snap = m.snapshot();
        assert_eq!(snap.match_hashes_requested, 10);
        assert_eq!(snap.match_blocks_returned, 7);
        assert_eq!(snap.scan_hashes_requested, 20);
        assert_eq!(snap.scan_blocks_returned, 15);
    }
}
