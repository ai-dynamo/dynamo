// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Track relay-side mutations so snapshot and delta publication stay bounded by
//! the buckets that actually changed.

use super::filter::{CuckooFilter, SLOTS};
use super::pages::BucketPages;
use super::snapshot::{SnapshotState, build_delta_for_buckets};

/// One delta entry carries the bucket index plus the four slot fingerprints.
pub(super) const DELTA_ENTRY_BYTES: usize = 4 + SLOTS * 2;

/// Cap deltas so a single missed update still has a recovery path through the
/// next full snapshot. Public so transport endpoints can check their message
/// caps against it at startup: a limit below this silently degrades into a
/// full-snapshot resync on every large delta.
pub const MAX_DELTA_BYTES: usize = 32 * 1024 * 1024;

/// Switch to a full snapshot when a delta would no longer be meaningfully
/// smaller or would approach the transport ceiling.
pub(super) fn churn_wants_full(dirty_buckets: usize, num_buckets: usize) -> bool {
    let delta_bytes = dirty_buckets.saturating_mul(DELTA_ENTRY_BYTES);
    let full_bytes = num_buckets * SLOTS * 2;
    delta_bytes.saturating_mul(2) >= full_bytes || delta_bytes >= MAX_DELTA_BYTES
}

/// One producer publish step; see [`SnapshotProducer::publish`].
pub enum Publish {
    /// First publish or shape change requires a full snapshot.
    Full(SnapshotState),
    Delta(Vec<u8>),
    Unchanged,
}

/// Track dirty buckets in a bitmap so hot-path marking stays allocation-free
/// and draining already yields sorted bucket indices.
#[derive(Default)]
struct DirtyBuckets {
    words: Vec<u64>,
    len: usize,
}

impl DirtyBuckets {
    fn for_buckets(num_buckets: usize) -> Self {
        Self {
            words: vec![0; num_buckets.div_ceil(64)],
            len: 0,
        }
    }

    fn mark(&mut self, bucket: usize) {
        let word = &mut self.words[bucket >> 6];
        let bit = 1u64 << (bucket & 63);
        self.len += usize::from(*word & bit == 0);
        *word |= bit;
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn len(&self) -> usize {
        self.len
    }

    fn clear(&mut self) {
        self.words.fill(0);
        self.len = 0;
    }

    /// Drain to a sorted bucket list and clear the bitmap.
    fn take_sorted(&mut self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.len);
        for (wi, word) in self.words.iter_mut().enumerate() {
            let mut w = std::mem::take(word);
            while w != 0 {
                out.push((wi << 6) | w.trailing_zeros() as usize);
                w &= w - 1;
            }
        }
        self.len = 0;
        out
    }
}

/// Keep one DC's filter in sync with the authoritative resident set while
/// producing full snapshots and incremental deltas.
pub struct SnapshotProducer {
    dc_worker_id: u64,
    filter: CuckooFilter,
    epoch: u64,
    /// Base state for the next delta. Shares pages with `filter.buckets` via
    /// copy-on-write, so taking a baseline is an `Arc`-bump, not a copy of the
    /// whole filter, and only pages that actually diverge get duplicated.
    last_shipped: BucketPages,
    /// Buckets touched since the last successful publish.
    dirty: DirtyBuckets,
}

impl SnapshotProducer {
    pub fn new(dc_worker_id: u64, expected: usize, seed: u64) -> Self {
        Self::new_with_epoch(dc_worker_id, expected, seed, 0)
    }

    /// Resume an existing epoch chain so a replacement producer does not look
    /// like a replay reset to consumers.
    pub fn new_with_epoch(dc_worker_id: u64, expected: usize, seed: u64, epoch: u64) -> Self {
        let filter = CuckooFilter::provisioned(expected, seed);
        let dirty = DirtyBuckets::for_buckets(filter.num_buckets());
        Self {
            dc_worker_id,
            filter,
            epoch,
            last_shipped: BucketPages::zeroed(0),
            dirty,
        }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub fn seed(&self) -> u64 {
        self.filter.seed()
    }
    pub fn len(&self) -> usize {
        self.filter.len()
    }
    pub fn num_buckets(&self) -> usize {
        self.filter.num_buckets()
    }
    pub fn is_empty(&self) -> bool {
        self.filter.is_empty()
    }

    /// Return false when the filter needs a rebuild from the authoritative set.
    pub fn insert(&mut self, h: u64) -> bool {
        // Split borrow: the filter reports mutated buckets straight into the
        // dirty bitmap, so kicked inserts stay allocation-free.
        let Self { filter, dirty, .. } = self;
        filter.insert_with(h, |bucket| dirty.mark(bucket))
    }
    pub fn remove(&mut self, h: u64) -> bool {
        if let Some(bucket) = self.filter.remove_tracked(h) {
            self.dirty.mark(bucket);
            return true;
        }
        false
    }

    /// Rebuild from the authoritative resident hashes so the next publish can
    /// re-baseline from a fresh filter.
    pub fn rebuild<I: IntoIterator<Item = u64>>(&mut self, resident: I, expected: usize) {
        let seed = self.filter.seed();
        let mut f = CuckooFilter::provisioned(expected, seed);
        for h in resident {
            f.insert(h);
        }
        self.dirty = DirtyBuckets::for_buckets(f.num_buckets());
        self.filter = f;
        self.last_shipped = BucketPages::zeroed(0);
    }

    fn snapshot_state(&self) -> SnapshotState {
        SnapshotState {
            buckets: self.filter.buckets.clone(),
            num_buckets: self.filter.num_buckets(),
            seed: self.filter.seed(),
            dc_worker_id: self.dc_worker_id,
            epoch: self.epoch,
        }
    }

    /// Capture a point-in-time snapshot and advance the epoch so chunks can be
    /// serialized without holding the caller's lock.
    pub fn full_snapshot(&mut self) -> SnapshotState {
        self.epoch += 1;
        // Cheap `Arc`-bump clone, not a flatten: pages diverge from this
        // baseline lazily, only where a later insert/remove actually touches
        // them (see `BucketPages::set`).
        self.last_shipped = self.filter.buckets.clone();
        self.dirty.clear();
        self.snapshot_state()
    }

    /// Capture the current filter without advancing state so a fresh subscriber
    /// can bootstrap before it starts consuming deltas.
    pub fn current_snapshot(&self) -> SnapshotState {
        self.snapshot_state()
    }

    /// Prefer a full snapshot when there is no stable delta base, otherwise
    /// ship the smallest useful update or nothing at all.
    pub fn publish(&mut self) -> Publish {
        if self.last_shipped.len() != self.filter.num_buckets() * SLOTS {
            return Publish::Full(self.full_snapshot());
        }
        if churn_wants_full(self.dirty.len(), self.filter.num_buckets()) {
            return Publish::Full(self.full_snapshot());
        }
        match self.delta() {
            Some(delta) => Publish::Delta(delta),
            None => Publish::Unchanged,
        }
    }

    /// Serialize only the buckets that actually changed since the last ship.
    fn delta(&mut self) -> Option<Vec<u8>> {
        if self.last_shipped.len() != self.filter.num_buckets() * SLOTS {
            return None;
        }
        if self.dirty.is_empty() {
            return None;
        }
        let dirty = self.dirty.take_sorted();
        let delta = build_delta_for_buckets(
            &self.last_shipped,
            &self.filter,
            self.dc_worker_id,
            self.epoch,
            self.epoch + 1,
            dirty.iter().copied(),
        );
        for b in dirty {
            let lo = b * SLOTS;
            self.last_shipped
                .write_array(lo, &self.filter.bucket_slots(b));
        }
        let delta = delta?;
        self.epoch += 1;
        Some(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::cuckoo::DEFAULT_FILTER_SEED;

    #[test]
    fn full_snapshot_establishes_delta_baseline() {
        let mut producer = SnapshotProducer::new(3, 64, DEFAULT_FILTER_SEED);
        assert!(producer.insert(11));
        let first_epoch = producer.full_snapshot().epoch();
        assert!(matches!(producer.publish(), Publish::Unchanged));
        assert!(producer.insert(22));
        let Publish::Delta(delta) = producer.publish() else {
            panic!("expected delta")
        };
        assert!(!delta.is_empty());
        assert_eq!(producer.epoch(), first_epoch + 1);
    }

    #[test]
    fn high_dirty_bucket_churn_falls_back_to_full() {
        let mut producer = SnapshotProducer::new(3, 64, DEFAULT_FILTER_SEED);
        producer.full_snapshot();
        for bucket in 0..producer.filter.num_buckets().div_ceil(2) {
            producer.dirty.mark(bucket);
        }
        assert!(matches!(producer.publish(), Publish::Full(_)));
    }
}
