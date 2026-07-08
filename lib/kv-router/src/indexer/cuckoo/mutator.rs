// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustc_hash::FxHashSet;

use super::addressing::CkfAddressing;
use super::bucket::{CuckooBucketStore, PackedBucket};
use crate::protocols::{ExternalSequenceBlockHash, KvCacheEventError};

const SPLITMIX_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DirtyBucket {
    pub(super) bucket: usize,
    pub(super) value: PackedBucket,
}

pub(super) struct CuckooInsertionScratch {
    touched: Vec<(usize, PackedBucket)>,
    dirty_buckets: Vec<usize>,
}

impl CuckooInsertionScratch {
    pub(super) fn new(max_kicks: usize) -> Result<Self, KvCacheEventError> {
        let capacity = max_kicks
            .checked_add(1)
            .ok_or(KvCacheEventError::CapacityExhausted)?;
        let mut touched = Vec::new();
        let mut dirty_buckets = Vec::new();
        touched
            .try_reserve_exact(capacity)
            .map_err(|_| KvCacheEventError::CapacityExhausted)?;
        dirty_buckets
            .try_reserve_exact(capacity)
            .map_err(|_| KvCacheEventError::CapacityExhausted)?;
        Ok(Self {
            touched,
            dirty_buckets,
        })
    }

    fn clear(&mut self) {
        self.touched.clear();
        self.dirty_buckets.clear();
    }
}

pub(super) struct DcWriterState {
    pub(super) resident: FxHashSet<ExternalSequenceBlockHash>,
    pub(super) rng: u64,
    pub(super) scratch: CuckooInsertionScratch,
}

impl DcWriterState {
    pub(super) fn new(
        expected_blocks: usize,
        max_kicks: usize,
        rng: u64,
    ) -> Result<Self, KvCacheEventError> {
        let mut resident = FxHashSet::default();
        resident
            .try_reserve(expected_blocks)
            .map_err(|_| KvCacheEventError::CapacityExhausted)?;
        Ok(Self {
            resident,
            rng,
            scratch: CuckooInsertionScratch::new(max_kicks)?,
        })
    }
}

pub(super) struct CuckooMutator<'a, S> {
    store: &'a S,
    addressing: &'a CkfAddressing,
    max_kicks: usize,
}

impl<'a, S: CuckooBucketStore> CuckooMutator<'a, S> {
    pub(super) fn new(store: &'a S, addressing: &'a CkfAddressing, max_kicks: usize) -> Self {
        Self {
            store,
            addressing,
            max_kicks,
        }
    }

    pub(super) fn insert(
        &self,
        hash: ExternalSequenceBlockHash,
        rng: &mut u64,
        scratch: &mut CuckooInsertionScratch,
        mut on_dirty: impl FnMut(DirtyBucket),
    ) -> Result<(), KvCacheEventError> {
        debug_assert!(self.store.bucket_count().is_power_of_two());
        scratch.clear();
        let probe = self.addressing.prepare(hash.0);

        if self.insert_into_empty(probe.bucket_a, probe.fingerprint, scratch)
            || self.insert_into_empty(probe.bucket_b, probe.fingerprint, scratch)
        {
            self.emit_final_dirty(scratch, &mut on_dirty);
            return Ok(());
        }

        let mut bucket = if next_random(rng) & 1 == 0 {
            probe.bucket_a
        } else {
            probe.bucket_b
        };
        let mut fingerprint = probe.fingerprint;

        for _ in 0..self.max_kicks {
            let before = self.store.load_bucket(bucket);
            let slot = (next_random(rng) as usize) & 3;
            let evicted = before.slot(slot);
            self.write(bucket, before, before.with_slot(slot, fingerprint), scratch);
            fingerprint = evicted;
            bucket = self.addressing.alternate_bucket(bucket, fingerprint);

            if self.insert_into_empty(bucket, fingerprint, scratch) {
                self.emit_final_dirty(scratch, &mut on_dirty);
                return Ok(());
            }
        }

        self.rollback(scratch);
        Err(KvCacheEventError::CapacityExhausted)
    }

    pub(super) fn remove(
        &self,
        hash: ExternalSequenceBlockHash,
        mut on_dirty: impl FnMut(DirtyBucket),
    ) -> Result<(), KvCacheEventError> {
        let probe = self.addressing.prepare(hash.0);
        for bucket in [probe.bucket_a, probe.bucket_b] {
            let before = self.store.load_bucket(bucket);
            let Some(slot) = before.first(probe.fingerprint) else {
                continue;
            };
            let after = before.with_slot(slot, 0);
            self.store.store_bucket(bucket, after);
            on_dirty(DirtyBucket {
                bucket,
                value: after,
            });
            return Ok(());
        }

        Err(KvCacheEventError::IndexerInvariantViolation)
    }

    fn insert_into_empty(
        &self,
        bucket: usize,
        fingerprint: u16,
        scratch: &mut CuckooInsertionScratch,
    ) -> bool {
        let before = self.store.load_bucket(bucket);
        let Some(slot) = before.first_empty() else {
            return false;
        };
        self.write(bucket, before, before.with_slot(slot, fingerprint), scratch);
        true
    }

    fn write(
        &self,
        bucket: usize,
        before: PackedBucket,
        after: PackedBucket,
        scratch: &mut CuckooInsertionScratch,
    ) {
        scratch.touched.push((bucket, before));
        scratch.dirty_buckets.push(bucket);
        self.store.store_bucket(bucket, after);
    }

    fn rollback(&self, scratch: &mut CuckooInsertionScratch) {
        for &(bucket, before) in scratch.touched.iter().rev() {
            self.store.store_bucket(bucket, before);
        }
        scratch.clear();
    }

    fn emit_final_dirty(
        &self,
        scratch: &mut CuckooInsertionScratch,
        on_dirty: &mut impl FnMut(DirtyBucket),
    ) {
        scratch.dirty_buckets.sort_unstable();
        scratch.dirty_buckets.dedup();
        for &bucket in &scratch.dirty_buckets {
            on_dirty(DirtyBucket {
                bucket,
                value: self.store.load_bucket(bucket),
            });
        }
        scratch.clear();
    }
}

pub(super) fn lane_rng_seed(seed: u64, lane: usize) -> u64 {
    let mut state = seed
        ^ (lane as u64)
            .wrapping_add(1)
            .wrapping_mul(SPLITMIX_INCREMENT);
    next_random(&mut state)
}

fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_INCREMENT);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}
