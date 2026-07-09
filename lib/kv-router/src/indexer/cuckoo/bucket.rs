// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use super::addressing::CkfProbe;
use super::{CkfBuildError, DC_COUNT};

const SLOT_COUNT: usize = 4;
const REPEATED_ONE: u64 = 0x0001_0001_0001_0001;
const HIGH_BITS: u64 = 0x8000_8000_8000_8000;
#[cfg(target_arch = "x86_64")]
const CACHE_LINE_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct PackedBucket(pub(super) u64);

impl PackedBucket {
    #[inline]
    pub(super) fn contains(self, fingerprint: u16) -> bool {
        debug_assert_ne!(fingerprint, 0);
        let repeated = u64::from(fingerprint) * REPEATED_ONE;
        let different = self.0 ^ repeated;
        different.wrapping_sub(REPEATED_ONE) & !different & HIGH_BITS != 0
    }

    #[inline]
    pub(super) fn first_empty(self) -> Option<usize> {
        self.first(0)
    }

    #[inline]
    pub(super) fn first(self, fingerprint: u16) -> Option<usize> {
        if self.0 as u16 == fingerprint {
            Some(0)
        } else if (self.0 >> 16) as u16 == fingerprint {
            Some(1)
        } else if (self.0 >> 32) as u16 == fingerprint {
            Some(2)
        } else if (self.0 >> 48) as u16 == fingerprint {
            Some(3)
        } else {
            None
        }
    }

    #[inline]
    pub(super) fn slot(self, slot: usize) -> u16 {
        debug_assert!(slot < SLOT_COUNT);
        (self.0 >> (slot * u16::BITS as usize)) as u16
    }

    #[inline]
    pub(super) fn with_slot(self, slot: usize, fingerprint: u16) -> Self {
        debug_assert!(slot < SLOT_COUNT);
        let shift = slot * u16::BITS as usize;
        let mask = u64::from(u16::MAX) << shift;
        Self((self.0 & !mask) | (u64::from(fingerprint) << shift))
    }

    #[cfg(test)]
    pub(super) fn scalar_contains(self, fingerprint: u16) -> bool {
        (0..SLOT_COUNT).any(|slot| self.slot(slot) == fingerprint)
    }
}

pub(super) trait CuckooBucketStore {
    fn load_bucket(&self, bucket: usize) -> PackedBucket;
    fn store_bucket(&self, bucket: usize, value: PackedBucket);
    fn bucket_count(&self) -> usize;
}

#[derive(Debug)]
pub(super) struct OwnedPackedCkfLane {
    buckets: Box<[Cell<u64>]>,
}

impl OwnedPackedCkfLane {
    pub(super) fn new(bucket_count: usize) -> Result<Self, CkfBuildError> {
        let allocation_bytes = bucket_count
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(CkfBuildError::CapacityOverflow)?;
        if allocation_bytes > isize::MAX as usize {
            return Err(CkfBuildError::CapacityOverflow);
        }

        let mut buckets = Vec::new();
        buckets
            .try_reserve_exact(bucket_count)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        buckets.extend((0..bucket_count).map(|_| Cell::new(0)));
        Ok(Self {
            buckets: buckets.into_boxed_slice(),
        })
    }

    pub(super) fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.set(0);
        }
    }

    pub(super) fn byte_len(&self) -> usize {
        std::mem::size_of_val(self.buckets.as_ref())
    }
}

impl CuckooBucketStore for OwnedPackedCkfLane {
    #[inline]
    fn load_bucket(&self, bucket: usize) -> PackedBucket {
        PackedBucket(self.buckets[bucket].get())
    }

    #[inline]
    fn store_bucket(&self, bucket: usize, value: PackedBucket) {
        self.buckets[bucket].set(value.0);
    }

    fn bucket_count(&self) -> usize {
        self.buckets.len()
    }
}

#[derive(Debug)]
pub(super) struct TransposedCkfTable<const D: usize> {
    bucket_count: usize,
    lanes: Box<[AtomicU64]>,
}

impl<const D: usize> TransposedCkfTable<D> {
    pub(super) fn new(bucket_count: usize) -> Result<Self, CkfBuildError> {
        debug_assert!((1..=DC_COUNT).contains(&D));
        let lane_count = bucket_count
            .checked_mul(D)
            .ok_or(CkfBuildError::CapacityOverflow)?;
        let allocation_bytes = lane_count
            .checked_mul(std::mem::size_of::<AtomicU64>())
            .ok_or(CkfBuildError::CapacityOverflow)?;
        if allocation_bytes > isize::MAX as usize {
            return Err(CkfBuildError::CapacityOverflow);
        }

        let mut lanes = Vec::new();
        lanes
            .try_reserve_exact(lane_count)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        lanes.extend((0..lane_count).map(|_| AtomicU64::new(0)));

        Ok(Self {
            bucket_count,
            lanes: lanes.into_boxed_slice(),
        })
    }

    #[cfg(test)]
    pub(super) fn lane(&self, lane: usize) -> TransposedDcView<'_, D> {
        debug_assert!(lane < D);
        TransposedDcView { table: self, lane }
    }

    pub(super) fn bucket_count(&self) -> usize {
        self.bucket_count
    }

    pub(super) fn store_image(&self, bucket: usize, lane: usize, value: PackedBucket) {
        debug_assert!(bucket < self.bucket_count);
        debug_assert!(lane < D);
        self.store(bucket, lane, value);
    }

    pub(super) fn load_image(&self, bucket: usize, lane: usize) -> PackedBucket {
        debug_assert!(bucket < self.bucket_count);
        debug_assert!(lane < D);
        self.load(bucket, lane)
    }

    pub(super) fn clear_lane(&self, lane: usize) {
        debug_assert!(lane < D);
        for bucket in 0..self.bucket_count {
            self.store(bucket, lane, PackedBucket::default());
        }
    }

    #[inline]
    pub(super) fn probe(&self, probe: CkfProbe) -> u16 {
        let mut present = 0u16;
        for lane in 0..D {
            let first = self.load(probe.bucket_a, lane);
            let second = self.load(probe.bucket_b, lane);
            let found = first.contains(probe.fingerprint) || second.contains(probe.fingerprint);
            present |= u16::from(found) << lane;
        }
        present
    }

    pub(super) fn prefetch_probe(&self, probe: CkfProbe) {
        self.prefetch_row(probe.bucket_a);
        if probe.bucket_b != probe.bucket_a {
            self.prefetch_row(probe.bucket_b);
        }
    }

    #[inline]
    fn load(&self, bucket: usize, lane: usize) -> PackedBucket {
        let index = bucket * D + lane;
        PackedBucket(self.lanes[index].load(Ordering::Relaxed))
    }

    #[inline]
    fn store(&self, bucket: usize, lane: usize, value: PackedBucket) {
        let index = bucket * D + lane;
        self.lanes[index].store(value.0, Ordering::Relaxed);
    }

    fn prefetch_row(&self, bucket: usize) {
        let index = bucket * D;
        let row = self.lanes.as_ptr().wrapping_add(index).cast::<u8>();
        prefetch_span(row, D * std::mem::size_of::<AtomicU64>());
    }
}

#[cfg(test)]
pub(super) struct TransposedDcView<'a, const D: usize> {
    table: &'a TransposedCkfTable<D>,
    lane: usize,
}

#[cfg(test)]
impl<const D: usize> CuckooBucketStore for TransposedDcView<'_, D> {
    #[inline]
    fn load_bucket(&self, bucket: usize) -> PackedBucket {
        self.table.load(bucket, self.lane)
    }

    #[inline]
    fn store_bucket(&self, bucket: usize, value: PackedBucket) {
        self.table.store(bucket, self.lane, value);
    }

    fn bucket_count(&self) -> usize {
        self.table.bucket_count
    }
}

#[cfg(target_arch = "x86_64")]
fn prefetch_span(start: *const u8, len: usize) {
    use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

    if len == 0 {
        return;
    }

    let address = start as usize;
    let mut offset = 0usize;
    loop {
        // SAFETY: Every prefetched address is within the allocated row. Prefetch is a hint and
        // does not dereference the pointer architecturally.
        unsafe { _mm_prefetch(start.add(offset).cast::<i8>(), _MM_HINT_T0) };
        let next = if offset == 0 {
            CACHE_LINE_BYTES - (address & (CACHE_LINE_BYTES - 1))
        } else {
            offset + CACHE_LINE_BYTES
        };
        if next >= len {
            break;
        }
        offset = next;
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn prefetch_span(_start: *const u8, _len: usize) {}
