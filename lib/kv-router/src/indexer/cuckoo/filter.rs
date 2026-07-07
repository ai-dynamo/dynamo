// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Approximate membership with deletion support, so a whole DC's cache fits in
//! a few MiB yet stays in sync under eviction churn.

use super::pages::BucketPages;

pub const SLOTS: usize = 4;
/// Separates the primary bucket hash from the fingerprint hash.
pub(super) const I1_SEED_TWEAK: u64 = 0xD1B5_4A32_D192_ED03;
const MAX_KICKS: usize = 500;
const TARGET_LOAD: f64 = 0.95;
/// Provision conservatively so churn and resize bursts stay insertable without
/// inflating filter size or snapshot bytes too much.
const SIZING_LOAD: f64 = 0.8;
/// Keep a shared default seed so both sides start from the same derivation
/// contract.
pub const DEFAULT_FILTER_SEED: u64 = 0x5DEE_CE66_D1B5_4A33;

#[inline]
pub(super) fn mix(mut x: u64, seed: u64) -> u64 {
    x ^= seed;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    x
}
#[inline]
pub(super) fn derive_fp(h: u64, seed: u64) -> u16 {
    let f = (mix(h, seed) >> 48) as u16;
    if f == 0 { 1 } else { f }
}
#[inline]
fn derive_i1(h: u64, seed: u64, mask: u64) -> usize {
    (mix(h, seed ^ I1_SEED_TWEAK) & mask) as usize
}
#[inline]
fn alt_bucket(i: usize, fp: u16, seed: u64, mask: u64) -> usize {
    let off = (mix(fp as u64, seed) & mask) | 1;
    ((i as u64) ^ off) as usize
}

/// Seeded cuckoo filter backed by page-granular copy-on-write fingerprints.
#[derive(Clone)]
pub struct CuckooFilter {
    pub(super) buckets: BucketPages,
    num_buckets: usize,
    pub(super) mask: u64,
    pub(super) seed: u64,
    rng: u64,
    pub(super) len: usize,
    /// Reused across kicked inserts so a displacement chain doesn't pay a
    /// fresh heap allocation every time both direct buckets are full.
    kick_scratch: Vec<(usize, [u16; SLOTS])>,
}

impl CuckooFilter {
    pub fn with_capacity_seeded(n: usize, seed: u64) -> Self {
        let need = ((n as f64 / (SLOTS as f64 * TARGET_LOAD)).ceil() as usize).max(1);
        Self::with_num_buckets(need.next_power_of_two(), seed)
    }

    /// Provision for `expected` elements with headroom for churn and resize
    /// bursts.
    pub fn provisioned(expected: usize, seed: u64) -> Self {
        let need = ((expected.max(1) as f64 / (SLOTS as f64 * SIZING_LOAD)).ceil() as usize).max(1);
        Self::with_num_buckets(need.next_power_of_two(), seed)
    }

    pub(super) fn with_num_buckets(nb: usize, seed: u64) -> Self {
        let nb = nb.max(2);
        Self {
            buckets: BucketPages::zeroed(nb * SLOTS),
            num_buckets: nb,
            mask: (nb as u64) - 1,
            seed,
            rng: seed | 1,
            len: 0,
            kick_scratch: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn bytes(&self) -> usize {
        self.buckets.len() * std::mem::size_of::<u16>()
    }
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }

    #[inline]
    fn slot(&self, b: usize, s: usize) -> usize {
        b * SLOTS + s
    }
    fn put_at(&mut self, bucket: usize, fingerprint: u16) -> bool {
        for slot in 0..SLOTS {
            let index = self.slot(bucket, slot);
            if self.buckets.get(index) == 0 {
                self.buckets.set(index, fingerprint);
                return true;
            }
        }
        false
    }

    fn bucket(&self, b: usize) -> [u16; SLOTS] {
        let lo = b * SLOTS;
        self.buckets.read_array(lo)
    }

    fn put_at_tracked(
        &mut self,
        b: usize,
        fp: u16,
        originals: &mut Vec<(usize, [u16; SLOTS])>,
    ) -> bool {
        for s in 0..SLOTS {
            let i = self.slot(b, s);
            if self.buckets.get(i) == 0 {
                originals.push((b, self.bucket(b)));
                self.buckets.set(i, fp);
                return true;
            }
        }
        false
    }
    pub(super) fn has_at(&self, b: usize, fp: u16) -> bool {
        self.buckets.contains_in_array::<SLOTS>(b * SLOTS, fp)
    }
    #[inline]
    fn next_rng(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    /// Return false only when the filter can no longer make room without a
    /// larger rebuild.
    pub fn insert(&mut self, h: u64) -> bool {
        self.insert_with(h, |_| {})
    }

    /// Report every mutated bucket to `on_dirty` so delta publication can stay
    /// sparse without paying a per-insert allocation on kicked inserts; a
    /// failed kick chain rolls back and reports nothing.
    pub(super) fn insert_with(&mut self, h: u64, mut on_dirty: impl FnMut(usize)) -> bool {
        let fp = derive_fp(h, self.seed);
        let i1 = derive_i1(h, self.seed, self.mask);
        if self.put_at(i1, fp) {
            self.len += 1;
            on_dirty(i1);
            return true;
        }
        let i2 = alt_bucket(i1, fp, self.seed, self.mask);
        if self.put_at(i2, fp) {
            self.len += 1;
            on_dirty(i2);
            return true;
        }
        // Track every pre-mutation bucket so rollback can restore the earliest
        // state even if the kick chain revisits a bucket. Reuse the filter's
        // scratch buffer instead of allocating fresh on every kicked insert;
        // it is always empty on entry and restored empty on every exit path.
        let mut originals = std::mem::take(&mut self.kick_scratch);
        let mut i = if self.next_rng() & 1 == 0 { i1 } else { i2 };
        let mut f = fp;
        for _ in 0..MAX_KICKS {
            let s = (self.next_rng() as usize) % SLOTS;
            let idx = self.slot(i, s);
            originals.push((i, self.bucket(i)));
            let displaced = self.buckets.get(idx);
            self.buckets.set(idx, f);
            f = displaced;
            i = alt_bucket(i, f, self.seed, self.mask);
            if self.put_at_tracked(i, f, &mut originals) {
                self.len += 1;
                for &(bucket, _) in &originals {
                    on_dirty(bucket);
                }
                originals.clear();
                self.kick_scratch = originals;
                return true;
            }
        }
        // Reverse rollback preserves the earliest pre-chain snapshot for each
        // bucket.
        for (b, slots) in originals.drain(..).rev() {
            let lo = b * SLOTS;
            self.buckets.write_array(lo, &slots);
        }
        self.kick_scratch = originals;
        false
    }

    pub fn contains(&self, h: u64) -> bool {
        let fp = derive_fp(h, self.seed);
        let i1 = derive_i1(h, self.seed, self.mask);
        if self.has_at(i1, fp) {
            return true;
        }
        let i2 = alt_bucket(i1, fp, self.seed, self.mask);
        self.has_at(i2, fp)
    }

    pub fn remove(&mut self, h: u64) -> bool {
        self.remove_tracked(h).is_some()
    }

    pub(super) fn remove_tracked(&mut self, h: u64) -> Option<usize> {
        let fp = derive_fp(h, self.seed);
        let i1 = derive_i1(h, self.seed, self.mask);
        let i2 = alt_bucket(i1, fp, self.seed, self.mask);
        for b in [i1, i2] {
            for s in 0..SLOTS {
                let idx = self.slot(b, s);
                if self.buckets.get(idx) == fp {
                    self.buckets.set(idx, 0);
                    self.len -= 1;
                    return Some(b);
                }
            }
        }
        None
    }

    /// Materialize the raw bucket array. Intended for diagnostics/tests; hot
    /// paths use page-aware bucket accessors.
    pub fn to_raw_buckets(&self) -> Vec<u16> {
        self.buckets.to_vec()
    }

    pub fn bucket_slots(&self, bucket: usize) -> [u16; SLOTS] {
        self.bucket(bucket)
    }

    pub(super) fn set_bucket_slots(&mut self, bucket: usize, slots: &[u16; SLOTS]) {
        self.buckets.write_array(bucket * SLOTS, slots);
    }
}
