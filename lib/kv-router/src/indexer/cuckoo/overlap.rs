// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Answer the routing question — how deep a contiguous prefix of the request
//! is cached in each DC — in logarithmic probes instead of a linear scan.

use super::filter::{CuckooFilter, I1_SEED_TWEAK, derive_fp, mix};

/// Precompute per-block probes once so large DC fan-outs pay the hash cost only
/// once per request.
pub struct Probe {
    fp: u16,
    m_index: u64,
    alt_base: u64,
}

impl Probe {
    /// Derive a probe from one sequence-hash so the search can reuse it or
    /// derive it lazily only at the indices it touches.
    #[inline]
    fn for_hash(h: u64, seed: u64) -> Probe {
        let fp = derive_fp(h, seed);
        Probe {
            fp,
            m_index: mix(h, seed ^ I1_SEED_TWEAK),
            alt_base: mix(fp as u64, seed),
        }
    }

    #[inline]
    pub fn bucket_indices(&self, mask: u64) -> (usize, usize) {
        let i1 = (self.m_index & mask) as usize;
        let i2 = ((i1 as u64) ^ ((self.alt_base & mask) | 1)) as usize;
        (i1, i2)
    }

    #[inline]
    pub fn fingerprint(&self) -> u16 {
        self.fp
    }
}

/// Precompute probes when the caller expects to reuse them across many DCs.
pub fn probes_for(seq: &[u64], seed: u64) -> Vec<Probe> {
    seq.iter().map(|&h| Probe::for_hash(h, seed)).collect()
}

/// Use authoritative misses to bound overlap depth; only hits can lie.
#[inline]
fn probe_present(filter: &CuckooFilter, p: &Probe) -> bool {
    let i1 = (p.m_index & filter.mask) as usize;
    if filter.has_at(i1, p.fp) {
        return true;
    }
    let i2 = ((i1 as u64) ^ ((p.alt_base & filter.mask) | 1)) as usize;
    filter.has_at(i2, p.fp)
}

/// Recheck the tail window so a mid-search false positive cannot inflate the
/// reported contiguous prefix.
pub const OVERLAP_VERIFY_WINDOW: usize = 8;

/// Use exponential plus binary search so deep prefixes stay logarithmic while
/// authoritative misses still bound the answer.
pub fn overlap_depth_searched(filter: &CuckooFilter, probes: &[Probe]) -> u32 {
    let n = probes.len();
    if n == 0 || !probe_present(filter, &probes[0]) {
        return 0;
    }
    // Grow exponentially so the first reliable miss appears without a linear
    // scan.
    let mut hi = 1usize;
    while hi < n && probe_present(filter, &probes[hi]) {
        hi <<= 1;
    }
    let mut lo = hi >> 1;
    let mut r = hi.min(n);
    while r - lo > 1 {
        let m = lo + (r - lo) / 2;
        if probe_present(filter, &probes[m]) {
            lo = m;
        } else {
            r = m;
        }
    }
    // Recheck the tail so a false positive cannot overstate the contiguous
    // prefix.
    let mut depth = r;
    for (k, probe) in probes
        .iter()
        .enumerate()
        .take(r)
        .skip(r.saturating_sub(OVERLAP_VERIFY_WINDOW))
    {
        if !probe_present(filter, probe) {
            depth = k;
            break;
        }
    }
    depth as u32
}
