// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use xxhash_rust::xxh3;

const ALTERNATE_BUCKET_DOMAIN: u64 = 0x9E37_79B9_7F4A_7C15;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CkfProbe {
    pub(super) fingerprint: u16,
    pub(super) bucket_a: usize,
    pub(super) bucket_b: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CkfAddressing {
    bucket_mask: usize,
    seed: u64,
}

impl CkfAddressing {
    pub(super) fn new(bucket_count: usize, seed: u64) -> Self {
        debug_assert!(bucket_count.is_power_of_two());
        debug_assert!(bucket_count >= 2);
        Self {
            bucket_mask: bucket_count - 1,
            seed,
        }
    }

    #[inline]
    pub(super) fn prepare(&self, sequence_hash: u64) -> CkfProbe {
        let mixed = xxh3::xxh3_64_with_seed(&sequence_hash.to_le_bytes(), self.seed);
        let fingerprint = nonzero_fingerprint(mixed as u16);
        let bucket_a = ((mixed >> 16) as usize) & self.bucket_mask;
        let bucket_b = self.alternate_bucket(bucket_a, fingerprint);
        CkfProbe {
            fingerprint,
            bucket_a,
            bucket_b,
        }
    }

    #[inline]
    pub(super) fn alternate_bucket(&self, bucket: usize, fingerprint: u16) -> usize {
        bucket ^ self.bucket_delta(fingerprint)
    }

    #[inline]
    fn bucket_delta(&self, fingerprint: u16) -> usize {
        // TODO(perf): Benchmark a seed/mask-specific fingerprint delta table before replacing
        // this hash, including its per-indexer memory and cache effects.
        let mixed = xxh3::xxh3_64_with_seed(
            &fingerprint.to_le_bytes(),
            self.seed ^ ALTERNATE_BUCKET_DOMAIN,
        );
        let delta = (mixed as usize) & self.bucket_mask;
        if delta == 0 { 1 } else { delta }
    }
}

#[inline]
fn nonzero_fingerprint(fingerprint: u16) -> u16 {
    if fingerprint == 0 { 1 } else { fingerprint }
}
