// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phantom layout, hash salting, and the shared wall-clock mapping.
//!
//! Every publisher and driver process derives the same per-phantom base, worker ID, salt, and
//! schedule from (manifest, layout, timing), so a query for phantom `i` is issued at the wall
//! instant phantom `i` publishes the events around it, whichever processes host them.

use anyhow::{Result, bail, ensure};
use serde::Serialize;

use crate::stream::Manifest;

/// Default first phantom worker ID: above the 2^53 discovery-safe publisher-ID space, so a
/// phantom never shares a publisher ID with a real direct-ZMQ publisher.
pub const DEFAULT_WORKER_ID_BASE: u64 = 0x7F00_0000_0000_0000;

/// MurmurHash3's 64-bit finalizer: a bijection on `u64`.
#[inline]
pub fn fmix64(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    value ^= value >> 33;
    value
}

/// Remap one block hash for a phantom copy. A bijection for a fixed key, so equality and
/// therefore every parent/child and lookup relation within a base survives, while copies with
/// different keys share no hashes (with overwhelming probability).
///
/// NOTE: Non-root sequence hashes are remapped independently rather than re-chained from the
/// remapped local hashes; neither arm's event-driven indexer recomputes sequence hashes from
/// local hashes, so the index structure is identical to a re-chained copy.
#[inline]
pub fn salt_hash(hash: u64, key: u64) -> u64 {
    fmix64(hash ^ key)
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PhantomLayout {
    /// Phantoms across every publisher process.
    pub total: u64,
    pub worker_id_base: u64,
    pub salt_seed: u64,
}

impl PhantomLayout {
    pub fn validate(&self, manifest: &Manifest) -> Result<()> {
        ensure!(self.total > 0, "--total-phantoms must be positive");
        ensure!(
            self.worker_id_base.checked_add(self.total).is_some(),
            "phantom worker IDs overflow u64"
        );
        ensure!(
            self.worker_id_base > (1 << 53),
            "--worker-id-base must exceed 2^53 to stay disjoint from real publisher IDs"
        );
        ensure!(!manifest.bases.is_empty(), "the stream has no bases");
        Ok(())
    }

    /// Contiguous phantom ranges share a base, so a process hosting a range loads few bases.
    pub fn base_of(&self, phantom: u64, bases: usize) -> usize {
        ((u128::from(phantom) * bases as u128) / u128::from(self.total)) as usize
    }

    pub fn worker_id(&self, phantom: u64) -> u64 {
        self.worker_id_base + phantom
    }

    pub fn salt_key(&self, phantom: u64) -> u64 {
        fmix64(self.salt_seed ^ fmix64(phantom.wrapping_add(1)))
    }

    /// Deterministic per-phantom timed-phase delay in `[0, spread_us)`.
    pub fn start_delay_us(&self, phantom: u64, spread_us: u64) -> u64 {
        if spread_us == 0 {
            return 0;
        }
        fmix64(phantom ^ 0x5DEE_CE66_D1CE_4E5B ^ self.salt_seed) % spread_us
    }
}

/// Aggregate natural rates (per virtual second) of `total` phantoms at speedup 1.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct NaturalRates {
    pub write_blocks: f64,
    pub stored_blocks: f64,
    pub removed_blocks: f64,
    pub events: f64,
    pub queries: f64,
    pub query_blocks: f64,
    /// Warm-up write blocks summed over phantoms (untimed).
    pub warmup_write_blocks: u64,
}

pub fn natural_rates(
    manifest: &Manifest,
    layout: &PhantomLayout,
    phantoms: std::ops::Range<u64>,
) -> NaturalRates {
    let span_s = manifest.span_us() as f64 / 1e6;
    let mut rates = NaturalRates::default();
    for phantom in phantoms {
        let base = &manifest.bases[layout.base_of(phantom, manifest.bases.len())];
        rates.write_blocks += base.timed.write_blocks() as f64;
        rates.stored_blocks += base.timed.stored_blocks as f64;
        rates.removed_blocks += base.timed.removed_blocks as f64;
        rates.events += base.timed.events as f64;
        rates.queries += base.queries as f64;
        rates.query_blocks += base.query_blocks as f64;
        rates.warmup_write_blocks += base.warmup.write_blocks();
    }
    rates.write_blocks /= span_s;
    rates.stored_blocks /= span_s;
    rates.removed_blocks /= span_s;
    rates.events /= span_s;
    rates.queries /= span_s;
    rates.query_blocks /= span_s;
    rates
}

/// Resolve the speedup from either an explicit value or a target aggregate write-block rate
/// (stored + removed blocks per wall second over all `total` phantoms).
pub fn resolve_speedup(
    manifest: &Manifest,
    layout: &PhantomLayout,
    speedup: Option<f64>,
    target_write_blocks_per_sec: Option<f64>,
) -> Result<f64> {
    let speedup = match (speedup, target_write_blocks_per_sec) {
        (Some(speedup), None) => speedup,
        (None, Some(target)) => {
            let natural = natural_rates(manifest, layout, 0..layout.total).write_blocks;
            ensure!(natural > 0.0, "the stream has no timed writes");
            target / natural
        }
        (Some(_), Some(_)) => bail!("pass --speedup or --target-write-blocks-per-sec, not both"),
        (None, None) => bail!("pass --speedup or --target-write-blocks-per-sec"),
    };
    ensure!(
        speedup.is_finite() && speedup > 0.0,
        "speedup {speedup} must be finite and positive"
    );
    Ok(speedup)
}

/// Wall-clock mapping of the timed section.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TimeMap {
    /// Unix microseconds at which virtual `t0_us` plays for a phantom with zero delay.
    pub start_at_unix_us: u64,
    pub t0_us: u64,
    pub speedup: f64,
}

impl TimeMap {
    pub fn wall_us(&self, ts_us: u64, delay_us: u64) -> u64 {
        let offset = ts_us.saturating_sub(self.t0_us) as f64 / self.speedup;
        self.start_at_unix_us + delay_us + offset as u64
    }

    /// Wall seconds the timed section covers at this speedup.
    pub fn coverage_s(&self, manifest: &Manifest) -> f64 {
        manifest.span_us() as f64 / 1e6 / self.speedup
    }
}

pub fn unix_now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_spreads_contiguous_ranges_over_bases() {
        let layout = PhantomLayout {
            total: 10,
            worker_id_base: DEFAULT_WORKER_ID_BASE,
            salt_seed: 0,
        };
        let bases: Vec<_> = (0..10).map(|phantom| layout.base_of(phantom, 4)).collect();
        assert_eq!(bases, vec![0, 0, 0, 1, 1, 2, 2, 2, 3, 3]);
        assert_ne!(layout.salt_key(0), layout.salt_key(1));
        assert!(layout.start_delay_us(3, 1000) < 1000);
    }

    #[test]
    fn salt_is_injective_on_a_sample_and_key_dependent() {
        let mut seen = std::collections::HashSet::new();
        for hash in 0..100_000u64 {
            assert!(seen.insert(salt_hash(hash, 7)));
        }
        assert_ne!(salt_hash(42, 7), salt_hash(42, 8));
    }

    #[test]
    fn time_map_scales_offsets() {
        let map = TimeMap {
            start_at_unix_us: 1_000_000,
            t0_us: 500,
            speedup: 2.0,
        };
        assert_eq!(map.wall_us(500, 0), 1_000_000);
        assert_eq!(map.wall_us(2_500, 10), 1_001_010);
    }
}
