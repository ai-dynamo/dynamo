// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sampled KV cache reuse ages.
//!
//! The router remembers when each worker last used a sample of prompt blocks.
//! When a later request reaches the same worker, every sampled block the worker
//! used before yields an age: how long the block sat unused on that worker.
//! A block inside the selected worker's cached prefix is a hit, labeled with
//! the tier that holds it. A block past the cached prefix is a miss that a
//! longer cache lifetime would have turned into a hit, so miss ages are not cut
//! off at the worker's current eviction age.
//!
//! A worker uses a block while a request routed to it holds the block, and
//! last used it when that request finished. The tracker sees only the prompt
//! blocks of requests that this router admits.

use std::{
    num::NonZeroUsize,
    sync::Arc,
    time::{Duration, Instant},
};

use dynamo_kv_router::{protocols::WorkerWithDpRank, scheduling::SelectedWorkerTierSnapshot};
use dynamo_runtime::config::environment_names::router::DYN_ROUTER_REUSE_AGE_SAMPLE_RATE;
use dynamo_tokens::SequenceHash;
use lru::LruCache;
use parking_lot::Mutex;

/// Upper bound on remembered (worker, block) pairs, about 100 bytes each. When
/// this bound rather than reuse ends a block's history, lower the sample rate.
const MAX_TRACKED_BLOCKS: usize = 1 << 18;

/// The storage tier that held a reused block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReuseTier {
    Device,
    HostPinned,
    /// Disk and external tiers, which the router reports together.
    Disk,
}

impl ReuseTier {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::HostPinned => "host_pinned",
            Self::Disk => "disk",
        }
    }
}

/// Ages of the sampled blocks of one routed request.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct ReuseAges {
    /// Blocks the selected worker still held, with the tier that held them.
    pub(crate) hits: Vec<(ReuseTier, Duration)>,
    /// Blocks the selected worker used before but no longer held.
    pub(crate) misses: Vec<Duration>,
}

#[derive(Debug, Clone, Copy)]
struct LastUse {
    at: Instant,
    /// Requests on the worker that still hold the block.
    holders: u32,
}

impl LastUse {
    fn held(at: Instant) -> Self {
        Self { at, holders: 1 }
    }

    fn released(at: Instant) -> Self {
        Self { at, holders: 0 }
    }
}

/// Remembers when each worker last used a sample of prompt blocks.
pub(crate) struct ReuseAgeTracker {
    /// Blocks whose sequence hash is at most this value are sampled.
    sample_threshold: u64,
    last_use: Mutex<LruCache<(WorkerWithDpRank, SequenceHash), LastUse>>,
}

impl ReuseAgeTracker {
    /// Returns a tracker when `DYN_ROUTER_REUSE_AGE_SAMPLE_RATE` is a positive fraction.
    pub(crate) fn from_env() -> Option<Arc<Self>> {
        sample_rate_from_lookup(|key| std::env::var(key).ok())
            .map(|sample_rate| Arc::new(Self::new(sample_rate, MAX_TRACKED_BLOCKS)))
    }

    pub(crate) fn new(sample_rate: f64, capacity: usize) -> Self {
        Self {
            // Sequence hashes are uniform, so a threshold keeps a fixed share
            // of blocks and every request samples the same ones.
            sample_threshold: (sample_rate * u64::MAX as f64) as u64,
            last_use: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).expect("tracker capacity is positive"),
            )),
        }
    }

    /// Measures the ages of the request's sampled blocks on `worker` and holds
    /// the blocks there until the returned lease drops.
    pub(crate) fn begin(
        self: &Arc<Self>,
        worker: WorkerWithDpRank,
        sequence_hashes: &[SequenceHash],
        tiers: &SelectedWorkerTierSnapshot,
        now: Instant,
    ) -> (ReuseAgeLease, ReuseAges) {
        let [device_end, host_end, cached_end] = cached_tier_ends(worker, tiers);
        let mut ages = ReuseAges::default();
        let mut held = Vec::new();
        let mut last_use = self.last_use.lock();
        for (position, &hash) in sequence_hashes.iter().enumerate() {
            if hash > self.sample_threshold {
                continue;
            }
            held.push(hash);
            let Some(entry) = last_use.get_mut(&(worker, hash)) else {
                last_use.put((worker, hash), LastUse::held(now));
                continue;
            };
            let in_use = entry.holders > 0;
            let age = if in_use {
                Duration::ZERO
            } else {
                now.saturating_duration_since(entry.at)
            };
            entry.holders = entry.holders.saturating_add(1);
            let tier = if position < device_end {
                Some(ReuseTier::Device)
            } else if position < host_end {
                Some(ReuseTier::HostPinned)
            } else if position < cached_end {
                Some(ReuseTier::Disk)
            } else {
                None
            };
            match tier {
                Some(tier) => ages.hits.push((tier, age)),
                // A block that another request still holds is not cached yet,
                // so no cache lifetime would turn it into a hit.
                None if in_use => {}
                None => ages.misses.push(age),
            }
        }
        drop(last_use);
        let lease = ReuseAgeLease {
            tracker: Arc::clone(self),
            worker,
            held,
        };
        (lease, ages)
    }

    fn release(&self, worker: WorkerWithDpRank, held: &[SequenceHash], now: Instant) {
        let mut last_use = self.last_use.lock();
        for &hash in held {
            match last_use.get_mut(&(worker, hash)) {
                Some(entry) => {
                    entry.at = now;
                    entry.holders = entry.holders.saturating_sub(1);
                }
                // Capacity evicted the entry while the request held it.
                None => {
                    last_use.put((worker, hash), LastUse::released(now));
                }
            }
        }
    }
}

/// Holds a request's sampled blocks on its worker; dropping it records their release.
pub(crate) struct ReuseAgeLease {
    tracker: Arc<ReuseAgeTracker>,
    worker: WorkerWithDpRank,
    held: Vec<SequenceHash>,
}

impl Drop for ReuseAgeLease {
    fn drop(&mut self) {
        self.tracker
            .release(self.worker, &self.held, Instant::now());
    }
}

/// Exclusive prefix ends of the device, host, and disk tiers on the selected
/// rank. Lower tiers extend the device prefix, and the snapshot reports their
/// ends cumulatively as the maximum over the worker's ranks.
fn cached_tier_ends(worker: WorkerWithDpRank, tiers: &SelectedWorkerTierSnapshot) -> [usize; 3] {
    let device = tiers
        .dp_device_blocks
        .iter()
        .find(|(dp_rank, _)| *dp_rank == worker.dp_rank)
        .map_or(tiers.gpu_blocks, |(_, blocks)| *blocks) as usize;
    let host = device.max(tiers.host_pinned_blocks as usize);
    let disk = host.max(tiers.disk_blocks as usize);
    [device, host, disk]
}

/// Parses the sample rate; unset or zero disables tracking.
fn sample_rate_from_lookup(get_env: impl Fn(&str) -> Option<String>) -> Option<f64> {
    let raw = get_env(DYN_ROUTER_REUSE_AGE_SAMPLE_RATE)?;
    match raw.trim().parse::<f64>() {
        Ok(sample_rate) if (0.0..=1.0).contains(&sample_rate) => {
            (sample_rate > 0.0).then_some(sample_rate)
        }
        _ => {
            tracing::warn!(
                env = DYN_ROUTER_REUSE_AGE_SAMPLE_RATE,
                value = %raw,
                "invalid KV reuse-age sample rate, expected a fraction in [0, 1]; reuse-age metrics stay disabled"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECOND: Duration = Duration::from_secs(1);

    fn worker(worker_id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(worker_id, 0)
    }

    fn device_prefix(blocks: u32) -> SelectedWorkerTierSnapshot {
        SelectedWorkerTierSnapshot {
            gpu_blocks: blocks,
            host_pinned_blocks: blocks,
            disk_blocks: blocks,
            ..Default::default()
        }
    }

    fn tracker() -> Arc<ReuseAgeTracker> {
        Arc::new(ReuseAgeTracker::new(1.0, 64))
    }

    impl ReuseAgeLease {
        fn release_at(mut self, now: Instant) {
            let held = std::mem::take(&mut self.held);
            self.tracker.release(self.worker, &held, now);
        }
    }

    #[test]
    fn sample_rate_requires_a_fraction() {
        let parse = |value: Option<&str>| {
            sample_rate_from_lookup(|key| {
                assert_eq!(key, DYN_ROUTER_REUSE_AGE_SAMPLE_RATE);
                value.map(str::to_string)
            })
        };
        assert_eq!(parse(None), None);
        assert_eq!(parse(Some("0")), None);
        assert_eq!(parse(Some("0.25")), Some(0.25));
        assert_eq!(parse(Some(" 1 ")), Some(1.0));
        assert_eq!(parse(Some("1.5")), None);
        assert_eq!(parse(Some("-0.1")), None);
        assert_eq!(parse(Some("NaN")), None);
        assert_eq!(parse(Some("often")), None);
    }

    #[test]
    fn ages_count_from_the_last_release_on_the_worker() {
        let tracker = tracker();
        let start = Instant::now();
        let (lease, ages) = tracker.begin(worker(1), &[11, 12, 13], &device_prefix(0), start);
        assert_eq!(ages, ReuseAges::default(), "first use has no age");
        lease.release_at(start + 10 * SECOND);

        let (_lease, ages) = tracker.begin(
            worker(1),
            &[11, 12, 13],
            &device_prefix(2),
            start + 70 * SECOND,
        );
        assert_eq!(
            ages,
            ReuseAges {
                hits: vec![
                    (ReuseTier::Device, 60 * SECOND),
                    (ReuseTier::Device, 60 * SECOND),
                ],
                misses: vec![60 * SECOND],
            }
        );
    }

    #[test]
    fn held_blocks_are_zero_age_hits_and_never_misses() {
        let tracker = tracker();
        let start = Instant::now();
        let (_holder, _) = tracker.begin(worker(1), &[11, 12], &device_prefix(0), start);

        let (_lease, ages) = tracker.begin(worker(1), &[11, 12], &device_prefix(1), start + SECOND);
        assert_eq!(
            ages,
            ReuseAges {
                hits: vec![(ReuseTier::Device, Duration::ZERO)],
                misses: vec![],
            }
        );
    }

    #[test]
    fn lower_tiers_extend_the_device_prefix() {
        let tracker = tracker();
        let start = Instant::now();
        let hashes = [11, 12, 13, 14];
        let (lease, _) = tracker.begin(worker(1), &hashes, &device_prefix(0), start);
        lease.release_at(start);

        let tiers = SelectedWorkerTierSnapshot {
            gpu_blocks: 1,
            host_pinned_blocks: 2,
            disk_blocks: 3,
            ..Default::default()
        };
        let (_lease, ages) = tracker.begin(worker(1), &hashes, &tiers, start + SECOND);
        assert_eq!(
            ages,
            ReuseAges {
                hits: vec![
                    (ReuseTier::Device, SECOND),
                    (ReuseTier::HostPinned, SECOND),
                    (ReuseTier::Disk, SECOND),
                ],
                misses: vec![SECOND],
            }
        );
    }

    #[test]
    fn device_tier_uses_the_selected_dp_rank() {
        let tiers = SelectedWorkerTierSnapshot {
            dp_device_blocks: vec![(0, 4), (1, 1)],
            gpu_blocks: 4,
            host_pinned_blocks: 4,
            disk_blocks: 4,
        };
        assert_eq!(
            cached_tier_ends(WorkerWithDpRank::new(1, 1), &tiers),
            [1, 4, 4]
        );
    }

    #[test]
    fn history_is_kept_per_worker() {
        let tracker = tracker();
        let start = Instant::now();
        let (lease, _) = tracker.begin(worker(1), &[11], &device_prefix(0), start);
        lease.release_at(start);

        let (_lease, ages) = tracker.begin(worker(2), &[11], &device_prefix(1), start + SECOND);
        assert_eq!(ages, ReuseAges::default());
    }

    #[test]
    fn only_sampled_hashes_are_tracked() {
        let tracker = Arc::new(ReuseAgeTracker::new(0.5, 64));
        let start = Instant::now();
        let hashes = [1, u64::MAX];
        let (lease, _) = tracker.begin(worker(1), &hashes, &device_prefix(0), start);
        lease.release_at(start);

        let (_lease, ages) = tracker.begin(worker(1), &hashes, &device_prefix(2), start + SECOND);
        assert_eq!(ages.hits, vec![(ReuseTier::Device, SECOND)]);
    }

    #[test]
    fn capacity_bounds_the_history() {
        let tracker = Arc::new(ReuseAgeTracker::new(1.0, 2));
        let start = Instant::now();
        for hash in [11, 12, 13] {
            let (lease, _) = tracker.begin(worker(1), &[hash], &device_prefix(0), start);
            lease.release_at(start);
        }

        let later = start + SECOND;
        let (_lease, forgotten) = tracker.begin(worker(1), &[11], &device_prefix(1), later);
        assert_eq!(
            forgotten,
            ReuseAges::default(),
            "the least recently used block left the history"
        );
        let (_lease, kept) = tracker.begin(worker(1), &[13], &device_prefix(1), later);
        assert_eq!(kept.hits, vec![(ReuseTier::Device, SECOND)]);
    }

    #[test]
    fn dropping_a_lease_releases_its_blocks() {
        let tracker = tracker();
        let start = Instant::now();
        let (lease, _) = tracker.begin(worker(1), &[11], &device_prefix(0), start);
        drop(lease);

        let (_lease, ages) = tracker.begin(worker(1), &[11], &device_prefix(1), Instant::now());
        assert_eq!(ages.hits.len(), 1);
        assert!(ages.misses.is_empty());
    }
}
