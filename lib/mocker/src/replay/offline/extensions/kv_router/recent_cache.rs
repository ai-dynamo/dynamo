// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in replay experiment: recent routed input demand, not resident KV cache.
//! Evictions and generated output blocks deliberately do not update this index.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use dynamo_kv_router::protocols::{OverlapScores, RouterEvent, WorkerId, WorkerWithDpRank};
use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use super::resident_cache::ResidentCache;

pub(super) const ENV: &str = "DYN_REPLAY_RECENT_CACHE";

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Mode {
    Observe,
    Fixed,
    #[default]
    Adaptive,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Config {
    pub ttl_secs: f64,
    pub threshold: f64,
    pub boost_credit: f64,
    pub mode: Mode,
    pub predict: bool,
    pub trace_decisions: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ttl_secs: 300.0,
            threshold: 0.5,
            boost_credit: 4.0,
            mode: Mode::Adaptive,
            predict: true,
            trace_decisions: false,
        }
    }
}

impl Config {
    pub(super) fn from_env() -> Result<Option<Self>> {
        let value = match std::env::var(ENV) {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(error) => return Err(error).context(ENV),
        };
        let config: Self = serde_json::from_str(&value).context(ENV)?;
        config.validate()?;
        Ok(Some(config))
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.ttl_secs.is_finite() && self.ttl_secs > 0.0,
            "{ENV}: ttl_secs must be finite and positive"
        );
        Duration::try_from_secs_f64(self.ttl_secs).context("recent-cache TTL is too large")?;
        ensure!(
            self.threshold.is_finite() && self.threshold > 0.0,
            "{ENV}: threshold must be finite and positive"
        );
        ensure!(
            self.boost_credit.is_finite() && self.boost_credit >= 0.0,
            "{ENV}: boost_credit must be finite and non-negative"
        );
        Ok(())
    }
}

#[derive(Serialize)]
struct Sample {
    time_secs: f64,
    ratio: f64,
    count: usize,
    capacity: u64,
    boosted: bool,
    selected_worker: Option<WorkerWithDpRank>,
    selected_reported_overlap_blocks: Option<u32>,
}

pub(super) struct Decision {
    sample: Sample,
    pub credit: Option<f64>,
}

pub(super) struct RecentCache {
    config: Config,
    ttl: Duration,
    epoch: Instant,
    entries: HashMap<WorkerWithDpRank, FxHashMap<SequenceHash, Instant>>,
    // Each block copy has exactly one expiry entry, including after refresh.
    expiry: BTreeSet<(Instant, WorkerWithDpRank, SequenceHash)>,
    capacities: BTreeMap<WorkerWithDpRank, u64>,
    capacity: u64,
    decisions: u64,
    decisions_above_threshold: u64,
    pressure_upward_crossings: u64,
    pressure_downward_crossings: u64,
    previous_above_threshold: Option<bool>,
    boosted_decisions: u64,
    mode_transitions: u64,
    previous_boosted: Option<bool>,
    pressure_sum: f64,
    peak_footprint: usize,
    peak_worker_pressure: f64,
    last_sample_second: Option<u64>,
    series: Vec<Sample>,
    route_counts: BTreeMap<WorkerWithDpRank, u64>,
    resident_cache: ResidentCache,
}

impl RecentCache {
    pub(super) fn new(config: Config, epoch: Instant) -> Result<Self> {
        config.validate()?;
        let ttl = Duration::try_from_secs_f64(config.ttl_secs)?;
        ensure!(
            epoch.checked_add(ttl).is_some(),
            "recent-cache TTL is too large"
        );
        Ok(Self {
            config,
            ttl,
            epoch,
            entries: HashMap::new(),
            expiry: BTreeSet::new(),
            capacities: BTreeMap::new(),
            capacity: 0,
            decisions: 0,
            decisions_above_threshold: 0,
            pressure_upward_crossings: 0,
            pressure_downward_crossings: 0,
            previous_above_threshold: None,
            boosted_decisions: 0,
            mode_transitions: 0,
            previous_boosted: None,
            pressure_sum: 0.0,
            peak_footprint: 0,
            peak_worker_pressure: 0.0,
            last_sample_second: None,
            series: Vec::new(),
            route_counts: BTreeMap::new(),
            resident_cache: ResidentCache::default(),
        })
    }

    pub(super) fn add_worker(&mut self, worker: WorkerWithDpRank, capacity: u64) {
        self.resident_cache.add_worker(worker, capacity);
        if let Some(previous) = self.capacities.insert(worker, capacity) {
            self.capacity -= previous;
        }
        self.capacity += capacity;
    }

    pub(super) fn remove_worker(&mut self, worker_id: WorkerId) {
        self.capacities.retain(|worker, capacity| {
            if worker.worker_id != worker_id {
                return true;
            }
            self.capacity -= *capacity;
            if let Some(entries) = self.entries.remove(worker) {
                for (hash, expiry) in entries {
                    self.expiry.remove(&(expiry, *worker, hash));
                }
            }
            false
        });
    }

    pub(super) fn finalize_worker_removal(&mut self, worker_id: WorkerId) {
        self.resident_cache.remove_worker(worker_id);
    }

    pub(super) fn observe_resident_event(&mut self, event: &RouterEvent) {
        self.resident_cache.observe(event);
    }

    pub(super) fn sample_resident(&mut self, now: Instant) {
        self.resident_cache
            .sample(now.saturating_duration_since(self.epoch).as_secs_f64());
    }

    pub(super) fn expire(&mut self, now: Instant) {
        while let Some(&(expiry, worker, hash)) = self.expiry.first() {
            if expiry > now {
                break;
            }
            self.expiry.pop_first();
            if let Some(entries) = self.entries.get_mut(&worker) {
                entries.remove(&hash);
            }
        }
    }

    pub(super) fn prepare(
        &mut self,
        now: Instant,
        hashes: &[SequenceHash],
        overlaps: &mut OverlapScores,
    ) -> Decision {
        self.expire(now);
        if self.config.predict {
            for (worker, entries) in &self.entries {
                let count = hashes
                    .iter()
                    .take_while(|hash| entries.contains_key(hash))
                    .count();
                if count > 0 {
                    let overlap = overlaps.scores.entry(*worker).or_default();
                    *overlap = (*overlap).max(u32::try_from(count).unwrap_or(u32::MAX));
                }
            }
        }
        let ratio = self.ratio();
        let boosted = match self.config.mode {
            Mode::Observe => false,
            Mode::Fixed => true,
            Mode::Adaptive => ratio > self.config.threshold,
        };
        Decision {
            sample: Sample {
                time_secs: now.saturating_duration_since(self.epoch).as_secs_f64(),
                ratio,
                count: self.expiry.len(),
                capacity: self.capacity,
                boosted,
                selected_worker: None,
                selected_reported_overlap_blocks: None,
            },
            credit: boosted.then_some(self.config.boost_credit),
        }
    }

    pub(super) fn admitted(
        &mut self,
        worker: WorkerWithDpRank,
        hashes: &[SequenceHash],
        now: Instant,
        reported_overlap_blocks: u32,
        decision: Decision,
    ) {
        self.sample_resident(now);
        let mut sample = decision.sample;
        sample.selected_worker = Some(worker);
        sample.selected_reported_overlap_blocks = Some(reported_overlap_blocks);
        self.decisions += 1;
        let above = sample.ratio > self.config.threshold;
        self.decisions_above_threshold += u64::from(above);
        if self
            .previous_above_threshold
            .is_some_and(|previous| previous != above)
        {
            self.pressure_upward_crossings += u64::from(above);
            self.pressure_downward_crossings += u64::from(!above);
        }
        self.previous_above_threshold = Some(above);
        self.boosted_decisions += u64::from(sample.boosted);
        self.mode_transitions += u64::from(
            self.previous_boosted
                .is_some_and(|previous| previous != sample.boosted),
        );
        self.previous_boosted = Some(sample.boosted);
        self.pressure_sum += sample.ratio;
        let second = sample.time_secs.floor() as u64;
        if self.config.trace_decisions || self.last_sample_second != Some(second) {
            self.last_sample_second = Some(second);
            self.series.push(sample);
        }
        *self.route_counts.entry(worker).or_default() += 1;
        let expiry = now + self.ttl;
        let entries = self.entries.entry(worker).or_default();
        for hash in hashes {
            if let Some(previous) = entries.insert(*hash, expiry) {
                self.expiry.remove(&(previous, worker, *hash));
            }
            self.expiry.insert((expiry, worker, *hash));
        }
        self.peak_footprint = self.peak_footprint.max(self.expiry.len());
        if let Some(&capacity) = self
            .capacities
            .get(&worker)
            .filter(|&&capacity| capacity > 0)
        {
            self.peak_worker_pressure = self
                .peak_worker_pressure
                .max(entries.len() as f64 / capacity as f64);
        }
    }

    fn ratio(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.expiry.len() as f64 / self.capacity as f64
    }

    fn diagnostics(&self) -> serde_json::Value {
        let workers = self
            .route_counts
            .keys()
            .chain(self.capacities.keys())
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|worker| {
                serde_json::json!({
                    "worker_id": worker.worker_id,
                    "dp_rank": worker.dp_rank,
                    "route_count": self.route_counts.get(&worker).copied().unwrap_or_default(),
                    "recent_count": self.entries.get(&worker).map_or(0, |entries| entries.len()),
                    "capacity": self.capacities.get(&worker).copied().unwrap_or_default(),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "config": self.config,
            "measurement": "recent_routed_input_block_copies",
            "decisions": self.decisions,
            "decisions_above_threshold": self.decisions_above_threshold,
            "decisions_at_or_below_threshold": self.decisions - self.decisions_above_threshold,
            "pressure_upward_crossings": self.pressure_upward_crossings,
            "pressure_downward_crossings": self.pressure_downward_crossings,
            "boosted_decisions": self.boosted_decisions,
            "mode_transitions": self.mode_transitions,
            "mean_decision_pressure": self.pressure_sum / self.decisions.max(1) as f64,
            "peak_footprint": self.peak_footprint,
            "peak_per_worker_pressure": self.peak_worker_pressure,
            "final_count": self.expiry.len(),
            "final_capacity": self.capacity,
            "series": self.series,
            "workers": workers,
            "resident_cache": self.resident_cache.diagnostics(),
        })
    }
}

impl Drop for RecentCache {
    fn drop(&mut self) {
        eprintln!("recent_cache_experiment={}", self.diagnostics());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admit(cache: &mut RecentCache, worker: WorkerWithDpRank, hashes: &[u64], now: Instant) {
        let decision = cache.prepare(now, hashes, &mut OverlapScores::new());
        cache.admitted(worker, hashes, now, 0, decision);
    }

    #[test]
    fn refresh_replaces_expiry_and_uses_virtual_time() {
        let start = Instant::now();
        let mut cache = RecentCache::new(
            Config {
                ttl_secs: 10.0,
                ..Config::default()
            },
            start,
        )
        .unwrap();
        let worker = WorkerWithDpRank::new(0, 0);
        cache.add_worker(worker, 10);
        admit(&mut cache, worker, &[1, 2], start);
        admit(&mut cache, worker, &[1], start + Duration::from_secs(9));
        assert_eq!(cache.expiry.len(), 2);
        cache.expire(start + Duration::from_secs(10));
        assert_eq!(cache.entries[&worker].len(), 1);
        assert!(cache.entries[&worker].contains_key(&1));
        cache.expire(start + Duration::from_secs(19));
        assert!(cache.expiry.is_empty());
    }

    #[test]
    fn pressure_counts_copies_and_dp_capacity_then_removes_worker() {
        let now = Instant::now();
        let mut cache = RecentCache::new(Config::default(), now).unwrap();
        let ranks = [
            WorkerWithDpRank::new(0, 0),
            WorkerWithDpRank::new(0, 1),
            WorkerWithDpRank::new(1, 0),
        ];
        for rank in ranks {
            cache.add_worker(rank, 4);
            admit(&mut cache, rank, &[1, 2], now);
        }
        assert_eq!(cache.ratio(), 0.5);
        assert_eq!(cache.expiry.len(), 6);
        cache.remove_worker(0);
        assert_eq!(cache.capacity, 4);
        assert_eq!(cache.expiry.len(), 2);
        assert_eq!(cache.ratio(), 0.5);
        assert!(!cache.entries.contains_key(&ranks[0]));
        assert!(!cache.entries.contains_key(&ranks[1]));
    }

    #[test]
    fn prediction_is_prefix_only_and_max_with_primary() {
        let now = Instant::now();
        let worker = WorkerWithDpRank::new(0, 0);
        let mut cache = RecentCache::new(Config::default(), now).unwrap();
        cache.add_worker(worker, 8);
        admit(&mut cache, worker, &[1, 2, 3], now);
        let mut overlaps = OverlapScores::new();
        cache.prepare(now, &[1, 9, 3], &mut overlaps);
        assert_eq!(overlaps.scores[&worker], 1);
        overlaps.scores.insert(worker, 4);
        cache.prepare(now, &[1, 2, 3], &mut overlaps);
        assert_eq!(overlaps.scores[&worker], 4);
    }

    #[test]
    fn observe_can_disable_prediction_and_adaptive_boost_expires() {
        let now = Instant::now();
        let worker = WorkerWithDpRank::new(0, 0);
        let mut cache = RecentCache::new(
            Config {
                ttl_secs: 1.0,
                mode: Mode::Observe,
                predict: false,
                ..Config::default()
            },
            now,
        )
        .unwrap();
        cache.add_worker(worker, 4);
        admit(&mut cache, worker, &[1, 2, 3], now);
        let mut overlaps = OverlapScores::new();
        assert!(
            cache
                .prepare(now, &[1, 2, 3], &mut overlaps)
                .credit
                .is_none()
        );
        assert!(overlaps.scores.is_empty());
        cache.config.mode = Mode::Adaptive;
        assert_eq!(cache.prepare(now, &[], &mut overlaps).credit, Some(4.0));
        let later = now + Duration::from_secs(1);
        assert!(cache.prepare(later, &[], &mut overlaps).credit.is_none());
        cache.config.mode = Mode::Fixed;
        assert_eq!(cache.prepare(later, &[], &mut overlaps).credit, Some(4.0));
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        for config in [
            Config {
                ttl_secs: 0.0,
                ..Config::default()
            },
            Config {
                threshold: f64::NAN,
                ..Config::default()
            },
            Config {
                boost_credit: -1.0,
                ..Config::default()
            },
        ] {
            assert!(RecentCache::new(config, Instant::now()).is_err());
        }
    }
}
