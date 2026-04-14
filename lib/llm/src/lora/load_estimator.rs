// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LORA Load Estimator
//!
//! Tracks LORA adapter usage over time to estimate load for allocation decisions.
//! Supports single-router (polling) and multi-router (event-based) modes.
//!
//! The primary load signal is **arrival count in a sliding window**, tracked by
//! a lock-free [`BucketedRateCounter`] per LoRA. An optional [`LoadPredictor`]
//! (e.g. EMA) can smooth the raw counts for the allocation controller.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dynamo_kv_router::protocols::{ActiveSequenceEvent, ActiveSequenceEventData};
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;

use crate::kv_router::ACTIVE_SEQUENCES_SUBJECT;
use crate::kv_router::scheduler::KvScheduler;
use crate::lora::config::PredictorType;
use crate::lora::predictor::{EmaPredictor, LoadPredictor};

// ─── BucketedRateCounter ────────────────────────────────────────────────────

/// Lock-free, epoch-based sliding-window rate counter.
///
/// Divides time into fixed-duration buckets. Each bucket has an atomic counter
/// and an epoch (the absolute bucket index it was last used for). Stale buckets
/// are detected by epoch mismatch and lazily reset via CAS.
pub struct BucketedRateCounter {
    buckets: Vec<AtomicU64>,
    epochs: Vec<AtomicU64>,
    epoch_start: Instant,
    bucket_duration: Duration,
    num_buckets: usize,
}

// SAFETY: Composed entirely of atomics and an immutable Instant.
unsafe impl Sync for BucketedRateCounter {}

impl std::fmt::Debug for BucketedRateCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BucketedRateCounter")
            .field("num_buckets", &self.num_buckets)
            .field("bucket_duration", &self.bucket_duration)
            .finish()
    }
}

impl BucketedRateCounter {
    pub fn new(num_buckets: usize, bucket_duration: Duration, now: Instant) -> Self {
        assert!(num_buckets > 0, "num_buckets must be > 0");
        let mut buckets = Vec::with_capacity(num_buckets);
        let mut epochs = Vec::with_capacity(num_buckets);
        for _ in 0..num_buckets {
            buckets.push(AtomicU64::new(0));
            epochs.push(AtomicU64::new(0));
        }
        Self {
            buckets,
            epochs,
            epoch_start: now,
            bucket_duration,
            num_buckets,
        }
    }

    /// Record a single arrival at time `now`. Lock-free.
    pub fn record(&self, now: Instant) {
        self.record_count(1, now);
    }

    /// Record `n` arrivals at time `now`. Lock-free.
    pub fn record_count(&self, n: u64, now: Instant) {
        if n == 0 {
            return;
        }
        let elapsed = now.duration_since(self.epoch_start);
        let global_bucket = elapsed.as_nanos() / self.bucket_duration.as_nanos();
        let global_bucket = global_bucket as u64;
        let index = (global_bucket as usize) % self.num_buckets;

        let current_epoch = self.epochs[index].load(Ordering::Acquire);
        if current_epoch == global_bucket {
            self.buckets[index].fetch_add(n, Ordering::Relaxed);
        } else if current_epoch < global_bucket {
            match self.epochs[index].compare_exchange(
                current_epoch,
                global_bucket,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.buckets[index].store(n, Ordering::Release);
                }
                Err(actual) => {
                    if actual == global_bucket {
                        self.buckets[index].fetch_add(n, Ordering::Relaxed);
                    }
                    // else: another thread advanced to a newer epoch; drop this record
                }
            }
        }
    }

    /// Count total arrivals within the sliding window ending at `now`.
    pub fn count(&self, now: Instant) -> u64 {
        let elapsed = now.duration_since(self.epoch_start);
        let global_bucket = (elapsed.as_nanos() / self.bucket_duration.as_nanos()) as u64;

        let mut total = 0u64;
        for i in 0..self.num_buckets {
            let epoch = self.epochs[i].load(Ordering::Acquire);
            let min_valid_epoch = global_bucket.saturating_sub(self.num_buckets as u64 - 1);
            if epoch >= min_valid_epoch && epoch <= global_bucket {
                total += self.buckets[i].load(Ordering::Relaxed);
            }
        }
        total
    }

    pub fn clear(&self) {
        for i in 0..self.num_buckets {
            self.buckets[i].store(0, Ordering::Release);
            self.epochs[i].store(0, Ordering::Release);
        }
    }
}

// ─── LoraLoadData ───────────────────────────────────────────────────────────

/// Per-LORA load data: lock-free hot-path atomics only.
struct LoraLoadData {
    active_count: AtomicUsize,
    rate_counter: BucketedRateCounter,
}

impl std::fmt::Debug for LoraLoadData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoraLoadData")
            .field("active_count", &self.active_count.load(Ordering::Relaxed))
            .field("rate_counter", &self.rate_counter)
            .finish()
    }
}

// ─── LoadEstimatorConfig ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LoadEstimatorConfig {
    pub poll_interval: Duration,
    /// Sliding window size for request-rate calculation.
    pub rate_window: Duration,
    pub buckets_per_second: u64,
    pub predictor_type: PredictorType,
    pub ema_alpha: f64,
}

impl Default for LoadEstimatorConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(5),
            rate_window: Duration::from_secs(30),
            buckets_per_second: 1,
            predictor_type: PredictorType::Ema,
            ema_alpha: 0.5,
        }
    }
}

impl LoadEstimatorConfig {
    /// Derive a config synchronized with the allocation controller.
    pub fn from_controller_timestep(timestep_secs: u64, multiplier: u64) -> Self {
        let min_window = crate::lora::config::MIN_RATE_WINDOW_SECS;
        Self {
            rate_window: Duration::from_secs((timestep_secs * multiplier).max(min_window)),
            ..Default::default()
        }
    }

    fn num_buckets(&self) -> usize {
        let secs = self.rate_window.as_secs().max(1);
        (secs * self.buckets_per_second).max(1) as usize
    }

    fn bucket_duration(&self) -> Duration {
        if self.buckets_per_second == 0 {
            Duration::from_secs(1)
        } else {
            Duration::from_nanos(1_000_000_000 / self.buckets_per_second)
        }
    }
}

// ─── LoadEstimator ──────────────────────────────────────────────────────────

/// Estimates LORA load based on arrival counts over a sliding time window.
///
/// The hot path (`increment_load`) is lock-free for existing LoRAs.
pub struct LoadEstimator {
    data: DashMap<String, LoraLoadData>,
    predictors: Mutex<HashMap<String, Box<dyn LoadPredictor>>>,
    config: parking_lot::RwLock<LoadEstimatorConfig>,
}

impl LoadEstimator {
    pub fn new() -> Self {
        Self::with_config(LoadEstimatorConfig::default())
    }

    pub fn with_config(config: LoadEstimatorConfig) -> Self {
        Self {
            data: DashMap::new(),
            predictors: Mutex::new(HashMap::new()),
            config: parking_lot::RwLock::new(config),
        }
    }

    /// Update the rate window at runtime.
    pub fn set_rate_window(&self, window: Duration) {
        let mut cfg = self.config.write();
        let old_window = cfg.rate_window;
        cfg.rate_window = window;
        let num_buckets = cfg.num_buckets();
        let bucket_duration = cfg.bucket_duration();
        drop(cfg);

        if old_window != window {
            let now = Instant::now();
            for mut entry in self.data.iter_mut() {
                let new_counter = BucketedRateCounter::new(num_buckets, bucket_duration, now);
                let old_active = entry.value().active_count.load(Ordering::Relaxed);
                *entry.value_mut() = LoraLoadData {
                    active_count: AtomicUsize::new(old_active),
                    rate_counter: new_counter,
                };
            }
        }

        tracing::info!(
            rate_window_secs = window.as_secs(),
            num_buckets,
            "LoadEstimator rate_window updated"
        );
    }

    /// Clear all windowed arrival data for a LoRA.
    pub fn clear_rate_counter(&self, lora_name: &str) {
        if let Some(entry) = self.data.get(lora_name) {
            entry.value().rate_counter.clear();
        }
    }

    pub fn start_polling(
        self: Arc<Self>,
        scheduler: Arc<KvScheduler>,
        component: Component,
    ) -> tokio::task::JoinHandle<()> {
        let cancel_token = component.drt().child_token();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self.config.read().poll_interval);
            tracing::info!("Started LORA load polling");

            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        tracing::debug!("LORA load polling task cancelled");
                        break;
                    }
                    _ = interval.tick() => {
                        let lora_counts = scheduler.get_active_lora_counts();
                        self.update_from_counts(lora_counts);
                    }
                }
            }
        })
    }

    pub fn start_event_subscription(
        self: Arc<Self>,
        component: Component,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            if let Err(e) = self.subscribe_to_events(component).await {
                tracing::error!("Error in LORA load event subscription: {}", e);
            }
        })
    }

    async fn subscribe_to_events(&self, component: Component) -> anyhow::Result<()> {
        let cancel_token = component.drt().child_token();
        let mut subscriber = EventSubscriber::for_component(&component, ACTIVE_SEQUENCES_SUBJECT)
            .await?
            .typed::<ActiveSequenceEvent>();

        tracing::info!("Started LORA load event subscription");

        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    tracing::debug!("LORA load event subscription cancelled");
                    break;
                }
                result = subscriber.next() => {
                    match result {
                        Some(Ok((_envelope, event))) => {
                            self.handle_event(event);
                        }
                        Some(Err(e)) => {
                            tracing::warn!("Error receiving LORA load event: {}", e);
                        }
                        None => {
                            tracing::warn!("LORA load event stream ended");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn handle_event(&self, event: ActiveSequenceEvent) {
        if let Some(lora_name) = event.lora_name {
            match event.data {
                ActiveSequenceEventData::AddRequest { .. } => {
                    self.increment_load(&lora_name);
                }
                ActiveSequenceEventData::Free => {
                    self.decrement_load(&lora_name);
                }
                ActiveSequenceEventData::MarkPrefillCompleted => {}
            }
        }
    }

    /// Increment load count for a LORA and record arrival. Lock-free for existing LoRAs.
    pub fn increment_load(&self, lora_name: &str) {
        let now = Instant::now();

        // Fast path: LoRA already exists
        if let Some(entry) = self.data.get(lora_name) {
            entry.value().active_count.fetch_add(1, Ordering::Relaxed);
            entry.value().rate_counter.record(now);
            return;
        }

        // Slow path: first time seeing this LoRA
        let cfg = self.config.read();
        let num_buckets = cfg.num_buckets();
        let bucket_duration = cfg.bucket_duration();
        drop(cfg);

        self.data
            .entry(lora_name.to_string())
            .and_modify(|data| {
                data.active_count.fetch_add(1, Ordering::Relaxed);
                data.rate_counter.record(now);
            })
            .or_insert_with(|| {
                let counter = BucketedRateCounter::new(num_buckets, bucket_duration, now);
                counter.record(now);
                LoraLoadData {
                    active_count: AtomicUsize::new(1),
                    rate_counter: counter,
                }
            });
    }

    /// Decrement in-flight count for a LORA.
    pub fn decrement_load(&self, lora_name: &str) {
        if let Some(entry) = self.data.get(lora_name) {
            entry.value().active_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn update_from_counts(&self, lora_counts: HashMap<String, usize>) {
        let now = Instant::now();
        let cfg = self.config.read();
        let num_buckets = cfg.num_buckets();
        let bucket_duration = cfg.bucket_duration();
        drop(cfg);

        for (lora_name, count) in &lora_counts {
            self.data
                .entry(lora_name.clone())
                .and_modify(|data| {
                    data.active_count.store(*count, Ordering::Relaxed);
                    data.rate_counter.record_count(*count as u64, now);
                })
                .or_insert_with(|| {
                    let counter = BucketedRateCounter::new(num_buckets, bucket_duration, now);
                    counter.record_count(*count as u64, now);
                    LoraLoadData {
                        active_count: AtomicUsize::new(*count),
                        rate_counter: counter,
                    }
                });
        }

        for entry in self.data.iter() {
            if !lora_counts.contains_key(entry.key()) {
                entry.value().active_count.store(0, Ordering::Relaxed);
            }
        }
    }

    /// Get current load using arrival count in the sliding window.
    pub fn get_current_load(&self) -> HashMap<String, usize> {
        let now = Instant::now();
        let cfg = self.config.read();
        let predictor_type = cfg.predictor_type;
        let ema_alpha = cfg.ema_alpha;
        drop(cfg);

        if predictor_type == PredictorType::None {
            return self
                .data
                .iter()
                .filter_map(|entry| {
                    let count = entry.value().rate_counter.count(now);
                    if count > 0 {
                        Some((entry.key().clone(), count as usize))
                    } else {
                        None
                    }
                })
                .collect();
        }

        let mut predictors = self.predictors.lock().unwrap_or_else(|e| e.into_inner());
        self.data
            .iter()
            .filter_map(|entry| {
                let lora_name = entry.key();
                let counter = &entry.value().rate_counter;

                let predictor = predictors
                    .entry(lora_name.clone())
                    .or_insert_with(|| Self::create_predictor(predictor_type, ema_alpha));

                predictor.update(counter, now);
                let load = predictor.predict();
                let load_rounded = load.round() as usize;

                if load_rounded > 0 {
                    Some((lora_name.clone(), load_rounded))
                } else {
                    None
                }
            })
            .collect()
    }

    fn create_predictor(predictor_type: PredictorType, ema_alpha: f64) -> Box<dyn LoadPredictor> {
        match predictor_type {
            PredictorType::None => unreachable!("should not create predictor for None type"),
            PredictorType::Ema => Box::new(EmaPredictor::new(ema_alpha)),
        }
    }

    /// Get raw arrival counts from the sliding-window rate counters.
    pub fn get_raw_arrival_counts(&self) -> HashMap<String, u64> {
        let now = Instant::now();
        self.data
            .iter()
            .filter_map(|entry| {
                let count = entry.value().rate_counter.count(now);
                if count > 0 {
                    Some((entry.key().clone(), count))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get instantaneous in-flight counts (for metrics).
    pub fn get_inflight_counts(&self) -> HashMap<String, usize> {
        self.data
            .iter()
            .filter_map(|entry| {
                let count = entry.value().active_count.load(Ordering::Relaxed);
                if count > 0 {
                    Some((entry.key().clone(), count))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for LoadEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucketed_rate_counter_basic() {
        let now = Instant::now();
        let counter = BucketedRateCounter::new(10, Duration::from_secs(1), now);

        counter.record(now);
        counter.record(now);
        counter.record(now);

        assert_eq!(counter.count(now), 3);
    }

    #[test]
    fn test_bucketed_rate_counter_expiry() {
        let start = Instant::now();
        let bucket_duration = Duration::from_secs(1);
        let counter = BucketedRateCounter::new(5, bucket_duration, start);

        counter.record(start);
        counter.record(start);

        let t2 = start + Duration::from_secs(2);
        counter.record(t2);

        assert_eq!(counter.count(t2), 3);

        let t6 = start + Duration::from_secs(6);
        assert_eq!(counter.count(t6), 1, "t=0 arrivals should have expired");

        let t8 = start + Duration::from_secs(8);
        assert_eq!(counter.count(t8), 0, "all arrivals should have expired");
    }

    #[test]
    fn test_increment_decrement_load() {
        let estimator = LoadEstimator::new();

        estimator.increment_load("lora-test");
        estimator.increment_load("lora-test");

        let load = estimator.get_current_load();
        assert_eq!(load.get("lora-test"), Some(&2));

        let inflight = estimator.get_inflight_counts();
        assert_eq!(inflight.get("lora-test"), Some(&2));

        estimator.decrement_load("lora-test");

        let inflight = estimator.get_inflight_counts();
        assert_eq!(inflight.get("lora-test"), Some(&1));

        // Windowed load still sees 2 arrivals
        let load = estimator.get_current_load();
        assert_eq!(load.get("lora-test"), Some(&2));
    }

    #[test]
    fn test_update_from_counts() {
        let estimator = LoadEstimator::new();

        let mut counts = HashMap::new();
        counts.insert("lora-math".to_string(), 5);
        counts.insert("lora-code".to_string(), 3);

        estimator.update_from_counts(counts);

        let load = estimator.get_current_load();
        assert_eq!(load.get("lora-math"), Some(&5));
        assert_eq!(load.get("lora-code"), Some(&3));
    }

    #[test]
    fn test_load_with_ema_predictor() {
        let config = LoadEstimatorConfig {
            predictor_type: PredictorType::Ema,
            ema_alpha: 1.0,
            ..Default::default()
        };
        let estimator = LoadEstimator::with_config(config);

        estimator.increment_load("lora-test");
        estimator.increment_load("lora-test");
        estimator.increment_load("lora-test");

        let load = estimator.get_current_load();
        assert_eq!(
            load.get("lora-test"),
            Some(&3),
            "EMA with alpha=1.0 should match raw count"
        );
    }
}
