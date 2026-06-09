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

/// Sentinel epoch value indicating a bucket is mid-rotation. Acts as a
/// transient lock: fast-path adders spin until the rotator publishes the new
/// epoch, and readers skip the bucket until then.
const BUCKET_ROTATING: u64 = u64::MAX;

/// Upper bound on the per-LoRA sliding-window bucket count. Caps the bucket
/// vector allocation (~16 bytes/bucket) so a pathological rate-window or
/// bucket-rate config cannot OOM. 1M buckets ≈ 16 MiB.
const MAX_BUCKETS: u64 = 1_000_000;

/// Lock-free, epoch-based sliding-window rate counter.
///
/// Divides time into fixed-duration buckets. Each bucket has an atomic counter
/// and an epoch (the absolute bucket index it was last used for). Stale buckets
/// are detected by epoch mismatch and lazily reset via a CAS-into-sentinel
/// protocol that prevents concurrent fast-path additions from being lost.
pub struct BucketedRateCounter {
    buckets: Vec<AtomicU64>,
    epochs: Vec<AtomicU64>,
    epoch_start: Instant,
    bucket_duration: Duration,
    num_buckets: usize,
    /// Absolute bucket index of the most recent recorded arrival, or `u64::MAX` if none recorded
    /// since creation/clear. Read by [`Self::has_recent_arrival`] for a rotation-safe "any arrival
    /// in the window" check — unlike [`Self::count`], it never transiently reads 0 while a bucket
    /// is mid-rotation, so it cannot drop a just-recorded short-lived arrival.
    last_arrival_bucket: AtomicU64,
}

// All fields (Vec<AtomicU64>, Instant, Duration, usize) are Sync, so Sync is
// auto-derived; no explicit unsafe impl needed.

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
            last_arrival_bucket: AtomicU64::new(u64::MAX),
        }
    }

    /// Record a single arrival at time `now`. Lock-free.
    pub fn record(&self, now: Instant) {
        self.record_count(1, now);
    }

    /// Record `n` arrivals at time `now`. Lock-free (spins only during the
    /// narrow rotation window when crossing a bucket boundary).
    ///
    /// Rotation protocol: when a stale epoch is observed, the writer CASes the
    /// epoch to the `BUCKET_ROTATING` sentinel, resets the bucket to its own
    /// contribution, and publishes the new epoch. Concurrent threads either
    /// take the fast path against the new epoch (preserving their adds) or
    /// observe the sentinel and spin until publish completes — so no fast-path
    /// add can be silently overwritten by the rotation.
    pub fn record_count(&self, n: u64, now: Instant) {
        if n == 0 {
            return;
        }
        let elapsed = now.duration_since(self.epoch_start);
        let global_bucket = (elapsed.as_nanos() / self.bucket_duration.as_nanos()) as u64;
        let index = (global_bucket as usize) % self.num_buckets;

        // Record the latest-arrival bucket for the rotation-safe recent-arrival check. A rare stale
        // record (an out-of-order older `now`) may store a slightly smaller value, but it is still
        // within ~1 bucket of the true latest and far inside the window, so retain_known's decision
        // is unaffected.
        self.last_arrival_bucket
            .store(global_bucket, Ordering::Relaxed);

        loop {
            let current_epoch = self.epochs[index].load(Ordering::Acquire);
            if current_epoch == global_bucket {
                // Fast path: bucket already belongs to our epoch.
                self.buckets[index].fetch_add(n, Ordering::Relaxed);
                return;
            }
            if current_epoch == BUCKET_ROTATING {
                // Another thread is rotating this bucket; wait for publish.
                std::hint::spin_loop();
                continue;
            }
            if current_epoch > global_bucket {
                // A newer epoch already owns this slot; drop the stale record.
                return;
            }
            // current_epoch < global_bucket: try to claim the rotation.
            if self.epochs[index]
                .compare_exchange(
                    current_epoch,
                    BUCKET_ROTATING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                // We own the bucket. Reset to our contribution, then publish
                // the new epoch. Concurrent readers/writers see ROTATING in
                // between and either spin or skip, never a stale value.
                self.buckets[index].store(n, Ordering::Release);
                self.epochs[index].store(global_bucket, Ordering::Release);
                return;
            }
            // Lost the CAS; another rotator won. Loop to observe new state.
        }
    }

    /// Count total arrivals within the sliding window ending at `now`.
    pub fn count(&self, now: Instant) -> u64 {
        let elapsed = now.duration_since(self.epoch_start);
        let global_bucket = (elapsed.as_nanos() / self.bucket_duration.as_nanos()) as u64;

        let mut total = 0u64;
        let min_valid_epoch = global_bucket.saturating_sub(self.num_buckets as u64 - 1);
        for i in 0..self.num_buckets {
            let epoch = self.epochs[i].load(Ordering::Acquire);
            // Skip buckets that are mid-rotation; their value will be visible
            // on the next read once the rotator publishes the new epoch.
            if epoch == BUCKET_ROTATING {
                continue;
            }
            if epoch >= min_valid_epoch && epoch <= global_bucket {
                total += self.buckets[i].load(Ordering::Relaxed);
            }
        }
        total
    }

    /// Whether any arrival was recorded within the sliding window ending at `now`.
    ///
    /// Rotation-safe single-atomic read (no per-bucket scan): checks the most-recent-arrival
    /// bucket against the window, so unlike [`Self::count`] it cannot transiently miss an arrival
    /// while a bucket is mid-rotation. Used by `retain_known` to avoid pruning a short-lived
    /// new-adapter's load signal during the bucket-rotation window.
    pub fn has_recent_arrival(&self, now: Instant) -> bool {
        let last = self.last_arrival_bucket.load(Ordering::Relaxed);
        if last == u64::MAX {
            return false; // nothing recorded since creation/clear
        }
        let elapsed = now.duration_since(self.epoch_start);
        let global_bucket = (elapsed.as_nanos() / self.bucket_duration.as_nanos()) as u64;
        // saturating_sub handles a last bucket slightly ahead of `global_bucket` (treated as just
        // now → recent); a last bucket older than the window yields a difference >= num_buckets.
        global_bucket.saturating_sub(last) < self.num_buckets as u64
    }

    pub fn clear(&self) {
        for i in 0..self.num_buckets {
            self.buckets[i].store(0, Ordering::Release);
            self.epochs[i].store(0, Ordering::Release);
        }
        // No arrivals remain in the window after a clear.
        self.last_arrival_bucket.store(u64::MAX, Ordering::Release);
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
            // saturating_mul: both operands are operator-supplied and could
            // otherwise overflow u64.
            rate_window: Duration::from_secs(
                timestep_secs.saturating_mul(multiplier).max(min_window),
            ),
            ..Default::default()
        }
    }

    fn num_buckets(&self) -> usize {
        let secs = self.rate_window.as_secs().max(1);
        // saturating_mul guards against overflow; the clamp then bounds the
        // bucket-vector allocation so a pathological (operator-supplied) window
        // or bucket rate cannot OOM/panic when a counter is created. At 16 bytes
        // per bucket (count + epoch atomics) the cap is ~16 MiB per LoRA.
        secs.saturating_mul(self.buckets_per_second)
            .clamp(1, MAX_BUCKETS) as usize
    }

    fn bucket_duration(&self) -> Duration {
        // Spread exactly `num_buckets` buckets across the configured rate
        // window. Deriving the duration from the (possibly clamped) bucket
        // count keeps `num_buckets * bucket_duration == rate_window`, so
        // clamping the count lengthens each bucket rather than silently
        // shrinking the retained window. For unclamped configs this equals the
        // requested 1s / buckets_per_second.
        let buckets = self.num_buckets() as u128; // num_buckets() >= 1
        let window_nanos = self.rate_window.as_nanos().max(1);
        let per = (window_nanos / buckets).clamp(1, u64::MAX as u128) as u64;
        Duration::from_nanos(per)
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

    /// Replace the full estimator config at runtime (rate window, bucket granularity,
    /// predictor type/alpha).
    ///
    /// Rebuilds EXISTING per-LoRA counters under the new bucket geometry when `rate_window` or
    /// `buckets_per_second` changes, and clears predictors when the geometry or the predictor
    /// type/alpha changes. The load-feed path can create counters BEFORE the controller applies
    /// its config (e.g. a KV active-sequence event arriving before `start_lora_controller` runs),
    /// so without this rebuild those early counters would keep the default bucketing forever.
    /// `active_count` (in-flight requests) is preserved; the windowed arrival history is restarted
    /// because it was measured against the old window. (Mirrors `set_rate_window`, but covers the
    /// full config.)
    pub fn set_config(&self, config: LoadEstimatorConfig) {
        let mut cfg = self.config.write();
        let old = cfg.clone();
        let geometry_changed = old.rate_window != config.rate_window
            || old.buckets_per_second != config.buckets_per_second;
        let predictor_changed = old.predictor_type != config.predictor_type
            || (old.ema_alpha - config.ema_alpha).abs() > f64::EPSILON;
        *cfg = config;
        let num_buckets = cfg.num_buckets();
        let bucket_duration = cfg.bucket_duration();
        drop(cfg);

        if geometry_changed {
            let now = Instant::now();
            for mut entry in self.data.iter_mut() {
                let old_active = entry.value().active_count.load(Ordering::Relaxed);
                *entry.value_mut() = LoraLoadData {
                    active_count: AtomicUsize::new(old_active),
                    rate_counter: BucketedRateCounter::new(num_buckets, bucket_duration, now),
                };
            }
        }
        // Predictors built under the old window/params are meaningless once the geometry or the
        // predictor type/alpha changes; clear them so smoothing restarts from scratch.
        if geometry_changed || predictor_changed {
            self.predictors
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
        }

        tracing::info!(
            num_buckets,
            geometry_changed,
            predictor_changed,
            "LoadEstimator config updated"
        );
    }

    /// Prune tracking data (and predictors) for any LoRA not in `known`. Bounds memory
    /// against unloaded adapters and unknown/typo request names that never get allocated.
    pub fn retain_known(&self, known: &std::collections::HashSet<&str>) {
        // Keep an entry if it is known, OR still has in-flight requests, OR has nonzero arrivals
        // within the current rate window. The in-flight check protects a LoRA whose arrival raced
        // this controller tick before its MDC reached the state tracker (dropping it would make
        // the matching LoadGuard / Free-event decrement a silent no-op). The recent-arrival check
        // additionally protects a SHORT request for a newly seen adapter that already completed
        // (active_count back to 0) before discovery reported the adapter: its arrival history must
        // survive at least one rate window so the controller still sees the load signal and can
        // allocate for it, instead of pruning the demand the instant the request finishes. The
        // recent-arrival check uses the rotation-safe `has_recent_arrival` (a single-atomic
        // last-arrival-bucket read), NOT `count`, so a bucket mid-rotation cannot transiently read
        // zero and drop the very signal this protects. Once the window slides past with no further
        // arrivals and the name is still unknown, it is pruned, so memory stays bounded against
        // unknown/typo request names.
        let now = Instant::now();
        self.data.retain(|name, data| {
            known.contains(name.as_str())
                || data.active_count.load(Ordering::Relaxed) > 0
                || data.rate_counter.has_recent_arrival(now)
        });
        self.predictors
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|name, _| known.contains(name.as_str()) || self.data.contains_key(name));
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
            // Window geometry changed; EMA estimates built over the old window
            // are meaningless — clear them so smoothing restarts from scratch.
            self.predictors
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
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
        // Remove the predictor entry so the next get_current_load() call starts
        // fresh rather than carrying the pre-clear EMA estimate forward.
        self.predictors
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(lora_name);
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
        let cancel_token = component.drt().child_token();
        tokio::spawn(async move {
            // Durable feed: reconnect on transient errors / stream end with capped backoff,
            // stopping only on cancellation. A failed subscribe must not silently disable KV
            // load tracking for the lifetime of the process.
            let mut backoff = Duration::from_secs(1);
            while !cancel_token.is_cancelled() {
                match self.subscribe_to_events(&component, &cancel_token).await {
                    Ok(()) => break, // cancelled cleanly
                    Err(e) => {
                        tracing::warn!(
                            "LORA load event subscription error: {e}; reconnecting in {backoff:?}"
                        );
                        tokio::select! {
                            _ = cancel_token.cancelled() => break,
                            _ = tokio::time::sleep(backoff) => {}
                        }
                        backoff = (backoff * 2).min(Duration::from_secs(30));
                    }
                }
            }
            tracing::debug!("LORA load event subscription task exiting");
        })
    }

    async fn subscribe_to_events(
        &self,
        component: &Component,
        cancel_token: &tokio_util::sync::CancellationToken,
    ) -> anyhow::Result<()> {
        let mut subscriber = EventSubscriber::for_component(component, ACTIVE_SEQUENCES_SUBJECT)
            .await?
            .typed::<ActiveSequenceEvent>();

        tracing::info!("Started LORA load event subscription");

        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => return Ok(()),
                result = subscriber.next() => {
                    match result {
                        Some(Ok((_envelope, event))) => {
                            self.handle_event(event);
                        }
                        Some(Err(e)) => {
                            tracing::warn!("Error receiving LORA load event: {}", e);
                        }
                        None => anyhow::bail!("LORA load event stream ended"),
                    }
                }
            }
        }
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
    ///
    /// Uses `saturating_sub` to guard against underflow: a duplicate, delayed,
    /// or out-of-order `Free` event when `active_count == 0` is silently
    /// ignored rather than wrapping to `usize::MAX`.
    pub fn decrement_load(&self, lora_name: &str) {
        if let Some(entry) = self.data.get(lora_name) {
            entry
                .value()
                .active_count
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    Some(v.saturating_sub(1))
                })
                .ok();
        }
    }

    /// Remove all tracking data for a LoRA. After this call the LoRA will no
    /// longer appear in [`get_current_load`] results. Useful when a LoRA is
    /// permanently unloaded and its stale rate-counter / predictor entries
    /// should be purged.
    pub fn remove_lora(&self, lora_name: &str) {
        self.data.remove(lora_name);
        if let Ok(mut predictors) = self.predictors.lock() {
            predictors.remove(lora_name);
        }
    }

    /// Update active counts from a polled snapshot.
    ///
    /// **Polling-mode caveat**: arrivals are approximated as the per-poll
    /// delta `max(0, current - prev)`, since worker snapshots do not expose
    /// request-start events. This is a *lower bound* on real arrivals:
    /// in-interval churn (e.g., 10 requests finishing while 10 new ones start
    /// — net delta 0) is invisible, and sub-interval oscillation is lost.
    /// Event-based mode (`handle_event` / `increment_load`) gives accurate
    /// arrival rates; prefer it when arrival precision matters.
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
                    let prev = data.active_count.load(Ordering::Relaxed);
                    data.active_count.store(*count, Ordering::Relaxed);
                    // Record only the delta (new arrivals since the last poll) to
                    // avoid double-counting sustained requests.  A request active
                    // for N ticks should contribute 1 arrival, not N.
                    let arrivals = count.saturating_sub(prev) as u64;
                    if arrivals > 0 {
                        data.rate_counter.record_count(arrivals, now);
                    }
                })
                .or_insert_with(|| {
                    let counter = BucketedRateCounter::new(num_buckets, bucket_duration, now);
                    // First observation: the entire count represents new arrivals.
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
        let result = self
            .data
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
            .collect();
        // Prune predictors for LoRAs that are no longer in self.data to prevent
        // unbounded growth and stale EMA reuse when names are recycled.
        predictors.retain(|name, _| self.data.contains_key(name));
        result
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
    fn set_config_rebuilds_existing_counter_geometry() {
        // The load-feed path can create a counter before the controller applies its config, so
        // set_config must rebuild EXISTING counters under the new bucket geometry (preserving the
        // in-flight active_count), not only counters created afterward.
        let est = LoadEstimator::with_config(LoadEstimatorConfig {
            rate_window: Duration::from_secs(10),
            buckets_per_second: 1,
            ..Default::default()
        });
        // Counter for "lora-a" created under the OLD geometry (10s @ 1/s = 10 buckets), with one
        // in-flight request.
        est.increment_load("lora-a");
        assert_eq!(est.data.get("lora-a").unwrap().rate_counter.num_buckets, 10);

        // Apply a finer bucket geometry at runtime.
        est.set_config(LoadEstimatorConfig {
            rate_window: Duration::from_secs(10),
            buckets_per_second: 4,
            ..Default::default()
        });

        let entry = est.data.get("lora-a").unwrap();
        assert_eq!(
            entry.rate_counter.num_buckets, 40,
            "existing counter must adopt the new geometry (10s @ 4/s = 40 buckets)"
        );
        assert_eq!(
            entry.active_count.load(Ordering::Relaxed),
            1,
            "in-flight active_count must survive the rebuild"
        );
    }

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
    fn test_decrement_load_saturates_at_zero() {
        let estimator = LoadEstimator::new();

        // Decrementing a never-seen LoRA is a no-op (data entry doesn't exist).
        estimator.decrement_load("never-seen");
        assert!(!estimator.get_inflight_counts().contains_key("never-seen"));

        // Over-decrement an existing entry: pre-fix, this wrapped to usize::MAX.
        estimator.increment_load("lora-test");
        estimator.decrement_load("lora-test");
        estimator.decrement_load("lora-test"); // would wrap without saturating_sub
        estimator.decrement_load("lora-test");

        let inflight = estimator.get_inflight_counts();
        // active_count == 0 is filtered out of get_inflight_counts.
        assert!(
            !inflight.contains_key("lora-test"),
            "expected saturated zero (filtered out); got {:?}",
            inflight.get("lora-test")
        );
    }

    #[test]
    fn test_update_from_counts_records_arrival_deltas() {
        let estimator = LoadEstimator::new();

        // First poll: 3 active → record 3 arrivals.
        let mut counts = HashMap::new();
        counts.insert("lora-a".to_string(), 3);
        estimator.update_from_counts(counts);
        assert_eq!(estimator.get_raw_arrival_counts().get("lora-a"), Some(&3));

        // Second poll: still 3 active (sustained) → delta is 0, no new arrivals.
        let mut counts = HashMap::new();
        counts.insert("lora-a".to_string(), 3);
        estimator.update_from_counts(counts);
        assert_eq!(
            estimator.get_raw_arrival_counts().get("lora-a"),
            Some(&3),
            "sustained traffic must not double-count arrivals"
        );

        // Third poll: 5 active (grew by 2) → record 2 new arrivals.
        let mut counts = HashMap::new();
        counts.insert("lora-a".to_string(), 5);
        estimator.update_from_counts(counts);
        assert_eq!(estimator.get_raw_arrival_counts().get("lora-a"), Some(&5));

        // Fourth poll: 2 active (shrank by 3) → no arrivals recorded; window stays.
        let mut counts = HashMap::new();
        counts.insert("lora-a".to_string(), 2);
        estimator.update_from_counts(counts);
        assert_eq!(
            estimator.get_raw_arrival_counts().get("lora-a"),
            Some(&5),
            "decreases must not record arrivals"
        );

        // In-flight reflects the latest snapshot, not the rolling window.
        assert_eq!(estimator.get_inflight_counts().get("lora-a"), Some(&2));
    }

    #[test]
    fn test_bucket_rotation_concurrent_no_lost_updates() {
        use std::sync::Arc;
        use std::thread;

        // Geometry: 100us per bucket, large window so nothing expires during the
        // test. Threads contend simultaneously across many bucket boundaries.
        let start = Instant::now();
        let bucket_duration = Duration::from_micros(100);
        let num_buckets = 10_000usize;
        let counter = Arc::new(BucketedRateCounter::new(
            num_buckets,
            bucket_duration,
            start,
        ));

        let threads_n: usize = 8;
        let per_thread: usize = 1_000;
        let step_micros: u64 = 50; // two records per bucket, so every other i crosses a boundary

        let mut handles = Vec::with_capacity(threads_n);
        for _ in 0..threads_n {
            let counter = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                for i in 0..per_thread {
                    let offset = Duration::from_micros(i as u64 * step_micros);
                    counter.record(start + offset);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let final_time = start + Duration::from_micros(per_thread as u64 * step_micros);
        let total = counter.count(final_time);
        let expected = (threads_n * per_thread) as u64;
        assert_eq!(
            total, expected,
            "expected {expected} arrivals across concurrent bucket rotations, got {total} \
             — rotation protocol lost updates"
        );
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

    #[test]
    fn retain_known_keeps_recent_arrival_after_short_request() {
        // A short request for a newly seen adapter can arrive and complete (active_count back to 0)
        // before the adapter appears in discovery. retain_known must keep its recent-arrival
        // history so the controller still sees the load signal for at least one rate window,
        // instead of pruning the demand the instant the request finishes.
        let est = LoadEstimator::new();
        est.increment_load("new-adapter"); // request arrives
        est.decrement_load("new-adapter"); // request completes immediately
        assert!(
            est.get_inflight_counts().get("new-adapter").is_none(),
            "no in-flight requests remain after the short request completes"
        );

        // Discovery has not reported it yet, so it is not "known".
        let known: std::collections::HashSet<&str> = std::collections::HashSet::new();
        est.retain_known(&known);

        assert!(
            est.data.contains_key("new-adapter"),
            "a recent arrival must survive retain_known for at least one rate window"
        );
        assert!(
            est.get_raw_arrival_counts()
                .get("new-adapter")
                .copied()
                .unwrap_or(0)
                > 0,
            "the load signal (recent arrival) must be preserved"
        );
    }

    #[test]
    fn retain_known_prunes_unknown_with_no_recent_arrivals() {
        // Once the rate window has slid past the arrival (no recent arrivals) and the name is still
        // unknown, retain_known must prune it so memory stays bounded against typo/unknown names.
        let est = LoadEstimator::new();
        est.increment_load("typo-adapter");
        est.decrement_load("typo-adapter");
        // Simulate the rate window fully sliding past the arrival without sleeping.
        est.clear_rate_counter("typo-adapter");
        assert_eq!(
            est.get_raw_arrival_counts()
                .get("typo-adapter")
                .copied()
                .unwrap_or(0),
            0,
            "precondition: no recent arrivals remain"
        );

        let known: std::collections::HashSet<&str> = std::collections::HashSet::new();
        est.retain_known(&known);

        assert!(
            !est.data.contains_key("typo-adapter"),
            "an unknown name with no in-flight requests and no recent arrivals must be pruned"
        );
    }
}
