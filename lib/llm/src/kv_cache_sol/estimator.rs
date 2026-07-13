// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use prometheus::{IntCounterVec, IntGauge};
use tokio_util::sync::CancellationToken;

use super::{KV_CACHE_SOL_TOPIC, KvCacheSolDomain, KvCacheSolEvent, KvCacheSolEventKind, now_ms};

#[derive(Default)]
pub(super) struct DomainCache {
    pub(super) expirations: HashMap<u64, u64>,
    // Keep exactly one ordered expiry entry per retained hash. A binary heap
    // leaves one stale entry behind on every refresh, so a hot prefix can grow
    // memory independently of max_cache_blocks even though `expirations`
    // remains bounded.
    pub(super) expiry_queue: BTreeSet<(u64, u64)>,
}

impl DomainCache {
    pub(super) fn prune(&mut self, watermark_ms: u64) {
        while let Some((expires_at, hash)) = self.expiry_queue.first().copied() {
            if expires_at > watermark_ms {
                break;
            }
            self.expiry_queue.remove(&(expires_at, hash));
            self.expirations.remove(&hash);
        }
    }

    pub(super) fn insert(&mut self, hash: u64, expires_at: u64) -> bool {
        match self.expirations.get_mut(&hash) {
            Some(current_expiry) => {
                // Events from different producers can have slightly skewed
                // timestamps. Never shorten an already-known retention window.
                if expires_at > *current_expiry {
                    self.expiry_queue.remove(&(*current_expiry, hash));
                    *current_expiry = expires_at;
                    self.expiry_queue.insert((expires_at, hash));
                }
                false
            }
            None => {
                self.expirations.insert(hash, expires_at);
                self.expiry_queue.insert((expires_at, hash));
                true
            }
        }
    }
}

struct PendingRequest {
    domain: KvCacheSolDomain,
    prompt_tokens: u64,
    hit_tokens: u64,
    expires_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SolObservation {
    Start {
        domain: KvCacheSolDomain,
        prompt_tokens: u64,
        hit_tokens: u64,
    },
    End {
        domain: KvCacheSolDomain,
        prompt_tokens: Option<u64>,
        comparable_hit_tokens: Option<u64>,
        observed_cached_tokens: Option<u64>,
    },
}

pub(super) struct KvCacheSolCore {
    horizon_ms: u64,
    max_cache_blocks: usize,
    max_pending_requests: usize,
    watermark_ms: u64,
    pub(super) caches: HashMap<KvCacheSolDomain, DomainCache>,
    pending: HashMap<(u64, String), PendingRequest>,
    cache_blocks: usize,
    degraded_until_ms: u64,
}

impl KvCacheSolCore {
    pub(super) fn new(
        horizon: Duration,
        max_cache_blocks: usize,
        max_pending_requests: usize,
    ) -> Self {
        Self {
            horizon_ms: horizon.as_millis() as u64,
            max_cache_blocks,
            max_pending_requests,
            watermark_ms: 0,
            caches: HashMap::new(),
            pending: HashMap::new(),
            cache_blocks: 0,
            degraded_until_ms: 0,
        }
    }

    pub(super) fn advance_to(&mut self, occurred_at_ms: u64) {
        self.watermark_ms = self.watermark_ms.max(occurred_at_ms);
        let mut expired_blocks = 0;
        self.caches.retain(|_, cache| {
            let before = cache.expirations.len();
            cache.prune(self.watermark_ms);
            expired_blocks += before.saturating_sub(cache.expirations.len());
            !cache.expirations.is_empty()
        });
        self.cache_blocks = self.cache_blocks.saturating_sub(expired_blocks);
        self.pending
            .retain(|_, pending| pending.expires_at_ms > self.watermark_ms);
    }

    pub(super) fn apply(&mut self, publisher_id: u64, event: KvCacheSolEvent) -> SolObservation {
        self.advance_to(event.occurred_at_ms);
        let cache = self.caches.entry(event.domain.clone()).or_default();
        let expires_at = event.occurred_at_ms.saturating_add(self.horizon_ms);

        match event.kind {
            KvCacheSolEventKind::RequestStart {
                prompt_sequence_hashes,
                prompt_tokens,
            } => {
                let hit_blocks = prompt_sequence_hashes
                    .iter()
                    .take_while(|hash| cache.expirations.contains_key(hash))
                    .count() as u64;
                let hit_tokens = (hit_blocks * event.domain.block_size as u64).min(prompt_tokens);
                if expires_at > self.watermark_ms {
                    for hash in prompt_sequence_hashes {
                        if !cache.expirations.contains_key(&hash)
                            && self.cache_blocks >= self.max_cache_blocks
                        {
                            self.degraded_until_ms = self.degraded_until_ms.max(expires_at);
                            break;
                        }
                        if cache.insert(hash, expires_at) {
                            self.cache_blocks += 1;
                        }
                    }
                }
                if expires_at > self.watermark_ms {
                    if self.pending.len() >= self.max_pending_requests {
                        self.degraded_until_ms = self.degraded_until_ms.max(expires_at);
                    } else {
                        self.pending.insert(
                            (publisher_id, event.request_id),
                            PendingRequest {
                                domain: event.domain.clone(),
                                prompt_tokens,
                                hit_tokens,
                                expires_at_ms: expires_at,
                            },
                        );
                    }
                }
                SolObservation::Start {
                    domain: event.domain,
                    prompt_tokens,
                    hit_tokens,
                }
            }
            KvCacheSolEventKind::RequestEnd {
                continuation_sequence_hashes,
                observed_cached_tokens,
                ..
            } => {
                if expires_at > self.watermark_ms {
                    for hash in continuation_sequence_hashes.into_values().flatten() {
                        if !cache.expirations.contains_key(&hash)
                            && self.cache_blocks >= self.max_cache_blocks
                        {
                            self.degraded_until_ms = self.degraded_until_ms.max(expires_at);
                            break;
                        }
                        if cache.insert(hash, expires_at) {
                            self.cache_blocks += 1;
                        }
                    }
                }
                let pending = self.pending.remove(&(publisher_id, event.request_id));
                if pending.is_none() && expires_at > self.watermark_ms {
                    self.degraded_until_ms = self.degraded_until_ms.max(expires_at);
                }
                SolObservation::End {
                    domain: pending
                        .as_ref()
                        .map(|pending| pending.domain.clone())
                        .unwrap_or(event.domain),
                    prompt_tokens: pending.as_ref().map(|pending| pending.prompt_tokens),
                    comparable_hit_tokens: pending.as_ref().map(|pending| pending.hit_tokens),
                    observed_cached_tokens,
                }
            }
        }
    }

    pub(super) fn cache_blocks(&self) -> usize {
        self.cache_blocks
    }

    pub(super) fn pending_requests(&self) -> usize {
        self.pending.len()
    }

    pub(super) fn degraded(&self) -> bool {
        self.watermark_ms < self.degraded_until_ms
    }

    fn mark_degraded_at(&mut self, occurred_at_ms: u64) {
        self.watermark_ms = self.watermark_ms.max(occurred_at_ms);
        self.degraded_until_ms = self
            .degraded_until_ms
            .max(occurred_at_ms.saturating_add(self.horizon_ms));
    }

    fn horizon_ms(&self) -> u64 {
        self.horizon_ms
    }
}

pub(super) struct EstimatorMetrics {
    pub(super) requests_total: IntCounterVec,
    pub(super) prompt_tokens_total: IntCounterVec,
    hit_tokens_total: IntCounterVec,
    observed_prompt_tokens_total: IntCounterVec,
    observed_cached_tokens_total: IntCounterVec,
    comparable_hit_tokens_total: IntCounterVec,
    pub(super) events_total: IntCounterVec,
    events_dropped_total: IntCounterVec,
    cache_blocks: IntGauge,
    pub(super) pending_requests: IntGauge,
    lag_seconds: prometheus::Gauge,
    horizon_seconds: IntGauge,
    pub(super) degraded: IntGauge,
}

impl EstimatorMetrics {
    pub(super) fn new(component: &Component, horizon: Duration) -> Result<Self> {
        let metrics = component.metrics();
        let labels = &["model", "model_config"];
        let result = Self {
            requests_total: metrics.create_intcountervec(
                "kv_cache_sol_requests_total",
                "Requests evaluated by the KV-cache speed-of-light estimator",
                labels,
                &[],
            )?,
            prompt_tokens_total: metrics.create_intcountervec(
                "kv_cache_sol_prompt_tokens_total",
                "Prompt tokens evaluated by the KV-cache speed-of-light estimator",
                labels,
                &[],
            )?,
            hit_tokens_total: metrics.create_intcountervec(
                "kv_cache_sol_hit_tokens_total",
                "Theoretical prompt cache-hit tokens",
                labels,
                &[],
            )?,
            observed_prompt_tokens_total: metrics.create_intcountervec(
                "kv_cache_sol_observed_prompt_tokens_total",
                "Prompt tokens with backend-observed cache metadata",
                labels,
                &[],
            )?,
            observed_cached_tokens_total: metrics.create_intcountervec(
                "kv_cache_sol_observed_cached_tokens_total",
                "Backend-observed cached prompt tokens",
                labels,
                &[],
            )?,
            comparable_hit_tokens_total: metrics.create_intcountervec(
                "kv_cache_sol_comparable_hit_tokens_total",
                "Theoretical hit tokens for requests with backend observations",
                labels,
                &[],
            )?,
            events_total: metrics.create_intcountervec(
                "kv_cache_sol_events_total",
                "KV-cache speed-of-light events consumed by type",
                &["event_type"],
                &[],
            )?,
            events_dropped_total: metrics.create_intcountervec(
                "kv_cache_sol_events_dropped_total",
                "Detected missing or invalid KV-cache speed-of-light events",
                &["reason"],
                &[],
            )?,
            cache_blocks: metrics.create_intgauge(
                "kv_cache_sol_cache_blocks",
                "Sequence hashes retained by the KV-cache speed-of-light estimator",
                &[],
            )?,
            pending_requests: metrics.create_intgauge(
                "kv_cache_sol_pending_requests",
                "Requests waiting for a matching end event",
                &[],
            )?,
            lag_seconds: metrics.create_gauge(
                "kv_cache_sol_lag_seconds",
                "Event occurrence-to-consumption lag in seconds",
                &[],
            )?,
            horizon_seconds: metrics.create_intgauge(
                "kv_cache_sol_horizon_seconds",
                "Configured theoretical cache retention horizon",
                &[],
            )?,
            degraded: metrics.create_intgauge(
                "kv_cache_sol_degraded",
                "Whether event loss, lateness, or capacity limits invalidate the estimate",
                &[],
            )?,
        };
        result.horizon_seconds.set(horizon.as_secs() as i64);
        Ok(result)
    }
}

pub struct KvCacheSolEstimator {
    cancel: CancellationToken,
}

impl KvCacheSolEstimator {
    pub async fn start(
        component: Component,
        horizon: Duration,
        max_cache_blocks: usize,
        max_pending_requests: usize,
    ) -> Result<Self> {
        let subscriber = EventSubscriber::for_namespace(component.namespace(), KV_CACHE_SOL_TOPIC)
            .await?
            .typed::<KvCacheSolEvent>();
        let metrics = Arc::new(EstimatorMetrics::new(&component, horizon)?);
        let cancel = component.drt().child_token();
        let task_cancel = cancel.clone();
        let runtime = component.drt().runtime().secondary();
        runtime.spawn(run_estimator(
            subscriber,
            KvCacheSolCore::new(horizon, max_cache_blocks, max_pending_requests),
            metrics,
            task_cancel,
        ));
        Ok(Self { cancel })
    }

    pub fn shutdown(&self) {
        self.cancel.cancel();
    }
}

impl Drop for KvCacheSolEstimator {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

pub(super) async fn run_estimator(
    mut subscriber: dynamo_runtime::transports::event_plane::TypedEventSubscriber<KvCacheSolEvent>,
    mut core: KvCacheSolCore,
    metrics: Arc<EstimatorMetrics>,
    cancel: CancellationToken,
) {
    let mut last_sequence: HashMap<u64, (u64, u64)> = HashMap::new();
    let mut maintenance = tokio::time::interval(Duration::from_secs(1));
    maintenance.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        let next = tokio::select! {
            _ = cancel.cancelled() => break,
            _ = maintenance.tick() => {
                let watermark_ms = now_ms();
                core.advance_to(watermark_ms);
                last_sequence.retain(|_, (_, last_seen_ms)| {
                    last_seen_ms.saturating_add(core.horizon_ms()) > watermark_ms
                });
                metrics.cache_blocks.set(core.cache_blocks() as i64);
                metrics.pending_requests.set(core.pending_requests() as i64);
                metrics.degraded.set(i64::from(core.degraded()));
                continue;
            }
            next = subscriber.next() => next,
        };
        let Some(result) = next else { break };
        let (envelope, event) = match result {
            Ok(value) => value,
            Err(error) => {
                metrics
                    .events_dropped_total
                    .with_label_values(&["decode_error"])
                    .inc();
                core.mark_degraded_at(now_ms());
                metrics.degraded.set(i64::from(core.degraded()));
                tracing::warn!(%error, "invalid KV-cache speed-of-light event");
                continue;
            }
        };
        if event.domain.block_size == 0 {
            metrics
                .events_dropped_total
                .with_label_values(&["invalid_domain"])
                .inc();
            core.mark_degraded_at(event.occurred_at_ms);
            metrics.degraded.set(i64::from(core.degraded()));
            continue;
        }
        last_sequence.retain(|_, (_, last_seen_ms)| {
            last_seen_ms.saturating_add(core.horizon_ms()) > event.occurred_at_ms
        });
        let sequence_gap = match last_sequence.insert(
            envelope.publisher_id,
            (event.producer_sequence, event.occurred_at_ms),
        ) {
            Some((previous, _)) => event.producer_sequence != previous.saturating_add(1),
            None => event.producer_sequence != 0,
        };
        if sequence_gap {
            metrics
                .events_dropped_total
                .with_label_values(&["sequence_gap"])
                .inc();
            core.mark_degraded_at(event.occurred_at_ms);
        }
        metrics
            .lag_seconds
            .set(now_ms().saturating_sub(event.occurred_at_ms) as f64 / 1000.0);
        let event_type = match event.kind {
            KvCacheSolEventKind::RequestStart { .. } => "request_start",
            KvCacheSolEventKind::RequestEnd { .. } => "request_end",
        };
        metrics.events_total.with_label_values(&[event_type]).inc();
        match core.apply(envelope.publisher_id, event) {
            SolObservation::Start {
                domain,
                prompt_tokens,
                hit_tokens,
            } => {
                let labels = &[domain.model.as_str(), domain.model_config.as_str()];
                metrics.requests_total.with_label_values(labels).inc();
                metrics
                    .prompt_tokens_total
                    .with_label_values(labels)
                    .inc_by(prompt_tokens);
                metrics
                    .hit_tokens_total
                    .with_label_values(labels)
                    .inc_by(hit_tokens);
            }
            SolObservation::End {
                domain,
                prompt_tokens,
                comparable_hit_tokens,
                observed_cached_tokens,
            } => {
                if let (Some(prompt), Some(hit), Some(observed)) =
                    (prompt_tokens, comparable_hit_tokens, observed_cached_tokens)
                {
                    let labels = &[domain.model.as_str(), domain.model_config.as_str()];
                    metrics
                        .observed_prompt_tokens_total
                        .with_label_values(labels)
                        .inc_by(prompt);
                    metrics
                        .comparable_hit_tokens_total
                        .with_label_values(labels)
                        .inc_by(hit);
                    metrics
                        .observed_cached_tokens_total
                        .with_label_values(labels)
                        .inc_by(observed.min(prompt));
                }
            }
        }
        metrics.cache_blocks.set(core.cache_blocks() as i64);
        metrics.pending_requests.set(core.pending_requests() as i64);
        metrics.degraded.set(i64::from(core.degraded()));
    }
}
