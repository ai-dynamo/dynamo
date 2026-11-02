// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Frontend's Prometheus metrics, served at `/metrics`.
//!
//! Series are labelled with the pool's name and region, never with anything
//! from the request body: a request's `model` is whatever the client sent,
//! and would mint a series per string. Pool frontends report what needs the
//! stream parsed, such as inter-token latency and token counts.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use axum::http::StatusCode;
use dynamo_global_router::affinity::{AffinityStats, CacheStats};
use dynamo_global_router::{InFlightTracker, Pool};
use dynamo_llm::http::service::metrics::generate_log_buckets;
use dynamo_runtime::metrics::prometheus_names::clamp_u64_to_i64;
use prometheus::{
    Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGaugeVec, Opts, Registry,
    TextEncoder,
};

use crate::PoolSnapshot;

const PREFIX: &str = "dynamo_global_frontend";
const POOL_LABELS: [&str; 2] = ["pool", "region"];
/// `requests_total`'s status for a request the pool sent no response to.
const NO_RESPONSE: &str = "no_response";
/// `requests_total`'s status for a request whose client left before the
/// pool's response headers arrived.
const CANCELLED: &str = "cancelled";

/// The Global Frontend's metrics, in a registry of their own.
#[derive(Debug)]
pub struct Metrics {
    registry: Registry,
    /// This process's region, for telling requests kept in it from the rest.
    region: Option<String>,
    tracker: Arc<InFlightTracker>,
    requests: IntCounterVec,
    rejected: IntCounterVec,
    response_errors: IntCounterVec,
    time_to_first_byte: HistogramVec,
    duration: HistogramVec,
    in_flight: IntGaugeVec,
    /// Present only with `--router-policy affinity`.
    affinity: Option<AffinityCounters>,
    /// A scrape resets and refills the gauges, so scrapes must not overlap.
    scrape: Mutex<()>,
}

/// Where one exported affinity series reads its cumulative count.
type Series = (&'static str, fn(&AffinityCounters) -> &AtomicU64);

/// How a request's affinity work ended. `skipped` and `timeout` are the
/// degraded path: `skipped` means the breaker was open and storage was never
/// consulted, `timeout` that the deadline ran out first.
const LOOKUPS: [Series; 7] = [
    ("hit", |counters| &counters.stats.hits),
    ("miss", |counters| &counters.stats.misses),
    ("conflict", |counters| &counters.stats.conflicts),
    ("error", |counters| &counters.stats.failures),
    ("timeout", |counters| &counters.stats.timeouts),
    ("skipped", |counters| &counters.stats.skipped),
    ("repair", |counters| &counters.stats.repairs),
];

/// Whether a lookup reached the backing store.
const CACHE: [Series; 3] = [
    ("hit", |counters| &counters.cache_stats.hits),
    ("miss", |counters| &counters.cache_stats.misses),
    ("invalidation", |counters| {
        &counters.cache_stats.invalidations
    }),
];

/// The affinity path keeps plain cumulative counters, which a scrape turns
/// into Prometheus counters by adding what has arrived since the last one.
#[derive(Debug)]
struct AffinityCounters {
    stats: Arc<AffinityStats>,
    cache_stats: Arc<CacheStats>,
    lookups: IntCounterVec,
    cache: IntCounterVec,
    /// What each series in `LOOKUPS` then `CACHE` last reported.
    reported: Mutex<Vec<u64>>,
}

impl AffinityCounters {
    /// Adds each series' arrivals since the last call. The caller holds the
    /// scrape lock, so no count is reported twice.
    fn drain(&self) {
        let mut reported = self.reported.lock().unwrap_or_else(PoisonError::into_inner);
        let series = LOOKUPS
            .iter()
            .map(|series| (&self.lookups, series))
            .chain(CACHE.iter().map(|series| (&self.cache, series)));
        for (index, (family, (outcome, count))) in series.enumerate() {
            let current = count(self).load(Ordering::Relaxed);
            // Saturating: these only ever grow, but a counter must never be
            // handed a negative delta.
            let arrived = current.saturating_sub(reported[index]);
            if arrived > 0 {
                family.with_label_values(&[outcome]).inc_by(arrived);
                reported[index] = current;
            }
        }
    }
}

impl Metrics {
    /// `region` is the region this process runs in, named as pools name
    /// theirs, which tells requests kept in it from the rest. Without it,
    /// their `local` label is `unknown`. `tracker` is the tracker the Global
    /// Frontend counts its requests in.
    pub fn new(region: Option<String>, tracker: Arc<InFlightTracker>) -> Self {
        let registry = Registry::new();
        let requests = register(
            &registry,
            IntCounterVec::new(
                Opts::new(
                    name("requests_total"),
                    "Requests forwarded to each pool, by the pool's response status, \
                     no_response, or cancelled if the client left first",
                ),
                &["pool", "region", "local", "status"],
            ),
        );
        let rejected = register(
            &registry,
            IntCounterVec::new(
                Opts::new(
                    name("rejected_requests_total"),
                    "Requests answered without a pool, by error type",
                ),
                &["error_type"],
            ),
        );
        let response_errors = register(
            &registry,
            IntCounterVec::new(
                Opts::new(
                    name("response_errors_total"),
                    "Successful responses whose body broke off before it ended",
                ),
                &POOL_LABELS,
            ),
        );
        // The buckets are the Dynamo frontend's defaults for time to first
        // token and request duration, so the two can share a dashboard.
        let time_to_first_byte = register(
            &registry,
            HistogramVec::new(
                HistogramOpts::new(
                    name("time_to_first_byte_seconds"),
                    "Time from receiving a request to the first body chunk of its \
                     pool's successful response",
                )
                .buckets(generate_log_buckets(0.001, 480.0, 18)),
                &POOL_LABELS,
            ),
        );
        let duration = register(
            &registry,
            HistogramVec::new(
                HistogramOpts::new(
                    name("request_duration_seconds"),
                    "Time from receiving a request to the end of its pool's successful \
                     response, or to the client leaving",
                )
                .buckets(generate_log_buckets(1.0, 512.0, 10)),
                &POOL_LABELS,
            ),
        );
        let in_flight = register(
            &registry,
            IntGaugeVec::new(
                Opts::new(
                    name("inflight_requests"),
                    "Requests this process has sent to each pool and not yet finished",
                ),
                &POOL_LABELS,
            ),
        );
        Self {
            registry,
            region,
            tracker,
            requests,
            rejected,
            response_errors,
            time_to_first_byte,
            duration,
            in_flight,
            affinity: None,
            scrape: Mutex::new(()),
        }
    }

    /// Exports the affinity path's counters, including the degraded path:
    /// requests that lost affinity to a storage error, to the deadline, or to
    /// an open breaker, and how much of the lookup load the cache absorbed.
    pub fn with_affinity(
        mut self,
        stats: Arc<AffinityStats>,
        cache_stats: Arc<CacheStats>,
    ) -> Self {
        let lookups = register(
            &self.registry,
            IntCounterVec::new(
                Opts::new(
                    name("affinity_lookups_total"),
                    "Session affinity decisions, by how the request's affinity work ended",
                ),
                &["outcome"],
            ),
        );
        let cache = register(
            &self.registry,
            IntCounterVec::new(
                Opts::new(
                    name("affinity_cache_total"),
                    "Session affinity store reads, by whether the in-memory cache answered them",
                ),
                &["outcome"],
            ),
        );
        self.affinity = Some(AffinityCounters {
            stats,
            cache_stats,
            lookups,
            cache,
            reported: Mutex::new(vec![0; LOOKUPS.len() + CACHE.len()]),
        });
        self
    }

    /// Counts a request answered without a pool. `error_type` is one of
    /// `prometheus_names::frontend_service::error_type`.
    pub(crate) fn rejected(&self, error_type: &str) {
        self.rejected.with_label_values(&[error_type]).inc();
    }

    /// Counts a request being sent to `pool` once [`Sending::answered`]
    /// gives the pool's answer, or as cancelled if it drops first: the
    /// client left before the pool's response headers arrived.
    pub(crate) fn sending<'a>(&'a self, pool: &'a Pool) -> Sending<'a> {
        Sending {
            metrics: self,
            pool,
            counted: false,
        }
    }

    fn count(&self, pool: &Pool, status: &str) {
        let local = match &self.region {
            None => "unknown",
            Some(region) if *region == pool.location.region => "true",
            Some(_) => "false",
        };
        self.requests
            .with_label_values(&[&pool.name, &pool.location.region, local, status])
            .inc();
    }

    /// Times a successful response from `pool` to a request received at
    /// `received`.
    pub(crate) fn response(&self, pool: &Pool, received: Instant) -> ResponseMetrics {
        let labels = [pool.name.as_str(), pool.location.region.as_str()];
        ResponseMetrics {
            received,
            first_byte: Some(self.time_to_first_byte.with_label_values(&labels)),
            duration: self.duration.with_label_values(&labels),
            errors: self.response_errors.with_label_values(&labels),
        }
    }

    /// Every family in Prometheus's text format, with each pool in `pools`
    /// read into the gauges now.
    pub(crate) fn render(&self, pools: &PoolSnapshot) -> prometheus::Result<String> {
        let _scrape = self.scrape.lock().unwrap_or_else(PoisonError::into_inner);
        // Pools that left the catalog lose their series.
        self.in_flight.reset();
        for pool in pools.pools() {
            self.in_flight
                .with_label_values(&[&pool.name, &pool.location.region])
                .set(clamp_u64_to_i64(self.tracker.get(&pool.id)));
        }
        if let Some(affinity) = &self.affinity {
            affinity.drain();
        }
        TextEncoder::new().encode_to_string(&self.registry.gather())
    }
}

fn name(suffix: &str) -> String {
    format!("{PREFIX}_{suffix}")
}

fn register<M>(registry: &Registry, metric: prometheus::Result<M>) -> M
where
    M: prometheus::core::Collector + Clone + 'static,
{
    let metric = metric.expect("metric options are statically valid");
    registry
        .register(Box::new(metric.clone()))
        .expect("each family is registered once");
    metric
}

/// A request on its way to a pool, counted once (see [`Metrics::sending`]).
#[derive(Debug)]
pub(crate) struct Sending<'a> {
    metrics: &'a Metrics,
    pool: &'a Pool,
    counted: bool,
}

impl Sending<'_> {
    /// Counts the pool's response status, or `None` if it sent no response.
    pub(crate) fn answered(mut self, status: Option<StatusCode>) {
        let status = status.as_ref().map_or(NO_RESPONSE, StatusCode::as_str);
        self.metrics.count(self.pool, status);
        self.counted = true;
    }
}

impl Drop for Sending<'_> {
    fn drop(&mut self) {
        if !self.counted {
            self.metrics.count(self.pool, CANCELLED);
        }
    }
}

/// One successful response's timing: the first chunk is its time to first
/// byte, a failed chunk counts as an error, and dropping it, when the
/// response ends or the client leaves, is its duration.
#[derive(Debug)]
pub(crate) struct ResponseMetrics {
    received: Instant,
    first_byte: Option<Histogram>,
    duration: Histogram,
    errors: IntCounter,
}

impl ResponseMetrics {
    pub(crate) fn chunk<T, E>(&mut self, chunk: &Result<T, E>) {
        match chunk {
            Ok(_) => {
                if let Some(first_byte) = self.first_byte.take() {
                    first_byte.observe(self.received.elapsed().as_secs_f64());
                }
            }
            Err(_) => self.errors.inc(),
        }
    }
}

impl Drop for ResponseMetrics {
    fn drop(&mut self) {
        self.duration.observe(self.received.elapsed().as_secs_f64());
    }
}
