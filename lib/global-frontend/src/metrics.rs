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

/// Why a request left a pool for another, as `failovers_total` labels it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FailoverReason {
    /// The pool sent no response: it refused the connection, did not accept
    /// it in time, or failed before its response headers.
    NoResponse,
    /// The pool answered 503.
    Unavailable,
    /// The pool answered the overload status (`DYN_HTTP_OVERLOAD_STATUS_CODE`).
    Overloaded,
}

impl FailoverReason {
    const ALL: [Self; 3] = [Self::NoResponse, Self::Unavailable, Self::Overloaded];

    fn label(self) -> &'static str {
        match self {
            Self::NoResponse => NO_RESPONSE,
            Self::Unavailable => "unavailable",
            Self::Overloaded => "overloaded",
        }
    }
}

/// The Global Frontend's metrics, in a registry of their own.
#[derive(Debug)]
pub struct Metrics {
    registry: Registry,
    /// This process's region, for telling requests kept in it from the rest.
    region: Option<String>,
    tracker: Arc<InFlightTracker>,
    requests: IntCounterVec,
    rejected: IntCounterVec,
    failovers: IntCounterVec,
    response_errors: IntCounterVec,
    time_to_first_byte: HistogramVec,
    duration: HistogramVec,
    in_flight: IntGaugeVec,
    /// Present only with `--router-policy affinity`.
    affinity: Option<AffinityCounters>,
    selection: Histogram,
    /// A scrape resets and refills the gauges, so scrapes must not overlap.
    scrape: Mutex<()>,
}

/// Where the affinity path keeps its cumulative counts.
#[derive(Debug)]
struct Sources {
    stats: Arc<AffinityStats>,
    cache: Arc<CacheStats>,
}

/// One exported series: its label value, and where it reads its cumulative
/// count.
type Series = (&'static str, fn(&Sources) -> &AtomicU64);

/// One affinity counter family.
struct Family {
    suffix: &'static str,
    help: &'static str,
    label: &'static str,
    series: &'static [Series],
}

const FAMILIES: [Family; 4] = [
    Family {
        suffix: "affinity_lookups_total",
        help: "Session requests whose affinity lookup found a live pin among the \
               candidates (hit) or not (miss)",
        label: "outcome",
        series: &[
            ("hit", |sources| &sources.stats.hits),
            ("miss", |sources| &sources.stats.misses),
        ],
    },
    // Named after the local router's `dynamo_router_shared_cache_errors_total`.
    // Every session request that routes without affinity counts once.
    Family {
        suffix: "affinity_store_errors_total",
        help: "Session requests that routed without affinity, by why: the store \
               failed (backend), ran out of budget (timeout), kept conflicting \
               (conflict), or was not consulted because the breaker was open \
               (breaker_open)",
        label: "kind",
        series: &[
            ("backend", |sources| &sources.stats.failures),
            ("timeout", |sources| &sources.stats.timeouts),
            ("conflict", |sources| &sources.stats.conflicts),
            ("breaker_open", |sources| &sources.stats.skipped),
        ],
    },
    Family {
        suffix: "affinity_store_writes_total",
        help: "Conditional writes sent to the affinity store, by why: a session \
               without a live pin (create), a session moving pools (move), a pin \
               past half its idle window (refresh), or an undecodable record \
               (repair)",
        label: "reason",
        series: &[
            ("create", |sources| &sources.stats.creates),
            ("move", |sources| &sources.stats.moves),
            ("refresh", |sources| &sources.stats.refreshes),
            ("repair", |sources| &sources.stats.repairs),
        ],
    },
    Family {
        suffix: "affinity_cache_total",
        help: "Session affinity store reads, by whether the in-memory cache \
               answered them, and cache entries invalidated",
        label: "outcome",
        series: &[
            ("hit", |sources| &sources.cache.hits),
            ("miss", |sources| &sources.cache.misses),
            ("invalidation", |sources| &sources.cache.invalidations),
        ],
    },
];

/// The affinity path keeps plain cumulative counters, which a scrape turns
/// into Prometheus counters by adding what has arrived since the last one.
#[derive(Debug)]
struct AffinityCounters {
    sources: Sources,
    /// One per entry of `FAMILIES`, in order.
    families: Vec<IntCounterVec>,
    /// What each series, in `FAMILIES` order, last reported.
    reported: Mutex<Vec<u64>>,
}

impl AffinityCounters {
    /// Adds each series' arrivals since the last call. The caller holds the
    /// scrape lock, so no count is reported twice.
    fn drain(&self) {
        let mut reported = self.reported.lock().unwrap_or_else(PoisonError::into_inner);
        let series = FAMILIES
            .iter()
            .zip(&self.families)
            .flat_map(|(family, counter)| {
                family.series.iter().map(move |series| (counter, series))
            });
        for (index, (counter, (label, count))) in series.enumerate() {
            let current = count(&self.sources).load(Ordering::Relaxed);
            // Saturating: these only ever grow, but a counter must never be
            // handed a negative delta.
            let arrived = current.saturating_sub(reported[index]);
            if arrived > 0 {
                counter.with_label_values(&[label]).inc_by(arrived);
                reported[index] = current;
            }
        }
    }
}

/// The time each session request spent in affinity storage, in
/// milliseconds, for [`AffinitySelector::with_store_timer`]. Pass it to
/// [`Metrics::with_affinity`] to export it.
///
/// [`AffinitySelector::with_store_timer`]: dynamo_global_router::AffinitySelector::with_store_timer
pub fn affinity_store_ms() -> Histogram {
    // Named and bucketed like the local router's
    // `dynamo_router_overhead_shared_cache_query_ms`, its own remote lookup on
    // the request path, so the two can share a dashboard.
    Histogram::with_opts(
        HistogramOpts::new(
            name("overhead_affinity_store_ms"),
            "Time each session request spent in affinity storage (lookup, write \
             and retry, up to --affinity-timeout-ms), in milliseconds",
        )
        .buckets(
            prometheus::exponential_buckets(0.01, 3.0, 17).expect("buckets are statically valid"),
        ),
    )
    .expect("metric options are statically valid")
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
        let failovers = register(
            &registry,
            IntCounterVec::new(
                Opts::new(
                    name("failovers_total"),
                    "Requests sent on to another pool because the pool before could not \
                     take them, by why: no_response, unavailable (503) or overloaded (the \
                     overload status). Counts once per extra pool tried",
                ),
                &["reason"],
            ),
        );
        for reason in FailoverReason::ALL {
            failovers.with_label_values(&[reason.label()]);
        }
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
        // Mirrors the local router's `dynamo_router_overhead_*_ms`. Unlabelled:
        // compare selection paths across runs.
        let selection = register(
            &registry,
            Histogram::with_opts(
                HistogramOpts::new(
                    name("overhead_selection_ms"),
                    "Time to select a pool for a request, in milliseconds",
                )
                .buckets(prometheus::exponential_buckets(0.01, 3.0, 17).expect("valid buckets")),
            ),
        );
        Self {
            registry,
            region,
            tracker,
            requests,
            rejected,
            failovers,
            response_errors,
            time_to_first_byte,
            duration,
            in_flight,
            affinity: None,
            selection,
            scrape: Mutex::new(()),
        }
    }

    /// Exports the affinity path's metrics, including the degraded path:
    /// requests that lost affinity to a storage error, to the deadline, to
    /// conflicts or to an open breaker; why the store was written; how much
    /// of the lookup load the cache absorbed; and `store_ms`, from
    /// [`affinity_store_ms`], the time requests spent in storage.
    pub fn with_affinity(
        mut self,
        stats: Arc<AffinityStats>,
        cache: Arc<CacheStats>,
        store_ms: Histogram,
    ) -> Self {
        let families = FAMILIES
            .iter()
            .map(|family| {
                register(
                    &self.registry,
                    IntCounterVec::new(
                        Opts::new(name(family.suffix), family.help),
                        &[family.label],
                    ),
                )
            })
            .collect();
        register(&self.registry, Ok(store_ms));
        let series = FAMILIES.iter().map(|family| family.series.len()).sum();
        self.affinity = Some(AffinityCounters {
            sources: Sources { stats, cache },
            families,
            reported: Mutex::new(vec![0; series]),
        });
        self
    }

    pub(crate) fn selected(&self, started: Instant) {
        self.selection
            .observe(started.elapsed().as_secs_f64() * 1000.0);
    }

    /// Counts a request answered without a pool. `error_type` is one of
    /// `prometheus_names::frontend_service::error_type`.
    pub(crate) fn rejected(&self, error_type: &str) {
        self.rejected.with_label_values(&[error_type]).inc();
    }

    /// Counts a request sent on to another pool for `reason`.
    pub(crate) fn failover(&self, reason: FailoverReason) {
        self.failovers.with_label_values(&[reason.label()]).inc();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample<'a>(text: &'a str, series: &str) -> Option<&'a str> {
        text.lines()
            .find_map(|line| line.strip_prefix(series)?.strip_prefix(' '))
    }

    #[test]
    fn affinity_counters_render_as_their_increase_since_the_last_scrape() {
        let stats = Arc::new(AffinityStats::default());
        let cache = Arc::new(CacheStats::default());
        let store_ms = affinity_store_ms();
        let metrics = Metrics::new(None, Arc::new(InFlightTracker::new())).with_affinity(
            stats.clone(),
            cache.clone(),
            store_ms.clone(),
        );
        let pools = PoolSnapshot::new(Vec::new());
        stats.hits.fetch_add(3, Ordering::Relaxed);
        stats.timeouts.fetch_add(1, Ordering::Relaxed);
        stats.skipped.fetch_add(2, Ordering::Relaxed);
        stats.refreshes.fetch_add(1, Ordering::Relaxed);
        cache.hits.fetch_add(5, Ordering::Relaxed);
        store_ms.observe(0.5);
        store_ms.observe(20.0);

        let text = metrics.render(&pools).unwrap();
        let expect = [
            (
                r#"dynamo_global_frontend_affinity_lookups_total{outcome="hit"}"#,
                "3",
            ),
            (
                r#"dynamo_global_frontend_affinity_store_errors_total{kind="timeout"}"#,
                "1",
            ),
            (
                r#"dynamo_global_frontend_affinity_store_errors_total{kind="breaker_open"}"#,
                "2",
            ),
            (
                r#"dynamo_global_frontend_affinity_store_writes_total{reason="refresh"}"#,
                "1",
            ),
            (
                r#"dynamo_global_frontend_affinity_cache_total{outcome="hit"}"#,
                "5",
            ),
            (
                "dynamo_global_frontend_overhead_affinity_store_ms_count",
                "2",
            ),
        ];
        for (series, value) in expect {
            assert_eq!(sample(&text, series), Some(value), "{series}");
        }

        // A second scrape adds only what arrived since the first.
        stats.hits.fetch_add(1, Ordering::Relaxed);
        let text = metrics.render(&pools).unwrap();
        assert_eq!(
            sample(
                &text,
                r#"dynamo_global_frontend_affinity_lookups_total{outcome="hit"}"#
            ),
            Some("4")
        );
    }
}
