// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Frontend's Prometheus metrics, served at `/metrics`.
//!
//! Series are labelled with the pool's name and region, never with anything
//! from the request body: a request's `model` is whatever the client sent,
//! and would mint a series per string. Pool frontends report what needs the
//! stream parsed, such as inter-token latency and token counts.
//!
//! With affinity routing, the metrics also show what the selector weighs and
//! decides (see [`Metrics::with_routing`]).

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use axum::http::StatusCode;
use dynamo_global_router::{AffinitySelector, InFlightTracker, Pool, PoolSignals};
use dynamo_llm::http::service::metrics::generate_log_buckets;
use dynamo_runtime::metrics::prometheus_names::clamp_u64_to_i64;
use prometheus::{
    GaugeVec, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge,
    IntGaugeVec, Opts, Registry, TextEncoder,
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
    routing: Option<Routing>,
    /// A scrape resets and refills the gauges, so scrapes must not overlap.
    scrape: Mutex<()>,
}

/// What the affinity selector weighs and decides, read at each scrape.
#[derive(Debug)]
struct Routing {
    signals: Arc<PoolSignals>,
    affinity: Arc<AffinitySelector>,
    occupancy: GaugeVec,
    round_trips: GaugeVec,
    selections: IntCounterVec,
    pinned: IntGauge,
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
            routing: None,
            scrape: Mutex::new(()),
        }
    }

    /// Adds what the affinity selector `affinity` weighs, from `signals`, and
    /// what it decides.
    pub fn with_routing(
        mut self,
        signals: Arc<PoolSignals>,
        affinity: Arc<AffinitySelector>,
    ) -> Self {
        let registry = &self.registry;
        let occupancy = register(
            registry,
            GaugeVec::new(
                Opts::new(
                    name("pool_occupancy_ratio"),
                    "How full each pool is against its own capacity, as the selector \
                     sees it, by resource: slots or kv. Absent while unknown",
                ),
                &["pool", "region", "resource"],
            ),
        );
        let round_trips = register(
            registry,
            GaugeVec::new(
                Opts::new(
                    name("region_round_trip_seconds"),
                    "The measured round trip to each region's pools",
                ),
                &["region"],
            ),
        );
        let selections = register(
            registry,
            IntCounterVec::new(
                Opts::new(
                    name("affinity_selections_total"),
                    "Requests placed by the affinity selector, by what became of their \
                     session's pin: stayed, moved, new or no_session",
                ),
                &["outcome"],
            ),
        );
        let pinned = register(
            registry,
            IntGauge::new(
                name("affinity_pinned_sessions"),
                "Sessions pinned in this process, including expired pins until the \
                 next sweep",
            ),
        );
        self.routing = Some(Routing {
            signals,
            affinity,
            occupancy,
            round_trips,
            selections,
            pinned,
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
        if let Some(routing) = &self.routing {
            routing.refresh(pools);
        }
        TextEncoder::new().encode_to_string(&self.registry.gather())
    }
}

impl Routing {
    fn refresh(&self, pools: &PoolSnapshot) {
        self.occupancy.reset();
        for pool in pools.pools() {
            let Some(occupancy) = self.signals.occupancy(&pool.id) else {
                continue;
            };
            let resources = [("slots", Some(occupancy.slots)), ("kv", occupancy.kv)];
            for (resource, value) in resources {
                if let Some(value) = value {
                    self.occupancy
                        .with_label_values(&[&pool.name, &pool.location.region, resource])
                        .set(value);
                }
            }
        }
        self.round_trips.reset();
        for (region, round_trip) in self.signals.region_round_trips().iter() {
            self.round_trips
                .with_label_values(&[region])
                .set(round_trip.as_secs_f64());
        }
        // The selector keeps the counts; each scrape catches the counters up.
        for (outcome, total) in self.affinity.selections() {
            let counter = self.selections.with_label_values(&[outcome]);
            counter.inc_by(total.saturating_sub(counter.get()));
        }
        let pinned = i64::try_from(self.affinity.pinned_sessions()).unwrap_or(i64::MAX);
        self.pinned.set(pinned);
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
    use std::collections::HashMap;
    use std::time::Duration;

    use dynamo_global_router::{LoadReport, PoolSelector, RoutingRequest};

    use super::*;
    use crate::catalog::tests::pool;

    #[test]
    fn shows_what_the_affinity_selector_weighs_and_decides() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let in_flight = Arc::new(InFlightTracker::new());
        let signals = Arc::new(PoolSignals::new(in_flight.clone()));
        // Blue has a report: half its slots, and 1,000 of 16,000 KV blocks
        // for 16 running requests (2,000 for all 32). Green has none.
        let window = Duration::from_secs(60);
        let report = LoadReport {
            in_flight: 32,
            generating: 16,
            max_concurrency: Some(64),
            window,
        };
        signals.record_load(&blue.id, report);
        signals.record_kv(&blue.id, 1000, 16000, window);
        let tokyo = ("ap-northeast-1".to_owned(), Duration::from_millis(135));
        signals.set_region_round_trips(HashMap::from([tokyo]));
        let region = blue.location.region.clone();
        let idle = Duration::from_secs(60);
        let affinity = Arc::new(AffinitySelector::new(signals.clone(), region, 2.0, idle));
        let metrics = Metrics::new(None, in_flight).with_routing(signals, affinity.clone());
        let pools = PoolSnapshot::new(vec![blue.clone(), green.clone()]);

        let request = RoutingRequest::new("qwen").with_session(Some("s"));
        affinity.select(&request, &[&blue, &green]).unwrap();
        let text = metrics.render(&pools).unwrap();
        let occupancy = "dynamo_global_frontend_pool_occupancy_ratio";
        let blue_labels = r#"pool="ohio/dynamo/blue",region="us-east-2""#;
        for line in [
            format!("{occupancy}{{{blue_labels},resource=\"slots\"}} 0.5"),
            format!("{occupancy}{{{blue_labels},resource=\"kv\"}} 0.125"),
            r#"dynamo_global_frontend_region_round_trip_seconds{region="ap-northeast-1"} 0.135"#
                .to_owned(),
            r#"dynamo_global_frontend_affinity_selections_total{outcome="new"} 1"#.to_owned(),
            "dynamo_global_frontend_affinity_pinned_sessions 1".to_owned(),
        ] {
            assert!(
                text.lines().any(|sample| sample == line),
                "{line} in\n{text}"
            );
        }
        assert!(!text.contains(&format!(r#"{occupancy}{{pool="ohio/dynamo/green""#)));

        // Counters catch up with the selector at each scrape.
        affinity.select(&request, &[&blue, &green]).unwrap();
        let text = metrics.render(&pools).unwrap();
        for outcome in ["new", "stayed"] {
            let line = format!(
                r#"dynamo_global_frontend_affinity_selections_total{{outcome="{outcome}"}} 1"#
            );
            assert!(
                text.lines().any(|sample| sample == line),
                "{line} in\n{text}"
            );
        }
    }
}
