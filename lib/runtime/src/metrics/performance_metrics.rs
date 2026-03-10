// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Initially contributed by Baseten @michaelfeil, feel free to tag on maintance.

//! Generic sliding-window performance metrics with a command-loop backend.
//!
//! All mutations and publishing are processed by a single worker thread via commands/events.
//! Hot-path recording from metric handles is lock-free at call site (bounded `try_send`).
//! Ideal for high-cardinality metrics.

use super::{MetricsHierarchy, create_metric, prometheus_names::build_component_metric_name};
use parking_lot::Mutex;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const SUFFIX_PER_SECOND: &str = "per_second";
const SUFFIX_AVG: &str = "avg";
const SUFFIX_NUMERATOR: &str = "numerator";
const SUFFIX_DENOMINATOR: &str = "denominator";
const SUFFIX_RATIO: &str = "ratio";
const SUFFIX_QUANTILE: &str = "quantile";
const LABEL_QUANTILE: &str = "quantile";
const DEFAULT_QUEUE_CAPACITY: usize = 16_384;
const WORKER_POLL: Duration = Duration::from_millis(100);
// guard against unbounded memory growth (should not happen for reasonable metrics)
const MAX_SAMPLES_PER_METRIC: usize = 100_000;
const DEFAULT_METRIC_WINDOW_SECONDS: f64 = 60.0;
const SAMPLE_OVERFLOW_LOG_INTERVAL: u64 = 5_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceMetricKind {
    Rate,
    Distribution,
    Ratio,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetricSpec {
    pub name: String,
    pub kind: PerformanceMetricKind,
    pub quantiles: Vec<f64>,
    pub sample_period_seconds: Option<f64>,
    pub window_seconds: f64,
}

impl PerformanceMetricSpec {
    pub fn rate(
        name: impl Into<String>,
        quantiles: Vec<f64>,
        sample_period_seconds: Option<f64>,
        window_seconds: Option<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            kind: PerformanceMetricKind::Rate,
            quantiles: normalize_quantiles(quantiles),
            sample_period_seconds: sample_period_seconds
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(|v| v.max(0.1)),
            window_seconds: window_seconds
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(|v| v.max(0.1))
                .unwrap_or(DEFAULT_METRIC_WINDOW_SECONDS),
        }
    }

    pub fn distribution(
        name: impl Into<String>,
        quantiles: Vec<f64>,
        window_seconds: Option<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            kind: PerformanceMetricKind::Distribution,
            quantiles: normalize_quantiles(quantiles),
            sample_period_seconds: None,
            window_seconds: window_seconds
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(|v| v.max(0.1))
                .unwrap_or(DEFAULT_METRIC_WINDOW_SECONDS),
        }
    }

    pub fn ratio(
        name: impl Into<String>,
        quantiles: Vec<f64>,
        window_seconds: Option<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            kind: PerformanceMetricKind::Ratio,
            quantiles: normalize_quantiles(quantiles),
            sample_period_seconds: None,
            window_seconds: window_seconds
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(|v| v.max(0.1))
                .unwrap_or(DEFAULT_METRIC_WINDOW_SECONDS),
        }
    }
}

#[derive(Debug, Clone)]
struct MetricSnapshot {
    pub name: String,
    pub kind: PerformanceMetricKind,
    pub rate_per_second: Option<f64>,
    pub average: Option<f64>,
    pub quantiles: Vec<(f64, f64)>,
    pub numerator_sum: Option<f64>,
    pub denominator_sum: Option<f64>,
    pub ratio: Option<f64>,
}

#[derive(Debug)]
pub struct PerformanceMetricsRegistry {
    inner: Arc<RegistryInner>,
}

impl Clone for PerformanceMetricsRegistry {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[derive(Clone)]
struct PerformanceMetricHandle {
    lease: Arc<MetricLease>,
}

#[derive(Clone)]
pub struct RateMetricHandle {
    inner: PerformanceMetricHandle,
}

#[derive(Clone)]
pub struct DistributionMetricHandle {
    inner: PerformanceMetricHandle,
}

#[derive(Clone)]
pub struct RatioMetricHandle {
    inner: PerformanceMetricHandle,
}

struct RequestMetricsFactoryInner {
    request: RateMetricHandle,
    input_tokens: RateMetricHandle,
    net_new_input_tokens: RateMetricHandle,
    ttft: DistributionMetricHandle,
    ttft_per_input_token: DistributionMetricHandle,
    ttft_per_net_new_input_token: DistributionMetricHandle,
    itl: DistributionMetricHandle,
    pre_first_token_cancellation: RateMetricHandle,
    mid_stream_cancellation: RateMetricHandle,
    successful_request: RateMetricHandle,
    itl_sample_rate: f64,
}

#[derive(Debug, Clone)]
/// Configuration for [`RequestMetricsFactory`].
///
/// Use [`Default`] for sensible presets and override only the fields you care about.
pub struct RequestMetricsOptions {
    pub request_quantiles: Vec<f64>,
    pub request_sample_period_seconds: Option<f64>,
    pub request_window_seconds: Option<f64>,
    pub input_tokens_quantiles: Vec<f64>,
    pub input_tokens_sample_period_seconds: Option<f64>,
    pub input_tokens_window_seconds: Option<f64>,
    pub ttft_quantiles: Vec<f64>,
    pub ttft_window_seconds: Option<f64>,
    pub ttft_per_input_token_quantiles: Vec<f64>,
    pub ttft_per_input_token_window_seconds: Option<f64>,
    pub itl_quantiles: Vec<f64>,
    pub itl_window_seconds: Option<f64>,
    pub pre_first_token_cancellation_quantiles: Vec<f64>,
    pub pre_first_token_cancellation_sample_period_seconds: Option<f64>,
    pub pre_first_token_cancellation_window_seconds: Option<f64>,
    pub mid_stream_cancellation_quantiles: Vec<f64>,
    pub mid_stream_cancellation_sample_period_seconds: Option<f64>,
    pub mid_stream_cancellation_window_seconds: Option<f64>,
    pub successful_request_quantiles: Vec<f64>,
    pub successful_request_sample_period_seconds: Option<f64>,
    pub successful_request_window_seconds: Option<f64>,
    pub itl_sample_rate: f64,
}

impl Default for RequestMetricsOptions {
    fn default() -> Self {
        Self {
            request_quantiles: vec![0.1, 0.5, 0.9],
            request_sample_period_seconds: Some(1.0),
            request_window_seconds: None,
            input_tokens_quantiles: vec![0.01, 0.1, 0.5, 0.9, 0.99],
            input_tokens_sample_period_seconds: Some(1.0),
            input_tokens_window_seconds: None,
            ttft_quantiles: vec![0.1, 0.5, 0.9, 0.99],
            ttft_window_seconds: None,
            ttft_per_input_token_quantiles: vec![0.1, 0.5, 0.9, 0.99],
            ttft_per_input_token_window_seconds: None,
            itl_quantiles: vec![0.1, 0.5, 0.9, 0.99, 0.999],
            itl_window_seconds: None,
            pre_first_token_cancellation_quantiles: vec![],
            pre_first_token_cancellation_sample_period_seconds: Some(1.0),
            pre_first_token_cancellation_window_seconds: None,
            mid_stream_cancellation_quantiles: vec![],
            mid_stream_cancellation_sample_period_seconds: Some(1.0),
            mid_stream_cancellation_window_seconds: None,
            successful_request_quantiles: vec![],
            successful_request_sample_period_seconds: Some(1.0),
            successful_request_window_seconds: None,
            itl_sample_rate: 0.05,
        }
    }
}

#[derive(Clone)]
/// Factory that creates one [`RequestMetric`] per request.
///
/// Calling [`RequestMetricsFactory::new`] registers a fixed set of sliding-window metrics under
/// the provided `metric_prefix`.
///
/// With `metric_prefix = "request_metrics"` and default [`RequestMetricsOptions`]:
///
/// | Base metric | Kind | Exported gauges |
/// | --- | --- | --- |
/// | `{prefix}_request` | Rate | `{prefix}_request_per_second`, `{prefix}_request_per_second_quantile{quantile="0.1|0.5|0.9"}` |
/// | `{prefix}_input_tokens` | Rate | `{prefix}_input_tokens_per_second`, `{prefix}_input_tokens_per_second_quantile{quantile="0.01|0.1|0.5|0.9|0.99"}` |
/// | `{prefix}_net_new_input_tokens` | Rate | `{prefix}_net_new_input_tokens_per_second`, `{prefix}_net_new_input_tokens_per_second_quantile{quantile="0.01|0.1|0.5|0.9|0.99"}` |
/// | `{prefix}_ttft_ms` | Distribution | `{prefix}_ttft_ms_avg`, `{prefix}_ttft_ms{quantile="0.1|0.5|0.9|0.99"}` |
/// | `{prefix}_ttft_ms_per_input_token` | Distribution | `{prefix}_ttft_ms_per_input_token_avg`, `{prefix}_ttft_ms_per_input_token{quantile="0.1|0.5|0.9|0.99"}` |
/// | `{prefix}_ttft_ms_per_net_new_input_token` | Distribution | `{prefix}_ttft_ms_per_net_new_input_token_avg`, `{prefix}_ttft_ms_per_net_new_input_token{quantile="0.1|0.5|0.9|0.99"}` |
/// | `{prefix}_itl_ms` | Distribution | `{prefix}_itl_ms_avg`, `{prefix}_itl_ms{quantile="0.1|0.5|0.9|0.99|0.999"}` |
/// | `{prefix}_pre_first_token_cancellation` | Rate | `{prefix}_pre_first_token_cancellation_per_second` |
/// | `{prefix}_mid_stream_cancellation` | Rate | `{prefix}_mid_stream_cancellation_per_second` |
/// | `{prefix}_successful_request` | Rate | `{prefix}_successful_request_per_second` |
///
/// Notes:
/// - Rate quantiles are exported on `*_per_second_quantile` with a `quantile` label.
/// - Distribution quantiles are exported on the base metric with a `quantile` label.
/// - Input tokens are recorded on first [`RequestMetric::record_tokens`] call.
///   If `new_request(input_tokens=0)` is used, first `total_tokens` is used as input-token value.
/// - Net-new metrics are recorded once and only when
///   [`RequestMetric::record_tokens`] is called with `cached_tokens=Some(...)` before first token.
pub struct RequestMetricsFactory {
    inner: Arc<RequestMetricsFactoryInner>,
}

/// Per-request recorder for TTFT/ITL and terminal outcomes.
pub struct RequestMetric {
    factory: Arc<RequestMetricsFactoryInner>,
    input_tokens: u64,
    started_at: Instant,
    last_token_at: Option<Instant>,
    last_total_tokens: u64,
    terminal: bool,
    cached_tokens_hint: Option<u64>,
    rng: SmallRng,
}

#[derive(Debug)]
struct RegistryInner {
    publish_interval: Duration,
    control_tx: Sender<ControlMessage>,
    data_tx: SyncSender<DataMessage>,
    dropped_events: AtomicU64,
    worker: Mutex<Option<JoinHandle<()>>>,
}

struct MetricLease {
    registry: Arc<RegistryInner>,
    metric_name: Arc<str>,
}

struct WorkerState {
    publish_interval: Duration,
    metrics: HashMap<String, MetricEntry>,
    publisher: PublisherState,
}

struct PublisherState {
    hierarchy: Arc<dyn MetricsHierarchy>,
    metric_prefix: String,
    labels: Vec<(String, String)>,
    handles: HashMap<String, MetricHandles>,
    next_publish_at: Instant,
}

#[derive(Clone)]
enum MetricHandles {
    Rate {
        per_second: prometheus::Gauge,
        quantiles: Vec<(f64, prometheus::Gauge)>,
    },
    Distribution {
        avg: prometheus::Gauge,
        quantiles: Vec<(f64, prometheus::Gauge)>,
    },
    Ratio {
        numerator: prometheus::Gauge,
        denominator: prometheus::Gauge,
        ratio: prometheus::Gauge,
        quantiles: Vec<(f64, prometheus::Gauge)>,
    },
}

impl MetricHandles {
    fn collectors(&self) -> Vec<Box<dyn prometheus::core::Collector>> {
        match self {
            MetricHandles::Rate {
                per_second,
                quantiles,
            } => {
                let mut out: Vec<Box<dyn prometheus::core::Collector>> =
                    vec![Box::new(per_second.clone())];
                out.extend(quantiles.iter().map(|(_, gauge)| {
                    Box::new(gauge.clone()) as Box<dyn prometheus::core::Collector>
                }));
                out
            }
            MetricHandles::Distribution { avg, quantiles } => {
                let mut out: Vec<Box<dyn prometheus::core::Collector>> =
                    vec![Box::new(avg.clone())];
                out.extend(quantiles.iter().map(|(_, gauge)| {
                    Box::new(gauge.clone()) as Box<dyn prometheus::core::Collector>
                }));
                out
            }
            MetricHandles::Ratio {
                numerator,
                denominator,
                ratio,
                quantiles,
            } => {
                let mut out: Vec<Box<dyn prometheus::core::Collector>> = vec![
                    Box::new(numerator.clone()),
                    Box::new(denominator.clone()),
                    Box::new(ratio.clone()),
                ];
                out.extend(quantiles.iter().map(|(_, gauge)| {
                    Box::new(gauge.clone()) as Box<dyn prometheus::core::Collector>
                }));
                out
            }
        }
    }
}

#[derive(Debug)]
struct MetricEntry {
    spec: PerformanceMetricSpec,
    state: MetricState,
    overflow_drops: u64,
}

#[derive(Debug)]
enum MetricState {
    Rate {
        counts: VecDeque<(Instant, u64)>,
    },
    Distribution {
        values: VecDeque<(Instant, f64)>,
    },
    Ratio {
        pairs: VecDeque<(Instant, f64, f64)>,
    },
}

enum ControlMessage {
    RegisterMetric {
        spec: PerformanceMetricSpec,
        resp: mpsc::Sender<anyhow::Result<()>>,
    },
    UnregisterMetric {
        name: String,
        resp: mpsc::Sender<anyhow::Result<()>>,
    },
    #[cfg(test)]
    SnapshotMetric {
        name: String,
        resp: mpsc::Sender<Option<MetricSnapshot>>,
    },
    #[cfg(test)]
    SnapshotAll {
        resp: mpsc::Sender<Vec<MetricSnapshot>>,
    },
    Shutdown,
}

enum DataMessage {
    Count {
        name: Arc<str>,
        count: u64,
        recorded_at: Instant,
    },
    Value {
        name: Arc<str>,
        value: f64,
        recorded_at: Instant,
    },
    Ratio {
        name: Arc<str>,
        numerator: f64,
        denominator: f64,
        recorded_at: Instant,
    },
}

impl PerformanceMetricsRegistry {
    fn new_attached_with(
        publish_interval: Duration,
        hierarchy: Arc<dyn MetricsHierarchy>,
        metric_prefix: String,
        labels: Vec<(String, String)>,
    ) -> Self {
        let publish_interval = publish_interval
            .max(Duration::from_secs(2))
            .min(Duration::from_secs(60));
        let (control_tx, control_rx) = mpsc::channel::<ControlMessage>();
        let (data_tx, data_rx) = mpsc::sync_channel::<DataMessage>(DEFAULT_QUEUE_CAPACITY);
        let worker = std::thread::spawn(move || {
            worker_loop(
                control_rx,
                data_rx,
                publish_interval,
                PublisherState {
                    hierarchy,
                    metric_prefix,
                    labels,
                    handles: HashMap::new(),
                    next_publish_at: Instant::now() + publish_interval,
                },
            )
        });

        Self {
            inner: Arc::new(RegistryInner {
                publish_interval,
                control_tx,
                data_tx,
                dropped_events: AtomicU64::new(0),
                worker: Mutex::new(Some(worker)),
            }),
        }
    }

    pub fn new_attached(
        publish_interval: Duration,
        hierarchy: Arc<dyn MetricsHierarchy>,
        metric_prefix: impl Into<String>,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<Self> {
        let labels_owned = labels
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect::<Vec<_>>();
        Ok(Self::new_attached_with(
            publish_interval,
            hierarchy,
            metric_prefix.into(),
            labels_owned,
        ))
    }

    pub fn new_attached_default(hierarchy: Arc<dyn MetricsHierarchy>) -> anyhow::Result<Self> {
        Self::new_attached(
            Duration::from_secs(5),
            hierarchy,
            "performance",
            &[] as &[(&str, &str)],
        )
    }

    pub fn dropped_events(&self) -> u64 {
        self.inner.dropped_events.load(Ordering::Relaxed)
    }

    pub fn new_rate_metric(
        &self,
        name: impl Into<String>,
        quantiles: Vec<f64>,
        sample_period_seconds: Option<f64>,
        window_seconds: Option<f64>,
    ) -> anyhow::Result<RateMetricHandle> {
        let name = name.into();
        self.inner.register_metric(PerformanceMetricSpec::rate(
            name.clone(),
            quantiles,
            sample_period_seconds,
            window_seconds,
        ))?;
        Ok(RateMetricHandle {
            inner: PerformanceMetricHandle {
                lease: Arc::new(MetricLease {
                    registry: Arc::clone(&self.inner),
                    metric_name: Arc::<str>::from(name),
                }),
            },
        })
    }

    pub fn new_distribution_metric(
        &self,
        name: impl Into<String>,
        quantiles: Vec<f64>,
        window_seconds: Option<f64>,
    ) -> anyhow::Result<DistributionMetricHandle> {
        let name = name.into();
        self.inner
            .register_metric(PerformanceMetricSpec::distribution(
                name.clone(),
                quantiles,
                window_seconds,
            ))?;
        Ok(DistributionMetricHandle {
            inner: PerformanceMetricHandle {
                lease: Arc::new(MetricLease {
                    registry: Arc::clone(&self.inner),
                    metric_name: Arc::<str>::from(name),
                }),
            },
        })
    }

    pub fn new_ratio_metric(
        &self,
        name: impl Into<String>,
        quantiles: Vec<f64>,
        window_seconds: Option<f64>,
    ) -> anyhow::Result<RatioMetricHandle> {
        let name = name.into();
        self.inner.register_metric(PerformanceMetricSpec::ratio(
            name.clone(),
            quantiles,
            window_seconds,
        ))?;
        Ok(RatioMetricHandle {
            inner: PerformanceMetricHandle {
                lease: Arc::new(MetricLease {
                    registry: Arc::clone(&self.inner),
                    metric_name: Arc::<str>::from(name),
                }),
            },
        })
    }

    pub fn unregister_metric(&self, name: impl Into<String>) -> anyhow::Result<()> {
        self.inner.unregister_metric(name.into())
    }

    #[cfg(test)]
    fn snapshot_all_for_test(&self) -> Vec<MetricSnapshot> {
        self.inner.snapshot_all()
    }
}

impl PerformanceMetricHandle {
    pub fn name(&self) -> &str {
        self.lease.metric_name.as_ref()
    }

    fn record_count(&self, count: u64) -> anyhow::Result<()> {
        self.lease
            .registry
            .record_count(Arc::clone(&self.lease.metric_name), count)
    }

    fn record_value(&self, value: f64) -> anyhow::Result<()> {
        self.lease
            .registry
            .record_value(Arc::clone(&self.lease.metric_name), value)
    }

    fn record_ratio(&self, numerator: f64, denominator: f64) -> anyhow::Result<()> {
        self.lease.registry.record_ratio(
            Arc::clone(&self.lease.metric_name),
            numerator,
            denominator,
        )
    }

    #[cfg(test)]
    fn snapshot_for_test(&self) -> Option<MetricSnapshot> {
        self.lease
            .registry
            .snapshot_metric(self.lease.metric_name.to_string())
    }
}

impl Drop for MetricLease {
    fn drop(&mut self) {
        self.registry
            .try_unregister_metric(self.metric_name.to_string());
    }
}

impl RateMetricHandle {
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn record_count(&self, count: u64) -> anyhow::Result<()> {
        self.inner.record_count(count)
    }

    #[cfg(test)]
    fn snapshot_for_test(&self) -> Option<MetricSnapshot> {
        self.inner.snapshot_for_test()
    }
}

impl DistributionMetricHandle {
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn record_value(&self, value: f64) -> anyhow::Result<()> {
        self.inner.record_value(value)
    }

    #[cfg(test)]
    fn snapshot_for_test(&self) -> Option<MetricSnapshot> {
        self.inner.snapshot_for_test()
    }
}

impl RatioMetricHandle {
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn record_ratio(&self, numerator: f64, denominator: f64) -> anyhow::Result<()> {
        self.inner.record_ratio(numerator, denominator)
    }

    #[cfg(test)]
    fn snapshot_for_test(&self) -> Option<MetricSnapshot> {
        self.inner.snapshot_for_test()
    }
}

impl RequestMetricsFactory {
    /// Create a factory from explicit options.
    pub fn new(
        registry: &PerformanceMetricsRegistry,
        metric_prefix: impl AsRef<str>,
        mut options: RequestMetricsOptions,
    ) -> anyhow::Result<Self> {
        options.itl_sample_rate = options.itl_sample_rate.clamp(0.0, 1.0);
        let metric_prefix = metric_prefix.as_ref();

        let request = registry.new_rate_metric(
            format!("{metric_prefix}_request"),
            options.request_quantiles.clone(),
            options.request_sample_period_seconds,
            options.request_window_seconds,
        )?;
        let input_tokens = registry.new_rate_metric(
            format!("{metric_prefix}_input_tokens"),
            options.input_tokens_quantiles.clone(),
            options.input_tokens_sample_period_seconds,
            options.input_tokens_window_seconds,
        )?;
        let net_new_input_tokens = registry.new_rate_metric(
            format!("{metric_prefix}_net_new_input_tokens"),
            options.input_tokens_quantiles.clone(),
            options.input_tokens_sample_period_seconds,
            options.input_tokens_window_seconds,
        )?;
        let ttft = registry.new_distribution_metric(
            format!("{metric_prefix}_ttft_ms"),
            options.ttft_quantiles.clone(),
            options.ttft_window_seconds,
        )?;
        let ttft_per_input_token = registry.new_distribution_metric(
            format!("{metric_prefix}_ttft_ms_per_input_token"),
            options.ttft_per_input_token_quantiles.clone(),
            options.ttft_per_input_token_window_seconds,
        )?;
        let ttft_per_net_new_input_token = registry.new_distribution_metric(
            format!("{metric_prefix}_ttft_ms_per_net_new_input_token"),
            options.ttft_per_input_token_quantiles.clone(),
            options.ttft_per_input_token_window_seconds,
        )?;
        let itl = registry.new_distribution_metric(
            format!("{metric_prefix}_itl_ms"),
            options.itl_quantiles.clone(),
            options.itl_window_seconds,
        )?;
        let pre_first_token_cancellation = registry.new_rate_metric(
            format!("{metric_prefix}_pre_first_token_cancellation"),
            options.pre_first_token_cancellation_quantiles.clone(),
            options.pre_first_token_cancellation_sample_period_seconds,
            options.pre_first_token_cancellation_window_seconds,
        )?;
        let mid_stream_cancellation = registry.new_rate_metric(
            format!("{metric_prefix}_mid_stream_cancellation"),
            options.mid_stream_cancellation_quantiles.clone(),
            options.mid_stream_cancellation_sample_period_seconds,
            options.mid_stream_cancellation_window_seconds,
        )?;
        let successful_request = registry.new_rate_metric(
            format!("{metric_prefix}_successful_request"),
            options.successful_request_quantiles.clone(),
            options.successful_request_sample_period_seconds,
            options.successful_request_window_seconds,
        )?;

        Ok(Self {
            inner: Arc::new(RequestMetricsFactoryInner {
                request,
                input_tokens,
                net_new_input_tokens,
                ttft,
                ttft_per_input_token,
                ttft_per_net_new_input_token,
                itl,
                pre_first_token_cancellation,
                mid_stream_cancellation,
                successful_request,
                itl_sample_rate: options.itl_sample_rate,
            }),
        })
    }

    /// Start tracking a single request.
    ///
    /// This records one request event immediately.
    ///
    /// Input tokens are recorded on the first [`RequestMetric::record_tokens`] call.
    pub fn new_request(&self, input_tokens: u64) -> RequestMetric {
        if let Err(e) = self.inner.request.record_count(1) {
            tracing::warn!(error = %e, "failed to record request rate metric");
        }

        RequestMetric {
            factory: Arc::clone(&self.inner),
            input_tokens,
            started_at: Instant::now(),
            last_token_at: None,
            last_total_tokens: 0,
            terminal: false,
            cached_tokens_hint: None,
            rng: SmallRng::from_rng(&mut rand::rng()),
        }
    }
}

impl RequestMetric {
    /// Record cumulative output token count for the request.
    ///
    /// On the first token this records TTFT in milliseconds. On later updates this records
    /// sampled ITL values in milliseconds per token.
    pub fn record_tokens(
        &mut self,
        total_tokens: u64,
        cached_tokens: Option<u64>,
    ) -> anyhow::Result<()> {
        if self.terminal {
            return Ok(());
        }

        if let Some(cached_tokens) = cached_tokens {
            self.cached_tokens_hint.get_or_insert(cached_tokens);
        }

        if total_tokens <= self.last_total_tokens {
            return Ok(());
        }

        let now = Instant::now();
        if self.last_token_at.is_none() {
            if self.input_tokens == 0 {
                self.input_tokens = total_tokens;
            }
            if self.input_tokens > 0 {
                self.factory.input_tokens.record_count(self.input_tokens)?;
            }

            let ttft_ms = now.duration_since(self.started_at).as_secs_f64() * 1000.0;
            self.factory.ttft.record_value(ttft_ms)?;
            if self.input_tokens > 0 {
                self.factory
                    .ttft_per_input_token
                    .record_value(ttft_ms / self.input_tokens as f64)?;
            }
            if let Some(cached_tokens) = self.cached_tokens_hint.take() {
                let net_new = self.input_tokens.saturating_sub(cached_tokens);
                if net_new > 0 {
                    self.factory.net_new_input_tokens.record_count(net_new)?;
                    self.factory
                        .ttft_per_net_new_input_token
                        .record_value(ttft_ms / net_new as f64)?;
                }
            }
            self.last_token_at = Some(now);
            self.last_total_tokens = total_tokens;
            return Ok(());
        }

        let last_token_at = self.last_token_at.unwrap_or(now);

        let delta_tokens = total_tokens - self.last_total_tokens;
        let elapsed_ms = now.duration_since(last_token_at).as_secs_f64() * 1000.0;
        if elapsed_ms > 0.0 {
            let itl_ms = elapsed_ms / delta_tokens as f64;
            let samples = sampled_count(delta_tokens, self.factory.itl_sample_rate, &mut self.rng);
            for _ in 0..samples {
                self.factory.itl.record_value(itl_ms)?;
            }
        }

        self.last_token_at = Some(now);
        self.last_total_tokens = total_tokens;
        Ok(())
    }

    /// Mark the request as cancelled.
    pub fn cancel(&mut self) -> anyhow::Result<()> {
        self.finalize_cancellation()
    }

    /// Mark the request as successful.
    pub fn success(&mut self) -> anyhow::Result<()> {
        if self.terminal {
            return Ok(());
        }
        self.factory.successful_request.record_count(1)?;
        self.terminal = true;
        Ok(())
    }

    fn finalize_cancellation(&mut self) -> anyhow::Result<()> {
        if self.terminal {
            return Ok(());
        }
        if self.last_token_at.is_some() {
            self.factory.mid_stream_cancellation.record_count(1)?;
        } else {
            self.factory.pre_first_token_cancellation.record_count(1)?;
        }
        self.terminal = true;
        Ok(())
    }
}

impl Drop for RequestMetric {
    fn drop(&mut self) {
        if !self.terminal
            && let Err(e) = self.finalize_cancellation()
        {
            tracing::warn!(
                error = %e,
                "failed to record request lifecycle cancellation during drop"
            );
        }
    }
}

impl RegistryInner {
    fn register_metric(&self, spec: PerformanceMetricSpec) -> anyhow::Result<()> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.control_tx
            .send(ControlMessage::RegisterMetric {
                spec,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("metrics worker is not running"))?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("metrics worker did not respond"))?
    }

    fn unregister_metric(&self, name: String) -> anyhow::Result<()> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.control_tx
            .send(ControlMessage::UnregisterMetric {
                name,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("metrics worker is not running"))?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("metrics worker did not respond"))?
    }

    fn try_unregister_metric(&self, name: String) {
        let (resp_tx, _resp_rx) = mpsc::channel();
        let _ = self.control_tx.send(ControlMessage::UnregisterMetric {
            name,
            resp: resp_tx,
        });
    }

    fn record_count(&self, name: Arc<str>, count: u64) -> anyhow::Result<()> {
        if count == 0 {
            return Ok(());
        }
        match self.data_tx.try_send(DataMessage::Count {
            name,
            count,
            recorded_at: Instant::now(),
        }) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => {
                self.note_dropped_event();
                Ok(())
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                anyhow::bail!("metrics worker is not running")
            }
        }
    }

    fn record_value(&self, name: Arc<str>, value: f64) -> anyhow::Result<()> {
        if !value.is_finite() || value < 0.0 {
            anyhow::bail!("value must be a finite non-negative number");
        }
        match self.data_tx.try_send(DataMessage::Value {
            name,
            value,
            recorded_at: Instant::now(),
        }) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => {
                self.note_dropped_event();
                Ok(())
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                anyhow::bail!("metrics worker is not running")
            }
        }
    }

    fn record_ratio(&self, name: Arc<str>, numerator: f64, denominator: f64) -> anyhow::Result<()> {
        if !numerator.is_finite()
            || !denominator.is_finite()
            || numerator < 0.0
            || denominator < 0.0
        {
            anyhow::bail!("numerator/denominator must be finite non-negative numbers");
        }
        match self.data_tx.try_send(DataMessage::Ratio {
            name,
            numerator,
            denominator,
            recorded_at: Instant::now(),
        }) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => {
                self.note_dropped_event();
                Ok(())
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                anyhow::bail!("metrics worker is not running")
            }
        }
    }

    #[cfg(test)]
    fn snapshot_metric(&self, name: String) -> Option<MetricSnapshot> {
        let (resp_tx, resp_rx) = mpsc::channel();
        if self
            .control_tx
            .send(ControlMessage::SnapshotMetric {
                name,
                resp: resp_tx,
            })
            .is_err()
        {
            return None;
        }
        resp_rx.recv().ok().flatten()
    }

    #[cfg(test)]
    fn snapshot_all(&self) -> Vec<MetricSnapshot> {
        let (resp_tx, resp_rx) = mpsc::channel();
        if self
            .control_tx
            .send(ControlMessage::SnapshotAll { resp: resp_tx })
            .is_err()
        {
            return vec![];
        }
        resp_rx.recv().unwrap_or_default()
    }

    fn note_dropped_event(&self) {
        let dropped = self.dropped_events.fetch_add(1, Ordering::Relaxed) + 1;
        if dropped == 1 || dropped.is_power_of_two() {
            tracing::warn!(dropped_events = dropped, "performance metrics queue full");
        }
    }
}

impl Drop for RegistryInner {
    fn drop(&mut self) {
        let _ = self.control_tx.send(ControlMessage::Shutdown);
        if let Some(handle) = self.worker.lock().take() {
            let _ = handle.join();
        }
    }
}

fn worker_loop(
    control_rx: Receiver<ControlMessage>,
    data_rx: Receiver<DataMessage>,
    publish_interval: Duration,
    publisher: PublisherState,
) {
    const MAX_DATA_DRAIN_PER_TICK: usize = 2_048;
    // Synthetic local benchmark indicates worker ingestion can sustain roughly ~1-5M events/s
    // (and higher in single-metric hot-path cases) when per-metric series stay short.
    let mut state = WorkerState {
        publish_interval,
        metrics: HashMap::new(),
        publisher,
    };
    let mut control_disconnected = false;
    let mut data_disconnected = false;

    loop {
        while let Ok(msg) = control_rx.try_recv() {
            if !handle_control_message(msg, &mut state) {
                tracing::debug!(
                    "performance metrics worker exiting after shutdown control message"
                );
                return;
            }
        }

        match data_rx.recv_timeout(WORKER_POLL) {
            Ok(msg) => {
                handle_data_message(msg, &mut state);
                let mut drained = 1usize;
                // Bound per-tick data draining so control/publish work cannot be starved
                // when producers keep the queue continuously non-empty.
                while drained < MAX_DATA_DRAIN_PER_TICK {
                    let Ok(msg) = data_rx.try_recv() else {
                        break;
                    };
                    handle_data_message(msg, &mut state);
                    drained += 1;
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                data_disconnected = true;
                if !control_disconnected {
                    tracing::warn!(
                        "performance metrics worker data channel disconnected before shutdown"
                    );
                }
            }
        }

        loop {
            match control_rx.try_recv() {
                Ok(msg) => {
                    if !handle_control_message(msg, &mut state) {
                        tracing::debug!(
                            "performance metrics worker exiting after shutdown control message"
                        );
                        return;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    control_disconnected = true;
                    if !data_disconnected {
                        tracing::warn!(
                            "performance metrics worker control channel disconnected before shutdown"
                        );
                    }
                    break;
                }
            }
        }

        maybe_auto_publish(&mut state);
        if control_disconnected && data_disconnected {
            tracing::warn!(
                "performance metrics worker exiting after both channels disconnected without explicit shutdown"
            );
            return;
        }
    }
}

fn handle_control_message(msg: ControlMessage, state: &mut WorkerState) -> bool {
    match msg {
        ControlMessage::RegisterMetric { spec, resp } => {
            let result = register_metric_in_state(state, spec);
            let _ = resp.send(result);
        }
        ControlMessage::UnregisterMetric { name, resp } => {
            let result = if state.metrics.remove(&name).is_some() {
                if let Some(handles) = state.publisher.handles.remove(&name) {
                    for collector in handles.collectors() {
                        let _ = state
                            .publisher
                            .hierarchy
                            .get_metrics_registry()
                            .remove_metric(collector);
                    }
                }
                Ok(())
            } else {
                Err(anyhow::anyhow!("unknown metric '{}'", name))
            };
            let _ = resp.send(result);
        }
        #[cfg(test)]
        ControlMessage::SnapshotMetric { name, resp } => {
            let now = Instant::now();
            let snapshot = state.metrics.get_mut(&name).map(|entry| {
                let window = effective_window_duration(entry);
                prune_entry(now, window, entry);
                snapshot_entry(now, window, entry)
            });
            let _ = resp.send(snapshot);
        }
        #[cfg(test)]
        ControlMessage::SnapshotAll { resp } => {
            let now = Instant::now();
            let snapshots = state
                .metrics
                .values_mut()
                .map(|entry| {
                    let window = effective_window_duration(entry);
                    prune_entry(now, window, entry);
                    snapshot_entry(now, window, entry)
                })
                .collect::<Vec<_>>();
            let _ = resp.send(snapshots);
        }
        ControlMessage::Shutdown => return false,
    }

    true
}

fn handle_data_message(msg: DataMessage, state: &mut WorkerState) {
    match msg {
        DataMessage::Count {
            name,
            count,
            recorded_at,
        } => {
            if let Some(entry) = state.metrics.get_mut(name.as_ref()) {
                let MetricEntry {
                    spec,
                    state,
                    overflow_drops,
                } = entry;
                if let MetricState::Rate { counts } = state {
                    counts.push_back((recorded_at, count));
                    let dropped = trim_to_sample_limit(counts);
                    if dropped > 0 {
                        *overflow_drops = overflow_drops.saturating_add(dropped as u64);
                        if should_log_sample_overflow(*overflow_drops) {
                            tracing::warn!(
                                metric = %name,
                                metric_kind = "rate",
                                max_samples = MAX_SAMPLES_PER_METRIC,
                                dropped_samples = *overflow_drops,
                                configured_window_seconds = spec.window_seconds,
                                retained_span_seconds = sample_span_from_rate(counts)
                                    .map(|span| span.as_secs_f64()),
                                "metric sample buffer exceeded capacity; dropping oldest samples"
                            );
                        }
                    }
                }
            }
        }
        DataMessage::Value {
            name,
            value,
            recorded_at,
        } => {
            if let Some(entry) = state.metrics.get_mut(name.as_ref()) {
                let MetricEntry {
                    spec,
                    state,
                    overflow_drops,
                } = entry;
                if let MetricState::Distribution { values } = state {
                    values.push_back((recorded_at, value));
                    let dropped = trim_to_sample_limit(values);
                    if dropped > 0 {
                        *overflow_drops = overflow_drops.saturating_add(dropped as u64);
                        if should_log_sample_overflow(*overflow_drops) {
                            tracing::warn!(
                                metric = %name,
                                metric_kind = "distribution",
                                max_samples = MAX_SAMPLES_PER_METRIC,
                                dropped_samples = *overflow_drops,
                                configured_window_seconds = spec.window_seconds,
                                retained_span_seconds = sample_span_from_distribution(values)
                                    .map(|span| span.as_secs_f64()),
                                "metric sample buffer exceeded capacity; dropping oldest samples"
                            );
                        }
                    }
                }
            }
        }
        DataMessage::Ratio {
            name,
            numerator,
            denominator,
            recorded_at,
        } => {
            if let Some(entry) = state.metrics.get_mut(name.as_ref()) {
                let MetricEntry {
                    spec,
                    state,
                    overflow_drops,
                } = entry;
                if let MetricState::Ratio { pairs } = state {
                    pairs.push_back((recorded_at, numerator, denominator));
                    let dropped = trim_to_sample_limit(pairs);
                    if dropped > 0 {
                        *overflow_drops = overflow_drops.saturating_add(dropped as u64);
                        if should_log_sample_overflow(*overflow_drops) {
                            tracing::warn!(
                                metric = %name,
                                metric_kind = "ratio",
                                max_samples = MAX_SAMPLES_PER_METRIC,
                                dropped_samples = *overflow_drops,
                                configured_window_seconds = spec.window_seconds,
                                retained_span_seconds = sample_span_from_ratio(pairs)
                                    .map(|span| span.as_secs_f64()),
                                "metric sample buffer exceeded capacity; dropping oldest samples"
                            );
                        }
                    }
                }
            }
        }
    }
}

fn register_metric_in_state(
    state: &mut WorkerState,
    spec: PerformanceMetricSpec,
) -> anyhow::Result<()> {
    if spec.name.trim().is_empty() {
        anyhow::bail!("metric name must not be empty");
    }
    if state.metrics.contains_key(&spec.name) {
        anyhow::bail!("metric '{}' already registered", spec.name);
    }

    let state_variant = match spec.kind {
        PerformanceMetricKind::Rate => MetricState::Rate {
            counts: VecDeque::new(),
        },
        PerformanceMetricKind::Distribution => MetricState::Distribution {
            values: VecDeque::new(),
        },
        PerformanceMetricKind::Ratio => MetricState::Ratio {
            pairs: VecDeque::new(),
        },
    };

    let mut specs = state
        .metrics
        .values()
        .map(|entry| entry.spec.clone())
        .collect::<Vec<_>>();
    specs.push(spec.clone());
    validate_metric_name_collisions(&specs, &state.publisher.metric_prefix)?;
    let handles = build_metric_handles_for_spec(&spec, &state.publisher)?;
    state.publisher.handles.insert(spec.name.clone(), handles);

    state.metrics.insert(
        spec.name.clone(),
        MetricEntry {
            spec,
            state: state_variant,
            overflow_drops: 0,
        },
    );
    Ok(())
}

fn maybe_auto_publish(state: &mut WorkerState) {
    let should_publish = Instant::now() >= state.publisher.next_publish_at;
    if !should_publish {
        return;
    }

    let _ = publish_state(state);
    state.publisher.next_publish_at = Instant::now() + state.publish_interval;
}

fn publish_state(state: &mut WorkerState) -> anyhow::Result<()> {
    let publisher = &mut state.publisher;

    let now = Instant::now();
    for entry in state.metrics.values_mut() {
        let window = effective_window_duration(entry);
        prune_entry(now, window, entry);
        let snapshot = snapshot_entry(now, window, entry);

        if let Some(handle) = publisher.handles.get(&snapshot.name) {
            match handle {
                MetricHandles::Rate {
                    per_second,
                    quantiles,
                } => {
                    per_second.set(snapshot.rate_per_second.unwrap_or(0.0));
                    for (q, gauge) in quantiles {
                        gauge.set(find_quantile(snapshot.quantiles.as_slice(), *q));
                    }
                }
                MetricHandles::Distribution { avg, quantiles } => {
                    avg.set(snapshot.average.unwrap_or(0.0));
                    for (q, gauge) in quantiles {
                        gauge.set(find_quantile(snapshot.quantiles.as_slice(), *q));
                    }
                }
                MetricHandles::Ratio {
                    numerator,
                    denominator,
                    ratio,
                    quantiles,
                } => {
                    numerator.set(snapshot.numerator_sum.unwrap_or(0.0));
                    denominator.set(snapshot.denominator_sum.unwrap_or(0.0));
                    ratio.set(snapshot.ratio.unwrap_or(0.0));
                    for (q, gauge) in quantiles {
                        gauge.set(find_quantile(snapshot.quantiles.as_slice(), *q));
                    }
                }
            }
        }
    }

    Ok(())
}

fn build_metric_handles_for_spec(
    spec: &PerformanceMetricSpec,
    publisher: &PublisherState,
) -> anyhow::Result<MetricHandles> {
    let base = format!("{}_{}", publisher.metric_prefix, spec.name);
    let label_refs = publisher
        .labels
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect::<Vec<_>>();

    match spec.kind {
        PerformanceMetricKind::Rate => {
            let per_second = create_metric::<prometheus::Gauge, dyn MetricsHierarchy>(
                publisher.hierarchy.as_ref(),
                &format!("{}_{}", base, SUFFIX_PER_SECOND),
                "Sliding-window throughput per second",
                &label_refs,
                None,
                None,
            )?;
            let quantiles = spec
                .quantiles
                .iter()
                .map(|q| {
                    let quantile_value = quantile_label_value(*q);
                    let mut quantile_labels = label_refs.clone();
                    quantile_labels.push((LABEL_QUANTILE, quantile_value.as_str()));
                    create_metric::<prometheus::Gauge, dyn MetricsHierarchy>(
                        publisher.hierarchy.as_ref(),
                        &format!("{}_{}_{}", base, SUFFIX_PER_SECOND, SUFFIX_QUANTILE),
                        "Sliding-window throughput quantile",
                        &quantile_labels,
                        None,
                        None,
                    )
                    .map(|g| (*q, g))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok(MetricHandles::Rate {
                per_second,
                quantiles,
            })
        }
        PerformanceMetricKind::Distribution => {
            let avg = create_metric::<prometheus::Gauge, dyn MetricsHierarchy>(
                publisher.hierarchy.as_ref(),
                &format!("{}_{}", base, SUFFIX_AVG),
                "Sliding-window average value",
                &label_refs,
                None,
                None,
            )?;
            let quantiles = spec
                .quantiles
                .iter()
                .map(|q| {
                    let quantile_value = quantile_label_value(*q);
                    let mut quantile_labels = label_refs.clone();
                    quantile_labels.push((LABEL_QUANTILE, quantile_value.as_str()));
                    create_metric::<prometheus::Gauge, dyn MetricsHierarchy>(
                        publisher.hierarchy.as_ref(),
                        &base,
                        "Sliding-window quantile value",
                        &quantile_labels,
                        None,
                        None,
                    )
                    .map(|g| (*q, g))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok(MetricHandles::Distribution { avg, quantiles })
        }
        PerformanceMetricKind::Ratio => {
            let numerator = create_metric::<prometheus::Gauge, dyn MetricsHierarchy>(
                publisher.hierarchy.as_ref(),
                &format!("{}_{}", base, SUFFIX_NUMERATOR),
                "Sliding-window numerator sum",
                &label_refs,
                None,
                None,
            )?;
            let denominator = create_metric::<prometheus::Gauge, dyn MetricsHierarchy>(
                publisher.hierarchy.as_ref(),
                &format!("{}_{}", base, SUFFIX_DENOMINATOR),
                "Sliding-window denominator sum",
                &label_refs,
                None,
                None,
            )?;
            let ratio = create_metric::<prometheus::Gauge, dyn MetricsHierarchy>(
                publisher.hierarchy.as_ref(),
                &format!("{}_{}", base, SUFFIX_RATIO),
                "Sliding-window weighted ratio",
                &label_refs,
                None,
                None,
            )?;
            let quantiles = spec
                .quantiles
                .iter()
                .map(|q| {
                    let quantile_value = quantile_label_value(*q);
                    let mut quantile_labels = label_refs.clone();
                    quantile_labels.push((LABEL_QUANTILE, quantile_value.as_str()));
                    create_metric::<prometheus::Gauge, dyn MetricsHierarchy>(
                        publisher.hierarchy.as_ref(),
                        &format!("{}_{}_{}", base, SUFFIX_RATIO, SUFFIX_QUANTILE),
                        "Sliding-window ratio quantile",
                        &quantile_labels,
                        None,
                        None,
                    )
                    .map(|g| (*q, g))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok(MetricHandles::Ratio {
                numerator,
                denominator,
                ratio,
                quantiles,
            })
        }
    }
}

fn sampled_count(tokens: u64, sample_rate: f64, rng: &mut SmallRng) -> u64 {
    if tokens == 0 || sample_rate <= 0.0 {
        return 0;
    }
    let expected = (tokens as f64 * sample_rate).min(50.0);
    let base = expected.floor() as u64;
    let frac = expected - base as f64;
    if frac > 0.0 && Rng::random::<f64>(rng) < frac {
        base + 1
    } else {
        base
    }
}

fn normalize_quantiles(quantiles: Vec<f64>) -> Vec<f64> {
    let mut normalized = quantiles
        .into_iter()
        .filter(|q| q.is_finite() && *q >= 0.0 && *q <= 1.0)
        .collect::<Vec<_>>();
    normalized.sort_by(f64::total_cmp);
    normalized.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    normalized
}

fn should_log_sample_overflow(overflow_count: u64) -> bool {
    overflow_count == 1
        || overflow_count.is_power_of_two()
        || (overflow_count >= SAMPLE_OVERFLOW_LOG_INTERVAL
            && overflow_count.is_multiple_of(SAMPLE_OVERFLOW_LOG_INTERVAL))
}

fn trim_to_sample_limit<T>(samples: &mut VecDeque<T>) -> usize {
    if samples.len() <= MAX_SAMPLES_PER_METRIC {
        return 0;
    }
    let overflow = samples.len() - MAX_SAMPLES_PER_METRIC;
    for _ in 0..overflow {
        samples.pop_front();
    }
    overflow
}

fn span_between(oldest: Option<Instant>, newest: Option<Instant>) -> Option<Duration> {
    match (oldest, newest) {
        (Some(oldest), Some(newest)) => {
            Some(newest.checked_duration_since(oldest).unwrap_or_default())
        }
        _ => None,
    }
}

fn sample_span_from_rate(samples: &VecDeque<(Instant, u64)>) -> Option<Duration> {
    span_between(
        samples.front().map(|(ts, _)| *ts),
        samples.back().map(|(ts, _)| *ts),
    )
}

fn sample_span_from_distribution(samples: &VecDeque<(Instant, f64)>) -> Option<Duration> {
    span_between(
        samples.front().map(|(ts, _)| *ts),
        samples.back().map(|(ts, _)| *ts),
    )
}

fn sample_span_from_ratio(samples: &VecDeque<(Instant, f64, f64)>) -> Option<Duration> {
    span_between(
        samples.front().map(|(ts, _, _)| *ts),
        samples.back().map(|(ts, _, _)| *ts),
    )
}

fn sample_span(state: &MetricState) -> Option<Duration> {
    match state {
        MetricState::Rate { counts } => sample_span_from_rate(counts),
        MetricState::Distribution { values } => sample_span_from_distribution(values),
        MetricState::Ratio { pairs } => sample_span_from_ratio(pairs),
    }
}

fn effective_window_duration(entry: &MetricEntry) -> Duration {
    let configured_window = Duration::from_secs_f64(entry.spec.window_seconds);
    if !sample_buffer_is_saturated(&entry.state) {
        return configured_window;
    }
    let span = sample_span(&entry.state).unwrap_or_default();
    configured_window.max(span).max(Duration::from_millis(100))
}

fn sample_buffer_is_saturated(state: &MetricState) -> bool {
    match state {
        MetricState::Rate { counts } => counts.len() >= MAX_SAMPLES_PER_METRIC,
        MetricState::Distribution { values } => values.len() >= MAX_SAMPLES_PER_METRIC,
        MetricState::Ratio { pairs } => pairs.len() >= MAX_SAMPLES_PER_METRIC,
    }
}

fn prune_entry(now: Instant, window_duration: Duration, entry: &mut MetricEntry) {
    let cutoff = now.checked_sub(window_duration).unwrap_or(now);
    match &mut entry.state {
        MetricState::Rate { counts } => {
            if counts.back().is_some_and(|(ts, _)| *ts < cutoff) {
                counts.clear();
            } else {
                while counts.front().is_some_and(|(ts, _)| *ts < cutoff) {
                    counts.pop_front();
                }
            }
        }
        MetricState::Distribution { values } => {
            if values.back().is_some_and(|(ts, _)| *ts < cutoff) {
                values.clear();
            } else {
                while values.front().is_some_and(|(ts, _)| *ts < cutoff) {
                    values.pop_front();
                }
            }
        }
        MetricState::Ratio { pairs } => {
            if pairs.back().is_some_and(|(ts, _, _)| *ts < cutoff) {
                pairs.clear();
            } else {
                while pairs.front().is_some_and(|(ts, _, _)| *ts < cutoff) {
                    pairs.pop_front();
                }
            }
        }
    }
}

fn snapshot_entry(now: Instant, window_duration: Duration, entry: &MetricEntry) -> MetricSnapshot {
    let window_seconds = window_duration.as_secs_f64().max(0.1);
    match &entry.state {
        MetricState::Rate { counts } => {
            let total: u64 = counts.iter().map(|(_, c)| *c).sum();
            let sample_period = entry.spec.sample_period_seconds.unwrap_or(1.0);
            let samples = build_rate_samples(now, counts, window_seconds, sample_period);
            let quantiles = compute_quantiles(samples, entry.spec.quantiles.as_slice());

            MetricSnapshot {
                name: entry.spec.name.clone(),
                kind: entry.spec.kind,
                rate_per_second: Some(total as f64 / window_seconds),
                average: None,
                quantiles,
                numerator_sum: None,
                denominator_sum: None,
                ratio: None,
            }
        }
        MetricState::Distribution { values } => {
            let series = values.iter().map(|(_, v)| *v).collect::<Vec<_>>();
            let average = if series.is_empty() {
                0.0
            } else {
                series.iter().sum::<f64>() / series.len() as f64
            };
            let quantiles = compute_quantiles(series, entry.spec.quantiles.as_slice());

            MetricSnapshot {
                name: entry.spec.name.clone(),
                kind: entry.spec.kind,
                rate_per_second: None,
                average: Some(average),
                quantiles,
                numerator_sum: None,
                denominator_sum: None,
                ratio: None,
            }
        }
        MetricState::Ratio { pairs } => {
            let numerator_sum: f64 = pairs.iter().map(|(_, n, _)| *n).sum();
            let denominator_sum: f64 = pairs.iter().map(|(_, _, d)| *d).sum();
            let ratio = if denominator_sum > 0.0 {
                numerator_sum / denominator_sum
            } else {
                0.0
            };
            let sample_ratios = pairs
                .iter()
                .filter_map(|(_, n, d)| if *d > 0.0 { Some(*n / *d) } else { None })
                .collect::<Vec<_>>();
            let quantiles = compute_quantiles(sample_ratios, entry.spec.quantiles.as_slice());

            MetricSnapshot {
                name: entry.spec.name.clone(),
                kind: entry.spec.kind,
                rate_per_second: None,
                average: None,
                quantiles,
                numerator_sum: Some(numerator_sum),
                denominator_sum: Some(denominator_sum),
                ratio: Some(ratio),
            }
        }
    }
}

fn build_rate_samples(
    now: Instant,
    counts: &VecDeque<(Instant, u64)>,
    window_seconds: f64,
    sample_period_seconds: f64,
) -> Vec<f64> {
    let window_seconds = window_seconds.max(0.1);
    let sample_period = sample_period_seconds.max(0.1);
    let bins = (window_seconds / sample_period).ceil().max(1.0) as usize;
    let start = now - Duration::from_secs_f64(window_seconds);

    let mut sums = vec![0u64; bins];
    for (ts, count) in counts {
        if *ts < start || *ts > now {
            continue;
        }
        let offset = ts.duration_since(start).as_secs_f64();
        let idx = (offset / sample_period).floor() as usize;
        let idx = idx.min(bins - 1);
        sums[idx] = sums[idx].saturating_add(*count);
    }

    sums.into_iter()
        .enumerate()
        .map(|(idx, c)| {
            let bucket_start = idx as f64 * sample_period;
            let bucket_end = ((idx + 1) as f64 * sample_period).min(window_seconds);
            let bucket_width = (bucket_end - bucket_start).max(1e-9);
            c as f64 / bucket_width
        })
        .collect::<Vec<_>>()
}

fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let clamped = p.clamp(0.0, 1.0);
    let idx = ((sorted_values.len() - 1) as f64 * clamped).round() as usize;
    sorted_values[idx.min(sorted_values.len() - 1)]
}

fn percentile_select(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let clamped = p.clamp(0.0, 1.0);
    let idx = ((values.len() - 1) as f64 * clamped).round() as usize;
    let idx = idx.min(values.len() - 1);
    let (_, nth, _) = values.select_nth_unstable_by(idx, f64::total_cmp);
    *nth
}

fn compute_quantiles(mut values: Vec<f64>, quantiles: &[f64]) -> Vec<(f64, f64)> {
    if quantiles.is_empty() {
        return vec![];
    }
    if values.is_empty() {
        return quantiles.iter().map(|q| (*q, 0.0)).collect::<Vec<_>>();
    }

    let should_use_select = {
        let value_count = values.len();
        let quantile_count = quantiles.len();
        if quantile_count <= 1 {
            true
        } else {
            // Empirical threshold: repeated select tends to win when the number of quantiles
            // is lower than ~70% of log2(sample_count); otherwise a single sort is cheaper.
            let threshold = ((value_count.max(2) as f64).log2() * 0.7).floor() as usize;
            quantile_count <= threshold.max(1)
        }
    };

    if should_use_select {
        quantiles
            .iter()
            .map(|q| (*q, percentile_select(&mut values, *q)))
            .collect::<Vec<_>>()
    } else {
        values.sort_unstable_by(f64::total_cmp);
        quantiles
            .iter()
            .map(|q| (*q, percentile(&values, *q)))
            .collect::<Vec<_>>()
    }
}

fn quantile_label_value(q: f64) -> String {
    let mut value = format!("{:.6}", q.clamp(0.0, 1.0));
    while value.ends_with('0') {
        value.pop();
    }
    if value.ends_with('.') {
        value.push('0');
    }
    value
}

fn find_quantile(values: &[(f64, f64)], q: f64) -> f64 {
    values
        .iter()
        .find(|(qq, _)| (*qq - q).abs() < 1e-9)
        .map(|(_, v)| *v)
        .unwrap_or(0.0)
}

fn validate_metric_name_collisions(
    specs: &[PerformanceMetricSpec],
    metric_prefix: &str,
) -> anyhow::Result<()> {
    let mut full_names = HashSet::new();
    for spec in specs {
        let base = format!("{}_{}", metric_prefix, spec.name);
        let mut names = Vec::new();
        match spec.kind {
            PerformanceMetricKind::Rate => {
                names.push(format!("{}_{}", base, SUFFIX_PER_SECOND));
                if !spec.quantiles.is_empty() {
                    names.push(format!(
                        "{}_{}_{}",
                        base, SUFFIX_PER_SECOND, SUFFIX_QUANTILE
                    ));
                }
            }
            PerformanceMetricKind::Distribution => {
                names.push(format!("{}_{}", base, SUFFIX_AVG));
                if !spec.quantiles.is_empty() {
                    names.push(base.clone());
                }
            }
            PerformanceMetricKind::Ratio => {
                names.push(format!("{}_{}", base, SUFFIX_NUMERATOR));
                names.push(format!("{}_{}", base, SUFFIX_DENOMINATOR));
                names.push(format!("{}_{}", base, SUFFIX_RATIO));
                if !spec.quantiles.is_empty() {
                    names.push(format!("{}_{}_{}", base, SUFFIX_RATIO, SUFFIX_QUANTILE));
                }
            }
        }

        for name in names {
            let canonical = build_component_metric_name(&name);
            if !full_names.insert(canonical.clone()) {
                anyhow::bail!(
                    "prometheus metric name collision after sanitization: '{}'; rename metric or quantiles",
                    canonical
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{MetricsHierarchy, MetricsRegistry};

    struct TestHierarchy {
        registry: MetricsRegistry,
    }

    impl TestHierarchy {
        fn new() -> Self {
            Self {
                registry: MetricsRegistry::new(),
            }
        }
    }

    impl MetricsHierarchy for TestHierarchy {
        fn basename(&self) -> String {
            "test_component".to_string()
        }

        fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
            vec![]
        }

        fn get_metrics_registry(&self) -> &MetricsRegistry {
            &self.registry
        }
    }

    #[test]
    fn ratio_snapshot_has_absolute_and_relative() {
        let hierarchy = Arc::new(TestHierarchy::new());
        let tracker = PerformanceMetricsRegistry::new_attached(
            Duration::from_secs(5),
            hierarchy,
            "performance",
            &[] as &[(&str, &str)],
        )
        .unwrap();
        let kv = tracker
            .new_ratio_metric("kv", vec![0.5, 0.9, 0.99], None)
            .unwrap();
        kv.record_ratio(80.0, 100.0).unwrap();
        kv.record_ratio(20.0, 50.0).unwrap();

        let s = kv.snapshot_for_test().unwrap();
        assert_eq!(s.kind, PerformanceMetricKind::Ratio);
        assert!((s.numerator_sum.unwrap_or_default() - 100.0).abs() < 1e-9);
        assert!((s.denominator_sum.unwrap_or_default() - 150.0).abs() < 1e-9);
        assert!((s.ratio.unwrap_or_default() - (100.0 / 150.0)).abs() < 1e-9);
        assert_eq!(s.quantiles.len(), 3);
    }

    #[test]
    fn rate_snapshot_has_quantiles() {
        let hierarchy = Arc::new(TestHierarchy::new());
        let tracker = PerformanceMetricsRegistry::new_attached(
            Duration::from_secs(5),
            hierarchy,
            "performance",
            &[] as &[(&str, &str)],
        )
        .unwrap();
        let request_gps = tracker
            .new_rate_metric("request_gps", vec![0.5, 0.9, 0.99], Some(1.0), None)
            .unwrap();
        request_gps.record_count(8).unwrap();
        request_gps.record_count(12).unwrap();
        let deadline = Instant::now() + Duration::from_millis(100);
        loop {
            let s = request_gps.snapshot_for_test().unwrap();
            assert_eq!(s.kind, PerformanceMetricKind::Rate);
            assert_eq!(s.quantiles.len(), 3);
            if s.rate_per_second.unwrap_or_default() > 0.0 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "rate_per_second did not become positive before timeout"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn rate_samples_use_trailing_bucket_actual_width() {
        let now = Instant::now();
        let window_seconds = 2.5;
        let sample_period_seconds = 1.0;
        let start = now - Duration::from_secs_f64(window_seconds);

        let mut counts = VecDeque::new();
        counts.push_back((start + Duration::from_millis(100), 10));
        counts.push_back((start + Duration::from_millis(2200), 10));

        let samples = build_rate_samples(now, &counts, window_seconds, sample_period_seconds);
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 10.0).abs() < 1e-9);
        assert!(samples[1].abs() < 1e-9);
        assert!((samples[2] - 20.0).abs() < 1e-9);
    }

    #[test]
    fn build_rate_samples_includes_window_boundaries() {
        let now = Instant::now();
        let window_seconds = 2.0;
        let sample_period_seconds = 1.0;
        let start = now - Duration::from_secs_f64(window_seconds);

        let mut counts = VecDeque::new();
        counts.push_back((start, 1)); // include lower bound
        counts.push_back((now, 1)); // include upper bound

        let samples = build_rate_samples(now, &counts, window_seconds, sample_period_seconds);
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 1.0).abs() < 1e-9);
        assert!((samples[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn quantiles_preserve_distinct_values() {
        let spec = PerformanceMetricSpec::distribution("ttft", vec![0.994, 0.995, 0.999], None);
        let labels = spec
            .quantiles
            .iter()
            .map(|q| quantile_label_value(*q))
            .collect::<HashSet<_>>();
        assert_eq!(labels.len(), spec.quantiles.len());
    }

    #[test]
    fn attach_without_registered_metrics_succeeds() {
        let hierarchy = Arc::new(TestHierarchy::new());
        let tracker = PerformanceMetricsRegistry::new_attached(
            Duration::from_secs(5),
            hierarchy.clone(),
            "performance",
            &[] as &[(&str, &str)],
        )
        .unwrap();
        assert_eq!(tracker.snapshot_all_for_test().len(), 0);
    }

    #[test]
    fn register_after_attach_and_unregister_work() {
        let hierarchy = Arc::new(TestHierarchy::new());
        let tracker = PerformanceMetricsRegistry::new_attached(
            Duration::from_secs(5),
            hierarchy.clone(),
            "performance",
            &[] as &[(&str, &str)],
        )
        .unwrap();
        let _tps = tracker
            .new_rate_metric("tps", vec![0.5], Some(1.0), None)
            .unwrap();

        let ttft = tracker
            .new_distribution_metric("ttft", vec![0.5], None)
            .unwrap();
        ttft.record_value(0.7).unwrap();
        assert!(ttft.snapshot_for_test().is_some());
        assert!(
            hierarchy
                .metrics()
                .prometheus_expfmt()
                .unwrap()
                .contains("performance_ttft_avg")
        );

        tracker.unregister_metric("ttft").unwrap();
        assert!(ttft.snapshot_for_test().is_none());
        assert!(
            !hierarchy
                .metrics()
                .prometheus_expfmt()
                .unwrap()
                .contains("performance_ttft_avg")
        );
    }

    #[test]
    fn register_validates_against_existing_suffixed_sanitized_names() {
        let hierarchy = Arc::new(TestHierarchy::new());
        let tracker = PerformanceMetricsRegistry::new_attached(
            Duration::from_secs(5),
            hierarchy.clone(),
            "baseten_frontend",
            &[] as &[(&str, &str)],
        )
        .unwrap();

        let _dist = tracker
            .new_distribution_metric("foo_per_second", vec![0.1, 0.5, 0.99, 0.999], None)
            .unwrap();

        let before = hierarchy.metrics().prometheus_expfmt().unwrap();
        assert!(before.contains("baseten_frontend_foo_per_second_avg"));
        assert!(before.contains("baseten_frontend_foo_per_second{"));
        assert!(before.contains("quantile=\"0.1\""));
        assert!(before.contains("quantile=\"0.5\""));
        assert!(before.contains("quantile=\"0.99\""));
        assert!(before.contains("quantile=\"0.999\""));

        let err = match tracker.new_rate_metric("foo", vec![0.1, 0.5, 0.99, 0.999], Some(1.0), None)
        {
            Ok(_) => panic!("expected collision error"),
            Err(err) => err.to_string(),
        };
        assert!(
            err.contains("prometheus metric name collision after sanitization"),
            "{err}"
        );

        let after = hierarchy.metrics().prometheus_expfmt().unwrap();
        assert!(after.contains("baseten_frontend_foo_per_second_avg"));
        assert!(after.contains("baseten_frontend_foo_per_second{"));
        assert!(after.contains("quantile=\"0.1\""));
        assert!(after.contains("quantile=\"0.5\""));
        assert!(after.contains("quantile=\"0.99\""));
        assert!(after.contains("quantile=\"0.999\""));
        assert!(!after.contains("baseten_frontend_foo_per_second_per_second"));
        assert!(!after.contains("baseten_frontend_foo_per_second_per_second_quantile"));
    }

    #[test]
    fn attached_metrics_export_expected_hierarchy_naming() {
        let hierarchy = Arc::new(TestHierarchy::new());
        let tracker = PerformanceMetricsRegistry::new_attached(
            Duration::from_secs(5),
            hierarchy.clone(),
            "baseten_frontend",
            &[] as &[(&str, &str)],
        )
        .unwrap();

        let _requests = tracker
            .new_rate_metric("requests", vec![0.1, 0.5, 0.99, 0.999], Some(1.0), None)
            .unwrap();

        let output = hierarchy.metrics().prometheus_expfmt().unwrap();
        assert!(output.contains("component_baseten_frontend_requests_per_second"));
        assert!(output.contains("component_baseten_frontend_requests_per_second_quantile{"));
        assert!(output.contains("quantile=\"0.1\""));
        assert!(output.contains("quantile=\"0.5\""));
        assert!(output.contains("quantile=\"0.99\""));
        assert!(output.contains("quantile=\"0.999\""));
    }

    #[test]
    fn request_metrics_new_request_records_request_and_input_tokens() {
        let hierarchy = Arc::new(TestHierarchy::new());
        let registry = PerformanceMetricsRegistry::new_attached(
            Duration::from_secs(5),
            hierarchy,
            "baseten_frontend",
            &[] as &[(&str, &str)],
        )
        .unwrap();

        let factory = RequestMetricsFactory::new(
            &registry,
            "request_metrics",
            RequestMetricsOptions::default(),
        )
        .unwrap();
        let _request = factory.new_request(123);

        let deadline = Instant::now() + Duration::from_millis(100);
        loop {
            let request_snapshot = factory.inner.request.snapshot_for_test().unwrap();
            if request_snapshot.rate_per_second.unwrap_or_default() > 0.0 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "request rate did not become positive before timeout"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn request_metrics_first_token_records_ttft_per_input_token() {
        let hierarchy = Arc::new(TestHierarchy::new());
        let registry = PerformanceMetricsRegistry::new_attached(
            Duration::from_secs(5),
            hierarchy,
            "baseten_frontend",
            &[] as &[(&str, &str)],
        )
        .unwrap();

        let factory = RequestMetricsFactory::new(
            &registry,
            "request_metrics",
            RequestMetricsOptions::default(),
        )
        .unwrap();
        let mut request = factory.new_request(100);
        std::thread::sleep(Duration::from_millis(1));
        request.record_tokens(1, None).unwrap();

        let s = factory
            .inner
            .ttft_per_input_token
            .snapshot_for_test()
            .unwrap();
        assert_eq!(s.kind, PerformanceMetricKind::Distribution);
        assert!(s.average.unwrap_or_default() >= 0.0);
    }

    #[test]
    fn request_metrics_records_input_tokens_on_first_record_tokens_call() {
        let hierarchy = Arc::new(TestHierarchy::new());
        let registry = PerformanceMetricsRegistry::new_attached(
            Duration::from_secs(5),
            hierarchy,
            "baseten_frontend",
            &[] as &[(&str, &str)],
        )
        .unwrap();

        let factory = RequestMetricsFactory::new(
            &registry,
            "request_metrics",
            RequestMetricsOptions::default(),
        )
        .unwrap();
        let mut request = factory.new_request(123);
        request.record_tokens(1, None).unwrap();

        let deadline = Instant::now() + Duration::from_millis(100);
        loop {
            let input_tokens_snapshot = factory.inner.input_tokens.snapshot_for_test().unwrap();
            if input_tokens_snapshot.rate_per_second.unwrap_or_default() > 0.0 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "input token rate did not become positive before timeout"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn request_metrics_can_infer_input_tokens_from_first_total_tokens() {
        let hierarchy = Arc::new(TestHierarchy::new());
        let registry = PerformanceMetricsRegistry::new_attached(
            Duration::from_secs(5),
            hierarchy,
            "baseten_frontend",
            &[] as &[(&str, &str)],
        )
        .unwrap();

        let factory = RequestMetricsFactory::new(
            &registry,
            "request_metrics",
            RequestMetricsOptions::default(),
        )
        .unwrap();
        let mut request = factory.new_request(0);
        request.record_tokens(129, None).unwrap();

        let deadline = Instant::now() + Duration::from_millis(100);
        loop {
            let input_tokens_snapshot = factory.inner.input_tokens.snapshot_for_test().unwrap();
            let ttft_per_input_token_snapshot = factory
                .inner
                .ttft_per_input_token
                .snapshot_for_test()
                .unwrap();
            if input_tokens_snapshot.rate_per_second.unwrap_or_default() > 0.0
                && ttft_per_input_token_snapshot.average.unwrap_or_default() > 0.0
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "inferred input token metrics did not become positive before timeout"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn request_metrics_cached_tokens_record_net_new_once_before_first_token() {
        let hierarchy = Arc::new(TestHierarchy::new());
        let registry = PerformanceMetricsRegistry::new_attached(
            Duration::from_secs(5),
            hierarchy,
            "baseten_frontend",
            &[] as &[(&str, &str)],
        )
        .unwrap();

        let factory = RequestMetricsFactory::new(
            &registry,
            "request_metrics",
            RequestMetricsOptions::default(),
        )
        .unwrap();
        let mut request = factory.new_request(100);
        std::thread::sleep(Duration::from_millis(1));
        request.record_tokens(1, Some(60)).unwrap();
        request.record_tokens(2, Some(50)).unwrap();

        let deadline = Instant::now() + Duration::from_millis(100);
        loop {
            let net_new_tokens_snapshot = factory
                .inner
                .net_new_input_tokens
                .snapshot_for_test()
                .unwrap();
            let ttft_per_net_new_snapshot = factory
                .inner
                .ttft_per_net_new_input_token
                .snapshot_for_test()
                .unwrap();
            if net_new_tokens_snapshot.rate_per_second.unwrap_or_default() > 0.0
                && ttft_per_net_new_snapshot.average.unwrap_or_default() > 0.0
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "net-new token metrics did not become positive before timeout"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn compute_quantiles_select_and_sort_paths_match_baseline() {
        let values = (0..2048)
            .map(|i| ((i * 37) % 997) as f64 / 3.0)
            .collect::<Vec<_>>();

        let quantiles_select = vec![0.1, 0.5, 0.9];
        let quantiles_sort = vec![0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999];

        let baseline = |qs: &[f64]| {
            let mut sorted = values.clone();
            sorted.sort_unstable_by(f64::total_cmp);
            qs.iter()
                .map(|q| (*q, percentile(sorted.as_slice(), *q)))
                .collect::<Vec<_>>()
        };

        let selected = compute_quantiles(values.clone(), quantiles_select.as_slice());
        let sorted = compute_quantiles(values.clone(), quantiles_sort.as_slice());

        assert_eq!(selected, baseline(quantiles_select.as_slice()));
        assert_eq!(sorted, baseline(quantiles_sort.as_slice()));
    }
}
