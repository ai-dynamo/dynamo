// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic sliding-window performance metrics tracker.
//!
//! This module provides:
//! - factory-style metric handles (`new_*_metric`)
//! - per-kind recording (`rate`, `distribution`, `ratio`)
//! - snapshot computation with optional quantiles
//! - optional Prometheus gauge publisher

use super::{MetricsHierarchy, create_metric, prometheus_names::build_component_metric_name};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const SUFFIX_PER_SECOND: &str = "per_second";
const SUFFIX_AVG: &str = "avg";
const SUFFIX_NUMERATOR: &str = "numerator";
const SUFFIX_DENOMINATOR: &str = "denominator";
const SUFFIX_RATIO: &str = "ratio";

/// Supported metric behaviors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceMetricKind {
    Rate,
    Distribution,
    Ratio,
}

/// Declarative metric configuration used by tracker and publisher.
#[derive(Debug, Clone)]
pub struct PerformanceMetricSpec {
    pub name: String,
    pub kind: PerformanceMetricKind,
    pub quantiles: Vec<f64>,
    pub sample_period_seconds: Option<f64>,
}

impl PerformanceMetricSpec {
    pub fn rate(
        name: impl Into<String>,
        quantiles: Vec<f64>,
        sample_period_seconds: Option<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            kind: PerformanceMetricKind::Rate,
            quantiles: normalize_quantiles(quantiles),
            sample_period_seconds: sample_period_seconds
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(|v| v.max(0.1)),
        }
    }

    pub fn distribution(name: impl Into<String>, quantiles: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            kind: PerformanceMetricKind::Distribution,
            quantiles: normalize_quantiles(quantiles),
            sample_period_seconds: None,
        }
    }

    pub fn ratio(name: impl Into<String>, quantiles: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            kind: PerformanceMetricKind::Ratio,
            quantiles: normalize_quantiles(quantiles),
            sample_period_seconds: None,
        }
    }
}

/// Sliding-window snapshot for a single metric.
#[derive(Debug, Clone)]
pub struct MetricSnapshot {
    pub name: String,
    pub kind: PerformanceMetricKind,
    pub rate_per_second: Option<f64>,
    pub average: Option<f64>,
    pub quantiles: Vec<(f64, f64)>,
    pub numerator_sum: Option<f64>,
    pub denominator_sum: Option<f64>,
    pub ratio: Option<f64>,
}

/// Sliding-window performance metric tracker.
///
/// Use `new_*_metric` methods to create handles and record values through them.
#[derive(Debug)]
pub struct PerformanceMetricsRegistry {
    inner: Arc<PerformanceMetricsRegistryInner>,
}

#[derive(Debug)]
struct PerformanceMetricsRegistryInner {
    window_duration: Duration,
    metrics: Mutex<HashMap<String, MetricEntry>>,
}

/// Attached metrics session with Prometheus publisher handles.
pub struct AttachedPerformanceMetrics {
    tracker: Arc<PerformanceMetricsRegistryInner>,
    publisher: PerformanceMetricsPrometheusPublisher,
}

/// Shared internal metric handle.
#[derive(Clone)]
struct PerformanceMetricHandle {
    tracker: Arc<PerformanceMetricsRegistryInner>,
    metric_name: String,
}

impl PerformanceMetricHandle {
    /// Metric name.
    pub fn name(&self) -> &str {
        &self.metric_name
    }

    fn record_count(&self, count: u64) -> anyhow::Result<()> {
        self.tracker.record_count_internal(&self.metric_name, count)
    }

    fn record_value(&self, value: f64) -> anyhow::Result<()> {
        self.tracker.record_value_internal(&self.metric_name, value)
    }

    fn record_ratio(&self, numerator: f64, denominator: f64) -> anyhow::Result<()> {
        self.tracker
            .record_ratio_internal(&self.metric_name, numerator, denominator)
    }

    fn snapshot(&self) -> Option<MetricSnapshot> {
        self.tracker.snapshot_metric(&self.metric_name)
    }
}

impl AttachedPerformanceMetrics {
    /// Window duration used for snapshot calculations.
    pub fn window_duration(&self) -> Duration {
        self.tracker.window_duration()
    }

    /// Snapshot all registered metrics.
    pub fn snapshot_all(&self) -> Vec<MetricSnapshot> {
        self.tracker.snapshot_all()
    }

    /// Update registered gauges from latest tracker snapshots.
    pub fn publish(&self) {
        self.publisher.publish();
    }

    /// Start periodic publish loop.
    pub fn start_auto_publish(&self, interval: Duration) {
        self.publisher.start_auto_publish(interval);
    }

    /// Stop periodic publish loop.
    pub fn stop_auto_publish(&self) {
        self.publisher.stop_auto_publish();
    }
}

/// Strongly typed handle for rate metrics.
#[derive(Clone)]
pub struct RateMetricHandle {
    inner: PerformanceMetricHandle,
}

impl RateMetricHandle {
    /// Metric name.
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Record units/events for a rate metric.
    pub fn record_count(&self, count: u64) -> anyhow::Result<()> {
        self.inner.record_count(count)
    }

    /// Get current snapshot.
    pub fn snapshot(&self) -> Option<MetricSnapshot> {
        self.inner.snapshot()
    }
}

/// Strongly typed handle for distribution metrics.
#[derive(Clone)]
pub struct DistributionMetricHandle {
    inner: PerformanceMetricHandle,
}

impl DistributionMetricHandle {
    /// Metric name.
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Record value sample.
    pub fn record_value(&self, value: f64) -> anyhow::Result<()> {
        self.inner.record_value(value)
    }

    /// Get current snapshot.
    pub fn snapshot(&self) -> Option<MetricSnapshot> {
        self.inner.snapshot()
    }
}

/// Strongly typed handle for ratio metrics.
#[derive(Clone)]
pub struct RatioMetricHandle {
    inner: PerformanceMetricHandle,
}

impl RatioMetricHandle {
    /// Metric name.
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Record (numerator, denominator) sample.
    pub fn record_ratio(&self, numerator: f64, denominator: f64) -> anyhow::Result<()> {
        self.inner.record_ratio(numerator, denominator)
    }

    /// Get current snapshot.
    pub fn snapshot(&self) -> Option<MetricSnapshot> {
        self.inner.snapshot()
    }
}

#[derive(Debug)]
struct MetricEntry {
    spec: PerformanceMetricSpec,
    state: MetricState,
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

impl PerformanceMetricsRegistry {
    /// Create tracker with bounded sliding window duration.
    pub fn new(window_duration: Duration) -> Self {
        Self {
            inner: Arc::new(PerformanceMetricsRegistryInner {
                window_duration: window_duration.max(Duration::from_secs(1)),
                metrics: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Window duration used for snapshot calculations.
    pub fn window_duration(&self) -> Duration {
        self.inner.window_duration()
    }

    /// Clear all in-memory metric state.
    pub fn clear(&self) {
        self.inner.clear();
    }

    fn register_and_build_handle(
        &self,
        name: impl Into<String>,
        register: impl FnOnce(&PerformanceMetricsRegistryInner, &str) -> anyhow::Result<()>,
    ) -> anyhow::Result<PerformanceMetricHandle> {
        let name = name.into();
        register(&self.inner, &name)?;
        Ok(PerformanceMetricHandle {
            tracker: Arc::clone(&self.inner),
            metric_name: name,
        })
    }

    /// Factory: create a rate metric handle.
    pub fn new_rate_metric(
        &self,
        name: impl Into<String>,
        quantiles: Vec<f64>,
        sample_period_seconds: Option<f64>,
    ) -> anyhow::Result<RateMetricHandle> {
        let inner = self.register_and_build_handle(name, |registry, metric_name| {
            registry.register_rate_metric(metric_name, quantiles, sample_period_seconds)
        })?;
        Ok(RateMetricHandle { inner })
    }

    /// Factory: create a distribution metric handle.
    pub fn new_distribution_metric(
        &self,
        name: impl Into<String>,
        quantiles: Vec<f64>,
    ) -> anyhow::Result<DistributionMetricHandle> {
        let inner = self.register_and_build_handle(name, |registry, metric_name| {
            registry.register_distribution_metric(metric_name, quantiles)
        })?;
        Ok(DistributionMetricHandle { inner })
    }

    /// Factory: create a ratio metric handle.
    pub fn new_ratio_metric(
        &self,
        name: impl Into<String>,
        quantiles: Vec<f64>,
    ) -> anyhow::Result<RatioMetricHandle> {
        let inner = self.register_and_build_handle(name, |registry, metric_name| {
            registry.register_ratio_metric(metric_name, quantiles)
        })?;
        Ok(RatioMetricHandle { inner })
    }

    /// Attach to a metrics hierarchy, consume builder, and return an attached metrics session.
    pub fn attach_to_hierarchy<H: MetricsHierarchy + ?Sized>(
        self,
        hierarchy: &H,
        metric_prefix: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<AttachedPerformanceMetrics> {
        let publisher = PerformanceMetricsPrometheusPublisher::attach_to_hierarchy(
            Arc::clone(&self.inner),
            hierarchy,
            metric_prefix,
            labels,
        )?;
        Ok(AttachedPerformanceMetrics {
            tracker: self.inner,
            publisher,
        })
    }
}

impl PerformanceMetricsRegistryInner {
    fn window_duration(&self) -> Duration {
        self.window_duration
    }

    fn clear(&self) {
        self.metrics.lock().clear();
    }

    fn register_metric(&self, spec: PerformanceMetricSpec) -> anyhow::Result<()> {
        let mut metrics = self.metrics.lock();
        if spec.name.trim().is_empty() {
            anyhow::bail!("metric name must not be empty");
        }
        if metrics.contains_key(&spec.name) {
            anyhow::bail!("metric '{}' already registered", spec.name);
        }

        let state = match spec.kind {
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

        metrics.insert(spec.name.clone(), MetricEntry { spec, state });
        Ok(())
    }

    fn register_rate_metric(
        &self,
        name: impl Into<String>,
        quantiles: Vec<f64>,
        sample_period_seconds: Option<f64>,
    ) -> anyhow::Result<()> {
        self.register_metric(PerformanceMetricSpec::rate(
            name,
            quantiles,
            sample_period_seconds,
        ))
    }

    fn register_distribution_metric(
        &self,
        name: impl Into<String>,
        quantiles: Vec<f64>,
    ) -> anyhow::Result<()> {
        self.register_metric(PerformanceMetricSpec::distribution(name, quantiles))
    }

    fn register_ratio_metric(
        &self,
        name: impl Into<String>,
        quantiles: Vec<f64>,
    ) -> anyhow::Result<()> {
        self.register_metric(PerformanceMetricSpec::ratio(name, quantiles))
    }

    fn record_count_internal(&self, metric_name: &str, count: u64) -> anyhow::Result<()> {
        if count == 0 {
            return Ok(());
        }
        let mut metrics = self.metrics.lock();
        let entry = metrics
            .get_mut(metric_name)
            .ok_or_else(|| anyhow::anyhow!("unknown metric '{}'", metric_name))?;

        match &mut entry.state {
            MetricState::Rate { counts } => counts.push_back((Instant::now(), count)),
            _ => anyhow::bail!("metric '{}' is not a rate metric", metric_name),
        }
        Ok(())
    }

    fn record_value_internal(&self, metric_name: &str, value: f64) -> anyhow::Result<()> {
        if !value.is_finite() || value < 0.0 {
            anyhow::bail!("value must be a finite non-negative number");
        }
        let mut metrics = self.metrics.lock();
        let entry = metrics
            .get_mut(metric_name)
            .ok_or_else(|| anyhow::anyhow!("unknown metric '{}'", metric_name))?;

        match &mut entry.state {
            MetricState::Distribution { values } => values.push_back((Instant::now(), value)),
            _ => anyhow::bail!("metric '{}' is not a distribution metric", metric_name),
        }
        Ok(())
    }

    fn record_ratio_internal(
        &self,
        metric_name: &str,
        numerator: f64,
        denominator: f64,
    ) -> anyhow::Result<()> {
        if !numerator.is_finite()
            || !denominator.is_finite()
            || numerator < 0.0
            || denominator < 0.0
        {
            anyhow::bail!("numerator/denominator must be finite non-negative numbers");
        }

        let mut metrics = self.metrics.lock();
        let entry = metrics
            .get_mut(metric_name)
            .ok_or_else(|| anyhow::anyhow!("unknown metric '{}'", metric_name))?;

        match &mut entry.state {
            MetricState::Ratio { pairs } => {
                pairs.push_back((Instant::now(), numerator, denominator))
            }
            _ => anyhow::bail!("metric '{}' is not a ratio metric", metric_name),
        }
        Ok(())
    }

    fn metric_specs(&self) -> Vec<PerformanceMetricSpec> {
        self.metrics
            .lock()
            .values()
            .map(|entry| entry.spec.clone())
            .collect::<Vec<_>>()
    }

    /// Snapshot one metric by name.
    fn snapshot_metric(&self, metric_name: &str) -> Option<MetricSnapshot> {
        let now = Instant::now();
        let mut metrics = self.metrics.lock();
        let entry = metrics.get_mut(metric_name)?;
        prune_entry(now, self.window_duration, entry);
        Some(snapshot_entry(now, self.window_duration, entry))
    }

    /// Snapshot all registered metrics.
    fn snapshot_all(&self) -> Vec<MetricSnapshot> {
        let now = Instant::now();
        let mut metrics = self.metrics.lock();
        metrics
            .values_mut()
            .map(|entry| {
                prune_entry(now, self.window_duration, entry);
                snapshot_entry(now, self.window_duration, entry)
            })
            .collect::<Vec<_>>()
    }
}

/// Prometheus gauge publisher for tracker snapshots.
pub struct PerformanceMetricsPrometheusPublisher {
    tracker: Arc<PerformanceMetricsRegistryInner>,
    handles: HashMap<String, MetricHandles>,
    auto_publish_worker: Mutex<Option<AutoPublishWorker>>,
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

struct AutoPublishWorker {
    stop_tx: Sender<()>,
    join_handle: JoinHandle<()>,
}

impl PerformanceMetricsPrometheusPublisher {
    /// Attach publisher to a metrics hierarchy and register gauges.
    ///
    /// Metric set is fixed at attach time; create metrics first.
    fn attach_to_hierarchy<H: MetricsHierarchy + ?Sized>(
        tracker: Arc<PerformanceMetricsRegistryInner>,
        hierarchy: &H,
        metric_prefix: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<Self> {
        let specs = tracker.metric_specs();
        if specs.is_empty() {
            anyhow::bail!("no metrics registered; create metrics before attaching publisher");
        }
        validate_metric_name_collisions(&specs, metric_prefix)?;
        let mut handles = HashMap::new();

        for spec in specs {
            let base = format!("{}_{}", metric_prefix, spec.name);
            let handles_entry = match spec.kind {
                PerformanceMetricKind::Rate => {
                    let per_second = create_metric::<prometheus::Gauge, H>(
                        hierarchy,
                        &format!("{}_{}", base, SUFFIX_PER_SECOND),
                        "Sliding-window throughput per second",
                        labels,
                        None,
                        None,
                    )?;
                    let quantiles = spec
                        .quantiles
                        .iter()
                        .map(|q| {
                            create_metric::<prometheus::Gauge, H>(
                                hierarchy,
                                &format!("{}_{}_{}", base, SUFFIX_PER_SECOND, quantile_suffix(*q)),
                                "Sliding-window throughput quantile",
                                labels,
                                None,
                                None,
                            )
                            .map(|g| (*q, g))
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    MetricHandles::Rate {
                        per_second,
                        quantiles,
                    }
                }
                PerformanceMetricKind::Distribution => {
                    let avg = create_metric::<prometheus::Gauge, H>(
                        hierarchy,
                        &format!("{}_{}", base, SUFFIX_AVG),
                        "Sliding-window average value",
                        labels,
                        None,
                        None,
                    )?;
                    let quantiles = spec
                        .quantiles
                        .iter()
                        .map(|q| {
                            create_metric::<prometheus::Gauge, H>(
                                hierarchy,
                                &format!("{}_{}", base, quantile_suffix(*q)),
                                "Sliding-window quantile value",
                                labels,
                                None,
                                None,
                            )
                            .map(|g| (*q, g))
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    MetricHandles::Distribution { avg, quantiles }
                }
                PerformanceMetricKind::Ratio => {
                    let numerator = create_metric::<prometheus::Gauge, H>(
                        hierarchy,
                        &format!("{}_{}", base, SUFFIX_NUMERATOR),
                        "Sliding-window numerator sum",
                        labels,
                        None,
                        None,
                    )?;
                    let denominator = create_metric::<prometheus::Gauge, H>(
                        hierarchy,
                        &format!("{}_{}", base, SUFFIX_DENOMINATOR),
                        "Sliding-window denominator sum",
                        labels,
                        None,
                        None,
                    )?;
                    let ratio = create_metric::<prometheus::Gauge, H>(
                        hierarchy,
                        &format!("{}_{}", base, SUFFIX_RATIO),
                        "Sliding-window weighted ratio",
                        labels,
                        None,
                        None,
                    )?;
                    let quantiles = spec
                        .quantiles
                        .iter()
                        .map(|q| {
                            create_metric::<prometheus::Gauge, H>(
                                hierarchy,
                                &format!("{}_{}_{}", base, SUFFIX_RATIO, quantile_suffix(*q)),
                                "Sliding-window ratio quantile",
                                labels,
                                None,
                                None,
                            )
                            .map(|g| (*q, g))
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    MetricHandles::Ratio {
                        numerator,
                        denominator,
                        ratio,
                        quantiles,
                    }
                }
            };
            handles.insert(spec.name, handles_entry);
        }

        Ok(Self {
            tracker,
            handles,
            auto_publish_worker: Mutex::new(None),
        })
    }

    /// Update registered gauges from latest tracker snapshots.
    pub fn publish(&self) {
        publish_snapshots(&self.tracker, &self.handles);
    }

    /// Start periodic publish loop. Existing loop (if any) is replaced.
    pub fn start_auto_publish(&self, interval: Duration) {
        self.stop_auto_publish();
        let tick = interval.max(Duration::from_millis(100));
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let tracker = Arc::clone(&self.tracker);
        let handles = self.handles.clone();
        let join_handle = std::thread::spawn(move || {
            loop {
                match stop_rx.recv_timeout(tick) {
                    Ok(_) => break,
                    Err(RecvTimeoutError::Timeout) => publish_snapshots(&tracker, &handles),
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
        });
        *self.auto_publish_worker.lock() = Some(AutoPublishWorker {
            stop_tx,
            join_handle,
        });
    }

    /// Stop periodic publish loop if running.
    pub fn stop_auto_publish(&self) {
        let worker = self.auto_publish_worker.lock().take();
        if let Some(worker) = worker {
            let _ = worker.stop_tx.send(());
            let _ = worker.join_handle.join();
        }
    }
}

impl Drop for PerformanceMetricsPrometheusPublisher {
    fn drop(&mut self) {
        self.stop_auto_publish();
    }
}

fn publish_snapshots(
    tracker: &Arc<PerformanceMetricsRegistryInner>,
    handles: &HashMap<String, MetricHandles>,
) {
    for snapshot in tracker.snapshot_all() {
        if let Some(handle) = handles.get(&snapshot.name) {
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
}

fn normalize_quantiles(quantiles: Vec<f64>) -> Vec<f64> {
    let mut normalized = quantiles
        .into_iter()
        .filter(|q| q.is_finite() && *q >= 0.0 && *q <= 1.0)
        .collect::<Vec<_>>();
    normalized.sort_by(f64::total_cmp);
    normalized.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    let mut seen_suffixes = HashSet::new();
    normalized
        .into_iter()
        .filter(|q| seen_suffixes.insert(quantile_suffix(*q)))
        .collect()
}

fn prune_entry(now: Instant, window_duration: Duration, entry: &mut MetricEntry) {
    let cutoff = now.checked_sub(window_duration).unwrap_or(now);
    match &mut entry.state {
        MetricState::Rate { counts } => {
            while counts.front().is_some_and(|(ts, _)| *ts < cutoff) {
                counts.pop_front();
            }
        }
        MetricState::Distribution { values } => {
            while values.front().is_some_and(|(ts, _)| *ts < cutoff) {
                values.pop_front();
            }
        }
        MetricState::Ratio { pairs } => {
            while pairs.front().is_some_and(|(ts, _, _)| *ts < cutoff) {
                pairs.pop_front();
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
            let mut sorted = samples;
            sorted.sort_by(f64::total_cmp);
            let quantiles = entry
                .spec
                .quantiles
                .iter()
                .map(|q| (*q, percentile(&sorted, *q)))
                .collect::<Vec<_>>();

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
            let mut series = values.iter().map(|(_, v)| *v).collect::<Vec<_>>();
            series.sort_by(f64::total_cmp);
            let average = if series.is_empty() {
                0.0
            } else {
                series.iter().sum::<f64>() / series.len() as f64
            };
            let quantiles = entry
                .spec
                .quantiles
                .iter()
                .map(|q| (*q, percentile(&series, *q)))
                .collect::<Vec<_>>();

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
            let mut sample_ratios = pairs
                .iter()
                .filter_map(|(_, n, d)| if *d > 0.0 { Some(*n / *d) } else { None })
                .collect::<Vec<_>>();
            sample_ratios.sort_by(f64::total_cmp);
            let quantiles = entry
                .spec
                .quantiles
                .iter()
                .map(|q| (*q, percentile(&sample_ratios, *q)))
                .collect::<Vec<_>>();

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
    let sample_period = sample_period_seconds.max(0.1);
    let bins = (window_seconds / sample_period).ceil().max(1.0) as usize;
    let start = now - Duration::from_secs_f64(window_seconds.max(0.1));

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
        .map(|c| c as f64 / sample_period)
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

fn quantile_suffix(q: f64) -> String {
    let mut pct = format!("{:.6}", q.clamp(0.0, 1.0) * 100.0);
    while pct.ends_with('0') {
        pct.pop();
    }
    if pct.ends_with('.') {
        pct.pop();
    }
    format!("p{}", pct.replace('.', "_"))
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
                names.extend(
                    spec.quantiles
                        .iter()
                        .map(|q| format!("{}_{}_{}", base, SUFFIX_PER_SECOND, quantile_suffix(*q))),
                );
            }
            PerformanceMetricKind::Distribution => {
                names.push(format!("{}_{}", base, SUFFIX_AVG));
                names.extend(
                    spec.quantiles
                        .iter()
                        .map(|q| format!("{}_{}", base, quantile_suffix(*q))),
                );
            }
            PerformanceMetricKind::Ratio => {
                names.push(format!("{}_{}", base, SUFFIX_NUMERATOR));
                names.push(format!("{}_{}", base, SUFFIX_DENOMINATOR));
                names.push(format!("{}_{}", base, SUFFIX_RATIO));
                names.extend(
                    spec.quantiles
                        .iter()
                        .map(|q| format!("{}_{}_{}", base, SUFFIX_RATIO, quantile_suffix(*q))),
                );
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
        let tracker = PerformanceMetricsRegistry::new(Duration::from_secs(60));
        let kv = tracker
            .new_ratio_metric("kv", vec![0.5, 0.9, 0.99])
            .unwrap();
        kv.record_ratio(80.0, 100.0).unwrap();
        kv.record_ratio(20.0, 50.0).unwrap();

        let s = kv.snapshot().unwrap();
        assert_eq!(s.kind, PerformanceMetricKind::Ratio);
        assert!((s.numerator_sum.unwrap_or_default() - 100.0).abs() < 1e-9);
        assert!((s.denominator_sum.unwrap_or_default() - 150.0).abs() < 1e-9);
        assert!((s.ratio.unwrap_or_default() - (100.0 / 150.0)).abs() < 1e-9);
        assert_eq!(s.quantiles.len(), 3);
    }

    #[test]
    fn rate_snapshot_has_quantiles() {
        let tracker = PerformanceMetricsRegistry::new(Duration::from_secs(10));
        let request_gps = tracker
            .new_rate_metric("request_gps", vec![0.5, 0.9, 0.99], Some(1.0))
            .unwrap();
        request_gps.record_count(8).unwrap();
        request_gps.record_count(12).unwrap();

        let s = request_gps.snapshot().unwrap();
        assert_eq!(s.kind, PerformanceMetricKind::Rate);
        assert!(s.rate_per_second.unwrap_or_default() >= 0.0);
        assert_eq!(s.quantiles.len(), 3);
    }

    #[test]
    fn quantiles_are_deduped_by_prometheus_suffix() {
        let spec = PerformanceMetricSpec::distribution("ttft", vec![0.994, 0.995, 0.999]);
        let suffixes = spec
            .quantiles
            .iter()
            .map(|q| quantile_suffix(*q))
            .collect::<HashSet<_>>();
        assert_eq!(suffixes.len(), spec.quantiles.len());
    }

    #[test]
    fn attach_without_registered_metrics_fails() {
        let tracker = PerformanceMetricsRegistry::new(Duration::from_secs(60));
        let hierarchy = TestHierarchy::new();
        let result = tracker.attach_to_hierarchy(&hierarchy, "performance", &[]);
        let err = match result {
            Ok(_) => panic!("attach should fail when no metrics are registered"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("no metrics registered"));
    }

    #[test]
    fn attach_consumes_builder_and_handles_keep_working() {
        let tracker = PerformanceMetricsRegistry::new(Duration::from_secs(60));
        let tps = tracker
            .new_rate_metric("tps", vec![0.5], Some(1.0))
            .unwrap();
        let hierarchy = TestHierarchy::new();
        let attached = tracker
            .attach_to_hierarchy(&hierarchy, "performance", &[])
            .unwrap();

        tps.record_count(3).unwrap();
        let snapshot = tps.snapshot().unwrap();
        assert!(snapshot.rate_per_second.unwrap_or_default() >= 0.0);
        attached.publish();
    }
}
