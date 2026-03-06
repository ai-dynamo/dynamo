// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic sliding-window performance metrics with a command-loop backend.
//!
//! All mutations and publishing are processed by a single worker thread via commands/events.
//! Hot-path recording from metric handles is lock-free at call site (bounded `try_send`).

use super::{MetricsHierarchy, create_metric, prometheus_names::build_component_metric_name};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const SUFFIX_PER_SECOND: &str = "per_second";
const SUFFIX_AVG: &str = "avg";
const SUFFIX_NUMERATOR: &str = "numerator";
const SUFFIX_DENOMINATOR: &str = "denominator";
const SUFFIX_RATIO: &str = "ratio";
const DEFAULT_QUEUE_CAPACITY: usize = 16_384;
const WORKER_POLL: Duration = Duration::from_millis(100);

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
    pub window_seconds: Option<f64>,
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
                .map(|v| v.max(0.1)),
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
                .map(|v| v.max(0.1)),
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
                .map(|v| v.max(0.1)),
        }
    }
}

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

#[derive(Debug)]
struct RegistryInner {
    window_duration: Duration,
    publish_interval: Duration,
    tx: SyncSender<WorkerMessage>,
    dropped_events: AtomicU64,
    worker: Mutex<Option<JoinHandle<()>>>,
}

struct MetricLease {
    registry: Arc<RegistryInner>,
    metric_name: String,
}

struct WorkerState {
    window_duration: Duration,
    publish_interval: Duration,
    metrics: HashMap<String, MetricEntry>,
    publisher: Option<PublisherState>,
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

enum WorkerMessage {
    RegisterMetric {
        spec: PerformanceMetricSpec,
        resp: mpsc::Sender<anyhow::Result<()>>,
    },
    UnregisterMetric {
        name: String,
        resp: mpsc::Sender<anyhow::Result<()>>,
    },
    RecordCount {
        name: String,
        count: u64,
        recorded_at: Instant,
    },
    RecordValue {
        name: String,
        value: f64,
        recorded_at: Instant,
    },
    RecordRatio {
        name: String,
        numerator: f64,
        denominator: f64,
        recorded_at: Instant,
    },
    SnapshotMetric {
        name: String,
        resp: mpsc::Sender<Option<MetricSnapshot>>,
    },
    SnapshotAll {
        resp: mpsc::Sender<Vec<MetricSnapshot>>,
    },
    AttachPublisher {
        hierarchy: Arc<dyn MetricsHierarchy>,
        metric_prefix: String,
        labels: Vec<(String, String)>,
        resp: mpsc::Sender<anyhow::Result<()>>,
    },
    Shutdown,
}

impl PerformanceMetricsRegistry {
    pub fn new(window_duration: Duration) -> Self {
        Self::new_with_publish_interval(window_duration, Duration::from_secs(5))
    }

    pub fn new_with_publish_interval(
        window_duration: Duration,
        publish_interval: Duration,
    ) -> Self {
        let window_duration = window_duration.max(Duration::from_secs(1));
        let publish_interval = publish_interval
            .max(Duration::from_secs(2))
            .min(Duration::from_secs(60));
        let (tx, rx) = mpsc::sync_channel::<WorkerMessage>(DEFAULT_QUEUE_CAPACITY);
        let worker = std::thread::spawn(move || worker_loop(rx, window_duration, publish_interval));

        Self {
            inner: Arc::new(RegistryInner {
                window_duration,
                publish_interval,
                tx,
                dropped_events: AtomicU64::new(0),
                worker: Mutex::new(Some(worker)),
            }),
        }
    }

    pub fn new_attached(
        window_duration: Duration,
        publish_interval: Duration,
        hierarchy: Arc<dyn MetricsHierarchy>,
        metric_prefix: impl Into<String>,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<Self> {
        let registry = Self::new_with_publish_interval(window_duration, publish_interval);
        registry.attach_to_hierarchy(hierarchy, metric_prefix.into(), labels)?;
        Ok(registry)
    }

    pub fn new_attached_with_options(
        hierarchy: Arc<dyn MetricsHierarchy>,
        window_seconds: Option<u64>,
        publish_cycle_seconds: Option<u64>,
        metric_prefix: Option<String>,
        labels: Option<Vec<(String, String)>>,
    ) -> anyhow::Result<Self> {
        let seconds = window_seconds.unwrap_or(60).max(1);
        let cycle = publish_cycle_seconds.unwrap_or(5).clamp(2, 60);
        let label_storage = labels.unwrap_or_default();
        let label_refs = label_storage
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect::<Vec<_>>();

        Self::new_attached(
            Duration::from_secs(seconds),
            Duration::from_secs(cycle),
            hierarchy,
            metric_prefix.unwrap_or_else(|| "performance".to_string()),
            &label_refs,
        )
    }

    pub fn new_attached_default(hierarchy: Arc<dyn MetricsHierarchy>) -> anyhow::Result<Self> {
        Self::new_attached_with_options(hierarchy, None, None, None, None)
    }

    pub fn window_duration(&self) -> Duration {
        self.inner.window_duration
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
                    metric_name: name,
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
                    metric_name: name,
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
                    metric_name: name,
                }),
            },
        })
    }

    pub fn unregister_metric(&self, name: impl Into<String>) -> anyhow::Result<()> {
        self.inner.unregister_metric(name.into())
    }

    pub fn snapshot_all(&self) -> Vec<MetricSnapshot> {
        self.inner.snapshot_all()
    }

    fn attach_to_hierarchy(
        &self,
        hierarchy: Arc<dyn MetricsHierarchy>,
        metric_prefix: String,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<()> {
        let labels_owned = labels
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect::<Vec<_>>();
        self.inner
            .attach_publisher(hierarchy, metric_prefix, labels_owned)
    }
}

impl PerformanceMetricHandle {
    pub fn name(&self) -> &str {
        &self.lease.metric_name
    }

    fn record_count(&self, count: u64) -> anyhow::Result<()> {
        self.lease
            .registry
            .record_count(self.lease.metric_name.clone(), count)
    }

    fn record_value(&self, value: f64) -> anyhow::Result<()> {
        self.lease
            .registry
            .record_value(self.lease.metric_name.clone(), value)
    }

    fn record_ratio(&self, numerator: f64, denominator: f64) -> anyhow::Result<()> {
        self.lease
            .registry
            .record_ratio(self.lease.metric_name.clone(), numerator, denominator)
    }

    fn snapshot(&self) -> Option<MetricSnapshot> {
        self.lease
            .registry
            .snapshot_metric(self.lease.metric_name.clone())
    }
}

impl Drop for MetricLease {
    fn drop(&mut self) {
        self.registry
            .try_unregister_metric(self.metric_name.clone());
    }
}

impl RateMetricHandle {
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn record_count(&self, count: u64) -> anyhow::Result<()> {
        self.inner.record_count(count)
    }

    pub fn snapshot(&self) -> Option<MetricSnapshot> {
        self.inner.snapshot()
    }
}

impl DistributionMetricHandle {
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn record_value(&self, value: f64) -> anyhow::Result<()> {
        self.inner.record_value(value)
    }

    pub fn snapshot(&self) -> Option<MetricSnapshot> {
        self.inner.snapshot()
    }
}

impl RatioMetricHandle {
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn record_ratio(&self, numerator: f64, denominator: f64) -> anyhow::Result<()> {
        self.inner.record_ratio(numerator, denominator)
    }

    pub fn snapshot(&self) -> Option<MetricSnapshot> {
        self.inner.snapshot()
    }
}

impl RegistryInner {
    fn register_metric(&self, spec: PerformanceMetricSpec) -> anyhow::Result<()> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx
            .send(WorkerMessage::RegisterMetric {
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
        self.tx
            .send(WorkerMessage::UnregisterMetric {
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
        let _ = self.tx.try_send(WorkerMessage::UnregisterMetric {
            name,
            resp: resp_tx,
        });
    }

    fn record_count(&self, name: String, count: u64) -> anyhow::Result<()> {
        if count == 0 {
            return Ok(());
        }
        match self.tx.try_send(WorkerMessage::RecordCount {
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

    fn record_value(&self, name: String, value: f64) -> anyhow::Result<()> {
        if !value.is_finite() || value < 0.0 {
            anyhow::bail!("value must be a finite non-negative number");
        }
        match self.tx.try_send(WorkerMessage::RecordValue {
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

    fn record_ratio(&self, name: String, numerator: f64, denominator: f64) -> anyhow::Result<()> {
        if !numerator.is_finite()
            || !denominator.is_finite()
            || numerator < 0.0
            || denominator < 0.0
        {
            anyhow::bail!("numerator/denominator must be finite non-negative numbers");
        }
        match self.tx.try_send(WorkerMessage::RecordRatio {
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

    fn snapshot_metric(&self, name: String) -> Option<MetricSnapshot> {
        let (resp_tx, resp_rx) = mpsc::channel();
        if self
            .tx
            .send(WorkerMessage::SnapshotMetric {
                name,
                resp: resp_tx,
            })
            .is_err()
        {
            return None;
        }
        resp_rx.recv().ok().flatten()
    }

    fn snapshot_all(&self) -> Vec<MetricSnapshot> {
        let (resp_tx, resp_rx) = mpsc::channel();
        if self
            .tx
            .send(WorkerMessage::SnapshotAll { resp: resp_tx })
            .is_err()
        {
            return vec![];
        }
        resp_rx.recv().unwrap_or_default()
    }

    fn attach_publisher(
        &self,
        hierarchy: Arc<dyn MetricsHierarchy>,
        metric_prefix: String,
        labels: Vec<(String, String)>,
    ) -> anyhow::Result<()> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx
            .send(WorkerMessage::AttachPublisher {
                hierarchy,
                metric_prefix,
                labels,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("metrics worker is not running"))?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("metrics worker did not respond"))?
    }

    fn note_dropped_event(&self) {
        let dropped = self.dropped_events.fetch_add(1, Ordering::Relaxed) + 1;
        if dropped == 1 || dropped.is_power_of_two() {
            eprintln!(
                "performance metrics queue full; dropped events total={}",
                dropped
            );
        }
    }
}

impl Drop for RegistryInner {
    fn drop(&mut self) {
        let _ = self.tx.send(WorkerMessage::Shutdown);
        if let Some(handle) = self.worker.lock().take() {
            let _ = handle.join();
        }
    }
}

fn worker_loop(rx: Receiver<WorkerMessage>, window_duration: Duration, publish_interval: Duration) {
    let mut state = WorkerState {
        window_duration,
        publish_interval,
        metrics: HashMap::new(),
        publisher: None,
    };

    loop {
        match rx.recv_timeout(WORKER_POLL) {
            Ok(msg) => {
                if !handle_message(msg, &mut state) {
                    return;
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }

        maybe_auto_publish(&mut state);
    }
}

fn handle_message(msg: WorkerMessage, state: &mut WorkerState) -> bool {
    match msg {
        WorkerMessage::RegisterMetric { spec, resp } => {
            let result = register_metric_in_state(state, spec);
            let _ = resp.send(result);
        }
        WorkerMessage::UnregisterMetric { name, resp } => {
            let result = if state.metrics.remove(&name).is_some() {
                if let Some(p) = state.publisher.as_mut() {
                    p.handles.remove(&name);
                }
                Ok(())
            } else {
                Err(anyhow::anyhow!("unknown metric '{}'", name))
            };
            let _ = resp.send(result);
        }
        WorkerMessage::RecordCount {
            name,
            count,
            recorded_at,
        } => {
            if let Some(entry) = state.metrics.get_mut(&name) {
                if let MetricState::Rate { counts } = &mut entry.state {
                    counts.push_back((recorded_at, count));
                }
            }
        }
        WorkerMessage::RecordValue {
            name,
            value,
            recorded_at,
        } => {
            if let Some(entry) = state.metrics.get_mut(&name) {
                if let MetricState::Distribution { values } = &mut entry.state {
                    values.push_back((recorded_at, value));
                }
            }
        }
        WorkerMessage::RecordRatio {
            name,
            numerator,
            denominator,
            recorded_at,
        } => {
            if let Some(entry) = state.metrics.get_mut(&name) {
                if let MetricState::Ratio { pairs } = &mut entry.state {
                    pairs.push_back((recorded_at, numerator, denominator));
                }
            }
        }
        WorkerMessage::SnapshotMetric { name, resp } => {
            let now = Instant::now();
            let snapshot = state.metrics.get_mut(&name).map(|entry| {
                let window = metric_window_duration(entry, state.window_duration);
                prune_entry(now, window, entry);
                snapshot_entry(now, window, entry)
            });
            let _ = resp.send(snapshot);
        }
        WorkerMessage::SnapshotAll { resp } => {
            let now = Instant::now();
            let snapshots = state
                .metrics
                .values_mut()
                .map(|entry| {
                    let window = metric_window_duration(entry, state.window_duration);
                    prune_entry(now, window, entry);
                    snapshot_entry(now, window, entry)
                })
                .collect::<Vec<_>>();
            let _ = resp.send(snapshots);
        }
        WorkerMessage::AttachPublisher {
            hierarchy,
            metric_prefix,
            labels,
            resp,
        } => {
            let result = attach_publisher_in_state(state, hierarchy, metric_prefix, labels);
            let _ = resp.send(result);
        }
        WorkerMessage::Shutdown => return false,
    }

    true
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

    if let Some(p) = state.publisher.as_mut() {
        validate_metric_name_collision_for_spec(spec.clone(), &p.metric_prefix, &p.handles)?;
        let handles = build_metric_handles_for_spec(&spec, p)?;
        p.handles.insert(spec.name.clone(), handles);
    }

    state.metrics.insert(
        spec.name.clone(),
        MetricEntry {
            spec,
            state: state_variant,
        },
    );
    Ok(())
}

fn attach_publisher_in_state(
    state: &mut WorkerState,
    hierarchy: Arc<dyn MetricsHierarchy>,
    metric_prefix: String,
    labels: Vec<(String, String)>,
) -> anyhow::Result<()> {
    validate_metric_name_collisions(
        &state
            .metrics
            .values()
            .map(|entry| entry.spec.clone())
            .collect::<Vec<_>>(),
        &metric_prefix,
    )?;

    let mut handles = HashMap::new();
    let mut pub_state = PublisherState {
        hierarchy,
        metric_prefix,
        labels,
        handles: HashMap::new(),
        next_publish_at: Instant::now() + state.publish_interval,
    };

    for entry in state.metrics.values() {
        let h = build_metric_handles_for_spec(&entry.spec, &pub_state)?;
        handles.insert(entry.spec.name.clone(), h);
    }
    pub_state.handles = handles;
    state.publisher = Some(pub_state);
    Ok(())
}

fn maybe_auto_publish(state: &mut WorkerState) {
    let should_publish = state
        .publisher
        .as_ref()
        .is_some_and(|p| Instant::now() >= p.next_publish_at);
    if !should_publish {
        return;
    }

    let _ = publish_state(state);
    if let Some(p) = state.publisher.as_mut() {
        p.next_publish_at = Instant::now() + state.publish_interval;
    }
}

fn publish_state(state: &mut WorkerState) -> anyhow::Result<()> {
    let Some(publisher) = state.publisher.as_mut() else {
        return Ok(());
    };

    let now = Instant::now();
    for entry in state.metrics.values_mut() {
        let window = metric_window_duration(entry, state.window_duration);
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
                    create_metric::<prometheus::Gauge, dyn MetricsHierarchy>(
                        publisher.hierarchy.as_ref(),
                        &format!("{}_{}_{}", base, SUFFIX_PER_SECOND, quantile_suffix(*q)),
                        "Sliding-window throughput quantile",
                        &label_refs,
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
                    create_metric::<prometheus::Gauge, dyn MetricsHierarchy>(
                        publisher.hierarchy.as_ref(),
                        &format!("{}_{}", base, quantile_suffix(*q)),
                        "Sliding-window quantile value",
                        &label_refs,
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
                    create_metric::<prometheus::Gauge, dyn MetricsHierarchy>(
                        publisher.hierarchy.as_ref(),
                        &format!("{}_{}_{}", base, SUFFIX_RATIO, quantile_suffix(*q)),
                        "Sliding-window ratio quantile",
                        &label_refs,
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

fn validate_metric_name_collision_for_spec(
    spec: PerformanceMetricSpec,
    metric_prefix: &str,
    existing: &HashMap<String, MetricHandles>,
) -> anyhow::Result<()> {
    let mut names = HashSet::new();
    for key in existing.keys() {
        names.insert(key.clone());
    }
    if names.contains(&spec.name) {
        anyhow::bail!("metric '{}' already registered", spec.name);
    }
    validate_metric_name_collisions(&[spec], metric_prefix)
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

fn metric_window_duration(entry: &MetricEntry, default_window: Duration) -> Duration {
    entry
        .spec
        .window_seconds
        .map(Duration::from_secs_f64)
        .unwrap_or(default_window)
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
            .new_ratio_metric("kv", vec![0.5, 0.9, 0.99], None)
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
            .new_rate_metric("request_gps", vec![0.5, 0.9, 0.99], Some(1.0), None)
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
        let spec = PerformanceMetricSpec::distribution("ttft", vec![0.994, 0.995, 0.999], None);
        let suffixes = spec
            .quantiles
            .iter()
            .map(|q| quantile_suffix(*q))
            .collect::<HashSet<_>>();
        assert_eq!(suffixes.len(), spec.quantiles.len());
    }

    #[test]
    fn attach_without_registered_metrics_succeeds() {
        let hierarchy = Arc::new(TestHierarchy::new());
        let tracker = PerformanceMetricsRegistry::new_attached(
            Duration::from_secs(60),
            Duration::from_secs(5),
            hierarchy,
            "performance",
            &[] as &[(&str, &str)],
        )
        .unwrap();
        assert_eq!(tracker.snapshot_all().len(), 0);
    }

    #[test]
    fn register_after_attach_and_unregister_work() {
        let hierarchy = Arc::new(TestHierarchy::new());
        let tracker = PerformanceMetricsRegistry::new_attached(
            Duration::from_secs(60),
            Duration::from_secs(5),
            hierarchy,
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
        assert!(ttft.snapshot().is_some());

        tracker.unregister_metric("ttft").unwrap();
        assert!(ttft.snapshot().is_none());
    }
}
