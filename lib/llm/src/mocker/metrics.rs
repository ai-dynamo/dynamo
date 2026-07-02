// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use dynamo_mocker::common::protocols::EngineType;
use dynamo_mocker::scheduler::MockerMetrics;
use dynamo_runtime::MetricsRegistry;
use prometheus::{
    Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge,
    IntGaugeVec, Opts,
};
use tokio::sync::OnceCell;

const VLLM_LABELS: &[&str] = &["model_name", "engine"];
const SGLANG_SCHEDULER_LABELS: &[&str] = &[
    "model_name",
    "engine_type",
    "tp_rank",
    "pp_rank",
    "moe_ep_rank",
];
const SGLANG_REQUEST_LABELS: &[&str] = &["model_name", "engine_type"];

pub(crate) struct NativeMockerMetrics {
    dp_size: u32,
    collectors: NativeCollectors,
    model_name: OnceCell<String>,
    state: Mutex<NativeMockerMetricsState>,
}

#[derive(Default)]
struct NativeMockerMetricsState {
    registered: bool,
    warned_model_mismatch: bool,
    latest_snapshots: HashMap<u32, MockerMetrics>,
    vllm_preemptions_seen: HashMap<u32, u64>,
    handles: Option<NativeMetricHandles>,
}

enum NativeCollectors {
    Vllm(VllmCollectors),
    Trtllm(TrtllmCollectors),
    Sglang(SglangCollectors),
}

struct VllmCollectors {
    num_requests_running: IntGaugeVec,
    num_requests_waiting: IntGaugeVec,
    kv_cache_usage_perc: GaugeVec,
    gpu_cache_usage_perc: GaugeVec,
    num_preemptions_total: IntCounterVec,
    time_to_first_token_seconds: HistogramVec,
    inter_token_latency_seconds: HistogramVec,
    e2e_request_latency_seconds: HistogramVec,
}

/// TRT-LLM runs on the vLLM scheduler core, so it exposes the same metric
/// shape as [`VllmCollectors`] but under the `trtllm:` native name prefix. It
/// binds into the same [`NativeMetricHandles::Vllm`] update path.
struct TrtllmCollectors {
    num_requests_running: IntGaugeVec,
    num_requests_waiting: IntGaugeVec,
    kv_cache_usage_perc: GaugeVec,
    gpu_cache_usage_perc: GaugeVec,
    num_preemptions_total: IntCounterVec,
    time_to_first_token_seconds: HistogramVec,
    inter_token_latency_seconds: HistogramVec,
    e2e_request_latency_seconds: HistogramVec,
}

struct SglangCollectors {
    num_running_reqs: IntGaugeVec,
    num_queue_reqs: IntGaugeVec,
    cache_hit_rate: GaugeVec,
    num_requests_total: IntCounterVec,
    time_to_first_token_seconds: HistogramVec,
    inter_token_latency_seconds: HistogramVec,
    e2e_request_latency_seconds: HistogramVec,
}

enum NativeMetricHandles {
    Vllm(HashMap<u32, VllmDpHandles>),
    Sglang(SglangHandles),
}

struct VllmDpHandles {
    num_requests_running: IntGauge,
    num_requests_waiting: IntGauge,
    kv_cache_usage_perc: Gauge,
    gpu_cache_usage_perc: Gauge,
    num_preemptions_total: IntCounter,
    request_metrics: NativeRequestMetricHandles,
}

struct SglangHandles {
    num_running_reqs: IntGauge,
    num_queue_reqs: IntGauge,
    cache_hit_rate: Gauge,
    request_metrics: NativeRequestMetricHandles,
}

#[derive(Clone)]
enum NativeRequestMetricHandles {
    Vllm {
        time_to_first_token_seconds: Histogram,
        inter_token_latency_seconds: Histogram,
        e2e_request_latency_seconds: Histogram,
    },
    Sglang {
        num_requests_total: IntCounter,
        time_to_first_token_seconds: Histogram,
        inter_token_latency_seconds: Histogram,
        e2e_request_latency_seconds: Histogram,
    },
}

pub(crate) struct NativeRequestTiming {
    handles: Option<NativeRequestMetricHandles>,
    start: Instant,
    last_token_at: Option<Instant>,
}

impl NativeMockerMetrics {
    pub(crate) fn new(engine_type: EngineType, dp_size: u32) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            dp_size,
            collectors: NativeCollectors::new(engine_type)?,
            model_name: OnceCell::new(),
            state: Mutex::new(NativeMockerMetricsState::default()),
        }))
    }

    pub(crate) fn register(&self, registry: &MetricsRegistry) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .expect("native mocker metrics lock poisoned");
        if state.registered {
            return Ok(());
        }

        self.collectors.register(registry)?;
        state.registered = true;
        Ok(())
    }

    pub(crate) fn update_scheduler_snapshot(&self, metrics: &MockerMetrics) {
        let mut state = self
            .state
            .lock()
            .expect("native mocker metrics lock poisoned");
        state
            .latest_snapshots
            .insert(metrics.dp_rank, metrics.clone());

        if let Some(model_name) = self.model_name.get() {
            self.bind_handles_locked(&mut state, model_name);
        }
        self.apply_snapshot_locked(&mut state, metrics);
    }

    pub(crate) async fn request_timing(
        &self,
        model_name: &str,
        dp_rank: u32,
        is_prefill: bool,
        start: Instant,
    ) -> NativeRequestTiming {
        if is_prefill {
            return NativeRequestTiming::disabled(start);
        }

        let Some(model_name) = self.ensure_model_name(model_name).await else {
            return NativeRequestTiming::disabled(start);
        };

        let handles = {
            let mut state = self
                .state
                .lock()
                .expect("native mocker metrics lock poisoned");
            self.bind_handles_locked(&mut state, &model_name);
            match state.handles.as_ref() {
                Some(NativeMetricHandles::Vllm(handles)) => handles
                    .get(&dp_rank)
                    .map(|handles| handles.request_metrics.clone()),
                Some(NativeMetricHandles::Sglang(handles)) => Some(handles.request_metrics.clone()),
                None => None,
            }
        };

        NativeRequestTiming::new(start, handles)
    }

    async fn ensure_model_name(&self, request_model_name: &str) -> Option<String> {
        let request_model_name = request_model_name.trim();
        if request_model_name.is_empty() {
            return None;
        }

        let requested = request_model_name.to_string();
        let model_name = self
            .model_name
            .get_or_try_init(|| async { Ok::<_, anyhow::Error>(requested.clone()) })
            .await
            .ok()?;

        if model_name != request_model_name {
            let mut state = self
                .state
                .lock()
                .expect("native mocker metrics lock poisoned");
            if !state.warned_model_mismatch {
                tracing::warn!(
                    first_model_name = %model_name,
                    request_model_name,
                    "mocker native metrics keep the first model_name label for this process"
                );
                state.warned_model_mismatch = true;
            }
        }

        Some(model_name.clone())
    }

    fn bind_handles_locked(&self, state: &mut NativeMockerMetricsState, model_name: &str) {
        if state.handles.is_some() {
            return;
        }

        state.handles = match &self.collectors {
            NativeCollectors::Vllm(collectors) => {
                let mut handles = HashMap::with_capacity(self.dp_size as usize);
                for dp_rank in 0..self.dp_size {
                    let engine = dp_rank.to_string();
                    let label_values = [model_name, engine.as_str()];
                    handles.insert(
                        dp_rank,
                        VllmDpHandles {
                            num_requests_running: collectors
                                .num_requests_running
                                .with_label_values(&label_values),
                            num_requests_waiting: collectors
                                .num_requests_waiting
                                .with_label_values(&label_values),
                            kv_cache_usage_perc: collectors
                                .kv_cache_usage_perc
                                .with_label_values(&label_values),
                            gpu_cache_usage_perc: collectors
                                .gpu_cache_usage_perc
                                .with_label_values(&label_values),
                            num_preemptions_total: collectors
                                .num_preemptions_total
                                .with_label_values(&label_values),
                            request_metrics: NativeRequestMetricHandles::Vllm {
                                time_to_first_token_seconds: collectors
                                    .time_to_first_token_seconds
                                    .with_label_values(&label_values),
                                inter_token_latency_seconds: collectors
                                    .inter_token_latency_seconds
                                    .with_label_values(&label_values),
                                e2e_request_latency_seconds: collectors
                                    .e2e_request_latency_seconds
                                    .with_label_values(&label_values),
                            },
                        },
                    );
                }
                Some(NativeMetricHandles::Vllm(handles))
            }
            NativeCollectors::Trtllm(collectors) => {
                // Identical shape to the vLLM path; bind into the same handles
                // so the snapshot-application logic is shared.
                let mut handles = HashMap::with_capacity(self.dp_size as usize);
                for dp_rank in 0..self.dp_size {
                    let engine = dp_rank.to_string();
                    let label_values = [model_name, engine.as_str()];
                    handles.insert(
                        dp_rank,
                        VllmDpHandles {
                            num_requests_running: collectors
                                .num_requests_running
                                .with_label_values(&label_values),
                            num_requests_waiting: collectors
                                .num_requests_waiting
                                .with_label_values(&label_values),
                            kv_cache_usage_perc: collectors
                                .kv_cache_usage_perc
                                .with_label_values(&label_values),
                            gpu_cache_usage_perc: collectors
                                .gpu_cache_usage_perc
                                .with_label_values(&label_values),
                            num_preemptions_total: collectors
                                .num_preemptions_total
                                .with_label_values(&label_values),
                            request_metrics: NativeRequestMetricHandles::Vllm {
                                time_to_first_token_seconds: collectors
                                    .time_to_first_token_seconds
                                    .with_label_values(&label_values),
                                inter_token_latency_seconds: collectors
                                    .inter_token_latency_seconds
                                    .with_label_values(&label_values),
                                e2e_request_latency_seconds: collectors
                                    .e2e_request_latency_seconds
                                    .with_label_values(&label_values),
                            },
                        },
                    );
                }
                Some(NativeMetricHandles::Vllm(handles))
            }
            NativeCollectors::Sglang(collectors) => {
                let scheduler_label_values = [model_name, "sglang", "0", "0", "0"];
                let request_label_values = [model_name, "sglang"];
                Some(NativeMetricHandles::Sglang(SglangHandles {
                    num_running_reqs: collectors
                        .num_running_reqs
                        .with_label_values(&scheduler_label_values),
                    num_queue_reqs: collectors
                        .num_queue_reqs
                        .with_label_values(&scheduler_label_values),
                    cache_hit_rate: collectors
                        .cache_hit_rate
                        .with_label_values(&scheduler_label_values),
                    request_metrics: NativeRequestMetricHandles::Sglang {
                        num_requests_total: collectors
                            .num_requests_total
                            .with_label_values(&request_label_values),
                        time_to_first_token_seconds: collectors
                            .time_to_first_token_seconds
                            .with_label_values(&request_label_values),
                        inter_token_latency_seconds: collectors
                            .inter_token_latency_seconds
                            .with_label_values(&request_label_values),
                        e2e_request_latency_seconds: collectors
                            .e2e_request_latency_seconds
                            .with_label_values(&request_label_values),
                    },
                }))
            }
        };

        let snapshots = state.latest_snapshots.values().cloned().collect::<Vec<_>>();
        for snapshot in snapshots {
            self.apply_snapshot_locked(state, &snapshot);
        }
    }

    fn apply_snapshot_locked(&self, state: &mut NativeMockerMetricsState, metrics: &MockerMetrics) {
        match state.handles.as_ref() {
            Some(NativeMetricHandles::Vllm(handles)) => {
                let Some(handles) = handles.get(&metrics.dp_rank) else {
                    return;
                };
                handles
                    .num_requests_running
                    .set(clamp_u64_to_i64(metrics.running_requests));
                handles
                    .num_requests_waiting
                    .set(clamp_u64_to_i64(metrics.waiting_requests));
                handles
                    .kv_cache_usage_perc
                    .set(metrics.gpu_cache_usage_perc);
                handles
                    .gpu_cache_usage_perc
                    .set(metrics.gpu_cache_usage_perc);

                let last_seen = state
                    .vllm_preemptions_seen
                    .entry(metrics.dp_rank)
                    .or_insert(0);
                if metrics.vllm_preemptions_total > *last_seen {
                    handles
                        .num_preemptions_total
                        .inc_by(metrics.vllm_preemptions_total - *last_seen);
                    *last_seen = metrics.vllm_preemptions_total;
                }
            }
            Some(NativeMetricHandles::Sglang(handles)) => {
                let running = state
                    .latest_snapshots
                    .values()
                    .map(|snapshot| snapshot.running_requests)
                    .sum::<u64>();
                let queued = state
                    .latest_snapshots
                    .values()
                    .map(|snapshot| snapshot.waiting_requests)
                    .sum::<u64>();
                let cache_hits = state
                    .latest_snapshots
                    .values()
                    .map(|snapshot| snapshot.sglang_cache_hit_tokens)
                    .sum::<u64>();
                let cache_total = state
                    .latest_snapshots
                    .values()
                    .map(|snapshot| snapshot.sglang_cache_total_tokens)
                    .sum::<u64>();

                handles.num_running_reqs.set(clamp_u64_to_i64(running));
                handles.num_queue_reqs.set(clamp_u64_to_i64(queued));
                handles.cache_hit_rate.set(if cache_total == 0 {
                    0.0
                } else {
                    cache_hits as f64 / cache_total as f64
                });
            }
            None => {}
        }
    }
}

impl NativeCollectors {
    fn new(engine_type: EngineType) -> Result<Self> {
        match engine_type {
            EngineType::Vllm => Ok(Self::Vllm(VllmCollectors::new()?)),
            EngineType::Trtllm => Ok(Self::Trtllm(TrtllmCollectors::new()?)),
            EngineType::Sglang => Ok(Self::Sglang(SglangCollectors::new()?)),
        }
    }

    fn register(&self, registry: &MetricsRegistry) -> Result<()> {
        match self {
            Self::Vllm(collectors) => collectors.register(registry),
            Self::Trtllm(collectors) => collectors.register(registry),
            Self::Sglang(collectors) => collectors.register(registry),
        }
    }
}

impl VllmCollectors {
    fn new() -> Result<Self> {
        Ok(Self {
            num_requests_running: IntGaugeVec::new(
                Opts::new(
                    "vllm:num_requests_running",
                    "Number of requests currently running on GPU.",
                ),
                VLLM_LABELS,
            )?,
            num_requests_waiting: IntGaugeVec::new(
                Opts::new(
                    "vllm:num_requests_waiting",
                    "Number of requests waiting to be processed.",
                ),
                VLLM_LABELS,
            )?,
            kv_cache_usage_perc: GaugeVec::new(
                Opts::new(
                    "vllm:kv_cache_usage_perc",
                    "KV-cache usage as a fraction of available cache blocks.",
                ),
                VLLM_LABELS,
            )?,
            gpu_cache_usage_perc: GaugeVec::new(
                Opts::new(
                    "vllm:gpu_cache_usage_perc",
                    "Compatibility alias for KV-cache usage as a fraction of available cache blocks.",
                ),
                VLLM_LABELS,
            )?,
            num_preemptions_total: IntCounterVec::new(
                Opts::new(
                    "vllm:num_preemptions_total",
                    "Cumulative number of request preemptions.",
                ),
                VLLM_LABELS,
            )?,
            time_to_first_token_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "vllm:time_to_first_token_seconds",
                    "Histogram of time to first token in seconds.",
                )
                .buckets(vllm_ttft_buckets()),
                VLLM_LABELS,
            )?,
            inter_token_latency_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "vllm:inter_token_latency_seconds",
                    "Histogram of inter-token latency in seconds.",
                )
                .buckets(vllm_itl_buckets()),
                VLLM_LABELS,
            )?,
            e2e_request_latency_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "vllm:e2e_request_latency_seconds",
                    "Histogram of e2e request latency in seconds.",
                )
                .buckets(vllm_e2e_buckets()),
                VLLM_LABELS,
            )?,
        })
    }

    fn register(&self, registry: &MetricsRegistry) -> Result<()> {
        registry.add_metric(Box::new(self.num_requests_running.clone()))?;
        registry.add_metric(Box::new(self.num_requests_waiting.clone()))?;
        registry.add_metric(Box::new(self.kv_cache_usage_perc.clone()))?;
        registry.add_metric(Box::new(self.gpu_cache_usage_perc.clone()))?;
        registry.add_metric(Box::new(self.num_preemptions_total.clone()))?;
        registry.add_metric(Box::new(self.time_to_first_token_seconds.clone()))?;
        registry.add_metric(Box::new(self.inter_token_latency_seconds.clone()))?;
        registry.add_metric(Box::new(self.e2e_request_latency_seconds.clone()))?;
        Ok(())
    }
}

impl TrtllmCollectors {
    fn new() -> Result<Self> {
        Ok(Self {
            num_requests_running: IntGaugeVec::new(
                Opts::new(
                    "trtllm:num_requests_running",
                    "Number of requests currently running on GPU.",
                ),
                VLLM_LABELS,
            )?,
            num_requests_waiting: IntGaugeVec::new(
                Opts::new(
                    "trtllm:num_requests_waiting",
                    "Number of requests waiting to be processed.",
                ),
                VLLM_LABELS,
            )?,
            kv_cache_usage_perc: GaugeVec::new(
                Opts::new(
                    "trtllm:kv_cache_usage_perc",
                    "KV-cache usage as a fraction of available cache blocks.",
                ),
                VLLM_LABELS,
            )?,
            gpu_cache_usage_perc: GaugeVec::new(
                Opts::new(
                    "trtllm:gpu_cache_usage_perc",
                    "Compatibility alias for KV-cache usage as a fraction of available cache blocks.",
                ),
                VLLM_LABELS,
            )?,
            num_preemptions_total: IntCounterVec::new(
                Opts::new(
                    "trtllm:num_preemptions_total",
                    "Cumulative number of request preemptions (always 0 under GUARANTEED_NO_EVICT).",
                ),
                VLLM_LABELS,
            )?,
            time_to_first_token_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "trtllm:time_to_first_token_seconds",
                    "Histogram of time to first token in seconds.",
                )
                .buckets(vllm_ttft_buckets()),
                VLLM_LABELS,
            )?,
            inter_token_latency_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "trtllm:inter_token_latency_seconds",
                    "Histogram of inter-token latency in seconds.",
                )
                .buckets(vllm_itl_buckets()),
                VLLM_LABELS,
            )?,
            e2e_request_latency_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "trtllm:e2e_request_latency_seconds",
                    "Histogram of e2e request latency in seconds.",
                )
                .buckets(vllm_e2e_buckets()),
                VLLM_LABELS,
            )?,
        })
    }

    fn register(&self, registry: &MetricsRegistry) -> Result<()> {
        registry.add_metric(Box::new(self.num_requests_running.clone()))?;
        registry.add_metric(Box::new(self.num_requests_waiting.clone()))?;
        registry.add_metric(Box::new(self.kv_cache_usage_perc.clone()))?;
        registry.add_metric(Box::new(self.gpu_cache_usage_perc.clone()))?;
        registry.add_metric(Box::new(self.num_preemptions_total.clone()))?;
        registry.add_metric(Box::new(self.time_to_first_token_seconds.clone()))?;
        registry.add_metric(Box::new(self.inter_token_latency_seconds.clone()))?;
        registry.add_metric(Box::new(self.e2e_request_latency_seconds.clone()))?;
        Ok(())
    }
}

impl SglangCollectors {
    fn new() -> Result<Self> {
        Ok(Self {
            num_running_reqs: IntGaugeVec::new(
                Opts::new("sglang:num_running_reqs", "The number of running requests."),
                SGLANG_SCHEDULER_LABELS,
            )?,
            num_queue_reqs: IntGaugeVec::new(
                Opts::new("sglang:num_queue_reqs", "The number of queued requests."),
                SGLANG_SCHEDULER_LABELS,
            )?,
            cache_hit_rate: GaugeVec::new(
                Opts::new("sglang:cache_hit_rate", "Cache hit rate."),
                SGLANG_SCHEDULER_LABELS,
            )?,
            num_requests_total: IntCounterVec::new(
                Opts::new("sglang:num_requests_total", "Number of requests processed."),
                SGLANG_REQUEST_LABELS,
            )?,
            time_to_first_token_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "sglang:time_to_first_token_seconds",
                    "Histogram of time to first token in seconds.",
                )
                .buckets(sglang_ttft_buckets()),
                SGLANG_REQUEST_LABELS,
            )?,
            inter_token_latency_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "sglang:inter_token_latency_seconds",
                    "Histogram of inter-token latency in seconds.",
                )
                .buckets(sglang_itl_buckets()),
                SGLANG_REQUEST_LABELS,
            )?,
            e2e_request_latency_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "sglang:e2e_request_latency_seconds",
                    "Histogram of End-to-end request latency in seconds",
                )
                .buckets(sglang_e2e_buckets()),
                SGLANG_REQUEST_LABELS,
            )?,
        })
    }

    fn register(&self, registry: &MetricsRegistry) -> Result<()> {
        registry.add_metric(Box::new(self.num_running_reqs.clone()))?;
        registry.add_metric(Box::new(self.num_queue_reqs.clone()))?;
        registry.add_metric(Box::new(self.cache_hit_rate.clone()))?;
        registry.add_metric(Box::new(self.num_requests_total.clone()))?;
        registry.add_metric(Box::new(self.time_to_first_token_seconds.clone()))?;
        registry.add_metric(Box::new(self.inter_token_latency_seconds.clone()))?;
        registry.add_metric(Box::new(self.e2e_request_latency_seconds.clone()))?;
        Ok(())
    }
}

impl NativeRequestTiming {
    fn new(start: Instant, handles: Option<NativeRequestMetricHandles>) -> Self {
        Self {
            handles,
            start,
            last_token_at: None,
        }
    }

    fn disabled(start: Instant) -> Self {
        Self::new(start, None)
    }

    pub(crate) fn record_tokens(&mut self, token_count: usize) {
        if token_count == 0 {
            return;
        }
        let Some(handles) = &self.handles else {
            return;
        };

        let now = Instant::now();
        let Some(last_token_at) = self.last_token_at.replace(now) else {
            handles.observe_ttft(now.duration_since(self.start).as_secs_f64());
            return;
        };

        let per_token_interval =
            now.duration_since(last_token_at).as_secs_f64() / token_count as f64;
        for _ in 0..token_count {
            handles.observe_itl(per_token_interval);
        }
    }

    pub(crate) fn record_normal_completion(&self) {
        let Some(handles) = &self.handles else {
            return;
        };

        handles.observe_e2e(self.start.elapsed().as_secs_f64());
        handles.inc_completed_requests();
    }
}

impl NativeRequestMetricHandles {
    fn observe_ttft(&self, value: f64) {
        match self {
            Self::Vllm {
                time_to_first_token_seconds,
                ..
            }
            | Self::Sglang {
                time_to_first_token_seconds,
                ..
            } => time_to_first_token_seconds.observe(value),
        }
    }

    fn observe_itl(&self, value: f64) {
        match self {
            Self::Vllm {
                inter_token_latency_seconds,
                ..
            }
            | Self::Sglang {
                inter_token_latency_seconds,
                ..
            } => inter_token_latency_seconds.observe(value),
        }
    }

    fn observe_e2e(&self, value: f64) {
        match self {
            Self::Vllm {
                e2e_request_latency_seconds,
                ..
            }
            | Self::Sglang {
                e2e_request_latency_seconds,
                ..
            } => e2e_request_latency_seconds.observe(value),
        }
    }

    fn inc_completed_requests(&self) {
        if let Self::Sglang {
            num_requests_total, ..
        } = self
        {
            num_requests_total.inc();
        }
    }
}

fn clamp_u64_to_i64(value: u64) -> i64 {
    value.min(i64::MAX as u64) as i64
}

fn vllm_ttft_buckets() -> Vec<f64> {
    vec![
        0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
        20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
    ]
}

fn vllm_itl_buckets() -> Vec<f64> {
    vec![
        0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
        20.0, 40.0, 80.0,
    ]
}

fn vllm_e2e_buckets() -> Vec<f64> {
    vec![
        0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0,
        240.0, 480.0, 960.0, 1920.0, 7680.0,
    ]
}

fn sglang_ttft_buckets() -> Vec<f64> {
    vec![
        0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0,
        200.0, 400.0,
    ]
}

fn sglang_itl_buckets() -> Vec<f64> {
    vec![
        0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.060, 0.080,
        0.100, 0.200, 0.400, 0.600, 0.800, 1.000, 2.000, 4.000, 6.000, 8.000,
    ]
}

fn sglang_e2e_buckets() -> Vec<f64> {
    vec![
        0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0,
        200.0, 400.0, 600.0, 1200.0, 1800.0, 2400.0,
    ]
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::time::{Duration, Instant};

    use dynamo_mocker::common::protocols::EngineType;
    use dynamo_mocker::scheduler::MockerMetrics;
    use dynamo_runtime::MetricsRegistry;
    use prometheus::proto::{Metric, MetricFamily, MetricType};

    use super::NativeMockerMetrics;

    fn gather_family(registry: &MetricsRegistry, name: &str) -> MetricFamily {
        registry
            .get_prometheus_registry()
            .gather()
            .into_iter()
            .find(|family| family.name() == name)
            .unwrap_or_else(|| panic!("missing metric family {name}"))
    }

    fn metric_labels(metric: &Metric) -> HashSet<String> {
        metric
            .get_label()
            .iter()
            .map(|label| label.name().to_string())
            .collect()
    }

    fn metric_with_label<'a>(
        family: &'a MetricFamily,
        label_name: &str,
        label_value: &str,
    ) -> &'a Metric {
        family
            .get_metric()
            .iter()
            .find(|metric| {
                metric
                    .get_label()
                    .iter()
                    .any(|label| label.name() == label_name && label.value() == label_value)
            })
            .unwrap_or_else(|| {
                panic!(
                    "missing metric in {} with {label_name}={label_value}",
                    family.name()
                )
            })
    }

    fn histogram_count(registry: &MetricsRegistry, name: &str) -> u64 {
        gather_family(registry, name)
            .get_metric()
            .iter()
            .map(|metric| metric.histogram.as_ref().unwrap().sample_count())
            .sum()
    }

    fn counter_value(registry: &MetricsRegistry, name: &str) -> f64 {
        gather_family(registry, name)
            .get_metric()
            .iter()
            .map(|metric| metric.counter.as_ref().unwrap().value())
            .sum()
    }

    #[tokio::test]
    async fn vllm_scrape_contract_uses_native_names_and_engine_labels() {
        let registry = MetricsRegistry::new();
        let metrics = NativeMockerMetrics::new(EngineType::Vllm, 2).unwrap();
        metrics.register(&registry).unwrap();
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 3, 10, 2, 4, 5, 0, 0));
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(1, 4, 10, 7, 8, 1, 0, 0));

        let mut timing = metrics
            .request_timing("llama", 0, false, Instant::now())
            .await;
        timing.record_tokens(1);
        timing.record_normal_completion();

        let running = gather_family(&registry, "vllm:num_requests_running");
        assert_eq!(running.get_field_type(), MetricType::GAUGE);
        let engine0 = metric_with_label(&running, "engine", "0");
        let engine1 = metric_with_label(&running, "engine", "1");
        assert_eq!(
            metric_labels(engine0),
            HashSet::from(["model_name".to_string(), "engine".to_string()])
        );
        assert_eq!(engine0.gauge.as_ref().unwrap().value(), 2.0);
        assert_eq!(engine1.gauge.as_ref().unwrap().value(), 7.0);

        assert_eq!(
            gather_family(&registry, "vllm:kv_cache_usage_perc").get_field_type(),
            MetricType::GAUGE
        );
        assert_eq!(
            gather_family(&registry, "vllm:gpu_cache_usage_perc").get_field_type(),
            MetricType::GAUGE
        );
        assert_eq!(
            gather_family(&registry, "vllm:num_preemptions_total").get_field_type(),
            MetricType::COUNTER
        );
        assert_eq!(
            gather_family(&registry, "vllm:time_to_first_token_seconds").get_field_type(),
            MetricType::HISTOGRAM
        );

        let text = registry.prometheus_expfmt_combined().unwrap();
        assert!(!text.contains("dynamo_component_vllm"));
        assert!(!text.contains("vllm_kv_cache_usage_perc"));
    }

    #[tokio::test]
    async fn vllm_preemption_counter_is_monotonic_across_snapshots() {
        let registry = MetricsRegistry::new();
        let metrics = NativeMockerMetrics::new(EngineType::Vllm, 1).unwrap();
        metrics.register(&registry).unwrap();
        let _ = metrics
            .request_timing("llama", 0, false, Instant::now())
            .await;

        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 0, 10, 0, 0, 3, 0, 0));
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 0, 10, 0, 0, 3, 0, 0));
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 0, 10, 0, 0, 2, 0, 0));
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 0, 10, 0, 0, 5, 0, 0));

        assert_eq!(counter_value(&registry, "vllm:num_preemptions_total"), 5.0);
    }

    #[tokio::test]
    async fn sglang_scrape_contract_uses_scheduler_and_request_label_sets() {
        let registry = MetricsRegistry::new();
        let metrics = NativeMockerMetrics::new(EngineType::Sglang, 2).unwrap();
        metrics.register(&registry).unwrap();
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 2, 10, 1, 3, 0, 2, 4));
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(1, 2, 10, 4, 5, 0, 1, 2));

        let mut timing = metrics
            .request_timing("llama", 0, false, Instant::now())
            .await;
        timing.record_tokens(1);
        timing.record_normal_completion();

        let running = gather_family(&registry, "sglang:num_running_reqs");
        assert_eq!(running.get_field_type(), MetricType::GAUGE);
        let running_metric = running.get_metric().first().unwrap();
        assert_eq!(running_metric.gauge.as_ref().unwrap().value(), 5.0);
        assert_eq!(
            metric_labels(running_metric),
            HashSet::from([
                "model_name".to_string(),
                "engine_type".to_string(),
                "tp_rank".to_string(),
                "pp_rank".to_string(),
                "moe_ep_rank".to_string(),
            ])
        );

        let queue = gather_family(&registry, "sglang:num_queue_reqs");
        assert_eq!(
            queue
                .get_metric()
                .first()
                .unwrap()
                .gauge
                .as_ref()
                .unwrap()
                .value(),
            8.0
        );
        let cache_hit_rate = gather_family(&registry, "sglang:cache_hit_rate");
        assert_eq!(
            cache_hit_rate
                .get_metric()
                .first()
                .unwrap()
                .gauge
                .as_ref()
                .unwrap()
                .value(),
            0.5
        );

        let requests = gather_family(&registry, "sglang:num_requests_total");
        assert_eq!(requests.get_field_type(), MetricType::COUNTER);
        let request_metric = requests.get_metric().first().unwrap();
        assert_eq!(
            metric_labels(request_metric),
            HashSet::from(["model_name".to_string(), "engine_type".to_string()])
        );
    }

    #[tokio::test]
    async fn request_timing_records_successful_token_lifecycle_only() {
        let registry = MetricsRegistry::new();
        let metrics = NativeMockerMetrics::new(EngineType::Sglang, 1).unwrap();
        metrics.register(&registry).unwrap();

        let mut timing = metrics
            .request_timing(
                "llama",
                0,
                false,
                Instant::now() - Duration::from_millis(10),
            )
            .await;
        timing.record_tokens(1);
        assert_eq!(
            histogram_count(&registry, "sglang:time_to_first_token_seconds"),
            1
        );
        assert_eq!(
            histogram_count(&registry, "sglang:inter_token_latency_seconds"),
            0
        );
        assert_eq!(
            histogram_count(&registry, "sglang:e2e_request_latency_seconds"),
            0
        );
        assert_eq!(counter_value(&registry, "sglang:num_requests_total"), 0.0);

        timing.record_tokens(1);
        timing.record_normal_completion();
        assert_eq!(
            histogram_count(&registry, "sglang:inter_token_latency_seconds"),
            1
        );
        assert_eq!(
            histogram_count(&registry, "sglang:e2e_request_latency_seconds"),
            1
        );
        assert_eq!(counter_value(&registry, "sglang:num_requests_total"), 1.0);

        let mut prefill_timing = metrics
            .request_timing("llama", 0, true, Instant::now() - Duration::from_millis(10))
            .await;
        prefill_timing.record_tokens(1);
        prefill_timing.record_normal_completion();
        assert_eq!(
            histogram_count(&registry, "sglang:e2e_request_latency_seconds"),
            1
        );
        assert_eq!(counter_value(&registry, "sglang:num_requests_total"), 1.0);
    }
}
