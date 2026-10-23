// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use anyhow::Result;
use dynamo_mocker::common::protocols::EngineType;
use dynamo_mocker::scheduler::MockerMetrics;
use dynamo_runtime::MetricsRegistry;
use dynamo_runtime::metrics::prometheus_names::{clamp_u64_to_i64, labels};
use prometheus::{
    Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge,
    IntGaugeVec, Opts,
};

const VLLM_LABELS: &[&str] = &["model_name", "engine"];
const SGLANG_SCHEDULER_LABELS: &[&str] = &[
    "model_name",
    "engine_type",
    "tp_rank",
    "pp_rank",
    "moe_ep_rank",
];
/// SGLang splits every request family except inter-token latency by `is_streaming`.
const SGLANG_REQUEST_LABELS: &[&str] = &["model_name", "engine_type", "is_streaming"];
const SGLANG_ITL_LABELS: &[&str] = &["model_name", "engine_type"];
const TRTLLM_LABELS: &[&str] = &["model_name", "engine_type"];
/// Labels `register_engine_metrics_callback` adds to every engine metric a Dynamo worker re-exports.
const DYNAMO_LABELS: &[&str] = &[
    labels::NAMESPACE,
    labels::COMPONENT,
    labels::ENDPOINT,
    labels::WORKER_ID,
    labels::MODEL,
];

/// Values for [`DYNAMO_LABELS`] other than `model`, taken from the worker's endpoint.
pub(crate) struct DynamoLabelValues {
    pub(crate) namespace: String,
    pub(crate) component: String,
    pub(crate) endpoint: String,
    pub(crate) worker_id: String,
}

pub(crate) struct NativeMockerMetrics {
    dp_size: u32,
    /// SGLang's `engine_type` label: `unified`, `prefill` or `decode`.
    sglang_engine_type: &'static str,
    collectors: NativeCollectors,
    /// Bound once at registration, so every series exists from startup like a real engine's.
    handles: OnceLock<NativeMetricHandles>,
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
    num_preemptions_total: IntCounterVec,
    time_to_first_token_seconds: HistogramVec,
    inter_token_latency_seconds: HistogramVec,
    request_time_per_output_token_seconds: HistogramVec,
    e2e_request_latency_seconds: HistogramVec,
}

/// Mirrors the TensorRT-LLM `MetricsCollector` families that `dynamo.trtllm` re-exports.
struct TrtllmCollectors {
    num_requests_running: IntGaugeVec,
    num_requests_waiting: IntGaugeVec,
    kv_cache_utilization: GaugeVec,
    time_to_first_token_seconds: HistogramVec,
    time_per_output_token_seconds: HistogramVec,
    e2e_request_latency_seconds: HistogramVec,
}

struct SglangCollectors {
    num_running_reqs: IntGaugeVec,
    num_queue_reqs: IntGaugeVec,
    cache_hit_rate: GaugeVec,
    num_requests_total: IntCounterVec,
    time_to_first_token_seconds: HistogramVec,
    inter_token_latency_seconds: HistogramVec,
    request_time_per_output_token_seconds: HistogramVec,
    e2e_request_latency_seconds: HistogramVec,
}

enum NativeMetricHandles {
    Vllm(HashMap<u32, VllmDpHandles>),
    Trtllm(TrtllmHandles),
    Sglang(SglangHandles),
}

struct VllmDpHandles {
    num_requests_running: IntGauge,
    num_requests_waiting: IntGauge,
    kv_cache_usage_perc: Gauge,
    num_preemptions_total: IntCounter,
    /// Highest scheduler preemption total seen, so the counter only moves forward.
    preemptions_seen: AtomicU64,
    request_metrics: NativeRequestMetricHandles,
}

/// One series per worker. `dynamo.trtllm` logs every attention-DP rank's stats into a single
/// collector, so each rank's report overwrites the gauges.
struct TrtllmHandles {
    num_requests_running: IntGauge,
    num_requests_waiting: IntGauge,
    kv_cache_utilization: Gauge,
    request_metrics: NativeRequestMetricHandles,
}

/// SGLang runs one scheduler, with its own gauges, per DP rank and one tokenizer for requests.
struct SglangHandles {
    schedulers: HashMap<u32, SglangSchedulerHandles>,
    request_metrics: NativeRequestMetricHandles,
}

struct SglangSchedulerHandles {
    num_running_reqs: IntGauge,
    num_queue_reqs: IntGauge,
    cache_hit_rate: Gauge,
}

#[derive(Clone)]
struct NativeRequestMetricHandles {
    engine: EngineType,
    /// SGLang prefill workers never record TTFT.
    time_to_first_token_seconds: Option<Histogram>,
    /// TensorRT-LLM has no inter-token latency histogram.
    inter_token_latency_seconds: Option<Histogram>,
    time_per_output_token_seconds: Histogram,
    e2e_request_latency_seconds: Histogram,
    /// Only SGLang counts completed requests.
    num_requests_total: Option<IntCounter>,
}

pub(crate) struct NativeRequestTiming {
    handles: Option<NativeRequestMetricHandles>,
    start: Instant,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    output_tokens: usize,
}

impl NativeMockerMetrics {
    pub(crate) fn new(
        engine_type: EngineType,
        dp_size: u32,
        sglang_engine_type: &'static str,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            dp_size,
            sglang_engine_type,
            collectors: NativeCollectors::new(engine_type, dp_size)?,
            handles: OnceLock::new(),
        }))
    }

    pub(crate) fn register(
        &self,
        registry: &MetricsRegistry,
        model_name: &str,
        dynamo: &DynamoLabelValues,
    ) -> Result<()> {
        if self.handles.get().is_some() {
            return Ok(());
        }

        self.collectors.register(registry)?;
        let _ = self.handles.set(self.bind_handles(model_name, dynamo));
        Ok(())
    }

    pub(crate) fn update_scheduler_snapshot(&self, metrics: &MockerMetrics) {
        match self.handles.get() {
            Some(NativeMetricHandles::Vllm(ranks)) => {
                let Some(handles) = ranks.get(&metrics.dp_rank) else {
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

                let seen = handles
                    .preemptions_seen
                    .fetch_max(metrics.vllm_preemptions_total, Ordering::Relaxed);
                if metrics.vllm_preemptions_total > seen {
                    handles
                        .num_preemptions_total
                        .inc_by(metrics.vllm_preemptions_total - seen);
                }
            }
            Some(NativeMetricHandles::Trtllm(handles)) => {
                handles
                    .num_requests_running
                    .set(clamp_u64_to_i64(metrics.running_requests));
                handles
                    .num_requests_waiting
                    .set(clamp_u64_to_i64(metrics.waiting_requests));
                handles
                    .kv_cache_utilization
                    .set(metrics.gpu_cache_usage_perc);
            }
            Some(NativeMetricHandles::Sglang(handles)) => {
                let Some(scheduler) = handles.schedulers.get(&metrics.dp_rank) else {
                    return;
                };
                scheduler
                    .num_running_reqs
                    .set(clamp_u64_to_i64(metrics.running_requests));
                scheduler
                    .num_queue_reqs
                    .set(clamp_u64_to_i64(metrics.waiting_requests));
                // No prefill tokens means no new observation, not a 0% cache hit rate.
                if metrics.sglang_cache_total_tokens > 0 {
                    scheduler.cache_hit_rate.set(
                        metrics.sglang_cache_hit_tokens as f64
                            / metrics.sglang_cache_total_tokens as f64,
                    );
                }
            }
            None => {}
        }
    }

    pub(crate) fn request_timing(&self, dp_rank: u32, start: Instant) -> NativeRequestTiming {
        let handles = match self.handles.get() {
            Some(NativeMetricHandles::Vllm(ranks)) => ranks
                .get(&dp_rank)
                .map(|handles| handles.request_metrics.clone()),
            Some(NativeMetricHandles::Trtllm(handles)) => Some(handles.request_metrics.clone()),
            Some(NativeMetricHandles::Sglang(handles)) => Some(handles.request_metrics.clone()),
            None => None,
        };

        NativeRequestTiming::new(start, handles)
    }

    fn bind_handles(&self, model_name: &str, dynamo: &DynamoLabelValues) -> NativeMetricHandles {
        let dynamo_values = [
            dynamo.namespace.as_str(),
            dynamo.component.as_str(),
            dynamo.endpoint.as_str(),
            dynamo.worker_id.as_str(),
            model_name,
        ];

        match &self.collectors {
            NativeCollectors::Vllm(collectors) => NativeMetricHandles::Vllm(
                (0..self.dp_size)
                    .map(|dp_rank| {
                        let engine = dp_rank.to_string();
                        let values = label_values(&[model_name, &engine], &dynamo_values);
                        let handles = VllmDpHandles {
                            num_requests_running: collectors
                                .num_requests_running
                                .with_label_values(&values),
                            num_requests_waiting: collectors
                                .num_requests_waiting
                                .with_label_values(&values),
                            kv_cache_usage_perc: collectors
                                .kv_cache_usage_perc
                                .with_label_values(&values),
                            num_preemptions_total: collectors
                                .num_preemptions_total
                                .with_label_values(&values),
                            preemptions_seen: AtomicU64::new(0),
                            request_metrics: NativeRequestMetricHandles {
                                engine: EngineType::Vllm,
                                time_to_first_token_seconds: Some(
                                    collectors
                                        .time_to_first_token_seconds
                                        .with_label_values(&values),
                                ),
                                inter_token_latency_seconds: Some(
                                    collectors
                                        .inter_token_latency_seconds
                                        .with_label_values(&values),
                                ),
                                time_per_output_token_seconds: collectors
                                    .request_time_per_output_token_seconds
                                    .with_label_values(&values),
                                e2e_request_latency_seconds: collectors
                                    .e2e_request_latency_seconds
                                    .with_label_values(&values),
                                num_requests_total: None,
                            },
                        };
                        (dp_rank, handles)
                    })
                    .collect(),
            ),
            NativeCollectors::Trtllm(collectors) => {
                let values = label_values(&[model_name, "trtllm"], &dynamo_values);
                NativeMetricHandles::Trtllm(TrtllmHandles {
                    num_requests_running: collectors
                        .num_requests_running
                        .with_label_values(&values),
                    num_requests_waiting: collectors
                        .num_requests_waiting
                        .with_label_values(&values),
                    kv_cache_utilization: collectors
                        .kv_cache_utilization
                        .with_label_values(&values),
                    request_metrics: NativeRequestMetricHandles {
                        engine: EngineType::Trtllm,
                        time_to_first_token_seconds: Some(
                            collectors
                                .time_to_first_token_seconds
                                .with_label_values(&values),
                        ),
                        inter_token_latency_seconds: None,
                        time_per_output_token_seconds: collectors
                            .time_per_output_token_seconds
                            .with_label_values(&values),
                        e2e_request_latency_seconds: collectors
                            .e2e_request_latency_seconds
                            .with_label_values(&values),
                        num_requests_total: None,
                    },
                })
            }
            NativeCollectors::Sglang(collectors) => {
                let engine_type = self.sglang_engine_type;
                // The mocker has no tensor, pipeline or expert parallelism, so each DP rank is a
                // rank-0 scheduler, as with plain data parallelism.
                let schedulers = (0..self.dp_size)
                    .map(|dp_rank| {
                        let dp_rank_label = dp_rank.to_string();
                        let mut engine_values = vec![model_name, engine_type, "0", "0", "0"];
                        if self.dp_size > 1 {
                            engine_values.push(&dp_rank_label);
                        }
                        let values = label_values(&engine_values, &dynamo_values);
                        let handles = SglangSchedulerHandles {
                            num_running_reqs: collectors
                                .num_running_reqs
                                .with_label_values(&values),
                            num_queue_reqs: collectors.num_queue_reqs.with_label_values(&values),
                            cache_hit_rate: collectors.cache_hit_rate.with_label_values(&values),
                        };
                        (dp_rank, handles)
                    })
                    .collect();
                // Dynamo always calls SGLang with `stream=True`.
                let request_values =
                    label_values(&[model_name, engine_type, "true"], &dynamo_values);
                let itl_values = label_values(&[model_name, engine_type], &dynamo_values);
                NativeMetricHandles::Sglang(SglangHandles {
                    schedulers,
                    request_metrics: NativeRequestMetricHandles {
                        engine: EngineType::Sglang,
                        time_to_first_token_seconds: (engine_type != "prefill").then(|| {
                            collectors
                                .time_to_first_token_seconds
                                .with_label_values(&request_values)
                        }),
                        inter_token_latency_seconds: Some(
                            collectors
                                .inter_token_latency_seconds
                                .with_label_values(&itl_values),
                        ),
                        time_per_output_token_seconds: collectors
                            .request_time_per_output_token_seconds
                            .with_label_values(&request_values),
                        e2e_request_latency_seconds: collectors
                            .e2e_request_latency_seconds
                            .with_label_values(&request_values),
                        num_requests_total: Some(
                            collectors
                                .num_requests_total
                                .with_label_values(&request_values),
                        ),
                    },
                })
            }
        }
    }
}

fn label_names(engine_labels: &[&'static str]) -> Vec<&'static str> {
    engine_labels.iter().chain(DYNAMO_LABELS).copied().collect()
}

fn label_values<'a>(engine_values: &[&'a str], dynamo_values: &[&'a str; 5]) -> Vec<&'a str> {
    engine_values.iter().chain(dynamo_values).copied().collect()
}

impl NativeCollectors {
    fn new(engine_type: EngineType, dp_size: u32) -> Result<Self> {
        match engine_type {
            EngineType::Vllm => Ok(Self::Vllm(VllmCollectors::new()?)),
            EngineType::Trtllm => Ok(Self::Trtllm(TrtllmCollectors::new()?)),
            EngineType::Sglang => Ok(Self::Sglang(SglangCollectors::new(dp_size)?)),
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
        let labels = label_names(VLLM_LABELS);
        Ok(Self {
            num_requests_running: IntGaugeVec::new(
                Opts::new(
                    "vllm:num_requests_running",
                    "Number of requests in model execution batches.",
                ),
                &labels,
            )?,
            num_requests_waiting: IntGaugeVec::new(
                Opts::new(
                    "vllm:num_requests_waiting",
                    "Number of requests waiting to be processed.",
                ),
                &labels,
            )?,
            kv_cache_usage_perc: GaugeVec::new(
                Opts::new(
                    "vllm:kv_cache_usage_perc",
                    "KV-cache usage. 1 means 100 percent usage.",
                ),
                &labels,
            )?,
            num_preemptions_total: IntCounterVec::new(
                Opts::new(
                    "vllm:num_preemptions_total",
                    "Cumulative number of preemption from the engine.",
                ),
                &labels,
            )?,
            time_to_first_token_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "vllm:time_to_first_token_seconds",
                    "Histogram of time to first token in seconds.",
                )
                .buckets(vllm_ttft_buckets()),
                &labels,
            )?,
            inter_token_latency_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "vllm:inter_token_latency_seconds",
                    "Histogram of inter-token latency in seconds.",
                )
                .buckets(vllm_itl_buckets()),
                &labels,
            )?,
            request_time_per_output_token_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "vllm:request_time_per_output_token_seconds",
                    "Histogram of time_per_output_token_seconds per request.",
                )
                .buckets(vllm_itl_buckets()),
                &labels,
            )?,
            e2e_request_latency_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "vllm:e2e_request_latency_seconds",
                    "Histogram of e2e request latency in seconds.",
                )
                .buckets(vllm_e2e_buckets()),
                &labels,
            )?,
        })
    }

    fn register(&self, registry: &MetricsRegistry) -> Result<()> {
        registry.add_metric(Box::new(self.num_requests_running.clone()))?;
        registry.add_metric(Box::new(self.num_requests_waiting.clone()))?;
        registry.add_metric(Box::new(self.kv_cache_usage_perc.clone()))?;
        registry.add_metric(Box::new(self.num_preemptions_total.clone()))?;
        registry.add_metric(Box::new(self.time_to_first_token_seconds.clone()))?;
        registry.add_metric(Box::new(self.inter_token_latency_seconds.clone()))?;
        registry.add_metric(Box::new(self.request_time_per_output_token_seconds.clone()))?;
        registry.add_metric(Box::new(self.e2e_request_latency_seconds.clone()))?;
        Ok(())
    }
}

impl TrtllmCollectors {
    // TensorRT-LLM's default buckets are the same lists as vLLM's TTFT, ITL and e2e buckets.
    fn new() -> Result<Self> {
        let labels = label_names(TRTLLM_LABELS);
        Ok(Self {
            num_requests_running: IntGaugeVec::new(
                Opts::new("trtllm_num_requests_running", "Number of active requests"),
                &labels,
            )?,
            num_requests_waiting: IntGaugeVec::new(
                Opts::new("trtllm_num_requests_waiting", "Number of queued requests"),
                &labels,
            )?,
            kv_cache_utilization: GaugeVec::new(
                Opts::new("trtllm_kv_cache_utilization", "KV cache utilization"),
                &labels,
            )?,
            time_to_first_token_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "trtllm_time_to_first_token_seconds",
                    "Histogram of time to first token in seconds.",
                )
                .buckets(vllm_ttft_buckets()),
                &labels,
            )?,
            time_per_output_token_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "trtllm_time_per_output_token_seconds",
                    "Histogram of time per output token in seconds.",
                )
                .buckets(vllm_itl_buckets()),
                &labels,
            )?,
            e2e_request_latency_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "trtllm_e2e_request_latency_seconds",
                    "Histogram of end to end request latency in seconds.",
                )
                .buckets(vllm_e2e_buckets()),
                &labels,
            )?,
        })
    }

    fn register(&self, registry: &MetricsRegistry) -> Result<()> {
        registry.add_metric(Box::new(self.num_requests_running.clone()))?;
        registry.add_metric(Box::new(self.num_requests_waiting.clone()))?;
        registry.add_metric(Box::new(self.kv_cache_utilization.clone()))?;
        registry.add_metric(Box::new(self.time_to_first_token_seconds.clone()))?;
        registry.add_metric(Box::new(self.time_per_output_token_seconds.clone()))?;
        registry.add_metric(Box::new(self.e2e_request_latency_seconds.clone()))?;
        Ok(())
    }
}

impl SglangCollectors {
    fn new(dp_size: u32) -> Result<Self> {
        // SGLang adds `dp_rank` to its scheduler labels when data parallelism is on.
        let mut scheduler_labels = SGLANG_SCHEDULER_LABELS.to_vec();
        if dp_size > 1 {
            scheduler_labels.push(labels::DP_RANK);
        }
        let scheduler_labels = label_names(&scheduler_labels);
        let request_labels = label_names(SGLANG_REQUEST_LABELS);
        let itl_labels = label_names(SGLANG_ITL_LABELS);
        Ok(Self {
            num_running_reqs: IntGaugeVec::new(
                Opts::new("sglang:num_running_reqs", "The number of running requests."),
                &scheduler_labels,
            )?,
            num_queue_reqs: IntGaugeVec::new(
                Opts::new(
                    "sglang:num_queue_reqs",
                    "The number of requests in the waiting queue.",
                ),
                &scheduler_labels,
            )?,
            cache_hit_rate: GaugeVec::new(
                Opts::new("sglang:cache_hit_rate", "The prefix cache hit rate."),
                &scheduler_labels,
            )?,
            num_requests_total: IntCounterVec::new(
                Opts::new("sglang:num_requests_total", "Number of requests processed."),
                &request_labels,
            )?,
            time_to_first_token_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "sglang:time_to_first_token_seconds",
                    "Histogram of time to first token in seconds.",
                )
                .buckets(sglang_ttft_buckets()),
                &request_labels,
            )?,
            inter_token_latency_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "sglang:inter_token_latency_seconds",
                    "Histogram of inter-token latency in seconds.",
                )
                .buckets(sglang_itl_buckets()),
                &itl_labels,
            )?,
            request_time_per_output_token_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "sglang:request_time_per_output_token_seconds",
                    "Per-request TPOT in seconds: (finished_time - first_token_time) / (completion_tokens - 1).",
                )
                .buckets(sglang_itl_buckets()),
                &request_labels,
            )?,
            e2e_request_latency_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "sglang:e2e_request_latency_seconds",
                    "Histogram of End-to-end request latency in seconds",
                )
                .buckets(sglang_e2e_buckets()),
                &request_labels,
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
        registry.add_metric(Box::new(self.request_time_per_output_token_seconds.clone()))?;
        registry.add_metric(Box::new(self.e2e_request_latency_seconds.clone()))?;
        Ok(())
    }
}

impl NativeRequestTiming {
    fn new(start: Instant, handles: Option<NativeRequestMetricHandles>) -> Self {
        Self {
            handles,
            start,
            first_token_at: None,
            last_token_at: None,
            output_tokens: 0,
        }
    }

    pub(crate) fn record_token(&mut self) {
        let Some(handles) = &self.handles else {
            return;
        };

        let now = Instant::now();
        self.output_tokens += 1;
        let Some(last_token_at) = self.last_token_at.replace(now) else {
            self.first_token_at = Some(now);
            if let Some(time_to_first_token_seconds) = &handles.time_to_first_token_seconds
                && !matches!(handles.engine, EngineType::Trtllm)
            {
                time_to_first_token_seconds.observe(now.duration_since(self.start).as_secs_f64());
            }
            return;
        };

        if let Some(inter_token_latency_seconds) = &handles.inter_token_latency_seconds {
            inter_token_latency_seconds.observe(now.duration_since(last_token_at).as_secs_f64());
        }
    }

    pub(crate) fn record_normal_completion(&self) {
        let Some(handles) = &self.handles else {
            return;
        };

        // TensorRT-LLM logs TTFT with the rest of a request's metrics once it finishes.
        if matches!(handles.engine, EngineType::Trtllm)
            && let (Some(time_to_first_token_seconds), Some(first)) =
                (&handles.time_to_first_token_seconds, self.first_token_at)
        {
            time_to_first_token_seconds.observe(first.duration_since(self.start).as_secs_f64());
        }

        // The first output token belongs to prefill, so TPOT spans the remaining tokens.
        let tpot = match (self.first_token_at, self.last_token_at) {
            (Some(first), Some(last)) if self.output_tokens > 1 => {
                last.duration_since(first).as_secs_f64() / (self.output_tokens - 1) as f64
            }
            _ => 0.0,
        };
        // vLLM records every TPOT, including 0 for single-token requests; SGLang and TensorRT-LLM
        // drop non-positive values.
        if tpot > 0.0 || matches!(handles.engine, EngineType::Vllm) {
            handles.time_per_output_token_seconds.observe(tpot);
        }

        // SGLang finishes a request after its KV handoff; vLLM and TensorRT-LLM end E2E at the
        // last token, so a prefill worker's handoff delay is not included.
        let end = match handles.engine {
            EngineType::Sglang => Instant::now(),
            _ => self.last_token_at.unwrap_or_else(Instant::now),
        };
        handles
            .e2e_request_latency_seconds
            .observe(end.duration_since(self.start).as_secs_f64());
        if let Some(num_requests_total) = &handles.num_requests_total {
            num_requests_total.inc();
        }
    }
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
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use dynamo_mocker::common::protocols::EngineType;
    use dynamo_mocker::scheduler::MockerMetrics;
    use dynamo_runtime::MetricsRegistry;
    use prometheus::proto::{Metric, MetricFamily, MetricType};

    use super::{DYNAMO_LABELS, DynamoLabelValues, NativeMockerMetrics};

    fn register_for_test(
        engine_type: EngineType,
        dp_size: u32,
        sglang_engine_type: &'static str,
    ) -> (MetricsRegistry, Arc<NativeMockerMetrics>) {
        let registry = MetricsRegistry::new();
        let metrics = NativeMockerMetrics::new(engine_type, dp_size, sglang_engine_type).unwrap();
        let dynamo = DynamoLabelValues {
            namespace: "dynamo".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
            worker_id: "1f".to_string(),
        };
        metrics.register(&registry, "llama", &dynamo).unwrap();
        (registry, metrics)
    }

    fn gather_family(registry: &MetricsRegistry, name: &str) -> MetricFamily {
        registry
            .get_prometheus_registry()
            .gather()
            .into_iter()
            .find(|family| family.name() == name)
            .unwrap_or_else(|| panic!("missing metric family {name}"))
    }

    fn label_set(engine_labels: &[&str]) -> HashSet<String> {
        engine_labels
            .iter()
            .chain(DYNAMO_LABELS)
            .map(|label| label.to_string())
            .collect()
    }

    fn metric_labels(metric: &Metric) -> HashSet<String> {
        metric
            .get_label()
            .iter()
            .map(|label| label.name().to_string())
            .collect()
    }

    fn label_value<'a>(metric: &'a Metric, label_name: &str) -> &'a str {
        metric
            .get_label()
            .iter()
            .find(|label| label.name() == label_name)
            .unwrap_or_else(|| panic!("missing label {label_name}"))
            .value()
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

    fn gauge_value(metric: &Metric) -> f64 {
        metric.gauge.as_ref().unwrap().value()
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

    #[test]
    fn vllm_scrape_contract_uses_native_names_and_engine_labels() {
        let (registry, metrics) = register_for_test(EngineType::Vllm, 2, "unified");

        // Series exist from registration, before any snapshot or request.
        let running = gather_family(&registry, "vllm:num_requests_running");
        assert_eq!(running.get_metric().len(), 2);
        assert_eq!(gauge_value(metric_with_label(&running, "engine", "1")), 0.0);

        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 3, 10, 2, 4, 5, 0, 0));
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(1, 4, 10, 7, 8, 1, 0, 0));
        let mut timing = metrics.request_timing(0, Instant::now());
        timing.record_token();
        timing.record_normal_completion();

        let running = gather_family(&registry, "vllm:num_requests_running");
        assert_eq!(running.get_field_type(), MetricType::GAUGE);
        let engine0 = metric_with_label(&running, "engine", "0");
        assert_eq!(metric_labels(engine0), label_set(&["model_name", "engine"]));
        assert_eq!(label_value(engine0, "model"), "llama");
        assert_eq!(label_value(engine0, "worker_id"), "1f");
        assert_eq!(gauge_value(engine0), 2.0);
        assert_eq!(gauge_value(metric_with_label(&running, "engine", "1")), 7.0);

        assert_eq!(
            gather_family(&registry, "vllm:kv_cache_usage_perc").get_field_type(),
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
        assert!(!text.contains("gpu_cache_usage_perc"));
    }

    #[test]
    fn vllm_preemption_counter_is_monotonic_across_snapshots() {
        let (registry, metrics) = register_for_test(EngineType::Vllm, 1, "unified");

        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 0, 10, 0, 0, 3, 0, 0));
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 0, 10, 0, 0, 3, 0, 0));
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 0, 10, 0, 0, 2, 0, 0));
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 0, 10, 0, 0, 5, 0, 0));

        assert_eq!(counter_value(&registry, "vllm:num_preemptions_total"), 5.0);
    }

    #[test]
    fn sglang_scrape_contract_uses_scheduler_and_request_label_sets() {
        let (registry, metrics) = register_for_test(EngineType::Sglang, 2, "unified");
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 2, 10, 1, 3, 0, 2, 4));
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(1, 2, 10, 4, 5, 0, 1, 2));

        let mut timing = metrics.request_timing(0, Instant::now());
        timing.record_token();
        timing.record_normal_completion();

        let running = gather_family(&registry, "sglang:num_running_reqs");
        assert_eq!(running.get_field_type(), MetricType::GAUGE);
        let rank1 = metric_with_label(&running, "dp_rank", "1");
        assert_eq!(
            metric_labels(rank1),
            label_set(&[
                "model_name",
                "engine_type",
                "tp_rank",
                "pp_rank",
                "moe_ep_rank",
                "dp_rank",
            ])
        );
        assert_eq!(label_value(rank1, "engine_type"), "unified");
        assert_eq!(gauge_value(rank1), 4.0);
        assert_eq!(
            gauge_value(metric_with_label(&running, "dp_rank", "0")),
            1.0
        );

        let queue = gather_family(&registry, "sglang:num_queue_reqs");
        assert_eq!(gauge_value(metric_with_label(&queue, "dp_rank", "1")), 5.0);
        let cache_hit_rate = gather_family(&registry, "sglang:cache_hit_rate");
        assert_eq!(
            gauge_value(metric_with_label(&cache_hit_rate, "dp_rank", "0")),
            0.5
        );

        assert_eq!(
            gather_family(&registry, "sglang:num_requests_total").get_field_type(),
            MetricType::COUNTER
        );
        for name in [
            "sglang:num_requests_total",
            "sglang:time_to_first_token_seconds",
            "sglang:request_time_per_output_token_seconds",
            "sglang:e2e_request_latency_seconds",
        ] {
            let family = gather_family(&registry, name);
            let metric = family.get_metric().first().unwrap();
            assert_eq!(
                metric_labels(metric),
                label_set(&["model_name", "engine_type", "is_streaming"]),
                "{name}"
            );
            assert_eq!(label_value(metric, "is_streaming"), "true", "{name}");
        }
        let itl = gather_family(&registry, "sglang:inter_token_latency_seconds");
        assert_eq!(
            metric_labels(itl.get_metric().first().unwrap()),
            label_set(&["model_name", "engine_type"])
        );
    }

    #[test]
    fn sglang_cache_hit_rate_ignores_passes_without_prefill_tokens() {
        let (registry, metrics) = register_for_test(EngineType::Sglang, 1, "unified");
        let cache_hit_rate = || {
            gauge_value(
                gather_family(&registry, "sglang:cache_hit_rate")
                    .get_metric()
                    .first()
                    .unwrap(),
            )
        };

        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 0, 10, 0, 0, 0, 5, 10));
        assert_eq!(cache_hit_rate(), 0.5);

        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 0, 10, 0, 0, 0, 0, 0));
        assert_eq!(cache_hit_rate(), 0.5);

        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 0, 10, 0, 0, 0, 0, 10));
        assert_eq!(cache_hit_rate(), 0.0);
    }

    #[test]
    fn request_timing_records_successful_token_lifecycle() {
        let (registry, metrics) = register_for_test(EngineType::Sglang, 1, "decode");

        let mut timing = metrics.request_timing(0, Instant::now() - Duration::from_millis(10));
        timing.record_token();
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
        assert_eq!(
            histogram_count(&registry, "sglang:request_time_per_output_token_seconds"),
            0
        );
        assert_eq!(counter_value(&registry, "sglang:num_requests_total"), 0.0);

        std::thread::sleep(Duration::from_millis(1));
        timing.record_token();
        timing.record_normal_completion();
        assert_eq!(
            histogram_count(&registry, "sglang:inter_token_latency_seconds"),
            1
        );
        assert_eq!(
            histogram_count(&registry, "sglang:e2e_request_latency_seconds"),
            1
        );
        assert_eq!(
            histogram_count(&registry, "sglang:request_time_per_output_token_seconds"),
            1
        );
        assert_eq!(counter_value(&registry, "sglang:num_requests_total"), 1.0);

        // A one-token request, such as a prefill worker's, has no TPOT.
        let mut single_token = metrics.request_timing(0, Instant::now());
        single_token.record_token();
        single_token.record_normal_completion();
        assert_eq!(
            histogram_count(&registry, "sglang:time_to_first_token_seconds"),
            2
        );
        assert_eq!(
            histogram_count(&registry, "sglang:e2e_request_latency_seconds"),
            2
        );
        assert_eq!(
            histogram_count(&registry, "sglang:request_time_per_output_token_seconds"),
            1
        );
        assert_eq!(counter_value(&registry, "sglang:num_requests_total"), 2.0);
    }

    #[test]
    fn vllm_records_zero_tpot_for_single_token_requests() {
        let (registry, metrics) = register_for_test(EngineType::Vllm, 1, "unified");

        let mut timing = metrics.request_timing(0, Instant::now());
        timing.record_token();
        timing.record_normal_completion();

        let family = gather_family(&registry, "vllm:request_time_per_output_token_seconds");
        let histogram = family.get_metric()[0].histogram.as_ref().unwrap();
        assert_eq!(histogram.sample_count(), 1);
        assert_eq!(histogram.sample_sum(), 0.0);
    }

    #[test]
    fn trtllm_scrape_contract_uses_native_names_and_last_rank_report() {
        let (registry, metrics) = register_for_test(EngineType::Trtllm, 2, "unified");
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(0, 3, 10, 2, 4, 0, 0, 0));
        metrics.update_scheduler_snapshot(&MockerMetrics::from_parts(1, 4, 10, 7, 8, 0, 0, 0));

        let mut timing = metrics.request_timing(1, Instant::now());
        timing.record_token();
        // TensorRT-LLM records TTFT only when the request finishes.
        assert_eq!(
            histogram_count(&registry, "trtllm_time_to_first_token_seconds"),
            0
        );
        std::thread::sleep(Duration::from_millis(1));
        timing.record_token();
        timing.record_normal_completion();
        assert_eq!(
            histogram_count(&registry, "trtllm_time_to_first_token_seconds"),
            1
        );

        let mut single_token = metrics.request_timing(0, Instant::now());
        single_token.record_token();
        single_token.record_normal_completion();

        let running = gather_family(&registry, "trtllm_num_requests_running");
        assert_eq!(running.get_field_type(), MetricType::GAUGE);
        assert_eq!(running.get_metric().len(), 1);
        let running_metric = running.get_metric().first().unwrap();
        assert_eq!(
            metric_labels(running_metric),
            label_set(&["model_name", "engine_type"])
        );
        assert_eq!(label_value(running_metric, "engine_type"), "trtllm");
        assert_eq!(gauge_value(running_metric), 7.0);
        let waiting = gather_family(&registry, "trtllm_num_requests_waiting");
        assert_eq!(gauge_value(&waiting.get_metric()[0]), 8.0);
        let utilization = gather_family(&registry, "trtllm_kv_cache_utilization");
        assert_eq!(gauge_value(&utilization.get_metric()[0]), 0.4);

        assert_eq!(
            histogram_count(&registry, "trtllm_time_to_first_token_seconds"),
            2
        );
        assert_eq!(
            histogram_count(&registry, "trtllm_time_per_output_token_seconds"),
            1
        );
        assert_eq!(
            histogram_count(&registry, "trtllm_e2e_request_latency_seconds"),
            2
        );

        let text = registry.prometheus_expfmt_combined().unwrap();
        assert!(!text.contains("trtllm:"));
        assert!(!text.contains("inter_token_latency"));
    }

    #[test]
    fn sglang_prefill_worker_skips_ttft() {
        let (registry, metrics) = register_for_test(EngineType::Sglang, 1, "prefill");

        let mut timing = metrics.request_timing(0, Instant::now());
        timing.record_token();
        timing.record_normal_completion();

        assert_eq!(
            histogram_count(&registry, "sglang:e2e_request_latency_seconds"),
            1
        );
        assert_eq!(counter_value(&registry, "sglang:num_requests_total"), 1.0);
        let text = registry.prometheus_expfmt_combined().unwrap();
        assert!(!text.contains("sglang:time_to_first_token_seconds_count"));
    }

    #[test]
    fn e2e_ends_at_last_token_except_sglang() {
        let handoff = Duration::from_millis(50);
        for (engine_type, family, ends_at_completion) in [
            (
                EngineType::Trtllm,
                "trtllm_e2e_request_latency_seconds",
                false,
            ),
            (
                EngineType::Sglang,
                "sglang:e2e_request_latency_seconds",
                true,
            ),
        ] {
            let (registry, metrics) = register_for_test(engine_type, 1, "prefill");
            let mut timing = metrics.request_timing(0, Instant::now());
            timing.record_token();
            std::thread::sleep(handoff);
            timing.record_normal_completion();

            let e2e = gather_family(&registry, family).get_metric()[0]
                .histogram
                .as_ref()
                .unwrap()
                .sample_sum();
            assert_eq!(e2e >= handoff.as_secs_f64(), ends_at_completion, "{family}");
        }
    }
}
