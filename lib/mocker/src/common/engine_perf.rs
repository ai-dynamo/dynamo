// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-level performance queries backed by AIC forward-pass metrics.
//!
//! This module deliberately sits next to, not inside, `perf_model`: the existing
//! `PerfModel` is a scheduler timing model used to sleep mock engine passes,
//! while this shim answers planner/router-style questions from FPM state.

use std::cmp;
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::time::Duration;

use aiconfigurator_core::{
    BackendKind, DataType, ENGINE_CONFIG_SCHEMA_VERSION, EngineConfig, FPM_VERSION,
    ForwardPassMetrics, ForwardPassPerfDiagnostics, ForwardPassPerfModel, ForwardPassPerfOptions,
    ParallelMapping, QuantizationConfig, QueuedRequestMetrics, ScheduledRequestMetrics,
};
use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::common::protocols::{ForwardPassSnapshot, MockEngineArgs, WorkerType};

const DEFAULT_AIC_SYSTEM: &str = "h200_sxm";
const MAX_CAPACITY_SEARCH_CANDIDATES: u32 = 128;
const MAX_KV_HIT_RATE_DISCOUNT: f64 = 0.95;
const AIC_NEXTN_KEY: &str = "nextn";
const AIC_NEXTN_ACCEPT_RATES_KEY: &str = "nextn_accept_rates";
const RAW_AIC_NEXTN_ACCEPT_RATES: &str = "0,0,0,0,0";

/// Engine limits needed by planner/router-level queries.
///
/// These also seed AIC correction bounds when callers do not pass explicit
/// `ForwardPassPerfOptions`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Validate)]
pub struct EnginePerfLimits {
    #[validate(range(min = 1))]
    pub max_num_batched_tokens: u32,
    #[validate(range(min = 1))]
    pub max_num_seqs: u32,
    #[validate(range(min = 1))]
    pub max_kv_tokens: u32,
}

/// AIC engine identity and model configuration accepted by the mocker shim.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AicEngineConfig {
    pub model_name: String,
    pub model_arch: Option<String>,
    pub system_name: String,
    pub systems_path: Option<String>,
    pub backend: String,
    pub backend_version: Option<String>,
    pub tp_size: u32,
    pub pp_size: u32,
    pub moe_tp_size: Option<u32>,
    pub moe_ep_size: Option<u32>,
    pub attention_dp_size: Option<u32>,
    pub weight_dtype: Option<String>,
    pub moe_dtype: Option<String>,
    pub activation_dtype: Option<String>,
    pub kv_cache_dtype: Option<String>,
    pub kv_block_size: Option<u32>,
    pub extra: BTreeMap<String, String>,
}

impl AicEngineConfig {
    pub fn into_aic_config(self) -> Result<EngineConfig> {
        // `model_arch` is intentionally not forwarded: aiconfigurator main
        // dropped it from `EngineConfig` (architecture is inferred from the HF
        // model_path) and its deserializer ignores a stray `model_arch` key.
        Ok(EngineConfig {
            schema_version: ENGINE_CONFIG_SCHEMA_VERSION,
            model_name: self.model_name,
            system_name: self.system_name,
            systems_path: self.systems_path.map(PathBuf::from),
            backend: parse_backend_kind(&self.backend)?,
            backend_version: self.backend_version,
            kv_block_size: self.kv_block_size,
            parallel: ParallelMapping {
                tp_size: self.tp_size,
                pp_size: self.pp_size,
                attention_dp_size: self.attention_dp_size,
                moe_tp_size: self.moe_tp_size,
                moe_ep_size: self.moe_ep_size,
            },
            quantization: QuantizationConfig {
                weight_dtype: self
                    .weight_dtype
                    .as_deref()
                    .map(parse_data_type)
                    .transpose()?,
                moe_dtype: self.moe_dtype.as_deref().map(parse_data_type).transpose()?,
                activation_dtype: self
                    .activation_dtype
                    .as_deref()
                    .map(parse_data_type)
                    .transpose()?,
                kv_cache_dtype: self
                    .kv_cache_dtype
                    .as_deref()
                    .map(parse_data_type)
                    .transpose()?,
            },
            speculative: None,
            extra: self.extra,
        })
    }
}

fn aic_config_for_raw_iteration_time(mut config: EngineConfig) -> EngineConfig {
    config.extra.insert(
        AIC_NEXTN_ACCEPT_RATES_KEY.to_string(),
        RAW_AIC_NEXTN_ACCEPT_RATES.to_string(),
    );
    config
}

fn aic_engine_config_for_raw_iteration_time(config: AicEngineConfig) -> Result<EngineConfig> {
    Ok(aic_config_for_raw_iteration_time(config.into_aic_config()?))
}

impl EnginePerfLimits {
    pub fn new(max_num_batched_tokens: u32, max_num_seqs: u32, max_kv_tokens: u32) -> Result<Self> {
        let limits = Self {
            max_num_batched_tokens,
            max_num_seqs,
            max_kv_tokens,
        };
        limits.validate().context("invalid engine perf limits")?;
        Ok(limits)
    }

    pub fn from_mock_engine_args(args: &MockEngineArgs) -> Result<Self> {
        let max_num_batched_tokens = to_u32(
            args.max_num_batched_tokens.unwrap_or(8192),
            "max_num_batched_tokens",
        )?;
        let max_num_seqs = to_u32(args.max_num_seqs.unwrap_or(512), "max_num_seqs")?;
        let max_kv_tokens = args
            .num_gpu_blocks
            .checked_mul(args.block_size)
            .context("num_gpu_blocks * block_size overflows usize")?;
        Self::new(
            max_num_batched_tokens,
            max_num_seqs,
            to_u32(max_kv_tokens, "max_kv_tokens")?,
        )
    }
}

/// Capacity search objective.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationTarget {
    Throughput,
    Latency,
}

impl Default for OptimizationTarget {
    fn default() -> Self {
        Self::Throughput
    }
}

/// Capacity query request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EngineCapacityRequest {
    pub isl: u32,
    pub osl: u32,
    pub ttft_sla: Option<Duration>,
    pub itl_sla: Option<Duration>,
    pub e2e_latency_sla: Option<Duration>,
    /// Average accepted tokens per decode forward, including the base token.
    ///
    /// Values below one or non-finite values are treated as one so existing raw
    /// decode capacity requests keep their historical behavior.
    #[serde(default = "default_accept_length")]
    pub accept_length: f64,
    /// Prefix-cache hit rate used to discount prefill compute only.
    ///
    /// Decode context/KV residency still uses the raw `isl`, because prefix
    /// reuse reduces prefill work but does not shrink the KV footprint for
    /// generated-token decode.
    pub kv_hit_rate: Option<f64>,
    pub optimization_target: OptimizationTarget,
}

/// Capacity query result.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EngineCapacity {
    pub rps: f64,
    pub ttft: Option<Duration>,
    pub itl: Option<Duration>,
    pub e2e_latency: Option<Duration>,
    pub eligible: bool,
}

#[derive(Clone, Debug, Default)]
pub struct EnginePerfModelInputs {
    pub engine_args: Option<MockEngineArgs>,
    pub aic_config: Option<AicEngineConfig>,
    pub worker_type: Option<WorkerType>,
    pub limits: Option<EnginePerfLimits>,
    pub options: Option<ForwardPassPerfOptions>,
    pub bootstrap_fpms: Vec<Vec<ForwardPassSnapshot>>,
}

#[derive(Clone, Debug)]
struct MovingAverage {
    samples: VecDeque<f64>,
    sum: f64,
    max_len: usize,
    seen_nonzero: bool,
}

impl MovingAverage {
    fn new(max_len: usize) -> Self {
        Self {
            samples: VecDeque::new(),
            sum: 0.0,
            max_len: max_len.max(1),
            seen_nonzero: false,
        }
    }

    fn add_after_first_nonzero(&mut self, value: f64) {
        if value > 0.0 {
            self.seen_nonzero = true;
        }
        if self.seen_nonzero {
            self.add(value.max(0.0));
        }
    }

    fn add(&mut self, value: f64) {
        self.samples.push_back(value);
        self.sum += value;
        while self.samples.len() > self.max_len {
            if let Some(old) = self.samples.pop_front() {
                self.sum -= old;
            }
        }
    }

    fn value(&self) -> f64 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.sum / self.samples.len() as f64
        }
    }
}

#[derive(Clone, Debug)]
struct AggLoadAverages {
    avg_isl: MovingAverage,
    avg_num_prefill: MovingAverage,
    avg_prefill_tokens: MovingAverage,
    avg_decode_len: MovingAverage,
}

impl AggLoadAverages {
    fn new(max_observations: usize) -> Self {
        Self {
            avg_isl: MovingAverage::new(max_observations),
            avg_num_prefill: MovingAverage::new(max_observations),
            avg_prefill_tokens: MovingAverage::new(max_observations),
            avg_decode_len: MovingAverage::new(max_observations),
        }
    }

    fn observe_iterations(&mut self, iterations: &[Vec<ForwardPassSnapshot>]) {
        for metrics_by_rank in iterations {
            self.observe_iteration(metrics_by_rank);
        }
    }

    fn observe_iteration(&mut self, metrics_by_rank: &[ForwardPassSnapshot]) {
        let Some(snapshot) = metrics_by_rank.iter().max_by_key(|snapshot| {
            u128::from(snapshot.sum_prefill_tokens)
                + u128::from(snapshot.sum_decode_kv_tokens)
                + u128::from(snapshot.num_decode_requests)
        }) else {
            return;
        };

        if snapshot.num_prefill_requests > 0 {
            self.avg_isl
                .add(snapshot.sum_prefill_tokens as f64 / f64::from(snapshot.num_prefill_requests));
            self.avg_num_prefill
                .add_after_first_nonzero(f64::from(snapshot.num_prefill_requests));
        } else {
            self.avg_num_prefill.add_after_first_nonzero(0.0);
        }
        self.avg_prefill_tokens
            .add_after_first_nonzero(snapshot.sum_prefill_tokens as f64);

        if snapshot.num_decode_requests > 0 {
            self.avg_decode_len.add(
                snapshot.sum_decode_kv_tokens as f64 / f64::from(snapshot.num_decode_requests),
            );
        }
    }

    fn typical_prefill_for_mixed_itl(&self) -> Result<(u32, u32)> {
        let tokens =
            ceil_positive_f64_to_u32(self.avg_prefill_tokens.value(), "avg prefill tokens")?;
        let requests =
            ceil_positive_f64_to_u32(self.avg_num_prefill.value(), "avg prefill requests")?;
        Ok((requests.max(u32::from(tokens > 0)), tokens))
    }
}

#[derive(Clone, Copy, Debug)]
struct PrefillChunkPlan {
    full_chunks: u64,
    tail_tokens: u64,
}

impl PrefillChunkPlan {
    fn new(tokens: u64, max_chunk: u64) -> Self {
        debug_assert!(max_chunk > 0);
        Self {
            full_chunks: tokens / max_chunk,
            tail_tokens: tokens % max_chunk,
        }
    }

    fn chunk_at(&self, iteration: u64, max_chunk: u64) -> u64 {
        if iteration < self.full_chunks {
            max_chunk
        } else if iteration == self.full_chunks {
            self.tail_tokens
        } else {
            0
        }
    }
}

#[derive(Clone, Debug)]
pub struct EnginePerfModel {
    model: ForwardPassPerfModel,
    worker_type: WorkerType,
    limits: EnginePerfLimits,
    attention_dp_size: usize,
    load_averages: AggLoadAverages,
}

impl EnginePerfModel {
    /// Build the best available model from whatever inputs the caller has.
    ///
    /// Explicit `aic_config` takes precedence. If it is absent and
    /// `engine_args.aic_backend` is set, `engine_args.aic_model_path` must also
    /// be set so the shim can derive a native AIC config. This side API does not
    /// receive the loaded `LocalModel`, so it cannot recover the source model path
    /// used by the live mocker entrypoint. Without `aic_backend`, the model starts
    /// in regression-only mode. `bootstrap_fpms` are tuned after construction.
    pub fn best_available(inputs: EnginePerfModelInputs) -> Result<Self> {
        let worker_type = resolve_worker_type(inputs.worker_type, inputs.engine_args.as_ref())?;
        let limits = resolve_limits(inputs.limits, inputs.engine_args.as_ref())?;
        let options = resolve_options(inputs.options, &limits);
        let load_averages = AggLoadAverages::new(options.max_observations);
        let aic_config = match inputs.aic_config {
            Some(config) => Some(aic_engine_config_for_raw_iteration_time(config)?),
            None => inputs
                .engine_args
                .as_ref()
                .map(aic_config_from_mock_engine_args)
                .transpose()?
                .flatten(),
        };
        let attention_dp_size = if let Some(size) = aic_config
            .as_ref()
            .and_then(|config| config.parallel.attention_dp_size)
        {
            size
        } else {
            inputs
                .engine_args
                .as_ref()
                .and_then(|args| args.aic_attention_dp_size)
                .map(|value| to_u32(value, "aic_attention_dp_size"))
                .transpose()?
                .unwrap_or(1)
        }
        .max(1) as usize;

        let model = match aic_config {
            Some(config) => ForwardPassPerfModel::best_available(config, options)
                .context("failed to create AIC forward-pass perf model")?,
            None => ForwardPassPerfModel::from_regression(options)
                .context("failed to create regression forward-pass perf model")?,
        };

        let mut this = Self {
            model,
            worker_type,
            limits,
            attention_dp_size,
            load_averages,
        };
        if !inputs.bootstrap_fpms.is_empty() {
            this.tune_with_fpms(&inputs.bootstrap_fpms)?;
        }
        Ok(this)
    }

    /// Build a strict native AIC model.
    ///
    /// This constructor fails when AIC does not support the supplied config.
    /// Use `best_available` when fallback regression is desired.
    pub fn from_native(
        aic_config: AicEngineConfig,
        worker_type: WorkerType,
        limits: EnginePerfLimits,
        options: Option<ForwardPassPerfOptions>,
    ) -> Result<Self> {
        let aic_config = aic_engine_config_for_raw_iteration_time(aic_config)?;
        let attention_dp_size = aic_config.parallel.attention_dp_size.unwrap_or(1).max(1) as usize;
        limits.validate().context("invalid engine perf limits")?;
        let resolved_options = resolve_options(options, &limits);
        let load_averages = AggLoadAverages::new(resolved_options.max_observations);
        let model = ForwardPassPerfModel::from_native(aic_config, resolved_options)
            .context("failed to create native AIC forward-pass perf model")?;
        Ok(Self {
            model,
            worker_type,
            limits,
            attention_dp_size,
            load_averages,
        })
    }

    /// Build a model that fits forward-pass duration directly from observed FPM wall times.
    pub fn from_regression(
        worker_type: WorkerType,
        limits: EnginePerfLimits,
        options: Option<ForwardPassPerfOptions>,
    ) -> Result<Self> {
        limits.validate().context("invalid engine perf limits")?;
        let resolved_options = resolve_options(options, &limits);
        let load_averages = AggLoadAverages::new(resolved_options.max_observations);
        let model = ForwardPassPerfModel::from_regression(resolved_options)
            .context("failed to create regression forward-pass perf model")?;
        Ok(Self {
            model,
            worker_type,
            limits,
            attention_dp_size: 1,
            load_averages,
        })
    }

    /// Estimate one scheduled forward-pass iteration.
    ///
    /// Input is one FPM per attention-DP rank. Only scheduled workload fields
    /// are used; queued fields and `wall_time` are ignored. The shim accepts
    /// current-version FPMs and unstamped local snapshots (`version == 0`) only.
    pub fn estimate_forward_pass_time(
        &self,
        metrics_by_rank: &[ForwardPassSnapshot],
    ) -> Result<Option<Duration>> {
        self.validate_metrics_by_rank(metrics_by_rank)?;
        let metrics = snapshots_to_aic_metrics(metrics_by_rank)?;
        self.estimate_aic_metrics(&metrics)
    }

    /// Tune the in-memory model with observed iterations.
    ///
    /// The outer slice is iterations; each inner slice is the per-attention-DP
    /// rank FPMs for that iteration. AIC uses scheduled workload fields as
    /// features and `wall_time` as the observed target. The shim accepts
    /// current-version FPMs and unstamped local snapshots (`version == 0`) only.
    pub fn tune_with_fpms(&mut self, iterations: &[Vec<ForwardPassSnapshot>]) -> Result<()> {
        for metrics_by_rank in iterations {
            self.validate_metrics_by_rank(metrics_by_rank)?;
        }
        let aic_iterations = iterations
            .iter()
            .map(|metrics_by_rank| snapshots_to_aic_metrics(metrics_by_rank))
            .collect::<Result<Vec<_>>>()?;
        self.model
            .tune_with_fpms(&aic_iterations)
            .context("failed to tune AIC forward-pass perf model")?;
        self.load_averages.observe_iterations(iterations);
        Ok(())
    }

    /// Return AIC model diagnostics, including native/fallback mode and tuning counts.
    pub fn diagnostics(&self) -> ForwardPassPerfDiagnostics {
        self.model.diagnostics()
    }

    /// Minimum native correction factor across currently ready AIC correction buckets.
    pub fn min_correction_factor(&self) -> Option<f64> {
        self.model.min_correction_factor()
    }

    /// Maximum native correction factor across currently ready AIC correction buckets.
    pub fn max_correction_factor(&self) -> Option<f64> {
        self.model.max_correction_factor()
    }

    /// Average native correction factor across currently ready AIC correction buckets.
    pub fn avg_correction_factor(&self) -> Option<f64> {
        self.model.avg_correction_factor()
    }

    /// Estimate queued prefill drain time.
    ///
    /// Prefill workers use only queued prefill fields. Aggregated workers use
    /// queued prefill plus the current scheduled decode load so the synthetic
    /// FPM remains a mixed workload.
    ///
    /// The current FPM queued-prefill fields, including unstamped local
    /// snapshots accepted as `version == 0`, carry raw queued prompt tokens and
    /// no KV-cache reuse estimate. Callers that want prefix-cache reuse included
    /// must adjust `queued_requests.sum_prefill_tokens` before calling this shim.
    /// FPM v1 exposes aggregate queued prefill load, not per-request queued state,
    /// so this helper uses a compressed full-chunks-plus-tail approximation.
    /// TODO: switch to iteration-level queue simulation when FPM includes accurate
    /// per-request queued prefill state and KV-cache reuse information.
    pub fn get_queued_prefill_time(
        &self,
        metrics_by_rank: &[ForwardPassSnapshot],
    ) -> Result<Option<Duration>> {
        if self.worker_type == WorkerType::Decode {
            return Ok(None);
        }
        self.validate_metrics_by_rank(metrics_by_rank)?;

        let remaining = metrics_by_rank
            .iter()
            .map(|snapshot| snapshot.sum_queued_prefill_tokens)
            .collect::<Vec<_>>();
        if remaining.iter().all(|tokens| *tokens == 0) {
            return Ok(Some(Duration::ZERO));
        }

        let max_chunk = u64::from(self.limits.max_num_batched_tokens);
        let plans = remaining
            .iter()
            .map(|tokens| PrefillChunkPlan::new(*tokens, max_chunk))
            .collect::<Vec<_>>();
        let mut breakpoints = vec![0];
        for plan in &plans {
            if plan.full_chunks > 0 {
                breakpoints.push(plan.full_chunks);
            }
            if plan.tail_tokens > 0 {
                breakpoints.push(plan.full_chunks + 1);
            }
        }
        breakpoints.sort_unstable();
        breakpoints.dedup();

        let mut total = Duration::ZERO;
        for window in breakpoints.windows(2) {
            let iteration = window[0];
            let repeat_count = window[1] - window[0];
            if repeat_count == 0 {
                continue;
            }
            let mut chunk_metrics = Vec::with_capacity(metrics_by_rank.len());
            for (snapshot, plan) in metrics_by_rank.iter().zip(plans.iter()) {
                let chunk = plan.chunk_at(iteration, max_chunk);
                let mut metrics = aic_identity_from_snapshot(snapshot);
                metrics.scheduled_requests.num_prefill_requests =
                    estimate_prefill_request_count(snapshot.num_queued_prefill, chunk);
                metrics.scheduled_requests.sum_prefill_tokens =
                    u64_to_u32(chunk, "queued prefill chunk tokens")?;
                if self.worker_type == WorkerType::Aggregated {
                    metrics.scheduled_requests.num_decode_requests = snapshot.num_decode_requests;
                    metrics.scheduled_requests.sum_decode_kv_tokens =
                        u64_to_u32(snapshot.sum_decode_kv_tokens, "scheduled decode KV tokens")?;
                    metrics.scheduled_requests.var_decode_kv_tokens = snapshot.var_decode_kv_tokens;
                }
                chunk_metrics.push(metrics);
            }
            let Some(duration) = self.estimate_aic_metrics(&chunk_metrics)? else {
                return Ok(None);
            };
            let repeated = mul_duration(duration, repeat_count)?;
            total = checked_add_duration(total, repeated, "queued prefill time")?;
        }
        Ok(Some(total))
    }

    /// Estimate scheduled decode iteration latency.
    ///
    /// Decode workers use only scheduled decode fields. Aggregated workers use
    /// scheduled decode plus current scheduled prefill; if the current FPM has
    /// no scheduled prefill, the helper uses the learned average scheduled
    /// prefill load from prior `tune_with_fpms` observations.
    pub fn get_scheduled_decode_itl(
        &self,
        metrics_by_rank: &[ForwardPassSnapshot],
    ) -> Result<Option<Duration>> {
        if self.worker_type == WorkerType::Prefill {
            return Ok(None);
        }
        self.validate_metrics_by_rank(metrics_by_rank)?;
        let metrics = metrics_by_rank
            .iter()
            .map(|snapshot| {
                let mut metrics = aic_identity_from_snapshot(snapshot);
                metrics.scheduled_requests.num_decode_requests = snapshot.num_decode_requests;
                metrics.scheduled_requests.sum_decode_kv_tokens =
                    u64_to_u32(snapshot.sum_decode_kv_tokens, "scheduled decode KV tokens")?;
                metrics.scheduled_requests.var_decode_kv_tokens = snapshot.var_decode_kv_tokens;
                if self.worker_type == WorkerType::Aggregated {
                    let (prefill_requests, prefill_tokens) = if snapshot.sum_prefill_tokens > 0
                        || snapshot.num_prefill_requests > 0
                    {
                        (
                            snapshot.num_prefill_requests,
                            u64_to_u32(snapshot.sum_prefill_tokens, "scheduled prefill tokens")?,
                        )
                    } else {
                        self.load_averages.typical_prefill_for_mixed_itl()?
                    };
                    metrics.scheduled_requests.num_prefill_requests = prefill_requests;
                    metrics.scheduled_requests.sum_prefill_tokens = prefill_tokens;
                    metrics.scheduled_requests.var_prefill_length = snapshot.var_prefill_length;
                    metrics.scheduled_requests.sum_prefill_kv_tokens = u64_to_u32(
                        snapshot.sum_prefill_kv_tokens,
                        "scheduled prefill KV tokens",
                    )?;
                }
                Ok(metrics)
            })
            .collect::<Result<Vec<_>>>()?;
        self.estimate_aic_metrics(&metrics)
    }

    /// Search for sustainable per-engine RPS under request shape and SLA constraints.
    ///
    /// Returns the best point found even when eligible SLA metrics fail; callers
    /// must inspect `EngineCapacity::eligible`.
    pub fn find_engine_capacity_rps(
        &self,
        request: EngineCapacityRequest,
    ) -> Result<Option<EngineCapacity>> {
        if request.isl == 0 {
            return Ok(None);
        }
        // Capacity search is intentionally bounded: full linear sweeps can be
        // caller-controlled through engine limits. TODO: replace coarse sampling
        // with a faster non-monotonic-safe search. We tried binary search, but
        // the perf model does not guarantee strictly increasing estimates,
        // which led to bad capacity choices.
        match self.worker_type {
            WorkerType::Prefill => self.find_prefill_capacity(&request),
            WorkerType::Decode => self.find_decode_capacity(&request),
            WorkerType::Aggregated => self.find_agg_capacity(&request),
        }
    }

    fn find_prefill_capacity(
        &self,
        request: &EngineCapacityRequest,
    ) -> Result<Option<EngineCapacity>> {
        let prefill_isl = effective_prefill_isl(request)?;
        let max_batch = self.prefill_max_batch(prefill_isl);
        if max_batch == 0 {
            return Ok(None);
        }

        let mut best = None;
        for batch_size in capacity_batch_sizes(max_batch) {
            let tokens = prefill_isl.saturating_mul(batch_size);
            let Some(ttft) = self.prefill_time_for_tokens(tokens)? else {
                return Ok(None);
            };
            if ttft.is_zero() {
                continue;
            }
            let rps = f64::from(batch_size) / ttft.as_secs_f64();
            let capacity = EngineCapacity {
                rps,
                ttft: Some(ttft),
                itl: None,
                e2e_latency: Some(ttft),
                eligible: sla_ok(Some(ttft), request.ttft_sla)
                    && sla_ok(None, request.itl_sla)
                    && sla_ok(Some(ttft), request.e2e_latency_sla),
            };
            best = select_capacity(best, capacity, request.optimization_target);
        }
        Ok(best)
    }

    fn find_decode_capacity(
        &self,
        request: &EngineCapacityRequest,
    ) -> Result<Option<EngineCapacity>> {
        if request.osl == 0 {
            return Ok(None);
        }
        let accept_length = normalized_accept_length(request.accept_length);
        let context_length = decode_context_length(request);
        let max_batch = self.decode_max_batch(context_length);
        if max_batch == 0 {
            return Ok(None);
        }
        let mut best = None;
        for batch_size in capacity_batch_sizes(max_batch) {
            let Some(forward_wall_time) = self.decode_time_for_batch(batch_size, context_length)?
            else {
                return Ok(None);
            };
            if forward_wall_time.is_zero() {
                continue;
            }
            let itl = div_duration_by_f64(forward_wall_time, accept_length, "decode ITL")?;
            if itl.is_zero() {
                continue;
            }
            let rps = f64::from(batch_size) / (f64::from(request.osl) * itl.as_secs_f64());
            let capacity = EngineCapacity {
                rps,
                ttft: None,
                itl: Some(itl),
                e2e_latency: None,
                eligible: sla_ok(None, request.ttft_sla)
                    && sla_ok(Some(itl), request.itl_sla)
                    && sla_ok(None, request.e2e_latency_sla),
            };
            best = select_capacity(best, capacity, request.optimization_target);
        }
        Ok(best)
    }

    fn find_agg_capacity(&self, request: &EngineCapacityRequest) -> Result<Option<EngineCapacity>> {
        if request.osl == 0 || self.limits.max_num_batched_tokens <= 1 {
            return Ok(None);
        }

        let prefill_isl = effective_prefill_isl(request)?;
        let accept_length = normalized_accept_length(request.accept_length);
        let context_length = decode_context_length(request);
        let kv_cap = self.decode_max_batch(context_length);
        let hard_cap = cmp::min(
            kv_cap,
            self.limits.max_num_batched_tokens.saturating_sub(1).max(1),
        );
        if hard_cap == 0 {
            return Ok(None);
        }

        let mut best = None;
        for batch_size in capacity_batch_sizes(hard_cap) {
            if !prefill_decode_balanced(
                prefill_isl,
                request.osl,
                batch_size,
                self.limits.max_num_batched_tokens,
                accept_length,
            )? {
                continue;
            }

            let decode_kv = batch_size.saturating_mul(context_length);
            let prefill_per_iter = cmp::min(
                self.limits.max_num_batched_tokens,
                aggregate_prefill_per_iter(prefill_isl, request.osl, batch_size, accept_length)?,
            );
            let Some(forward_wall_time) =
                self.mixed_time(prefill_per_iter, batch_size, decode_kv)?
            else {
                return Ok(None);
            };
            if forward_wall_time.is_zero() {
                continue;
            }
            let itl = div_duration_by_f64(forward_wall_time, accept_length, "aggregate ITL")?;
            if itl.is_zero() {
                continue;
            }

            let ttft_prefill_tokens = prefill_per_iter.saturating_add(prefill_isl);
            let Some(ttft) = self.agg_ttft(ttft_prefill_tokens, batch_size, decode_kv)? else {
                return Ok(None);
            };
            let decode_tail = mul_duration(itl, u64::from(request.osl.saturating_sub(1)))?;
            let e2e = checked_add_duration(ttft, decode_tail, "aggregate E2E latency")?;
            let rps = f64::from(batch_size) / (f64::from(request.osl) * itl.as_secs_f64());
            let eligible = sla_ok(Some(ttft), request.ttft_sla)
                && sla_ok(Some(itl), request.itl_sla)
                && sla_ok(Some(e2e), request.e2e_latency_sla);
            let capacity = EngineCapacity {
                rps,
                ttft: Some(ttft),
                itl: Some(itl),
                e2e_latency: Some(e2e),
                eligible,
            };
            best = select_capacity(best, capacity, request.optimization_target);
        }
        Ok(best)
    }

    fn prefill_time_for_tokens(&self, tokens: u32) -> Result<Option<Duration>> {
        self.prefill_chunk_time(u64::from(tokens), |chunk| {
            let metrics = synthetic_prefill_by_rank(chunk, self.attention_dp_size)?;
            self.estimate_aic_metrics(&metrics)
        })
    }

    fn decode_time_for_batch(
        &self,
        batch_size: u32,
        context_length: u32,
    ) -> Result<Option<Duration>> {
        let metrics = synthetic_decode_by_rank(
            batch_size,
            batch_size.saturating_mul(context_length),
            self.attention_dp_size,
        )?;
        self.estimate_aic_metrics(&metrics)
    }

    fn mixed_time(
        &self,
        prefill_tokens: u32,
        decode_requests: u32,
        decode_kv: u32,
    ) -> Result<Option<Duration>> {
        let metrics = synthetic_mixed_by_rank(
            prefill_tokens,
            decode_requests,
            decode_kv,
            self.attention_dp_size,
        )?;
        self.estimate_aic_metrics(&metrics)
    }

    fn agg_ttft(
        &self,
        queued_prefill_tokens: u32,
        current_decode_requests: u32,
        current_decode_kv: u32,
    ) -> Result<Option<Duration>> {
        self.prefill_chunk_time(u64::from(queued_prefill_tokens), |chunk| {
            let metrics = synthetic_mixed_by_rank(
                chunk,
                current_decode_requests,
                current_decode_kv,
                self.attention_dp_size,
            )?;
            self.estimate_aic_metrics(&metrics)
        })
    }

    fn prefill_chunk_time<F>(&self, tokens: u64, mut estimate_chunk: F) -> Result<Option<Duration>>
    where
        F: FnMut(u32) -> Result<Option<Duration>>,
    {
        let plan = PrefillChunkPlan::new(tokens, u64::from(self.limits.max_num_batched_tokens));
        let mut total = Duration::ZERO;
        if plan.full_chunks > 0 {
            let Some(duration) = estimate_chunk(self.limits.max_num_batched_tokens)? else {
                return Ok(None);
            };
            let repeated = mul_duration(duration, plan.full_chunks)?;
            total = checked_add_duration(total, repeated, "prefill chunk time")?;
        }
        if plan.tail_tokens > 0 {
            let tail = u64_to_u32(plan.tail_tokens, "prefill tail tokens")?;
            let Some(duration) = estimate_chunk(tail)? else {
                return Ok(None);
            };
            total = checked_add_duration(total, duration, "prefill chunk time")?;
        }
        Ok(Some(total))
    }

    fn prefill_max_batch(&self, isl: u32) -> u32 {
        if self.limits.max_num_seqs == 0 || self.limits.max_num_batched_tokens == 0 {
            return 0;
        }
        if isl > self.limits.max_num_batched_tokens {
            return 1;
        }
        cmp::min(
            self.limits.max_num_seqs,
            self.limits.max_num_batched_tokens / isl.max(1),
        )
    }

    fn decode_max_batch(&self, context_length: u32) -> u32 {
        if context_length == 0 {
            return 0;
        }
        cmp::min(
            self.limits.max_num_seqs,
            self.limits.max_kv_tokens / context_length,
        )
    }

    fn validate_metrics_by_rank(&self, metrics_by_rank: &[ForwardPassSnapshot]) -> Result<()> {
        ensure!(
            !metrics_by_rank.is_empty(),
            "at least one attention-DP rank metric is required"
        );
        ensure!(
            metrics_by_rank.len() == self.attention_dp_size,
            "expected {} attention-DP rank metrics, got {}",
            self.attention_dp_size,
            metrics_by_rank.len()
        );
        if self.attention_dp_size > 1 {
            let mut seen = vec![false; self.attention_dp_size];
            for snapshot in metrics_by_rank {
                validate_fpm_version(snapshot)?;
                let rank = snapshot.dp_rank as usize;
                ensure!(
                    rank < self.attention_dp_size,
                    "dp_rank {} out of range for attention_dp_size {}",
                    snapshot.dp_rank,
                    self.attention_dp_size
                );
                ensure!(!seen[rank], "duplicate dp_rank {}", snapshot.dp_rank);
                seen[rank] = true;
            }
        } else {
            validate_fpm_version(&metrics_by_rank[0])?;
        }
        Ok(())
    }

    fn estimate_aic_metrics(&self, metrics: &[ForwardPassMetrics]) -> Result<Option<Duration>> {
        let ms = self
            .model
            .estimate_forward_pass_time_ms(metrics)
            .context("failed to estimate AIC forward-pass time")?;
        ms.map(|value| {
            ensure!(
                value.is_finite(),
                "AIC forward-pass estimate must be finite, got {value}"
            );
            Duration::try_from_secs_f64(value.max(0.0) / 1000.0)
                .with_context(|| format!("invalid AIC forward-pass estimate {value} ms"))
        })
        .transpose()
    }
}

pub fn snapshots_to_aic_metrics(
    metrics_by_rank: &[ForwardPassSnapshot],
) -> Result<Vec<ForwardPassMetrics>> {
    ensure!(
        !metrics_by_rank.is_empty(),
        "at least one attention-DP rank metric is required"
    );
    metrics_by_rank
        .iter()
        .map(snapshot_to_aic_metrics)
        .collect()
}

pub fn snapshot_to_aic_metrics(snapshot: &ForwardPassSnapshot) -> Result<ForwardPassMetrics> {
    validate_fpm_version(snapshot)?;
    let mut metrics = aic_identity_from_snapshot(snapshot);
    metrics.wall_time = snapshot.wall_time_secs;
    metrics.scheduled_requests = ScheduledRequestMetrics {
        num_prefill_requests: snapshot.num_prefill_requests,
        sum_prefill_tokens: u64_to_u32(snapshot.sum_prefill_tokens, "scheduled prefill tokens")?,
        var_prefill_length: snapshot.var_prefill_length,
        sum_prefill_kv_tokens: u64_to_u32(
            snapshot.sum_prefill_kv_tokens,
            "scheduled prefill KV tokens",
        )?,
        num_decode_requests: snapshot.num_decode_requests,
        sum_decode_kv_tokens: u64_to_u32(
            snapshot.sum_decode_kv_tokens,
            "scheduled decode KV tokens",
        )?,
        var_decode_kv_tokens: snapshot.var_decode_kv_tokens,
    };
    metrics.queued_requests = QueuedRequestMetrics {
        num_prefill_requests: snapshot.num_queued_prefill,
        sum_prefill_tokens: u64_to_u32(
            snapshot.sum_queued_prefill_tokens,
            "queued prefill tokens",
        )?,
        var_prefill_length: snapshot.var_queued_prefill_length,
        num_decode_requests: snapshot.num_queued_decode,
        sum_decode_kv_tokens: u64_to_u32(
            snapshot.sum_queued_decode_kv_tokens,
            "queued decode KV tokens",
        )?,
        var_decode_kv_tokens: snapshot.var_queued_decode_kv_tokens,
    };
    Ok(metrics)
}

fn validate_fpm_version(snapshot: &ForwardPassSnapshot) -> Result<()> {
    ensure!(
        snapshot.version == 0 || snapshot.version == FPM_VERSION,
        "unsupported FPM version {}; expected {}",
        snapshot.version,
        FPM_VERSION
    );
    Ok(())
}

fn aic_identity_from_snapshot(snapshot: &ForwardPassSnapshot) -> ForwardPassMetrics {
    ForwardPassMetrics {
        version: if snapshot.version == 0 {
            FPM_VERSION
        } else {
            snapshot.version
        },
        worker_id: snapshot.worker_id.clone(),
        dp_rank: snapshot.dp_rank,
        counter_id: snapshot.counter_id,
        wall_time: snapshot.wall_time_secs,
        scheduled_requests: ScheduledRequestMetrics::default(),
        queued_requests: QueuedRequestMetrics::default(),
    }
}

/// Derive native AIC config fields from serialized mocker engine args.
///
/// `aic_backend` opts into native AIC modeling, and `aic_model_path` is required
/// in this side API because no loaded `LocalModel` is available as a fallback.
pub fn aic_config_from_mock_engine_args(args: &MockEngineArgs) -> Result<Option<EngineConfig>> {
    let Some(backend) = args.aic_backend.as_deref() else {
        return Ok(None);
    };
    let Some(model_name) = args.aic_model_path.clone() else {
        bail!("aic_model_path is required when aic_backend is set");
    };
    let mut extra = BTreeMap::new();
    if let Some(nextn) = args.aic_nextn {
        extra.insert(AIC_NEXTN_KEY.to_string(), nextn.to_string());
    }
    extra.insert(
        AIC_NEXTN_ACCEPT_RATES_KEY.to_string(),
        RAW_AIC_NEXTN_ACCEPT_RATES.to_string(),
    );
    Ok(Some(aic_config_for_raw_iteration_time(EngineConfig {
        schema_version: ENGINE_CONFIG_SCHEMA_VERSION,
        model_name,
        system_name: args
            .aic_system
            .clone()
            .unwrap_or_else(|| DEFAULT_AIC_SYSTEM.to_string()),
        systems_path: args.aic_systems_path.clone().map(PathBuf::from),
        backend: parse_backend_kind(backend)?,
        backend_version: args.aic_backend_version.clone(),
        kv_block_size: Some(to_u32(args.block_size, "block_size")?),
        parallel: ParallelMapping {
            tp_size: to_u32(args.aic_tp_size.unwrap_or(1), "aic_tp_size")?,
            pp_size: 1,
            attention_dp_size: args
                .aic_attention_dp_size
                .map(|value| to_u32(value, "aic_attention_dp_size"))
                .transpose()?,
            moe_tp_size: args
                .aic_moe_tp_size
                .map(|value| to_u32(value, "aic_moe_tp_size"))
                .transpose()?,
            moe_ep_size: args
                .aic_moe_ep_size
                .map(|value| to_u32(value, "aic_moe_ep_size"))
                .transpose()?,
        },
        // Native analytics path: quant dtypes are parsed by `parse_data_type`
        // into `aiconfigurator_core::DataType` (accepted: bfloat16, float16,
        // fp8, fp8_static, fp8_block, nvfp4, int8, int4, w4afp8, w4a16_mxfp4,
        // w4a8_mxfp4_mxfp8) -- a separate vocabulary from the Python callback
        // path's per-field AIC quant-mode names. This path also does not model
        // communication dtype (`QuantizationConfig` has no comm field), so
        // `aic_comm_dtype` is intentionally not applied here.
        quantization: QuantizationConfig {
            weight_dtype: args
                .aic_gemm_dtype
                .as_deref()
                .map(parse_data_type)
                .transpose()?,
            moe_dtype: args
                .aic_moe_dtype
                .as_deref()
                .map(parse_data_type)
                .transpose()?,
            activation_dtype: args
                .aic_fmha_dtype
                .as_deref()
                .map(parse_data_type)
                .transpose()?,
            kv_cache_dtype: args
                .aic_kv_cache_dtype
                .as_deref()
                .map(parse_data_type)
                .transpose()?,
        },
        speculative: None,
        extra,
    })))
}

fn resolve_worker_type(
    worker_type: Option<WorkerType>,
    engine_args: Option<&MockEngineArgs>,
) -> Result<WorkerType> {
    worker_type
        .or_else(|| engine_args.map(|args| args.worker_type))
        .ok_or_else(|| anyhow!("worker_type is required when engine_args is not provided"))
}

fn resolve_limits(
    limits: Option<EnginePerfLimits>,
    engine_args: Option<&MockEngineArgs>,
) -> Result<EnginePerfLimits> {
    let limits = match (limits, engine_args) {
        (Some(limits), _) => Ok(limits),
        (None, Some(args)) => EnginePerfLimits::from_mock_engine_args(args),
        (None, None) => Err(anyhow!(
            "limits are required when engine_args is not provided"
        )),
    }?;
    limits.validate().context("invalid engine perf limits")?;
    Ok(limits)
}

fn resolve_options(
    options: Option<ForwardPassPerfOptions>,
    limits: &EnginePerfLimits,
) -> ForwardPassPerfOptions {
    match options {
        Some(options) => options,
        None => ForwardPassPerfOptions {
            max_num_tokens: limits.max_num_batched_tokens,
            max_batch_size: limits.max_num_seqs,
            max_kv_tokens: limits.max_kv_tokens,
            ..ForwardPassPerfOptions::default()
        },
    }
}

fn parse_backend_kind(value: &str) -> Result<BackendKind> {
    match value {
        "trtllm" => Ok(BackendKind::Trtllm),
        "sglang" => Ok(BackendKind::Sglang),
        "vllm" => Ok(BackendKind::Vllm),
        other => bail!("invalid AIC backend {other:?}; expected trtllm, sglang, or vllm"),
    }
}

fn parse_data_type(value: &str) -> Result<DataType> {
    match value {
        "bfloat16" => Ok(DataType::Bfloat16),
        "float16" => Ok(DataType::Float16),
        "fp8" => Ok(DataType::Fp8),
        "fp8_static" => Ok(DataType::Fp8Static),
        "fp8_block" => Ok(DataType::Fp8Block),
        "nvfp4" => Ok(DataType::Nvfp4),
        "int8" => Ok(DataType::Int8),
        "int4" => Ok(DataType::Int4),
        "w4afp8" => Ok(DataType::W4afp8),
        "w4a16_mxfp4" => Ok(DataType::W4a16Mxfp4),
        "w4a8_mxfp4_mxfp8" => Ok(DataType::W4a8Mxfp4Mxfp8),
        other => bail!("invalid AIC dtype {other:?}"),
    }
}

fn empty_aic_metrics() -> ForwardPassMetrics {
    ForwardPassMetrics::default()
}

fn estimate_prefill_request_count(num_requests: u32, chunk_tokens: u64) -> u32 {
    if chunk_tokens == 0 {
        0
    } else if num_requests == 0 {
        1
    } else {
        num_requests
    }
}

fn synthetic_prefill_by_rank(tokens: u32, ranks: usize) -> Result<Vec<ForwardPassMetrics>> {
    let ranks = ranks.max(1);
    (0..ranks)
        .map(|rank| {
            let rank_tokens = split_total(tokens, ranks, rank);
            let mut metrics = empty_aic_metrics();
            metrics.dp_rank = rank as u32;
            metrics.scheduled_requests.sum_prefill_tokens = rank_tokens;
            metrics.scheduled_requests.num_prefill_requests = u32::from(rank_tokens > 0);
            Ok(metrics)
        })
        .collect()
}

fn synthetic_decode_by_rank(
    num_requests: u32,
    kv_tokens: u32,
    ranks: usize,
) -> Result<Vec<ForwardPassMetrics>> {
    let ranks = ranks.max(1);
    (0..ranks)
        .map(|rank| {
            let rank_requests = split_total(num_requests, ranks, rank);
            let rank_kv = split_total(kv_tokens, ranks, rank);
            let mut metrics = empty_aic_metrics();
            metrics.dp_rank = rank as u32;
            metrics.scheduled_requests.num_decode_requests = rank_requests;
            metrics.scheduled_requests.sum_decode_kv_tokens = rank_kv;
            Ok(metrics)
        })
        .collect()
}

fn synthetic_mixed_by_rank(
    prefill_tokens: u32,
    decode_requests: u32,
    decode_kv_tokens: u32,
    ranks: usize,
) -> Result<Vec<ForwardPassMetrics>> {
    let ranks = ranks.max(1);
    (0..ranks)
        .map(|rank| {
            let rank_prefill = split_total(prefill_tokens, ranks, rank);
            let rank_decode_requests = split_total(decode_requests, ranks, rank);
            let rank_decode_kv = split_total(decode_kv_tokens, ranks, rank);
            let mut metrics = empty_aic_metrics();
            metrics.dp_rank = rank as u32;
            metrics.scheduled_requests.sum_prefill_tokens = rank_prefill;
            metrics.scheduled_requests.num_prefill_requests = u32::from(rank_prefill > 0);
            metrics.scheduled_requests.sum_decode_kv_tokens = rank_decode_kv;
            metrics.scheduled_requests.num_decode_requests = rank_decode_requests;
            Ok(metrics)
        })
        .collect()
}

fn split_total(total: u32, ranks: usize, rank: usize) -> u32 {
    let ranks_u32 = ranks as u32;
    let base = total / ranks_u32;
    let remainder = total % ranks_u32;
    base + u32::from((rank as u32) < remainder)
}

fn decode_context_length(request: &EngineCapacityRequest) -> u32 {
    request.isl.saturating_add(request.osl / 2).max(1)
}

fn effective_prefill_isl(request: &EngineCapacityRequest) -> Result<u32> {
    let scale = 1.0 - clamp_kv_hit_rate(request.kv_hit_rate);
    let tokens = ceil_positive_f64_to_u32(
        f64::from(request.isl) * scale,
        "effective prefill ISL after kv_hit_rate discount",
    )?;
    Ok(tokens.max(1))
}

fn default_accept_length() -> f64 {
    1.0
}

fn normalized_accept_length(value: f64) -> f64 {
    if value.is_finite() && value > 1.0 {
        value
    } else {
        1.0
    }
}

fn clamp_kv_hit_rate(kv_hit_rate: Option<f64>) -> f64 {
    let Some(value) = kv_hit_rate else {
        return 0.0;
    };
    if !value.is_finite() {
        return 0.0;
    }
    value.clamp(0.0, MAX_KV_HIT_RATE_DISCOUNT)
}

fn prefill_decode_balanced(
    isl: u32,
    osl: u32,
    batch_size: u32,
    max_num_batched_tokens: u32,
    accept_length: f64,
) -> Result<bool> {
    let prefill_budget = max_num_batched_tokens.saturating_sub(batch_size);
    if prefill_budget == 0 {
        return Ok(false);
    }
    Ok(aggregate_prefill_per_iter(isl, osl, batch_size, accept_length)? <= prefill_budget)
}

fn aggregate_prefill_per_iter(
    isl: u32,
    osl: u32,
    batch_size: u32,
    accept_length: f64,
) -> Result<u32> {
    ceil_positive_f64_to_u32(
        f64::from(batch_size) * normalized_accept_length(accept_length) * f64::from(isl)
            / f64::from(osl.max(1)),
        "aggregate prefill per iteration",
    )
}

fn capacity_batch_sizes(max_batch: u32) -> Vec<u32> {
    if max_batch == 0 {
        return Vec::new();
    }
    if max_batch <= MAX_CAPACITY_SEARCH_CANDIDATES {
        return (1..=max_batch).collect();
    }

    let span = u64::from(max_batch - 1);
    let denominator = u64::from(MAX_CAPACITY_SEARCH_CANDIDATES - 1);
    (0..MAX_CAPACITY_SEARCH_CANDIDATES)
        .map(|index| 1 + ((u64::from(index) * span) / denominator) as u32)
        .collect()
}

fn sla_ok(value: Option<Duration>, sla: Option<Duration>) -> bool {
    match (value, sla) {
        (_, None) => true,
        (Some(value), Some(sla)) => value <= sla,
        (None, Some(_)) => false,
    }
}

fn select_capacity(
    current: Option<EngineCapacity>,
    candidate: EngineCapacity,
    target: OptimizationTarget,
) -> Option<EngineCapacity> {
    let Some(current) = current else {
        return Some(candidate);
    };
    match (current.eligible, candidate.eligible) {
        (false, true) => Some(candidate),
        (true, false) => Some(current),
        _ => match target {
            OptimizationTarget::Throughput => {
                if candidate.rps > current.rps {
                    Some(candidate)
                } else {
                    Some(current)
                }
            }
            OptimizationTarget::Latency => {
                let current_latency = current
                    .e2e_latency
                    .or(current.ttft)
                    .or(current.itl)
                    .unwrap_or(Duration::MAX);
                let candidate_latency = candidate
                    .e2e_latency
                    .or(candidate.ttft)
                    .or(candidate.itl)
                    .unwrap_or(Duration::MAX);
                if candidate_latency < current_latency {
                    Some(candidate)
                } else {
                    Some(current)
                }
            }
        },
    }
}

fn checked_add_duration(lhs: Duration, rhs: Duration, context: &str) -> Result<Duration> {
    lhs.checked_add(rhs)
        .ok_or_else(|| anyhow!("{context} overflow"))
}

fn div_duration_by_f64(duration: Duration, divisor: f64, context: &str) -> Result<Duration> {
    ensure!(
        divisor.is_finite() && divisor > 0.0,
        "{context} divisor must be a positive finite value, got {divisor}"
    );
    Duration::try_from_secs_f64(duration.as_secs_f64() / divisor)
        .with_context(|| format!("{context} overflow dividing {duration:?} by {divisor}"))
}

fn mul_duration(duration: Duration, factor: u64) -> Result<Duration> {
    let seconds = u128::from(duration.as_secs())
        .checked_mul(u128::from(factor))
        .ok_or_else(|| anyhow!("duration overflow multiplying {duration:?} by {factor}"))?;
    let nanos = u128::from(duration.subsec_nanos())
        .checked_mul(u128::from(factor))
        .ok_or_else(|| anyhow!("duration overflow multiplying {duration:?} by {factor}"))?;
    let total_seconds = seconds
        .checked_add(nanos / 1_000_000_000)
        .ok_or_else(|| anyhow!("duration overflow multiplying {duration:?} by {factor}"))?;
    ensure!(
        total_seconds <= u128::from(u64::MAX),
        "duration overflow multiplying {duration:?} by {factor}"
    );
    Ok(Duration::new(
        total_seconds as u64,
        (nanos % 1_000_000_000) as u32,
    ))
}

fn to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{name} exceeds u32::MAX"))
}

fn u64_to_u32(value: u64, name: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{name} exceeds u32::MAX"))
}

fn ceil_positive_f64_to_u32(value: f64, name: &str) -> Result<u32> {
    ensure!(value.is_finite(), "{name} must be finite");
    if value <= 0.0 {
        return Ok(0);
    }
    let value = value.ceil();
    ensure!(value <= f64::from(u32::MAX), "{name} exceeds u32::MAX");
    Ok(value as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> EnginePerfLimits {
        EnginePerfLimits {
            max_num_batched_tokens: 8192,
            max_num_seqs: 512,
            max_kv_tokens: 2_000_000,
        }
    }

    fn fast_options() -> ForwardPassPerfOptions {
        ForwardPassPerfOptions {
            min_observations: 2,
            max_observations: 16,
            ..ForwardPassPerfOptions::default()
        }
    }

    fn prefill_observation(tokens: u64, wall_time_secs: f64) -> ForwardPassSnapshot {
        ForwardPassSnapshot {
            num_prefill_requests: 1,
            sum_prefill_tokens: tokens,
            wall_time_secs,
            ..Default::default()
        }
    }

    fn decode_observation(
        num_requests: u32,
        kv_tokens: u64,
        wall_time_secs: f64,
    ) -> ForwardPassSnapshot {
        ForwardPassSnapshot {
            num_decode_requests: num_requests,
            sum_decode_kv_tokens: kv_tokens,
            wall_time_secs,
            ..Default::default()
        }
    }

    fn mixed_observation(
        prefill_tokens: u64,
        num_decode_requests: u32,
        decode_kv_tokens: u64,
        wall_time_secs: f64,
    ) -> ForwardPassSnapshot {
        ForwardPassSnapshot {
            num_prefill_requests: u32::from(prefill_tokens > 0),
            sum_prefill_tokens: prefill_tokens,
            num_decode_requests,
            sum_decode_kv_tokens: decode_kv_tokens,
            wall_time_secs,
            ..Default::default()
        }
    }

    fn with_rank(mut snapshot: ForwardPassSnapshot, rank: u32) -> ForwardPassSnapshot {
        snapshot.dp_rank = rank;
        snapshot
    }

    #[test]
    fn snapshot_conversion_preserves_fields() {
        let snapshot = ForwardPassSnapshot {
            version: FPM_VERSION,
            worker_id: "worker-a".to_string(),
            dp_rank: 3,
            counter_id: 99,
            num_prefill_requests: 1,
            sum_prefill_tokens: 10,
            var_prefill_length: 2.0,
            sum_prefill_kv_tokens: 3,
            num_decode_requests: 4,
            sum_decode_kv_tokens: 50,
            var_decode_kv_tokens: 6.0,
            num_queued_prefill: 7,
            sum_queued_prefill_tokens: 80,
            var_queued_prefill_length: 9.0,
            num_queued_decode: 10,
            sum_queued_decode_kv_tokens: 11,
            var_queued_decode_kv_tokens: 12.0,
            wall_time_secs: 0.013,
        };
        let converted = snapshot_to_aic_metrics(&snapshot).unwrap();
        assert_eq!(converted.version, FPM_VERSION);
        assert_eq!(converted.worker_id, "worker-a");
        assert_eq!(converted.dp_rank, 3);
        assert_eq!(converted.counter_id, 99);
        assert_eq!(converted.scheduled_requests.sum_prefill_tokens, 10);
        assert_eq!(converted.scheduled_requests.sum_prefill_kv_tokens, 3);
        assert_eq!(converted.scheduled_requests.num_decode_requests, 4);
        assert_eq!(converted.queued_requests.num_prefill_requests, 7);
        assert_eq!(converted.queued_requests.sum_decode_kv_tokens, 11);
        assert_eq!(converted.wall_time, 0.013);
    }

    #[test]
    fn snapshot_conversion_rejects_unsupported_fpm_version() {
        let snapshot = ForwardPassSnapshot {
            version: FPM_VERSION + 1,
            ..Default::default()
        };
        let err = snapshot_to_aic_metrics(&snapshot).unwrap_err();
        assert!(err.to_string().contains("unsupported FPM version"));
    }

    #[test]
    fn explicit_aic_config_preserves_extra() {
        let mut extra = BTreeMap::new();
        extra.insert(
            "moe_workload_distribution".to_string(),
            "balanced".to_string(),
        );
        let config = AicEngineConfig {
            model_name: "model".to_string(),
            model_arch: Some("arch".to_string()),
            system_name: "h200_sxm".to_string(),
            systems_path: Some("/tmp/aic-systems".to_string()),
            backend: "vllm".to_string(),
            backend_version: None,
            tp_size: 1,
            pp_size: 1,
            moe_tp_size: None,
            moe_ep_size: None,
            attention_dp_size: Some(1),
            weight_dtype: None,
            moe_dtype: None,
            activation_dtype: None,
            kv_cache_dtype: None,
            kv_block_size: None,
            extra: extra.clone(),
        };
        let converted = config.into_aic_config().unwrap();
        assert_eq!(converted.extra, extra);
        assert_eq!(
            converted.systems_path.as_deref(),
            Some(std::path::Path::new("/tmp/aic-systems"))
        );
    }

    #[test]
    fn raw_iteration_time_aic_config_forces_zero_accept_rates() {
        let mut extra = BTreeMap::new();
        extra.insert(AIC_NEXTN_KEY.to_string(), "3".to_string());
        extra.insert(
            AIC_NEXTN_ACCEPT_RATES_KEY.to_string(),
            "0.85,0.3,0,0,0".to_string(),
        );
        let config = AicEngineConfig {
            model_name: "model".to_string(),
            model_arch: Some("arch".to_string()),
            system_name: "h200_sxm".to_string(),
            systems_path: None,
            backend: "vllm".to_string(),
            backend_version: None,
            tp_size: 1,
            pp_size: 1,
            moe_tp_size: None,
            moe_ep_size: None,
            attention_dp_size: Some(1),
            weight_dtype: None,
            moe_dtype: None,
            activation_dtype: None,
            kv_cache_dtype: None,
            kv_block_size: None,
            extra,
        };

        let config = aic_engine_config_for_raw_iteration_time(config).unwrap();

        assert_eq!(config.extra.get(AIC_NEXTN_KEY), Some(&"3".to_string()));
        assert_eq!(
            config.extra.get(AIC_NEXTN_ACCEPT_RATES_KEY),
            Some(&RAW_AIC_NEXTN_ACCEPT_RATES.to_string())
        );
    }

    #[test]
    fn mock_engine_args_aic_config_forces_zero_accept_rates() {
        let args = MockEngineArgs::builder()
            .aic_backend(Some("vllm".to_string()))
            .aic_model_path(Some("model".to_string()))
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("0.85,0.3,0,0,0".to_string()))
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        let config = aic_config_from_mock_engine_args(&args).unwrap().unwrap();

        assert_eq!(config.extra.get(AIC_NEXTN_KEY), Some(&"2".to_string()));
        assert_eq!(
            config.extra.get(AIC_NEXTN_ACCEPT_RATES_KEY),
            Some(&RAW_AIC_NEXTN_ACCEPT_RATES.to_string())
        );
    }

    #[test]
    fn mock_engine_args_json_forwards_aic_quantization() {
        let args = MockEngineArgs::from_json_str(
            &serde_json::json!({
                "aic_backend": "vllm",
                "aic_model_path": "model",
                "aic_gemm_dtype": "fp8_block",
                "aic_moe_dtype": "w4a16_mxfp4",
                "aic_fmha_dtype": "bfloat16",
                "aic_kv_cache_dtype": "fp8",
            })
            .to_string(),
        )
        .unwrap();

        let config = aic_config_from_mock_engine_args(&args).unwrap().unwrap();

        assert_eq!(config.quantization.weight_dtype, Some(DataType::Fp8Block));
        assert_eq!(config.quantization.moe_dtype, Some(DataType::W4a16Mxfp4));
        assert_eq!(
            config.quantization.activation_dtype,
            Some(DataType::Bfloat16)
        );
        assert_eq!(config.quantization.kv_cache_dtype, Some(DataType::Fp8));
    }

    #[test]
    fn snapshot_conversion_rejects_u32_overflow() {
        let snapshot = ForwardPassSnapshot {
            sum_prefill_tokens: u64::from(u32::MAX) + 1,
            ..Default::default()
        };
        let err = snapshot_to_aic_metrics(&snapshot).unwrap_err();
        assert!(err.to_string().contains("scheduled prefill tokens"));
    }

    #[test]
    fn engine_perf_limits_reject_zero_values() {
        for limits in [
            EnginePerfLimits {
                max_num_batched_tokens: 0,
                max_num_seqs: 1,
                max_kv_tokens: 1,
            },
            EnginePerfLimits {
                max_num_batched_tokens: 1,
                max_num_seqs: 0,
                max_kv_tokens: 1,
            },
            EnginePerfLimits {
                max_num_batched_tokens: 1,
                max_num_seqs: 1,
                max_kv_tokens: 0,
            },
        ] {
            let err =
                EnginePerfModel::from_regression(WorkerType::Decode, limits, None).unwrap_err();
            assert!(err.to_string().contains("invalid engine perf limits"));
        }
    }

    #[test]
    fn capacity_batch_sizes_are_bounded_and_include_endpoints() {
        assert_eq!(capacity_batch_sizes(0), Vec::<u32>::new());
        assert_eq!(capacity_batch_sizes(3), vec![1, 2, 3]);

        let exact = capacity_batch_sizes(MAX_CAPACITY_SEARCH_CANDIDATES);
        assert_eq!(exact.len(), MAX_CAPACITY_SEARCH_CANDIDATES as usize);
        assert_eq!(exact[0], 1);
        assert_eq!(exact[exact.len() - 1], MAX_CAPACITY_SEARCH_CANDIDATES);

        let sampled = capacity_batch_sizes(MAX_CAPACITY_SEARCH_CANDIDATES + 1);
        assert_eq!(sampled.len(), MAX_CAPACITY_SEARCH_CANDIDATES as usize);
        assert_eq!(sampled[0], 1);
        assert_eq!(
            sampled[sampled.len() - 1],
            MAX_CAPACITY_SEARCH_CANDIDATES + 1
        );
        assert!(sampled.windows(2).all(|window| window[0] < window[1]));
    }

    #[test]
    fn prefill_chunk_plan_uses_full_chunks_plus_tail() {
        let plan = PrefillChunkPlan::new(250, 100);
        assert_eq!(plan.full_chunks, 2);
        assert_eq!(plan.tail_tokens, 50);
        assert_eq!(plan.chunk_at(0, 100), 100);
        assert_eq!(plan.chunk_at(1, 100), 100);
        assert_eq!(plan.chunk_at(2, 100), 50);
        assert_eq!(plan.chunk_at(3, 100), 0);
    }

    #[test]
    fn prefill_time_uses_compressed_full_chunks_and_tail() {
        let small_limits = EnginePerfLimits {
            max_num_batched_tokens: 100,
            max_num_seqs: 4,
            max_kv_tokens: 1_000_000,
        };
        let mut model = EnginePerfModel::from_regression(
            WorkerType::Prefill,
            small_limits,
            Some(fast_options()),
        )
        .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![prefill_observation(50, 0.005)],
                vec![prefill_observation(100, 0.010)],
            ])
            .unwrap();

        let duration = model.prefill_time_for_tokens(250).unwrap().unwrap();
        assert!((duration.as_secs_f64() - 0.025).abs() < 1e-9);
    }

    #[test]
    fn regression_model_returns_none_until_bootstrapped() {
        let model = EnginePerfModel::from_regression(WorkerType::Decode, limits(), None).unwrap();
        let snapshot = ForwardPassSnapshot {
            num_decode_requests: 1,
            sum_decode_kv_tokens: 128,
            ..Default::default()
        };
        assert!(
            model
                .get_scheduled_decode_itl(&[snapshot])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn unsupported_worker_helpers_return_none() {
        let prefill =
            EnginePerfModel::from_regression(WorkerType::Prefill, limits(), None).unwrap();
        let decode = EnginePerfModel::from_regression(WorkerType::Decode, limits(), None).unwrap();
        assert!(
            prefill
                .get_scheduled_decode_itl(&[ForwardPassSnapshot::default()])
                .unwrap()
                .is_none()
        );
        assert!(
            decode
                .get_queued_prefill_time(&[ForwardPassSnapshot::default()])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn synthetic_work_is_split_across_attention_dp_ranks() {
        let metrics = synthetic_decode_by_rank(5, 101, 2).unwrap();
        assert_eq!(metrics.len(), 2);
        assert_eq!(
            metrics
                .iter()
                .map(|m| m.scheduled_requests.num_decode_requests)
                .sum::<u32>(),
            5
        );
        assert_eq!(
            metrics
                .iter()
                .map(|m| m.scheduled_requests.sum_decode_kv_tokens)
                .sum::<u32>(),
            101
        );

        let mixed = synthetic_mixed_by_rank(100, 5, 101, 2).unwrap();
        assert_eq!(
            mixed
                .iter()
                .map(|m| m.scheduled_requests.num_decode_requests)
                .sum::<u32>(),
            5
        );
        assert_eq!(
            mixed
                .iter()
                .map(|m| m.scheduled_requests.sum_decode_kv_tokens)
                .sum::<u32>(),
            101
        );
    }

    #[test]
    fn queued_prefill_helper_ignores_scheduled_work() {
        let mut model =
            EnginePerfModel::from_regression(WorkerType::Prefill, limits(), Some(fast_options()))
                .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![prefill_observation(100, 0.010)],
                vec![prefill_observation(200, 0.020)],
            ])
            .unwrap();

        let base = ForwardPassSnapshot {
            num_queued_prefill: 1,
            sum_queued_prefill_tokens: 100,
            ..Default::default()
        };
        let noisy = ForwardPassSnapshot {
            num_queued_prefill: 1,
            sum_queued_prefill_tokens: 100,
            num_prefill_requests: 99,
            sum_prefill_tokens: 999_999,
            num_decode_requests: 99,
            sum_decode_kv_tokens: 999_999,
            ..Default::default()
        };
        let left = model.get_queued_prefill_time(&[base]).unwrap().unwrap();
        let right = model.get_queued_prefill_time(&[noisy]).unwrap().unwrap();
        assert!((left.as_secs_f64() - right.as_secs_f64()).abs() < 1e-9);
    }

    #[test]
    fn queued_prefill_helper_handles_large_aggregate_chunk_count() {
        let small_limits = EnginePerfLimits {
            max_num_batched_tokens: 1,
            max_num_seqs: 4,
            max_kv_tokens: 1_000_000,
        };
        let mut model = EnginePerfModel::from_regression(
            WorkerType::Prefill,
            small_limits,
            Some(fast_options()),
        )
        .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![prefill_observation(1, 0.001)],
                vec![prefill_observation(2, 0.002)],
            ])
            .unwrap();

        let snapshot = ForwardPassSnapshot {
            num_queued_prefill: 1,
            sum_queued_prefill_tokens: 1_000,
            ..Default::default()
        };
        let duration = model.get_queued_prefill_time(&[snapshot]).unwrap().unwrap();
        assert!((duration.as_secs_f64() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn scheduled_decode_helper_ignores_queued_decode() {
        let mut model =
            EnginePerfModel::from_regression(WorkerType::Decode, limits(), Some(fast_options()))
                .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![decode_observation(1, 100, 0.010)],
                vec![decode_observation(2, 200, 0.020)],
            ])
            .unwrap();

        let base = ForwardPassSnapshot {
            num_decode_requests: 1,
            sum_decode_kv_tokens: 100,
            ..Default::default()
        };
        let queued = ForwardPassSnapshot {
            num_decode_requests: 1,
            sum_decode_kv_tokens: 100,
            num_queued_decode: 64,
            sum_queued_decode_kv_tokens: 1_000_000,
            ..Default::default()
        };
        let left = model.get_scheduled_decode_itl(&[base]).unwrap().unwrap();
        let right = model.get_scheduled_decode_itl(&[queued]).unwrap().unwrap();
        assert!((left.as_secs_f64() - right.as_secs_f64()).abs() < 1e-9);
    }

    #[test]
    fn tune_with_fpms_accepts_multiple_attention_dp_ranks() {
        let mut args = MockEngineArgs::default();
        args.worker_type = WorkerType::Decode;
        args.aic_attention_dp_size = Some(2);
        let mut model = EnginePerfModel::best_available(EnginePerfModelInputs {
            engine_args: Some(args),
            options: Some(fast_options()),
            ..Default::default()
        })
        .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![
                    with_rank(decode_observation(1, 100, 0.010), 0),
                    with_rank(decode_observation(2, 200, 0.020), 1),
                ],
                vec![
                    with_rank(decode_observation(1, 150, 0.015), 0),
                    with_rank(decode_observation(3, 300, 0.030), 1),
                ],
            ])
            .unwrap();
        let prediction = model
            .get_scheduled_decode_itl(&[
                with_rank(decode_observation(1, 100, 0.0), 0),
                with_rank(decode_observation(2, 200, 0.0), 1),
            ])
            .unwrap();
        assert!(prediction.is_some());
    }

    #[test]
    fn attention_dp_rank_validation_rejects_duplicate_ranks() {
        let mut args = MockEngineArgs::default();
        args.worker_type = WorkerType::Decode;
        args.aic_attention_dp_size = Some(2);
        let model = EnginePerfModel::best_available(EnginePerfModelInputs {
            engine_args: Some(args),
            options: Some(fast_options()),
            ..Default::default()
        })
        .unwrap();
        let err = model
            .estimate_forward_pass_time(&[
                with_rank(decode_observation(1, 100, 0.0), 0),
                with_rank(decode_observation(1, 100, 0.0), 0),
            ])
            .unwrap_err();
        assert!(err.to_string().contains("duplicate dp_rank"));
    }

    #[test]
    fn aggregated_helpers_keep_mixed_workload_shape() {
        let mut model = EnginePerfModel::from_regression(
            WorkerType::Aggregated,
            limits(),
            Some(fast_options()),
        )
        .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![mixed_observation(100, 1, 100, 0.020)],
                vec![mixed_observation(100, 2, 200, 0.040)],
                vec![mixed_observation(200, 1, 100, 0.030)],
            ])
            .unwrap();

        let queued_prefill_with_decode = ForwardPassSnapshot {
            num_queued_prefill: 1,
            sum_queued_prefill_tokens: 100,
            num_decode_requests: 2,
            sum_decode_kv_tokens: 200,
            ..Default::default()
        };
        assert!(
            model
                .get_queued_prefill_time(&[queued_prefill_with_decode])
                .unwrap()
                .is_some()
        );

        let decode_only_input = ForwardPassSnapshot {
            num_decode_requests: 1,
            sum_decode_kv_tokens: 100,
            ..Default::default()
        };
        assert!(
            model
                .get_scheduled_decode_itl(&[decode_only_input])
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn aggregated_capacity_ttft_includes_new_request_prefill() {
        let small_limits = EnginePerfLimits {
            max_num_batched_tokens: 50,
            max_num_seqs: 1,
            max_kv_tokens: 1_000_000,
        };
        let mut model = EnginePerfModel::from_regression(
            WorkerType::Aggregated,
            small_limits,
            Some(fast_options()),
        )
        .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![mixed_observation(10, 1, 105, 0.011)],
                vec![mixed_observation(25, 1, 105, 0.026)],
                vec![mixed_observation(50, 1, 105, 0.051)],
            ])
            .unwrap();

        let capacity = model
            .find_engine_capacity_rps(EngineCapacityRequest {
                isl: 100,
                osl: 10,
                ttft_sla: None,
                itl_sla: None,
                e2e_latency_sla: None,
                accept_length: 1.0,
                kv_hit_rate: None,
                optimization_target: OptimizationTarget::Latency,
            })
            .unwrap()
            .unwrap();
        assert!(capacity.ttft.unwrap().as_secs_f64() > 0.08);
    }

    #[test]
    fn decode_capacity_accept_length_discounts_itl_and_rps() {
        let mut model =
            EnginePerfModel::from_regression(WorkerType::Decode, limits(), Some(fast_options()))
                .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![decode_observation(1, 100, 0.010)],
                vec![decode_observation(2, 200, 0.020)],
            ])
            .unwrap();

        let base = model
            .find_engine_capacity_rps(EngineCapacityRequest {
                isl: 100,
                osl: 10,
                ttft_sla: None,
                itl_sla: None,
                e2e_latency_sla: None,
                accept_length: 1.0,
                kv_hit_rate: None,
                optimization_target: OptimizationTarget::Throughput,
            })
            .unwrap()
            .unwrap();
        let spec = model
            .find_engine_capacity_rps(EngineCapacityRequest {
                isl: 100,
                osl: 10,
                ttft_sla: None,
                itl_sla: None,
                e2e_latency_sla: None,
                accept_length: 2.0,
                kv_hit_rate: None,
                optimization_target: OptimizationTarget::Throughput,
            })
            .unwrap()
            .unwrap();

        assert!(spec.rps > base.rps);
        assert!(spec.itl.unwrap() < base.itl.unwrap());
    }

    #[test]
    fn aggregated_capacity_accept_length_tightens_prefill_balance() {
        assert!(prefill_decode_balanced(100, 100, 33, 100, 2.0).unwrap());
        assert!(!prefill_decode_balanced(100, 100, 34, 100, 2.0).unwrap());
        assert_eq!(aggregate_prefill_per_iter(100, 100, 33, 2.0).unwrap(), 66);

        let small_limits = EnginePerfLimits {
            max_num_batched_tokens: 100,
            max_num_seqs: 100,
            max_kv_tokens: 1_000_000,
        };
        let mut model = EnginePerfModel::from_regression(
            WorkerType::Aggregated,
            small_limits,
            Some(fast_options()),
        )
        .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![mixed_observation(10, 1, 150, 0.010)],
                vec![mixed_observation(50, 1, 150, 0.050)],
                vec![mixed_observation(100, 1, 150, 0.100)],
            ])
            .unwrap();

        let capacity = model
            .find_engine_capacity_rps(EngineCapacityRequest {
                isl: 100,
                osl: 100,
                ttft_sla: None,
                itl_sla: None,
                e2e_latency_sla: None,
                accept_length: 2.0,
                kv_hit_rate: None,
                optimization_target: OptimizationTarget::Throughput,
            })
            .unwrap()
            .unwrap();

        let inferred_batch = capacity.rps * 100.0 * capacity.itl.unwrap().as_secs_f64();
        assert!(inferred_batch <= 34.0);
    }

    #[test]
    fn aggregated_capacity_returns_error_on_e2e_overflow() {
        let large_limits = EnginePerfLimits {
            max_num_batched_tokens: 8192,
            max_num_seqs: 1,
            max_kv_tokens: u32::MAX,
        };
        let mut model = EnginePerfModel::from_regression(
            WorkerType::Aggregated,
            large_limits,
            Some(fast_options()),
        )
        .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![mixed_observation(1, 1, 2_147_483_648, 1.0e12)],
                vec![mixed_observation(2, 1, 2_147_483_649, 1.0e12)],
            ])
            .unwrap();

        let err = model
            .find_engine_capacity_rps(EngineCapacityRequest {
                isl: 1,
                osl: u32::MAX,
                ttft_sla: None,
                itl_sla: None,
                e2e_latency_sla: None,
                accept_length: 1.0,
                kv_hit_rate: None,
                optimization_target: OptimizationTarget::Throughput,
            })
            .unwrap_err();

        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn prefill_capacity_batches_requests_within_limits() {
        let small_limits = EnginePerfLimits {
            max_num_batched_tokens: 400,
            max_num_seqs: 4,
            max_kv_tokens: 1_000_000,
        };
        let mut model = EnginePerfModel::from_regression(
            WorkerType::Prefill,
            small_limits,
            Some(fast_options()),
        )
        .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![prefill_observation(100, 0.020)],
                vec![prefill_observation(400, 0.050)],
            ])
            .unwrap();

        let capacity = model
            .find_engine_capacity_rps(EngineCapacityRequest {
                isl: 100,
                osl: 10,
                ttft_sla: None,
                itl_sla: None,
                e2e_latency_sla: None,
                accept_length: 1.0,
                kv_hit_rate: None,
                optimization_target: OptimizationTarget::Throughput,
            })
            .unwrap()
            .unwrap();

        assert!(capacity.rps > 70.0);
        assert!(capacity.ttft.is_some());
        assert_eq!(capacity.e2e_latency, capacity.ttft);
        assert!(capacity.itl.is_none());
    }

    #[test]
    fn prefill_capacity_kv_hit_rate_discounts_prefill_compute() {
        let single_seq_limits = EnginePerfLimits {
            max_num_batched_tokens: 8192,
            max_num_seqs: 1,
            max_kv_tokens: 1_000_000,
        };
        let mut model = EnginePerfModel::from_regression(
            WorkerType::Prefill,
            single_seq_limits,
            Some(fast_options()),
        )
        .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![prefill_observation(100, 0.020)],
                vec![prefill_observation(400, 0.080)],
            ])
            .unwrap();

        let base = model
            .find_engine_capacity_rps(EngineCapacityRequest {
                isl: 400,
                osl: 10,
                ttft_sla: None,
                itl_sla: None,
                e2e_latency_sla: None,
                accept_length: 1.0,
                kv_hit_rate: Some(0.0),
                optimization_target: OptimizationTarget::Throughput,
            })
            .unwrap()
            .unwrap();
        let hit = model
            .find_engine_capacity_rps(EngineCapacityRequest {
                isl: 400,
                osl: 10,
                ttft_sla: None,
                itl_sla: None,
                e2e_latency_sla: None,
                accept_length: 1.0,
                kv_hit_rate: Some(0.5),
                optimization_target: OptimizationTarget::Throughput,
            })
            .unwrap()
            .unwrap();

        assert!(hit.rps > base.rps);
        assert!(hit.ttft.unwrap() < base.ttft.unwrap());
    }

    #[test]
    fn aggregated_capacity_kv_hit_rate_keeps_raw_context_for_kv_cap() {
        let raw_context_limited = EnginePerfLimits {
            max_num_batched_tokens: 8192,
            max_num_seqs: 1,
            max_kv_tokens: 1_100,
        };
        let many_seq_same_kv = EnginePerfLimits {
            max_num_batched_tokens: 8192,
            max_num_seqs: 100,
            max_kv_tokens: 1_100,
        };
        let mut one_seq = EnginePerfModel::from_regression(
            WorkerType::Aggregated,
            raw_context_limited,
            Some(fast_options()),
        )
        .unwrap();
        let mut many_seq = EnginePerfModel::from_regression(
            WorkerType::Aggregated,
            many_seq_same_kv,
            Some(fast_options()),
        )
        .unwrap();
        let training = vec![
            vec![mixed_observation(10, 1, 1_050, 0.011)],
            vec![mixed_observation(20, 1, 1_050, 0.021)],
            vec![mixed_observation(40, 1, 1_050, 0.041)],
        ];
        one_seq.tune_with_fpms(&training).unwrap();
        many_seq.tune_with_fpms(&training).unwrap();

        let request = EngineCapacityRequest {
            isl: 1_000,
            osl: 100,
            ttft_sla: None,
            itl_sla: None,
            e2e_latency_sla: None,
            accept_length: 1.0,
            kv_hit_rate: Some(0.9),
            optimization_target: OptimizationTarget::Throughput,
        };
        let one_seq_capacity = one_seq
            .find_engine_capacity_rps(request.clone())
            .unwrap()
            .unwrap();
        let many_seq_capacity = many_seq.find_engine_capacity_rps(request).unwrap().unwrap();

        assert!((one_seq_capacity.rps - many_seq_capacity.rps).abs() < 1e-9);
    }

    #[test]
    fn decode_capacity_returns_none_when_kv_cannot_fit_one_sequence() {
        let small_limits = EnginePerfLimits {
            max_num_batched_tokens: 8192,
            max_num_seqs: 4,
            max_kv_tokens: 50,
        };
        let mut model = EnginePerfModel::from_regression(
            WorkerType::Decode,
            small_limits,
            Some(fast_options()),
        )
        .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![decode_observation(1, 100, 0.010)],
                vec![decode_observation(2, 200, 0.020)],
            ])
            .unwrap();

        let capacity = model
            .find_engine_capacity_rps(EngineCapacityRequest {
                isl: 100,
                osl: 10,
                ttft_sla: None,
                itl_sla: None,
                e2e_latency_sla: None,
                accept_length: 1.0,
                kv_hit_rate: None,
                optimization_target: OptimizationTarget::Throughput,
            })
            .unwrap();

        assert!(capacity.is_none());
    }

    #[test]
    fn capacity_marks_unsupported_sla_metrics_ineligible() {
        let mut prefill =
            EnginePerfModel::from_regression(WorkerType::Prefill, limits(), Some(fast_options()))
                .unwrap();
        prefill
            .tune_with_fpms(&vec![
                vec![prefill_observation(100, 0.010)],
                vec![prefill_observation(200, 0.020)],
            ])
            .unwrap();
        let prefill_capacity = prefill
            .find_engine_capacity_rps(EngineCapacityRequest {
                isl: 100,
                osl: 10,
                ttft_sla: None,
                itl_sla: Some(Duration::from_secs(1)),
                e2e_latency_sla: None,
                accept_length: 1.0,
                kv_hit_rate: None,
                optimization_target: OptimizationTarget::Throughput,
            })
            .unwrap()
            .unwrap();
        assert!(!prefill_capacity.eligible);

        let mut decode =
            EnginePerfModel::from_regression(WorkerType::Decode, limits(), Some(fast_options()))
                .unwrap();
        decode
            .tune_with_fpms(&vec![
                vec![decode_observation(1, 100, 0.010)],
                vec![decode_observation(2, 200, 0.020)],
            ])
            .unwrap();
        let decode_capacity = decode
            .find_engine_capacity_rps(EngineCapacityRequest {
                isl: 100,
                osl: 10,
                ttft_sla: Some(Duration::from_secs(1)),
                itl_sla: None,
                e2e_latency_sla: None,
                accept_length: 1.0,
                kv_hit_rate: None,
                optimization_target: OptimizationTarget::Throughput,
            })
            .unwrap()
            .unwrap();
        assert!(!decode_capacity.eligible);
    }

    #[test]
    fn decode_capacity_returns_best_point_after_training() {
        let mut model =
            EnginePerfModel::from_regression(WorkerType::Decode, limits(), Some(fast_options()))
                .unwrap();
        model
            .tune_with_fpms(&vec![
                vec![decode_observation(1, 100, 0.010)],
                vec![decode_observation(2, 200, 0.020)],
            ])
            .unwrap();
        let capacity = model
            .find_engine_capacity_rps(EngineCapacityRequest {
                isl: 100,
                osl: 10,
                ttft_sla: None,
                itl_sla: Some(Duration::from_secs_f64(1.0)),
                e2e_latency_sla: None,
                accept_length: 1.0,
                kv_hit_rate: None,
                optimization_target: OptimizationTarget::Throughput,
            })
            .unwrap()
            .unwrap();
        assert!(capacity.rps > 0.0);
        assert!(capacity.itl.is_some());
        assert!(capacity.ttft.is_none());
    }
}
