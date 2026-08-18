// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduler-local data structures.
//!
//! Serializable user configuration lives in `crate::engine::EngineConfig`. The
//! `MockEngineArgs` type below is only the materialized, rank-local view used
//! by the mechanically moved scheduler algorithms.

use std::sync::Arc;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::engine::KvEvent;
use crate::engine::common::hashing::Token;
use crate::engine::common::perf_model::PerfModel;

/// Sink for neutral KV-cache events emitted by a rank.
pub(crate) trait KvCacheEventSink: Send + Sync {
    fn publish(&self, event: KvEvent) -> anyhow::Result<()>;
}

/// Optional neutral event sink used by the native KV managers.
#[derive(Clone, Default)]
pub(crate) struct KvEventPublishers {
    event_sink: Option<Arc<dyn KvCacheEventSink>>,
}

impl KvEventPublishers {
    pub(crate) fn new(event_sink: Option<Arc<dyn KvCacheEventSink>>) -> Self {
        Self { event_sink }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.event_sink.is_none()
    }

    pub(crate) fn publish(
        &self,
        event: KvEvent,
        _block_token_ids: Option<&[Vec<u32>]>,
    ) -> anyhow::Result<()> {
        if let Some(sink) = self.event_sink.as_ref() {
            sink.publish(event)?;
        }
        Ok(())
    }
}

/// Rank-local forward-pass metrics before conversion to the public neutral DTO.
#[derive(Debug, Clone, Default)]
pub(crate) struct ForwardPassSnapshot {
    pub num_prefill_requests: u32,
    pub sum_prefill_tokens: u64,
    pub var_prefill_length: f64,
    pub sum_prefill_kv_tokens: u64,
    pub num_decode_requests: u32,
    pub sum_decode_kv_tokens: u64,
    pub var_decode_kv_tokens: f64,
    pub num_queued_prefill: u32,
    pub sum_queued_prefill_tokens: u64,
    pub var_queued_prefill_length: f64,
    pub num_queued_decode: u32,
    pub sum_queued_decode_kv_tokens: u64,
    pub var_queued_decode_kv_tokens: f64,
    pub wall_time_secs: f64,
}

/// Internal request shape consumed by the moved schedulers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct DirectRequest {
    pub tokens: Vec<Token>,
    pub max_output_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_ids: Option<Vec<Token>>,
    pub uuid: Option<Uuid>,
    pub arrival_timestamp_ms: Option<f64>,
}

impl DirectRequest {
    #[inline]
    pub(crate) fn effective_max_output_tokens(&self) -> usize {
        self.output_token_ids
            .as_ref()
            .map_or(self.max_output_tokens, Vec::len)
    }
}

/// Cost of materializing one prompt in native G1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PrefillCost {
    pub new_blocks: usize,
    pub new_tokens: usize,
    pub cached_tokens: usize,
    pub active_cached_tokens: usize,
}

/// One output signal produced by a scheduler pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OutputSignal {
    pub uuid: Uuid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_id: Option<Token>,
    pub completed: bool,
    #[serde(default)]
    pub rejected: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff_delay_ms: Option<f64>,
    /// Prompt tokens served from KV cache at admission (scheduler truth,
    /// post-eviction). Set once, on the request's first output signal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PreemptionMode {
    #[default]
    Lifo,
    Fifo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum EngineType {
    #[default]
    Vllm,
    Sglang,
    Trtllm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SchedulingPolicy {
    #[default]
    Vllm,
    TrtllmGuaranteedNoEvict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum WorkerType {
    #[default]
    Aggregated,
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum KvTransferTimingMode {
    #[default]
    FullPrompt,
    DestinationMissing,
}

/// Materialized SGLang controls used by the moved scheduler implementation.
#[derive(Debug, Clone, Default)]
pub(crate) struct SglangArgs {
    pub schedule_policy: Option<String>,
    pub page_size: Option<usize>,
    pub max_prefill_tokens: Option<usize>,
    pub chunked_prefill_size: Option<usize>,
    pub clip_max_new_tokens: Option<usize>,
    pub schedule_conservativeness: Option<f64>,
}

/// Rank-local view of `EngineConfig`.
///
/// This is deliberately not serializable and performs no provider loading or
/// validation. `EngineConfig::validate` is the single configuration
/// boundary; `scheduler::rank::core_args` materializes this view.
#[derive(Debug, Clone, Builder)]
#[builder(pattern = "owned", build_fn(public))]
pub(crate) struct MockEngineArgs {
    #[builder(default = "EngineType::Vllm")]
    pub engine_type: EngineType,
    #[builder(default = "16_384")]
    pub num_gpu_blocks: usize,
    #[builder(default = "64")]
    pub block_size: usize,
    #[builder(default = "None")]
    pub max_model_len: Option<usize>,
    #[builder(default = "Some(256)")]
    pub max_num_seqs: Option<usize>,
    #[builder(default = "Some(8_192)")]
    pub max_num_batched_tokens: Option<usize>,
    #[builder(default = "true")]
    pub enable_prefix_caching: bool,
    #[builder(default = "true")]
    pub enable_chunked_prefill: bool,
    #[builder(default = "1.0")]
    pub speedup_ratio: f64,
    #[builder(default = "1.0")]
    pub decode_speedup_ratio: f64,
    #[builder(default = "WorkerType::Aggregated")]
    pub worker_type: WorkerType,
    #[builder(default = "Arc::new(PerfModel::Polynomial)")]
    pub perf_model: Arc<PerfModel>,
    #[builder(default = "None")]
    pub aic_nextn: Option<usize>,
    #[builder(default = "None")]
    pub aic_nextn_accept_rates: Option<String>,
    #[builder(default = "42")]
    pub aic_mtp_seed: u64,
    #[builder(default = "None")]
    pub kv_bytes_per_token: Option<usize>,
    #[builder(default = "None")]
    pub kv_transfer_bandwidth: Option<f64>,
    #[builder(default = "KvTransferTimingMode::FullPrompt")]
    pub kv_transfer_timing_mode: KvTransferTimingMode,
    #[builder(default = "PreemptionMode::Lifo")]
    pub preemption_mode: PreemptionMode,
    #[builder(default = "None")]
    pub sglang: Option<SglangArgs>,
    #[builder(default = "false")]
    pub emit_kv_events: bool,
    #[builder(default = "false")]
    pub emit_kv_token_ids: bool,
}

impl MockEngineArgs {
    #[cfg(test)]
    pub(crate) fn builder() -> MockEngineArgsBuilder {
        MockEngineArgsBuilder::default()
    }

    pub(crate) fn scheduling_policy(&self) -> SchedulingPolicy {
        match self.engine_type {
            EngineType::Trtllm => SchedulingPolicy::TrtllmGuaranteedNoEvict,
            EngineType::Vllm | EngineType::Sglang => SchedulingPolicy::Vllm,
        }
    }
}
