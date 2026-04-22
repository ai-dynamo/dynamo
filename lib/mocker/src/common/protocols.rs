// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use dynamo_kv_router::config::RouterQueuePolicy;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;
use validator::Validate;

use crate::common::perf_model::PerfModel;
use dynamo_kv_router::protocols::KvCacheEvent;
use dynamo_tokens::blocks::UniqueBlock;
use dynamo_tokens::{BlockHash, PositionalLineageHash, SequenceHash, Token};

/// Metadata marker type for kvbm-logical blocks in the mocker's G1 pool.
#[derive(Clone, Debug)]
pub struct G1;

/// Eviction strategy for the kvbm-logical inactive pool.
///
/// `Lineage` is the default and matches kvbm-logical's own default — it evicts
/// leaf blocks first, which subsumes the preemption-priority behaviour that the
/// mocker's old `LRUEvictor::push_front` provided.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum MockerEvictionBackend {
    Lru,
    MultiLru,
    #[default]
    Lineage,
}

/// Trait for publishing KV cache events.
/// This abstracts the runtime dependency so mocker components can remain generic.
pub trait KvCacheEventSink: Send + Sync {
    fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()>;
}

/// Raw KV event payload used by transport-specific publishers such as the
/// vLLM-native ZMQ event stream.
#[derive(Debug, Clone)]
pub struct RawKvEvent {
    pub event: KvCacheEvent,
    pub block_token_ids: Option<Vec<Vec<u32>>>,
}

/// Trait for publishing transport-specific raw KV event payloads.
pub trait RawKvEventSink: Send + Sync {
    fn publish(&self, event: RawKvEvent) -> anyhow::Result<()>;
}

/// Shared KV event publisher bundle used by schedulers and KV managers.
#[derive(Clone, Default)]
pub struct KvEventPublishers {
    event_sink: Option<Arc<dyn KvCacheEventSink>>,
    raw_sink: Option<Arc<dyn RawKvEventSink>>,
}

impl KvEventPublishers {
    pub fn new(
        event_sink: Option<Arc<dyn KvCacheEventSink>>,
        raw_sink: Option<Arc<dyn RawKvEventSink>>,
    ) -> Self {
        Self {
            event_sink,
            raw_sink,
        }
    }

    pub fn raw_enabled(&self) -> bool {
        self.raw_sink.is_some()
    }

    pub fn is_empty(&self) -> bool {
        self.event_sink.is_none() && self.raw_sink.is_none()
    }

    pub fn publish(
        &self,
        event: KvCacheEvent,
        block_token_ids: Option<&[Vec<u32>]>,
    ) -> anyhow::Result<()> {
        if let Some(sink) = self.event_sink.as_ref() {
            sink.publish(event.clone())?;
        }

        if let Some(sink) = self.raw_sink.as_ref() {
            sink.publish(RawKvEvent {
                event,
                block_token_ids: block_token_ids.map(|token_ids| token_ids.to_vec()),
            })?;
        }

        Ok(())
    }
}

/// Per-iteration forward pass snapshot, mirroring the Python `ForwardPassMetrics`
/// schema in `components/src/dynamo/common/forward_pass_metrics.py`.
///
/// Produced by the scheduler core after each `execute_pass_internal()` call.
/// The runtime-dependent layer (`lib/llm`) wraps this with identity fields
/// (worker_id, dp_rank, counter_id) and serializes to msgpack for the event plane.
#[derive(Debug, Clone, Default)]
pub struct ForwardPassSnapshot {
    // -- scheduled requests (executed this iteration) --
    pub num_prefill_requests: u32,
    pub sum_prefill_tokens: u64,
    pub var_prefill_length: f64,
    pub sum_prefill_kv_tokens: u64,
    pub num_decode_requests: u32,
    pub sum_decode_kv_tokens: u64,
    pub var_decode_kv_tokens: f64,
    // -- queued requests (waiting, not scheduled) --
    pub num_queued_prefill: u32,
    pub sum_queued_prefill_tokens: u64,
    pub var_queued_prefill_length: f64,
    pub num_queued_decode: u32,
    pub sum_queued_decode_kv_tokens: u64,
    pub var_queued_decode_kv_tokens: f64,
    // -- timing --
    pub wall_time_secs: f64,
}

/// Trait for publishing forward pass metrics snapshots.
/// This abstracts the FPM publishing pipeline so mocker schedulers remain generic.
pub trait FpmSink: Send + Sync {
    fn publish(&self, snapshot: ForwardPassSnapshot) -> anyhow::Result<()>;
}

/// Optional FPM sink used by schedulers.
/// Wraps `Option<Arc<dyn FpmSink>>` for ergonomic passing and no-op default behavior.
#[derive(Clone, Default)]
pub struct FpmPublisher {
    sink: Option<Arc<dyn FpmSink>>,
}

impl FpmPublisher {
    pub fn new(sink: Option<Arc<dyn FpmSink>>) -> Self {
        Self { sink }
    }

    pub fn publish(&self, snapshot: ForwardPassSnapshot) -> anyhow::Result<()> {
        if let Some(sink) = &self.sink {
            sink.publish(snapshot)?;
        }
        Ok(())
    }
}

pub type NumBlocks = usize;

/// Represents different block movement operations in the cache
/// For Use and Promote variants, block hashes are included for KV event publishing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MoveBlock {
    Use(
        Vec<UniqueBlock>,
        Vec<BlockHash>,
        Vec<PositionalLineageHash>,
        Option<Vec<Vec<u32>>>,
        Option<UniqueBlock>,
    ),
    Destroy(Vec<UniqueBlock>),
    Deref(Vec<UniqueBlock>),
    Promote(
        Uuid,
        SequenceHash,
        Option<u64>,
        BlockHash,
        PositionalLineageHash,
        Option<Vec<u32>>,
    ),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MoveBlockResponse {
    Store(Vec<SequenceHash>, Option<u64>),
    Remove(Vec<SequenceHash>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectRequest {
    pub tokens: Vec<Token>,
    pub max_output_tokens: usize,
    pub uuid: Option<Uuid>,
    pub dp_rank: u32,
    pub arrival_timestamp_ms: Option<f64>,
}

/// Represents the cost of prefilling content in the cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillCost {
    pub new_blocks: usize,
    pub new_tokens: usize,
    /// Number of tokens already cached (prefix hit).
    /// isl = cached_tokens + new_tokens
    pub cached_tokens: usize,
}

impl PrefillCost {
    pub fn predict_prefill_compute(
        &self,
        new_tokens: Option<usize>,
        perf_model: &PerfModel,
    ) -> f64 {
        let tokens = new_tokens.unwrap_or(self.new_tokens);
        let isl = self.cached_tokens + tokens;
        perf_model.predict_prefill_time(1, isl, self.cached_tokens)
    }
}

/// Signal for output token generation with completion status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSignal {
    pub uuid: Uuid,
    pub completed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff_delay_ms: Option<f64>,
}

/// Preemption policy for evicting decode requests under memory pressure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PreemptionMode {
    /// Evict the newest request (matches vLLM v1 default)
    #[default]
    Lifo,
    /// Evict the oldest request
    Fifo,
}

/// Engine type for selecting scheduling and KV cache simulation behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EngineType {
    /// vLLM-style scheduling with hash-based block KV cache
    #[default]
    Vllm,
    /// SGLang-style scheduling with radix-tree KV cache
    Sglang,
}

/// Worker type for disaggregated serving configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WorkerType {
    /// Standard aggregated worker handling both prefill and decode
    #[default]
    Aggregated,
    /// Dedicated prefill worker in disaggregated mode
    Prefill,
    /// Dedicated decode worker in disaggregated mode
    Decode,
}

/// Configuration for reasoning/thinking token output in the mocker.
///
/// When set, the mocker wraps the first portion of each response in thinking
/// boundary tokens: `[start_token, random..., end_token, random...]`.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ReasoningConfig {
    pub start_thinking_token_id: u32,
    pub end_thinking_token_id: u32,
    #[validate(range(min = 0.0, max = 1.0))]
    pub thinking_ratio: f64,
}

impl ReasoningConfig {
    /// Number of thinking tokens (including start/end boundaries) for a given osl.
    /// Returns 0 if osl < 2 (thinking disabled). Otherwise clamps to [2, osl].
    pub fn num_thinking_tokens(&self, max_output_tokens: usize) -> usize {
        if max_output_tokens < 2 {
            return 0;
        }
        let raw = (max_output_tokens as f64 * self.thinking_ratio).floor() as usize;
        if raw == 0 {
            return 0;
        }
        raw.max(2).min(max_output_tokens)
    }

    /// Number of response tokens after the thinking block.
    pub fn num_response_tokens(&self, max_output_tokens: usize) -> usize {
        max_output_tokens.saturating_sub(self.num_thinking_tokens(max_output_tokens))
    }
}

/// SGLang-specific configuration parameters.
///
/// Grouped into a nested struct to keep the `MockEngineArgs` namespace clean,
/// following the same pattern as [`ReasoningConfig`].
#[derive(Debug, Clone, Serialize, Deserialize, Validate, Default)]
pub struct SglangArgs {
    /// Scheduling policy: "fifo"/"fcfs" or "lpm". Default: "fifo".
    pub schedule_policy: Option<String>,
    /// Radix cache page size in tokens. Default: 1.
    #[validate(range(min = 1))]
    pub page_size: Option<usize>,
    /// Maximum prefill tokens budget per batch. Default: 16384.
    #[validate(range(min = 1))]
    pub max_prefill_tokens: Option<usize>,
    /// Chunked prefill size (max tokens per chunk). Default: 8192.
    #[validate(range(min = 1))]
    pub chunked_prefill_size: Option<usize>,
    /// Clip max new tokens for admission budget. Default: 4096.
    #[validate(range(min = 1))]
    pub clip_max_new_tokens: Option<usize>,
    /// Schedule conservativeness factor (0.0–1.0). Default: 1.0.
    #[validate(range(min = 0.0, max = 1.0))]
    pub schedule_conservativeness: Option<f64>,
}

/// Configuration arguments for MockEngine
#[derive(Debug, Clone, Serialize, Deserialize, Builder, Validate)]
#[builder(pattern = "owned", build_fn(public))]
pub struct MockEngineArgs {
    /// Engine type: vLLM or SGLang simulation
    #[builder(default = "EngineType::Vllm")]
    pub engine_type: EngineType,

    #[builder(default = "16384")]
    #[validate(range(min = 1))]
    pub num_gpu_blocks: usize,

    #[builder(default = "0")]
    pub block_size: usize,

    // This was 1024 in the past but reverted back to 256
    #[builder(default = Some(256))]
    #[validate(range(min = 1))]
    pub max_num_seqs: Option<usize>,

    // default for open api server, for llm class it's 16384
    #[builder(default = Some(8192))]
    #[validate(range(min = 1))]
    pub max_num_batched_tokens: Option<usize>,

    #[builder(default = true)]
    pub enable_prefix_caching: bool,

    #[builder(default = true)]
    pub enable_chunked_prefill: bool,

    #[builder(default = "1.0")]
    #[validate(range(min = 0.0))]
    pub speedup_ratio: f64,

    /// Additional speedup multiplier applied only to decode steps.
    /// Models speculative decoding (e.g. Eagle) where decode throughput improves
    /// without affecting prefill latency. The effective decode speedup is
    /// `speedup_ratio * decode_speedup_ratio`.
    #[builder(default = "1.0")]
    #[validate(range(min = 0.0))]
    pub decode_speedup_ratio: f64,

    #[builder(default = "1")]
    #[validate(range(min = 1))]
    pub dp_size: u32,

    /// Optional startup time in seconds to simulate engine initialization delay
    #[builder(default = "None")]
    #[validate(range(min = 0.0))]
    pub startup_time: Option<f64>,

    /// Worker type for disaggregated serving (Aggregated, Prefill, or Decode)
    #[builder(default = "WorkerType::Aggregated")]
    pub worker_type: WorkerType,

    /// Original planner profile NPZ path used to materialize `perf_model`.
    #[builder(default = "None")]
    pub planner_profile_data: Option<PathBuf>,

    /// Performance model for timing predictions (not serialized, loaded from planner_profile_data)
    #[serde(skip)]
    #[builder(default = "Arc::new(PerfModel::default())")]
    pub perf_model: Arc<PerfModel>,

    /// If set, indicates direct AIC SDK calls should be used.
    /// The value is the backend name (e.g., "sglang", "vllm").
    /// The Python layer reads this and overrides perf_model with an Aiconfigurator callback.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_backend: Option<String>,

    /// AIC GPU system name (e.g., "h200_sxm"). Required when aic_backend is set.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_system: Option<String>,

    /// AIC backend engine version (e.g., "0.12.0" for vLLM, "0.5.6.post2" for SGLang).
    /// If None, uses the default version for the backend.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_backend_version: Option<String>,

    /// Tensor parallel size for AIC latency prediction.
    /// Only affects AIC performance model lookups, not mocker scheduling.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_tp_size: Option<usize>,

    /// HuggingFace model path for AIC latency prediction (e.g., "nvidia/Llama-3.1-8B-Instruct-FP8").
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_model_path: Option<String>,

    /// MoE tensor-parallel size for AIC latency prediction (e.g., 4 for pure MoE-TP).
    /// Required for MoE models; must satisfy: aic_tp_size * aic_attention_dp_size == aic_moe_tp_size * aic_moe_ep_size.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_moe_tp_size: Option<usize>,

    /// MoE expert-parallel size for AIC latency prediction (e.g., 4 for pure EP).
    /// Required for MoE models; must satisfy: aic_tp_size * aic_attention_dp_size == aic_moe_tp_size * aic_moe_ep_size.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_moe_ep_size: Option<usize>,

    /// Attention data-parallel size for AIC latency prediction (default: 1).
    /// Corresponds to the `dp` dimension in AIC CLI output.
    /// Must satisfy: aic_tp_size * aic_attention_dp_size == aic_moe_tp_size * aic_moe_ep_size.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_attention_dp_size: Option<usize>,

    /// Enable worker-local KV indexer for tracking this worker's own KV cache state
    #[builder(default = "false")]
    pub enable_local_indexer: bool,

    /// Bootstrap port for disaggregated serving rendezvous.
    /// Prefill workers listen on this port; decode workers connect to it.
    /// If None, bootstrap rendezvous is disabled.
    #[builder(default = "None")]
    pub bootstrap_port: Option<u16>,

    /// KV cache bytes per token, auto-computed from model config by Python CLI.
    /// Formula: num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
    #[builder(default = "None")]
    pub kv_bytes_per_token: Option<usize>,

    /// KV cache transfer bandwidth in GB/s for disaggregated serving latency simulation.
    /// Default: 64.0 (inter-node InfiniBand). Set to 0 to disable KV transfer delay.
    /// For intra-node NVLink, typical value is ~450.
    #[builder(default = "None")]
    #[validate(range(min = 0.0))]
    pub kv_transfer_bandwidth: Option<f64>,

    /// Reasoning/thinking token configuration.
    /// When set, the mocker wraps output in thinking boundary tokens.
    #[builder(default = "None")]
    pub reasoning: Option<ReasoningConfig>,

    /// ZMQ port for publishing KV events in vLLM's native wire format.
    /// When set, the scheduler publishes to a ZMQ PUB socket instead of directly to NATS.
    /// A KvEventPublisher relay subscribes to this socket and forwards events to NATS.
    #[builder(default = "None")]
    pub zmq_kv_events_port: Option<u16>,

    /// ZMQ ROUTER port for replay of buffered KV event batches.
    /// When set alongside `zmq_kv_events_port`, the mocker binds a ROUTER socket
    /// that streams back buffered batches by sequence number on request.
    /// Port is offset by dp_rank (replay_port + dp_rank).
    #[builder(default = "None")]
    pub zmq_replay_port: Option<u16>,

    /// Preemption mode for decode eviction under memory pressure.
    /// Lifo (default) evicts the newest request; Fifo evicts the oldest.
    #[builder(default)]
    pub preemption_mode: PreemptionMode,

    /// Optional replay-only override for the router queue policy.
    #[builder(default = "None")]
    pub router_queue_policy: Option<RouterQueuePolicy>,

    /// SGLang-specific configuration. Only used when `engine_type == Sglang`.
    #[builder(default = "None")]
    pub sglang: Option<SglangArgs>,
}

impl Default for MockEngineArgs {
    fn default() -> MockEngineArgs {
        MockEngineArgsBuilder::default()
            .build()
            .expect("Failed to build default MockEngineArgs")
            .normalized()
            .expect("Failed to normalize default MockEngineArgs")
    }
}

impl MockEngineArgs {
    const DEFAULT_VLLM_BLOCK_SIZE: usize = 64;
    const DEFAULT_SGLANG_BLOCK_SIZE: usize = 1;

    pub fn builder() -> MockEngineArgsBuilder {
        MockEngineArgsBuilder::default()
    }

    pub fn normalized(mut self) -> anyhow::Result<Self> {
        match self.engine_type {
            EngineType::Vllm => {
                if self.block_size == 0 {
                    self.block_size = Self::DEFAULT_VLLM_BLOCK_SIZE;
                }
            }
            EngineType::Sglang => {
                let page_size = self.sglang.as_ref().and_then(|sglang| sglang.page_size);
                match (self.block_size, page_size) {
                    (0, None) => {
                        self.block_size = Self::DEFAULT_SGLANG_BLOCK_SIZE;
                    }
                    (0, Some(page_size)) => {
                        self.block_size = page_size;
                    }
                    (block_size, Some(page_size)) if block_size == page_size => {}
                    (_, Some(page_size)) => {
                        return Err(anyhow::anyhow!(
                            "engine_type=sglang requires block_size and sglang.page_size to match when both are set, got block_size={} and sglang.page_size={page_size}",
                            self.block_size,
                        ));
                    }
                    (_, None) => {}
                }
            }
        }

        if self.engine_type == EngineType::Sglang
            && let Some(chunked_prefill_size) = self
                .sglang
                .as_ref()
                .and_then(|sglang| sglang.chunked_prefill_size)
            && chunked_prefill_size % self.block_size != 0
        {
            return Err(anyhow::anyhow!(
                "engine_type=sglang requires sglang.chunked_prefill_size to be divisible by block_size, got chunked_prefill_size={} and block_size={}",
                chunked_prefill_size,
                self.block_size,
            ));
        }

        self.validate()
            .map_err(|error| anyhow::anyhow!("Failed to validate MockEngineArgs: {error}"))?;
        if self.block_size == 0 {
            return Err(anyhow::anyhow!("block_size must be greater than 0"));
        }

        Ok(self)
    }

    pub fn is_prefill(&self) -> bool {
        self.worker_type == WorkerType::Prefill
    }

    pub fn is_decode(&self) -> bool {
        self.worker_type == WorkerType::Decode
    }

    pub fn needs_kv_publisher(&self) -> bool {
        self.enable_prefix_caching && !self.is_decode()
    }

    /// Create MockEngineArgs from a JSON file containing extra engine arguments
    pub fn from_json_file(path: &Path) -> anyhow::Result<Self> {
        let file_content = std::fs::read_to_string(path)?;
        Self::from_json_str(&file_content)
    }

    pub fn from_json_str(content: &str) -> anyhow::Result<Self> {
        let mut builder = Self::builder();
        let extra_args: HashMap<String, serde_json::Value> = serde_json::from_str(content)?;

        // Define valid field names
        let valid_fields: HashSet<&str> = [
            "engine_type",
            "num_gpu_blocks",
            "block_size",
            "max_num_seqs",
            "max_num_batched_tokens",
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "speedup_ratio",
            "decode_speedup_ratio",
            "dp_size",
            "startup_time",
            "worker_type",
            "is_prefill",
            "is_decode",
            "planner_profile_data",
            "aic_backend",
            "aic_system",
            "aic_backend_version",
            "aic_tp_size",
            "aic_model_path",
            "aic_moe_tp_size",
            "aic_moe_ep_size",
            "aic_attention_dp_size",
            "enable_local_indexer",
            "bootstrap_port",
            "kv_bytes_per_token",
            "kv_transfer_bandwidth",
            "reasoning",
            "zmq_kv_events_port",
            "zmq_replay_port",
            "preemption_mode",
            "router_queue_policy",
            "sglang",
            "has_perf_model",
        ]
        .iter()
        .cloned()
        .collect();

        // Check for invalid arguments
        let invalid_args: Vec<String> = extra_args
            .keys()
            .filter(|key| !valid_fields.contains(key.as_str()))
            .cloned()
            .collect();

        if !invalid_args.is_empty() {
            return Err(anyhow::anyhow!(
                "Invalid arguments found in JSON file: {}. Valid arguments are: {:?}",
                invalid_args.join(", "),
                valid_fields
            ));
        }

        // Apply each extra argument to the builder
        if let Some(value) = extra_args.get("engine_type")
            && let Some(s) = value.as_str()
        {
            let engine_type = match s {
                "vllm" => EngineType::Vllm,
                "sglang" => EngineType::Sglang,
                other => {
                    return Err(anyhow::anyhow!(
                        "Invalid engine_type '{}'. Must be 'vllm' or 'sglang'.",
                        other
                    ));
                }
            };
            builder = builder.engine_type(engine_type);
        }

        if let Some(value) = extra_args.get("num_gpu_blocks")
            && let Some(num) = value.as_u64()
        {
            builder = builder.num_gpu_blocks(num as usize);
        }

        if let Some(value) = extra_args.get("block_size")
            && let Some(num) = value.as_u64()
        {
            builder = builder.block_size(num as usize);
        }

        if let Some(value) = extra_args.get("max_num_seqs") {
            if value.is_null() {
                builder = builder.max_num_seqs(None);
            } else if let Some(num) = value.as_u64() {
                builder = builder.max_num_seqs(Some(num as usize));
            }
        }

        if let Some(value) = extra_args.get("max_num_batched_tokens") {
            if value.is_null() {
                builder = builder.max_num_batched_tokens(None);
            } else if let Some(num) = value.as_u64() {
                builder = builder.max_num_batched_tokens(Some(num as usize));
            }
        }

        if let Some(value) = extra_args.get("enable_prefix_caching")
            && let Some(enabled) = value.as_bool()
        {
            builder = builder.enable_prefix_caching(enabled);
        }

        if let Some(value) = extra_args.get("enable_chunked_prefill")
            && let Some(enabled) = value.as_bool()
        {
            builder = builder.enable_chunked_prefill(enabled);
        }

        if let Some(value) = extra_args.get("speedup_ratio")
            && let Some(num) = value.as_f64()
        {
            builder = builder.speedup_ratio(num);
        }

        if let Some(value) = extra_args.get("decode_speedup_ratio")
            && let Some(num) = value.as_f64()
        {
            builder = builder.decode_speedup_ratio(num);
        }

        if let Some(value) = extra_args.get("dp_size")
            && let Some(num) = value.as_u64()
        {
            builder = builder.dp_size(num as u32);
        }

        if let Some(value) = extra_args.get("startup_time")
            && let Some(num) = value.as_f64()
        {
            builder = builder.startup_time(Some(num));
        }

        if let Some(value) = extra_args.get("enable_local_indexer")
            && let Some(enabled) = value.as_bool()
        {
            builder = builder.enable_local_indexer(enabled);
        }

        if let Some(value) = extra_args.get("bootstrap_port")
            && let Some(port) = value.as_u64()
        {
            builder = builder.bootstrap_port(Some(port as u16));
        }

        if let Some(value) = extra_args.get("kv_bytes_per_token")
            && let Some(num) = value.as_u64()
        {
            builder = builder.kv_bytes_per_token(Some(num as usize));
        }

        if let Some(value) = extra_args.get("kv_transfer_bandwidth")
            && let Some(num) = value.as_f64()
        {
            builder = builder.kv_transfer_bandwidth(Some(num));
        }

        if let Some(value) = extra_args.get("reasoning")
            && !value.is_null()
        {
            let cfg: ReasoningConfig = serde_json::from_value(value.clone())
                .map_err(|e| anyhow::anyhow!("Failed to parse reasoning config: {}", e))?;
            builder = builder.reasoning(Some(cfg));
        }

        if let Some(value) = extra_args.get("zmq_kv_events_port")
            && let Some(port) = value.as_u64()
        {
            builder = builder.zmq_kv_events_port(Some(port as u16));
        }

        if let Some(value) = extra_args.get("zmq_replay_port")
            && let Some(port) = value.as_u64()
        {
            builder = builder.zmq_replay_port(Some(port as u16));
        }

        if let Some(value) = extra_args.get("preemption_mode")
            && let Some(mode_str) = value.as_str()
        {
            let mode = match mode_str {
                "lifo" => PreemptionMode::Lifo,
                "fifo" => PreemptionMode::Fifo,
                _ => {
                    return Err(anyhow::anyhow!(
                        "Invalid preemption_mode: '{}'. Must be 'lifo' or 'fifo'.",
                        mode_str
                    ));
                }
            };
            builder = builder.preemption_mode(mode);
        }

        if let Some(value) = extra_args.get("router_queue_policy")
            && let Some(policy_str) = value.as_str()
        {
            let policy = policy_str.parse().map_err(|e: String| anyhow::anyhow!(e))?;
            builder = builder.router_queue_policy(Some(policy));
        }

        if let Some(value) = extra_args.get("sglang")
            && !value.is_null()
        {
            let cfg: SglangArgs = serde_json::from_value(value.clone())
                .map_err(|e| anyhow::anyhow!("Failed to parse sglang config: {}", e))?;
            builder = builder.sglang(Some(cfg));
        }

        let worker_type = if let Some(value) = extra_args.get("worker_type") {
            match value.as_str() {
                Some("aggregated") => WorkerType::Aggregated,
                Some("prefill") => WorkerType::Prefill,
                Some("decode") => WorkerType::Decode,
                Some(other) => {
                    return Err(anyhow::anyhow!(
                        "Invalid worker_type '{}'. Must be 'aggregated', 'prefill', or 'decode'.",
                        other
                    ));
                }
                None => {
                    return Err(anyhow::anyhow!(
                        "Invalid worker_type: expected string value."
                    ));
                }
            }
        } else {
            let is_prefill = extra_args
                .get("is_prefill")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let is_decode = extra_args
                .get("is_decode")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            match (is_prefill, is_decode) {
                (false, false) => WorkerType::Aggregated,
                (true, false) => WorkerType::Prefill,
                (false, true) => WorkerType::Decode,
                (true, true) => {
                    return Err(anyhow::anyhow!(
                        "Invalid worker configuration: is_prefill and is_decode cannot both be true."
                    ));
                }
            }
        };
        builder = builder.worker_type(worker_type);

        // Load performance model from NPZ file if provided.
        let perf_model = if let Some(path_str) = extra_args.get("planner_profile_data")
            && let Some(path_str) = path_str.as_str()
        {
            let npz_path = PathBuf::from(path_str);
            builder = builder.planner_profile_data(Some(npz_path.clone()));
            match PerfModel::from_npz(&npz_path) {
                Ok(model) => {
                    tracing::info!("Successfully loaded performance model from: {:?}", npz_path);
                    Arc::new(model)
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to load performance model from {:?}: {}. Falling back to polynomial model.",
                        npz_path,
                        e
                    );
                    Arc::new(PerfModel::default())
                }
            }
        } else {
            Arc::new(PerfModel::default())
        };
        builder = builder.perf_model(perf_model);

        // Check for AIC direct mode fields
        if let Some(backend) = extra_args.get("aic_backend")
            && let Some(backend_str) = backend.as_str()
        {
            builder = builder.aic_backend(Some(backend_str.to_string()));
        }
        if let Some(system) = extra_args.get("aic_system")
            && let Some(s) = system.as_str()
        {
            builder = builder.aic_system(Some(s.to_string()));
        }
        if let Some(version) = extra_args.get("aic_backend_version")
            && let Some(s) = version.as_str()
        {
            builder = builder.aic_backend_version(Some(s.to_string()));
        }
        if let Some(tp) = extra_args.get("aic_tp_size")
            && let Some(n) = tp.as_u64()
        {
            builder = builder.aic_tp_size(Some(n as usize));
        }
        if let Some(mp) = extra_args.get("aic_model_path")
            && let Some(s) = mp.as_str()
        {
            builder = builder.aic_model_path(Some(s.to_string()));
        }
        if let Some(v) = extra_args.get("aic_moe_tp_size")
            && let Some(n) = v.as_u64()
        {
            builder = builder.aic_moe_tp_size(Some(n as usize));
        }
        if let Some(v) = extra_args.get("aic_moe_ep_size")
            && let Some(n) = v.as_u64()
        {
            builder = builder.aic_moe_ep_size(Some(n as usize));
        }
        if let Some(v) = extra_args.get("aic_attention_dp_size")
            && let Some(n) = v.as_u64()
        {
            builder = builder.aic_attention_dp_size(Some(n as usize));
        }
        // Build the MockEngineArgs with either defaults or overridden values
        builder
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build MockEngineArgs: {}", e))
            .and_then(Self::normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_mock_engine_args_json_round_trip_preserves_worker_type_and_nulls() {
        let args = MockEngineArgs::builder()
            .worker_type(WorkerType::Decode)
            .max_num_seqs(None)
            .max_num_batched_tokens(None)
            .reasoning(None)
            .sglang(None)
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        let payload = serde_json::json!({
            "engine_type": "vllm",
            "num_gpu_blocks": args.num_gpu_blocks,
            "block_size": args.block_size,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": args.enable_prefix_caching,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "speedup_ratio": args.speedup_ratio,
            "decode_speedup_ratio": args.decode_speedup_ratio,
            "dp_size": args.dp_size,
            "startup_time": args.startup_time,
            "worker_type": "decode",
            "planner_profile_data": args.planner_profile_data,
            "aic_backend": args.aic_backend,
            "aic_system": args.aic_system,
            "aic_backend_version": args.aic_backend_version,
            "aic_tp_size": args.aic_tp_size,
            "aic_model_path": args.aic_model_path,
            "enable_local_indexer": args.enable_local_indexer,
            "bootstrap_port": args.bootstrap_port,
            "kv_bytes_per_token": args.kv_bytes_per_token,
            "kv_transfer_bandwidth": args.kv_transfer_bandwidth,
            "reasoning": args.reasoning,
            "zmq_kv_events_port": args.zmq_kv_events_port,
            "zmq_replay_port": args.zmq_replay_port,
            "preemption_mode": "lifo",
            "router_queue_policy": args.router_queue_policy.map(|policy| policy.to_string()),
            "sglang": args.sglang,
            "has_perf_model": true,
        });

        let restored = MockEngineArgs::from_json_str(&payload.to_string()).unwrap();

        assert_eq!(restored.worker_type, WorkerType::Decode);
        assert_eq!(restored.max_num_seqs, None);
        assert_eq!(restored.max_num_batched_tokens, None);
    }

    #[test]
    fn test_unique_block_default_uniqueness() {
        // Create 10 default UniqueBlock instances
        let blocks: Vec<UniqueBlock> = (0..10).map(|_| UniqueBlock::default()).collect();

        // Extract UUIDs from each block
        let mut uuids = Vec::new();
        for block in blocks {
            match block {
                UniqueBlock::PartialBlock(uuid) => uuids.push(uuid),
                _ => panic!("Expected UuidIdentifier variant"),
            }
        }

        // Check that all UUIDs are unique by comparing each with every other
        for i in 0..uuids.len() {
            for j in i + 1..uuids.len() {
                assert_ne!(
                    uuids[i], uuids[j],
                    "UUID at index {} and {} are identical",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_normalized_sglang_uses_page_size_alias_for_block_size() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .sglang(Some(SglangArgs {
                page_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.block_size, 16);
    }

    #[test]
    fn test_normalized_sglang_accepts_equal_block_size_and_page_size() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(8)
            .sglang(Some(SglangArgs {
                page_size: Some(8),
                ..Default::default()
            }))
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.block_size, 8);
    }

    #[test]
    fn test_normalized_sglang_rejects_mismatched_block_size_and_page_size() {
        let error = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(8)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                ..Default::default()
            }))
            .build()
            .unwrap()
            .normalized()
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("block_size and sglang.page_size to match"),
            "unexpected error: {error}",
        );
    }

    #[test]
    fn test_normalized_sglang_defaults_block_size_to_one() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.block_size, 1);
    }

    #[test]
    fn test_from_json_file_normalizes_sglang_page_size() {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path().join("args.json");
        std::fs::write(
            &path,
            serde_json::to_string(&json!({
                "engine_type": "sglang",
                "sglang": {
                    "page_size": 32
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let args = MockEngineArgs::from_json_file(&path).unwrap();
        assert_eq!(args.block_size, 32);
    }

    #[test]
    fn test_normalized_sglang_rejects_chunked_prefill_not_divisible_by_block_size() {
        let error = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(6),
                ..Default::default()
            }))
            .build()
            .unwrap()
            .normalized()
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("chunked_prefill_size to be divisible by block_size"),
            "unexpected error: {error}",
        );
    }

    #[test]
    fn test_normalized_sglang_accepts_chunked_prefill_divisible_by_block_size() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(8),
                ..Default::default()
            }))
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.block_size, 4);
    }
}
