// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use dynamo_kv_router::config::RouterQueuePolicy;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;
use validator::Validate;

use crate::common::perf_model::PerfModel;
use dynamo_kv_router::protocols::{KvCacheEvent, StorageTier};
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

    fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        _storage_tier: StorageTier,
    ) -> anyhow::Result<()> {
        self.publish(event)
    }
}

/// Raw KV event payload used by transport-specific publishers such as the
/// vLLM-native ZMQ event stream.
#[derive(Debug, Clone)]
pub struct RawKvEvent {
    pub event: KvCacheEvent,
    pub block_token_ids: Option<Vec<Vec<u32>>>,
    pub storage_tier: StorageTier,
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
        self.publish_with_storage_tier(event, block_token_ids, StorageTier::Device)
    }

    pub fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        block_token_ids: Option<&[Vec<u32>]>,
        storage_tier: StorageTier,
    ) -> anyhow::Result<()> {
        if let Some(sink) = self.event_sink.as_ref() {
            sink.publish_with_storage_tier(event.clone(), storage_tier)?;
        }

        if let Some(sink) = self.raw_sink.as_ref() {
            sink.publish(RawKvEvent {
                event,
                block_token_ids: block_token_ids.map(|token_ids| token_ids.to_vec()),
                storage_tier,
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

/// Keeps omitted JSON fields distinct from explicit `null` so serde can replace
/// the old hand-written parser without losing Python `dump_json()` round trips.
#[derive(Debug, Clone, Default)]
enum OptionalConfigValue<T> {
    #[default]
    Missing,
    Present(Option<T>),
}

impl<'de, T> Deserialize<'de> for OptionalConfigValue<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(Self::Present)
    }
}

impl<T> OptionalConfigValue<T> {
    fn into_nullable(self) -> Option<Option<T>> {
        match self {
            Self::Missing => None,
            Self::Present(value) => Some(value),
        }
    }

    fn into_non_null(self, field: &str) -> Result<Option<T>, String> {
        match self {
            Self::Missing => Ok(None),
            Self::Present(Some(value)) => Ok(Some(value)),
            Self::Present(None) => Err(format!("{field} must not be null")),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct MockEngineArgsSerde {
    engine_type: OptionalConfigValue<String>,
    num_gpu_blocks: OptionalConfigValue<usize>,
    block_size: OptionalConfigValue<usize>,
    max_num_seqs: OptionalConfigValue<usize>,
    max_num_batched_tokens: OptionalConfigValue<usize>,
    enable_prefix_caching: OptionalConfigValue<bool>,
    enable_chunked_prefill: OptionalConfigValue<bool>,
    speedup_ratio: OptionalConfigValue<f64>,
    decode_speedup_ratio: OptionalConfigValue<f64>,
    dp_size: OptionalConfigValue<u32>,
    startup_time: OptionalConfigValue<f64>,
    worker_type: OptionalConfigValue<String>,
    is_prefill: OptionalConfigValue<bool>,
    is_decode: OptionalConfigValue<bool>,
    planner_profile_data: OptionalConfigValue<PathBuf>,
    aic_backend: OptionalConfigValue<String>,
    aic_system: OptionalConfigValue<String>,
    aic_backend_version: OptionalConfigValue<String>,
    aic_tp_size: OptionalConfigValue<usize>,
    aic_model_path: OptionalConfigValue<String>,
    aic_moe_tp_size: OptionalConfigValue<usize>,
    aic_moe_ep_size: OptionalConfigValue<usize>,
    aic_attention_dp_size: OptionalConfigValue<usize>,
    gpu_memory_utilization: OptionalConfigValue<f64>,
    mem_fraction_static: OptionalConfigValue<f64>,
    enable_local_indexer: OptionalConfigValue<bool>,
    bootstrap_port: OptionalConfigValue<u16>,
    kv_bytes_per_token: OptionalConfigValue<usize>,
    kv_transfer_bandwidth: OptionalConfigValue<f64>,
    num_g2_blocks: OptionalConfigValue<usize>,
    offload_batch_size: OptionalConfigValue<usize>,
    bandwidth_g1_to_g2_gbps: OptionalConfigValue<f64>,
    bandwidth_g2_to_g1_gbps: OptionalConfigValue<f64>,
    reasoning: OptionalConfigValue<ReasoningConfig>,
    zmq_kv_events_port: OptionalConfigValue<u16>,
    zmq_replay_port: OptionalConfigValue<u16>,
    preemption_mode: OptionalConfigValue<String>,
    router_queue_policy: OptionalConfigValue<String>,
    sglang: OptionalConfigValue<SglangArgs>,
    #[serde(rename = "has_perf_model")]
    _has_perf_model: OptionalConfigValue<serde_json::Value>,
}

fn parse_engine_type(value: &str) -> Result<EngineType, String> {
    match value {
        "vllm" => Ok(EngineType::Vllm),
        "sglang" => Ok(EngineType::Sglang),
        other => Err(format!(
            "Invalid engine_type '{other}'. Must be 'vllm' or 'sglang'."
        )),
    }
}

fn parse_worker_type(value: &str) -> Result<WorkerType, String> {
    match value {
        "aggregated" => Ok(WorkerType::Aggregated),
        "prefill" => Ok(WorkerType::Prefill),
        "decode" => Ok(WorkerType::Decode),
        other => Err(format!(
            "Invalid worker_type '{other}'. Must be 'aggregated', 'prefill', or 'decode'."
        )),
    }
}

fn parse_preemption_mode(value: &str) -> Result<PreemptionMode, String> {
    match value {
        "lifo" => Ok(PreemptionMode::Lifo),
        "fifo" => Ok(PreemptionMode::Fifo),
        other => Err(format!(
            "Invalid preemption_mode: '{other}'. Must be 'lifo' or 'fifo'."
        )),
    }
}

fn load_perf_model(path: &Path) -> Arc<PerfModel> {
    match PerfModel::from_npz(path) {
        Ok(model) => {
            tracing::info!("Successfully loaded performance model from: {:?}", path);
            Arc::new(model)
        }
        Err(e) => {
            tracing::error!(
                "Failed to load performance model from {:?}: {}. Falling back to polynomial model.",
                path,
                e
            );
            Arc::new(PerfModel::default())
        }
    }
}

/// Configuration arguments for MockEngine
#[derive(Debug, Clone, Serialize, Deserialize, Builder, Validate)]
#[serde(try_from = "MockEngineArgsSerde")]
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

    /// GPU memory fraction for AIC KV capacity estimation with vLLM.
    #[builder(default = "None")]
    #[validate(range(min = 0.0, max = 1.0))]
    pub gpu_memory_utilization: Option<f64>,

    /// Static memory fraction for AIC KV capacity estimation with SGLang.
    #[builder(default = "None")]
    #[validate(range(min = 0.0, max = 1.0))]
    pub mem_fraction_static: Option<f64>,

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

    /// KVBM G2 (host DRAM) block capacity. When the `kvbm-offload`
    /// feature is enabled, setting this explicitly opts the mocker into
    /// G2 offload simulation. When unset, no G2 offload engine is attached.
    #[builder(default = "None")]
    #[validate(range(min = 1))]
    pub num_g2_blocks: Option<usize>,

    /// Batch size for the G1→G2 offload pipeline. Offloads are grouped
    /// into batches of this size before being handed to the worker.
    /// Only consulted when the `kvbm-offload` feature is enabled;
    /// falls back to the `KvbmOffloadConfig` default when unset.
    #[builder(default = "None")]
    #[validate(range(min = 1))]
    pub offload_batch_size: Option<usize>,

    /// G1→G2 offload bandwidth in GB/s for the PS-queue simulation.
    /// Only consulted when the `kvbm-offload` feature is enabled;
    /// falls back to the `KvbmOffloadConfig` default (host DRAM PCIe
    /// ballpark) when unset.
    #[builder(default = "None")]
    #[validate(range(min = 0.0))]
    pub bandwidth_g1_to_g2_gbps: Option<f64>,

    /// G2→G1 onboard bandwidth in GB/s for the PS-queue simulation.
    /// Only consulted when the `kvbm-offload` feature is enabled;
    /// falls back to the `KvbmOffloadConfig` default when unset.
    #[builder(default = "None")]
    #[validate(range(min = 0.0))]
    pub bandwidth_g2_to_g1_gbps: Option<f64>,

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

impl TryFrom<MockEngineArgsSerde> for MockEngineArgs {
    type Error = String;

    fn try_from(compat: MockEngineArgsSerde) -> Result<Self, Self::Error> {
        let mut builder = Self::builder();

        if let Some(engine_type) = compat.engine_type.into_non_null("engine_type")? {
            builder = builder.engine_type(parse_engine_type(&engine_type)?);
        }
        if let Some(Some(num_gpu_blocks)) = compat.num_gpu_blocks.into_nullable() {
            builder = builder.num_gpu_blocks(num_gpu_blocks);
        }
        if let Some(block_size) = compat.block_size.into_non_null("block_size")? {
            builder = builder.block_size(block_size);
        }
        if let Some(max_num_seqs) = compat.max_num_seqs.into_nullable() {
            builder = builder.max_num_seqs(max_num_seqs);
        }
        if let Some(max_num_batched_tokens) = compat.max_num_batched_tokens.into_nullable() {
            builder = builder.max_num_batched_tokens(max_num_batched_tokens);
        }
        if let Some(enable_prefix_caching) = compat
            .enable_prefix_caching
            .into_non_null("enable_prefix_caching")?
        {
            builder = builder.enable_prefix_caching(enable_prefix_caching);
        }
        if let Some(enable_chunked_prefill) = compat
            .enable_chunked_prefill
            .into_non_null("enable_chunked_prefill")?
        {
            builder = builder.enable_chunked_prefill(enable_chunked_prefill);
        }
        if let Some(speedup_ratio) = compat.speedup_ratio.into_non_null("speedup_ratio")? {
            builder = builder.speedup_ratio(speedup_ratio);
        }
        if let Some(decode_speedup_ratio) = compat
            .decode_speedup_ratio
            .into_non_null("decode_speedup_ratio")?
        {
            builder = builder.decode_speedup_ratio(decode_speedup_ratio);
        }
        if let Some(dp_size) = compat.dp_size.into_non_null("dp_size")? {
            builder = builder.dp_size(dp_size);
        }
        if let Some(startup_time) = compat.startup_time.into_nullable() {
            builder = builder.startup_time(startup_time);
        }

        let worker_type = if let Some(worker_type) =
            compat.worker_type.into_non_null("worker_type")?
        {
            parse_worker_type(&worker_type)?
        } else {
            let is_prefill = compat
                .is_prefill
                .into_non_null("is_prefill")?
                .unwrap_or(false);
            let is_decode = compat
                .is_decode
                .into_non_null("is_decode")?
                .unwrap_or(false);

            match (is_prefill, is_decode) {
                (false, false) => WorkerType::Aggregated,
                (true, false) => WorkerType::Prefill,
                (false, true) => WorkerType::Decode,
                (true, true) => {
                    return Err(
                        "Invalid worker configuration: is_prefill and is_decode cannot both be true."
                            .to_string(),
                    );
                }
            }
        };
        builder = builder.worker_type(worker_type);

        if let Some(planner_profile_data) = compat.planner_profile_data.into_nullable() {
            builder = builder.planner_profile_data(planner_profile_data.clone());
            if let Some(path) = planner_profile_data {
                builder = builder.perf_model(load_perf_model(&path));
            }
        }

        if let Some(aic_backend) = compat.aic_backend.into_nullable() {
            builder = builder.aic_backend(aic_backend);
        }
        if let Some(aic_system) = compat.aic_system.into_nullable() {
            builder = builder.aic_system(aic_system);
        }
        if let Some(aic_backend_version) = compat.aic_backend_version.into_nullable() {
            builder = builder.aic_backend_version(aic_backend_version);
        }
        if let Some(aic_tp_size) = compat.aic_tp_size.into_nullable() {
            builder = builder.aic_tp_size(aic_tp_size);
        }
        if let Some(aic_model_path) = compat.aic_model_path.into_nullable() {
            builder = builder.aic_model_path(aic_model_path);
        }
        if let Some(aic_moe_tp_size) = compat.aic_moe_tp_size.into_nullable() {
            builder = builder.aic_moe_tp_size(aic_moe_tp_size);
        }
        if let Some(aic_moe_ep_size) = compat.aic_moe_ep_size.into_nullable() {
            builder = builder.aic_moe_ep_size(aic_moe_ep_size);
        }
        if let Some(aic_attention_dp_size) = compat.aic_attention_dp_size.into_nullable() {
            builder = builder.aic_attention_dp_size(aic_attention_dp_size);
        }
        if let Some(gpu_memory_utilization) = compat.gpu_memory_utilization.into_nullable() {
            builder = builder.gpu_memory_utilization(gpu_memory_utilization);
        }
        if let Some(mem_fraction_static) = compat.mem_fraction_static.into_nullable() {
            builder = builder.mem_fraction_static(mem_fraction_static);
        }
        if let Some(enable_local_indexer) = compat
            .enable_local_indexer
            .into_non_null("enable_local_indexer")?
        {
            builder = builder.enable_local_indexer(enable_local_indexer);
        }
        if let Some(bootstrap_port) = compat.bootstrap_port.into_nullable() {
            builder = builder.bootstrap_port(bootstrap_port);
        }
        if let Some(kv_bytes_per_token) = compat.kv_bytes_per_token.into_nullable() {
            builder = builder.kv_bytes_per_token(kv_bytes_per_token);
        }
        if let Some(kv_transfer_bandwidth) = compat.kv_transfer_bandwidth.into_nullable() {
            builder = builder.kv_transfer_bandwidth(kv_transfer_bandwidth);
        }
        if let Some(num_g2_blocks) = compat.num_g2_blocks.into_nullable() {
            builder = builder.num_g2_blocks(num_g2_blocks);
        }
        if let Some(offload_batch_size) = compat.offload_batch_size.into_nullable() {
            builder = builder.offload_batch_size(offload_batch_size);
        }
        if let Some(bandwidth_g1_to_g2_gbps) = compat.bandwidth_g1_to_g2_gbps.into_nullable() {
            builder = builder.bandwidth_g1_to_g2_gbps(bandwidth_g1_to_g2_gbps);
        }
        if let Some(bandwidth_g2_to_g1_gbps) = compat.bandwidth_g2_to_g1_gbps.into_nullable() {
            builder = builder.bandwidth_g2_to_g1_gbps(bandwidth_g2_to_g1_gbps);
        }
        if let Some(reasoning) = compat.reasoning.into_nullable() {
            builder = builder.reasoning(reasoning);
        }
        if let Some(zmq_kv_events_port) = compat.zmq_kv_events_port.into_nullable() {
            builder = builder.zmq_kv_events_port(zmq_kv_events_port);
        }
        if let Some(zmq_replay_port) = compat.zmq_replay_port.into_nullable() {
            builder = builder.zmq_replay_port(zmq_replay_port);
        }
        if let Some(preemption_mode) = compat.preemption_mode.into_non_null("preemption_mode")? {
            builder = builder.preemption_mode(parse_preemption_mode(&preemption_mode)?);
        }
        if let Some(router_queue_policy) = compat.router_queue_policy.into_nullable() {
            let router_queue_policy = router_queue_policy
                .map(|policy| policy.parse().map_err(|e: String| e))
                .transpose()?;
            builder = builder.router_queue_policy(router_queue_policy);
        }
        if let Some(sglang) = compat.sglang.into_nullable() {
            builder = builder.sglang(sglang);
        }

        builder
            .build()
            .map_err(|e| format!("Failed to build MockEngineArgs: {e}"))?
            .normalized()
            .map_err(|e| e.to_string())
    }
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
        let mut deserializer = serde_json::Deserializer::from_str(content);
        serde_path_to_error::deserialize(&mut deserializer)
            .map_err(|error| anyhow::anyhow!("{error}"))
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
    fn test_mock_engine_args_json_rejects_unknown_and_invalid_types() {
        let unknown = MockEngineArgs::from_json_str(&json!({"unknown": true}).to_string())
            .expect_err("unknown fields should be rejected");
        assert!(
            unknown.to_string().contains("unknown field"),
            "unexpected error: {unknown}",
        );

        let invalid =
            MockEngineArgs::from_json_str(&json!({"gpu_memory_utilization": "bad"}).to_string())
                .expect_err("wrongly typed fields should be rejected");
        assert!(
            invalid.to_string().contains("gpu_memory_utilization"),
            "unexpected error: {invalid}",
        );
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
