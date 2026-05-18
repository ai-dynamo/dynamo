// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `LLMEngine` trait plus registration-metadata and output-construction helpers.
//!
//! The trait takes the same `PreprocessedRequest` / `LLMEngineOutput` types used
//! across preprocessing, routing, and the frontend — no separate data-shape
//! translation layer for Rust engines.
//!
//! Object-safety: every instance method takes `&self`. `Arc<dyn LLMEngine>` is
//! the handle `Worker` drives the lifecycle through.

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use tokio::sync::watch;

use crate::error::DynamoError;

pub use dynamo_llm::kv_router::publisher::KvEventPublisher;
pub use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
pub use dynamo_llm::protocols::common::preprocessor::{
    BootstrapInfo, PrefillResult, PreprocessedRequest,
};
pub use dynamo_llm::protocols::common::{
    FinishReason, OutputOptions, SamplingOptions, StopConditions,
};
pub use dynamo_protocols::types::CompletionUsage;
pub use dynamo_runtime::engine::AsyncEngineContext;

/// Per-request handle wrapping the runtime context. `Deref`s to
/// `dyn AsyncEngineContext` so engine code uses it transparently.
pub struct GenerateContext {
    inner: Arc<dyn AsyncEngineContext>,
    /// Decode-mode first-token signal. `Some` only on decode-mode requests;
    /// `None` otherwise.
    first_token: Option<watch::Sender<bool>>,
}

impl GenerateContext {
    pub fn new(
        inner: Arc<dyn AsyncEngineContext>,
        first_token: Option<watch::Sender<bool>>,
    ) -> Self {
        Self { inner, first_token }
    }

    /// Clone the underlying runtime context Arc — for spawned tasks
    /// outliving `generate`'s scope.
    pub fn inner_arc(&self) -> Arc<dyn AsyncEngineContext> {
        self.inner.clone()
    }

    /// Fire the first-token signal. Idempotent; no-op on non-decode
    /// requests. Engines normally don't need this — the framework
    /// auto-fires on the first non-empty chunk. Use only when first-token
    /// is observable via a side channel before the main stream yields.
    pub fn notify_first_token(&self) {
        if let Some(tx) = &self.first_token {
            let _ = tx.send(true);
        }
    }

    /// Framework-internal: borrow the underlying Sender for cross-boundary
    /// threading (PyO3 mirrors this handle into Python's `Context` so
    /// `notify_first_token()` fires the same signal). Rust engines should
    /// call [`notify_first_token`](Self::notify_first_token) instead.
    pub fn first_token_sender(&self) -> Option<&watch::Sender<bool>> {
        self.first_token.as_ref()
    }
}

impl Deref for GenerateContext {
    type Target = dyn AsyncEngineContext;
    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

/// Registration metadata returned by [`LLMEngine::start`].
///
/// `Worker` consumes this to build a `ModelDeploymentCard` and register the
/// model with discovery. `None` on an optional field means "don't advertise":
/// the router sees no value and falls back to round-robin (for scheduling
/// hints) or its configured defaults. Engines without a traditional KV cache
/// can leave `kv_cache_block_size` and `total_kv_blocks` unset.
#[derive(Clone, Debug, Default)]
pub struct EngineConfig {
    /// Canonical model identifier (e.g. HF repo name).
    pub model: String,
    /// Public-facing model name advertised to clients. Defaults to `model`.
    pub served_model_name: Option<String>,
    /// Maximum context length the engine supports, in tokens.
    pub context_length: Option<u32>,
    /// KV cache block size, in tokens. Used by KV-aware routing. `None`
    /// means the engine has no block-structured KV cache; KV-aware routing
    /// falls back to round-robin for this backend.
    pub kv_cache_block_size: Option<u32>,
    /// Total number of KV cache blocks available to the engine. `None`
    /// means "not advertised"; the planner treats the backend as having
    /// no KV-capacity hint.
    pub total_kv_blocks: Option<u64>,
    /// Maximum number of concurrent in-flight sequences.
    pub max_num_seqs: Option<u64>,
    /// Maximum tokens the engine will process in a single batched step.
    pub max_num_batched_tokens: Option<u64>,
    /// Number of data-parallel ranks this worker hosts. Defaults to 1.
    /// The router uses this to enumerate per-rank load when scoring.
    pub data_parallel_size: Option<u32>,
    /// Global index of the first DP rank this worker hosts. Defaults to 0.
    /// Non-zero only under multi-worker DP layouts where each worker owns a
    /// sub-range (e.g. vLLM hybrid/external LB, SGLang DP-attention across
    /// multiple nodes). The router enumerates ranks
    /// `[data_parallel_start_rank, data_parallel_start_rank + data_parallel_size)`.
    pub data_parallel_start_rank: Option<u32>,
    /// Bootstrap host this prefill worker advertises to decode peers.
    ///
    /// Only meaningful for backends with a Dynamo-level host/port
    /// handshake (today: SGLang). Backends whose KV transport is
    /// internal — TRT-LLM uses TRT-LLM's transceiver, vLLM uses vLLM's
    /// `NixlConnector` — should leave this `None`.
    ///
    /// Engines that do use it set this in `start()` after the engine
    /// has resolved its bootstrap address (SGLang reads
    /// `tokenizer_manager.server_args.disaggregation_bootstrap_port`).
    /// When both `bootstrap_host` and `bootstrap_port` are `Some`,
    /// `Worker` publishes them via
    /// `ModelRuntimeConfig::disaggregated_endpoint` so the frontend's
    /// `PrefillRouter` can take its optimised "Bootstrap path" (route
    /// decode concurrent with prefill instead of waiting for prefill
    /// to drain).
    pub bootstrap_host: Option<String>,
    /// Bootstrap port for disaggregated KV transfer. See `bootstrap_host`.
    pub bootstrap_port: Option<u16>,
    /// Engine-specific metadata copied into `ModelRuntimeConfig.runtime_data`.
    pub runtime_data: HashMap<String, serde_json::Value>,
}

/// Inference engine trait.
///
/// Lifecycle:
///   1. Construct the engine (typically via a backend-specific `from_args`).
///   2. `start()` — start the engine, return `EngineConfig` metadata.
///   3. `generate()` — called for each request (concurrent calls expected).
///   4. `abort()` — called when a request is cancelled (optional, default no-op).
///   5. `cleanup()` — called once on shutdown, release all resources.
#[async_trait]
pub trait LLMEngine: Send + Sync + 'static {
    /// Start the engine and return registration metadata.
    ///
    /// After this returns, the engine MUST be ready to accept `generate()`
    /// calls. `Worker` will register the model and begin serving immediately.
    /// Use interior mutability for any state allocated here.
    ///
    /// `worker_id` is an opaque, runtime-allocated unique identifier for
    /// this worker. It is stable from `start()` onward for the worker's
    /// lifetime and unique across replicas in the cluster. Engines that
    /// need a per-worker key for cluster-wide bookkeeping (e.g. TRT-LLM's
    /// 10-bit `disagg_machine_id` snowflake field) should derive it from
    /// this value rather than hashing host/pid or asking operators for a
    /// CLI override. The internal mechanism (discovery instance ID) is
    /// not part of the contract — engines should treat it as opaque.
    ///
    /// `start()` is async and may take minutes for real backends (e.g.
    /// compiling a model graph on an accelerator). Emit
    /// `tracing::info!` checkpoints so operators see progress — this
    /// call is otherwise a silent window between process launch and
    /// endpoint serving.
    async fn start(&self, worker_id: u64) -> Result<EngineConfig, DynamoError>;

    /// Yield streaming response chunks for a single request.
    ///
    /// Called concurrently for multiple in-flight requests. The returned
    /// stream MUST poll `ctx.is_stopped()` between yields; on cancellation,
    /// emit a terminal `Ok(chunk)` with `FinishReason::Cancelled`.
    ///
    /// Stream item: `Result<LLMEngineOutput, DynamoError>`.
    ///   * `Ok(chunk)` carries normal output. Exactly one terminal `Ok`
    ///     chunk (one with `finish_reason` set) must be the last item
    ///     yielded, and no items may follow it.
    ///   * `Err(dynamo_err)` carries a typed mid-stream failure (e.g.
    ///     `BackendError::InvalidArgument`). It is itself terminal — the
    ///     framework forwards it as `Annotated::error` and stops polling
    ///     the stream. Use this instead of yielding an `Ok` chunk with
    ///     `FinishReason::Error` when you want the typed `BackendError`
    ///     variant preserved end-to-end.
    ///
    /// `completion_usage` on the terminal is optional but recommended —
    /// the frontend aggregates it when present. In debug builds, the
    /// framework wraps the stream in a validator that panics on contract
    /// violations.
    ///
    /// The returned stream is `'static`: clone or move any state from
    /// `&self` or `request` into the stream body before constructing it.
    /// Use [`chunk::token`] for non-terminal chunks and
    /// [`LLMEngineOutput::cancelled`] / `::stop` / `::length` / `::error`
    /// for terminal chunks (combine with [`LLMEngineOutputExt`] for
    /// fluent field setting).
    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError>;

    /// Abort an in-flight request (optional, default no-op).
    ///
    /// Called by the framework only when `ctx.stopped()` or `ctx.killed()`
    /// fires — i.e. when the client or operator explicitly cancels. It is
    /// NOT called when the response stream is simply dropped (e.g. TCP
    /// reset, consumer-side timeout without cancellation).
    ///
    /// For cleanup that must happen on ANY drop path (releasing an
    /// accelerator slot, freeing a request handle), put the release logic
    /// inside the `generate` stream body using RAII — a guard whose
    /// `Drop` runs when the stream is dropped, however that happens. Use
    /// `abort` only for out-of-band notifications (e.g. telling a remote
    /// scheduler to cancel compute early).
    async fn abort(&self, _ctx: Arc<dyn AsyncEngineContext>) {}

    /// Drain in-flight engine work before shutdown (optional, default no-op).
    ///
    /// Called once during graceful shutdown after the discovery unregister
    /// + grace-period sleep, but before [`cleanup`](LLMEngine::cleanup).
    /// Use it for backend-side draining that must complete while the
    /// distributed runtime (NATS / etcd) is still alive — e.g. waiting for
    /// in-flight NIXL KV transfers on prefill workers (issue #7319), so
    /// downstream decode workers don't observe a use-after-free on freed
    /// GPU memory.
    ///
    /// Failures are logged and swallowed; shutdown proceeds regardless.
    async fn drain(&self) -> Result<(), DynamoError> {
        Ok(())
    }

    /// Release all engine resources. Called exactly once.
    ///
    /// `Worker` guarantees:
    ///
    /// * `cleanup` runs after [`start`](LLMEngine::start) succeeded
    ///   *and* on shutdown — the common case.
    /// * `cleanup` also runs after `start` raised, on the partial
    ///   state the engine may have allocated before failing (inner
    ///   LLM handle, sockets, background tasks). Implementations
    ///   **must** be null-safe: guard each resource with an `is
    ///   None` / `Option::is_some` check so a partially constructed
    ///   engine can be released without panic.
    /// * `cleanup` is **not** called when `start` was never invoked
    ///   (pre-start shutdown via SIGTERM during distributed runtime
    ///   construction). Engines whose constructors allocate
    ///   resources must release them via `Drop` rather than rely on
    ///   `cleanup`.
    ///
    /// `cleanup` must also be idempotent: a second call after a
    /// successful first call must return `Ok(())` without re-entering
    /// teardown (NCCL groups and similar fail noisily on double-free).
    async fn cleanup(&self) -> Result<(), DynamoError>;

    /// KV event sources advertised by this engine, one per dp_rank.
    /// Empty by default (engine opts out of KV-aware routing).
    async fn kv_event_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
        Ok(Vec::new())
    }

    /// Metrics snapshot sources, one per dp_rank. Empty by default.
    async fn metrics_sources(&self) -> Result<Vec<MetricsSource>, DynamoError> {
        Ok(Vec::new())
    }
}

/// Invoked once with a freshly-built publisher; engine drives `publish`
/// from its own thread thereafter.
pub type OnPublisherReady =
    Box<dyn FnOnce(Arc<KvEventPublisher>) -> Result<(), DynamoError> + Send + 'static>;

/// Worker reads this on a fixed interval. MUST be a cheap member-field read.
pub type SnapshotFn = Arc<dyn Fn() -> Option<Metrics> + Send + Sync>;

/// KV event source descriptor. Two flavors: subscribe to an engine-provided
/// ZMQ PUB, or hand a publisher to the engine and let it drive `publish`
/// from its own thread (for engines whose event API blocks the caller).
pub enum KvEventSource {
    Zmq {
        endpoint: String,
        topic: String,
        dp_rank: u32,
    },
    Push {
        on_ready: OnPublisherReady,
        dp_rank: u32,
    },
}

impl KvEventSource {
    /// Data-parallel rank this source publishes for.
    pub fn dp_rank(&self) -> u32 {
        match self {
            KvEventSource::Zmq { dp_rank, .. } | KvEventSource::Push { dp_rank, .. } => *dp_rank,
        }
    }
}

/// One metrics-snapshot source per data-parallel rank.
pub struct MetricsSource {
    /// Worker polls this on a fixed interval. MUST be a cheap member-field
    /// read — engine-internal calls land in the 10s of ms and stall the
    /// publish loop. Conformance kit enforces a 1 ms ceiling.
    pub snapshot: SnapshotFn,
    pub dp_rank: u32,
}

/// Worker-level metrics snapshot consumed by the KV router.
///
/// `kv_used_blocks` is the primary load signal the router scores against.
/// `Option` because the engine may not have a snapshot yet on the first few
/// ticks after `start()`; `Worker` skips the publish call in that case.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Metrics {
    /// Number of KV blocks currently occupied across all in-flight requests.
    pub kv_used_blocks: Option<u64>,
}

/// Non-terminal chunk constructor. Terminal chunks come from upstream
/// [`LLMEngineOutput::cancelled`] / `::stop` / `::length` / `::error`.
pub mod chunk {
    use super::LLMEngineOutput;

    /// Non-terminal chunk carrying a single token.
    pub fn token(id: u32) -> LLMEngineOutput {
        LLMEngineOutput {
            token_ids: vec![id],
            ..Default::default()
        }
    }
}

/// Fluent setters for [`LLMEngineOutput`] — combine with upstream
/// constructors (`LLMEngineOutput::length()`, `::cancelled()`, etc.) to
/// avoid the `let mut output = ...; output.field = ...;` pattern.
///
/// ```ignore
/// use dynamo_backend_common::{LLMEngineOutput, LLMEngineOutputExt, usage};
///
/// yield LLMEngineOutput::length()
///     .with_tokens(vec![final_id])
///     .with_usage(usage(prompt_len, n));
/// ```
pub trait LLMEngineOutputExt: Sized {
    /// Replace `token_ids`.
    fn with_tokens(self, tokens: Vec<u32>) -> Self;
    /// Attach usage stats.
    fn with_usage(self, usage: CompletionUsage) -> Self;
}

impl LLMEngineOutputExt for LLMEngineOutput {
    fn with_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.token_ids = tokens;
        self
    }
    fn with_usage(mut self, usage: CompletionUsage) -> Self {
        self.completion_usage = Some(usage);
        self
    }
}

/// Build a [`CompletionUsage`] from prompt and completion counts.
/// `total_tokens` saturates on overflow (realistic LLM contexts are far
/// from `u32::MAX`).
pub fn usage(prompt_tokens: u32, completion_tokens: u32) -> CompletionUsage {
    CompletionUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
        prompt_tokens_details: None,
        completion_tokens_details: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_token_sets_only_token_ids() {
        let c = chunk::token(42);
        assert_eq!(c.token_ids, vec![42]);
        assert!(c.finish_reason.is_none());
        assert!(c.completion_usage.is_none());
    }

    #[test]
    fn ext_with_tokens_and_with_usage() {
        let terminal = LLMEngineOutput::length()
            .with_tokens(vec![1, 2, 3])
            .with_usage(usage(10, 3));
        assert_eq!(terminal.token_ids, vec![1, 2, 3]);
        assert!(matches!(terminal.finish_reason, Some(FinishReason::Length)));
        assert_eq!(terminal.completion_usage.unwrap().total_tokens, 13);
    }

    #[test]
    fn usage_sums_totals() {
        let u = usage(7, 11);
        assert_eq!(u.total_tokens, 18);
    }

    #[test]
    fn usage_saturates_on_overflow() {
        let u = usage(u32::MAX, 10);
        assert_eq!(u.total_tokens, u32::MAX);
    }
}
