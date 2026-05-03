// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mocker backend — wraps `dynamo-mocker`'s scheduler core in the
//! [`LLMEngine`] contract.
//!
//! A single scheduler drives every in-flight request: each forward
//! pass emits one token per active sequence, paced by the mocker's
//! performance model. That scheduling contract is full-fidelity — a
//! real engine behaves the same way from the framework's perspective
//! — which makes this example a good stand-in for AIPerf / pipeline
//! load tests.
//!
//! What is *not* modelled here: KV-event publishing, forward-pass
//! metrics, disaggregated-serving bootstrap. Those live in
//! `lib/llm/src/mocker.rs` and need a `Component` handle that
//! `LLMEngine::start()` does not expose today; they will land with a
//! future trait extension. Use the planner/router-facing mocker if
//! you need the event plane.

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, CommonArgs, DynamoError, EngineConfig, ErrorType, LLMEngine,
    LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest, WorkerConfig, chunk, usage,
};
use dynamo_mocker::common::protocols::{
    DirectRequest, EngineType, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use dynamo_mocker::engine::create_engine;
use dynamo_mocker::scheduler::SchedulerHandle;
use futures::stream::BoxStream;
use rand::Rng;
use tokio::sync::{OnceCell, mpsc};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Single-rank example — the scheduler always runs at dp_rank 0.
const DP_RANK: u32 = 0;

/// Fallback when the client does not set `stop_conditions.max_tokens`. The
/// conformance suite exercises this path (single-generate uses `None`).
const DEFAULT_MAX_TOKENS: usize = 16;

#[derive(clap::Parser, Debug)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    about = "Dynamo mocker backend — serves the dynamo-mocker scheduler through the backend-common LLMEngine trait."
)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    /// Friendly model name advertised to the frontend.
    #[arg(long, default_value = "mocker-model")]
    model_name: String,

    /// HF repo or local path providing tokenizer + chat template. Leave
    /// empty for name-only registration (no templating).
    #[arg(long, default_value = "")]
    model_path: String,

    /// KV cache block size in tokens. 0 lets the mocker pick its default
    /// (64 for vLLM).
    #[arg(long, default_value_t = 0)]
    block_size: usize,

    /// Total KV cache blocks the mocker can allocate.
    #[arg(long, default_value_t = 16384)]
    num_gpu_blocks: usize,

    /// Maximum concurrent sequences the scheduler will admit.
    #[arg(long, default_value_t = 256)]
    max_num_seqs: usize,

    /// Maximum tokens the scheduler will process in a single forward pass.
    #[arg(long, default_value_t = 8192)]
    max_num_batched_tokens: usize,

    /// Wall-clock speedup multiplier relative to the perf model. Higher =
    /// faster simulated forward passes.
    #[arg(long, default_value_t = 1.0)]
    speedup_ratio: f64,

    /// Context length (tokens) advertised to clients. Bump this for
    /// long-context workloads; the effective ceiling is
    /// `num_gpu_blocks * block_size`.
    #[arg(long, default_value_t = 8192)]
    context_length: u32,
}

fn build_engine_args(args: &Args) -> Result<MockEngineArgs, DynamoError> {
    let built = MockEngineArgs::builder()
        .engine_type(EngineType::Vllm)
        .block_size(args.block_size)
        .num_gpu_blocks(args.num_gpu_blocks)
        .max_num_seqs(Some(args.max_num_seqs))
        .max_num_batched_tokens(Some(args.max_num_batched_tokens))
        .speedup_ratio(args.speedup_ratio)
        .dp_size(1)
        .build()
        .map_err(|e| invalid_arg(format!("mocker args: {e}")))?;
    built
        .normalized()
        .map_err(|e| invalid_arg(format!("mocker args: {e}")))
}

/// Per-request state held by the engine for as long as the request is
/// in flight. `tx` receives per-forward-pass signals from the shared
/// dispatcher; `ctx` lets `cleanup()` signal the stream to terminate
/// cleanly (yielding a `Cancelled` terminal) instead of racing the
/// channel-drop path.
struct ActiveEntry {
    tx: mpsc::UnboundedSender<OutputSignal>,
    ctx: Arc<dyn AsyncEngineContext>,
}

/// Removes the request's entry from the active-requests map on any stream
/// exit path — natural completion, cancellation, or an early drop.
struct ActiveRequestGuard {
    uuid: Uuid,
    active: Arc<DashMap<Uuid, ActiveEntry>>,
}

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        self.active.remove(&self.uuid);
    }
}

pub struct MockerBackend {
    model_name: String,
    context_length: u32,
    engine_args: MockEngineArgs,
    cancel: CancellationToken,
    active: Arc<DashMap<Uuid, ActiveEntry>>,
    request_tx: OnceCell<mpsc::UnboundedSender<DirectRequest>>,
    /// Held for its lifetime side-effect only: the scheduler's internal
    /// `CancelGuard` fires on drop, so keeping the handle alive keeps
    /// the scheduler tasks running for the engine's lifetime.
    #[allow(dead_code)]
    scheduler: OnceCell<Box<dyn SchedulerHandle>>,
}

impl MockerBackend {
    fn new(model_name: String, context_length: u32, engine_args: MockEngineArgs) -> Self {
        MockerBackend {
            model_name,
            context_length,
            engine_args,
            cancel: CancellationToken::new(),
            active: Arc::new(DashMap::new()),
            request_tx: OnceCell::new(),
            scheduler: OnceCell::new(),
        }
    }

    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = match argv {
            Some(a) => <Args as clap::Parser>::try_parse_from(a),
            None => <Args as clap::Parser>::try_parse(),
        }
        .map_err(|e| invalid_arg(e.to_string()))?;

        let engine_args = build_engine_args(&args)?;
        let engine = Self::new(args.model_name.clone(), args.context_length, engine_args);
        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            model_name: args.model_path,
            served_model_name: Some(args.model_name),
            ..Default::default()
        };
        Ok((engine, config))
    }
}

#[async_trait]
impl LLMEngine for MockerBackend {
    async fn start(&self) -> Result<EngineConfig, DynamoError> {
        // Reject double-start BEFORE calling `create_engine`. The
        // scheduler's internal `CancelGuard` fires `self.cancel.cancel()`
        // on drop, so if we create a second scheduler and then error
        // out, dropping it would kill the first one too.
        if self.scheduler.initialized() {
            return Err(engine_shutdown("mocker backend already started"));
        }

        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
        let scheduler = create_engine(
            self.engine_args.clone(),
            DP_RANK,
            Some(output_tx),
            KvEventPublishers::default(),
            Some(self.cancel.clone()),
            FpmPublisher::default(),
        );

        // The `initialized()` check + these `set()` calls are not atomic,
        // so concurrent `start()` callers could both pass the check and
        // race here. The Worker framework calls `start()` exactly once,
        // which is the contract we rely on. We still map `set()` errors
        // to a well-typed error instead of unwrapping.
        self.request_tx
            .set(scheduler.request_sender())
            .map_err(|_| engine_shutdown("mocker backend already started"))?;
        self.scheduler
            .set(scheduler)
            .map_err(|_| engine_shutdown("mocker backend already started"))?;

        // Fan-out: the scheduler emits one batch per forward pass; each
        // signal carries the per-request uuid so we route it to the
        // matching stream. `biased` ensures shutdown wins over pending
        // batches so we stop forwarding once cancel fires.
        let active = self.active.clone();
        let cancel = self.cancel.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => break,
                    batch = output_rx.recv() => {
                        let Some(batch) = batch else { break };
                        for signal in batch {
                            if let Some(entry) = active.get(&signal.uuid) {
                                let _ = entry.tx.send(signal);
                            }
                        }
                    }
                }
            }
        });

        tracing::info!(
            model = %self.model_name,
            num_gpu_blocks = self.engine_args.num_gpu_blocks,
            block_size = self.engine_args.block_size,
            speedup_ratio = self.engine_args.speedup_ratio,
            "mocker backend started"
        );

        Ok(EngineConfig {
            model: self.model_name.clone(),
            served_model_name: Some(self.model_name.clone()),
            context_length: Some(self.context_length),
            kv_cache_block_size: Some(self.engine_args.block_size as u32),
            total_kv_blocks: Some(self.engine_args.num_gpu_blocks as u64),
            max_num_seqs: self.engine_args.max_num_seqs.map(|v| v as u64),
            max_num_batched_tokens: self.engine_args.max_num_batched_tokens.map(|v| v as u64),
        })
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, LLMEngineOutput>, DynamoError> {
        let request_tx = self
            .request_tx
            .get()
            .ok_or_else(|| engine_shutdown("generate called before start"))?
            .clone();

        let uuid = ctx.id().parse().unwrap_or_else(|_| Uuid::new_v4());
        let prompt_len = request.token_ids.len() as u32;
        let max_output_tokens = request
            .stop_conditions
            .max_tokens
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_MAX_TOKENS);

        // max_tokens == 0: nothing to generate, but still need a terminal.
        // Skip the scheduler round-trip entirely — the mocker protocol
        // requires max_output_tokens >= 1.
        if max_output_tokens == 0 {
            return Ok(Box::pin(async_stream::stream! {
                // Honour cancellation even on this fast path so the
                // terminal's finish_reason reflects what the client asked for.
                if ctx.is_stopped() {
                    yield LLMEngineOutput::cancelled().with_usage(usage(prompt_len, 0));
                } else {
                    yield LLMEngineOutput::length().with_usage(usage(prompt_len, 0));
                }
            }));
        }

        let direct = DirectRequest {
            tokens: request.token_ids.clone(),
            max_output_tokens,
            uuid: Some(uuid),
            dp_rank: DP_RANK,
            arrival_timestamp_ms: request.request_timestamp_ms,
        };

        let (tx, mut rx) = mpsc::unbounded_channel::<OutputSignal>();
        self.active.insert(
            uuid,
            ActiveEntry {
                tx,
                ctx: ctx.clone(),
            },
        );

        if request_tx.send(direct).is_err() {
            self.active.remove(&uuid);
            return Err(engine_shutdown("scheduler is not accepting requests"));
        }

        let guard = ActiveRequestGuard {
            uuid,
            active: self.active.clone(),
        };

        Ok(Box::pin(async_stream::stream! {
            let _guard = guard;
            let mut generated: u32 = 0;

            loop {
                // `biased` is load-bearing — don't reorder or remove.
                // Two reasons we need it:
                //   1. Prefer cancellation over a pending signal: if
                //      both arms are ready, yield `Cancelled` instead of
                //      one more token.
                //   2. Cleanup ordering: `cleanup()` fires
                //      `ctx.stop_generating()` and then lets `self.active`
                //      drop (closing `tx`). A stream polled after that
                //      sees both `ctx.stopped()` ready AND
                //      `rx.recv() -> None`; biased picks `stopped()`, so
                //      we yield `Cancelled` instead of the spurious
                //      "channel closed before completion" error.
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        yield LLMEngineOutput::cancelled()
                            .with_usage(usage(prompt_len, generated));
                        break;
                    }
                    maybe_signal = rx.recv() => {
                        let Some(signal) = maybe_signal else {
                            yield LLMEngineOutput::error(
                                "mocker backend: scheduler channel closed before completion".to_string(),
                            );
                            break;
                        };
                        generated += 1;
                        let token_id = rand::rng().random_range(1000..2000);

                        if signal.completed {
                            // Defensive: the scheduler sets `completed`
                            // exactly when `generated == max_output_tokens`.
                            // If it ever fires early, surface it instead of
                            // silently truncating.
                            if (generated as usize) < max_output_tokens {
                                yield LLMEngineOutput::error(format!(
                                    "mocker backend: scheduler signalled completion at {generated}/{max_output_tokens} tokens"
                                ));
                                break;
                            }
                            yield LLMEngineOutput::length()
                                .with_tokens(vec![token_id])
                                .with_usage(usage(prompt_len, generated));
                            break;
                        }
                        yield chunk::token(token_id);
                    }
                }
            }
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        // The stream body's `biased` select on `ctx.stopped()` drives
        // the user-visible cancellation — it yields `Cancelled` and the
        // request is removed from `self.active` via the guard's Drop.
        //
        // Scheduler-side: there is no abort API on `SchedulerHandle`
        // today, so the mocker scheduler will continue spending
        // simulated compute on this uuid until max_output_tokens.
        // Future cleanup will need an abort hook in `dynamo-mocker`.
        tracing::debug!(request_id = ctx.id(), "mocker backend: abort requested");
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        // Signal every in-flight stream to terminate via its own
        // `ctx.stopped()` path — each yields a `Cancelled` terminal and
        // the guard drops its active-requests entry. This avoids the
        // race where `active.clear()` would drop channels first and
        // streams would see `rx.recv() → None` as a "channel closed"
        // error instead of a clean cancellation.
        //
        // Collect contexts first, then drop the iterator before firing
        // `stop_generating()`. Holding `DashMap::iter()`'s read locks
        // while a stream tries to `active.remove()` (via its guard)
        // would briefly contend; this pattern sidesteps it.
        let ctxs: Vec<_> = self
            .active
            .iter()
            .map(|entry| entry.value().ctx.clone())
            .collect();
        for ctx in ctxs {
            ctx.stop_generating();
        }
        self.cancel.cancel();
        tracing::info!("mocker backend: cleanup invoked");
        Ok(())
    }
}

fn invalid_arg(msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(msg)
        .build()
}

fn engine_shutdown(msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::EngineShutdown))
        .message(msg)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;
    use dynamo_backend_common::{FinishReason, SamplingOptions, StopConditions};
    use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
    use futures::StreamExt;

    fn test_engine() -> MockerBackend {
        let args = Args::try_parse_from(["bin"]).unwrap();
        let engine_args = build_engine_args(&args).unwrap();
        MockerBackend::new(args.model_name, args.context_length, engine_args)
    }

    fn request(max_tokens: Option<u32>) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("mocker-model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions {
                max_tokens,
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(Default::default())
            .build()
            .unwrap()
    }

    #[test]
    fn args_parse_with_defaults() {
        let args = Args::try_parse_from(["bin"]).unwrap();
        assert_eq!(args.model_name, "mocker-model");
        assert_eq!(args.max_num_seqs, 256);
        assert_eq!(args.num_gpu_blocks, 16384);
        assert_eq!(args.context_length, 8192);
        assert_eq!(args.common.namespace, "dynamo");
    }

    #[test]
    fn build_engine_args_normalizes_block_size() {
        let args = Args::try_parse_from(["bin"]).unwrap();
        let engine_args = build_engine_args(&args).unwrap();
        // vLLM's default block size after normalization is 64.
        assert_eq!(engine_args.block_size, 64);
    }

    #[tokio::test]
    async fn start_returns_advertised_metadata() {
        let engine = test_engine();
        let cfg = engine.start().await.unwrap();
        assert_eq!(cfg.model, "mocker-model");
        assert_eq!(cfg.kv_cache_block_size, Some(64));
        assert_eq!(cfg.total_kv_blocks, Some(16384));
        assert_eq!(cfg.max_num_seqs, Some(256));
        assert_eq!(cfg.context_length, Some(8192));
        engine.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn double_start_errors_without_breaking_engine() {
        // Regression: a second `start()` used to cancel the live engine's
        // shared token (via the second scheduler's `CancelGuard` firing
        // on drop), leaving the engine unusable. The fix is to reject
        // the second call BEFORE creating a second scheduler.
        let engine = test_engine();
        engine.start().await.unwrap();

        let err = engine.start().await.unwrap_err();
        assert_eq!(
            err.error_type(),
            ErrorType::Backend(BackendError::EngineShutdown)
        );

        // The engine must still serve after the rejected double-start.
        let ctx = Context::new(());
        let stream = engine
            .generate(request(Some(2)), ctx.context())
            .await
            .expect("engine must still be usable after rejected double-start");
        let chunks: Vec<_> = stream.collect().await;
        let terminal = chunks.last().expect("at least a terminal");
        assert!(
            matches!(terminal.finish_reason, Some(FinishReason::Length)),
            "stream must still complete normally, got {:?}",
            terminal.finish_reason
        );

        engine.cleanup().await.unwrap();
    }

    #[test]
    fn from_args_produces_valid_config() {
        let (_engine, config) = MockerBackend::from_args(Some(vec![
            "bin".to_string(),
            "--model-name".to_string(),
            "custom-served".to_string(),
            "--model-path".to_string(),
            "org/repo".to_string(),
            "--namespace".to_string(),
            "ns".to_string(),
            "--component".to_string(),
            "comp".to_string(),
        ]))
        .unwrap();
        assert_eq!(config.namespace, "ns");
        assert_eq!(config.component, "comp");
        assert_eq!(config.model_name, "org/repo");
        assert_eq!(config.served_model_name.as_deref(), Some("custom-served"));
    }

    #[tokio::test]
    async fn generate_before_start_is_an_error() {
        let engine = test_engine();
        let ctx = Context::new(());
        let result = engine.generate(request(Some(1)), ctx.context()).await;
        let Err(err) = result else {
            panic!("expected generate() to fail before start()");
        };
        assert_eq!(
            err.error_type(),
            ErrorType::Backend(BackendError::EngineShutdown)
        );
    }

    #[tokio::test]
    async fn generate_with_zero_max_tokens_emits_empty_length_terminal() {
        let engine = test_engine();
        engine.start().await.unwrap();

        let ctx = Context::new(());
        let stream = engine
            .generate(request(Some(0)), ctx.context())
            .await
            .expect("stream");
        let chunks: Vec<_> = stream.collect().await;

        assert_eq!(chunks.len(), 1, "zero max_tokens should yield one terminal");
        assert!(chunks[0].token_ids.is_empty());
        assert!(matches!(
            chunks[0].finish_reason,
            Some(FinishReason::Length)
        ));
        let u = chunks[0].completion_usage.as_ref().unwrap();
        assert_eq!(u.prompt_tokens, 3);
        assert_eq!(u.completion_tokens, 0);

        engine.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn generate_with_zero_max_tokens_honours_cancellation() {
        // The zero-max-tokens fast path checks `ctx.is_stopped()` before
        // choosing its terminal reason. Lock that in so a future refactor
        // can't silently fall back to a Length terminal on cancelled
        // requests.
        let engine = test_engine();
        engine.start().await.unwrap();

        let ctx = Context::new(());
        let ctrl = ctx.context();
        ctrl.stop_generating();
        let stream = engine
            .generate(request(Some(0)), ctrl)
            .await
            .expect("stream");
        let chunks: Vec<_> = stream.collect().await;

        assert_eq!(chunks.len(), 1);
        assert!(matches!(
            chunks[0].finish_reason,
            Some(FinishReason::Cancelled)
        ));

        engine.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn generate_runs_to_completion() {
        let engine = test_engine();
        engine.start().await.unwrap();

        let ctx = Context::new(());
        let ctrl = ctx.context();
        let stream = engine
            .generate(request(Some(4)), ctrl)
            .await
            .expect("stream");
        let chunks: Vec<_> = stream.collect().await;

        assert!(!chunks.is_empty(), "expected at least one chunk");
        let terminal = chunks.last().unwrap();
        assert!(
            matches!(terminal.finish_reason, Some(FinishReason::Length)),
            "expected Length terminal, got {:?}",
            terminal.finish_reason
        );
        let u = terminal.completion_usage.as_ref().unwrap();
        assert_eq!(u.prompt_tokens, 3);
        assert_eq!(u.completion_tokens, 4);

        engine.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn generate_cancellation_yields_cancelled_chunk() {
        let engine = test_engine();
        engine.start().await.unwrap();

        let ctx = Context::new(());
        let ctrl = ctx.context();
        let stream = engine
            .generate(request(Some(10_000)), ctrl.clone())
            .await
            .expect("stream");
        ctrl.stop_generating();

        let chunks: Vec<_> = stream.collect().await;
        let terminal = chunks.last().expect("at least a terminal");
        assert!(
            matches!(terminal.finish_reason, Some(FinishReason::Cancelled)),
            "expected Cancelled terminal, got {:?}",
            terminal.finish_reason
        );

        engine.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn mocker_passes_conformance() {
        let engine = test_engine();
        dynamo_backend_common::testing::run_conformance(engine)
            .await
            .expect("mocker backend must satisfy conformance");
    }
}
