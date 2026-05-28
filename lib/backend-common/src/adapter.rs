// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bridges [`LLMEngine`] (the author-facing trait) to [`AsyncEngine`]
//! (the trait `Ingress::for_engine` consumes).
//!
//! Decode-mode disagg defers `engine.abort()` until the first chunk to
//! avoid orphaning the prefill peer's NIXL KV transfer.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use futures::StreamExt;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::disagg::DisaggregationMode;
use crate::engine::{GenerateContext, LLMEngine};

/// Cancels its token on Drop so the monitor task exits cleanly when the
/// response stream is gone.
struct CancelMonitorGuard {
    drop_token: CancellationToken,
}

impl Drop for CancelMonitorGuard {
    fn drop(&mut self) {
        self.drop_token.cancel();
    }
}

pub(crate) struct EngineAdapter {
    engine: Arc<dyn LLMEngine>,
    mode: DisaggregationMode,
}

impl EngineAdapter {
    pub(crate) fn new(engine: Arc<dyn LLMEngine>, mode: DisaggregationMode) -> Self {
        Self { engine, mode }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for EngineAdapter
{
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (request, handle) = input.into_parts();
        let ctx: Arc<dyn AsyncEngineContext> = handle.context();

        // Decode workers defer engine.abort() until first-token to protect
        // in-flight NIXL transfers. The Sender goes to the engine (via
        // GenerateContext + the stream wrapper's auto-fire); the Receiver
        // gates the monitor's abort call.
        let (ft_tx, mut ft_rx) = if self.mode.is_decode() {
            let (tx, rx) = watch::channel(false);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        let gen_ctx = GenerateContext::new(ctx.clone(), ft_tx.clone());
        let chunks = self
            .engine
            .generate(request, gen_ctx)
            .await
            .map_err(Error::from)?;

        let drop_token = CancellationToken::new();
        let monitor_token = drop_token.clone();
        let abort_engine = self.engine.clone();
        let abort_ctx = ctx.clone();
        tokio::spawn(async move {
            // Wait for cancellation; drop_token arm = natural completion, no abort.
            let cancelled = tokio::select! {
                _ = abort_ctx.stopped() => {
                    tracing::debug!(request_id = abort_ctx.id(), "cancellation observed (stopped)");
                    true
                }
                _ = abort_ctx.killed() => {
                    tracing::debug!(request_id = abort_ctx.id(), "cancellation observed (killed)");
                    true
                }
                _ = monitor_token.cancelled() => false,
            };
            if !cancelled {
                return;
            }
            // `biased`: if first-token AND drop both fire in the same cycle,
            // prefer first-token — the request reached the abortable state.
            // `wait_for` `Err(Closed)` (all senders dropped) means the
            // request was torn down before first-token; treat as drop.
            if let Some(rx) = &mut ft_rx
                && !*rx.borrow()
            {
                tracing::debug!(
                    request_id = abort_ctx.id(),
                    "deferring engine.abort() until first-token observed"
                );
                tokio::select! {
                    biased;
                    res = rx.wait_for(|v| *v) => {
                        if res.is_err() {
                            return;
                        }
                    }
                    _ = monitor_token.cancelled() => return,
                }
            }
            abort_engine.abort(abort_ctx).await;
        });
        let guard = CancelMonitorGuard { drop_token };

        #[cfg(debug_assertions)]
        let chunks = crate::validate::wrap(chunks);

        let stream_ctx = ctx.clone();
        let mapped = async_stream::stream! {
            let _guard = guard;
            let mut inner = chunks;
            let mut chunk_count: usize = 0;
            let mut signalled = false;
            while let Some(item) = inner.next().await {
                chunk_count += 1;
                match item {
                    Ok(chunk) => {
                        // First non-empty chunk releases the deferred abort.
                        // Token-less chunks (SGLang's bootstrap handshake) don't count.
                        if !signalled
                            && !chunk.token_ids.is_empty()
                            && let Some(tx) = &ft_tx
                        {
                            // Receiver is held by the monitor task; send only
                            // fails if it panicked, in which case the abort is
                            // already moot.
                            let _ = tx.send(true);
                            signalled = true;
                        }
                        let is_terminal = chunk.finish_reason.is_some();
                        yield Annotated::from_data(chunk);
                        if is_terminal {
                            break;
                        }
                    }
                    Err(dynamo_err) => {
                        tracing::debug!(
                            request_id = stream_ctx.id(),
                            error = %dynamo_err,
                            "engine stream yielded typed error",
                        );
                        yield Annotated::from_err(dynamo_err);
                        break;
                    }
                }
            }
            tracing::debug!(
                request_id = stream_ctx.id(),
                chunks = chunk_count,
                cancelled = stream_ctx.is_stopped(),
                "stream complete"
            );
        };

        Ok(ResponseStream::new(Box::pin(mapped), ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineConfig, FinishReason, LLMEngineOutputExt, chunk, usage};
    use crate::error::{BackendError, DynamoError, ErrorType};
    use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_runtime::pipeline::Context;
    use futures::StreamExt;
    use futures::stream::BoxStream;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock engine: yields a canned list of chunks with a per-chunk delay, and
    /// records how many times `abort` is called.
    struct MockEngine {
        chunks: Vec<LLMEngineOutput>,
        per_chunk_delay_ms: u64,
        abort_calls: Arc<AtomicUsize>,
        setup_err: Option<fn() -> DynamoError>,
    }

    impl MockEngine {
        fn new(chunks: Vec<LLMEngineOutput>) -> (Arc<Self>, Arc<AtomicUsize>) {
            let counter = Arc::new(AtomicUsize::new(0));
            let eng = Arc::new(Self {
                chunks,
                per_chunk_delay_ms: 0,
                abort_calls: counter.clone(),
                setup_err: None,
            });
            (eng, counter)
        }
    }

    #[async_trait]
    impl LLMEngine for MockEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig::default())
        }

        async fn generate(
            &self,
            _request: PreprocessedRequest,
            context: GenerateContext,
        ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
            if let Some(make_err) = self.setup_err {
                return Err(make_err());
            }
            let chunks = self.chunks.clone();
            let delay_ms = self.per_chunk_delay_ms;
            let ctx = context.inner_arc();
            Ok(Box::pin(async_stream::stream! {
                for c in chunks {
                    if delay_ms > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    }
                    if ctx.is_stopped() { break; }
                    yield Ok(c);
                }
            }))
        }

        async fn abort(&self, _context: Arc<dyn AsyncEngineContext>) {
            self.abort_calls.fetch_add(1, Ordering::SeqCst);
        }

        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
    }

    fn make_request(token_ids: Vec<u32>) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("mock".to_string())
            .token_ids(token_ids)
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn adapter_maps_chunks_to_outputs() {
        let (engine, abort_ct) = MockEngine::new(vec![
            chunk::token(11),
            LLMEngineOutput::length()
                .with_tokens(vec![22])
                .with_usage(usage(3, 2)),
        ]);
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Aggregated);

        let input = Context::new(make_request(vec![1, 2, 3]));
        let stream = adapter.generate(input).await.unwrap();
        let collected: Vec<_> = stream.collect().await;

        assert_eq!(collected.len(), 2);
        let first = collected[0].data.as_ref().unwrap();
        assert_eq!(first.token_ids, vec![11]);
        assert!(first.finish_reason.is_none());

        let second = collected[1].data.as_ref().unwrap();
        assert_eq!(second.token_ids, vec![22]);
        assert!(matches!(second.finish_reason, Some(FinishReason::Length)));
        assert_eq!(
            abort_ct.load(Ordering::SeqCst),
            0,
            "clean completion must not call engine.abort"
        );
    }

    #[tokio::test]
    async fn adapter_cancellation_triggers_engine_abort() {
        let engine = Arc::new(MockEngine {
            chunks: (0..100).map(chunk::token).collect(),
            per_chunk_delay_ms: 20,
            abort_calls: Arc::new(AtomicUsize::new(0)),
            setup_err: None,
        });
        let abort_ct = engine.abort_calls.clone();
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Aggregated);

        let input: Context<PreprocessedRequest> = Context::new(make_request(vec![1]));
        let ctrl = input.context();
        let mut stream = adapter.generate(input).await.unwrap();

        // Read one chunk, then trigger cancellation.
        let _first = stream.next().await.expect("at least one chunk");
        ctrl.stop_generating();

        let drained = tokio::time::timeout(std::time::Duration::from_millis(500), async {
            while stream.next().await.is_some() {}
        })
        .await;
        assert!(
            drained.is_ok(),
            "stream did not terminate after cancellation"
        );

        // Give the monitor task time to schedule and call abort(). 100ms
        // leaves headroom under CI load without making the test slow.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert_eq!(
            abort_ct.load(Ordering::SeqCst),
            1,
            "engine.abort should be called exactly once on cancellation"
        );
    }

    #[tokio::test]
    async fn adapter_engine_setup_error_propagates() {
        let engine = Arc::new(MockEngine {
            chunks: vec![],
            per_chunk_delay_ms: 0,
            abort_calls: Arc::new(AtomicUsize::new(0)),
            setup_err: Some(|| {
                DynamoError::builder()
                    .error_type(ErrorType::Backend(BackendError::Unknown))
                    .message("init failed")
                    .build()
            }),
        });
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Aggregated);

        let input = Context::new(make_request(vec![1]));
        let err = adapter.generate(input).await.unwrap_err();
        assert!(err.to_string().contains("init failed"));
    }

    /// Engine that yields one regular chunk, then a terminal cancel chunk when
    /// `ctx.is_stopped()` becomes true. Verifies the adapter forwards the
    /// terminal to downstream rather than dropping it on the break.
    struct TerminalOnCancelEngine;

    #[async_trait]
    impl LLMEngine for TerminalOnCancelEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig::default())
        }
        async fn generate(
            &self,
            _request: PreprocessedRequest,
            ctx: GenerateContext,
        ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
            let ctx = ctx.inner_arc();
            Ok(Box::pin(async_stream::stream! {
                yield Ok(chunk::token(1));
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    if ctx.is_stopped() {
                        yield Ok(LLMEngineOutput::cancelled().with_usage(usage(3, 1)));
                        break;
                    }
                }
            }))
        }
        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn adapter_forwards_terminal_cancel_chunk_to_downstream() {
        let adapter = EngineAdapter::new(
            Arc::new(TerminalOnCancelEngine),
            DisaggregationMode::Aggregated,
        );
        let input: Context<PreprocessedRequest> = Context::new(make_request(vec![1]));
        let ctrl = input.context();
        let mut stream = adapter.generate(input).await.unwrap();

        let _first = stream.next().await.expect("first chunk");
        ctrl.stop_generating();

        let rest: Vec<_> = stream.collect().await;
        assert_eq!(
            rest.len(),
            1,
            "downstream must receive the engine's terminal cancel chunk"
        );
        let terminal = rest[0].data.as_ref().unwrap();
        assert!(matches!(
            terminal.finish_reason,
            Some(FinishReason::Cancelled)
        ));
    }

    #[tokio::test]
    async fn adapter_surfaces_typed_invalid_argument_error() {
        let engine = Arc::new(MockEngine {
            chunks: vec![],
            per_chunk_delay_ms: 0,
            abort_calls: Arc::new(AtomicUsize::new(0)),
            setup_err: Some(|| {
                DynamoError::builder()
                    .error_type(ErrorType::Backend(BackendError::InvalidArgument))
                    .message("bad param")
                    .build()
            }),
        });
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Aggregated);

        let input = Context::new(make_request(vec![1]));
        let err = adapter.generate(input).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("BackendInvalidArgument"), "got: {msg}");
        assert!(msg.contains("bad param"), "got: {msg}");
    }

    /// Engine that yields one chunk and then a typed `Err(DynamoError)`,
    /// proving the adapter forwards a mid-stream typed error as
    /// `Annotated::error` with the `BackendError` variant intact.
    struct TypedMidStreamErrEngine;

    #[async_trait]
    impl LLMEngine for TypedMidStreamErrEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig::default())
        }
        async fn generate(
            &self,
            _request: PreprocessedRequest,
            _ctx: GenerateContext,
        ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
            Ok(Box::pin(async_stream::stream! {
                yield Ok(chunk::token(1));
                yield Err(DynamoError::builder()
                    .error_type(ErrorType::Backend(BackendError::InvalidArgument))
                    .message("bad mid-stream")
                    .build());
            }))
        }
        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn adapter_forwards_typed_mid_stream_error_as_annotated_error() {
        let adapter = EngineAdapter::new(
            Arc::new(TypedMidStreamErrEngine),
            DisaggregationMode::Aggregated,
        );
        let input = Context::new(make_request(vec![1]));
        let mut stream = adapter.generate(input).await.unwrap();

        let first = stream.next().await.expect("first chunk");
        assert!(first.data.is_some(), "first item carries data");

        let err_item = stream.next().await.expect("typed error item");
        assert!(err_item.is_error(), "second item must be Annotated::error");
        let err = err_item.error.expect("typed DynamoError carried through");
        assert_eq!(
            err.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument),
            "typed BackendError variant must survive end-to-end"
        );
        assert!(err.to_string().contains("bad mid-stream"));

        // No items after the typed error.
        assert!(stream.next().await.is_none());
    }

    // -------------------------------------------------------------------
    // Deferred-abort behaviour for decode-mode workers.
    // -------------------------------------------------------------------

    use tokio::sync::Notify;

    /// Engine whose `generate` parks on a barrier until the test releases
    /// it. Records how many times `abort()` was called so we can assert
    /// on timing.
    struct ParkedEngine {
        release: Arc<Notify>,
        abort_calls: Arc<AtomicUsize>,
    }

    impl ParkedEngine {
        fn new() -> (Arc<Self>, Arc<Notify>, Arc<AtomicUsize>) {
            let release = Arc::new(Notify::new());
            let abort_calls = Arc::new(AtomicUsize::new(0));
            (
                Arc::new(Self {
                    release: release.clone(),
                    abort_calls: abort_calls.clone(),
                }),
                release,
                abort_calls,
            )
        }
    }

    #[async_trait]
    impl LLMEngine for ParkedEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig::default())
        }
        async fn generate(
            &self,
            _request: PreprocessedRequest,
            _ctx: GenerateContext,
        ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
            let release = self.release.clone();
            Ok(Box::pin(async_stream::stream! {
                release.notified().await;
                yield Ok(chunk::token(42));
                yield Ok(LLMEngineOutput::length().with_usage(usage(1, 1)));
            }))
        }
        async fn abort(&self, _ctx: Arc<dyn AsyncEngineContext>) {
            self.abort_calls.fetch_add(1, Ordering::SeqCst);
        }
        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
    }

    /// Cancellation before the first chunk must NOT fire `engine.abort()`
    /// until the first chunk lands — early aborts orphan the prefill peer's
    /// NIXL transfer.
    #[tokio::test(start_paused = true)]
    async fn decode_defers_abort_until_first_chunk() {
        let (engine, release, abort_calls) = ParkedEngine::new();
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Decode);

        let input: Context<PreprocessedRequest> = Context::new(make_request(vec![1]));
        let ctrl = input.context();
        let mut stream = adapter.generate(input).await.unwrap();

        ctrl.stop_generating();
        tokio::time::advance(std::time::Duration::from_millis(100)).await;
        assert_eq!(
            abort_calls.load(Ordering::SeqCst),
            0,
            "decode worker must not call engine.abort before first-token"
        );

        release.notify_one();
        let _ = stream.next().await;
        while stream.next().await.is_some() {}

        tokio::time::advance(std::time::Duration::from_millis(100)).await;
        assert_eq!(
            abort_calls.load(Ordering::SeqCst),
            1,
            "abort must fire exactly once after first-token observed"
        );
    }

    /// Aggregated-mode fires `engine.abort()` immediately — only decode
    /// opts into deferral.
    #[tokio::test(start_paused = true)]
    async fn aggregated_fires_abort_immediately() {
        let (engine, release, abort_calls) = ParkedEngine::new();
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Aggregated);

        let input: Context<PreprocessedRequest> = Context::new(make_request(vec![1]));
        let ctrl = input.context();
        let mut stream = adapter.generate(input).await.unwrap();

        ctrl.stop_generating();
        tokio::time::advance(std::time::Duration::from_millis(100)).await;
        assert_eq!(
            abort_calls.load(Ordering::SeqCst),
            1,
            "aggregated worker must fire abort immediately on cancellation"
        );

        release.notify_one();
        while stream.next().await.is_some() {}
    }

    /// Engine that fires the side-channel first-token notify on entry and
    /// then parks, modelling an engine (e.g. TRT-LLM reading an aqueue) that
    /// observes first-token before the main `generate` stream yields anything.
    struct SideChannelEngine {
        release: Arc<Notify>,
        abort_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl LLMEngine for SideChannelEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig::default())
        }
        async fn generate(
            &self,
            _request: PreprocessedRequest,
            ctx: GenerateContext,
        ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
            ctx.notify_first_token();
            let release = self.release.clone();
            Ok(Box::pin(async_stream::stream! {
                release.notified().await;
                yield Ok(LLMEngineOutput::length().with_usage(usage(1, 0)));
            }))
        }
        async fn abort(&self, _ctx: Arc<dyn AsyncEngineContext>) {
            self.abort_calls.fetch_add(1, Ordering::SeqCst);
        }
        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
    }

    /// Engine firing `ctx.notify_first_token()` from a side channel
    /// releases the deferred abort even when no chunk has flowed.
    #[tokio::test(start_paused = true)]
    async fn decode_side_channel_hook_releases_deferred_abort() {
        let release = Arc::new(Notify::new());
        let abort_calls = Arc::new(AtomicUsize::new(0));
        let engine = Arc::new(SideChannelEngine {
            release: release.clone(),
            abort_calls: abort_calls.clone(),
        });
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Decode);

        let input: Context<PreprocessedRequest> = Context::new(make_request(vec![1]));
        let ctrl = input.context();
        let mut stream = adapter.generate(input).await.unwrap();

        ctrl.stop_generating();
        tokio::time::advance(std::time::Duration::from_millis(100)).await;
        assert_eq!(
            abort_calls.load(Ordering::SeqCst),
            1,
            "side-channel notify must release the deferred abort"
        );

        release.notify_one();
        while stream.next().await.is_some() {}
    }

    /// Stream drop before first-token must NOT fire abort. The monitor's
    /// `drop_token` arm exits the deferred wait without calling abort.
    #[tokio::test(start_paused = true)]
    async fn decode_stream_drop_without_first_token_does_not_abort() {
        let (engine, _release, abort_calls) = ParkedEngine::new();
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Decode);

        let input: Context<PreprocessedRequest> = Context::new(make_request(vec![1]));
        let ctrl = input.context();
        let stream = adapter.generate(input).await.unwrap();

        ctrl.stop_generating();
        drop(stream);

        tokio::time::advance(std::time::Duration::from_millis(100)).await;
        assert_eq!(
            abort_calls.load(Ordering::SeqCst),
            0,
            "stream drop before first-token must not fire engine.abort"
        );
    }
}
