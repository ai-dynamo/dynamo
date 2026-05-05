// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bridges [`LLMEngine`] (the author-facing trait) to [`AsyncEngine`]
//! (the trait `Ingress::for_engine` consumes).
//!
//! The adapter is purely structural: destructure [`SingleIn`], forward to the
//! engine, wrap outputs in [`Annotated`]. No data-shape translation — authors
//! work with the same types the rest of the Rust pipeline uses.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use tokio::task::JoinHandle;

use crate::engine::LLMEngine;

/// Aborts the spawned cancellation-monitor task when the response stream
/// is dropped.
struct CancelMonitorGuard(JoinHandle<()>);

impl Drop for CancelMonitorGuard {
    fn drop(&mut self) {
        self.0.abort();
    }
}

pub(crate) struct EngineAdapter {
    engine: Arc<dyn LLMEngine>,
}

impl EngineAdapter {
    pub(crate) fn new(engine: Arc<dyn LLMEngine>) -> Self {
        Self { engine }
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

        let chunks = self
            .engine
            .generate(request, ctx.clone())
            .await
            .map_err(Error::from)?;

        // Cancellation monitor: awaits stop or kill, then notifies the engine.
        let abort_engine = self.engine.clone();
        let abort_ctx = ctx.clone();
        let monitor = tokio::spawn(async move {
            tokio::select! {
                _ = abort_ctx.stopped() => {
                    tracing::debug!(
                        request_id = abort_ctx.id(),
                        "cancellation observed (stopped)"
                    );
                }
                _ = abort_ctx.killed() => {
                    tracing::debug!(
                        request_id = abort_ctx.id(),
                        "cancellation observed (killed)"
                    );
                }
            }
            abort_engine.abort(abort_ctx.clone()).await;
        });
        let guard = CancelMonitorGuard(monitor);

        #[cfg(debug_assertions)]
        let chunks = crate::validate::wrap(chunks);

        let ctx_for_stream = ctx.clone();
        let mapped = async_stream::stream! {
            let _guard = guard;
            let mut inner = chunks;
            let mut chunk_count: usize = 0;
            let mut cancelled = false;
            while let Some(chunk) = inner.next().await {
                chunk_count += 1;
                // Record cancellation whenever the context reports it, but
                // don't exit on that alone — a well-behaved engine will
                // respond to `stop_generating()` by emitting its own
                // terminal `Cancelled` chunk, and downstream must see it.
                // We exit only on the engine's terminal chunk so the
                // engine's final frame is always forwarded. Broken
                // engines that never emit a terminal end the loop when
                // their inner stream ends (the conformance kit catches
                // the contract violation separately).
                if ctx_for_stream.is_stopped() {
                    cancelled = true;
                }
                let is_terminal = chunk.finish_reason.is_some();
                yield Annotated::from_data(chunk);
                if is_terminal {
                    break;
                }
            }
            tracing::debug!(
                request_id = ctx_for_stream.id(),
                chunks = chunk_count,
                cancelled,
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
        async fn start(&self) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig::default())
        }

        async fn generate(
            &self,
            _request: PreprocessedRequest,
            context: Arc<dyn AsyncEngineContext>,
        ) -> Result<BoxStream<'static, LLMEngineOutput>, DynamoError> {
            if let Some(make_err) = self.setup_err {
                return Err(make_err());
            }
            let chunks = self.chunks.clone();
            let delay_ms = self.per_chunk_delay_ms;
            Ok(Box::pin(async_stream::stream! {
                for c in chunks {
                    if delay_ms > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    }
                    if context.is_stopped() { break; }
                    yield c;
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
        let (engine, _abort_ct) = MockEngine::new(vec![
            chunk::token(11),
            LLMEngineOutput::length()
                .with_tokens(vec![22])
                .with_usage(usage(3, 2)),
        ]);
        let adapter = EngineAdapter::new(engine);

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
        let adapter = EngineAdapter::new(engine);

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
        let adapter = EngineAdapter::new(engine);

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
        async fn start(&self) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig::default())
        }
        async fn generate(
            &self,
            _request: PreprocessedRequest,
            ctx: Arc<dyn AsyncEngineContext>,
        ) -> Result<BoxStream<'static, LLMEngineOutput>, DynamoError> {
            Ok(Box::pin(async_stream::stream! {
                yield chunk::token(1);
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    if ctx.is_stopped() {
                        yield LLMEngineOutput::cancelled().with_usage(usage(3, 1));
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
        let adapter = EngineAdapter::new(Arc::new(TerminalOnCancelEngine));
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
        let adapter = EngineAdapter::new(engine);

        let input = Context::new(make_request(vec![1]));
        let err = adapter.generate(input).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("BackendInvalidArgument"), "got: {msg}");
        assert!(msg.contains("bad param"), "got: {msg}");
    }
}
