// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bridges [`LLMEngine`] (the author-facing trait) to [`AsyncEngine`]
//! (the trait `Ingress::for_engine` consumes).
//!
//! Decode-mode disagg defers `engine.abort()` until the first chunk to
//! avoid orphaning the prefill peer's NIXL KV transfer.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

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
use opentelemetry::trace::{SpanContext, SpanId, Status, TraceFlags, TraceId, TraceState};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::disagg::DisaggregationMode;
use crate::engine::{GenerateContext, LLMEngine};

/// Test-only override count. Production reads check this first so tests can
/// force-enable the recording fast-path without touching `OTEL_EXPORT_ENABLED`
/// (which would require `unsafe { set_var }` and race with concurrent
/// `getenv` callers). Tests acquire an `OtlpExportOverride` guard; multiple
/// concurrent guards stack (counter-based), and the default is restored when
/// the last guard drops.
static OTLP_EXPORT_OVERRIDES: AtomicUsize = AtomicUsize::new(0);

/// OTLP-export gate. When the runtime isn't exporting traces (the operator
/// hasn't set `OTEL_EXPORT_ENABLED=1`), the per-chunk ttft capture and
/// terminal attribute writes are dead work — the span has nowhere to go but
/// the local `tracing` event stream, which doesn't read these fields. The
/// span itself still gets created so log events stay correlated with the
/// request_id.
///
/// Read uncached (~tens of ns per request) — small cost, but it removes a
/// class of test-ordering hazards that a cached version would introduce.
fn is_otlp_export_enabled() -> bool {
    if OTLP_EXPORT_OVERRIDES.load(Ordering::Relaxed) > 0 {
        return true;
    }
    std::env::var("OTEL_EXPORT_ENABLED")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Record ITL percentile attrs on the span. No-op on fewer than 2 token
/// chunks (no sample available). Mutates `samples` in place (sort) so we
/// don't pay for a second allocation; caller doesn't reuse after.
///
/// Percentile method: **nearest-rank** (`idx = round((n-1) * frac)`). For
/// small streams (3–4 samples) the reported p50 is the closest-rank value,
/// not the linear-interpolated median — expect ~1-sample jitter relative to
/// "textbook" definitions. The choice keeps the math cheap and is stable
/// against off-by-one comparisons across consecutive requests.
fn record_itl_distribution(span: &tracing::Span, samples: &mut [f64]) {
    if samples.is_empty() {
        return;
    }
    let avg = samples.iter().sum::<f64>() / samples.len() as f64;
    span.record("avg_itl_ms", format!("{avg:.2}").as_str());
    // partial_cmp can yield None only for NaN; Instant deltas never NaN, so
    // unwrap_or(Equal) is a safe fallback that keeps order stable.
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = |frac: f64| -> f64 {
        let idx = ((samples.len() - 1) as f64 * frac).round() as usize;
        samples[idx]
    };
    span.record("itl_p50_ms", format!("{:.2}", p(0.50)).as_str());
    span.record("itl_p99_ms", format!("{:.2}", p(0.99)).as_str());
    span.record(
        "itl_max_ms",
        format!("{:.2}", samples[samples.len() - 1]).as_str(),
    );
}

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

        // Per-request worker-side span. Nests under `handle_payload` (set up
        // by the runtime's NATS ingress) so the trace tree has a contiguous
        // worker layer. Attributes get filled in across the stream lifecycle:
        // - `model`, `input_tokens`, `disagg_role` on entry,
        // - `prefill_trace_id` / `prefill_span_id` on decode (when the prefill
        //   peer embedded its trace identity in disaggregated_params),
        // - `ttft_ms` on first non-empty chunk,
        // - `output_tokens`, `finish_reason`, `cancelled`, `itl_*_ms` on
        //   terminal.
        let span = tracing::info_span!(
            target: "request_span",
            "engine.generate",
            model = %request.model,
            input_tokens = request.token_ids.len(),
            disagg_role = self.mode.as_str(),
            prefill_trace_id = tracing::field::Empty,
            prefill_span_id = tracing::field::Empty,
            ttft_ms = tracing::field::Empty,
            output_tokens = tracing::field::Empty,
            finish_reason = tracing::field::Empty,
            cancelled = tracing::field::Empty,
            error_kind = tracing::field::Empty,
            avg_itl_ms = tracing::field::Empty,
            itl_p50_ms = tracing::field::Empty,
            itl_p99_ms = tracing::field::Empty,
            itl_max_ms = tracing::field::Empty,
        );

        // Decode-side: attach a real OTel Link from the engine.generate span
        // to the prefill peer's span. Tempo / Jaeger render this as a
        // cross-trace edge — operators don't have to copy-paste trace IDs.
        // We also record the IDs as attributes for log-analysis fallback
        // (JSONL log readers can't render OTel Links).
        //
        // Reads from `prefill_trace_link` (framework-owned, typed), keeping
        // the disagg trace contract separate from engine-owned
        // `disaggregated_params` payloads.
        if self.mode.is_decode()
            && let Some(prefill) = request.prefill_result.as_ref()
            && let Some(link) = prefill.prefill_trace_link.as_ref()
        {
            span.record("prefill_trace_id", link.trace_id.as_str());
            span.record("prefill_span_id", link.span_id.as_str());
            if let (Ok(trace_id), Ok(span_id)) = (
                TraceId::from_hex(&link.trace_id),
                SpanId::from_hex(&link.span_id),
            ) {
                span.add_link(SpanContext::new(
                    trace_id,
                    span_id,
                    TraceFlags::SAMPLED,
                    true, // is_remote
                    TraceState::default(),
                ));
            }
        }

        // Prefill-side: capture this worker's trace identity for embedding
        // into the terminal chunk's `prefill_trace_link`. `in_scope` reads
        // from our `engine.generate` span (the DistributedTraceIdLayer has
        // populated it in JSONL mode); yields None in non-JSONL deployments.
        let prefill_trace_link: Option<dynamo_llm::protocols::common::preprocessor::TraceLink> =
            if self.mode.is_prefill() {
                let link = span.in_scope(|| {
                    dynamo_runtime::logging::get_distributed_tracing_context().map(|tc| {
                        dynamo_llm::protocols::common::preprocessor::TraceLink {
                            trace_id: tc.trace_id,
                            span_id: tc.span_id,
                        }
                    })
                });
                if link.is_none() {
                    tracing::debug!(
                        "disagg trace linking inactive — no DistributedTraceContext \
                         on engine.generate span (DistributedTraceIdLayer requires JSONL \
                         mode + OTEL_EXPORT_ENABLED)"
                    );
                }
                link
            } else {
                None
            };

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

        let gen_ctx =
            GenerateContext::with_metadata(ctx.clone(), ft_tx.clone(), handle.metadata().clone());
        // `.instrument()` the setup call so a setup-time error lands on the
        // same span as the streaming body.
        let chunks = self
            .engine
            .generate(request, gen_ctx)
            .instrument(span.clone())
            .await
            .map_err(|e| {
                span.record("error_kind", "setup_failed");
                // Short, stable description — full error message is
                // available via the trace_id-correlated log stream.
                span.set_status(Status::error("setup_failed"));
                Error::from(e)
            })?;
        let request_start = Instant::now();

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
        let stream_span = span.clone();
        let should_record_attrs = is_otlp_export_enabled();
        let is_prefill_mode = self.mode.is_prefill();
        let mapped = async_stream::stream! {
            let _guard = guard;
            let mut inner = chunks;
            let mut chunk_count: usize = 0;
            let mut output_token_count: usize = 0;
            let mut signalled = false;
            // ITL samples (ms) — millisecond gap between successive non-empty
            // token chunks. Aggregate; we only render percentiles at terminal
            // so the per-chunk overhead is one timestamp + one Vec push.
            let mut itl_samples_ms: Vec<f64> = Vec::new();
            let mut last_token_at: Option<Instant> = None;
            while let Some(item) = inner.next().await {
                chunk_count += 1;
                match item {
                    Ok(mut chunk) => {
                        // First non-empty chunk releases the deferred abort.
                        // Token-less chunks (SGLang's bootstrap handshake) don't count.
                        if !signalled && !chunk.token_ids.is_empty() {
                            if should_record_attrs {
                                // ttft = time from setup completion to first token.
                                // Recorded once; subsequent non-empty chunks don't update.
                                let ttft_ms = request_start.elapsed().as_secs_f64() * 1000.0;
                                stream_span.record("ttft_ms", format!("{:.2}", ttft_ms).as_str());
                                last_token_at = Some(Instant::now());
                            }
                            if let Some(tx) = &ft_tx {
                                // Receiver is held by the monitor task; send only
                                // fails if it panicked, in which case the abort is
                                // already moot.
                                let _ = tx.send(true);
                            }
                            signalled = true;
                        } else if should_record_attrs
                            && !chunk.token_ids.is_empty()
                            && let Some(prev) = last_token_at
                        {
                            // Subsequent non-empty chunks contribute one ITL sample.
                            let now = Instant::now();
                            itl_samples_ms.push(now.duration_since(prev).as_secs_f64() * 1000.0);
                            last_token_at = Some(now);
                        }
                        if should_record_attrs {
                            output_token_count += chunk.token_ids.len();
                        }
                        let is_terminal = chunk.finish_reason.is_some();
                        if should_record_attrs && is_terminal {
                            stream_span.record("output_tokens", output_token_count);
                            if let Some(reason) = chunk.finish_reason.as_ref() {
                                stream_span.record("finish_reason", format!("{:?}", reason).as_str());
                            }
                            stream_span.record("cancelled", stream_ctx.is_stopped());
                            record_itl_distribution(&stream_span, &mut itl_samples_ms);
                        }
                        // Prefill-side: stamp our trace identity into
                        // `prefill_trace_link` (framework-owned, typed). The
                        // decode peer reads from prefill_result.prefill_trace_link.
                        // We never touch the engine's disaggregated_params
                        // payload — it stays opaque to the framework.
                        if is_prefill_mode
                            && is_terminal
                            && let Some(link) = &prefill_trace_link
                        {
                            chunk.prefill_trace_link = Some(link.clone());
                        }
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
                        if should_record_attrs {
                            let error_kind = format!("{:?}", dynamo_err.error_type());
                            stream_span.record("error_kind", error_kind.as_str());
                            stream_span.record("output_tokens", output_token_count);
                            stream_span.record("cancelled", stream_ctx.is_stopped());
                            record_itl_distribution(&stream_span, &mut itl_samples_ms);
                            // Surface as OTel status so Tempo / Jaeger render
                            // the span as errored (red, counts in error-rate
                            // dashboards). Use the variant name as a short,
                            // machine-correlatable description — full error
                            // message is available via the trace_id-
                            // correlated log stream.
                            stream_span.set_status(Status::error(error_kind));
                        }
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
        // `stream_span.record(...)` from the closure mutates the same span
        // we just dropped — `Span` is a cheap handle, clones share storage.

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

    // -------------------------------------------------------------------
    // Auto-span observation. Uses a custom tracing layer that records field
    // values on `engine.generate` spans into a shared map, then asserts on
    // the values the adapter recorded across the stream lifecycle.
    // -------------------------------------------------------------------

    use std::collections::HashMap;
    use std::sync::Mutex;
    use tracing::field::{Field, Visit};
    use tracing::span;
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::{Context as TraceCtx, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;
    use tracing_subscriber::util::SubscriberInitExt;

    #[derive(Default, Clone)]
    struct CapturedFields(Arc<Mutex<HashMap<String, String>>>);

    struct FieldRecorder<'a>(&'a mut HashMap<String, String>);

    impl<'a> Visit for FieldRecorder<'a> {
        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
        fn record_i64(&mut self, field: &Field, value: i64) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
        fn record_u64(&mut self, field: &Field, value: u64) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
        fn record_bool(&mut self, field: &Field, value: bool) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }
    }

    struct CaptureLayer {
        out: CapturedFields,
        span_name: &'static str,
    }

    impl<S> Layer<S> for CaptureLayer
    where
        S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(&self, attrs: &span::Attributes<'_>, _id: &span::Id, _ctx: TraceCtx<'_, S>) {
            if attrs.metadata().name() != self.span_name {
                return;
            }
            let mut out = self.out.0.lock().unwrap();
            attrs.record(&mut FieldRecorder(&mut out));
        }

        fn on_record(&self, id: &span::Id, values: &span::Record<'_>, ctx: TraceCtx<'_, S>) {
            if !ctx.span(id).is_some_and(|s| s.name() == self.span_name) {
                return;
            }
            let mut out = self.out.0.lock().unwrap();
            values.record(&mut FieldRecorder(&mut out));
        }
    }

    /// RAII counter on `OTLP_EXPORT_OVERRIDES`. Concurrent test guards stack;
    /// the default OTLP-off behavior is restored when the last guard drops.
    /// No env-var mutation, no `unsafe`.
    struct OtlpExportOverride;

    impl OtlpExportOverride {
        fn enable() -> Self {
            OTLP_EXPORT_OVERRIDES.fetch_add(1, Ordering::Relaxed);
            Self
        }
    }

    impl Drop for OtlpExportOverride {
        fn drop(&mut self) {
            OTLP_EXPORT_OVERRIDES.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Bundle: OTLP override + tracing-subscriber dispatch guard. Tests hold
    /// this for the duration of the captured assertions.
    struct CaptureGuard {
        _otlp: OtlpExportOverride,
        _dispatch: tracing::dispatcher::DefaultGuard,
    }

    /// Force-enable the recording fast-path and install a fresh subscriber
    /// that captures every `engine.generate` field write into a shared map.
    fn install_capture() -> (CapturedFields, CaptureGuard) {
        let otlp = OtlpExportOverride::enable();
        let captured = CapturedFields::default();
        let layer = CaptureLayer {
            out: captured.clone(),
            span_name: "engine.generate",
        };
        let dispatch = tracing_subscriber::registry().with(layer).set_default();
        (
            captured,
            CaptureGuard {
                _otlp: otlp,
                _dispatch: dispatch,
            },
        )
    }

    #[tokio::test]
    async fn auto_span_records_initial_attrs() {
        let (captured, _guard) = install_capture();
        let (engine, _abort) = MockEngine::new(vec![]);
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Aggregated);
        let input = Context::new(make_request(vec![1, 2, 3, 4, 5]));
        let _stream = adapter.generate(input).await.unwrap();

        let fields = captured.0.lock().unwrap().clone();
        assert_eq!(fields.get("model").map(String::as_str), Some("mock"));
        assert_eq!(fields.get("input_tokens").map(String::as_str), Some("5"));
        assert_eq!(fields.get("disagg_role").map(String::as_str), Some("agg"));
    }

    #[tokio::test]
    async fn auto_span_records_terminal_attrs_on_clean_stream() {
        let (captured, _guard) = install_capture();
        let (engine, _abort) = MockEngine::new(vec![
            chunk::token(11),
            chunk::token(22),
            LLMEngineOutput::length()
                .with_tokens(vec![33])
                .with_usage(usage(3, 3)),
        ]);
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Aggregated);
        let input = Context::new(make_request(vec![1, 2, 3]));
        let stream = adapter.generate(input).await.unwrap();
        let _: Vec<_> = stream.collect().await;

        let fields = captured.0.lock().unwrap().clone();
        // 3 chunks, 1 token each → 3 total. Sum-not-count is the contract.
        assert_eq!(fields.get("output_tokens").map(String::as_str), Some("3"));
        assert!(
            fields
                .get("finish_reason")
                .is_some_and(|v| v.contains("Length")),
            "got: {:?}",
            fields.get("finish_reason")
        );
        assert_eq!(fields.get("cancelled").map(String::as_str), Some("false"));
        assert!(
            fields.contains_key("ttft_ms"),
            "ttft_ms missing; got fields: {fields:?}"
        );
        // 3 token chunks → 2 ITL samples → all four ITL attrs populated.
        for key in ["avg_itl_ms", "itl_p50_ms", "itl_p99_ms", "itl_max_ms"] {
            assert!(
                fields.contains_key(key),
                "{key} missing; got fields: {fields:?}"
            );
        }
    }

    #[tokio::test]
    async fn auto_span_marks_cancelled_when_stream_stopped() {
        let (captured, _guard) = install_capture();
        let adapter = EngineAdapter::new(
            Arc::new(TerminalOnCancelEngine),
            DisaggregationMode::Aggregated,
        );
        let input: Context<PreprocessedRequest> = Context::new(make_request(vec![1, 2, 3]));
        let ctrl = input.context();
        let mut stream = adapter.generate(input).await.unwrap();
        let _ = stream.next().await;
        ctrl.stop_generating();
        let _: Vec<_> = stream.collect().await;

        let fields = captured.0.lock().unwrap().clone();
        assert_eq!(fields.get("cancelled").map(String::as_str), Some("true"));
    }

    #[tokio::test]
    async fn auto_span_records_error_kind_on_typed_error() {
        let (captured, _guard) = install_capture();
        let adapter = EngineAdapter::new(
            Arc::new(TypedMidStreamErrEngine),
            DisaggregationMode::Aggregated,
        );
        let input = Context::new(make_request(vec![1]));
        let stream = adapter.generate(input).await.unwrap();
        let _: Vec<_> = stream.collect().await;

        let fields = captured.0.lock().unwrap().clone();
        let err_kind = fields.get("error_kind").cloned().unwrap_or_default();
        assert!(
            err_kind.contains("InvalidArgument"),
            "expected error_kind to contain InvalidArgument, got: {err_kind:?}"
        );
    }

    /// Regression guard: the `engine.generate` span must be created
    /// unconditionally — never gated on `is_otlp_export_enabled()`.
    /// Companion to `auto_span_records_initial_attrs`, which runs WITH
    /// the OTLP override and so wouldn't catch a future refactor that
    /// gates the `info_span!` itself on the fast-path flag.
    #[tokio::test]
    async fn auto_span_fires_without_otlp_override() {
        // Install ONLY the capture layer — no `OtlpExportOverride::enable()`.
        let captured = CapturedFields::default();
        let layer = CaptureLayer {
            out: captured.clone(),
            span_name: "engine.generate",
        };
        let _dispatch = tracing_subscriber::registry().with(layer).set_default();

        let (engine, _abort) = MockEngine::new(vec![chunk::token(7), LLMEngineOutput::length()]);
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Aggregated);
        let input = Context::new(make_request(vec![1, 2, 3]));
        let stream = adapter.generate(input).await.unwrap();
        let _: Vec<_> = stream.collect().await;

        let fields = captured.0.lock().unwrap().clone();
        assert_eq!(
            fields.get("model").map(String::as_str),
            Some("mock"),
            "engine.generate span must be created with `model` attr even when OTLP export is off"
        );
        assert_eq!(
            fields.get("input_tokens").map(String::as_str),
            Some("3"),
            "engine.generate span must record input_tokens at entry"
        );
        assert_eq!(
            fields.get("disagg_role").map(String::as_str),
            Some("agg"),
            "engine.generate span must record disagg_role at entry"
        );
    }

    /// Pure unit test of the percentile helper. Five evenly-spaced samples;
    /// median is the middle, p99 lands on max for n=5.
    #[test]
    fn record_itl_distribution_computes_percentiles() {
        let (captured, _guard) = install_capture();
        let span = tracing::info_span!(
            target: "request_span",
            "engine.generate",
            avg_itl_ms = tracing::field::Empty,
            itl_p50_ms = tracing::field::Empty,
            itl_p99_ms = tracing::field::Empty,
            itl_max_ms = tracing::field::Empty,
        );
        let mut samples = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        record_itl_distribution(&span, &mut samples);

        let fields = captured.0.lock().unwrap().clone();
        assert_eq!(fields.get("avg_itl_ms").map(String::as_str), Some("30.00"));
        assert_eq!(fields.get("itl_p50_ms").map(String::as_str), Some("30.00"));
        assert_eq!(fields.get("itl_p99_ms").map(String::as_str), Some("50.00"));
        assert_eq!(fields.get("itl_max_ms").map(String::as_str), Some("50.00"));
    }

    /// Empty sample set — no-op, no panic, no fields recorded.
    #[test]
    fn record_itl_distribution_no_op_when_empty() {
        let (captured, _guard) = install_capture();
        let span = tracing::info_span!(
            target: "request_span",
            "engine.generate",
            avg_itl_ms = tracing::field::Empty,
        );
        let mut samples: Vec<f64> = vec![];
        record_itl_distribution(&span, &mut samples);
        assert!(!captured.0.lock().unwrap().contains_key("avg_itl_ms"));
    }

    /// Decode-side: `prefill_result` exists but carries no `dynamo_trace`
    /// payload — the link branch must skip cleanly without recording attrs
    /// or panicking.
    #[tokio::test]
    async fn auto_span_no_link_when_dynamo_trace_missing() {
        use dynamo_llm::protocols::common::preprocessor::PrefillResult;

        let (captured, _guard) = install_capture();
        let (engine, _abort) = MockEngine::new(vec![chunk::token(1), LLMEngineOutput::length()]);
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Decode);

        let mut request = make_request(vec![1, 2, 3]);
        request.prefill_result = Some(PrefillResult {
            // Engine packed its own kv-transfer payload; no Dynamo trace meta.
            disaggregated_params: serde_json::json!({
                "engine_specific": "value",
            }),
            prefill_trace_link: None,
            prompt_tokens_details: None,
        });
        let input = Context::new(request);
        let stream = adapter.generate(input).await.unwrap();
        let _: Vec<_> = stream.collect().await;

        let fields = captured.0.lock().unwrap().clone();
        assert!(!fields.contains_key("prefill_trace_id"));
        assert!(!fields.contains_key("prefill_span_id"));
    }

    /// Decode-side: `prefill_trace_link` is set but the hex IDs are
    /// malformed — the link branch records the raw strings as attrs but
    /// skips the OTel Link construction (TraceId::from_hex rejects bad hex).
    /// The test asserts the attrs are still recorded so operators get the
    /// fallback path; if `add_link` ever panics on bad input the test catches
    /// the regression.
    #[tokio::test]
    async fn auto_span_handles_malformed_hex_gracefully() {
        use dynamo_llm::protocols::common::preprocessor::{PrefillResult, TraceLink};

        let (captured, _guard) = install_capture();
        let (engine, _abort) = MockEngine::new(vec![chunk::token(1), LLMEngineOutput::length()]);
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Decode);

        let mut request = make_request(vec![1, 2, 3]);
        request.prefill_result = Some(PrefillResult {
            disaggregated_params: serde_json::json!({}),
            prefill_trace_link: Some(TraceLink {
                trace_id: "not-valid-hex".to_string(),
                span_id: "ALSO-INVALID".to_string(),
            }),
            prompt_tokens_details: None,
        });
        let input = Context::new(request);
        let stream = adapter.generate(input).await.unwrap();
        let _: Vec<_> = stream.collect().await;

        let fields = captured.0.lock().unwrap().clone();
        // Attrs still set (fallback for log analysis).
        assert_eq!(
            fields.get("prefill_trace_id").map(String::as_str),
            Some("not-valid-hex")
        );
        assert_eq!(
            fields.get("prefill_span_id").map(String::as_str),
            Some("ALSO-INVALID")
        );
        // No panic — Link branch must have skipped via `if let Ok(...)`.
    }

    /// Decode-side: when the upstream request carries a
    /// `prefill_result.prefill_trace_link`, the adapter must record
    /// `prefill_trace_id` and `prefill_span_id` on its `engine.generate`
    /// span so operators can hop traces.
    #[tokio::test]
    async fn auto_span_records_prefill_link_on_decode() {
        use dynamo_llm::protocols::common::preprocessor::{PrefillResult, TraceLink};

        let (captured, _guard) = install_capture();
        let (engine, _abort) = MockEngine::new(vec![chunk::token(1), LLMEngineOutput::length()]);
        let adapter = EngineAdapter::new(engine, DisaggregationMode::Decode);

        let mut request = make_request(vec![1, 2, 3]);
        request.prefill_result = Some(PrefillResult {
            disaggregated_params: serde_json::json!({}),
            prefill_trace_link: Some(TraceLink {
                trace_id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
                span_id: "bbbbbbbbbbbbbbbb".to_string(),
            }),
            prompt_tokens_details: None,
        });
        let input = Context::new(request);
        let stream = adapter.generate(input).await.unwrap();
        let _: Vec<_> = stream.collect().await;

        let fields = captured.0.lock().unwrap().clone();
        assert_eq!(
            fields.get("prefill_trace_id").map(String::as_str),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
        assert_eq!(
            fields.get("prefill_span_id").map(String::as_str),
            Some("bbbbbbbbbbbbbbbb")
        );
    }
}
