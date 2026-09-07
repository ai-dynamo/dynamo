// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conformance test kit for [`LLMEngine`] and [`RawEngine`] implementations.
//!
//! Engines wire themselves into the test suite with one call —
//! [`run_conformance`] for token engines, [`run_encode_conformance`] for
//! Encode-role engines, and [`run_raw_conformance`] for raw media engines:
//!
//! ```ignore
//! #[tokio::test]
//! async fn my_engine_satisfies_contract() {
//!     dynamo_backend_common::testing::run_conformance(MyEngine::new_for_test)
//!         .await
//!         .expect("conformance");
//! }
//! ```
//!
//! The kit takes a factory rather than a pre-built engine so it can
//! construct one engine for the main lifecycle test and a second,
//! pristine engine for the "cleanup before start" check — the latter
//! mirrors `Worker`'s post-start-failure cleanup path and would not
//! work on an already-started engine.
//!
//! Gated behind the `testing` cargo feature; intended for `[dev-dependencies]`.

use std::sync::Arc;
use std::time::Duration;

use dynamo_llm::protocols::common::preprocessor::{MultimodalData, PreprocessedRequest};
use dynamo_llm::protocols::common::{FinishReason, OutputOptions, SamplingOptions, StopConditions};
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use futures::StreamExt;

use crate::engine::{GenerateContext, LLMEngine, RawEngine};
use crate::metrics::{EngineMetrics, TestHierarchy};
use ConformanceFailure::*;

const DEFAULT_CANCEL_DEADLINE: Duration = Duration::from_secs(2);

/// Fresh, non-cancelled context suitable for a single `generate` call.
pub fn mock_context() -> Arc<dyn AsyncEngineContext> {
    Context::<()>::new(()).context()
}

/// Context that auto-triggers `stop_generating()` after `after` has elapsed.
///
/// Must be called from within a running tokio runtime (uses `tokio::spawn`).
pub fn cancelling_context(after: Duration) -> Arc<dyn AsyncEngineContext> {
    let ctx = Context::<()>::new(()).context();
    let ctx2 = ctx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(after).await;
        ctx2.stop_generating();
    });
    ctx
}

/// Which conformance check failed, and why.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConformanceFailure {
    StartFailed(String),
    EmptyModelInConfig,
    /// A `RawEngine` populated `EngineConfig.llm` — raw media engines have no
    /// token pipeline, so they must leave the registration sub-record `None`.
    RawEngineAdvertisedLlmRegistration,
    GenerateFailed(String),
    NoChunksYielded,
    ChunkAfterTerminal,
    NoTerminalChunk,
    StreamYieldedError(String),
    ConcurrentGenerateFailed(String),
    CancellationNotObserved {
        after: Duration,
    },
    CancellationIgnored,
    CleanupFailed(String),
    SecondCleanupFailed(String),
    CleanupWithoutStartFailed(String),
    KvEventSourcesFailed(String),
    KvEventSourcesNotIdempotent,
    SetupMetricsFailed(String),
    ComponentMetricsNotIdempotent,
    /// The engine's terminal `completion_usage.completion_tokens` doesn't
    /// match the sum of `chunk.token_ids.len()` it emitted across the
    /// stream. The framework records `output_tokens` from the chunk-token
    /// sum; a divergence means the engine's internal bookkeeping disagrees
    /// with what it actually streamed.
    CompletionTokensMismatch {
        chunked: usize,
        reported: u32,
    },
    /// Encode conformance requires exactly one terminal handoff chunk.
    EncodeChunkCount {
        count: usize,
    },
    /// Encode terminals use the standard `Stop` finish reason.
    EncodeTerminalExpected,
    /// Encode workers produce a handoff payload, not generated tokens.
    EncodeTokensNotEmpty {
        count: usize,
    },
    /// Encode terminal usage must describe a zero-token handoff response.
    EncodeUsageMismatch {
        expected_prompt: u32,
        prompt: u32,
        completion: u32,
        total: u32,
    },
    /// The encoder handoff contract is object-shaped at every boundary.
    EncoderResultExpectedObject,
}

impl std::fmt::Display for ConformanceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartFailed(m) => write!(f, "start() failed: {m}"),
            EmptyModelInConfig => write!(f, "EngineConfig.model is empty"),
            RawEngineAdvertisedLlmRegistration => write!(
                f,
                "RawEngine populated EngineConfig.llm; raw media engines have no \
                 token pipeline and must leave it None"
            ),
            GenerateFailed(m) => write!(f, "generate() failed: {m}"),
            NoChunksYielded => write!(f, "generate() stream yielded no chunks"),
            ChunkAfterTerminal => write!(f, "chunk yielded after terminal chunk"),
            NoTerminalChunk => write!(f, "stream ended without a terminal chunk"),
            StreamYieldedError(m) => write!(f, "engine stream yielded Err: {m}"),
            ConcurrentGenerateFailed(m) => {
                write!(f, "concurrent generate() calls failed: {m}")
            }
            CancellationNotObserved { after } => write!(
                f,
                "stream did not terminate within {after:?} after cancellation"
            ),
            CancellationIgnored => write!(
                f,
                "stream terminated but terminal chunk's finish_reason was not Cancelled \
                 (engine must emit FinishReason::Cancelled when it observes cancellation)"
            ),
            CleanupFailed(m) => write!(f, "cleanup() failed: {m}"),
            SecondCleanupFailed(m) => {
                write!(f, "second cleanup() call failed (must be idempotent): {m}")
            }
            CleanupWithoutStartFailed(m) => write!(
                f,
                "cleanup() failed on a never-started engine: {m} \
                 (Worker calls cleanup() after start() raises, so engines must \
                 be null-safe against partial / no allocation)"
            ),
            KvEventSourcesFailed(m) => write!(f, "kv_event_sources() failed: {m}"),
            KvEventSourcesNotIdempotent => write!(
                f,
                "kv_event_sources() returned different dp_rank set on a second call \
                 (the descriptor list must be stable for the engine's lifetime)"
            ),
            SetupMetricsFailed(m) => write!(f, "setup_metrics() failed: {m}"),
            ComponentMetricsNotIdempotent => write!(
                f,
                "setup_metrics().dp_ranks returned different ranks across calls \
                 (the rank set must be stable for the engine's lifetime)"
            ),
            CompletionTokensMismatch { chunked, reported } => write!(
                f,
                "engine emitted {chunked} tokens across the stream but reported \
                 completion_usage.completion_tokens = {reported} on the terminal \
                 (engine bookkeeping diverges from streamed output)"
            ),
            EncodeChunkCount { count } => write!(
                f,
                "encode generate() must yield exactly one terminal chunk; got {count}"
            ),
            EncodeTerminalExpected => write!(
                f,
                "encode generate() must yield a terminal chunk with \
                 finish_reason = FinishReason::Stop"
            ),
            EncodeTokensNotEmpty { count } => write!(
                f,
                "encode terminal chunk must have empty token_ids; got {count} tokens"
            ),
            EncodeUsageMismatch {
                expected_prompt,
                prompt,
                completion,
                total,
            } => write!(
                f,
                "encode terminal usage must report prompt={expected_prompt}, completion=0, \
                 total={expected_prompt}; got prompt={prompt}, completion={completion}, \
                 total={total}"
            ),
            EncoderResultExpectedObject => write!(
                f,
                "encode terminal chunk must carry an object-shaped encoder_result"
            ),
        }
    }
}

impl std::error::Error for ConformanceFailure {}

/// Run the full conformance suite against an engine.
///
/// Takes a factory rather than a built engine so the kit can construct
/// a second, pristine engine for the "cleanup before start" check.
pub async fn run_conformance<E, F>(mut factory: F) -> Result<(), ConformanceFailure>
where
    E: LLMEngine,
    F: FnMut() -> E,
{
    let engine = factory();

    // 1. start() returns non-empty model.
    let config = engine
        .start(0)
        .await
        .map_err(|e| StartFailed(e.to_string()))?;
    if config.model.is_empty() {
        return Err(EmptyModelInConfig);
    }

    // 2. KV-aware-routing source descriptors satisfy their contracts:
    //    - kv_event_sources doesn't error; rank set is stable across calls
    //    - setup_metrics doesn't error against a synthetic EngineMetrics
    //    - returned MetricsBindings.dp_ranks are stable across calls
    //
    //    Run before generate() to match Worker's actual call order
    //    (publishers wire up between start() and serve).
    check_kv_event_sources(&engine).await?;
    check_setup_metrics(&engine).await?;

    // 4. A plain generate() yields a well-formed stream ending in a terminal chunk.
    check_single_generate(&engine, &config.model).await?;

    // 5. Interleaved generate() calls both complete — catches shared-state bugs.
    //    Uses tokio::join! under the test runtime (single-threaded by default),
    //    so this is interleaving rather than true parallelism.
    check_concurrent_generates(&engine, &config.model, LlmConformanceMode::Token).await?;

    // 6. Cancellation is observed within a bounded deadline.
    check_cancellation(
        &engine,
        &config.model,
        LlmConformanceMode::Token,
        DEFAULT_CANCEL_DEADLINE,
    )
    .await?;

    // 7. cleanup() succeeds and is idempotent.
    engine
        .cleanup()
        .await
        .map_err(|e| CleanupFailed(e.to_string()))?;
    engine
        .cleanup()
        .await
        .map_err(|e| SecondCleanupFailed(e.to_string()))?;

    // 8. cleanup() is safe on a never-started engine — mirrors the path
    //    `Worker` takes after `start()` raises. Engines must guard each
    //    allocated resource with a null-check.
    let fresh = factory();
    fresh
        .cleanup()
        .await
        .map_err(|e| CleanupWithoutStartFailed(e.to_string()))?;

    Ok(())
}

/// Run the Encode-role conformance suite against an [`LLMEngine`].
///
/// Encode engines have a narrower response shape than token generators, but
/// share the same routing, metrics, concurrency, cancellation, and cleanup
/// lifecycle contracts.
pub async fn run_encode_conformance<E, F>(mut factory: F) -> Result<(), ConformanceFailure>
where
    E: LLMEngine,
    F: FnMut() -> E,
{
    let engine = factory();
    let config = engine
        .start(0)
        .await
        .map_err(|e| StartFailed(e.to_string()))?;
    if config.model.is_empty() {
        return Err(EmptyModelInConfig);
    }

    check_kv_event_sources(&engine).await?;
    check_setup_metrics(&engine).await?;
    check_encode_generate(&engine, &config.model).await?;
    check_concurrent_generates(&engine, &config.model, LlmConformanceMode::Encode).await?;
    check_cancellation(
        &engine,
        &config.model,
        LlmConformanceMode::Encode,
        DEFAULT_CANCEL_DEADLINE,
    )
    .await?;

    engine
        .cleanup()
        .await
        .map_err(|e| CleanupFailed(e.to_string()))?;
    engine
        .cleanup()
        .await
        .map_err(|e| SecondCleanupFailed(e.to_string()))?;

    let fresh = factory();
    fresh
        .cleanup()
        .await
        .map_err(|e| CleanupWithoutStartFailed(e.to_string()))?;

    Ok(())
}

async fn check_encode_generate<E: LLMEngine>(
    engine: &E,
    model: &str,
) -> Result<(), ConformanceFailure> {
    let request = encode_request(model);
    let expected_prompt = request.token_ids.len() as u32;
    let stream = engine
        .generate(request, GenerateContext::new(mock_context(), None))
        .await
        .map_err(|e| GenerateFailed(e.to_string()))?;
    let items: Vec<_> = stream.collect().await;
    validate_encode_items(items, expected_prompt)
}

fn validate_encode_items(
    items: Vec<Result<crate::engine::LLMEngineOutput, crate::error::DynamoError>>,
    expected_prompt: u32,
) -> Result<(), ConformanceFailure> {
    if items.len() != 1 {
        return Err(EncodeChunkCount { count: items.len() });
    }
    let chunk = items
        .into_iter()
        .next()
        .expect("length checked")
        .map_err(|e| StreamYieldedError(e.to_string()))?;

    if !matches!(chunk.finish_reason, Some(FinishReason::Stop)) {
        return Err(EncodeTerminalExpected);
    }
    if !chunk.token_ids.is_empty() {
        return Err(EncodeTokensNotEmpty {
            count: chunk.token_ids.len(),
        });
    }
    if !chunk.encoder_result.as_ref().is_some_and(|v| v.is_object()) {
        return Err(EncoderResultExpectedObject);
    }
    if let Some(usage) = chunk.completion_usage.as_ref()
        && (usage.prompt_tokens != expected_prompt
            || usage.completion_tokens != 0
            || usage.total_tokens != expected_prompt)
    {
        return Err(EncodeUsageMismatch {
            expected_prompt,
            prompt: usage.prompt_tokens,
            completion: usage.completion_tokens,
            total: usage.total_tokens,
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum LlmConformanceMode {
    Token,
    Encode,
}

fn request(model: &str) -> PreprocessedRequest {
    // Keep conformance smokes bounded for real LLM engines. The separate
    // cancellation check still requests enough tokens to catch ignored cancels.
    request_with_max_tokens(model, Some(8))
}

fn request_with_max_tokens(model: &str, max_tokens: Option<u32>) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(vec![1, 2, 3])
        .stop_conditions(StopConditions {
            max_tokens,
            ..Default::default()
        })
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions::default())
        .build()
        .expect("build request")
}

fn encode_request(model: &str) -> PreprocessedRequest {
    let multi_modal_data = std::collections::HashMap::from([(
        "image".to_string(),
        vec![MultimodalData::RawUrl(
            "data:image/png;base64,AA==".to_string(),
        )],
    )]);
    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(vec![1, 2, 3])
        .multi_modal_data(Some(multi_modal_data))
        .mm_processor_kwargs(Some(serde_json::json!({ "min_pixels": 64 })))
        .stop_conditions(StopConditions {
            max_tokens: Some(8),
            ..Default::default()
        })
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions::default())
        .build()
        .expect("build encode request")
}

async fn check_single_generate<E: LLMEngine>(
    engine: &E,
    model: &str,
) -> Result<(), ConformanceFailure> {
    let ctx = mock_context();
    let stream = engine
        .generate(request(model), GenerateContext::new(ctx, None))
        .await
        .map_err(|e| GenerateFailed(e.to_string()))?;
    let items: Vec<_> = stream.collect().await;

    if items.is_empty() {
        return Err(NoChunksYielded);
    }
    let mut chunks = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Ok(c) => chunks.push(c),
            Err(e) => return Err(StreamYieldedError(e.to_string())),
        }
    }
    let mut terminal_idx = None;
    for (i, c) in chunks.iter().enumerate() {
        if c.finish_reason.is_some() {
            if terminal_idx.is_some() {
                return Err(ChunkAfterTerminal);
            }
            terminal_idx = Some(i);
        }
    }
    let terminal_idx = match terminal_idx {
        Some(i) if i == chunks.len() - 1 => i,
        Some(_) => return Err(ChunkAfterTerminal),
        None => return Err(NoTerminalChunk),
    };

    // Engine bookkeeping self-consistency: if the engine reports its own
    // completion_tokens count on the terminal chunk, it must agree with the
    // tokens it actually emitted. Skip when the engine doesn't report.
    if let Some(usage) = chunks[terminal_idx].completion_usage.as_ref() {
        let chunked: usize = chunks.iter().map(|c| c.token_ids.len()).sum();
        if chunked != usage.completion_tokens as usize {
            return Err(CompletionTokensMismatch {
                chunked,
                reported: usage.completion_tokens,
            });
        }
    }
    Ok(())
}

async fn check_concurrent_generates<E: LLMEngine>(
    engine: &E,
    model: &str,
    mode: LlmConformanceMode,
) -> Result<(), ConformanceFailure> {
    // 8 in-flight streams — enough to catch state-tramping under interleaved
    // polls. Under a single-threaded test runtime this is interleaving rather
    // than true parallelism, but it still exercises shared-state correctness.
    const CONCURRENT: usize = 8;
    let futs = (0..CONCURRENT).map(|_| async {
        let ctx = mock_context();
        let request = match mode {
            LlmConformanceMode::Token => request(model),
            LlmConformanceMode::Encode => encode_request(model),
        };
        let expected_prompt = request.token_ids.len() as u32;
        let stream = engine
            .generate(request, GenerateContext::new(ctx, None))
            .await
            .map_err(|e| ConcurrentGenerateFailed(e.to_string()))?;
        match mode {
            LlmConformanceMode::Token => {
                let n = stream.count().await;
                if n == 0 {
                    Err(ConcurrentGenerateFailed("stream was empty".to_string()))
                } else {
                    Ok(())
                }
            }
            LlmConformanceMode::Encode => {
                validate_encode_items(stream.collect().await, expected_prompt)
            }
        }
    });
    for result in futures::future::join_all(futs).await {
        result?;
    }
    Ok(())
}

async fn check_kv_event_sources<E: LLMEngine>(engine: &E) -> Result<(), ConformanceFailure> {
    let first = engine
        .kv_event_sources()
        .await
        .map_err(|e| KvEventSourcesFailed(e.to_string()))?;
    let second = engine
        .kv_event_sources()
        .await
        .map_err(|e| KvEventSourcesFailed(e.to_string()))?;
    let ranks_a: Vec<u32> = first.iter().map(|s| s.dp_rank()).collect();
    let ranks_b: Vec<u32> = second.iter().map(|s| s.dp_rank()).collect();
    if ranks_a != ranks_b {
        return Err(KvEventSourcesNotIdempotent);
    }
    Ok(())
}

async fn check_setup_metrics<E: LLMEngine>(engine: &E) -> Result<(), ConformanceFailure> {
    let make_ctx = |metrics: &'static EngineMetrics| crate::engine::MetricsCtx {
        model: "test-model",
        component: "test",
        model_load_time_seconds: 0.0,
        metrics,
    };
    // Leaking is fine in a test — the EngineMetrics handle is short-lived
    // and we need a 'static borrow for both calls. Alternative would be
    // separate `EngineMetrics` per call with a thread_local; cleaner to leak.
    let metrics: &'static EngineMetrics = Box::leak(Box::new(EngineMetrics::from_hierarchy(
        TestHierarchy::new(),
    )));

    let bindings_a = engine
        .setup_metrics(make_ctx(metrics))
        .await
        .map_err(|e| SetupMetricsFailed(e.to_string()))?;
    let bindings_b = engine
        .setup_metrics(make_ctx(metrics))
        .await
        .map_err(|e| SetupMetricsFailed(e.to_string()))?;

    if bindings_a.dp_ranks != bindings_b.dp_ranks {
        return Err(ComponentMetricsNotIdempotent);
    }
    // `on_publisher_ready` callbacks from both bindings are dropped without
    // invocation — they're FnOnce, so this just confirms engines aren't
    // capturing side-effects we'd inadvertently fire twice.
    Ok(())
}

async fn check_cancellation<E: LLMEngine>(
    engine: &E,
    model: &str,
    mode: LlmConformanceMode,
    deadline: Duration,
) -> Result<(), ConformanceFailure> {
    // Request enough tokens that an engine which ignores cancellation
    // can't finish naturally before the deadline fires.
    const LONG_MAX_TOKENS: u32 = 10_000;

    let ctx = mock_context();
    let request = match mode {
        LlmConformanceMode::Token => request_with_max_tokens(model, Some(LONG_MAX_TOKENS)),
        LlmConformanceMode::Encode => encode_request(model),
    };
    let stream = engine
        .generate(request, GenerateContext::new(ctx.clone(), None))
        .await
        .map_err(|e| GenerateFailed(e.to_string()))?;

    // Cancel as soon as the stream is live. The engine's body hasn't been
    // polled yet, so its first `is_stopped()` check will observe the flag
    // regardless of engine speed — no timer race.
    ctx.stop_generating();

    let items = tokio::time::timeout(deadline, async {
        let mut s = stream;
        let mut out = Vec::new();
        while let Some(c) = s.next().await {
            out.push(c);
        }
        out
    })
    .await
    .map_err(|_| CancellationNotObserved { after: deadline })?;

    match items.last() {
        Some(Ok(c)) if matches!(c.finish_reason, Some(FinishReason::Cancelled)) => Ok(()),
        Some(Ok(_)) => Err(CancellationIgnored),
        Some(Err(e)) => Err(StreamYieldedError(e.to_string())),
        None => Err(NoChunksYielded),
    }
}

// ---------------------------------------------------------------------------
// RawEngine conformance — the raw-media analog of the suite above.
//
// Raw responses are opaque JSON objects: no `finish_reason`, no token
// bookkeeping, and no `kv_event_sources`. So the raw checks pin only the
// modality-neutral contract — a well-formed stream, concurrency safety, and
// prompt cancellation (measured purely as "the stream terminates", since
// there's no Cancelled marker to assert).
// ---------------------------------------------------------------------------

/// Run the raw-media conformance suite against a [`RawEngine`] implementation.
///
/// Like [`run_conformance`], takes a factory so the kit can build a second,
/// pristine engine for the "cleanup before start" check.
pub async fn run_raw_conformance<E, F>(mut factory: F) -> Result<(), ConformanceFailure>
where
    E: RawEngine,
    F: FnMut() -> E,
{
    let engine = factory();

    // 1. start() returns a non-empty model and — for a raw engine — leaves
    //    the token-pipeline `llm` registration sub-record unset.
    let config = engine
        .start(0)
        .await
        .map_err(|e| StartFailed(e.to_string()))?;
    if config.model.is_empty() {
        return Err(EmptyModelInConfig);
    }
    if config.llm.is_some() {
        return Err(RawEngineAdvertisedLlmRegistration);
    }

    // 2. A plain generate() yields a well-formed (non-empty, all-Ok) stream.
    check_single_generate_raw(&engine).await?;

    // 3. Interleaved generate() calls all complete — catches shared-state bugs.
    check_concurrent_generates_raw(&engine).await?;

    // 4. Cancellation is observed within a bounded deadline.
    check_cancellation_raw(&engine, DEFAULT_CANCEL_DEADLINE).await?;

    // 5. cleanup() succeeds and is idempotent.
    engine
        .cleanup()
        .await
        .map_err(|e| CleanupFailed(e.to_string()))?;
    engine
        .cleanup()
        .await
        .map_err(|e| SecondCleanupFailed(e.to_string()))?;

    // 6. cleanup() is safe on a never-started engine — mirrors Worker's
    //    post-start-failure path.
    let fresh = factory();
    fresh
        .cleanup()
        .await
        .map_err(|e| CleanupWithoutStartFailed(e.to_string()))?;

    Ok(())
}

/// A raw request body. `steps` lets the cancellation check ask for a long
/// stream that only ends early if the engine observes cancellation; engines
/// that don't read it just serve their normal (short) response.
fn raw_request(steps: Option<u64>) -> serde_json::Value {
    match steps {
        Some(n) => serde_json::json!({ "prompt": "ping", "steps": n }),
        None => serde_json::json!({ "prompt": "ping" }),
    }
}

async fn check_single_generate_raw<E: RawEngine>(engine: &E) -> Result<(), ConformanceFailure> {
    let ctx = mock_context();
    let stream = engine
        .generate(raw_request(None), GenerateContext::new(ctx, None))
        .await
        .map_err(|e| GenerateFailed(e.to_string()))?;
    let items: Vec<_> = stream.collect().await;
    if items.is_empty() {
        return Err(NoChunksYielded);
    }
    for item in items {
        if let Err(e) = item {
            return Err(StreamYieldedError(e.to_string()));
        }
    }
    Ok(())
}

async fn check_concurrent_generates_raw<E: RawEngine>(
    engine: &E,
) -> Result<(), ConformanceFailure> {
    const CONCURRENT: usize = 8;
    let futs = (0..CONCURRENT).map(|_| async {
        let ctx = mock_context();
        let stream = engine
            .generate(raw_request(None), GenerateContext::new(ctx, None))
            .await
            .map_err(|e| ConcurrentGenerateFailed(e.to_string()))?;
        let n = stream.count().await;
        if n == 0 {
            Err(ConcurrentGenerateFailed("stream was empty".to_string()))
        } else {
            Ok(())
        }
    });
    for result in futures::future::join_all(futs).await {
        result?;
    }
    Ok(())
}

async fn check_cancellation_raw<E: RawEngine>(
    engine: &E,
    deadline: Duration,
) -> Result<(), ConformanceFailure> {
    // Ask for a long stream so an engine that ignores cancellation can't
    // drain it within the deadline.
    const LONG_STEPS: u64 = 100_000;

    let ctx = mock_context();
    let stream = engine
        .generate(
            raw_request(Some(LONG_STEPS)),
            GenerateContext::new(ctx.clone(), None),
        )
        .await
        .map_err(|e| GenerateFailed(e.to_string()))?;

    // Cancel before the first poll, mirroring check_cancellation.
    ctx.stop_generating();

    // Raw responses carry no finish_reason, so conformance is purely "the
    // stream terminates promptly". Draining to completion within the deadline
    // is the success condition.
    tokio::time::timeout(deadline, async {
        let mut s = stream;
        while s.next().await.is_some() {}
    })
    .await
    .map_err(|_| CancellationNotObserved { after: deadline })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{
        EngineConfig, LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest, usage,
    };
    use crate::error::DynamoError;
    use async_trait::async_trait;
    use futures::stream::BoxStream;

    /// Minimal engine that opts out of everything except `start`/`cleanup`
    /// and a custom `setup_metrics`. Other trait methods that
    /// `check_setup_metrics` doesn't touch are stubbed with `unreachable!`.
    struct ConfigurableMetricsEngine {
        dp_ranks: Vec<u32>,
    }

    #[async_trait]
    impl LLMEngine for ConfigurableMetricsEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig {
                model: "mock".to_string(),
                ..EngineConfig::default()
            })
        }
        async fn generate(
            &self,
            _request: PreprocessedRequest,
            _ctx: crate::engine::GenerateContext,
        ) -> Result<
            BoxStream<'static, Result<crate::engine::LLMEngineOutput, DynamoError>>,
            DynamoError,
        > {
            unreachable!()
        }
        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
        async fn setup_metrics(
            &self,
            _ctx: crate::engine::MetricsCtx<'_>,
        ) -> Result<crate::engine::MetricsBindings, DynamoError> {
            Ok(crate::engine::MetricsBindings {
                dp_ranks: self.dp_ranks.clone(),
                on_publisher_ready: None,
            })
        }
    }

    /// Engines that opt out entirely (returning an empty `dp_ranks`) are
    /// acceptable — opt-out is the default.
    #[tokio::test]
    async fn check_setup_metrics_accepts_opt_out() {
        let engine = ConfigurableMetricsEngine { dp_ranks: vec![] };
        let result = check_setup_metrics(&engine).await;
        assert!(result.is_ok(), "opt-out should pass: {:?}", result);
    }

    /// Engines declaring a non-empty rank set pass when stable across calls.
    #[tokio::test]
    async fn check_setup_metrics_accepts_stable_ranks() {
        let engine = ConfigurableMetricsEngine {
            dp_ranks: vec![0, 1, 2],
        };
        assert!(check_setup_metrics(&engine).await.is_ok());
    }

    #[derive(Clone, Copy)]
    enum EncodeMockResponse {
        Valid,
        MissingEncoderResult,
        WrongCount,
        WrongFinish,
        Tokens,
        BadUsage,
        Empty,
    }

    struct EncodeConformanceMock {
        response: EncodeMockResponse,
        honor_cancel: bool,
    }

    fn encode_mock(response: EncodeMockResponse) -> EncodeConformanceMock {
        EncodeConformanceMock {
            response,
            honor_cancel: true,
        }
    }

    fn valid_encode_chunk() -> LLMEngineOutput {
        LLMEngineOutput::encode_terminal(serde_json::Map::from_iter([(
            "handle".to_string(),
            serde_json::json!("sample-encoder:test"),
        )]))
    }

    #[async_trait]
    impl LLMEngine for EncodeConformanceMock {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig {
                model: "encode-mock".to_string(),
                ..EngineConfig::default()
            })
        }

        async fn generate(
            &self,
            request: PreprocessedRequest,
            ctx: GenerateContext,
        ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
            assert!(
                request
                    .multi_modal_data
                    .as_ref()
                    .is_some_and(|data| !data.is_empty()),
                "encode conformance request must contain multi_modal_data"
            );
            assert_eq!(
                request.mm_processor_kwargs,
                Some(serde_json::json!({ "min_pixels": 64 })),
                "encode conformance request must contain mm_processor_kwargs"
            );
            let chunks = match self.response {
                EncodeMockResponse::Valid => vec![Ok(valid_encode_chunk())],
                EncodeMockResponse::MissingEncoderResult => vec![Ok(LLMEngineOutput::stop())],
                EncodeMockResponse::WrongCount => {
                    vec![Ok(valid_encode_chunk()), Ok(valid_encode_chunk())]
                }
                EncodeMockResponse::WrongFinish => {
                    let mut chunk = valid_encode_chunk();
                    chunk.finish_reason = Some(FinishReason::Length);
                    vec![Ok(chunk)]
                }
                EncodeMockResponse::Tokens => {
                    let mut chunk = valid_encode_chunk();
                    chunk.token_ids = vec![1];
                    vec![Ok(chunk)]
                }
                EncodeMockResponse::BadUsage => {
                    vec![Ok(valid_encode_chunk().with_usage(usage(3, 1)))]
                }
                EncodeMockResponse::Empty => vec![],
            };
            let honor_cancel = self.honor_cancel;
            let ctx = ctx.inner_arc();
            Ok(Box::pin(async_stream::stream! {
                if honor_cancel && ctx.is_stopped() {
                    yield Ok(LLMEngineOutput::cancelled());
                    return;
                }
                for chunk in chunks {
                    yield chunk;
                }
            }))
        }

        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn encode_mock_without_usage_satisfies_conformance() {
        run_encode_conformance(|| encode_mock(EncodeMockResponse::Valid))
            .await
            .expect("encode conformance");
    }

    #[tokio::test]
    async fn encode_conformance_rejects_missing_encoder_result() {
        let result =
            run_encode_conformance(|| encode_mock(EncodeMockResponse::MissingEncoderResult)).await;
        assert!(
            matches!(result, Err(EncoderResultExpectedObject)),
            "expected EncoderResultExpectedObject, got {result:?}"
        );
    }

    #[tokio::test]
    async fn encode_conformance_rejects_wrong_chunk_count() {
        let engine = encode_mock(EncodeMockResponse::WrongCount);
        let result = check_encode_generate(&engine, "encode-mock").await;
        assert!(matches!(result, Err(EncodeChunkCount { count: 2 })));
    }

    #[tokio::test]
    async fn encode_conformance_rejects_wrong_finish_reason() {
        let engine = encode_mock(EncodeMockResponse::WrongFinish);
        let result = check_encode_generate(&engine, "encode-mock").await;
        assert!(matches!(result, Err(EncodeTerminalExpected)));
    }

    #[tokio::test]
    async fn encode_conformance_rejects_generated_tokens() {
        let engine = encode_mock(EncodeMockResponse::Tokens);
        let result = check_encode_generate(&engine, "encode-mock").await;
        assert!(matches!(result, Err(EncodeTokensNotEmpty { count: 1 })));
    }

    #[tokio::test]
    async fn encode_conformance_rejects_inconsistent_usage() {
        let engine = encode_mock(EncodeMockResponse::BadUsage);
        let result = check_encode_generate(&engine, "encode-mock").await;
        assert!(matches!(result, Err(EncodeUsageMismatch { .. })));
    }

    #[tokio::test]
    async fn encode_conformance_rejects_ignored_cancellation() {
        let engine = EncodeConformanceMock {
            response: EncodeMockResponse::Valid,
            honor_cancel: false,
        };
        let result = check_cancellation(
            &engine,
            "encode-mock",
            LlmConformanceMode::Encode,
            Duration::from_millis(150),
        )
        .await;
        assert!(matches!(result, Err(CancellationIgnored)));
    }

    #[tokio::test]
    async fn encode_conformance_validates_concurrent_streams() {
        let engine = encode_mock(EncodeMockResponse::Empty);
        let result =
            check_concurrent_generates(&engine, "encode-mock", LlmConformanceMode::Encode).await;
        assert!(matches!(result, Err(EncodeChunkCount { count: 0 })));
    }

    /// Minimal `RawEngine` for exercising the raw conformance kit itself.
    /// With `honor_cancel = false`, `generate` ignores `is_stopped()` so the
    /// cancellation check can be shown to have teeth.
    struct RawConformanceMock {
        honor_cancel: bool,
    }

    #[async_trait]
    impl RawEngine for RawConformanceMock {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig {
                model: "raw-mock".to_string(),
                ..EngineConfig::default()
            })
        }

        async fn generate(
            &self,
            request: serde_json::Value,
            context: GenerateContext,
        ) -> Result<BoxStream<'static, Result<serde_json::Value, DynamoError>>, DynamoError>
        {
            let steps = request.get("steps").and_then(|v| v.as_u64()).unwrap_or(2);
            let honor = self.honor_cancel;
            let ctx = context.inner_arc();
            Ok(Box::pin(async_stream::stream! {
                for i in 0..steps {
                    if honor && ctx.is_stopped() {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(2)).await;
                    yield Ok(serde_json::json!({ "progress": i }));
                }
                yield Ok(serde_json::json!({ "data": [{ "url": "data:," }] }));
            }))
        }

        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn raw_mock_satisfies_conformance() {
        run_raw_conformance(|| RawConformanceMock { honor_cancel: true })
            .await
            .expect("raw conformance");
    }

    #[tokio::test]
    async fn raw_conformance_flags_ignored_cancellation() {
        // An engine that never checks is_stopped() can't drain LONG_STEPS
        // within the deadline, so the cancellation check must fail.
        let engine = RawConformanceMock {
            honor_cancel: false,
        };
        let result = check_cancellation_raw(&engine, Duration::from_millis(150)).await;
        assert!(
            matches!(result, Err(CancellationNotObserved { .. })),
            "expected CancellationNotObserved, got {result:?}"
        );
    }
}
