// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conformance test kit for [`LLMEngine`] implementations.
//!
//! Engines wire themselves into the test suite with one call:
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

use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_llm::protocols::common::{FinishReason, OutputOptions, SamplingOptions, StopConditions};
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use futures::StreamExt;

use crate::engine::{GenerateContext, LLMEngine};
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
pub enum ConformanceFailure {
    StartFailed(String),
    EmptyModelInConfig,
    GenerateFailed(String),
    NoChunksYielded,
    ChunkAfterTerminal,
    NoTerminalChunk,
    StreamYieldedError(String),
    ConcurrentGenerateFailed(String),
    CancellationNotObserved { after: Duration },
    CancellationIgnored,
    CleanupFailed(String),
    SecondCleanupFailed(String),
    CleanupWithoutStartFailed(String),
    KvEventSourcesFailed(String),
    KvEventSourcesNotIdempotent,
    MetricsSourcesFailed(String),
    MetricsSourcesNotIdempotent,
    MetricsSnapshotTooSlow { took: Duration },
}

impl std::fmt::Display for ConformanceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartFailed(m) => write!(f, "start() failed: {m}"),
            EmptyModelInConfig => write!(f, "EngineConfig.model is empty"),
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
            MetricsSourcesFailed(m) => write!(f, "metrics_sources() failed: {m}"),
            MetricsSourcesNotIdempotent => write!(
                f,
                "metrics_sources() returned different dp_rank set on a second call \
                 (the descriptor list must be stable for the engine's lifetime)"
            ),
            MetricsSnapshotTooSlow { took } => write!(
                f,
                "SnapshotSource.snapshot took {took:?} (must be a cheap field read, \
                 < 1 ms; an engine-internal call would land in the 10s of ms)"
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

    // 2. A plain generate() yields a well-formed stream ending in a terminal chunk.
    check_single_generate(&engine, &config.model).await?;

    // 3. Interleaved generate() calls both complete — catches shared-state bugs.
    //    Uses tokio::join! under the test runtime (single-threaded by default),
    //    so this is interleaving rather than true parallelism.
    check_concurrent_generates(&engine, &config.model).await?;

    // 4. Cancellation is observed within a bounded deadline.
    check_cancellation(&engine, &config.model, DEFAULT_CANCEL_DEADLINE).await?;

    // 5. KV-aware-routing source descriptors satisfy their contracts:
    //    - kv_event_sources / metrics_sources don't error
    //    - rank sets are stable across repeated calls
    //    - SnapshotSource.snapshot is a cheap field read (< 1 ms)
    check_kv_event_sources(&engine).await?;
    check_metrics_sources(&engine).await?;

    // 6. cleanup() succeeds and is idempotent.
    engine
        .cleanup()
        .await
        .map_err(|e| CleanupFailed(e.to_string()))?;
    engine
        .cleanup()
        .await
        .map_err(|e| SecondCleanupFailed(e.to_string()))?;

    // 6. cleanup() is safe on a never-started engine — mirrors the path
    //    `Worker` takes after `start()` raises. Engines must guard each
    //    allocated resource with a null-check.
    let fresh = factory();
    fresh
        .cleanup()
        .await
        .map_err(|e| CleanupWithoutStartFailed(e.to_string()))?;

    Ok(())
}

fn request(model: &str) -> PreprocessedRequest {
    request_with_max_tokens(model, None)
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
    match terminal_idx {
        Some(i) if i == chunks.len() - 1 => Ok(()),
        Some(_) => Err(ChunkAfterTerminal),
        None => Err(NoTerminalChunk),
    }
}

async fn check_concurrent_generates<E: LLMEngine>(
    engine: &E,
    model: &str,
) -> Result<(), ConformanceFailure> {
    // 8 in-flight streams — enough to catch state-tramping under interleaved
    // polls. Under a single-threaded test runtime this is interleaving rather
    // than true parallelism, but it still exercises shared-state correctness.
    const CONCURRENT: usize = 8;
    let futs = (0..CONCURRENT).map(|_| async {
        let ctx = mock_context();
        let stream = engine
            .generate(request(model), GenerateContext::new(ctx, None))
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

/// Ceiling for a snapshot read. An engine that accidentally calls into its
/// underlying inference engine here lands in the 10s of ms and stalls the
/// publish loop.
const SNAPSHOT_MAX_LATENCY: Duration = Duration::from_millis(1);

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

async fn check_metrics_sources<E: LLMEngine>(engine: &E) -> Result<(), ConformanceFailure> {
    let first = engine
        .metrics_sources()
        .await
        .map_err(|e| MetricsSourcesFailed(e.to_string()))?;
    let second = engine
        .metrics_sources()
        .await
        .map_err(|e| MetricsSourcesFailed(e.to_string()))?;
    let ranks_a: Vec<u32> = first.iter().map(|s| s.dp_rank).collect();
    let ranks_b: Vec<u32> = second.iter().map(|s| s.dp_rank).collect();
    if ranks_a != ranks_b {
        return Err(MetricsSourcesNotIdempotent);
    }
    // Probe snapshot latency on every returned source. The closure is what
    // `Worker` invokes under the GIL on a tokio interval; if it's slow
    // here it'll stall the publish loop in production. Take min-of-3 so
    // a contended CI runner doesn't flake on a single-sample outlier.
    for src in &first {
        let took = (0..3)
            .map(|_| {
                let started = std::time::Instant::now();
                let _ = (src.snapshot)();
                started.elapsed()
            })
            .min()
            .unwrap_or_default();
        if took > SNAPSHOT_MAX_LATENCY {
            return Err(MetricsSnapshotTooSlow { took });
        }
    }
    Ok(())
}

async fn check_cancellation<E: LLMEngine>(
    engine: &E,
    model: &str,
    deadline: Duration,
) -> Result<(), ConformanceFailure> {
    // Request enough tokens that an engine which ignores cancellation
    // can't finish naturally before the deadline fires.
    const LONG_MAX_TOKENS: u32 = 10_000;

    let ctx = mock_context();
    let stream = engine
        .generate(
            request_with_max_tokens(model, Some(LONG_MAX_TOKENS)),
            GenerateContext::new(ctx.clone(), None),
        )
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
