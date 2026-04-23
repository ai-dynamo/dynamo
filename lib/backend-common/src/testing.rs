// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conformance test kit for [`LLMEngine`] implementations.
//!
//! Engines wire themselves into the test suite with one call:
//!
//! ```ignore
//! #[tokio::test]
//! async fn my_engine_satisfies_contract() {
//!     let engine = MyEngine::new_for_test();
//!     dynamo_backend_common::testing::run_conformance(engine)
//!         .await
//!         .expect("conformance");
//! }
//! ```
//!
//! Gated behind the `testing` cargo feature; intended for `[dev-dependencies]`.

use std::sync::Arc;
use std::time::Duration;

use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_llm::protocols::common::{FinishReason, OutputOptions, SamplingOptions, StopConditions};
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use futures::StreamExt;

use crate::engine::LLMEngine;
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
    ConcurrentGenerateFailed(String),
    CancellationNotObserved { after: Duration },
    CancellationIgnored,
    CleanupFailed(String),
    SecondCleanupFailed(String),
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
        }
    }
}

impl std::error::Error for ConformanceFailure {}

/// Run the full conformance suite against an engine.
pub async fn run_conformance<E: LLMEngine>(engine: E) -> Result<(), ConformanceFailure> {
    // 1. start() returns non-empty model.
    let config = engine
        .start()
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

    // 5. cleanup() succeeds and is idempotent.
    engine
        .cleanup()
        .await
        .map_err(|e| CleanupFailed(e.to_string()))?;
    engine
        .cleanup()
        .await
        .map_err(|e| SecondCleanupFailed(e.to_string()))?;

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
        .generate(request(model), ctx)
        .await
        .map_err(|e| GenerateFailed(e.to_string()))?;
    let chunks: Vec<_> = stream.collect().await;

    if chunks.is_empty() {
        return Err(NoChunksYielded);
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
            .generate(request(model), ctx)
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
            ctx.clone(),
        )
        .await
        .map_err(|e| GenerateFailed(e.to_string()))?;

    // Cancel as soon as the stream is live. The engine's body hasn't been
    // polled yet, so its first `is_stopped()` check will observe the flag
    // regardless of engine speed — no timer race.
    ctx.stop_generating();

    let chunks = tokio::time::timeout(deadline, async {
        let mut s = stream;
        let mut out = Vec::new();
        while let Some(c) = s.next().await {
            out.push(c);
        }
        out
    })
    .await
    .map_err(|_| CancellationNotObserved { after: deadline })?;

    match chunks.last() {
        Some(c) if matches!(c.finish_reason, Some(FinishReason::Cancelled)) => Ok(()),
        Some(_) => Err(CancellationIgnored),
        None => Err(NoChunksYielded),
    }
}
