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
pub enum ConformanceFailure {
    StartFailed(String),
    EmptyModelInConfig,
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
        }
    }
}

impl std::error::Error for ConformanceFailure {}

/// In-process microbenchmark driver for the unified-backend bridge.
///
/// Drives [`LLMEngine::generate`] through the production [`EngineAdapter`] with
/// [`mock_context`]-style contexts — no NATS / etcd. Used by
/// `benchmarks/unified_backend/` to isolate the Rust↔Python bridge + GIL cost:
/// a `PyLLMEngine`-wrapped Python engine vs the GIL-free [`BenchFloorEngine`],
/// both through the same path.
pub mod bench {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use async_stream::stream;
    use async_trait::async_trait;
    use futures::stream::{BoxStream, StreamExt};

    use dynamo_runtime::pipeline::{AsyncEngine, Context};

    use crate::adapter::EngineAdapter;
    use crate::disagg::DisaggregationMode;
    use crate::engine::{
        EngineConfig, GenerateContext, LLMEngine, LLMEngineOutput, LLMEngineOutputExt,
        OutputOptions, PreprocessedRequest, SamplingOptions, StopConditions, TopLogprob, chunk,
        usage,
    };
    use crate::error::DynamoError;

    /// Cap on synthesised top-k alternatives; matches `sample_engine.py`.
    const MAX_LOGPROBS: u32 = 20;

    /// One benchmark configuration point.
    #[derive(Clone, Debug)]
    pub struct BenchWorkload {
        pub model: String,
        /// Number of prompt token IDs to send.
        pub prompt_len: usize,
        /// Tokens each request generates.
        pub max_tokens: u32,
        /// Synthetic top-k logprobs per token, or `None` to omit logprobs.
        pub logprobs_k: Option<u32>,
        /// In-flight requests held concurrently (the GIL-contention knob).
        pub concurrency: usize,
        /// Total requests to run before the measurement window closes.
        pub total_requests: usize,
    }

    /// Aggregated results of one [`run_load`] invocation.
    #[derive(Clone, Debug)]
    pub struct BenchStats {
        pub requests: usize,
        pub total_output_tokens: usize,
        pub wall_seconds: f64,
        pub tokens_per_sec: f64,
        pub ttft_p50_ms: f64,
        pub ttft_p99_ms: f64,
        pub itl_p50_ms: f64,
        pub itl_p99_ms: f64,
    }

    struct ReqStat {
        /// `None` when the request produced no token-bearing chunks — kept out
        /// of the TTFT pool rather than recorded as a spurious 0.0.
        ttft_ms: Option<f64>,
        itl_ms: Vec<f64>,
        output_tokens: usize,
    }

    /// Nearest-rank percentile (matches `adapter::record_itl_distribution`).
    fn nearest_rank(samples: &mut [f64], frac: f64) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (((samples.len() - 1) as f64) * frac).round() as usize;
        samples[idx]
    }

    fn build_request(w: &BenchWorkload) -> PreprocessedRequest {
        let token_ids: Vec<u32> = (0..w.prompt_len as u32).map(|i| i % 32000).collect();
        PreprocessedRequest::builder()
            .model(w.model.clone())
            .token_ids(token_ids)
            .stop_conditions(StopConditions {
                max_tokens: Some(w.max_tokens),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions {
                logprobs: w.logprobs_k,
                ..Default::default()
            })
            .build()
            .expect("build benchmark request")
    }

    /// Run a single request through the engine, recording TTFT, per-token
    /// ITL, and the streamed token count. `None` on a setup error.
    async fn run_one(adapter: &EngineAdapter, w: &BenchWorkload) -> Option<ReqStat> {
        let input = Context::new(build_request(w));
        let t0 = Instant::now();
        let mut stream = match adapter.generate(input).await {
            Ok(s) => s,
            Err(e) => {
                // Surface rather than silently drop — an excluded request would
                // otherwise inflate the reported throughput with no diagnostic.
                tracing::warn!(error = %e, "bench request generate() failed; excluded from sample");
                return None;
            }
        };
        let mut ttft_ms: Option<f64> = None;
        let mut last = t0;
        let mut itl_ms = Vec::new();
        let mut output_tokens = 0usize;
        while let Some(ann) = stream.next().await {
            // Count only token-bearing chunks for TTFT/ITL, matching
            // `EngineAdapter`'s ITL recording — a token-less chunk (e.g. an
            // empty terminal or a bootstrap handshake) is not a token event.
            let n = ann.data.as_ref().map_or(0, |c| c.token_ids.len());
            if n == 0 {
                continue;
            }
            let now = Instant::now();
            match ttft_ms {
                None => ttft_ms = Some((now - t0).as_secs_f64() * 1000.0),
                Some(_) => itl_ms.push((now - last).as_secs_f64() * 1000.0),
            }
            last = now;
            output_tokens += n;
        }
        Some(ReqStat {
            ttft_ms,
            itl_ms,
            output_tokens,
        })
    }

    /// Drive an `LLMEngine` through the production [`EngineAdapter`] (the
    /// unified-backend path) holding `workload.concurrency` requests in flight
    /// until `total_requests` complete, returning aggregated stats.
    ///
    /// Must be called from within a tokio runtime (uses `tokio::spawn`).
    pub async fn run_load(
        engine: Arc<dyn LLMEngine>,
        mode: DisaggregationMode,
        workload: BenchWorkload,
    ) -> BenchStats {
        let adapter = Arc::new(EngineAdapter::new(engine, mode));
        let next = Arc::new(AtomicUsize::new(0));
        let collected = Arc::new(Mutex::new(Vec::<ReqStat>::with_capacity(
            workload.total_requests,
        )));
        let workload = Arc::new(workload);

        let start = Instant::now();
        let mut handles = Vec::with_capacity(workload.concurrency);
        for _ in 0..workload.concurrency {
            let adapter = adapter.clone();
            let next = next.clone();
            let collected = collected.clone();
            let workload = workload.clone();
            handles.push(tokio::spawn(async move {
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= workload.total_requests {
                        break;
                    }
                    if let Some(stat) = run_one(&adapter, &workload).await {
                        collected
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push(stat);
                    }
                }
            }));
        }
        for h in handles {
            let _ = h.await;
        }
        let wall_seconds = start.elapsed().as_secs_f64();

        let collected = Arc::try_unwrap(collected)
            .map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner()))
            .unwrap_or_default();

        let total_output_tokens: usize = collected.iter().map(|s| s.output_tokens).sum();
        let mut ttfts: Vec<f64> = collected.iter().filter_map(|s| s.ttft_ms).collect();
        let mut itls: Vec<f64> = collected
            .iter()
            .flat_map(|s| s.itl_ms.iter().copied())
            .collect();

        BenchStats {
            requests: collected.len(),
            total_output_tokens,
            wall_seconds,
            tokens_per_sec: if wall_seconds > 0.0 {
                total_output_tokens as f64 / wall_seconds
            } else {
                0.0
            },
            ttft_p50_ms: nearest_rank(&mut ttfts, 0.50),
            ttft_p99_ms: nearest_rank(&mut ttfts, 0.99),
            itl_p50_ms: nearest_rank(&mut itls, 0.50),
            itl_p99_ms: nearest_rank(&mut itls, 0.99),
        }
    }

    /// Synthetic logprobs matching `sample_engine.py`'s shape, so the floor's
    /// per-chunk payload size matches the Python engine's.
    fn stamp_logprobs(out: &mut LLMEngineOutput, token_id: u32, top_k: u32) {
        let top_k = top_k.min(MAX_LOGPROBS);
        let selected_lp = -0.1 * f64::from(token_id % 10);
        out.log_probs = Some(vec![selected_lp]);
        if top_k > 0 {
            let mut entries: Vec<TopLogprob> = Vec::with_capacity(top_k as usize + 1);
            entries.push(TopLogprob {
                rank: 1,
                token_id,
                token: Some(format!("token_id:{token_id}")),
                logprob: selected_lp,
                bytes: None,
            });
            for r in 1..=top_k {
                let alt_id = (token_id + r) % 32000;
                entries.push(TopLogprob {
                    rank: r + 1,
                    token_id: alt_id,
                    token: Some(format!("token_id:{alt_id}")),
                    logprob: selected_lp - 0.1 * f64::from(r),
                    bytes: None,
                });
            }
            out.top_logprobs = Some(vec![entries]);
        }
    }

    /// GIL-free reference engine: emits `max_tokens` rotating token IDs (with
    /// optional per-token delay and synthetic logprobs), mirroring
    /// `sample_engine.py`'s output minus the Python. The delta vs the
    /// `PyLLMEngine`-wrapped engine through [`run_load`] is the bridge + GIL cost.
    pub struct BenchFloorEngine {
        per_token_delay: Duration,
    }

    impl BenchFloorEngine {
        /// `per_token_delay_ms` should match `sample_engine.py`'s `--delay`
        /// for an apples-to-apples comparison; pass `0.0` to expose pure
        /// bridge + GIL overhead with no pacing.
        pub fn new(per_token_delay_ms: f64) -> Self {
            Self {
                per_token_delay: Duration::from_secs_f64((per_token_delay_ms / 1000.0).max(0.0)),
            }
        }
    }

    #[async_trait]
    impl LLMEngine for BenchFloorEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig {
                model: "bench-floor".to_string(),
                ..Default::default()
            })
        }

        async fn generate(
            &self,
            request: PreprocessedRequest,
            ctx: GenerateContext,
        ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
            // Clamp to >= 1 so the stream always yields a terminal chunk
            // (max_tokens == 0 would otherwise produce an empty stream).
            let max_new = request.stop_conditions.max_tokens.unwrap_or(16).max(1);
            let logprobs_k = request.output_options.logprobs;
            let prompt_len = request.token_ids.len() as u32;
            let delay = self.per_token_delay;

            let s = stream! {
                for i in 0..max_new {
                    if ctx.is_stopped() {
                        yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_len, i)));
                        return;
                    }
                    if !delay.is_zero() {
                        tokio::time::sleep(delay).await;
                    }
                    let token_id = (i + 1) % 32000;
                    let mut out = if i == max_new - 1 {
                        LLMEngineOutput::length()
                            .with_tokens(vec![token_id])
                            .with_usage(usage(prompt_len, max_new))
                    } else {
                        chunk::token(token_id)
                    };
                    if let Some(k) = logprobs_k {
                        stamp_logprobs(&mut out, token_id, k);
                    }
                    yield Ok(out);
                }
            };
            Ok(Box::pin(s))
        }

        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// `run_load` drives the floor through the real `EngineAdapter`,
        /// counts every streamed token, and reports positive throughput.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn floor_run_load_counts_all_tokens() {
            let engine: Arc<dyn LLMEngine> = Arc::new(BenchFloorEngine::new(0.0));
            let workload = BenchWorkload {
                model: "bench-model".to_string(),
                prompt_len: 16,
                max_tokens: 8,
                logprobs_k: Some(3),
                concurrency: 4,
                total_requests: 32,
            };
            let stats = run_load(engine, DisaggregationMode::Aggregated, workload).await;

            assert_eq!(stats.requests, 32);
            // Each request yields one token per step for `max_tokens` steps.
            assert_eq!(stats.total_output_tokens, 32 * 8);
            assert!(stats.tokens_per_sec > 0.0, "stats: {stats:?}");
            assert!(stats.itl_p50_ms >= 0.0);
        }
    }
}

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
    check_concurrent_generates(&engine, &config.model).await?;

    // 6. Cancellation is observed within a bounded deadline.
    check_cancellation(&engine, &config.model, DEFAULT_CANCEL_DEADLINE).await?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineConfig, PreprocessedRequest};
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
}
