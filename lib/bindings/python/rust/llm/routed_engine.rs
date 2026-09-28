// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, LazyLock};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};
use tokio_stream::StreamExt;
use tracing::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use dynamo_llm::entrypoint::PrefillRoutedEngine;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_llm::protocols::common::timing::RequestTracker;
use dynamo_llm::request_trace;
use dynamo_runtime::logging::{DistributedTraceContext, otel_parent_context_from_distributed};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated as RsAnnotated;

use crate::to_pyerr;

#[pyclass]
pub struct RoutedEngine {
    inner: PrefillRoutedEngine,
    kv_cache_block_size: usize,
}

impl RoutedEngine {
    pub fn new(inner: PrefillRoutedEngine, kv_cache_block_size: usize) -> Self {
        Self {
            inner,
            kv_cache_block_size,
        }
    }
}

// Bound work submitted to Tokio's otherwise unbounded blocking-task queue.
// The byte budget covers retained prompt vectors, including spare capacity.
const MAX_PENDING_REPLAYS: usize = 4;
const REPLAY_MEMORY_KIB: usize = 32 * 1024;
static REPLAY_TASKS: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(MAX_PENDING_REPLAYS)));
static REPLAY_MEMORY: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(REPLAY_MEMORY_KIB)));

/// Reserve hashing work without making inference wait for trace capacity.
fn reserve_replay(
    tasks: &Arc<Semaphore>,
    memory: &Arc<Semaphore>,
    token_capacity: usize,
) -> Option<(OwnedSemaphorePermit, OwnedSemaphorePermit)> {
    let kib = token_capacity
        .checked_mul(std::mem::size_of::<dynamo_llm::protocols::TokenIdType>())?
        .div_ceil(1024);
    let task = tasks.clone().try_acquire_owned().ok()?;
    let memory = memory
        .clone()
        .try_acquire_many_owned(u32::try_from(kib).ok()?)
        .ok()?;
    Some((task, memory))
}

struct PendingRequestEnd {
    data: Option<RequestEndData>,
    output_tokens: usize,
}

struct RequestEndData {
    request_id: String,
    tracker: Arc<RequestTracker>,
    token_ids: Arc<Vec<dynamo_llm::protocols::TokenIdType>>,
    kv_cache_block_size: usize,
    replayable: bool,
}

impl PendingRequestEnd {
    /// Count raw output across router attempts before Python post-processing.
    fn record_output(&mut self, token_count: usize) {
        if token_count > 0 {
            self.tracker().record_first_token();
            self.output_tokens = self.output_tokens.saturating_add(token_count);
        }
    }

    fn tracker(&self) -> &RequestTracker {
        self.data
            .as_ref()
            .expect("request end data")
            .tracker
            .as_ref()
    }
}

impl Drop for PendingRequestEnd {
    fn drop(&mut self) {
        let Some(data) = self.data.take() else {
            return;
        };
        // Router attempts overwrite their individual output counts. The raw
        // stream total includes tokens delivered before a worker migration.
        if (self.output_tokens as u64) > data.tracker.osl_tokens() {
            data.tracker.record_osl(self.output_tokens);
        }
        // The routed stream has been dropped before this guard, so the worker
        // and router have finished writing into the shared tracker. Hashing a
        // long prompt and writing a trace must never delay the client stream.
        if data.tracker.total_time_ms().is_none() {
            data.tracker.record_finish();
        }
        let RequestEndData {
            request_id,
            tracker,
            token_ids,
            kv_cache_block_size,
            replayable,
        } = data;
        if replayable
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
            && let Some(permits) =
                reserve_replay(&REPLAY_TASKS, &REPLAY_MEMORY, token_ids.capacity())
        {
            runtime.spawn_blocking(move || {
                let _permits = permits;
                request_trace::emit_python_routed_request_end(
                    request_id,
                    &tracker,
                    &token_ids,
                    kv_cache_block_size,
                    true,
                );
            });
        } else {
            // Preserve request-end metadata even when hashing is unavailable.
            // This publishes to the bounded trace bus without hashing or I/O.
            request_trace::emit_python_routed_request_end(
                request_id,
                &tracker,
                &[],
                kv_cache_block_size,
                false,
            );
        }
    }
}

fn replayable_text_request(request: &PreprocessedRequest, block_size: usize) -> bool {
    block_size != 0
        && request.prompt_embeds.is_none()
        && request.multi_modal_data.is_none()
        && request.multi_modal_uuids.is_none()
        && request.mm_routing_info.is_none()
        && request.media_io_kwargs.is_none()
        && !request.extra_args.as_ref().is_some_and(|args| {
            args.get("mm_placeholders").is_some()
                || args.get("mm_hashes").is_some()
                || args.get("expanded_token_ids").is_some()
        })
        && request.sampling_options.n.unwrap_or(1) == 1
        && request.sampling_options.best_of.unwrap_or(1) == 1
}

#[pymethods]
impl RoutedEngine {
    /// Send a preprocessed request through the Rust prefill-routed pipeline.
    #[pyo3(signature = (preprocessed, context=None))]
    fn generate<'p>(
        &self,
        py: Python<'p>,
        preprocessed: PyObject,
        context: Option<crate::context::Context>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let mut request: PreprocessedRequest =
            depythonize(preprocessed.bind(py)).map_err(to_pyerr)?;
        let trace = if request_trace::config::capture_enabled()
            && request_trace::policy().emit_request_end_records()
        {
            let tracker = request
                .tracker
                .get_or_insert_with(|| Arc::new(RequestTracker::new()))
                .clone();
            let token_ids = request.token_ids.clone();
            let replayable = replayable_text_request(&request, self.kv_cache_block_size);
            Some((tracker, token_ids, replayable))
        } else {
            None
        };
        let request_context = if let Some(parent_context) = context.as_ref() {
            let parent_metadata = parent_context.metadata_snapshot();
            let parent_context = parent_context.inner();
            let child_context = SingleIn::with_id_and_metadata(
                request,
                parent_context.id().to_string(),
                parent_metadata,
            );
            let child_controller = child_context.context();
            parent_context.link_child(child_controller.clone());
            if parent_context.is_killed() {
                child_controller.kill();
            } else if parent_context.is_stopped() {
                child_controller.stop_generating();
            }
            child_context
        } else {
            SingleIn::new(request)
        };
        let trace = trace.map(|(tracker, token_ids, replayable)| {
            (
                request_context.id().to_string(),
                tracker,
                token_ids,
                replayable,
            )
        });
        let kv_cache_block_size = self.kv_cache_block_size;
        let inner = self.inner.clone();

        // Re-parent onto the caller's trace: this future runs on a fresh
        // Tokio task where Span::current() is empty (ai-dynamo/dynamo#11397).
        let dispatch_span = dispatch_span(
            context.as_ref().and_then(|c| c.trace_context()),
            request_context.id(),
        );

        crate::future_into_py(
            py,
            async move {
                let mut stream = inner.generate(request_context).await.map_err(to_pyerr)?;
                let mut trace =
                    trace.map(
                        |(request_id, tracker, token_ids, replayable)| PendingRequestEnd {
                            data: Some(RequestEndData {
                                request_id,
                                tracker,
                                token_ids,
                                kv_cache_block_size,
                                replayable,
                            }),
                            output_tokens: 0,
                        },
                    );
                let task_context = stream.context();
                let (tx, rx) = tokio::sync::mpsc::channel::<RsAnnotated<PyObject>>(32);

                tokio::spawn(async move {
                    loop {
                        let response = tokio::select! {
                            _ = tx.closed() => {
                                task_context.stop_generating();
                                break;
                            }
                            response = stream.next() => response,
                        };

                        let Some(response) = response else {
                            break;
                        };

                        if let (Some(trace), Some(output)) =
                            (trace.as_mut(), response.data.as_ref())
                        {
                            trace.record_output(output.token_ids.len());
                        }

                        let py_response = Python::with_gil(|py| {
                            response.map_data(|data| {
                                pythonize(py, &data)
                                    .map(|obj| obj.unbind())
                                    .map_err(|e| format!("pythonize failed: {e}"))
                            })
                        });

                        if tx.send(py_response).await.is_err() {
                            task_context.stop_generating();
                            break;
                        }
                    }
                    drop(stream);
                    drop(trace);
                });

                Ok(crate::AsyncResponseStream::new(rx, true))
            }
            .instrument(dispatch_span),
        )
    }
}

fn dispatch_span(
    trace_context: Option<&DistributedTraceContext>,
    request_id: &str,
) -> tracing::Span {
    match trace_context {
        Some(tc) => {
            let span = tracing::info_span!(
                target: "request_span",
                "routed_engine.generate",
                request_id = request_id,
                trace_id = tc.trace_id.as_str(),
                parent_id = tc.span_id.as_str(),
                trace_flags = tc.trace_flags.as_str(),
                tracestate = tc.tracestate.as_deref(),
                x_request_id = tc.x_request_id.as_deref(),
            );
            if let Some(context) = otel_parent_context_from_distributed(tc) {
                let _ = span.set_parent(context);
            }
            span
        }
        None => tracing::Span::current(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_runtime::logging::{DistributedTraceIdLayer, inject_trace_headers_into_map};
    use opentelemetry::trace::{TraceContextExt, TraceId, TracerProvider as _};
    use tracing_subscriber::layer::SubscriberExt;

    #[test]
    fn request_end_emits_replay_and_metadata_under_overload_and_cancellation() {
        // Isolate the process-wide trace policy and bus from other tests.
        const CHILD: &str = "DYNAMO_ROUTED_TRACE_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "llm::routed_engine::tests::request_end_emits_replay_and_metadata_under_overload_and_cancellation",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("DYN_REQUEST_TRACE", "true")
                .env("DYN_REQUEST_TRACE_CAPACITY", "1024")
                .env("DYN_REQUEST_TRACE_RECORDS", "request_end")
                .env("DYN_REQUEST_TRACE_SINKS", "stderr")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            for line in String::from_utf8_lossy(&output.stderr)
                .lines()
                .filter(|line| line.starts_with("200 simulated"))
            {
                eprintln!("{line}");
            }
            return;
        }

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            use std::time::Duration;
            use tokio_util::sync::CancellationToken;

            let shutdown = CancellationToken::new();
            request_trace::init_from_env_with_shutdown(shutdown.clone())
                .await
                .unwrap();
            let mut rows = request_trace::subscribe();
            let make_pending = |id: &str| PendingRequestEnd {
                data: Some(RequestEndData {
                    request_id: id.to_string(),
                    tracker: Arc::new(RequestTracker::new()),
                    token_ids: Arc::new(vec![1, 2, 3, 4]),
                    kv_cache_block_size: 2,
                    replayable: true,
                }),
                output_tokens: 0,
            };

            let mut migrated = make_pending("migration");
            migrated.tracker().record_osl(2);
            migrated.record_output(2);
            migrated.tracker().record_osl(3);
            migrated.record_output(3);
            drop(migrated);
            let row = tokio::time::timeout(Duration::from_secs(5), rows.recv())
                .await
                .unwrap()
                .unwrap();
            let request = row.request.unwrap();
            assert_eq!(request.request_id, "migration");
            assert_eq!(request.output_tokens, Some(5));
            assert!(request.replay.is_some());
            assert!(request.ttft_ms.is_some());
            assert!(request.total_time_ms.is_some());

            // Wait for the hashing closure to return its permits after publishing.
            let all_tasks = tokio::time::timeout(
                Duration::from_secs(5),
                REPLAY_TASKS
                    .clone()
                    .acquire_many_owned(MAX_PENDING_REPLAYS as u32),
            )
            .await
            .unwrap()
            .unwrap();
            let all_memory = tokio::time::timeout(Duration::from_secs(5),
                REPLAY_MEMORY.clone().acquire_many_owned(REPLAY_MEMORY_KIB as u32))
                .await.unwrap().unwrap();
            drop(all_memory);
            let mut overloaded = make_pending("overload");
            overloaded.record_output(1);
            drop(overloaded);
            let request = rows.try_recv().unwrap().request.unwrap();
            assert_eq!(request.request_id, "overload");
            assert_eq!(request.output_tokens, Some(1));
            assert!(request.replay.is_none());
            assert!(request.total_time_ms.is_some());
            drop(all_tasks);

            let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
            let pending = make_pending("cancellation");
            let task = tokio::spawn(async move {
                let mut pending = pending;
                pending.record_output(2);
                ready_tx.send(()).unwrap();
                std::future::pending::<()>().await;
                drop(pending);
            });
            ready_rx.await.unwrap();
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
            let row = tokio::time::timeout(Duration::from_secs(5), rows.recv())
                .await
                .unwrap()
                .unwrap();
            let request = row.request.unwrap();
            assert_eq!(request.request_id, "cancellation");
            assert_eq!(request.output_tokens, Some(2));
            assert!(request.replay.is_some());
            assert!(request.total_time_ms.is_some());
            let _all_tasks = tokio::time::timeout(
                Duration::from_secs(5),
                REPLAY_TASKS
                    .clone()
                    .acquire_many_owned(MAX_PENDING_REPLAYS as u32),
            )
            .await
            .unwrap()
            .unwrap();
            let all_memory = tokio::time::timeout(Duration::from_secs(5),
                REPLAY_MEMORY.clone().acquire_many_owned(REPLAY_MEMORY_KIB as u32))
                .await.unwrap().unwrap();
            drop(all_memory);
            assert!(matches!(
                rows.try_recv(),
                Err(tokio::sync::broadcast::error::TryRecvError::Empty)
            ));
            drop(_all_tasks);

            // One-frame responses can complete faster than long prompts hash.
            // Exercise that burst through the real drop guard and trace bus.
            let started = std::time::Instant::now();
            for i in 0..200 {
                let mut pending = make_pending(&format!("burst-{i}"));
                pending.data.as_mut().unwrap().token_ids = Arc::new(vec![1; 131_072]);
                pending.data.as_mut().unwrap().kv_cache_block_size = 64;
                pending.record_output(1);
                drop(pending);
            }
            let submission_time = started.elapsed();
            let mut replay_rows = 0;
            let mut ids = std::collections::HashSet::new();
            for _ in 0..200 {
                let row = tokio::time::timeout(Duration::from_secs(5), rows.recv())
                    .await.unwrap().unwrap();
                let request = row.request.unwrap();
                assert_eq!(request.output_tokens, Some(1));
                assert!(request.total_time_ms.is_some());
                assert!(ids.insert(request.request_id));
                replay_rows += usize::from(request.replay.is_some());
            }
            eprintln!("200 simulated one-frame completion guards, 131072-token prompts: submitted in {submission_time:?}, drained in {:?}; {replay_rows} replay rows, {} metadata-only rows",
                started.elapsed(), 200 - replay_rows);
            shutdown.cancel();
        });
    }

    #[test]
    fn replay_reservation_bounds_tasks_and_retained_memory() {
        let tasks = Arc::new(Semaphore::new(2));
        let memory = Arc::new(Semaphore::new(8));
        let first = reserve_replay(&tasks, &memory, 1024).unwrap();
        let second = reserve_replay(&tasks, &memory, 1024).unwrap();
        assert!(reserve_replay(&tasks, &memory, 1).is_none());
        drop(first);
        assert!(reserve_replay(&tasks, &memory, 1025).is_none());
        // A failed memory reservation must return its task permit.
        assert_eq!(tasks.available_permits(), 1);
        drop(second);
        assert_eq!(tasks.available_permits(), 2);
        assert_eq!(memory.available_permits(), 8);
        assert!(reserve_replay(&tasks, &memory, usize::MAX).is_none());
    }

    #[test]
    fn replay_hashes_only_plain_single_choice_text() {
        let mut request = PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .eos_token_ids(vec![])
            .annotations(vec![])
            .build()
            .unwrap();
        assert!(replayable_text_request(&request, 16));
        assert!(!replayable_text_request(&request, 0));

        request.multi_modal_data = Some(Default::default());
        assert!(!replayable_text_request(&request, 16));
        request.multi_modal_data = None;
        request.extra_args = Some(serde_json::json!({"mm_placeholders": []}));
        assert!(!replayable_text_request(&request, 16));
        request.extra_args = None;
        request.sampling_options.n = Some(2);
        assert!(!replayable_text_request(&request, 16));
    }

    #[test]
    fn request_end_preserves_router_finish_time() {
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_finish();
        let router_elapsed_ms = tracker.total_time_ms();

        drop(PendingRequestEnd {
            data: Some(RequestEndData {
                request_id: "finished-request".to_string(),
                tracker: tracker.clone(),
                token_ids: Arc::new(Vec::new()),
                kv_cache_block_size: 0,
                replayable: false,
            }),
            output_tokens: 0,
        });

        assert_eq!(tracker.total_time_ms(), router_elapsed_ms);
    }

    #[test]
    fn request_end_counts_output_across_migration_attempts() {
        let tracker = Arc::new(RequestTracker::new());
        let mut pending = PendingRequestEnd {
            data: Some(RequestEndData {
                request_id: "migrated-request".to_string(),
                tracker: tracker.clone(),
                token_ids: Arc::new(Vec::new()),
                kv_cache_block_size: 0,
                replayable: false,
            }),
            output_tokens: 0,
        };

        // The first worker emits two tokens before a retriable failure.
        tracker.record_osl(2);
        pending.record_output(2);
        // Migration reuses the tracker, but the replacement counts only its
        // own three tokens. The client has received five across both workers.
        tracker.record_osl(3);
        pending.record_output(3);
        pending.record_output(0);
        drop(pending);

        assert_eq!(tracker.osl_tokens(), 5);
        assert!(tracker.total_time_ms().is_some());
    }

    #[test]
    fn request_end_preserves_larger_router_output_count() {
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_osl(7);
        drop(PendingRequestEnd {
            data: Some(RequestEndData {
                request_id: "cancelled-request".to_string(),
                tracker: tracker.clone(),
                token_ids: Arc::new(Vec::new()),
                kv_cache_block_size: 0,
                replayable: false,
            }),
            output_tokens: 5,
        });
        assert_eq!(tracker.osl_tokens(), 7);
    }

    fn make_trace_context(
        trace_id: &str,
        span_id: &str,
        trace_flags: &str,
    ) -> DistributedTraceContext {
        serde_json::from_value(serde_json::json!({
            "trace_id": trace_id,
            "span_id": span_id,
            "trace_flags": trace_flags,
            "tracestate": "vendor=dynamo",
            "x_request_id": "xr-1",
        }))
        .expect("DistributedTraceContext deserializes from trace_id + span_id")
    }

    fn with_otel_subscriber<T>(f: impl FnOnce() -> T) -> T {
        let subscriber = tracing_subscriber::registry().with(tracing_opentelemetry::layer());
        tracing::subscriber::with_default(subscriber, f)
    }

    #[test]
    fn dispatch_span_reparents_to_captured_trace_context() {
        with_otel_subscriber(|| {
            let tc =
                make_trace_context("0123456789abcdef0123456789abcdef", "0123456789abcdef", "01");
            let span = dispatch_span(Some(&tc), "req-1");
            let otel_ctx = span.context();
            let span_context = otel_ctx.span().span_context().clone();
            assert_eq!(
                span_context.trace_id(),
                TraceId::from_hex("0123456789abcdef0123456789abcdef").unwrap(),
                "dispatch span must inherit the captured trace_id"
            );
            assert!(span_context.is_sampled());
        });
    }

    #[test]
    fn dispatch_span_preserves_unsampled_trace_flags() {
        with_otel_subscriber(|| {
            let tc =
                make_trace_context("0123456789abcdef0123456789abcdef", "0123456789abcdef", "00");
            let span = dispatch_span(Some(&tc), "req-unsampled");
            let otel_ctx = span.context();
            let span_context = otel_ctx.span().span_context().clone();
            assert_eq!(
                span_context.trace_id(),
                TraceId::from_hex("0123456789abcdef0123456789abcdef").unwrap(),
                "trace identity must propagate even when not sampled"
            );
            assert!(
                !span_context.is_sampled(),
                "a non-sampled parent must not be re-sampled across the Python boundary"
            );
        });
    }

    #[test]
    fn dispatch_span_tolerates_malformed_trace_ids() {
        with_otel_subscriber(|| {
            let tc = make_trace_context("not-hex", "also-not-hex", "01");
            let span = dispatch_span(Some(&tc), "req-2");
            let otel_ctx = span.context();
            assert!(!otel_ctx.span().span_context().is_valid());
        });
    }

    #[test]
    fn dispatch_span_falls_back_to_current_span_without_trace_context() {
        with_otel_subscriber(|| {
            let outer = tracing::info_span!("outer");
            let _enter = outer.enter();
            let span = dispatch_span(None, "req-3");
            assert_eq!(
                span.id(),
                tracing::Span::current().id(),
                "without a trace context the previous behavior must be preserved"
            );
        });
    }

    #[test]
    fn dispatch_span_uses_request_span_target() {
        with_otel_subscriber(|| {
            let tc =
                make_trace_context("0123456789abcdef0123456789abcdef", "0123456789abcdef", "01");
            let span = dispatch_span(Some(&tc), "req-target");
            assert_eq!(
                span.metadata().map(|m| m.target()),
                Some("request_span"),
                "dispatch span must use the always-on request-plane target"
            );
        });
    }

    #[test]
    fn dispatch_span_feeds_distributed_layer_for_header_injection() {
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder().build();
        let tracer = provider.tracer("test");
        // Not SubscriberInitExt::set_default — that installs a global LogTracer.
        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::registry()
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .with(DistributedTraceIdLayer),
        );
        let tc = make_trace_context("0123456789abcdef0123456789abcdef", "0123456789abcdef", "00");
        let span = dispatch_span(Some(&tc), "req-inject");
        let _enter = span.enter();
        let mut headers = std::collections::HashMap::new();

        inject_trace_headers_into_map(&mut headers);

        let traceparent = headers
            .get("traceparent")
            .expect("dispatch span must yield a traceparent for downstream injection");
        assert!(
            traceparent.starts_with("00-0123456789abcdef0123456789abcdef-"),
            "injected traceparent must carry the captured trace_id, got {traceparent}"
        );
        assert!(
            traceparent.ends_with("-00"),
            "unsampled trace_flags must survive injection, got {traceparent}"
        );
        assert_eq!(
            headers.get("tracestate").map(String::as_str),
            Some("vendor=dynamo")
        );
        assert_eq!(
            headers.get("x-request-id").map(String::as_str),
            Some("xr-1")
        );
    }
}
