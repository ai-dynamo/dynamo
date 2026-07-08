// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP handler for the token-in/token-out `Generate` API
//! (`POST /inference/v1/generate`).
//!
//! This is an experimental engine-native endpoint, **disabled by default**;
//! opt in via the `enable_engine_apis` builder flag or the
//! `DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE` env var. When enabled it registers
//! a frontend-native handler that preserves the complete request in an opaque
//! backend envelope. Streaming (`stream=true`) remains unimplemented.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::post,
};
use dynamo_runtime::pipeline::{AsyncEngineContext, AsyncEngineContextProvider, Context};
use futures::StreamExt;
use serde::Serialize;
use tracing::Instrument;

use super::disconnect::create_connection_monitor;
use super::metrics::{CancellationLabels, ErrorType, HttpQueueGuard, ResponseMetricCollector};
use super::openai::{
    check_model_serving_ready, check_ready, context_from_headers, get_body_limit,
    get_or_create_request_id, smart_json_error_middleware,
};
use super::{RouteDoc, service_v2};
use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::timing::RequestTracker;
use crate::protocols::common::{SamplingOptions, StopConditions};
use crate::protocols::openai::generate::{
    GenerateRequest, GenerateResponse, GenerateResponseOptions, SamplingParams, StreamOptions,
};
use crate::protocols::{Annotated, common::llm_backend::LLMEngineOutput};

const X_REQUEST_ID_HEADER: &str = "x-request-id";
const X_DATA_PARALLEL_RANK_HEADER: &str = "x-data-parallel-rank";

#[derive(Debug)]
struct GenerateRequestContext {
    request_id: String,
    data_parallel_rank: Option<u32>,
}

/// vLLM-style nested error body: `{"error": {"message", "type", "code"}}`.
#[derive(Serialize, Debug)]
struct GenerateError {
    error: GenerateErrorBody,
}

#[derive(Serialize, Debug)]
struct GenerateErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: u16,
}

/// Create an Axum [`Router`] for the token-in/token-out `Generate` endpoint.
/// If no path is provided, the default path is `/inference/v1/generate`.
pub fn generate_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/inference/v1/generate".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(handler_generate))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

/// Build a vLLM-style nested-`error` response.
fn generate_error_response(code: StatusCode, error_type: &str, message: String) -> Response {
    (
        code,
        Json(GenerateError {
            error: GenerateErrorBody {
                message,
                error_type: error_type.to_string(),
                code: code.as_u16(),
            },
        }),
    )
        .into_response()
}

/// Resolve the request metadata that vLLM keeps outside the public JSON body.
fn resolve_generate_request_context(
    headers: &HeaderMap,
    body_request_id: Option<&str>,
) -> GenerateRequestContext {
    let request_id = headers
        .get(X_REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
        .or_else(|| body_request_id.map(ToOwned::to_owned))
        .unwrap_or_else(|| get_or_create_request_id(headers));
    let data_parallel_rank = headers
        .get(X_DATA_PARALLEL_RANK_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse().ok());

    GenerateRequestContext {
        request_id,
        data_parallel_rank,
    }
}

/// Convert vLLM's lower-is-higher priority to Dynamo's higher-is-higher scale.
fn dynamo_routing_priority(vllm_priority: i32) -> i32 {
    vllm_priority.saturating_neg()
}

fn generate_dispatch_span(request_id: &str) -> tracing::Span {
    tracing::info_span!(target: "request_span", "generate", request_id = %request_id)
}

async fn run_until_killed<T>(
    context: &dyn AsyncEngineContext,
    operation: impl std::future::Future<Output = T>,
) -> Option<T> {
    tokio::pin!(operation);
    tokio::select! {
        biased;

        // Preserve an ownership-bearing result if it completes concurrently;
        // callers re-check the context before using it.
        result = &mut operation => Some(result),
        _ = context.killed() => None,
    }
}

fn generate_cancelled_response() -> Response {
    generate_error_response(
        StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
        "request_cancelled",
        "request was cancelled".to_string(),
    )
}

fn generate_internal_error_response() -> Response {
    generate_error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal_error",
        "internal server error".to_string(),
    )
}

/// Borrowed worker envelope for vLLM-specific request fields.
///
/// `token_ids` are intentionally absent: `PreprocessedRequest.token_ids` is
/// the canonical routing and wire representation, and the worker reconstructs
/// the vLLM request from that field.
#[derive(Serialize)]
struct VllmTitoEnvelope<'a> {
    request_id: &'a str,
    sampling_params: &'a SamplingParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<&'a StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_salt: Option<&'a str>,
    priority: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_transfer_params: Option<&'a serde_json::Map<String, serde_json::Value>>,
    #[serde(flatten)]
    passthrough: &'a serde_json::Map<String, serde_json::Value>,
}

impl<'a> VllmTitoEnvelope<'a> {
    fn new(request: &'a GenerateRequest, request_id: &'a str) -> Self {
        let GenerateRequest {
            request_id: _,
            token_ids: _,
            sampling_params,
            model,
            stream,
            stream_options,
            cache_salt,
            priority,
            kv_transfer_params,
            passthrough,
        } = request;
        Self {
            request_id,
            sampling_params,
            model: model.as_deref(),
            stream: *stream,
            stream_options: stream_options.as_ref(),
            cache_salt: cache_salt.as_deref(),
            priority: *priority,
            kv_transfer_params: kv_transfer_params.as_ref(),
            passthrough,
        }
    }
}

/// Project routing controls while retaining all engine-owned fields in
/// `extra_args.vllm_tito`. The backend remains the authority for interpreting
/// every vLLM-specific field.
fn preprocessed_from_generate(
    request: &GenerateRequest,
    model: &str,
    data_parallel_rank: Option<u32>,
    request_id: &str,
) -> anyhow::Result<PreprocessedRequest> {
    let sampling = &request.sampling_params;
    let max_tokens = sampling.max_tokens();
    let routing_priority = dynamo_routing_priority(request.priority);
    let vllm_tito = serde_json::to_value(VllmTitoEnvelope::new(request, request_id))?;
    let tracker = Arc::new(RequestTracker::new());
    tracker.record_isl(request.token_ids.len(), None);

    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(request.token_ids.clone())
        .stop_conditions(StopConditions {
            max_tokens,
            min_tokens: sampling.min_tokens(),
            ignore_eos: Some(sampling.ignore_eos()),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            n: Some(1),
            ..Default::default()
        })
        .output_options(Default::default())
        .routing(Some(crate::protocols::common::preprocessor::RoutingHints {
            dp_rank: data_parallel_rank,
            expected_output_tokens: max_tokens,
            cache_namespace: request.cache_salt.clone(),
            // `priority_jump` is a boost-only scheduler input. Preserve penalties
            // in signed `priority`, matching the standard preprocessor projection.
            priority_jump: Some(routing_priority.max(0) as f64),
            priority: Some(routing_priority),
            ..Default::default()
        }))
        .extra_args(Some(serde_json::json!({
            // Do not copy token_ids into this envelope. The worker must rebuild
            // that field from PreprocessedRequest.token_ids after routing.
            "vllm_tito": vllm_tito,
        })))
        .tracker(Some(tracker))
        .build()
        .map_err(|error| anyhow::anyhow!("failed to build PreprocessedRequest: {error}"))
}

/// Metrics adapter for the raw engine stream used by `/inference/v1/generate`.
///
/// Unlike the OpenAI text endpoints, Generate deliberately bypasses the
/// tokenizer/postprocessor pipeline that emits `LLMMetricAnnotation`. Its
/// token IDs are already rendered, so observe the same response metrics from
/// the raw token deltas while leaving tokenizer and media metrics untouched.
struct GenerateMetricCollector {
    response: ResponseMetricCollector,
    http_queue: Option<HttpQueueGuard>,
    tracker: Arc<RequestTracker>,
    input_tokens: usize,
    output_tokens: usize,
}

impl GenerateMetricCollector {
    fn new(
        metrics: Arc<super::metrics::Metrics>,
        model: &str,
        tracker: Arc<RequestTracker>,
        input_tokens: usize,
    ) -> Self {
        Self {
            response: metrics.clone().create_response_collector(model),
            http_queue: Some(metrics.create_http_queue_guard(model)),
            tracker,
            input_tokens,
            output_tokens: 0,
        }
    }

    fn observe(&mut self, annotated: &Annotated<LLMEngineOutput>) {
        let Some(output) = annotated.data.as_ref() else {
            return;
        };

        if let Some(worker) = self.tracker.get_worker_info() {
            self.response.set_worker_info(
                worker.prefill_worker_id,
                worker.prefill_dp_rank,
                self.tracker.prefill_worker_type().map(String::from),
                worker.decode_worker_id,
                worker.decode_dp_rank,
                self.tracker.decode_worker_type().map(String::from),
            );
        }

        let cached_tokens = output
            .completion_usage
            .as_ref()
            .and_then(|usage| usage.prompt_tokens_details.as_ref())
            .and_then(|details| details.cached_tokens)
            .map(|tokens| tokens as usize);
        self.response.observe_cached_tokens(cached_tokens);

        let chunk_tokens = output.token_ids.len();
        self.output_tokens += chunk_tokens;
        self.response.observe_current_osl(self.output_tokens);
        if self.response.is_first_token()
            && chunk_tokens > 0
            && let Some(guard) = self.http_queue.take()
        {
            drop(guard);
        }
        self.response
            .observe_response(self.input_tokens, chunk_tokens);
    }
}

/// Resolve, route, and dispatch a frontend-native token-in/token-out request.
async fn handler_generate(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<GenerateRequest>,
) -> Response {
    if let Err(response) = check_ready(&state) {
        return response.into_response();
    }

    if let Err(message) = request.validate() {
        return generate_error_response(StatusCode::BAD_REQUEST, "invalid_request_error", message);
    }

    if request.stream {
        return generate_error_response(
            StatusCode::NOT_IMPLEMENTED,
            "not_implemented",
            "streaming (stream=true) is not implemented for /inference/v1/generate yet".to_string(),
        );
    }
    let response_options = request.response_options();

    let model = match &request.model {
        Some(model) => model.clone(),
        None => {
            let models = state.manager().list_generate_models();
            match models.len() {
                1 => models.into_iter().next().unwrap(),
                0 => {
                    return generate_error_response(
                        StatusCode::NOT_FOUND,
                        "not_found",
                        "no generate-capable model is registered".to_string(),
                    );
                }
                _ => {
                    return generate_error_response(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        "multiple models are registered; specify `model` in the request"
                            .to_string(),
                    );
                }
            }
        }
    };

    if let Err(response) = check_model_serving_ready(&state, &model) {
        return response.into_response();
    }

    let engine = match state.manager().get_generate_engine(&model) {
        Ok(engine) => engine,
        Err(error) => {
            let (status, error_type) = match error {
                crate::discovery::ModelManagerError::ModelUnavailable(_) => {
                    (StatusCode::SERVICE_UNAVAILABLE, "service_unavailable")
                }
                _ => (StatusCode::NOT_FOUND, "not_found"),
            };
            return generate_error_response(status, error_type, error.to_string());
        }
    };

    let request_context = resolve_generate_request_context(&headers, request.request_id.as_deref());
    let preprocessed = match preprocessed_from_generate(
        &request,
        &model,
        request_context.data_parallel_rank,
        &request_context.request_id,
    ) {
        Ok(preprocessed) => preprocessed,
        Err(error) => {
            return generate_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                error.to_string(),
            );
        }
    };

    let request_id = request_context.request_id;
    let context: Context<PreprocessedRequest> =
        match context_from_headers(preprocessed, request_id.clone(), &headers) {
            Ok(context) => context,
            Err(response) => return response.into_response(),
        };
    let engine_context = context.context();
    let cancellation_labels = CancellationLabels {
        model: state.manager().metric_model_for(&model).to_string(),
        endpoint: super::metrics::Endpoint::Generate.to_string(),
        request_type: "unary".to_string(),
    };
    let (mut connection_handle, _stream_handle) = create_connection_monitor(
        engine_context,
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    let dispatch_span = generate_dispatch_span(&request_id);
    // Unary work must outlive the Axum handler so dropping the handler can signal
    // the armed connection monitor. The detached dispatch observes that kill at
    // each backend await point and then exits promptly.
    let response = match tokio::spawn(
        generate_dispatch(
            engine,
            context,
            request_id,
            model,
            state.clone(),
            response_options,
        )
        .instrument(dispatch_span),
    )
    .await
    {
        Ok(response) => response,
        Err(error) => {
            tracing::error!(%error, "generate dispatch task panicked");
            generate_internal_error_response()
        }
    };

    connection_handle.disarm();
    response
}

async fn generate_dispatch(
    engine: crate::types::openai::generate::GenerateStreamingEngine,
    context: Context<PreprocessedRequest>,
    request_id: String,
    model: String,
    state: Arc<service_v2::State>,
    response_options: GenerateResponseOptions,
) -> Response {
    let metric_model = state.manager().metric_model_for(&model).to_string();
    let input_tokens = context.content().token_ids.len();
    let tracker = context.content().tracker.clone().unwrap_or_else(|| {
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_isl(input_tokens, None);
        tracker
    });
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &metric_model,
        super::metrics::Endpoint::Generate,
        false,
        &request_id,
    );
    let mut metric_collector =
        GenerateMetricCollector::new(state.metrics_clone(), &metric_model, tracker, input_tokens);
    let request_context = context.context();
    let generate_result =
        match run_until_killed(request_context.as_ref(), engine.generate(context)).await {
            Some(result) => result,
            None => {
                inflight_guard.mark_error(ErrorType::Cancelled);
                return generate_cancelled_response();
            }
        };
    if request_context.is_killed() {
        inflight_guard.mark_error(ErrorType::Cancelled);
        return generate_cancelled_response();
    }
    let stream = match generate_result {
        Ok(stream) => stream,
        Err(error) => {
            let was_cancelled = request_context.is_killed()
                || super::metrics::request_was_cancelled(error.as_ref());
            let was_rejected = super::metrics::request_was_rejected(error.as_ref());
            inflight_guard.mark_error(if was_cancelled {
                ErrorType::Cancelled
            } else if was_rejected {
                ErrorType::Unavailable
            } else {
                ErrorType::Internal
            });
            if was_cancelled {
                return generate_cancelled_response();
            }
            if was_rejected {
                tracing::warn!(%request_id, error = %format!("{error:#}"), "engine rejected generate request");
                state
                    .metrics_clone()
                    .inc_rejection(&metric_model, super::metrics::Endpoint::Generate);
                return generate_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "service_unavailable",
                    "engine rejected the request".to_string(),
                );
            }
            tracing::error!(%request_id, error = %format!("{error:#}"), "engine generate call failed");
            return generate_internal_error_response();
        }
    };

    let engine_context = stream.context();
    let stream = stream.inspect(move |annotated| metric_collector.observe(annotated));
    let response_result = match run_until_killed(
        request_context.as_ref(),
        GenerateResponse::from_annotated_stream_with_options(
            stream,
            request_id.clone(),
            response_options,
        ),
    )
    .await
    {
        Some(result) => result,
        None => {
            inflight_guard.mark_error(ErrorType::Cancelled);
            return generate_cancelled_response();
        }
    };
    match response_result {
        Ok(response) => {
            if request_context.is_killed() || engine_context.is_killed() {
                inflight_guard.mark_error(ErrorType::Cancelled);
                return generate_cancelled_response();
            }
            if !response.is_complete_unary() {
                inflight_guard.mark_error(ErrorType::Internal);
                tracing::error!(%request_id, "generate stream ended without a complete choice");
                return generate_internal_error_response();
            }
            inflight_guard.mark_ok();
            Json(response).into_response()
        }
        Err(error) => {
            if request_context.is_killed()
                || engine_context.is_killed()
                || super::metrics::request_was_cancelled(error.as_ref())
            {
                inflight_guard.mark_error(ErrorType::Cancelled);
                return generate_cancelled_response();
            }
            inflight_guard.mark_error(ErrorType::Internal);
            tracing::error!(%request_id, %error, "failed to fold generate stream");
            generate_internal_error_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::Future,
        pin::Pin,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        task::{Context as TaskContext, Poll},
        time::Duration,
    };

    use super::service_v2::{HttpService, VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV};
    use super::*;
    use crate::http::service::metrics::{Endpoint, RequestType, Status};
    use crate::protocols::{Annotated, common::llm_backend::LLMEngineOutput};
    use dynamo_runtime::{
        engine::{AsyncEngine, ResponseStream},
        pipeline::{Error, ManyOut, SingleIn},
    };
    use futures::Stream;
    use tokio::sync::Notify;
    use tokio_util::sync::CancellationToken;
    use tracing::field::{Field, Visit};
    use tracing::{Subscriber, span};
    use tracing_subscriber::Layer;
    use tracing_subscriber::prelude::*;

    #[derive(Clone, Copy)]
    enum PendingPhase {
        Generate,
        Stream,
    }

    struct PendingOperation {
        started: Arc<Notify>,
        dropped: Arc<AtomicBool>,
        polled: bool,
    }

    impl PendingOperation {
        fn new(started: Arc<Notify>, dropped: Arc<AtomicBool>) -> Self {
            Self {
                started,
                dropped,
                polled: false,
            }
        }

        fn mark_started(&mut self) {
            if !self.polled {
                self.polled = true;
                self.started.notify_one();
            }
        }
    }

    impl Future for PendingOperation {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
            self.get_mut().mark_started();
            Poll::Pending
        }
    }

    impl Stream for PendingOperation {
        type Item = Annotated<LLMEngineOutput>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
            self.get_mut().mark_started();
            Poll::Pending
        }
    }

    impl Drop for PendingOperation {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    struct PendingEngine {
        phase: PendingPhase,
        started: Arc<Notify>,
        dropped: Arc<AtomicBool>,
    }

    struct TerminalEngine(crate::protocols::common::FinishReason);

    struct CancelledEngine;

    struct MetricEngine;

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for CancelledEngine
    {
        async fn generate(
            &self,
            _request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            Err(dynamo_runtime::error::DynamoError::builder()
                .error_type(dynamo_runtime::error::ErrorType::Cancelled)
                .message("backend cancelled before opening a stream")
                .build()
                .into())
        }
    }

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for TerminalEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
                index: Some(0),
                finish_reason: Some(self.0.clone()),
                ..Default::default()
            })]);
            Ok(ResponseStream::new(Box::pin(stream), request.context()))
        }
    }

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for MetricEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let first = futures::stream::once(async {
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![10],
                    index: Some(0),
                    ..Default::default()
                })
            });
            let second = futures::stream::once(async {
                tokio::time::sleep(Duration::from_millis(1)).await;
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![11],
                    index: Some(0),
                    finish_reason: Some(crate::protocols::common::FinishReason::Stop),
                    completion_usage: Some(dynamo_protocols::types::CompletionUsage {
                        prompt_tokens: 3,
                        completion_tokens: 2,
                        total_tokens: 5,
                        prompt_tokens_details: Some(dynamo_protocols::types::PromptTokensDetails {
                            audio_tokens: None,
                            cached_tokens: Some(2),
                        }),
                        completion_tokens_details: None,
                    }),
                    ..Default::default()
                })
            });
            let stream = first.chain(second);
            Ok(ResponseStream::new(Box::pin(stream), request.context()))
        }
    }

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for PendingEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let operation = PendingOperation::new(self.started.clone(), self.dropped.clone());
            match self.phase {
                PendingPhase::Generate => {
                    operation.await;
                    unreachable!("pending generate operation completed")
                }
                PendingPhase::Stream => {
                    Ok(ResponseStream::new(Box::pin(operation), request.context()))
                }
            }
        }
    }

    #[derive(Clone)]
    struct RequestIdCaptureLayer(Arc<Mutex<Option<String>>>);

    impl<S: Subscriber> Layer<S> for RequestIdCaptureLayer {
        fn on_new_span(
            &self,
            attrs: &span::Attributes<'_>,
            _id: &span::Id,
            _context: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut visitor = RequestIdVisitor::default();
            attrs.record(&mut visitor);
            if visitor.request_id.is_some() {
                *self.0.lock().unwrap() = visitor.request_id;
            }
        }
    }

    #[derive(Default)]
    struct RequestIdVisitor {
        request_id: Option<String>,
    }

    impl Visit for RequestIdVisitor {
        fn record_str(&mut self, field: &Field, value: &str) {
            if field.name() == "request_id" {
                self.request_id = Some(value.to_string());
            }
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "request_id" {
                self.request_id = Some(format!("{value:?}"));
            }
        }
    }

    /// Spin up an `HttpService` bound to an ephemeral port and return the port
    /// plus the run handle. Mirrors the reqwest-based router tests in
    /// `service_v2`.
    async fn serve(enable_generate: Option<bool>) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let builder = HttpService::builder().port(port);
        let builder = match enable_generate {
            Some(enabled) => builder.enable_engine_apis(enabled),
            None => builder,
        };
        let service = builder.build().unwrap();
        let cancel_token = CancellationToken::new();
        let handle = tokio::spawn(async move {
            service.run_with_listener(cancel_token, listener).await.ok();
        });
        // Give the server a moment to start listening.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        (port, handle)
    }

    #[tokio::test]
    async fn generate_route_no_model_returns_structured_404() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "not_found");
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_streaming_returns_501() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{},"stream":true}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_IMPLEMENTED.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "not_implemented");
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_rejects_empty_token_ids() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[],"sampling_params":{}}"#)
            .send()
            .await
            .expect("generate request failed");

        assert_eq!(resp.status().as_u16(), StatusCode::BAD_REQUEST.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(
            body["error"]["message"].as_str().is_some_and(
                |message| message.contains("token_ids must contain at least one token")
            )
        );
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_enforces_vllm_rust_request_rules() {
        let (port, handle) = serve(Some(true)).await;
        let client = reqwest::Client::new();
        let invalid = [
            r#"{"token_ids":[1],"sampling_params":{},"stream_options":{"include_usage":true}}"#,
            r#"{"token_ids":[1],"sampling_params":{"max_tokens":0}}"#,
            r#"{"token_ids":[1],"sampling_params":{"prompt_logprobs":-2}}"#,
            r#"{"token_ids":[1],"sampling_params":{"min_tokens":3,"max_tokens":2}}"#,
        ];

        for body in invalid {
            let resp = client
                .post(format!("http://localhost:{port}/inference/v1/generate"))
                .header("content-type", "application/json")
                .body(body)
                .send()
                .await
                .expect("generate request failed");
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body: serde_json::Value = resp.json().await.expect("json body");
            assert_eq!(body["error"]["type"], "invalid_request_error");
        }

        handle.abort();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn generate_route_404_by_default() {
        temp_env::async_with_vars(
            [(VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV, None::<&str>)],
            async {
                let (port, handle) = serve(None).await;
                let resp = reqwest::Client::new()
                    .post(format!("http://localhost:{}/inference/v1/generate", port))
                    .header("content-type", "application/json")
                    .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
                    .send()
                    .await
                    .expect("generate request failed");
                assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
                handle.abort();
            },
        )
        .await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn generate_route_is_registered_when_enabled_by_env() {
        temp_env::async_with_vars(
            [(VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV, Some("1"))],
            async {
                let (port, handle) = serve(None).await;
                let resp = reqwest::Client::new()
                    .post(format!("http://localhost:{}/inference/v1/generate", port))
                    .header("content-type", "application/json")
                    .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
                    .send()
                    .await
                    .expect("generate request failed");
                assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
                let body: serde_json::Value = resp.json().await.expect("json body");
                assert_eq!(body["error"]["type"], "not_found");
                handle.abort();
            },
        )
        .await;
    }

    #[test]
    fn engine_fields_reach_envelope_with_resolved_id_and_cache_namespace() {
        let raw = serde_json::json!({
            "request_id": "req-forward",
            "token_ids": [1, 2],
            "sampling_params": {
                "max_tokens": 8,
                "future_sampling_field": {"nested": true}
            },
            "model": "test-model",
            "stream": true,
            "stream_options": {"include_usage": true},
            "cache_salt": "tenant-a",
            "features": {"future_feature": [1, 2, 3]},
            "priority": 7,
            "kv_transfer_params": {"remote": "worker-a"},
            "future_top_level_field": {"anything": "works"}
        });
        let request: GenerateRequest =
            serde_json::from_value(raw.clone()).expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(&request, "test-model", None, "resolved-request")
                .expect("build request");
        assert_eq!(preprocessed.stop_conditions.max_tokens, Some(8));
        assert_eq!(preprocessed.stop_conditions.min_tokens, None);
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.expected_output_tokens),
            Some(8)
        );
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.priority),
            Some(-7),
            "vLLM lower-is-higher priority must be inverted for Dynamo routing"
        );
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.priority_jump),
            Some(0.0)
        );
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.cache_namespace.as_deref()),
            Some("tenant-a")
        );
        let envelope = preprocessed
            .extra_args
            .as_ref()
            .and_then(|extra| extra.get("vllm_tito"))
            .expect("vllm_tito envelope");

        let mut expected_envelope = raw;
        expected_envelope["request_id"] = serde_json::json!("resolved-request");
        let expected_token_ids = expected_envelope
            .as_object_mut()
            .and_then(|object| object.remove("token_ids"))
            .expect("token_ids in client request");
        assert_eq!(preprocessed.token_ids, vec![1, 2]);
        assert_eq!(
            preprocessed
                .tracker
                .as_ref()
                .and_then(|tracker| tracker.isl_tokens()),
            Some(2)
        );
        assert_eq!(expected_token_ids, serde_json::json!([1, 2]));
        assert_eq!(envelope, &expected_envelope);
        assert!(envelope.get("token_ids").is_none());
    }

    #[test]
    fn omitted_max_tokens_stays_omitted_in_control_shadow() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {},
            "model": "test-model"
        }))
        .expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(&request, "test-model", None, "resolved-request")
                .expect("build request");
        assert_eq!(preprocessed.stop_conditions.max_tokens, None);
        assert_eq!(preprocessed.stop_conditions.min_tokens, None);
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.expected_output_tokens),
            None
        );
    }

    #[test]
    fn explicit_zero_min_tokens_stays_explicit_in_control_shadow() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {"min_tokens": 0},
            "model": "test-model"
        }))
        .expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(&request, "test-model", None, "resolved-request")
                .expect("build request");
        assert_eq!(preprocessed.stop_conditions.min_tokens, Some(0));
    }

    #[test]
    fn generate_request_context_matches_vllm_header_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert(X_REQUEST_ID_HEADER, "header-request".parse().unwrap());
        headers.insert(X_DATA_PARALLEL_RANK_HEADER, "3".parse().unwrap());

        let context = resolve_generate_request_context(&headers, Some("body-request"));

        assert_eq!(context.request_id, "header-request");
        assert_eq!(context.data_parallel_rank, Some(3));
    }

    #[test]
    fn generate_request_context_falls_back_and_ignores_invalid_dp_rank() {
        let mut headers = HeaderMap::new();
        headers.insert(X_DATA_PARALLEL_RANK_HEADER, "invalid".parse().unwrap());

        let context = resolve_generate_request_context(&headers, Some("body-request"));

        assert_eq!(context.request_id, "body-request");
        assert_eq!(context.data_parallel_rank, None);
    }

    #[test]
    fn generate_dispatch_span_uses_resolved_request_id() {
        let captured_request_id = Arc::new(Mutex::new(None));
        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::registry().with(RequestIdCaptureLayer(captured_request_id.clone())),
        );

        let _dispatch_span = generate_dispatch_span("header-request");

        assert_eq!(
            captured_request_id.lock().unwrap().as_deref(),
            Some("header-request")
        );
    }

    fn dispatch_test_context() -> Context<PreprocessedRequest> {
        Context::new(
            PreprocessedRequest::builder()
                .model("test-model".to_string())
                .token_ids(vec![1])
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .build()
                .expect("build dispatch test request"),
        )
    }

    fn metric_value<'a>(
        families: &'a [prometheus::proto::MetricFamily],
        name: &str,
        labels: &[(&str, &str)],
    ) -> &'a prometheus::proto::Metric {
        let family = families
            .iter()
            .find(|family| family.name() == name)
            .unwrap_or_else(|| panic!("missing metric family {name}"));
        family
            .get_metric()
            .iter()
            .find(|metric| {
                labels.iter().all(|(expected_name, expected_value)| {
                    metric.get_label().iter().any(|label| {
                        label.name() == *expected_name && label.value() == *expected_value
                    })
                })
            })
            .unwrap_or_else(|| panic!("missing {name} series with labels {labels:?}"))
    }

    fn histogram_sample_count_or_zero(
        families: &[prometheus::proto::MetricFamily],
        name: &str,
        labels: &[(&str, &str)],
    ) -> u64 {
        families
            .iter()
            .find(|family| family.name() == name)
            .and_then(|family| {
                family.get_metric().iter().find(|metric| {
                    labels.iter().all(|(expected_name, expected_value)| {
                        metric.get_label().iter().any(|label| {
                            label.name() == *expected_name && label.value() == *expected_value
                        })
                    })
                })
            })
            .map(|metric| metric.get_histogram().get_sample_count())
            .unwrap_or(0)
    }

    fn assert_cancelled_dispatch_metrics(state: &service_v2::State) {
        let metric_model = state.manager().metric_model_for("test-model");
        let metrics = state.metrics_clone();
        assert_eq!(metrics.get_inflight_count(metric_model), 0);
        assert_eq!(
            metrics.get_request_counter(
                metric_model,
                &Endpoint::Generate,
                &RequestType::Unary,
                &Status::Error,
                &ErrorType::Cancelled,
            ),
            1
        );

        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();
        let families = registry.gather();
        let model_labels = [("model", metric_model)];
        assert_eq!(
            metric_value(&families, "dynamo_frontend_queued_requests", &model_labels,)
                .get_gauge()
                .value(),
            0.0
        );
        assert_eq!(
            histogram_sample_count_or_zero(
                &families,
                "dynamo_frontend_time_to_first_token_seconds",
                &model_labels,
            ),
            0
        );
        assert_eq!(
            histogram_sample_count_or_zero(
                &families,
                "dynamo_frontend_output_sequence_tokens",
                &model_labels,
            ),
            0
        );
    }

    async fn await_cancelled_dispatch(
        task: tokio::task::JoinHandle<Response>,
        dropped: &AtomicBool,
        state: &service_v2::State,
    ) {
        let response = tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .expect("dispatch did not stop promptly after request kill")
            .expect("dispatch task panicked");
        assert_eq!(response.status().as_u16(), 499);
        assert!(dropped.load(Ordering::SeqCst));
        assert_cancelled_dispatch_metrics(state);
    }

    async fn assert_request_kill_interrupts_pending(phase: PendingPhase) {
        let started = Arc::new(Notify::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(PendingEngine {
                phase,
                started: started.clone(),
                dropped: dropped.clone(),
            });
        let context = dispatch_test_context();
        let request_context = context.context();
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let task = tokio::spawn(generate_dispatch(
            engine,
            context,
            "req-pending-dispatch".to_string(),
            "test-model".to_string(),
            state.clone(),
            GenerateResponseOptions::default(),
        ));

        started.notified().await;
        assert_eq!(
            state
                .metrics_clone()
                .get_inflight_count(state.manager().metric_model_for("test-model")),
            1
        );
        request_context.kill();

        await_cancelled_dispatch(task, dropped.as_ref(), state.as_ref()).await;
    }

    async fn dispatch_terminal_finish_reason(
        finish_reason: crate::protocols::common::FinishReason,
    ) -> (Response, Arc<service_v2::State>) {
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(TerminalEngine(finish_reason));
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let response = generate_dispatch(
            engine,
            dispatch_test_context(),
            "req-terminal-dispatch".to_string(),
            "test-model".to_string(),
            state.clone(),
            GenerateResponseOptions::default(),
        )
        .await;
        (response, state)
    }

    #[tokio::test]
    async fn request_kill_interrupts_pending_engine_generate() {
        assert_request_kill_interrupts_pending(PendingPhase::Generate).await;
    }

    #[tokio::test]
    async fn request_kill_interrupts_pending_response_stream() {
        assert_request_kill_interrupts_pending(PendingPhase::Stream).await;
    }

    #[tokio::test]
    async fn backend_error_finish_returns_sanitized_500() {
        let secret = "sensitive backend failure";
        let (response, _state) = dispatch_terminal_finish_reason(
            crate::protocols::common::FinishReason::Error(secret.to_string()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read error response");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("parse error response");
        assert_eq!(body["error"]["message"], "internal server error");
        assert!(!body.to_string().contains(secret));
    }

    #[tokio::test]
    async fn backend_cancelled_finish_returns_499() {
        let (response, state) =
            dispatch_terminal_finish_reason(crate::protocols::common::FinishReason::Cancelled)
                .await;

        assert_eq!(response.status().as_u16(), 499);
        assert_cancelled_dispatch_metrics(state.as_ref());
    }

    #[tokio::test]
    async fn immediate_engine_cancellation_returns_499() {
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(CancelledEngine);
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();

        let response = generate_dispatch(
            engine,
            dispatch_test_context(),
            "req-immediate-cancel".to_string(),
            "test-model".to_string(),
            state.clone(),
            GenerateResponseOptions::default(),
        )
        .await;

        assert_eq!(response.status().as_u16(), 499);
        assert_cancelled_dispatch_metrics(state.as_ref());
    }

    #[tokio::test]
    async fn zero_token_success_drains_queue_without_output_metrics() {
        let (response, state) =
            dispatch_terminal_finish_reason(crate::protocols::common::FinishReason::Stop).await;

        assert_eq!(response.status(), StatusCode::OK);
        let metric_model = state.manager().metric_model_for("test-model");
        assert_eq!(
            state.metrics_clone().get_request_counter(
                metric_model,
                &Endpoint::Generate,
                &RequestType::Unary,
                &Status::Success,
                &ErrorType::None,
            ),
            1
        );

        let registry = prometheus::Registry::new();
        state.metrics_clone().register(&registry).unwrap();
        let families = registry.gather();
        let model_labels = [("model", metric_model)];
        assert_eq!(
            metric_value(&families, "dynamo_frontend_queued_requests", &model_labels,)
                .get_gauge()
                .value(),
            0.0
        );
        assert_eq!(
            histogram_sample_count_or_zero(
                &families,
                "dynamo_frontend_time_to_first_token_seconds",
                &model_labels,
            ),
            0
        );
        assert_eq!(
            histogram_sample_count_or_zero(
                &families,
                "dynamo_frontend_output_sequence_tokens",
                &model_labels,
            ),
            0
        );
    }

    #[tokio::test]
    async fn successful_generate_populates_frontend_metrics() {
        const MODEL: &str = "generate-metric-test-model";
        const WORKER_ID: &str = "987654321";
        const DP_RANK: &str = "3";

        let tracker = Arc::new(RequestTracker::new());
        tracker.record_isl(3, None);
        tracker.record_worker(
            WORKER_ID.parse().unwrap(),
            Some(DP_RANK.parse().unwrap()),
            crate::discovery::WORKER_TYPE_DECODE,
        );
        let context = Context::new(
            PreprocessedRequest::builder()
                .model(MODEL.to_string())
                .token_ids(vec![1, 2, 3])
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .tracker(Some(tracker))
                .build()
                .expect("build metric test request"),
        );
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(MetricEngine);
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let metric_model = state.manager().metric_model_for(MODEL).to_string();
        let registry = prometheus::Registry::new();
        state.metrics_clone().register(&registry).unwrap();

        let response = generate_dispatch(
            engine,
            context,
            "req-generate-metrics".to_string(),
            MODEL.to_string(),
            state.clone(),
            GenerateResponseOptions::default(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(state.metrics_clone().get_inflight_count(&metric_model), 0);
        assert_eq!(
            state.metrics_clone().get_request_counter(
                &metric_model,
                &Endpoint::Generate,
                &RequestType::Unary,
                &Status::Success,
                &ErrorType::None,
            ),
            1
        );

        let families = registry.gather();
        let model_labels = [("model", metric_model.as_str())];
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_requests_started_total",
                &[("model", metric_model.as_str()), ("endpoint", "generate")],
            )
            .get_counter()
            .value(),
            1.0
        );
        assert_eq!(
            metric_value(&families, "dynamo_frontend_active_requests", &model_labels)
                .get_gauge()
                .value(),
            0.0
        );
        assert_eq!(
            metric_value(&families, "dynamo_frontend_queued_requests", &model_labels)
                .get_gauge()
                .value(),
            0.0
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_request_duration_seconds",
                &model_labels,
            )
            .get_histogram()
            .get_sample_count(),
            1
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_input_sequence_tokens",
                &model_labels,
            )
            .get_histogram()
            .get_sample_sum(),
            3.0
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_output_sequence_tokens",
                &model_labels,
            )
            .get_histogram()
            .get_sample_sum(),
            2.0
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_output_tokens_total",
                &model_labels,
            )
            .get_counter()
            .value(),
            2.0
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_time_to_first_token_seconds",
                &model_labels,
            )
            .get_histogram()
            .get_sample_count(),
            1
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_inter_token_latency_seconds",
                &model_labels,
            )
            .get_histogram()
            .get_sample_count(),
            1
        );
        assert_eq!(
            metric_value(&families, "dynamo_frontend_cached_tokens", &model_labels,)
                .get_histogram()
                .get_sample_sum(),
            2.0
        );

        let worker_labels = [WORKER_ID, DP_RANK, crate::discovery::WORKER_TYPE_DECODE];
        assert_eq!(
            crate::http::service::metrics::WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE
                .with_label_values(&worker_labels)
                .get(),
            3
        );
        assert!(
            crate::http::service::metrics::WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE
                .with_label_values(&worker_labels)
                .get()
                > 0.0
        );
        assert!(
            crate::http::service::metrics::WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE
                .with_label_values(&worker_labels)
                .get()
                > 0.0
        );
    }

    #[test]
    fn generate_control_shadow_carries_dp_rank_and_inverted_priority() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {},
            "priority": -7
        }))
        .expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(&request, "test-model", Some(3), "resolved-request")
                .expect("build request");
        let routing = preprocessed.routing.as_ref().expect("routing hints");

        assert_eq!(routing.dp_rank, Some(3));
        assert_eq!(routing.priority, Some(7));
        assert_eq!(routing.priority_jump, Some(7.0));
    }

    #[test]
    fn priority_inversion_saturates_at_i32_min() {
        assert_eq!(dynamo_routing_priority(i32::MIN), i32::MAX);
    }
}
