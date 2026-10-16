// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-native `POST`/`PUT /generate`, in a pluggable [`GenerateFormat`].
//!
//! The format owns every request and response field: it builds the routed
//! request, joins the worker's outputs into client responses, and continues
//! migrated requests. This adapter routes, dispatches, streams, and accounts.

use std::sync::Arc;

use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::post,
};
use dynamo_runtime::error::DynamoError;
use dynamo_runtime::pipeline::{AsyncEngineContext, AsyncEngineContextProvider, Context};
use futures::{Stream, StreamExt, pin_mut};
use parking_lot::Mutex;
use serde_json::Value;
use tracing::Instrument;

use super::disconnect::{
    ConnectionHandle, create_connection_monitor, monitor_for_disconnects_with_error,
};
use super::error::SanitizedError;
use super::metrics::{CancellationLabels, ErrorType, HttpQueueGuard, ResponseMetricCollector};
use super::openai::{
    check_model_serving_ready, check_ready, get_body_limit, get_or_create_request_id,
    preprocessed_context_from_headers,
};
use super::{RouteDoc, service_v2};
use crate::generate_format::{
    Bound, ClientError, ClientRequest, GenerateExchange, Opened, SharedFormat, Worker,
};
use crate::protocols::Annotated;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use crate::protocols::common::timing::RequestTracker;
use crate::protocols::response_merge::PREFILL_OUTPUT_ANNOTATION;

pub(super) const DEFAULT_PATH: &str = "/generate";

type SharedExchange = Arc<Mutex<Box<dyn GenerateExchange>>>;

struct GenerateRoute {
    state: Arc<service_v2::State>,
    format: SharedFormat,
}

fn canonical_generate_models(
    manager: &crate::discovery::ModelManager,
    models: Vec<String>,
) -> Vec<String> {
    models
        .into_iter()
        .map(|model| manager.resolve_canonical_name(&model))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub fn router(
    state: Arc<service_v2::State>,
    path: Option<String>,
    format: SharedFormat,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| DEFAULT_PATH.to_string());
    let docs = vec![
        RouteDoc::new(axum::http::Method::POST, &path),
        RouteDoc::new(axum::http::Method::PUT, &path),
    ];
    let router = Router::new()
        .route(&path, post(handler).put(handler))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(Arc::new(GenerateRoute { state, format }));
    (docs, router)
}

const CANCELLED: StatusCode = match StatusCode::from_u16(499) {
    Ok(status) => status,
    Err(_) => StatusCode::BAD_REQUEST,
};

fn error_response(format: &SharedFormat, status: StatusCode, message: &str) -> Response {
    (status, Json(format.error_body(status, message))).into_response()
}

fn client_error(format: &SharedFormat, error: ClientError) -> Response {
    let body = error
        .body
        .unwrap_or_else(|| format.error_body(error.status, &error.message));
    (error.status, Json(body)).into_response()
}

/// A failure's metric label, status, and client message.
fn classify(error: &(dyn std::error::Error + 'static)) -> (ErrorType, StatusCode, String) {
    use super::metrics::*;
    // Deadline is checked before overload so a chain carrying both markers keeps
    // the deadline outcome, matching the OpenAI surface.
    if request_deadline_exceeded(error) {
        let message = REQUEST_DEADLINE_EXCEEDED_MESSAGE.to_string();
        (ErrorType::Cancelled, StatusCode::TOO_MANY_REQUESTS, message)
    } else if request_was_rejected(error) {
        let message = SanitizedError::Overloaded.to_string();
        (
            ErrorType::Overload,
            StatusCode::SERVICE_UNAVAILABLE,
            message,
        )
    } else if request_was_unavailable(error) {
        let message = SanitizedError::Unavailable.to_string();
        (
            ErrorType::Unavailable,
            StatusCode::SERVICE_UNAVAILABLE,
            message,
        )
    } else if let Some(error) = super::error::find_canonical_error_in_chain(error)
        .filter(|e| e.class() == dynamo_runtime::error::ErrorClass::InvalidRequest)
    {
        let message = error
            .public_message()
            .unwrap_or("Invalid request")
            .to_string();
        (ErrorType::Validation, StatusCode::BAD_REQUEST, message)
    } else if request_was_cancelled(error) {
        (
            ErrorType::Cancelled,
            CANCELLED,
            SanitizedError::Cancelled.to_string(),
        )
    } else {
        let message = SanitizedError::Internal.to_string();
        (
            ErrorType::Internal,
            StatusCode::INTERNAL_SERVER_ERROR,
            message,
        )
    }
}

fn stream_error(
    format: &SharedFormat,
    error: &(dyn std::error::Error + 'static),
) -> (ErrorType, String) {
    let (error_type, status, message) = classify(error);
    (error_type, format.error_body(status, &message).to_string())
}

fn adapt_openai_error(format: &SharedFormat, response: super::openai::ErrorResponse) -> Response {
    let (status, Json(error)) = response;
    error_response(format, status, error.message())
}

async fn run_until_killed<T>(
    context: &dyn AsyncEngineContext,
    operation: impl std::future::Future<Output = T>,
) -> Option<T> {
    tokio::pin!(operation);
    tokio::select! {
        biased;
        result = &mut operation => Some(result),
        _ = context.killed() => None,
    }
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
    worker_info_observed: bool,
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
            worker_info_observed: false,
        }
    }

    /// Copy routed worker labels once, after the tracker has populated them.
    fn observe_worker_info(&mut self) {
        if self.worker_info_observed {
            return;
        }

        let Some(worker) = self.tracker.get_worker_info() else {
            return;
        };
        self.response.set_worker_info(
            worker.prefill_worker_id,
            worker.prefill_dp_rank,
            self.tracker.prefill_worker_type().map(String::from),
            worker.decode_worker_id,
            worker.decode_dp_rank,
            self.tracker.decode_worker_type().map(String::from),
        );
        self.worker_info_observed = true;
    }

    fn observe(&mut self, annotated: &mut Annotated<LLMEngineOutput>) {
        let Some(output) = annotated.data.as_mut() else {
            return;
        };

        if let Some(routing_data) = output.routing_data.take() {
            if let Some(timing) = routing_data.timing {
                self.tracker.set_external_timing(timing);
            }
            if let Some(worker_id) = routing_data.worker_id {
                self.tracker.set_external_worker_info(worker_id);
            }
            if let Some(token_ids) = routing_data.token_ids {
                self.tracker.set_external_query_token_ids(token_ids);
            }
        }

        self.observe_worker_info();

        let cached_tokens = output
            .completion_usage
            .as_ref()
            // A migrated attempt includes already-delivered output tokens in
            // its prompt. Ignore that attempt-local usage for this logical
            // request and let the RequestTracker fallback run on drop.
            .filter(|usage| usage.prompt_tokens as usize == self.input_tokens)
            .and_then(|usage| usage.prompt_tokens_details.as_ref())
            .and_then(|details| details.cached_tokens)
            .map(|tokens| tokens as usize);
        self.response.observe_cached_tokens(cached_tokens);

        // RetryManager appends delivered tokens to the retried request and only
        // yields newly generated deltas, so this remains exact across migration.
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

impl Drop for GenerateMetricCollector {
    fn drop(&mut self) {
        // Matching backend usage is authoritative when present. The response
        // collector latches it during streaming; this logical-request router
        // estimate fills missing or migration-expanded attempt usage.
        self.response
            .observe_cached_tokens(self.tracker.cached_tokens());
    }
}

async fn handler(
    State(route): State<Arc<GenerateRoute>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let GenerateRoute { state, format } = route.as_ref();
    if let Err(response) = check_ready(state) {
        return adapt_openai_error(format, response);
    }
    let Opened {
        request_id,
        stream: streaming,
        model,
        exchange,
    } = match format.open(ClientRequest {
        body: &body,
        headers: &headers,
        default_request_id: &get_or_create_request_id(&headers),
    }) {
        Ok(opened) => opened,
        Err(error) => return client_error(format, error),
    };
    let capability = format.capability();
    let model = match model {
        Some(model) => state.manager().resolve_canonical_name(&model),
        None => {
            let mut models = canonical_generate_models(
                state.manager(),
                state
                    .manager()
                    .list_generate_models_for_capability(capability),
            );
            match models.len() {
                1 => models.remove(0),
                0 => {
                    let message = "no generate-capable model is registered";
                    return error_response(format, StatusCode::NOT_FOUND, message);
                }
                _ => {
                    let message = "multiple models are registered; specify the model";
                    return error_response(format, StatusCode::BAD_REQUEST, message);
                }
            }
        }
    };
    if let Err(response) = check_model_serving_ready(state, &model) {
        return adapt_openai_error(format, response);
    }
    let selection = match state
        .manager()
        .get_generate_engine_for_capability_with_routing(&model, capability)
    {
        Ok(selection) => selection,
        Err(error) => {
            let status = match error {
                crate::discovery::ModelManagerError::ModelUnavailable(_) => {
                    StatusCode::SERVICE_UNAVAILABLE
                }
                _ => StatusCode::NOT_FOUND,
            };
            return error_response(format, status, &error.to_string());
        }
    };
    let mut exchange = exchange;
    let worker = Worker {
        model: &model,
        engine_info: &selection.engine_info,
        kv_cache_block_size: selection.kv_cache_block_size,
        lora_name: selection.lora_name.as_deref(),
        tower_connector_lora_enabled: selection.tower_connector_lora_enabled,
    };
    let Bound {
        mut request,
        migration,
    } = match exchange.bind(&worker) {
        Ok(bound) => bound,
        Err(error) => return client_error(format, error),
    };
    // The router records TTFT and ITL into the request's tracker.
    let tracker = Arc::new(RequestTracker::new());
    tracker.record_isl(request.token_ids.len(), None);
    request.tracker = Some(tracker.clone());
    let exchange: SharedExchange = Arc::new(Mutex::new(exchange));
    match migration {
        Ok(()) => request.continuation = Some(exchange.clone()),
        Err(reason) => tracing::debug!(%request_id, %reason, "generate request cannot migrate"),
    }

    let context: Context<PreprocessedRequest> =
        match preprocessed_context_from_headers(request, request_id.clone(), &headers) {
            Ok(context) => context,
            Err(response) => return adapt_openai_error(format, response),
        };
    let cancellation_labels = CancellationLabels {
        model: state.manager().metric_model_for(&model).to_string(),
        endpoint: super::metrics::Endpoint::Generate.to_string(),
        request_type: if streaming { "streaming" } else { "unary" }.to_string(),
    };
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        context.context(),
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    let dispatch_span =
        tracing::info_span!(target: "request_span", "generate", request_id = %request_id);
    let dispatch = Dispatch {
        tracker,
        request_id,
        model,
        state: state.clone(),
        stream_handle,
        streaming,
        exchange,
        format: format.clone(),
    };
    let response = match tokio::spawn(
        dispatch
            .run(selection.engine, context)
            .instrument(dispatch_span),
    )
    .await
    {
        Ok(response) => response,
        Err(error) => {
            tracing::error!(%error, "generate dispatch task panicked");
            let status = StatusCode::INTERNAL_SERVER_ERROR;
            error_response(format, status, "internal server error")
        }
    };

    connection_handle.disarm();
    response
}

/// Feed one worker item to the exchange.
fn feed(
    exchange: &SharedExchange,
    delta: Annotated<LLMEngineOutput>,
    emit: &mut dyn FnMut(&Value),
) -> Result<(), DynamoError> {
    let prefill = delta.event.as_deref() == Some(PREFILL_OUTPUT_ANNOTATION);
    match delta.into_data()? {
        Some(output) => exchange.lock().chunk(output, prefill, emit),
        None => Ok(()),
    }
}

/// Streaming: each client chunk as one SSE event.
fn events(
    stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
    exchange: SharedExchange,
) -> impl Stream<Item = Result<Vec<Event>, DynamoError>> {
    async_stream::try_stream! {
        pin_mut!(stream);
        // One buffer reused across chunks.
        let mut buffer = Vec::new();
        let mut render = move |value: &Value, events: &mut Vec<Event>| {
            buffer.clear();
            if serde_json::to_writer(&mut buffer, value).is_ok()
                && let Ok(json) = std::str::from_utf8(&buffer)
            {
                events.push(Event::default().data(json));
            }
        };
        while let Some(delta) = stream.next().await {
            let mut out = Vec::new();
            feed(&exchange, delta, &mut |value| render(value, &mut out))?;
            yield out;
        }
        let mut out = Vec::new();
        // Release the lock before `?` can yield.
        let last = exchange.lock().finish();
        if let Some(last) = last? {
            render(&last, &mut out);
        }
        yield out;
    }
}

struct Dispatch {
    tracker: Arc<RequestTracker>,
    request_id: String,
    model: String,
    state: Arc<service_v2::State>,
    stream_handle: ConnectionHandle,
    streaming: bool,
    exchange: SharedExchange,
    format: SharedFormat,
}

impl Dispatch {
    async fn run(
        self,
        engine: crate::types::openai::generate::GenerateStreamingEngine,
        context: Context<PreprocessedRequest>,
    ) -> Response {
        let Self {
            tracker,
            request_id,
            model,
            state,
            stream_handle,
            streaming,
            exchange,
            format,
        } = self;
        let mut inflight_guard = state.metrics_clone().create_inflight_guard(
            state.manager().metric_model_for(&model),
            super::metrics::Endpoint::Generate,
            streaming,
            &request_id,
        );
        let cancelled = |inflight_guard: &mut super::metrics::InflightGuard| {
            inflight_guard.mark_error(ErrorType::Cancelled);
            error_response(&format, CANCELLED, "request was cancelled")
        };
        let request_context = context.context();
        let Some(generate_result) =
            run_until_killed(request_context.as_ref(), engine.generate(context)).await
        else {
            return cancelled(&mut inflight_guard);
        };
        if request_context.is_killed() {
            return cancelled(&mut inflight_guard);
        }
        let stream = match generate_result {
            Ok(stream) => stream,
            Err(error) => {
                let (mut error_type, status, message) = classify(error.as_ref());
                if request_context.is_killed() || status == CANCELLED {
                    return cancelled(&mut inflight_guard);
                }
                if error_type == ErrorType::Overload {
                    // Pre-stream rejections count as unavailable, with a rejection metric.
                    error_type = ErrorType::Unavailable;
                    state
                        .metrics_clone()
                        .inc_rejection(&model, super::metrics::Endpoint::Generate);
                }
                if error_type == ErrorType::Internal {
                    tracing::error!(%request_id, error = %format!("{error:#}"), "generate engine call failed");
                } else {
                    tracing::warn!(%request_id, error = %format!("{error:#}"), "generate request failed before streaming");
                }
                inflight_guard.mark_error(error_type);
                return error_response(&format, status, &message);
            }
        };

        let engine_context = stream.context();
        let input_tokens = tracker.isl_tokens().unwrap_or_default();
        let metric_model = state.manager().metric_model_for(&model).to_string();
        let mut collector = GenerateMetricCollector::new(
            state.metrics_clone(),
            &metric_model,
            tracker,
            input_tokens,
        );
        let stream = stream.map(move |mut item| {
            collector.observe(&mut item);
            item
        });
        if !streaming {
            let fold = async {
                pin_mut!(stream);
                while let Some(delta) = stream.next().await {
                    feed(&exchange, delta, &mut |_| {})?;
                }
                exchange.lock().finish()?.ok_or_else(|| {
                    DynamoError::msg("the generate format returned no unary response")
                })
            };
            return match run_until_killed(request_context.as_ref(), fold).await {
                Some(Ok(response))
                    if !request_context.is_killed() && !engine_context.is_killed() =>
                {
                    inflight_guard.mark_ok();
                    Json(response).into_response()
                }
                Some(Err(error)) => {
                    let (error_type, status, message) = classify(&error);
                    if error_type == ErrorType::Internal {
                        tracing::error!(%request_id, %error, "generate stream failed");
                    }
                    inflight_guard.mark_error(error_type);
                    error_response(&format, status, &message)
                }
                _ => cancelled(&mut inflight_guard),
            };
        }
        let stream = events(stream, exchange).flat_map(|result| {
            futures::stream::iter(match result {
                Ok(events) => events.into_iter().map(Ok).collect(),
                Err(error) => vec![Err(axum::Error::new(error))],
            })
        });
        let stream = monitor_for_disconnects_with_error(
            stream,
            engine_context,
            inflight_guard,
            stream_handle,
            move |error| stream_error(&format, error),
        );
        let mut response = Sse::new(stream);
        if let Some(keep_alive) = state.sse_keep_alive() {
            response = response.keep_alive(KeepAlive::default().interval(keep_alive));
        }
        response.into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use dynamo_runtime::engine::{AsyncEngine, ResponseStream};
    use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
    type Input = SingleIn<PreprocessedRequest>;
    type Output = ManyOut<Annotated<LLMEngineOutput>>;

    struct FailingEngine(dynamo_runtime::error::ErrorType);

    #[async_trait::async_trait]
    impl AsyncEngine<Input, Output, Error> for FailingEngine {
        async fn generate(&self, _request: Input) -> Result<Output, Error> {
            let mut error = DynamoError::builder()
                .error_type(self.0)
                .message("private worker details");
            if self.0 == dynamo_runtime::error::ErrorType::DeadlineExceeded {
                error = error.reason(
                    dynamo_runtime::error::ErrorReason::new("router.queue_deadline_exceeded")
                        .unwrap(),
                );
            }
            Err(error.build().into())
        }
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
    use crate::http::service::metrics::{Endpoint, RequestType, Status};
    use dynamo_runtime::protocols::maybe_error::MaybeError;

    fn sglang() -> SharedFormat {
        Arc::new(crate::protocols::sglang::SglangFormat::native())
    }

    fn dispatch(
        state: Arc<service_v2::State>,
        streaming: bool,
        stream_handle: ConnectionHandle,
    ) -> Dispatch {
        let body = serde_json::json!({"input_ids": [1], "stream": streaming}).to_string();
        let Opened { exchange, .. } = sglang()
            .open(ClientRequest {
                body: body.as_bytes(),
                headers: &HeaderMap::new(),
                default_request_id: "request-id",
            })
            .unwrap();
        Dispatch {
            tracker: Arc::new(RequestTracker::new()),
            request_id: "req".to_string(),
            model: "test-model".to_string(),
            state,
            stream_handle,
            streaming,
            exchange: Arc::new(Mutex::new(exchange)),
            format: sglang(),
        }
    }

    async fn json(response: Response) -> Value {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[test]
    fn model_aliases_are_deduplicated_before_implicit_selection() {
        let manager = crate::discovery::ModelManager::new();
        assert!(manager.register_alias("alias", "primary"));
        let models = ["primary", "alias", "other"].map(String::from).to_vec();
        assert_eq!(
            canonical_generate_models(&manager, models),
            ["other", "primary"]
        );
    }

    #[test]
    fn stream_errors_keep_their_classification_and_shape() {
        use dynamo_runtime::error::{ErrorReason, ErrorType as DynamoErrorType};

        let typed = |error_type, message: &str| {
            DynamoError::builder()
                .error_type(error_type)
                .message(message)
                .build()
        };
        let deadline = DynamoError::builder()
            .error_type(DynamoErrorType::WorkerOverloaded)
            .message("worker overloaded")
            .cause(
                DynamoError::builder()
                    .error_type(DynamoErrorType::DeadlineExceeded)
                    .reason(ErrorReason::new("router.queue_deadline_exceeded").unwrap())
                    .message("internal deadline detail")
                    .build(),
            )
            .build();
        for (error, expected_type, expected_message) in [
            (
                typed(DynamoErrorType::InvalidArgument, "bad params"),
                ErrorType::Validation,
                "Invalid request",
            ),
            (
                typed(DynamoErrorType::Unavailable, "x"),
                ErrorType::Unavailable,
                "Service temporarily unavailable",
            ),
            (
                typed(DynamoErrorType::WorkerUnavailable, "x"),
                ErrorType::Unavailable,
                "Service temporarily unavailable",
            ),
            (
                typed(DynamoErrorType::Cancelled, "x"),
                ErrorType::Cancelled,
                "Request cancelled",
            ),
            (
                DynamoError::msg("backend failure"),
                ErrorType::Internal,
                "Internal server error",
            ),
            // Deadline wins over overload.
            (deadline, ErrorType::Cancelled, "request deadline exceeded"),
        ] {
            let (error_type, body) = stream_error(&sglang(), &axum::Error::new(error));
            let body: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(error_type, expected_type);
            assert_eq!(
                body,
                serde_json::json!({"error": {"message": expected_message}})
            );
        }
    }

    #[tokio::test]
    async fn dispatch_errors_preserve_status_shape_and_metrics() {
        use dynamo_runtime::error::ErrorType as Backend;
        for (kind, status, metric, message) in [
            (
                Backend::WorkerUnavailable,
                StatusCode::SERVICE_UNAVAILABLE,
                ErrorType::Unavailable,
                "Service temporarily unavailable",
            ),
            (
                Backend::DeadlineExceeded,
                StatusCode::TOO_MANY_REQUESTS,
                ErrorType::Cancelled,
                "request deadline exceeded",
            ),
        ] {
            for streaming in [false, true] {
                let state = service_v2::HttpService::builder()
                    .build()
                    .unwrap()
                    .state_clone();
                let (tx, _rx) = tokio::sync::oneshot::channel();
                let response = dispatch(
                    state.clone(),
                    streaming,
                    ConnectionHandle::create_disabled(tx),
                )
                .run(Arc::new(FailingEngine(kind)), dispatch_test_context())
                .await;
                assert_eq!(response.status(), status);
                assert_eq!(json(response).await["error"]["message"], message);
                let request_type = if streaming {
                    RequestType::Stream
                } else {
                    RequestType::Unary
                };
                assert_eq!(
                    state.metrics_clone().get_request_counter(
                        state.manager().metric_model_for("test-model"),
                        &Endpoint::Generate,
                        &request_type,
                        &Status::Error,
                        &metric
                    ),
                    1
                );
            }
        }
    }

    struct ReplayEngine(Vec<Annotated<LLMEngineOutput>>);

    #[async_trait::async_trait]
    impl AsyncEngine<Input, Output, Error> for ReplayEngine {
        async fn generate(&self, request: Input) -> Result<Output, Error> {
            let stream = Box::pin(futures::stream::iter(self.0.clone()));
            Ok(ResponseStream::new(stream, request.context()))
        }
    }

    /// A simulated SGLang worker whose first `failures` attempts drop after
    /// two chunks, as when an engine dies mid-stream.
    struct FlakyWorker {
        calls: std::sync::atomic::AtomicUsize,
        failures: usize,
        truth: crate::generate_format::testkit::sim::Truth,
    }

    #[async_trait::async_trait]
    impl AsyncEngine<Input, Output, Error> for FlakyWorker {
        async fn generate(&self, request: Input) -> Result<Output, Error> {
            use crate::generate_format::{GenerateFormat, Phase, testkit::sim};
            let call = self
                .calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let (request, context) = request.transfer(());
            let body = request.extra_args.as_ref().unwrap()["sglang_tito"].clone();
            let chunks = sim::stream(
                &self.truth,
                body.as_object().unwrap(),
                &request.token_ids,
                sim::Role::Aggregated,
                &Value::Null,
            );
            let format = crate::protocols::sglang::SglangFormat::native();
            let mut items: Vec<_> = chunks
                .into_iter()
                .map(|chunk| {
                    Annotated::from_data(format.inspect(chunk, Phase::Aggregated).unwrap().output)
                })
                .collect();
            if call < self.failures {
                items.truncate(2);
                items.push(Annotated::from_err(
                    DynamoError::builder()
                        .error_type(dynamo_runtime::error::ErrorType::Backend(
                            dynamo_runtime::error::BackendError::StreamIncomplete,
                        ))
                        .message("engine died")
                        .build(),
                ));
            }
            let stream = Box::pin(futures::stream::iter(items));
            Ok(ResponseStream::new(stream, context.context()))
        }
    }

    /// Generate's migration operator in front of a worker.
    struct Migrating(Arc<crate::migration::Migration>, Arc<FlakyWorker>);

    #[async_trait::async_trait]
    impl AsyncEngine<Input, Output, Error> for Migrating {
        async fn generate(&self, request: Input) -> Result<Output, Error> {
            use dynamo_runtime::pipeline::Operator;
            self.0.generate(request, self.1.clone()).await
        }
    }

    #[tokio::test]
    async fn a_request_migrated_mid_stream_returns_what_an_unbroken_one_does() {
        let truth = crate::generate_format::testkit::sim::Truth {
            prompt: vec![1, 2, 3],
            prompt_logprobs: vec![0.0, -0.5, -0.25],
            output: (10..15).map(|id| (id, -1.0)).collect(),
            stop: false,
            row_bytes: 1,
            experts: vec![0; 16],
            cuts: (1..=5).collect(),
            cached_tokens: 2,
        };
        let respond = |stream: bool, failures: usize| {
            let truth = truth.clone();
            async move {
                let body = serde_json::json!({
                    "input_ids": [1, 2, 3], "stream": stream, "return_logprob": true,
                    "logprob_start_len": 0, "sampling_params": {"max_new_tokens": 5},
                })
                .to_string();
                let Opened { mut exchange, .. } = sglang()
                    .open(ClientRequest {
                        body: body.as_bytes(),
                        headers: &HeaderMap::new(),
                        default_request_id: "request-id",
                    })
                    .unwrap();
                let worker = crate::generate_format::testkit::worker(&Value::Null);
                let Bound {
                    mut request,
                    migration,
                } = exchange.bind(&worker).unwrap();
                assert!(migration.is_ok());
                let exchange: SharedExchange = Arc::new(Mutex::new(exchange));
                request.continuation = Some(exchange.clone());
                let worker = Arc::new(FlakyWorker {
                    calls: Default::default(),
                    failures,
                    truth,
                });
                let card = crate::model_card::ModelDeploymentCard::with_name_only("test-model");
                let metrics = Arc::new(super::super::metrics::Metrics::new());
                let migration = crate::migration::Migration::for_generate(&card, 3, metrics);
                let state = service_v2::HttpService::builder()
                    .build()
                    .unwrap()
                    .state_clone();
                let (tx, _rx) = tokio::sync::oneshot::channel();
                let mut dispatch = dispatch(state, stream, ConnectionHandle::create_disabled(tx));
                dispatch.exchange = exchange;
                let response = dispatch
                    .run(
                        Arc::new(Migrating(migration, worker.clone())),
                        Context::new(request),
                    )
                    .await;
                assert_eq!(response.status(), StatusCode::OK);
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let calls = worker.calls.load(std::sync::atomic::Ordering::Relaxed);
                (String::from_utf8(body.to_vec()).unwrap(), calls)
            }
        };
        for stream in [false, true] {
            let (unbroken, calls) = respond(stream, 0).await;
            assert_eq!(calls, 1);
            let (migrated, calls) = respond(stream, 2).await;
            assert_eq!(calls, 3, "two failed attempts, then one that finishes");
            assert_eq!(migrated, unbroken, "stream={stream}");
        }
    }

    #[tokio::test]
    async fn vllm_format_preserves_binding_terminal_status_and_wire_shape() {
        use crate::protocols::common::FinishReason;
        for (finish, status) in [
            (FinishReason::Length, StatusCode::OK),
            (FinishReason::Cancelled, CANCELLED),
            (
                FinishReason::Error("private engine detail".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ] {
            let format = crate::generate_format::named("vllm").unwrap();
            let mut headers = HeaderMap::new();
            headers.insert("x-request-id", "header-id".parse().unwrap());
            headers.insert("x-data-parallel-rank", "3".parse().unwrap());
            let mut opened = format.open(ClientRequest {
                body: br#"{"token_ids":[1],"sampling_params":{},"request_id":"body-id","priority":-7}"#,
                headers: &headers, default_request_id: "default-id",
            }).unwrap();
            assert_eq!(opened.request_id, "header-id");
            let bound = opened
                .exchange
                .bind(&crate::generate_format::testkit::worker(&Value::Null))
                .unwrap();
            assert!(bound.migration.is_err());
            let hints = bound.request.routing.as_ref().unwrap();
            assert_eq!(hints.dp_rank, Some(3));
            assert_eq!(hints.priority, Some(7));
            let state = service_v2::HttpService::builder()
                .build()
                .unwrap()
                .state_clone();
            let (tx, _rx) = tokio::sync::oneshot::channel();
            let mut dispatch = dispatch(state, false, ConnectionHandle::create_disabled(tx));
            dispatch.exchange = Arc::new(Mutex::new(opened.exchange));
            dispatch.format = format;
            let output = LLMEngineOutput {
                token_ids: vec![2, 3],
                finish_reason: Some(finish),
                ..Default::default()
            };
            let response = dispatch
                .run(
                    Arc::new(ReplayEngine(vec![Annotated::from_data(output)])),
                    Context::new(bound.request),
                )
                .await;
            assert_eq!(response.status(), status);
            let body = json(response).await;
            if status == StatusCode::OK {
                assert_eq!(body["request_id"], "header-id");
                assert_eq!(body["choices"][0]["token_ids"], serde_json::json!([2, 3]));
                assert_eq!(body["choices"][0]["finish_reason"], "length");
            } else {
                assert_eq!(body["error"]["code"], status.as_u16());
                assert!(!body.to_string().contains("private engine detail"));
            }
        }
    }

    #[tokio::test]
    async fn unary_dispatch_joins_chunks_and_maps_stream_errors() {
        let chunk = |ids: Value| {
            Annotated::from_data(LLMEngineOutput {
                engine_data: Some(serde_json::json!({"sglang_response": {"output_ids": ids}})),
                ..Default::default()
            })
        };
        let invalid = DynamoError::builder()
            .error_type(dynamo_runtime::error::ErrorType::InvalidArgument)
            .message("bad sampling params")
            .build();
        for (outputs, status, body) in [
            (
                vec![chunk(serde_json::json!([5])), chunk(serde_json::json!([6]))],
                StatusCode::OK,
                serde_json::json!({"output_ids": [5, 6]}),
            ),
            (
                vec![chunk(serde_json::json!([5])), Annotated::from_err(invalid)],
                StatusCode::BAD_REQUEST,
                serde_json::json!({"error": {"message": "Invalid request"}}),
            ),
            (
                vec![Annotated::from_data(LLMEngineOutput::default())],
                StatusCode::INTERNAL_SERVER_ERROR,
                serde_json::json!({"error": {"message": "Internal server error"}}),
            ),
        ] {
            let state = service_v2::HttpService::builder()
                .build()
                .unwrap()
                .state_clone();
            let (tx, _rx) = tokio::sync::oneshot::channel();
            let response = dispatch(state, false, ConnectionHandle::create_disabled(tx))
                .run(Arc::new(ReplayEngine(outputs)), dispatch_test_context())
                .await;
            assert_eq!(response.status(), status);
            assert_eq!(json(response).await, body);
        }
    }
}
