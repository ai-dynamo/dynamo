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
use super::metrics::{CancellationLabels, ErrorType};
use super::openai::{
    check_model_serving_ready, check_ready, find_invalid_argument_in_chain, get_body_limit,
    get_or_create_request_id, preprocessed_context_from_headers,
};
use super::{RouteDoc, service_v2};
use crate::generate_format::{ClientRequest, GenerateExchange, Opened, SharedFormat};
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
    (status, Json(format.error_body(message))).into_response()
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
    } else if let Some(error) = find_invalid_argument_in_chain(error) {
        let message = error.message().to_string();
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

/// A stream error already rendered in the request's format.
#[derive(Debug, thiserror::Error)]
#[error("{body}")]
struct RenderedError {
    error_type: ErrorType,
    body: String,
}

/// The stream monitor's formatter: stream errors arrive rendered; the
/// monitor's own get the default shape.
fn stream_error(error: &(dyn std::error::Error + 'static)) -> (ErrorType, String) {
    let mut current = Some(error);
    while let Some(source) = current {
        if let Some(rendered) = source.downcast_ref::<RenderedError>() {
            return (rendered.error_type.clone(), rendered.body.clone());
        }
        current = source.source();
    }
    let (error_type, _, message) = classify(error);
    let body = serde_json::json!({"error": {"message": message}});
    (error_type, body.to_string())
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

async fn handler(
    State(route): State<Arc<GenerateRoute>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let GenerateRoute { state, format } = route.as_ref();
    if let Err(response) = check_ready(state) {
        return adapt_openai_error(format, response);
    }
    let capability = format.capability();
    let models = canonical_generate_models(
        state.manager(),
        state
            .manager()
            .list_generate_models_for_capability(capability),
    );
    let model = match models.len() {
        1 => models.into_iter().next().unwrap(),
        0 => {
            let message = "no generate-capable model is registered";
            return error_response(format, StatusCode::NOT_FOUND, message);
        }
        _ => {
            let message = "multiple models serve this generate format; configure a model-specific generate endpoint";
            return error_response(format, StatusCode::BAD_REQUEST, message);
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

    let Opened {
        mut request,
        request_id,
        stream: streaming,
        migration,
        exchange,
    } = match format.open(ClientRequest {
        body: &body,
        headers: &headers,
        engine_info: &selection.engine_info,
        default_request_id: &get_or_create_request_id(&headers),
    }) {
        Ok(opened) => opened,
        Err(error) => {
            let body = error
                .body
                .unwrap_or_else(|| format.error_body(&error.message));
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    };
    request.model = model.clone();
    // The router records TTFT and ITL into the request's tracker.
    let tracker = Arc::new(RequestTracker::new());
    tracker.record_isl(request.token_ids.len(), None);
    request.tracker = Some(tracker);
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
        let rendering = format.clone();
        let stream = events(stream, exchange).flat_map(move |result| {
            futures::stream::iter(match result {
                Ok(events) => events.into_iter().map(Ok).collect(),
                Err(error) => {
                    let (error_type, _, message) = classify(&error);
                    let body = rendering.error_body(&message).to_string();
                    vec![Err(axum::Error::new(RenderedError { error_type, body }))]
                }
            })
        });
        let stream = monitor_for_disconnects_with_error(
            stream,
            engine_context,
            inflight_guard,
            stream_handle,
            stream_error,
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

    use crate::http::service::generate::tests::{WorkerUnavailableEngine, dispatch_test_context};
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
                engine_info: &Value::Null,
                default_request_id: "request-id",
            })
            .unwrap();
        Dispatch {
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

    #[tokio::test]
    async fn worker_unavailable_dispatch_returns_503() {
        let state = service_v2::HttpService::builder()
            .build()
            .unwrap()
            .state_clone();
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let response = dispatch(state.clone(), true, ConnectionHandle::create_disabled(tx))
            .run(Arc::new(WorkerUnavailableEngine), dispatch_test_context())
            .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json(response).await;
        assert_eq!(body["error"]["message"], "Service temporarily unavailable");
        assert!(!body.to_string().contains("unknown endpoint"));
        assert_eq!(
            state.metrics_clone().get_request_counter(
                state.manager().metric_model_for("test-model"),
                &Endpoint::Generate,
                &RequestType::Stream,
                &Status::Error,
                &ErrorType::Unavailable,
            ),
            1
        );
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
                "bad params",
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
            let (error_type, body) = stream_error(&axum::Error::new(error));
            let body: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(error_type, expected_type);
            assert_eq!(
                body,
                serde_json::json!({"error": {"message": expected_message}})
            );
        }
    }

    struct DeadlineEngine;

    #[async_trait::async_trait]
    impl
        dynamo_runtime::engine::AsyncEngine<
            dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
            dynamo_runtime::pipeline::ManyOut<Annotated<LLMEngineOutput>>,
            dynamo_runtime::pipeline::Error,
        > for DeadlineEngine
    {
        async fn generate(
            &self,
            _request: dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
        ) -> Result<
            dynamo_runtime::pipeline::ManyOut<Annotated<LLMEngineOutput>>,
            dynamo_runtime::pipeline::Error,
        > {
            Err(DynamoError::builder()
                .error_type(dynamo_runtime::error::ErrorType::DeadlineExceeded)
                .reason(
                    dynamo_runtime::error::ErrorReason::new("router.queue_deadline_exceeded")
                        .unwrap(),
                )
                .message("router deadline exceeded before stream start")
                .build()
                .into())
        }
    }

    #[tokio::test]
    async fn pre_stream_deadline_maps_to_http_429_with_cancelled_metric() {
        let state = service_v2::HttpService::builder()
            .build()
            .unwrap()
            .state_clone();
        let context = dispatch_test_context();
        let (mut connection_handle, stream_handle) = create_connection_monitor(
            context.context(),
            None,
            CancellationLabels {
                model: "test-model".to_string(),
                endpoint: Endpoint::Generate.to_string(),
                request_type: "streaming".to_string(),
            },
        )
        .await;
        let response = dispatch(state.clone(), true, stream_handle)
            .run(Arc::new(DeadlineEngine), context)
            .await;
        connection_handle.disarm();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            json(response).await["error"]["message"],
            "request deadline exceeded"
        );
        assert_eq!(
            state.metrics_clone().get_request_counter(
                state.manager().metric_model_for("test-model"),
                &Endpoint::Generate,
                &RequestType::Stream,
                &Status::Error,
                &ErrorType::Cancelled,
            ),
            1
        );
    }

    struct ReplayEngine(Vec<Annotated<LLMEngineOutput>>);

    #[async_trait::async_trait]
    impl
        dynamo_runtime::engine::AsyncEngine<
            dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
            dynamo_runtime::pipeline::ManyOut<Annotated<LLMEngineOutput>>,
            dynamo_runtime::pipeline::Error,
        > for ReplayEngine
    {
        async fn generate(
            &self,
            request: dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
        ) -> Result<
            dynamo_runtime::pipeline::ManyOut<Annotated<LLMEngineOutput>>,
            dynamo_runtime::pipeline::Error,
        > {
            let stream = Box::pin(futures::stream::iter(self.0.clone()));
            Ok(dynamo_runtime::engine::ResponseStream::new(
                stream,
                request.context(),
            ))
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
    impl
        dynamo_runtime::engine::AsyncEngine<
            dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
            dynamo_runtime::pipeline::ManyOut<Annotated<LLMEngineOutput>>,
            dynamo_runtime::pipeline::Error,
        > for FlakyWorker
    {
        async fn generate(
            &self,
            request: dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
        ) -> Result<
            dynamo_runtime::pipeline::ManyOut<Annotated<LLMEngineOutput>>,
            dynamo_runtime::pipeline::Error,
        > {
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
            Ok(dynamo_runtime::engine::ResponseStream::new(
                stream,
                context.context(),
            ))
        }
    }

    /// Generate's migration operator in front of a worker.
    struct Migrating(Arc<crate::migration::Migration>, Arc<FlakyWorker>);

    #[async_trait::async_trait]
    impl
        dynamo_runtime::engine::AsyncEngine<
            dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
            dynamo_runtime::pipeline::ManyOut<Annotated<LLMEngineOutput>>,
            dynamo_runtime::pipeline::Error,
        > for Migrating
    {
        async fn generate(
            &self,
            request: dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
        ) -> Result<
            dynamo_runtime::pipeline::ManyOut<Annotated<LLMEngineOutput>>,
            dynamo_runtime::pipeline::Error,
        > {
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
                let Opened {
                    mut request,
                    migration,
                    exchange,
                    ..
                } = sglang()
                    .open(ClientRequest {
                        body: body.as_bytes(),
                        headers: &HeaderMap::new(),
                        engine_info: &Value::Null,
                        default_request_id: "request-id",
                    })
                    .unwrap();
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
                serde_json::json!({"error": {"message": "bad sampling params"}}),
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
