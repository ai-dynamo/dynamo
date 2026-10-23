// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-native `POST`/`PUT /generate`, in a pluggable [`GenerateFormat`].
//!
//! The format owns every request and response field: it builds the routed
//! request and joins the worker's outputs into client responses. This adapter routes, dispatches, streams, and accounts.

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
use super::metrics::{CancellationLabels, ErrorType, HttpQueueGuard, ResponseMetricCollector};
use super::openai::{
    check_model_serving_ready, check_ready, get_body_limit, get_or_create_request_id,
    preprocessed_context_from_headers,
};
use super::{RouteDoc, service_v2};
use crate::generate_format::{
    Bound, ClientError, ClientRequest, ErrorPhase, GenerateExchange, Opened, SharedFormat, Worker,
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
    let mut docs = vec![RouteDoc::new(axum::http::Method::POST, &path)];
    let mut methods = post(handler);
    if format.supports_put() {
        docs.push(RouteDoc::new(axum::http::Method::PUT, &path));
        methods = methods.put(handler);
    }
    let router = Router::new()
        .route(&path, methods)
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
    error
        .response
        .map(|response| *response)
        .unwrap_or_else(|| error_response(format, error.status, &error.message))
}

fn stream_error(
    format: &SharedFormat,
    error: &(dyn std::error::Error + 'static),
) -> (ErrorType, String) {
    let crate::generate_format::WorkerError {
        kind: error_type,
        status,
        message,
    } = format.worker_error(error, ErrorPhase::Stream { cancelled: false });
    (error_type, format.error_body(status, &message).to_string())
}

fn adapt_openai_error(format: &SharedFormat, response: super::openai::ErrorResponse) -> Response {
    let error = ClientError {
        status: response.0,
        message: response.1.0.message().to_string(),
        response: Some(Box::new(response.into_response())),
    };
    client_error(format, format.frontend_error(error))
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
        response: ResponseMetricCollector,
        http_queue: HttpQueueGuard,
        tracker: Arc<RequestTracker>,
        input_tokens: usize,
    ) -> Self {
        Self {
            response,
            http_queue: Some(http_queue),
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
            self.tracker.ingest_routing_data(routing_data);
        }

        self.observe_worker_info();

        let cached_tokens = output
            .completion_usage
            .as_ref()
            // Ignore usage that does not match the client's prompt and let
            // the RequestTracker fallback run on drop.
            .filter(|usage| usage.prompt_tokens as usize == self.input_tokens)
            .and_then(|usage| usage.prompt_tokens_details.as_ref())
            .and_then(|details| details.cached_tokens)
            .map(|tokens| tokens as usize);
        self.response.observe_cached_tokens(cached_tokens);

        // Worker outputs contain newly generated token deltas.
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
        // estimate fills missing or mismatched usage.
        self.response
            .observe_cached_tokens(self.tracker.cached_tokens());
    }
}

async fn handler(
    State(route): State<Arc<GenerateRoute>>,
    headers: HeaderMap,
    body: Result<Bytes, axum::extract::rejection::BytesRejection>,
) -> Response {
    let GenerateRoute { state, format } = route.as_ref();
    let body = match body {
        Ok(body) => body,
        Err(error) => return client_error(format, format.body_error(error, &headers)),
    };
    let Opened {
        request_id,
        stream: streaming,
        model,
        input_tokens,
        validation,
        bind,
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
        Some(model) => Ok(state.manager().resolve_canonical_name(&model)),
        None => {
            let mut models = canonical_generate_models(
                state.manager(),
                state
                    .manager()
                    .list_generate_models_for_capability(capability),
            );
            match models.len() {
                1 => Ok(models.remove(0)),
                0 => Err((
                    StatusCode::NOT_FOUND,
                    "no generate-capable model is registered",
                )),
                _ => Err((StatusCode::BAD_REQUEST, format.ambiguous_model_message())),
            }
        }
    };
    let metric_model = model
        .as_ref()
        .ok()
        .map(|model| state.manager().metric_model_for(model))
        .unwrap_or(crate::discovery::UNKNOWN_METRIC_MODEL);
    let tracker = Arc::new(RequestTracker::new());
    tracker.record_isl(input_tokens, None);
    let dispatch_span = tracing::info_span!(
        target: "request_span", "generate", request_id = %request_id, model = %metric_model,
        input_tokens = tracing::field::Empty, output_tokens = tracing::field::Empty,
        ttft_ms = tracing::field::Empty, avg_itl_ms = tracing::field::Empty,
        prefill_worker_id = tracing::field::Empty, decode_worker_id = tracing::field::Empty,
    );
    let (mut inflight_guard, collector) = dispatch_span.in_scope(|| {
        let inflight = state.metrics_clone().create_inflight_guard(
            metric_model,
            super::metrics::Endpoint::Generate,
            streaming,
            &request_id,
        );
        // Generate prompts are already tokenized; count them before any output.
        inflight.request_metrics().add_input_tokens(input_tokens);
        let collector = GenerateMetricCollector::new(
            inflight.response_collector(),
            state.metrics_clone().create_http_queue_guard(metric_model),
            tracker.clone(),
            input_tokens,
        );
        (inflight, collector)
    });
    let mut reject = |response: Response| {
        inflight_guard.mark_error(match response.status() {
            StatusCode::NOT_FOUND => ErrorType::NotFound,
            StatusCode::NOT_IMPLEMENTED => ErrorType::NotImplemented,
            StatusCode::TOO_MANY_REQUESTS => ErrorType::Overload,
            StatusCode::SERVICE_UNAVAILABLE => ErrorType::Unavailable,
            status if status.as_u16() == 499 => ErrorType::Cancelled,
            status if status.as_u16() == 529 => ErrorType::Overload,
            status if status.is_client_error() => ErrorType::Validation,
            _ => ErrorType::Internal,
        });
        response
    };
    if let Err(response) = check_ready(state) {
        return reject(adapt_openai_error(format, response));
    }
    if let Err(error) = validation {
        return reject(client_error(format, error));
    }
    let model = match model {
        Ok(model) => model,
        Err((status, message)) => return reject(error_response(format, status, message)),
    };
    if let Err(response) = check_model_serving_ready(state, &model) {
        return reject(adapt_openai_error(format, response));
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
            return reject(error_response(format, status, &error.to_string()));
        }
    };
    let worker = Worker {
        model: &model,
        kv_cache_block_size: selection.kv_cache_block_size,
        lora_name: selection.lora_name.as_deref(),
        tower_connector_lora_enabled: selection.tower_connector_lora_enabled,
    };
    let Bound {
        mut request,
        exchange,
    } = match bind(&worker) {
        Ok(bound) => bound,
        Err(error) => return reject(client_error(format, error)),
    };
    request.tracker = Some(tracker.clone());
    let exchange: SharedExchange = Arc::new(Mutex::new(exchange));
    let context: Context<PreprocessedRequest> =
        match preprocessed_context_from_headers(request, request_id.clone(), &headers) {
            Ok(context) => context,
            Err(response) => return reject(adapt_openai_error(format, response)),
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

    let dispatch = Dispatch {
        inflight_guard,
        collector,
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
) -> impl Stream<Item = Result<Event, axum::Error>> {
    async_stream::try_stream! {
        pin_mut!(stream);
        let mut out = Vec::new();
        let mut buffer = Vec::new();
        let mut render = |value: &Value| {
            buffer.clear();
            serde_json::to_writer(&mut buffer, value).map_err(axum::Error::new)?;
            let json = std::str::from_utf8(&buffer).map_err(axum::Error::new)?;
            Ok::<_, axum::Error>(Event::default().data(json))
        };
        while let Some(delta) = stream.next().await {
            feed(&exchange, delta, &mut |value| {
                out.push(render(value));
            }).map_err(axum::Error::new)?;
            for event in out.drain(..) {
                yield event?;
            }
        }
        // Release the lock before `?` or `yield` can suspend the stream.
        let last = exchange.lock().finish();
        if let Some(last) = last.map_err(axum::Error::new)? {
            yield render(&last)?;
        }
    }
}

struct Dispatch {
    inflight_guard: super::metrics::InflightGuard,
    collector: GenerateMetricCollector,
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
            mut inflight_guard,
            mut collector,
            request_id,
            model,
            state,
            stream_handle,
            streaming,
            exchange,
            format,
        } = self;
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
                let crate::generate_format::WorkerError {
                    kind: mut error_type,
                    status,
                    message,
                } = format.worker_error(error.as_ref(), ErrorPhase::Dispatch);
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
                    let crate::generate_format::WorkerError {
                        kind: error_type,
                        status,
                        message,
                    } = format.worker_error(
                        &error,
                        ErrorPhase::Stream {
                            cancelled: request_context.is_killed() || engine_context.is_killed(),
                        },
                    );
                    if error_type == ErrorType::Internal {
                        tracing::error!(%request_id, %error, "generate stream failed");
                    }
                    inflight_guard.mark_error(error_type);
                    error_response(&format, status, &message)
                }
                _ => cancelled(&mut inflight_guard),
            };
        }
        let stream = events(stream, exchange);
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
mod tests;
