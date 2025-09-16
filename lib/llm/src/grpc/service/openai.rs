// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::{
    engine::AsyncEngineContext,
    pipeline::{AsyncEngineContextProvider, Context},
    protocols::annotated::AnnotationsProvider,
};
use futures::{Stream, StreamExt, stream};
use std::sync::Arc;

use crate::protocols::openai::completions::{
    NvCreateCompletionRequest, NvCreateCompletionResponse,
};
use crate::types::Annotated;

use super::kserve;

// [gluo NOTE] These are common utilities that should be shared between frontends
use crate::http::service::{
    disconnect::{ConnectionHandle, create_connection_monitor},
    metrics::{Endpoint, InflightGuard, process_response_and_observe_metrics},
};

use tonic::Status;

/// Dynamo Annotation for the request ID
pub const ANNOTATION_REQUEST_ID: &str = "request_id";

// [gluo NOTE] strip down version of lib/llm/src/http/service/openai.rs
// dupliating it here as the original file has coupling with HTTP objects.

/// OpenAI Completions Request Handler
///
/// This method will handle the incoming request for the `/v1/completions endpoint`. The endpoint is a "source"
/// for an [`super::OpenAICompletionsStreamingEngine`] and will return a stream of
/// responses which will be forward to the client.
///
/// Note: For all requests, streaming or non-streaming, we always call the engine with streaming enabled. For
/// non-streaming requests, we will fold the stream into a single response as part of this handler.
pub async fn completion_response_stream(
    state: Arc<kserve::State>,
    request: NvCreateCompletionRequest,
) -> Result<impl Stream<Item = Annotated<NvCreateCompletionResponse>>, Status> {
    // create the context for the request
    // [WIP] from request id.
    let request_id = get_or_create_request_id(request.inner.user.as_deref());
    let request = Context::with_id(request, request_id.clone());
    let context = request.context();

    // create the connection handles
    let (mut connection_handle, stream_handle) =
        create_connection_monitor(context.clone(), Some(state.metrics_clone())).await;

    let streaming = request.inner.stream.unwrap_or(false);
    // update the request to always stream
    let request = request.map(|mut req| {
        req.inner.stream = Some(true);
        req
    });

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = &request.inner.model;

    // todo - error handling should be more robust
    let engine = state
        .manager()
        .get_completions_engine(model)
        .map_err(|_| Status::not_found("model not found"))?;

    let http_queue_guard = state.metrics_clone().create_http_queue_guard(model);

    let inflight_guard =
        state
            .metrics_clone()
            .create_inflight_guard(model, Endpoint::Completions, streaming);

    let mut response_collector = state.metrics_clone().create_response_collector(model);

    // prepare to process any annotations
    let annotations = request.annotations();

    // issue the generate call on the engine
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| Status::internal(format!("Failed to generate completions: {}", e)))?;

    // capture the context to cancel the stream if the client disconnects
    let ctx = stream.context();

    // prepare any requested annotations
    let annotations = annotations.map_or(Vec::new(), |annotations| {
        annotations
            .iter()
            .filter_map(|annotation| {
                if annotation == ANNOTATION_REQUEST_ID {
                    Annotated::<NvCreateCompletionResponse>::from_annotation(
                        ANNOTATION_REQUEST_ID,
                        &request_id,
                    )
                    .ok()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    });

    // apply any annotations to the front of the stream
    let stream = stream::iter(annotations).chain(stream);

    // Tap on the stream to collect response metrics and handle http_queue_guard
    let mut http_queue_guard = Some(http_queue_guard);
    let stream = stream.inspect(move |response| {
        // Calls observe_response() on each token - drops http_queue_guard on first token
        process_response_and_observe_metrics(
            response,
            &mut response_collector,
            &mut http_queue_guard,
        );
    });

    let stream = grpc_monitor_for_disconnects(stream, ctx, inflight_guard, stream_handle);

    // if we got here, then we will return a response and the potentially long running task has completed successfully
    // without need to be cancelled.
    connection_handle.disarm();

    Ok(stream)
}

/// This method will consume an AsyncEngineStream and monitor for disconnects or context cancellation.
/// This is gRPC variant of `monitor_for_disconnects` as that implementation has SSE specific handling.
/// Should decouple and reuse `monitor_for_disconnects`
///
/// Uses `tokio::select!` to choose between receiving responses from the source stream or detecting when
/// the context is stopped. If the context is stopped, we break the stream. If the source stream ends
/// naturally, we mark the request as successful and send the final `[DONE]` event.
pub fn grpc_monitor_for_disconnects<T>(
    stream: impl Stream<Item = Annotated<T>>,
    context: Arc<dyn AsyncEngineContext>,
    mut inflight_guard: InflightGuard,
    mut stream_handle: ConnectionHandle,
) -> impl Stream<Item = Annotated<T>> {
    stream_handle.arm();
    async_stream::stream! {
        tokio::pin!(stream);
        loop {
            tokio::select! {
                event = stream.next() => {
                    match event {
                        Some(response) => {
                            yield response;
                        }
                        None => {
                            // Stream ended normally
                            inflight_guard.mark_ok();
                            stream_handle.disarm();
                            break;
                        }
                    }
                }
                _ = context.stopped() => {
                    tracing::trace!("Context stopped; breaking stream");
                    break;
                }
            }
        }
    }
}

/// Get the request ID from a primary source, or lastly create a new one if not present
// TODO: Similar function exists in lib/llm/src/http/service/openai.rs but with different signature and more complex logic (distributed tracing, headers)
fn get_or_create_request_id(primary: Option<&str>) -> String {
    // Try to get the request ID from the primary source
    if let Some(primary) = primary
        && let Ok(uuid) = uuid::Uuid::parse_str(primary)
    {
        return uuid.to_string();
    }

    // Try to parse the request ID as a UUID, or generate a new one if missing/invalid
    let uuid = uuid::Uuid::new_v4();
    uuid.to_string()
}
