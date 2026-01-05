// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
use super::kserve::inference;

// [gluo NOTE] These are common utilities that should be shared between frontends
use crate::http::service::{
    disconnect::{ConnectionHandle, create_connection_monitor},
    metrics::{Endpoint, InflightGuard, process_response_and_observe_metrics},
};
use dynamo_async_openai::types::{CompletionFinishReason, CreateCompletionRequest, Prompt};

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

impl TryFrom<inference::ModelInferRequest> for NvCreateCompletionRequest {
    type Error = Status;

    fn try_from(request: inference::ModelInferRequest) -> Result<Self, Self::Error> {
        // Protocol requires if `raw_input_contents` is used to hold input data,
        // it must be used for all inputs.
        if !request.raw_input_contents.is_empty()
            && request.inputs.len() != request.raw_input_contents.len()
        {
            return Err(Status::invalid_argument(
                "`raw_input_contents` must be used for all inputs",
            ));
        }

        // iterate through inputs
        let mut text_input = None;
        let mut stream = false;
        for (idx, input) in request.inputs.iter().enumerate() {
            match input.name.as_str() {
                "text_input" => {
                    if input.datatype != "BYTES" {
                        return Err(Status::invalid_argument(format!(
                            "Expected 'text_input' to be of type BYTES for string input, got {:?}",
                            input.datatype
                        )));
                    }
                    if input.shape != vec![1] && input.shape != vec![1, 1] {
                        return Err(Status::invalid_argument(format!(
                            "Expected 'text_input' to have shape [1], got {:?}",
                            input.shape
                        )));
                    }
                    match &input.contents {
                        Some(content) => {
                            let bytes = content.bytes_contents.first().ok_or_else(|| {
                                Status::invalid_argument(
                                    "'text_input' must contain exactly one element",
                                )
                            })?;
                            text_input = Some(String::from_utf8_lossy(bytes).to_string());
                        }
                        None => {
                            let raw_input =
                                request.raw_input_contents.get(idx).ok_or_else(|| {
                                    Status::invalid_argument("Missing raw input for 'text_input'")
                                })?;
                            if raw_input.len() < 4 {
                                return Err(Status::invalid_argument(
                                    "'text_input' raw input must be length-prefixed (>= 4 bytes)",
                                ));
                            }
                            // We restrict the 'text_input' only contain one element, only need to
                            // parse the first element. Skip first four bytes that is used to store
                            // the length of the input.
                            text_input = Some(String::from_utf8_lossy(&raw_input[4..]).to_string());
                        }
                    }
                }
                "streaming" | "stream" => {
                    if input.datatype != "BOOL" {
                        return Err(Status::invalid_argument(format!(
                            "Expected '{}' to be of type BOOL, got {:?}",
                            input.name, input.datatype
                        )));
                    }
                    if input.shape != vec![1] {
                        return Err(Status::invalid_argument(format!(
                            "Expected 'stream' to have shape [1], got {:?}",
                            input.shape
                        )));
                    }
                    match &input.contents {
                        Some(content) => {
                            stream = *content.bool_contents.first().ok_or_else(|| {
                                Status::invalid_argument(
                                    "'stream' must contain exactly one element",
                                )
                            })?;
                        }
                        None => {
                            let raw_input =
                                request.raw_input_contents.get(idx).ok_or_else(|| {
                                    Status::invalid_argument("Missing raw input for 'stream'")
                                })?;
                            if raw_input.is_empty() {
                                return Err(Status::invalid_argument(
                                    "'stream' raw input must contain at least one byte",
                                ));
                            }
                            stream = raw_input[0] != 0;
                        }
                    }
                }
                _ => {
                    return Err(Status::invalid_argument(format!(
                        "Invalid input name: {}, supported inputs are 'text_input', 'stream'",
                        input.name
                    )));
                }
            }
        }

        // return error if text_input is None
        let text_input = match text_input {
            Some(input) => input,
            None => {
                return Err(Status::invalid_argument(
                    "Missing required input: 'text_input'",
                ));
            }
        };

        Ok(NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: request.model_name,
                prompt: Prompt::String(text_input),
                stream: Some(stream),
                user: if request.id.is_empty() {
                    None
                } else {
                    Some(request.id.clone())
                },
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: None,
            unsupported_fields: Default::default(),
        })
    }
}

impl TryFrom<NvCreateCompletionResponse> for inference::ModelInferResponse {
    type Error = anyhow::Error;

    fn try_from(response: NvCreateCompletionResponse) -> Result<Self, Self::Error> {
        let mut outputs = vec![];
        let mut text_output = vec![];
        let mut finish_reason = vec![];
        for choice in &response.inner.choices {
            text_output.push(choice.text.clone());
            let reason_str = match choice.finish_reason.as_ref() {
                Some(CompletionFinishReason::Stop) => "stop",
                Some(CompletionFinishReason::Length) => "length",
                Some(CompletionFinishReason::ContentFilter) => "content_filter",
                None => "",
            };
            finish_reason.push(reason_str.to_string());
        }
        outputs.push(inference::model_infer_response::InferOutputTensor {
            name: "text_output".to_string(),
            datatype: "BYTES".to_string(),
            shape: vec![text_output.len() as i64],
            contents: Some(inference::InferTensorContents {
                bytes_contents: text_output
                    .into_iter()
                    .map(|text| text.as_bytes().to_vec())
                    .collect(),
                ..Default::default()
            }),
            ..Default::default()
        });
        outputs.push(inference::model_infer_response::InferOutputTensor {
            name: "finish_reason".to_string(),
            datatype: "BYTES".to_string(),
            shape: vec![finish_reason.len() as i64],
            contents: Some(inference::InferTensorContents {
                bytes_contents: finish_reason
                    .into_iter()
                    .map(|text| text.as_bytes().to_vec())
                    .collect(),
                ..Default::default()
            }),
            ..Default::default()
        });

        Ok(inference::ModelInferResponse {
            model_name: response.inner.model,
            model_version: "1".to_string(),
            id: response.inner.id,
            outputs,
            parameters: ::std::collections::HashMap::<String, inference::InferParameter>::new(),
            raw_output_contents: vec![],
        })
    }
}

impl TryFrom<NvCreateCompletionResponse> for inference::ModelStreamInferResponse {
    type Error = anyhow::Error;

    fn try_from(response: NvCreateCompletionResponse) -> Result<Self, Self::Error> {
        match inference::ModelInferResponse::try_from(response) {
            Ok(response) => Ok(inference::ModelStreamInferResponse {
                infer_response: Some(response),
                ..Default::default()
            }),
            Err(e) => Ok(inference::ModelStreamInferResponse {
                infer_response: None,
                error_message: format!("Failed to convert response: {}", e),
            }),
        }
    }
}
