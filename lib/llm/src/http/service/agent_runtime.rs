// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::sync::{Arc, LazyLock};

use axum::body::to_bytes;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use dynamo_agent_rt::{
    AgentRuntime, AuthorizationScope, CanonicalJsonFingerprinter, IdempotencyKey,
    InMemoryCheckpointStore, InferenceFuture, InferenceIntent, InferenceInvoker, InferenceOutput,
    InferenceRequest, ModelStepKind, OpenAiResponses, ResponseId, ResponsesOutputInterpreter,
    ResponsesRequestMaterializer, ResponsesStreamEventInterpreter, RunStreamResult, RunTurn,
    RuntimeAuthorization, RuntimeLimits, SystemClock, UuidGenerator,
};
use dynamo_protocols::types::responses::{
    ResponseCompletedEvent, ResponseCreatedEvent, ResponseInProgressEvent,
    ResponseOutputItemDoneEvent, ResponseStreamEvent, Status,
};
use dynamo_runtime::pipeline::AsyncEngineContextProvider;
use futures::{Stream, StreamExt, stream};
use thiserror::Error;

use super::disconnect::create_connection_monitor;
use super::metrics::{CancellationLabels, Endpoint};
use super::{openai, service_v2};
use crate::protocols::common::extensions::NvExt;
use crate::protocols::openai::responses::stream_converter::ResponseEventSerializer;
use crate::protocols::openai::responses::{NvCreateResponse, NvResponse, ResponseParams};
use crate::request_template::{RequestTemplate, resolve_request_model};

static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("DYN_ENABLE_AGENT_RT_POC")
        .ok()
        .and_then(|value| value.parse::<bool>().ok())
        .unwrap_or(false)
});

pub(super) fn enabled() -> bool {
    *ENABLED
}

pub(super) type ResponsesAgentRuntime = AgentRuntime<
    OpenAiResponses,
    InMemoryCheckpointStore<OpenAiResponses>,
    ResponsesRequestMaterializer,
    CanonicalJsonFingerprinter,
    DynamoResponsesInvoker,
    ResponsesOutputInterpreter,
    UuidGenerator,
    SystemClock,
>;

pub(super) fn new_responses_runtime() -> Arc<ResponsesAgentRuntime> {
    Arc::new(AgentRuntime::new(
        InMemoryCheckpointStore::default(),
        ResponsesRequestMaterializer::default(),
        CanonicalJsonFingerprinter,
        DynamoResponsesInvoker,
        ResponsesOutputInterpreter::default(),
        UuidGenerator,
        SystemClock,
    ))
}

/// Ephemeral Dynamo ingress state forwarded across model steps.
///
/// Raw headers remain request-scoped and are never written to checkpoints.
#[derive(Clone)]
pub(super) struct DynamoResponsesContext {
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    request_id: String,
    headers: HeaderMap,
    nvext: Option<NvExt>,
}

impl DynamoResponsesContext {
    pub(super) fn new(
        state: Arc<service_v2::State>,
        template: Option<RequestTemplate>,
        request_id: String,
        headers: HeaderMap,
        nvext: Option<NvExt>,
    ) -> Self {
        Self {
            state,
            template,
            request_id,
            headers,
            nvext,
        }
    }
}

pub(super) async fn handle_responses(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    headers: HeaderMap,
    request: NvCreateResponse,
) -> Result<Response, openai::ErrorResponse> {
    let streaming = request.inner.stream.unwrap_or(false);
    let store = request.inner.store.unwrap_or(false);
    let response_params = ResponseParams::from_create_response(&request.inner);
    let parent_response_id = request
        .inner
        .previous_response_id
        .as_deref()
        .map(ResponseId::from);
    let request_id = openai::get_or_create_request_id(&headers);
    let idempotency_key = header_value(&headers, "idempotency-key")
        .or_else(|| header_value(&headers, "x-idempotency-key"))
        .unwrap_or_else(|| request_id.clone());
    let authorization = RuntimeAuthorization {
        scope: AuthorizationScope {
            tenant_id: header_value(&headers, "x-dynamo-tenant-id")
                .unwrap_or_else(|| "local".to_owned()),
            principal_id: header_value(&headers, "x-dynamo-principal-id")
                .unwrap_or_else(|| "local".to_owned()),
        },
        permitted_connectors: BTreeSet::new(),
        limits: RuntimeLimits::default(),
    };
    let invocation_context =
        DynamoResponsesContext::new(state.clone(), template, request_id, headers, request.nvext);
    let step_kind = if parent_response_id.is_some() {
        ModelStepKind::ClientToolContinuation
    } else {
        ModelStepKind::Initial
    };
    let command = RunTurn {
        request: request.inner,
        parent_response_id,
        authorization,
        idempotency_key: IdempotencyKey::from(idempotency_key),
        invocation_context,
        inference_intent: InferenceIntent { step_kind },
        lease_duration_millis: 120_000,
    };

    if streaming {
        let result = state
            .responses_agent_runtime()
            .clone()
            .run_stream(command, ResponsesStreamEventInterpreter::default())
            .await
            .map_err(|error| {
                openai::ErrorMessage::internal_server_error(&format!(
                    "Agent runtime failed: {error}"
                ))
            })?;
        let stream: AgentResponsesStream = match result {
            RunStreamResult::Live(stream) => {
                Box::pin(stream.map(|event| event.map_err(axum::Error::new)))
            }
            RunStreamResult::Existing(record) => {
                let response = record.response.clone().ok_or_else(|| {
                    openai::ErrorMessage::internal_server_error(
                        "Agent runtime turn exists but has no replayable response",
                    )
                })?;
                Box::pin(stream::iter(
                    committed_response_events(response).into_iter().map(Ok),
                ))
            }
        };
        return Ok(sse_response(
            stream,
            ResponseEventSerializer::new(&response_params),
            state.sse_keep_alive(),
        ));
    }

    let result = state
        .responses_agent_runtime()
        .run_unary(command)
        .await
        .map_err(|error| {
            openai::ErrorMessage::internal_server_error(&format!("Agent runtime failed: {error}"))
        })?;
    let response = result.record().response.clone().ok_or_else(|| {
        openai::ErrorMessage::internal_server_error(
            "Agent runtime turn exists but has no replayable response",
        )
    })?;
    let response = NvResponse {
        inner: response,
        nvext: None,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        store,
    };

    Ok(axum::Json(response).into_response())
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

fn committed_response_events(
    response: dynamo_protocols::types::responses::Response,
) -> Vec<ResponseStreamEvent> {
    let mut initial = response.clone();
    initial.status = Status::InProgress;
    initial.output.clear();
    let mut sequence_number = 0_u64;
    let mut events = Vec::with_capacity(response.output.len() + 3);
    events.push(ResponseStreamEvent::ResponseCreated(ResponseCreatedEvent {
        sequence_number,
        response: initial.clone(),
    }));
    sequence_number += 1;
    events.push(ResponseStreamEvent::ResponseInProgress(
        ResponseInProgressEvent {
            sequence_number,
            response: initial,
        },
    ));
    sequence_number += 1;
    for (output_index, item) in response.output.iter().cloned().enumerate() {
        events.push(ResponseStreamEvent::ResponseOutputItemDone(
            ResponseOutputItemDoneEvent {
                sequence_number,
                output_index: output_index as u32,
                item,
            },
        ));
        sequence_number += 1;
    }
    events.push(ResponseStreamEvent::ResponseCompleted(
        ResponseCompletedEvent {
            sequence_number,
            response,
        },
    ));
    events
}

type AgentResponsesStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<ResponseStreamEvent, axum::Error>> + Send>>;

fn sse_response(
    stream: AgentResponsesStream,
    serializer: ResponseEventSerializer,
    keep_alive: Option<std::time::Duration>,
) -> Response {
    let stream = async_stream::stream! {
        tokio::pin!(stream);
        while let Some(event) = stream.next().await {
            let event = match event {
                Ok(event) => event,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            yield serializer.serialize(&event).map_err(axum::Error::new);
        }
        yield Ok::<Event, axum::Error>(Event::default().data("[DONE]"));
    };
    let mut response = Sse::new(stream);
    if let Some(keep_alive) = keep_alive {
        response = response.keep_alive(KeepAlive::default().interval(keep_alive));
    }
    response.into_response()
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;

    use futures::stream;

    use crate::protocols::openai::responses::ResponseParams;
    use crate::protocols::openai::responses::stream_converter::ResponseEventSerializer;

    use super::{committed_response_events, sse_response};

    #[tokio::test]
    async fn completed_response_is_exposed_as_codex_compatible_sse() {
        let inner = serde_json::from_value(serde_json::json!({
            "created_at": 1,
            "id": "resp_public",
            "model": "model",
            "object": "response",
            "output": [{
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "hello",
                    "annotations": [],
                    "logprobs": null
                }]
            }],
            "status": "completed"
        }))
        .unwrap();
        let stream = Box::pin(stream::iter(
            committed_response_events(inner).into_iter().map(Ok),
        ));
        let response = sse_response(
            stream,
            ResponseEventSerializer::new(&ResponseParams::default()),
            None,
        );

        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        let output_done = body.find("event: response.output_item.done").unwrap();
        let completed = body.find("event: response.completed").unwrap();
        assert!(output_done < completed);
        assert!(body.contains("resp_public"));
        assert!(body.contains("hello"));
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DynamoResponsesInvoker;

#[derive(Debug, Error)]
pub(super) enum DynamoResponsesInvocationError {
    #[error("{0}")]
    Dynamo(String),
    #[error("failed to read Dynamo Responses body: {0}")]
    Body(#[from] axum::Error),
    #[error("failed to decode Dynamo Responses body: {0}")]
    Decode(#[from] serde_json::Error),
}

impl InferenceInvoker<OpenAiResponses> for DynamoResponsesInvoker {
    type Context = DynamoResponsesContext;
    type Error = DynamoResponsesInvocationError;

    fn invoke<'a>(
        &'a self,
        request: &'a InferenceRequest<OpenAiResponses, Self::Context>,
    ) -> InferenceFuture<'a, OpenAiResponses, Self::Error> {
        Box::pin(async move {
            let inner = request.request.clone();
            let streaming = inner.stream.unwrap_or(false);
            let model = resolve_request_model(
                inner.model.as_deref().unwrap_or(""),
                request.context.template.as_ref(),
            )
            .to_owned();
            let pipeline_request = openai::context_from_headers(
                NvCreateResponse {
                    inner,
                    nvext: request.context.nvext.clone(),
                },
                request.context.request_id.clone(),
                &request.context.headers,
            )
            .map_err(|(status, message)| {
                DynamoResponsesInvocationError::Dynamo(format!(
                    "failed to rebuild Dynamo request context ({status}): {}",
                    message.message()
                ))
            })?;
            let engine_context = pipeline_request.context();
            let labels = CancellationLabels {
                model: request
                    .context
                    .state
                    .manager()
                    .metric_model_for(&model)
                    .to_owned(),
                endpoint: Endpoint::Responses.to_string(),
                request_type: if streaming {
                    "agent_runtime_stream"
                } else {
                    "agent_runtime_unary"
                }
                .to_owned(),
            };
            let (mut connection_handle, stream_handle) = create_connection_monitor(
                engine_context,
                Some(request.context.state.metrics_clone()),
                labels,
            )
            .await;

            if streaming {
                let stream = openai::responses_native_stream(
                    request.context.state.clone(),
                    request.context.template.clone(),
                    pipeline_request,
                    stream_handle,
                )
                .await
                .map_err(|(status, message)| {
                    DynamoResponsesInvocationError::Dynamo(format!(
                        "Dynamo Responses invocation failed ({status}): {}",
                        message.message()
                    ))
                })?;
                connection_handle.disarm();
                return Ok(InferenceOutput::Streaming(Box::pin(stream.map(Ok))));
            }

            let response = openai::responses(
                request.context.state.clone(),
                request.context.template.clone(),
                pipeline_request,
                stream_handle,
            )
            .await
            .map_err(|(status, message)| {
                DynamoResponsesInvocationError::Dynamo(format!(
                    "Dynamo Responses invocation failed ({status}): {}",
                    message.message()
                ))
            })?;
            connection_handle.disarm();

            if !response.status().is_success() {
                return Err(DynamoResponsesInvocationError::Dynamo(format!(
                    "Dynamo Responses invocation returned {}",
                    response.status()
                )));
            }
            let body = to_bytes(response.into_body(), openai::get_body_limit()).await?;
            let response: NvResponse = serde_json::from_slice(&body)?;
            Ok(InferenceOutput::Unary(Box::new(response.inner)))
        })
    }
}
