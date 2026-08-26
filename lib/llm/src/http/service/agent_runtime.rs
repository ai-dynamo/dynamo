// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
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
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use futures::{Stream, StreamExt, stream};
use thiserror::Error;

use super::disconnect::create_connection_monitor;
use super::metrics::{CancellationLabels, Endpoint};
use super::{metadata, openai, service_v2};
use crate::protocols::agents::{agent_context_header_values, session_affinity_header_value};
use crate::protocols::common::extensions::{
    AGENT_CONTEXT_CONTEXT_KEY, AgentContext, NvExt, SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId,
};
use crate::protocols::common::input_trigger::classify_response_request;
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

/// Dynamo-owned, explicitly filtered ingress data forwarded across model steps.
///
/// Authorization credentials, caller-supplied identity, routing headers, and
/// lifecycle-finalization hints are intentionally absent. The carrier is
/// ephemeral and is never written to agent runtime checkpoints.
#[derive(Clone)]
struct DynamoInvocationCarrier {
    metadata: BTreeMap<String, String>,
    agent_context: Option<crate::protocols::agents::AgentContextHeaderValues>,
    session_affinity: Option<SessionAffinityId>,
    trace_request_id: Option<String>,
}

impl DynamoInvocationCarrier {
    fn from_headers(headers: &HeaderMap) -> Result<Self, metadata::MetadataHeaderError> {
        let metadata = metadata::extract_metadata_from_http(headers)?;
        let agent_context = agent_context_header_values(headers).map(|mut values| {
            // Agent-runtime turns do not own Dynamo session lifecycle. In
            // particular, an external session-final hint must not evict KV
            // state on every internal model step.
            values.session_final = None;
            values
        });
        let session_affinity = session_affinity_header_value(headers).map(SessionAffinityId::new);
        let trace_request_id = crate::request_trace::is_enabled()
            .then(|| header_value(headers, "x-request-id"))
            .flatten();
        Ok(Self {
            metadata,
            agent_context,
            session_affinity,
            trace_request_id,
        })
    }

    fn context(&self, request: NvCreateResponse, request_id: String) -> Context<NvCreateResponse> {
        let input_trigger = classify_response_request(&request);
        let mut request = Context::with_id_and_metadata(request, request_id, self.metadata.clone());
        if let Some(trace_request_id) = &self.trace_request_id {
            request.insert(
                crate::request_trace::X_REQUEST_ID_CONTEXT_KEY,
                trace_request_id.clone(),
            );
        }
        if let Some(values) = &self.agent_context {
            let mut agent_context = AgentContext::from(values.clone());
            agent_context.input_trigger = Some(input_trigger);
            request.insert(AGENT_CONTEXT_CONTEXT_KEY, agent_context);
        }
        if let Some(session_affinity) = &self.session_affinity {
            request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_affinity.clone());
        }
        request
    }
}

/// Ephemeral Dynamo ingress state forwarded across model steps.
#[derive(Clone)]
pub(super) struct DynamoResponsesContext {
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    request_id: String,
    carrier: DynamoInvocationCarrier,
    nvext: Option<NvExt>,
}

impl DynamoResponsesContext {
    fn new(
        state: Arc<service_v2::State>,
        template: Option<RequestTemplate>,
        request_id: String,
        carrier: DynamoInvocationCarrier,
        nvext: Option<NvExt>,
    ) -> Self {
        Self {
            state,
            template,
            request_id,
            carrier,
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
    let carrier = DynamoInvocationCarrier::from_headers(&headers)
        .map_err(|error| openai::ErrorMessage::request_headers_too_large(&error.to_string()))?;
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
        DynamoResponsesContext::new(state.clone(), template, request_id, carrier, request.nvext);
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
    use axum::http::HeaderMap;

    use futures::stream;

    use crate::protocols::common::extensions::{
        AGENT_CONTEXT_CONTEXT_KEY, AgentContext, InputTrigger, SESSION_AFFINITY_CONTEXT_KEY,
        SessionAffinityId,
    };
    use crate::protocols::openai::responses::ResponseParams;
    use crate::protocols::openai::responses::stream_converter::ResponseEventSerializer;

    use super::{DynamoInvocationCarrier, committed_response_events, sse_response};

    #[test]
    fn invocation_carrier_forwards_only_typed_ingress_data() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer client-secret".parse().unwrap());
        headers.insert("x-dynamo-tenant-id", "spoofed-tenant".parse().unwrap());
        headers.insert("thread-id", "thread-123".parse().unwrap());
        headers.insert("x-codex-parent-thread-id", "thread-parent".parse().unwrap());
        headers.insert("x-dynamo-session-final", "true".parse().unwrap());
        headers.insert("x-dynamo-meta-policy-class", "latency".parse().unwrap());
        headers.insert(
            "x-dynamo-meta-authorization",
            "Bearer nested-secret".parse().unwrap(),
        );

        let carrier = DynamoInvocationCarrier::from_headers(&headers).unwrap();
        assert_eq!(
            carrier.metadata.get("policy-class").map(String::as_str),
            Some("latency")
        );
        assert!(!carrier.metadata.contains_key("authorization"));
        assert_eq!(
            carrier
                .agent_context
                .as_ref()
                .map(|context| context.session_id.as_str()),
            Some("thread-123")
        );
        assert_eq!(
            carrier
                .agent_context
                .as_ref()
                .and_then(|context| context.parent_session_id.as_deref()),
            Some("thread-parent")
        );
        assert_eq!(
            carrier
                .agent_context
                .as_ref()
                .and_then(|context| context.session_final),
            None
        );
        assert_eq!(
            carrier
                .session_affinity
                .as_ref()
                .map(SessionAffinityId::as_str),
            Some("thread-123")
        );

        let request = serde_json::from_value(serde_json::json!({
            "model": "model",
            "input": "hello"
        }))
        .unwrap();
        let context = carrier.context(request, "request-1".to_string());
        assert_eq!(
            context.metadata().get("policy-class").map(String::as_str),
            Some("latency")
        );
        let agent_context = context
            .get::<AgentContext>(AGENT_CONTEXT_CONTEXT_KEY)
            .expect("agent context attached");
        assert_eq!(agent_context.session_final, None);
        assert_eq!(agent_context.kv_hints, None);
        assert_eq!(agent_context.input_trigger, Some(InputTrigger::UserMessage));
        let affinity = context
            .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .expect("session affinity attached");
        assert_eq!(affinity.as_str(), "thread-123");
    }

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
            let pipeline_request = request.context.carrier.context(
                NvCreateResponse {
                    inner,
                    nvext: request.context.nvext.clone(),
                },
                request.context.request_id.clone(),
            );
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
