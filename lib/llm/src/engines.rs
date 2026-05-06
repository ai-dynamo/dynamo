// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;

use async_stream::stream;
use async_trait::async_trait;

use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{Error, ManyIn, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;

#[cfg(test)]
use dynamo_runtime::engine::RequestStream;

use crate::protocols::openai::{
    chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
    completions::{NvCreateCompletionRequest, NvCreateCompletionResponse, prompt_to_string},
};
use crate::types::openai::embeddings::NvCreateEmbeddingRequest;
use crate::types::openai::embeddings::NvCreateEmbeddingResponse;
use dynamo_protocols::types::realtime::{
    EventType, RealtimeAPIError, RealtimeClientEvent, RealtimeServerEvent,
    RealtimeServerEventError, RealtimeServerEventSessionUpdated,
};

//
// The engines are each in their own crate under `lib/engines`
//

#[derive(Debug, Clone)]
pub struct MultiNodeConfig {
    /// How many nodes / hosts we are using
    pub num_nodes: u32,
    /// Unique consecutive integer to identify this node
    pub node_rank: u32,
    /// host:port of head / control node
    pub leader_addr: String,
}

impl Default for MultiNodeConfig {
    fn default() -> Self {
        MultiNodeConfig {
            num_nodes: 1,
            node_rank: 0,
            leader_addr: "".to_string(),
        }
    }
}

//
// Example echo engines
//

/// How long to sleep between echoed tokens.
/// Default is 10ms which gives us 100 tok/s.
/// Can be configured via the DYN_TOKEN_ECHO_DELAY_MS environment variable.
pub static TOKEN_ECHO_DELAY: LazyLock<Duration> = LazyLock::new(|| {
    const DEFAULT_DELAY_MS: u64 = 10;

    let delay_ms = env::var("DYN_TOKEN_ECHO_DELAY_MS")
        .ok()
        .and_then(|val| val.parse::<u64>().ok())
        .unwrap_or(DEFAULT_DELAY_MS);

    Duration::from_millis(delay_ms)
});

/// Engine that accepts un-preprocessed requests and echos the prompt back as the response
/// Useful for testing ingress such as service-http.
struct EchoEngine {}

/// Validate Engine that verifies request data
pub struct ValidateEngine<E> {
    inner: E,
}

impl<E> ValidateEngine<E> {
    pub fn new(inner: E) -> Self {
        Self { inner }
    }
}

/// Engine that dispatches requests to either OpenAICompletions
/// or OpenAIChatCompletions engine
pub struct EngineDispatcher<E> {
    inner: E,
}

impl<E> EngineDispatcher<E> {
    pub fn new(inner: E) -> Self {
        EngineDispatcher { inner }
    }
}

/// Trait on request types that allows us to validate the data
pub trait ValidateRequest {
    fn validate(&self) -> Result<(), anyhow::Error>;
}

/// Trait that allows handling both completion and chat completions requests
#[async_trait]
pub trait StreamingEngine: Send + Sync {
    async fn handle_completion(
        &self,
        req: SingleIn<NvCreateCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error>;

    async fn handle_chat(
        &self,
        req: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error>;
}

/// Trait that allows handling embedding requests
#[async_trait]
pub trait EmbeddingEngine: Send + Sync {
    async fn handle_embedding(
        &self,
        req: SingleIn<NvCreateEmbeddingRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateEmbeddingResponse>>, Error>;
}

pub fn make_echo_engine() -> Arc<dyn StreamingEngine> {
    let engine = EchoEngine {};
    let data = EngineDispatcher::new(engine);
    Arc::new(data)
}

/// Bidirectional echo engine over the OpenAI Realtime event surface.
///
/// For each [`RealtimeClientEvent`] the client sends, the engine emits a
/// single [`RealtimeServerEvent`]:
/// * [`RealtimeClientEvent::SessionUpdate`] → [`RealtimeServerEvent::SessionUpdated`]
///   echoing the requested `session` payload back.
/// * Every other variant → [`RealtimeServerEvent::Error`] with
///   `code = "echo_engine_unsupported"` and a message naming the variant.
///
/// Used by the experimental `/v1/realtime` WebSocket endpoint to demonstrate
/// end-to-end bidirectional plumbing. The mock is intentionally minimal:
/// real session lifecycle, audio buffering, transcription, and response
/// generation belong to a downstream engine, not here.
pub struct EchoBidirectionalEngine {}

#[async_trait]
impl AsyncEngine<ManyIn<RealtimeClientEvent>, ManyOut<Annotated<RealtimeServerEvent>>, Error>
    for EchoBidirectionalEngine
{
    async fn generate(
        &self,
        mut incoming: ManyIn<RealtimeClientEvent>,
    ) -> Result<ManyOut<Annotated<RealtimeServerEvent>>, Error> {
        let ctx = incoming.context();
        let session_id = ctx.id().to_string();
        let ctx_for_stream = ctx.clone();

        let output = stream! {
            let ctx = ctx_for_stream;
            let mut chunk_index: u64 = 0;

            while let Some(client_event) = incoming.next().await {
                if ctx.is_stopped() {
                    break;
                }
                chunk_index += 1;
                let event_id = format!("event_{session_id}_{chunk_index}");

                let server_event = match client_event {
                    RealtimeClientEvent::SessionUpdate(req) => {
                        RealtimeServerEvent::SessionUpdated(RealtimeServerEventSessionUpdated {
                            event_id,
                            session: req.session,
                        })
                    }
                    other => RealtimeServerEvent::Error(RealtimeServerEventError {
                        event_id,
                        error: RealtimeAPIError {
                            r#type: "invalid_request_error".to_string(),
                            code: Some("echo_engine_unsupported".to_string()),
                            message: format!(
                                "echo engine does not support client event {}",
                                other.event_type()
                            ),
                            param: None,
                            event_id: None,
                        },
                    }),
                };

                yield Annotated {
                    id: Some(chunk_index.to_string()),
                    data: Some(server_event),
                    event: None,
                    comment: None,
                    error: None,
                };
            }
        };

        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for EchoEngine
{
    async fn generate(
        &self,
        incoming_request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = incoming_request.transfer(());
        let ctx = context.context();
        let mut deltas = request.response_generator(ctx.id().to_string());
        let Some(req) = request.inner.messages.into_iter().next_back() else {
            anyhow::bail!("Empty chat messages in request");
        };

        let prompt = match req {
            dynamo_protocols::types::ChatCompletionRequestMessage::User(user_msg) => {
                match user_msg.content {
                    dynamo_protocols::types::ChatCompletionRequestUserMessageContent::Text(
                        prompt,
                    ) => prompt,
                    _ => anyhow::bail!("Invalid request content field, expected Content::Text"),
                }
            }
            _ => anyhow::bail!("Invalid request type, expected User message"),
        };

        let output = stream! {
            let mut id = 1;
            for c in prompt.chars() {
                // we are returning characters not tokens, so there will be some postprocessing overhead
                tokio::time::sleep(*TOKEN_ECHO_DELAY).await;
                let response = deltas.create_choice(0, Some(c.to_string()), None, None, None);
                yield Annotated{ id: Some(id.to_string()), data: Some(response), event: None, comment: None, error: None };
                id += 1;
            }

            let response = deltas.create_choice(0, None, Some(dynamo_protocols::types::FinishReason::Stop), None, None);
            yield Annotated { id: Some(id.to_string()), data: Some(response), event: None, comment: None, error: None };
        };

        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateCompletionRequest>,
        ManyOut<Annotated<NvCreateCompletionResponse>>,
        Error,
    > for EchoEngine
{
    async fn generate(
        &self,
        incoming_request: SingleIn<NvCreateCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        let (request, context) = incoming_request.transfer(());
        let ctx = context.context();
        let deltas = request.response_generator(ctx.id().to_string());
        let chars_string = prompt_to_string(&request.inner.prompt);
        let output = stream! {
            let mut id = 1;
            for c in chars_string.chars() {
                tokio::time::sleep(*TOKEN_ECHO_DELAY).await;
                let response = deltas.create_choice(0, Some(c.to_string()), None, None);
                yield Annotated{ id: Some(id.to_string()), data: Some(response), event: None, comment: None, error: None };
                id += 1;
            }
            let response = deltas.create_choice(0, None, Some(dynamo_protocols::types::CompletionFinishReason::Stop), None);
            yield Annotated { id: Some(id.to_string()), data: Some(response), event: None, comment: None, error: None };

        };

        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateEmbeddingRequest>,
        ManyOut<Annotated<NvCreateEmbeddingResponse>>,
        Error,
    > for EchoEngine
{
    async fn generate(
        &self,
        _incoming_request: SingleIn<NvCreateEmbeddingRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateEmbeddingResponse>>, Error> {
        unimplemented!()
    }
}

#[async_trait]
impl<E, Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error> for ValidateEngine<E>
where
    E: AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error> + Send + Sync,
    Req: ValidateRequest + Send + Sync + 'static,
    Resp: Send + Sync + 'static,
{
    async fn generate(
        &self,
        incoming_request: SingleIn<Req>,
    ) -> Result<ManyOut<Annotated<Resp>>, Error> {
        let (request, context) = incoming_request.into_parts();

        // Validate the request first
        if let Err(validation_error) = request.validate() {
            return Err(anyhow::anyhow!("Validation failed: {}", validation_error));
        }

        // Forward to inner engine if validation passes
        let validated_request = SingleIn::rejoin(request, context);
        self.inner.generate(validated_request).await
    }
}

#[async_trait]
impl<E> StreamingEngine for EngineDispatcher<E>
where
    E: AsyncEngine<
            SingleIn<NvCreateCompletionRequest>,
            ManyOut<Annotated<NvCreateCompletionResponse>>,
            Error,
        > + AsyncEngine<
            SingleIn<NvCreateChatCompletionRequest>,
            ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
            Error,
        > + AsyncEngine<
            SingleIn<NvCreateEmbeddingRequest>,
            ManyOut<Annotated<NvCreateEmbeddingResponse>>,
            Error,
        > + Send
        + Sync,
{
    async fn handle_completion(
        &self,
        req: SingleIn<NvCreateCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        self.inner.generate(req).await
    }

    async fn handle_chat(
        &self,
        req: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        self.inner.generate(req).await
    }
}

#[async_trait]
impl<E> EmbeddingEngine for EngineDispatcher<E>
where
    E: AsyncEngine<
            SingleIn<NvCreateEmbeddingRequest>,
            ManyOut<Annotated<NvCreateEmbeddingResponse>>,
            Error,
        > + Send
        + Sync,
{
    async fn handle_embedding(
        &self,
        req: SingleIn<NvCreateEmbeddingRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateEmbeddingResponse>>, Error> {
        self.inner.generate(req).await
    }
}

pub struct EmbeddingEngineAdapter(Arc<dyn EmbeddingEngine>);

impl EmbeddingEngineAdapter {
    pub fn new(engine: Arc<dyn EmbeddingEngine>) -> Self {
        EmbeddingEngineAdapter(engine)
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateEmbeddingRequest>,
        ManyOut<Annotated<NvCreateEmbeddingResponse>>,
        Error,
    > for EmbeddingEngineAdapter
{
    async fn generate(
        &self,
        req: SingleIn<NvCreateEmbeddingRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateEmbeddingResponse>>, Error> {
        self.0.handle_embedding(req).await
    }
}

pub struct StreamingEngineAdapter(Arc<dyn StreamingEngine>);

impl StreamingEngineAdapter {
    pub fn new(engine: Arc<dyn StreamingEngine>) -> Self {
        StreamingEngineAdapter(engine)
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateCompletionRequest>,
        ManyOut<Annotated<NvCreateCompletionResponse>>,
        Error,
    > for StreamingEngineAdapter
{
    async fn generate(
        &self,
        req: SingleIn<NvCreateCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        self.0.handle_completion(req).await
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for StreamingEngineAdapter
{
    async fn generate(
        &self,
        req: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        self.0.handle_chat(req).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::pipeline::Context;
    use futures::stream;

    fn make_input(events: Vec<RealtimeClientEvent>) -> ManyIn<RealtimeClientEvent> {
        RequestStream::new(Box::pin(stream::iter(events)), Context::new(()).context())
    }

    fn parse_client_event(json: serde_json::Value) -> RealtimeClientEvent {
        serde_json::from_value(json).expect("valid realtime client event")
    }

    /// SessionUpdate frames are echoed back as a single SessionUpdated server event
    /// that carries the same Session payload — the round-trip the merge-blocker test
    /// in `http_websocket.rs` relies on.
    #[tokio::test]
    async fn echo_bidirectional_session_update_roundtrip() {
        let session_update = parse_client_event(serde_json::json!({
            "type": "session.update",
            "session": { "type": "realtime", "model": "gpt-realtime" }
        }));

        let engine = EchoBidirectionalEngine {};
        let mut response_stream = engine
            .generate(make_input(vec![session_update]))
            .await
            .expect("generate");

        let chunk = response_stream.next().await.expect("one server event");
        assert!(response_stream.next().await.is_none(), "exactly one event");

        match chunk.data.expect("annotated payload") {
            RealtimeServerEvent::SessionUpdated(updated) => {
                assert!(!updated.event_id.is_empty());
                let json = serde_json::to_value(&updated.session).unwrap();
                assert_eq!(json["type"], "realtime");
                assert_eq!(json["model"], "gpt-realtime");
            }
            other => panic!("expected session.updated, got {other:?}"),
        }
    }

    /// Any non-SessionUpdate client event is rejected with a single Error server event
    /// naming the wire-tag of the offending variant.
    #[tokio::test]
    async fn echo_bidirectional_unknown_event_emits_error() {
        let append = parse_client_event(serde_json::json!({
            "type": "input_audio_buffer.append",
            "audio": ""
        }));

        let engine = EchoBidirectionalEngine {};
        let mut response_stream = engine
            .generate(make_input(vec![append]))
            .await
            .expect("generate");

        let chunk = response_stream.next().await.expect("one server event");
        assert!(response_stream.next().await.is_none(), "exactly one event");

        match chunk.data.expect("annotated payload") {
            RealtimeServerEvent::Error(err) => {
                assert_eq!(err.error.code.as_deref(), Some("echo_engine_unsupported"));
                assert!(err.error.message.contains("input_audio_buffer.append"));
            }
            other => panic!("expected error, got {other:?}"),
        }
    }
}
