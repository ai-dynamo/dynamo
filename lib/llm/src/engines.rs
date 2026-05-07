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
    EventType, MaxOutputTokens, RealtimeAPIError, RealtimeClientEvent, RealtimeResponse,
    RealtimeResponseStatus, RealtimeServerEvent, RealtimeServerEventError,
    RealtimeServerEventResponseAudioDelta, RealtimeServerEventResponseAudioDone,
    RealtimeServerEventResponseCreated, RealtimeServerEventResponseDone,
    RealtimeServerEventSessionUpdated,
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

/// How many base64 characters per [`response.output_audio.delta`] frame the
/// echo engine emits. Picked small enough to demonstrate streaming (multiple
/// deltas per append) on typical audio buffers, large enough to keep the
/// frame count reasonable. Has no semantic meaning beyond pacing.
const ECHO_AUDIO_DELTA_CHUNK_LEN: usize = 64;

/// Bidirectional echo engine over the OpenAI Realtime event surface.
///
/// For each [`RealtimeClientEvent`] the client sends, the engine emits a
/// short stream of [`RealtimeServerEvent`]s that demonstrates the round-trip:
/// * [`RealtimeClientEvent::SessionUpdate`] → one
///   [`RealtimeServerEvent::SessionUpdated`] echoing the requested `session`.
/// * [`RealtimeClientEvent::InputAudioBufferAppend`] → a spec-shaped response
///   envelope:
///   1. [`RealtimeServerEvent::ResponseCreated`] (`status: in_progress`)
///   2. one or more [`RealtimeServerEvent::ResponseOutputAudioDelta`] frames
///      echoing the base64 audio back in `ECHO_AUDIO_DELTA_CHUNK_LEN`-char
///      chunks
///   3. [`RealtimeServerEvent::ResponseOutputAudioDone`]
///   4. [`RealtimeServerEvent::ResponseDone`] (`status: completed`)
/// * Every other variant → [`RealtimeServerEvent::Error`] with
///   `code = "echo_engine_unsupported"` and a message naming the variant.
///
/// Used by the experimental `/v1/realtime` WebSocket endpoint to demonstrate
/// end-to-end bidirectional plumbing. The mock is intentionally minimal:
/// real session lifecycle, audio buffering, transcription, VAD turn
/// detection, conversation-item bookkeeping, and gating audio output on a
/// `response.create` trigger all belong to a downstream engine, not here.
/// Notably, this echo emits the response envelope *immediately on append*
/// rather than waiting for `response.create` — a real engine would gate it.
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
            let mut turn: u64 = 0;
            let mut frame: u64 = 0;

            while let Some(client_event) = incoming.next().await {
                if ctx.is_stopped() {
                    break;
                }
                turn += 1;
                let response_id = format!("resp_{session_id}_{turn}");
                let item_id = format!("item_{session_id}_{turn}");

                match client_event {
                    RealtimeClientEvent::SessionUpdate(req) => {
                        frame += 1;
                        yield annotated_event(
                            frame,
                            RealtimeServerEvent::SessionUpdated(
                                RealtimeServerEventSessionUpdated {
                                    event_id: format!("event_{session_id}_{frame}"),
                                    session: req.session,
                                },
                            ),
                        );
                    }
                    RealtimeClientEvent::InputAudioBufferAppend(req) => {
                        // Spec shape: response.created → output_audio.delta×N
                        // → output_audio.done → response.done. Chunking is on
                        // base64-character boundaries (not byte boundaries),
                        // which is fine for transport echo — receivers
                        // concatenate the deltas before decoding.
                        frame += 1;
                        yield annotated_event(
                            frame,
                            RealtimeServerEvent::ResponseCreated(
                                RealtimeServerEventResponseCreated {
                                    event_id: format!("event_{session_id}_{frame}"),
                                    response: echo_response(
                                        &response_id,
                                        RealtimeResponseStatus::InProgress,
                                    ),
                                },
                            ),
                        );

                        let chars: Vec<char> = req.audio.chars().collect();
                        for chunk in chars.chunks(ECHO_AUDIO_DELTA_CHUNK_LEN) {
                            if ctx.is_stopped() {
                                break;
                            }
                            frame += 1;
                            yield annotated_event(
                                frame,
                                RealtimeServerEvent::ResponseOutputAudioDelta(
                                    RealtimeServerEventResponseAudioDelta {
                                        event_id: format!("event_{session_id}_{frame}"),
                                        response_id: response_id.clone(),
                                        item_id: item_id.clone(),
                                        output_index: 0,
                                        content_index: 0,
                                        delta: chunk.iter().collect(),
                                    },
                                ),
                            );
                        }
                        if ctx.is_stopped() {
                            break;
                        }
                        frame += 1;
                        yield annotated_event(
                            frame,
                            RealtimeServerEvent::ResponseOutputAudioDone(
                                RealtimeServerEventResponseAudioDone {
                                    event_id: format!("event_{session_id}_{frame}"),
                                    response_id: response_id.clone(),
                                    item_id: item_id.clone(),
                                    output_index: 0,
                                    content_index: 0,
                                },
                            ),
                        );
                        frame += 1;
                        yield annotated_event(
                            frame,
                            RealtimeServerEvent::ResponseDone(
                                RealtimeServerEventResponseDone {
                                    event_id: format!("event_{session_id}_{frame}"),
                                    response: echo_response(
                                        &response_id,
                                        RealtimeResponseStatus::Completed,
                                    ),
                                },
                            ),
                        );
                    }
                    other => {
                        frame += 1;
                        yield annotated_event(
                            frame,
                            RealtimeServerEvent::Error(RealtimeServerEventError {
                                event_id: format!("event_{session_id}_{frame}"),
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
                        );
                    }
                }
            }
        };

        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

fn annotated_event(frame: u64, event: RealtimeServerEvent) -> Annotated<RealtimeServerEvent> {
    Annotated {
        id: Some(frame.to_string()),
        data: Some(event),
        event: None,
        comment: None,
        error: None,
    }
}

/// Minimal `RealtimeResponse` payload for the echo engine's `response.created`
/// and `response.done` envelope frames. Real engines populate `output`,
/// `usage`, etc.; the mock leaves them empty so the spec shape is intact
/// without pretending to have generated tokens.
fn echo_response(id: &str, status: RealtimeResponseStatus) -> RealtimeResponse {
    RealtimeResponse {
        audio: None,
        conversation_id: None,
        id: id.to_string(),
        max_output_tokens: MaxOutputTokens::Inf,
        metadata: None,
        object: "realtime.response".to_string(),
        output: Vec::new(),
        output_modalities: vec!["audio".to_string()],
        status,
        status_details: None,
        usage: None,
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

    /// `input_audio_buffer.append` produces a spec-shaped response envelope:
    /// `response.created` → `response.output_audio.delta`×N →
    /// `response.output_audio.done` → `response.done`. Frame IDs
    /// (response_id, item_id) are stable across the turn so receivers can
    /// group them. Statuses transition InProgress → Completed.
    #[tokio::test]
    async fn echo_bidirectional_append_streams_response_envelope_with_audio_deltas() {
        // 200 base64 chars → 4 deltas (3×64 + 1×8) inside the envelope.
        let audio = "A".repeat(200);
        let append = parse_client_event(serde_json::json!({
            "type": "input_audio_buffer.append",
            "audio": audio.clone(),
        }));

        let engine = EchoBidirectionalEngine {};
        let mut response_stream = engine
            .generate(make_input(vec![append]))
            .await
            .expect("generate");

        let mut events: Vec<RealtimeServerEvent> = Vec::new();
        while let Some(chunk) = response_stream.next().await {
            events.push(chunk.data.expect("annotated payload"));
        }

        // Envelope order: created, deltas..., audio.done, response.done.
        let expected_deltas = 200_usize.div_ceil(ECHO_AUDIO_DELTA_CHUNK_LEN);
        assert_eq!(events.len(), expected_deltas + 3);

        let RealtimeServerEvent::ResponseCreated(created) = &events[0] else {
            panic!(
                "first event should be response.created, got {:?}",
                events[0]
            );
        };
        assert!(matches!(
            created.response.status,
            RealtimeResponseStatus::InProgress
        ));
        let response_id = created.response.id.clone();

        let RealtimeServerEvent::ResponseDone(done) = events.last().unwrap() else {
            panic!(
                "last event should be response.done, got {:?}",
                events.last().unwrap()
            );
        };
        assert!(matches!(
            done.response.status,
            RealtimeResponseStatus::Completed
        ));
        assert_eq!(done.response.id, response_id);

        let deltas: Vec<&RealtimeServerEventResponseAudioDelta> = events
            .iter()
            .filter_map(|e| match e {
                RealtimeServerEvent::ResponseOutputAudioDelta(d) => Some(d),
                _ => None,
            })
            .collect();
        assert_eq!(deltas.len(), expected_deltas);
        assert_eq!(
            deltas.iter().map(|d| d.delta.as_str()).collect::<String>(),
            audio,
            "concatenated deltas should equal the input audio"
        );
        assert!(deltas.iter().all(|d| d.response_id == response_id));

        let audio_done: &RealtimeServerEventResponseAudioDone = events
            .iter()
            .find_map(|e| match e {
                RealtimeServerEvent::ResponseOutputAudioDone(d) => Some(d),
                _ => None,
            })
            .expect("response.output_audio.done missing");
        assert_eq!(audio_done.response_id, response_id);
    }

    /// Client events the echo engine doesn't model (anything other than
    /// SessionUpdate / InputAudioBufferAppend) are rejected with a single
    /// Error server event naming the wire-tag of the offending variant.
    #[tokio::test]
    async fn echo_bidirectional_unknown_event_emits_error() {
        let item_create = parse_client_event(serde_json::json!({
            "type": "conversation.item.create",
            "item": { "type": "message", "role": "user", "content": [] }
        }));

        let engine = EchoBidirectionalEngine {};
        let mut response_stream = engine
            .generate(make_input(vec![item_create]))
            .await
            .expect("generate");

        let chunk = response_stream.next().await.expect("one server event");
        assert!(response_stream.next().await.is_none(), "exactly one event");

        match chunk.data.expect("annotated payload") {
            RealtimeServerEvent::Error(err) => {
                assert_eq!(err.error.code.as_deref(), Some("echo_engine_unsupported"));
                assert!(err.error.message.contains("conversation.item.create"));
            }
            other => panic!("expected error, got {other:?}"),
        }
    }
}
