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

/// Bidirectional echo engine: consumes a stream of `NvCreateChatCompletionRequest`s
/// and, for each one, emits character-by-character `NvCreateChatCompletionStreamResponse`
/// chunks followed by a terminal `FinishReason::Stop`. Used by the experimental
/// `/v1/realtime` WebSocket endpoint to demonstrate end-to-end bidirectional plumbing.
pub struct EchoBidirectionalEngine {}

#[async_trait]
impl
    AsyncEngine<
        ManyIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for EchoBidirectionalEngine
{
    async fn generate(
        &self,
        mut incoming: ManyIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let ctx = incoming.context();
        let session_id = ctx.id().to_string();
        let ctx_for_stream = ctx.clone();

        let output = stream! {
            let ctx = ctx_for_stream;
            let mut id: u64 = 1;
            let mut chunk_index: u64 = 0;

            while let Some(req) = incoming.next().await {
                if ctx.is_stopped() {
                    break;
                }
                chunk_index += 1;

                let summary = req
                    .inner
                    .messages
                    .iter()
                    .next_back()
                    .and_then(|msg| match msg {
                        dynamo_protocols::types::ChatCompletionRequestMessage::User(user_msg) => {
                            match &user_msg.content {
                                dynamo_protocols::types::ChatCompletionRequestUserMessageContent::Text(prompt) => Some(prompt.clone()),
                                _ => None,
                            }
                        }
                        _ => None,
                    })
                    .unwrap_or_else(|| format!("<chunk {chunk_index}: non-text content>"));

                let mut deltas = req.response_generator(format!("{session_id}-{chunk_index}"));

                for c in summary.chars() {
                    if ctx.is_stopped() {
                        break;
                    }
                    tokio::time::sleep(*TOKEN_ECHO_DELAY).await;
                    let response = deltas.create_choice(0, Some(c.to_string()), None, None);
                    yield Annotated {
                        id: Some(id.to_string()),
                        data: Some(response),
                        event: None,
                        comment: None,
                        error: None,
                    };
                    id += 1;
                }

                if !ctx.is_stopped() {
                    let response = deltas.create_choice(
                        0,
                        None,
                        Some(dynamo_protocols::types::FinishReason::Stop),
                        None,
                    );
                    yield Annotated {
                        id: Some(id.to_string()),
                        data: Some(response),
                        event: None,
                        comment: None,
                        error: None,
                    };
                    id += 1;
                }
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
                let response = deltas.create_choice(0, Some(c.to_string()), None, None);
                yield Annotated{ id: Some(id.to_string()), data: Some(response), event: None, comment: None, error: None };
                id += 1;
            }

            let response =
                deltas.create_choice(0, None, Some(dynamo_protocols::types::FinishReason::Stop), None);
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
            let response = deltas.create_choice(
                0,
                None,
                Some(dynamo_protocols::types::CompletionFinishReason::Stop),
                None,
            );
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

    fn make_user_request(text: &str) -> NvCreateChatCompletionRequest {
        let body = serde_json::json!({
            "model": "echo",
            "messages": [{ "role": "user", "content": text }],
        });
        serde_json::from_value(body).expect("valid chat completion request")
    }

    fn collect_text(
        annotated_chunks: &[Annotated<NvCreateChatCompletionStreamResponse>],
    ) -> String {
        use dynamo_protocols::types::ChatCompletionMessageContent;
        annotated_chunks
            .iter()
            .filter_map(|chunk| chunk.data.as_ref())
            .flat_map(|resp| resp.inner.choices.iter())
            .filter_map(|choice| match choice.delta.content.as_ref()? {
                ChatCompletionMessageContent::Text(s) => Some(s.clone()),
                _ => None,
            })
            .collect()
    }

    fn count_finish_stops(
        annotated_chunks: &[Annotated<NvCreateChatCompletionStreamResponse>],
    ) -> usize {
        use dynamo_protocols::types::FinishReason;
        annotated_chunks
            .iter()
            .filter_map(|chunk| chunk.data.as_ref())
            .flat_map(|resp| resp.inner.choices.iter())
            .filter(|c| matches!(c.finish_reason, Some(FinishReason::Stop)))
            .count()
    }

    fn make_input(
        requests: Vec<NvCreateChatCompletionRequest>,
    ) -> ManyIn<NvCreateChatCompletionRequest> {
        RequestStream::new(Box::pin(stream::iter(requests)), Context::new(()).context())
    }

    /// Drive a fixed sequence of two requests through the bidirectional echo engine
    /// and assert: every char of every prompt is echoed in order, and each request
    /// gets its own terminal `FinishReason::Stop`.
    #[tokio::test]
    async fn echo_bidirectional_emits_per_char_then_finish() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            // SAFETY: runs at most once, before any worker thread reads the env;
            // `TOKEN_ECHO_DELAY` captures it via `LazyLock` on first use.
            unsafe {
                std::env::set_var("DYN_TOKEN_ECHO_DELAY_MS", "0");
            }
        });

        let engine = EchoBidirectionalEngine {};
        let input = make_input(vec![make_user_request("hi"), make_user_request("ok")]);

        let mut response_stream = engine.generate(input).await.expect("generate");
        let mut chunks = Vec::new();
        while let Some(chunk) = response_stream.next().await {
            chunks.push(chunk);
        }

        assert_eq!(collect_text(&chunks), "hiok");
        assert_eq!(count_finish_stops(&chunks), 2);
    }

    // Cancellation is verified at the WebSocket integration level rather than here:
    // `TOKEN_ECHO_DELAY` is a `LazyLock` that captures `DYN_TOKEN_ECHO_DELAY_MS` once
    // per process, so tests sharing the binary cannot independently dial the per-char
    // delay. The integration test in lib/llm/tests/http_websocket.rs exercises the
    // client-disconnect path through the full handler.
}
