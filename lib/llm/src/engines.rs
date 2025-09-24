// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;

use async_stream::stream;
use async_trait::async_trait;

use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;

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
            dynamo_async_openai::types::ChatCompletionRequestMessage::User(user_msg) => {
                match user_msg.content {
                    dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
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
                yield Annotated{ id: Some(id.to_string()), data: Some(response), event: None, comment: None };
                id += 1;
            }

            let response = deltas.create_choice(0, None, None, Some(dynamo_async_openai::types::FinishReason::Stop), None);
            yield Annotated { id: Some(id.to_string()), data: Some(response), event: None, comment: None };
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
                yield Annotated{ id: Some(id.to_string()), data: Some(response), event: None, comment: None };
                id += 1;
            }
            let response = deltas.create_choice(0, None, Some(dynamo_async_openai::types::CompletionFinishReason::Stop), None);
            yield Annotated { id: Some(id.to_string()), data: Some(response), event: None, comment: None };

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
