// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Chat inference independently of HTTP request extraction and response delivery.

use crate::protocols::{
    Annotated,
    openai::{
        ParsingOptions,
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
    },
};
use async_trait::async_trait;
use dynamo_runtime::{
    error::DynamoError,
    pipeline::{Context, ManyOut},
};

mod dynamo;
pub use dynamo::DynamoChatController;

/// The canonical model name and a bounded-cardinality label for HTTP metrics.
pub struct ChatModel {
    pub name: String,
    pub metric_name: String,
}

/// Typed output shared by JSON aggregation and SSE delivery.
///
/// The stream must retain the input request's cancellation context and typed
/// errors. Parser options must describe the engine that produced this stream.
pub struct ChatOutput {
    pub stream: ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    pub parsing_options: ParsingOptions,
    /// Some parsers buffer all visible output; HTTP must keep that connection alive.
    pub can_defer_all_output: bool,
}

/// Failures before a response stream exists. HTTP mapping belongs to the handler.
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("Model not found")]
    ModelNotFound,
    #[error("Model unavailable: {0}")]
    ModelUnavailable(String),
    #[error(transparent)]
    InvalidRequest(DynamoError),
    #[error(transparent)]
    Generation(anyhow::Error),
}

/// Replaceable implementation behind the shared chat-completions HTTP API.
///
/// HTTP owns decoding, templates, common API validation, metadata extraction,
/// metrics and JSON/SSE delivery. The controller owns model resolution,
/// model-specific readiness, parser selection and inference. Process readiness
/// and endpoint enablement remain HTTP service concerns.
#[async_trait]
pub trait ChatController: Send + Sync {
    fn resolve_model(&self, model: &str) -> ChatModel;

    /// Check model readiness before request dispatch. An unknown model may be
    /// rejected here or by `generate`; the Dynamo implementation defers it to
    /// generation to preserve validation and error precedence.
    fn check_model_ready(&self, model: &str) -> Result<(), ChatError>;

    /// Takes ownership of the validated request without serializing or copying
    /// its payload. Both unary and streaming HTTP requests use this operation.
    async fn generate(
        &self,
        request: Context<NvCreateChatCompletionRequest>,
    ) -> Result<ChatOutput, ChatError>;
}
