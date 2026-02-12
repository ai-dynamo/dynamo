// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Speculative next-turn prefill for reasoning models.
//!
//! After an assistant turn completes, we know what the next turn's prompt prefix
//! will look like: the full conversation history (with thinking content stripped by
//! the Jinja template for non-last assistant turns). We render it, tokenize it,
//! and send a `max_tokens=1` request through the pipeline to warm the KV cache.

use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use dynamo_async_openai::types::ChatCompletionRequestMessage;
use futures::Stream;
use futures::stream::StreamExt;
use minijinja::value::Value;

use dynamo_runtime::engine::AsyncEngine;
use dynamo_runtime::pipeline::{Context as PipelineContext, Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;

use crate::preprocessor::ANNOTATION_LLM_METRICS;
use crate::preprocessor::prompt::{OAIChatLikeRequest, OAIPromptFormatter};
use crate::protocols::common::llm_backend::{BackendOutput, PreprocessedRequest};
use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use crate::tokenizers::traits::Tokenizer;

/// A minimal `OAIChatLikeRequest` for speculative next-turn prefill.
/// Holds the full conversation (including a new assistant message) and
/// renders with `add_generation_prompt = false` so the result is the
/// exact prefix the next user turn will see.
pub struct SpeculativePrefillRequest {
    messages: Vec<ChatCompletionRequestMessage>,
}

impl SpeculativePrefillRequest {
    pub fn new(messages: Vec<ChatCompletionRequestMessage>) -> Self {
        Self { messages }
    }
}

impl OAIChatLikeRequest for SpeculativePrefillRequest {
    fn model(&self) -> String {
        "speculative_prefill".to_string()
    }

    fn messages(&self) -> Value {
        let json = serde_json::to_value(&self.messages).unwrap();
        Value::from_serialize(&json)
    }

    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        Some(&self.messages)
    }

    fn should_add_generation_prompt(&self) -> bool {
        false
    }
}

/// Optionally wraps a chat completion response stream to enable speculative
/// next-turn prefill. When `nvext.speculative_prefill` is set, the returned
/// stream accumulates the assistant response text and, on completion, spawns
/// a background task that renders the next-turn prefix and fires a
/// `max_tokens=1` request through the pipeline to warm the KV cache.
///
/// When the flag is not set, returns the stream unmodified with zero overhead.
pub fn maybe_wrap_stream(
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>,
    request: &NvCreateChatCompletionRequest,
    next: &Arc<
        dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>,
    >,
    formatter: &Arc<dyn OAIPromptFormatter>,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>> {
    let enabled = request
        .nvext
        .as_ref()
        .and_then(|ext| ext.speculative_prefill)
        .unwrap_or(false);

    if !enabled {
        return stream;
    }

    let (tx, rx) = tokio::sync::oneshot::channel::<String>();

    let next = next.clone();
    let formatter = formatter.clone();
    let tokenizer = tokenizer.clone();
    let messages = request.inner.messages.clone();
    tokio::spawn(async move {
        let Ok(response_text) = rx.await else {
            return;
        };
        if let Err(e) = prefill_task(next, formatter, tokenizer, messages, response_text).await {
            tracing::warn!(error = %e, "Speculative prefill failed");
        }
    });

    let mut accumulated_text = String::new();
    let mut prefill_tx = Some(tx);
    Box::pin(stream.map(move |item| {
        if let Some(ref resp) = item.data {
            for choice in &resp.choices {
                if let Some(dynamo_async_openai::types::ChatCompletionMessageContent::Text(
                    ref text,
                )) = choice.delta.content
                {
                    accumulated_text.push_str(text);
                }
            }
        }

        if item.event.as_deref() == Some(ANNOTATION_LLM_METRICS)
            && item.data.is_none()
            && let Some(tx) = prefill_tx.take()
        {
            let _ = tx.send(std::mem::take(&mut accumulated_text));
        }

        item
    }))
}

/// Fire-and-forget task that renders the next-turn prefix and sends it
/// through the pipeline as a `max_tokens=1` request to warm the KV cache.
async fn prefill_task(
    next: Arc<
        dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>,
    >,
    formatter: Arc<dyn OAIPromptFormatter>,
    tokenizer: Arc<dyn Tokenizer>,
    original_messages: Vec<ChatCompletionRequestMessage>,
    response_text: String,
) -> Result<()> {
    let assistant_msg = ChatCompletionRequestMessage::Assistant(
        dynamo_async_openai::types::ChatCompletionRequestAssistantMessage {
            content: Some(
                dynamo_async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(
                    response_text,
                ),
            ),
            ..Default::default()
        },
    );

    let mut messages = original_messages;
    messages.push(assistant_msg);

    let prefill_request = SpeculativePrefillRequest::new(messages);
    let formatted_prompt = formatter.render(&prefill_request)?;
    let encoding = tokenizer.encode(&formatted_prompt)?;
    let token_ids = encoding.token_ids().to_vec();

    tracing::debug!(
        num_tokens = token_ids.len(),
        "Speculative prefill: sending next-turn prefix"
    );

    let preprocessed = PreprocessedRequest::builder()
        .model("speculative_prefill".to_string())
        .token_ids(token_ids)
        .stop_conditions(StopConditions {
            max_tokens: Some(1),
            ..Default::default()
        })
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions::default())
        .eos_token_ids(vec![])
        .annotations(vec![])
        .build()?;

    let context = PipelineContext::with_id(preprocessed, uuid::Uuid::new_v4().to_string());
    let _ = next.generate(context).await;

    Ok(())
}
