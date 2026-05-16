// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent, FinishReason,
};
use dynamo_runtime::protocols::annotated::Annotated;

pub fn last_user_text(req: &NvCreateChatCompletionRequest, chunk_index: u64) -> String {
    req.inner
        .messages
        .iter()
        .next_back()
        .and_then(|msg| match msg {
            ChatCompletionRequestMessage::User(user_msg) => match &user_msg.content {
                ChatCompletionRequestUserMessageContent::Text(prompt) => Some(prompt.clone()),
                _ => None,
            },
            _ => None,
        })
        .unwrap_or_else(|| format!("<chunk {chunk_index}: non-text content>"))
}

pub fn chunk_finished(chunk: &Annotated<NvCreateChatCompletionStreamResponse>) -> bool {
    chunk
        .data
        .as_ref()
        .map(|data| {
            data.inner
                .choices
                .iter()
                .any(|choice| choice.finish_reason == Some(FinishReason::Stop))
        })
        .unwrap_or(false)
}
