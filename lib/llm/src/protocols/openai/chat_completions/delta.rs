// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse};
use crate::protocols::common;

/// Provides a method for generating a [`DeltaGenerator`] from a chat completion request.
impl NvCreateChatCompletionRequest {
    /// Creates a [`DeltaGenerator`] instance based on the chat completion request.
    ///
    /// # Returns
    /// * [`DeltaGenerator`] configured with model name and response options.
    pub fn response_generator(&self) -> DeltaGenerator {
        let options = DeltaGeneratorOptions {
            enable_usage: true,
            enable_logprobs: self.inner.logprobs.unwrap_or(false),
        };

        DeltaGenerator::new(self.inner.model.clone(), options)
    }
}

/// Configuration options for the [`DeltaGenerator`], controlling response behavior.
#[derive(Debug, Clone, Default)]
pub struct DeltaGeneratorOptions {
    /// Determines whether token usage statistics should be included in the response.
    pub enable_usage: bool,
    /// Determines whether log probabilities should be included in the response.
    pub enable_logprobs: bool,
}

/// Generates incremental chat completion responses in a streaming fashion.
#[derive(Debug, Clone)]
pub struct DeltaGenerator {
    /// Unique identifier for the chat completion session.
    id: String,
    /// Object type, representing a streamed chat completion response.
    object: String,
    /// Timestamp (Unix epoch) when the response was created.
    created: u32,
    /// Model name used for generating responses.
    model: String,
    /// Optional system fingerprint for version tracking.
    system_fingerprint: Option<String>,
    /// Optional service tier information for the response.
    service_tier: Option<async_openai::types::ServiceTierResponse>,
    /// Tracks token usage for the completion request.
    usage: async_openai::types::CompletionUsage,
    /// Counter tracking the number of messages issued.
    msg_counter: u64,
    /// Configuration options for response generation.
    options: DeltaGeneratorOptions,
}

impl DeltaGenerator {
    /// Creates a new [`DeltaGenerator`] instance with the specified model and options.
    ///
    /// # Arguments
    /// * `model` - The model name used for response generation.
    /// * `options` - Configuration options for enabling usage and log probabilities.
    ///
    /// # Returns
    /// * A new instance of [`DeltaGenerator`].
    pub fn new(model: String, options: DeltaGeneratorOptions) -> Self {
        // SAFETY: Casting from `u64` to `u32` could lead to precision loss after `u32::MAX`,
        // but this will not be an issue until 2106.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;

        let usage = async_openai::types::CompletionUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        };

        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model,
            system_fingerprint: None,
            service_tier: None,
            usage,
            msg_counter: 0,
            options,
        }
    }

    /// Updates the prompt token usage count.
    ///
    /// # Arguments
    /// * `isl` - The number of prompt tokens used.
    pub fn update_isl(&mut self, isl: u32) {
        self.usage.prompt_tokens = isl;
    }

    /// Creates a choice within a chat completion response.
    ///
    /// # Arguments
    /// * `index` - The index of the choice in the completion response.
    /// * `text` - The text content for the response.
    /// * `finish_reason` - The reason why the response finished (e.g., stop, length, etc.).
    /// * `logprobs` - Optional log probabilities of the generated tokens.
    ///
    /// # Returns
    /// * An [`async_openai::types::CreateChatCompletionStreamResponse`] instance representing the choice.
    #[allow(deprecated)]
    pub fn create_choice(
        &self,
        index: u32,
        text: Option<String>,
        finish_reason: Option<async_openai::types::FinishReason>,
        logprobs: Option<async_openai::types::ChatChoiceLogprobs>,
    ) -> async_openai::types::CreateChatCompletionStreamResponse {
        // TODO: Update for tool calling
        let delta = async_openai::types::ChatCompletionStreamResponseDelta {
            role: if self.msg_counter == 0 {
                Some(async_openai::types::Role::Assistant)
            } else {
                None
            },
            content: text,
            tool_calls: None,
            function_call: None,
            refusal: None,
        };

        let choice = async_openai::types::ChatChoiceStream {
            index,
            delta,
            finish_reason,
            logprobs,
        };

        let choices = vec![choice];

        async_openai::types::CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices,
            usage: if self.options.enable_usage {
                Some(self.usage.clone())
            } else {
                None
            },
            service_tier: self.service_tier.clone(),
        }
    }
}

/// Implements the [`crate::protocols::openai::DeltaGeneratorExt`] trait for [`DeltaGenerator`], allowing
/// it to transform backend responses into OpenAI-style streaming responses.
impl crate::protocols::openai::DeltaGeneratorExt<NvCreateChatCompletionStreamResponse>
    for DeltaGenerator
{
    /// Converts a backend response into a structured OpenAI-style streaming response.
    ///
    /// # Arguments
    /// * `delta` - The backend response containing generated text and metadata.
    ///
    /// # Returns
    /// * `Ok(NvCreateChatCompletionStreamResponse)` if conversion succeeds.
    /// * `Err(anyhow::Error)` if an error occurs.
    fn choice_from_postprocessor(
        &mut self,
        delta: crate::protocols::common::llm_backend::BackendOutput,
    ) -> anyhow::Result<NvCreateChatCompletionStreamResponse> {
        // Aggregate token usage if enabled.
        if self.options.enable_usage {
            self.usage.completion_tokens += delta.token_ids.len() as u32;
        }

        // TODO: Implement log probabilities aggregation.
        let logprobs = None;

        // Map backend finish reasons to OpenAI's finish reasons.
        let finish_reason = match delta.finish_reason {
            Some(common::FinishReason::EoS) => Some(async_openai::types::FinishReason::Stop),
            Some(common::FinishReason::Stop) => Some(async_openai::types::FinishReason::Stop),
            Some(common::FinishReason::Length) => Some(async_openai::types::FinishReason::Length),
            Some(common::FinishReason::Cancelled) => Some(async_openai::types::FinishReason::Stop),
            Some(common::FinishReason::Error(err_msg)) => {
                return Err(anyhow::anyhow!(err_msg));
            }
            None => None,
        };

        // Create the streaming response.
        let index = 0;
        let stream_response = self.create_choice(index, delta.text, finish_reason, logprobs);

        Ok(NvCreateChatCompletionStreamResponse {
            inner: stream_response,
        })
    }
}
