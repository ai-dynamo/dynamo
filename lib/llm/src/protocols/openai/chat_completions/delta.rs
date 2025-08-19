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
use crate::{
    protocols::common::{self},
    types::TokenIdType,
};

/// Provides a method for generating a [`DeltaGenerator`] from a chat completion request.
impl NvCreateChatCompletionRequest {
    /// Creates a [`DeltaGenerator`] instance based on the chat completion request.
    ///
    /// # Returns
    /// * [`DeltaGenerator`] configured with model name and response options.
    pub fn response_generator(&self) -> DeltaGenerator {
        let options = DeltaGeneratorOptions {
            enable_usage: true,
            enable_logprobs: self.inner.logprobs.unwrap_or(false)
                || self.inner.top_logprobs.unwrap_or(0) > 0,
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
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // SAFETY: Casting from `u64` to `u32` could lead to precision loss after `u32::MAX`,
        // but this will not be an issue until 2106.
        let now: u32 = now.try_into().expect("timestamp exceeds u32::MAX");

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

    pub fn create_logprobs(
        &self,
        tokens: Vec<common::llm_backend::TokenType>,
        token_ids: Vec<TokenIdType>,
        logprobs: Option<common::llm_backend::LogProbs>,
        top_logprobs: Option<common::llm_backend::TopLogprobs>,
    ) -> Option<async_openai::types::ChatChoiceLogprobs> {
        if !self.options.enable_logprobs || logprobs.is_none() {
            return None;
        }

        let toks = tokens
            .into_iter()
            .zip(token_ids)
            .map(|(token, token_id)| (token.unwrap_or_default(), token_id))
            .collect::<Vec<(String, TokenIdType)>>();
        let tok_lps = toks
            .iter()
            .zip(logprobs.unwrap())
            .map(|(_, lp)| lp as f32)
            .collect::<Vec<f32>>();

        let content = top_logprobs.map(|top_logprobs| {
            toks.iter()
                .zip(tok_lps)
                .zip(top_logprobs)
                .map(|(((t, tid), lp), top_lps)| {
                    let mut found_selected_token = false;
                    let mut converted_top_lps = top_lps
                        .iter()
                        .map(|top_lp| {
                            let top_t = top_lp.token.clone().unwrap_or_default();
                            let top_tid = top_lp.token_id;
                            found_selected_token = found_selected_token || top_tid == *tid;
                            async_openai::types::TopLogprobs {
                                token: top_t,
                                logprob: top_lp.logprob as f32,
                                bytes: None,
                            }
                        })
                        .collect::<Vec<async_openai::types::TopLogprobs>>();
                    if !found_selected_token {
                        // If the selected token is not in the top logprobs, add it
                        converted_top_lps.push(async_openai::types::TopLogprobs {
                            token: t.clone(),
                            logprob: lp,
                            bytes: None,
                        });
                    }
                    async_openai::types::ChatCompletionTokenLogprob {
                        token: t.clone(),
                        logprob: lp,
                        bytes: None,
                        top_logprobs: converted_top_lps,
                    }
                })
                .collect()
        });

        Some(async_openai::types::ChatChoiceLogprobs {
            content,
            refusal: None,
        })
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
        let delta = async_openai::types::ChatCompletionStreamResponseDelta {
            content: text,
            function_call: None,
            tool_calls: None,
            role: if self.msg_counter == 0 {
                Some(async_openai::types::Role::Assistant)
            } else {
                None
            },
            refusal: None,
        };

        let choice = async_openai::types::ChatChoiceStream {
            index,
            delta,
            finish_reason,
            logprobs,
        };

        let choices = vec![choice];

        let mut usage = self.usage.clone();
        if self.options.enable_usage {
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
        }

        async_openai::types::CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices,
            usage: if self.options.enable_usage {
                Some(usage)
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
            // SAFETY: Casting from `usize` to `u32` could lead to precision loss after `u32::MAX`,
            // but this will not be an issue until context lengths exceed 4_294_967_295.
            let token_length: u32 = delta
                .token_ids
                .len()
                .try_into()
                .expect("token_ids length exceeds u32::MAX");

            self.usage.completion_tokens += token_length;
        }

        let logprobs = self.create_logprobs(
            delta.tokens,
            delta.token_ids,
            delta.log_probs,
            delta.top_logprobs,
        );

        // Map backend finish reasons to OpenAI's finish reasons.
        let finish_reason = match delta.finish_reason {
            Some(common::FinishReason::EoS) => Some(async_openai::types::FinishReason::Stop),
            Some(common::FinishReason::Stop) => Some(async_openai::types::FinishReason::Stop),
            Some(common::FinishReason::Length) => Some(async_openai::types::FinishReason::Length),
            Some(common::FinishReason::Cancelled) => Some(async_openai::types::FinishReason::Stop),
            Some(common::FinishReason::ContentFilter) => {
                Some(async_openai::types::FinishReason::ContentFilter)
            }
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

    fn get_isl(&self) -> Option<u32> {
        Some(self.usage.prompt_tokens)
    }
}
