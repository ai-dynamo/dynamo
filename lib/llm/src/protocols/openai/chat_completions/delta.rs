// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse};
use crate::{
    local_model::runtime_config::ModelRuntimeConfig,
    protocols::common::{self},
    types::TokenIdType,
};
use anyhow::Context;
use dynamo_parsers::{ParserResult, ReasoningParser, ReasoningParserType, ReasoningParserWrapper};

/// Provides a method for generating a [`DeltaGenerator`] from a chat completion request.
impl NvCreateChatCompletionRequest {
    /// Creates a [`DeltaGenerator`] instance based on the chat completion request.
    ///
    /// # Arguments
    /// * `request_id` - The request ID to use for the chat completion response ID.
    ///
    /// # Returns
    /// * [`DeltaGenerator`] configured with model name and response options.
    pub fn response_generator(&self, request_id: String) -> DeltaGenerator {
        let options = DeltaGeneratorOptions {
            enable_usage: self
                .inner
                .stream_options
                .as_ref()
                .map(|opts| opts.include_usage)
                .unwrap_or(false),
            enable_logprobs: self.inner.logprobs.unwrap_or(false)
                || self.inner.top_logprobs.unwrap_or(0) > 0,
            runtime_config: ModelRuntimeConfig::default(),
        };

        DeltaGenerator::new(self.inner.model.clone(), options, request_id)
    }
}

/// Configuration options for the [`DeltaGenerator`], controlling response behavior.
#[derive(Debug, Clone, Default)]
pub struct DeltaGeneratorOptions {
    /// Determines whether token usage statistics should be included in the response.
    pub enable_usage: bool,
    /// Determines whether log probabilities should be included in the response.
    pub enable_logprobs: bool,

    pub runtime_config: ModelRuntimeConfig,
}

/// Generates incremental chat completion responses in a streaming fashion.
pub struct DeltaGenerator {
    /// Unique identifier for the chat completion session.
    id: String,
    /// Object type, representing a streamed chat completion response.
    object: String,
    /// Timestamp (Unix epoch) when the response was created.
    created: u32,
    model: String,
    /// Optional system fingerprint for version tracking.
    system_fingerprint: Option<String>,
    /// Optional service tier information for the response.
    service_tier: Option<dynamo_async_openai::types::ServiceTierResponse>,
    /// Tracks token usage for the completion request.
    usage: dynamo_async_openai::types::CompletionUsage,
    /// Counter tracking the number of messages issued.
    msg_counter: u64,
    /// Configuration options for response generation.
    options: DeltaGeneratorOptions,

    /// Reasoning Parser object
    /// This is used to parse reasoning content in the response.
    /// None means no reasoning parsing will be performed.
    reasoning_parser: Option<ReasoningParserWrapper>,

    /// Tokenizer for accurate reasoning token counting
    tokenizer: Option<std::sync::Arc<dyn crate::tokenizers::traits::Tokenizer>>,
    /// Counter for reasoning tokens (separate from completion_tokens)
    reasoning_tokens: u32,
}

impl std::fmt::Debug for DeltaGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaGenerator")
            .field("id", &self.id)
            .field("object", &self.object)
            .field("created", &self.created)
            .field("model", &self.model)
            .field("system_fingerprint", &self.system_fingerprint)
            .field("service_tier", &self.service_tier)
            .field("usage", &self.usage)
            .field("msg_counter", &self.msg_counter)
            .field("options", &self.options)
            .field("reasoning_parser", &self.reasoning_parser)
            .field("tokenizer", &"<tokenizer>") // Don't try to debug the tokenizer
            .field("reasoning_tokens", &self.reasoning_tokens)
            .finish()
    }
}

impl DeltaGenerator {
    /// Creates a new [`DeltaGenerator`] instance with the specified model and options.
    ///
    /// # Arguments
    /// * `model` - The model name used for response generation.
    /// * `options` - Configuration options for enabling usage and log probabilities.
    /// * `request_id` - The request ID to use for the chat completion response.
    ///
    /// # Returns
    /// * A new instance of [`DeltaGenerator`].
    pub fn new(model: String, options: DeltaGeneratorOptions, request_id: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // SAFETY: Casting from `u64` to `u32` could lead to precision loss after `u32::MAX`,
        // but this will not be an issue until 2106.
        let now: u32 = now.try_into().expect("timestamp exceeds u32::MAX");

        let usage = dynamo_async_openai::types::CompletionUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        };

        // Reasoning parser type
        // If no parser is specified (None), no reasoning parsing will be performed
        let reasoning_parser = options
            .runtime_config
            .reasoning_parser
            .as_deref()
            .map(ReasoningParserType::get_reasoning_parser_from_name);

        let chatcmpl_id = format!("chatcmpl-{request_id}");

        Self {
            id: chatcmpl_id,
            object: "chat.completion.chunk".to_string(),
            created: now,
            model,
            system_fingerprint: None,
            service_tier: None,
            usage,
            msg_counter: 0,
            options,
            reasoning_parser,
            tokenizer: None,
            reasoning_tokens: 0,
        }
    }

    /// Update runtime configuration and reconfigure the reasoning parser accordingly.
    pub fn set_reasoning_parser(&mut self, runtime_config: ModelRuntimeConfig) {
        self.options.runtime_config = runtime_config.clone();
        match self.options.runtime_config.reasoning_parser.as_deref() {
            Some(name) => {
                self.reasoning_parser =
                    Some(ReasoningParserType::get_reasoning_parser_from_name(name));
            }
            None => {
                self.reasoning_parser = None;
            }
        }
    }

    /// Updates the prompt token usage count.
    ///
    /// # Arguments
    /// * `isl` - The number of prompt tokens used.
    pub fn update_isl(&mut self, isl: u32) {
        self.usage.prompt_tokens = isl;
    }

    /// Sets the tokenizer for accurate reasoning token counting.
    ///
    /// # Arguments
    /// * `tokenizer` - Optional tokenizer for token counting
    pub fn set_tokenizer(
        &mut self,
        tokenizer: Option<std::sync::Arc<dyn crate::tokenizers::traits::Tokenizer>>,
    ) {
        self.tokenizer = tokenizer;
    }

    /// Count tokens in reasoning vs normal text with fallback strategy.
    ///
    /// # Arguments
    /// * `reasoning_text` - Text content identified as reasoning
    /// * `normal_text` - Text content identified as normal response
    /// * `total_tokens` - Total number of tokens in the chunk
    ///
    /// # Returns
    /// * `(reasoning_token_count, normal_token_count)` - Tuple of token counts
    fn count_reasoning_tokens(
        &self,
        reasoning_text: &str,
        normal_text: &str,
        total_tokens: u32,
    ) -> (u32, u32) {
        // PRIMARY: Try accurate tokenizer-based counting
        if let Some(_tokenizer) = &self.tokenizer {
            match self.tokenize_accurately(reasoning_text, normal_text) {
                Ok((reasoning_count, normal_count)) => {
                    tracing::debug!(
                        reasoning_tokens = reasoning_count,
                        normal_tokens = normal_count,
                        method = "tokenizer_accurate",
                        "Token counting successful"
                    );
                    return (reasoning_count, normal_count);
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        reasoning_text_len = reasoning_text.len(),
                        normal_text_len = normal_text.len(),
                        "Tokenizer encoding failed, falling back to approximation"
                    );
                }
            }
        } else {
            tracing::debug!(
                reasoning_text_len = reasoning_text.len(),
                normal_text_len = normal_text.len(),
                "No tokenizer available, using approximation"
            );
        }

        // FALLBACK: Character-ratio approximation (only when necessary)
        self.approximate_token_counts(reasoning_text, normal_text, total_tokens)
    }

    /// Accurate tokenizer-based counting.
    ///
    /// # Arguments
    /// * `reasoning_text` - Text content identified as reasoning
    /// * `normal_text` - Text content identified as normal response
    ///
    /// # Returns
    /// * `Result<(reasoning_count, normal_count), anyhow::Error>` - Token counts or error
    fn tokenize_accurately(
        &self,
        reasoning_text: &str,
        normal_text: &str,
    ) -> anyhow::Result<(u32, u32)> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No tokenizer available"))?;

        // Handle empty text cases efficiently
        let reasoning_count = if reasoning_text.is_empty() {
            0
        } else {
            tokenizer
                .encode(reasoning_text)
                .with_context(|| format!("Failed to encode reasoning text: '{}'", reasoning_text))?
                .token_ids()
                .len() as u32
        };

        let normal_count = if normal_text.is_empty() {
            0
        } else {
            tokenizer
                .encode(normal_text)
                .with_context(|| format!("Failed to encode normal text: '{}'", normal_text))?
                .token_ids()
                .len() as u32
        };

        Ok((reasoning_count, normal_count))
    }

    /// Fallback approximation (only when accurate method fails).
    ///
    /// # Arguments
    /// * `reasoning_text` - Text content identified as reasoning
    /// * `normal_text` - Text content identified as normal response
    /// * `total_tokens` - Total number of tokens in the chunk
    ///
    /// # Returns
    /// * `(reasoning_count, normal_count)` - Approximated token counts
    fn approximate_token_counts(
        &self,
        reasoning_text: &str,
        normal_text: &str,
        total_tokens: u32,
    ) -> (u32, u32) {
        let total_chars = reasoning_text.len() + normal_text.len();

        if total_chars == 0 {
            tracing::debug!("Empty text content, no tokens to count");
            return (0, total_tokens);
        }

        let reasoning_ratio = reasoning_text.len() as f32 / total_chars as f32;
        let reasoning_tokens = (total_tokens as f32 * reasoning_ratio).round() as u32;
        let normal_tokens = total_tokens.saturating_sub(reasoning_tokens);

        tracing::warn!(
            reasoning_tokens,
            normal_tokens,
            reasoning_ratio,
            total_chars,
            total_tokens,
            method = "character_approximation",
            "Using fallback token counting method"
        );

        (reasoning_tokens, normal_tokens)
    }

    pub fn create_logprobs(
        &self,
        tokens: Vec<common::llm_backend::TokenType>,
        token_ids: &[TokenIdType],
        logprobs: Option<common::llm_backend::LogProbs>,
        top_logprobs: Option<common::llm_backend::TopLogprobs>,
    ) -> Option<dynamo_async_openai::types::ChatChoiceLogprobs> {
        if !self.options.enable_logprobs || logprobs.is_none() {
            return None;
        }

        let toks = tokens
            .into_iter()
            .zip(token_ids)
            .map(|(token, token_id)| (token.unwrap_or_default(), *token_id))
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
                            dynamo_async_openai::types::TopLogprobs {
                                token: top_t,
                                logprob: top_lp.logprob as f32,
                                bytes: None,
                            }
                        })
                        .collect::<Vec<dynamo_async_openai::types::TopLogprobs>>();
                    if !found_selected_token {
                        // If the selected token is not in the top logprobs, add it
                        converted_top_lps.push(dynamo_async_openai::types::TopLogprobs {
                            token: t.clone(),
                            logprob: lp,
                            bytes: None,
                        });
                    }
                    dynamo_async_openai::types::ChatCompletionTokenLogprob {
                        token: t.clone(),
                        logprob: lp,
                        bytes: None,
                        top_logprobs: converted_top_lps,
                    }
                })
                .collect()
        });

        Some(dynamo_async_openai::types::ChatChoiceLogprobs {
            content,
            refusal: None,
        })
    }

    fn create_reasoning_content(
        &mut self,
        text: &Option<String>,
        token_ids: &[u32],
    ) -> Option<ParserResult> {
        // If no reasoning parser is configured, return None
        let reasoning_parser = self.reasoning_parser.as_mut()?;

        let text_ref = text.as_deref().unwrap_or("");
        if text_ref.is_empty() && token_ids.is_empty() {
            return None;
        }
        let parser_result =
            reasoning_parser.parse_reasoning_streaming_incremental(text_ref, token_ids);

        Some(parser_result)
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
    /// * An [`dynamo_async_openai::types::CreateChatCompletionStreamResponse`] instance representing the choice.
    #[allow(deprecated)]
    pub fn create_choice(
        &mut self,
        index: u32,
        text: Option<String>,
        reasoning_content: Option<String>,
        finish_reason: Option<dynamo_async_openai::types::FinishReason>,
        logprobs: Option<dynamo_async_openai::types::ChatChoiceLogprobs>,
    ) -> NvCreateChatCompletionStreamResponse {
        let delta = dynamo_async_openai::types::ChatCompletionStreamResponseDelta {
            content: text,
            function_call: None,
            tool_calls: None,
            role: if self.msg_counter == 0 {
                Some(dynamo_async_openai::types::Role::Assistant)
            } else {
                None
            },
            refusal: None,
            reasoning_content,
        };

        let choice = dynamo_async_openai::types::ChatChoiceStream {
            index,
            delta,
            finish_reason,
            logprobs,
        };

        let choices = vec![choice];

        // According to OpenAI spec: when stream_options.include_usage is true,
        // all intermediate chunks should have usage: null
        // The final usage chunk will be sent separately with empty choices
        dynamo_async_openai::types::CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices,
            usage: None, // Always None for chunks with content/choices
            service_tier: self.service_tier.clone(),
        }
    }

    /// Creates a final usage-only chunk for OpenAI compliance.
    /// This should be sent after the last content chunk when stream_options.include_usage is true.
    ///
    /// # Returns
    /// * A [`CreateChatCompletionStreamResponse`] with empty choices and usage stats.
    pub fn create_usage_chunk(&self) -> NvCreateChatCompletionStreamResponse {
        let mut usage = self.usage.clone();
        usage.total_tokens = usage.prompt_tokens.saturating_add(usage.completion_tokens);

        // Add reasoning tokens to response if reasoning parser is configured or reasoning tokens detected
        if self.reasoning_parser.is_some() || self.reasoning_tokens > 0 {
            let mut details = usage.completion_tokens_details.unwrap_or_default();

            if self.reasoning_tokens > 0 {
                details.reasoning_tokens = Some(self.reasoning_tokens);
                // Include reasoning tokens in total count
                usage.total_tokens = usage.total_tokens.saturating_add(self.reasoning_tokens);

                tracing::info!(
                    prompt_tokens = usage.prompt_tokens,
                    completion_tokens = usage.completion_tokens,
                    reasoning_tokens = self.reasoning_tokens,
                    total_tokens = usage.total_tokens,
                    "Final usage statistics with reasoning tokens"
                );
            } else {
                details.reasoning_tokens = None;

                tracing::debug!(
                    prompt_tokens = usage.prompt_tokens,
                    completion_tokens = usage.completion_tokens,
                    total_tokens = usage.total_tokens,
                    "Usage statistics - reasoning parser configured but no reasoning tokens detected"
                );
            }

            usage.completion_tokens_details = Some(details);
        }

        dynamo_async_openai::types::CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![], // Empty choices for usage-only chunk
            usage: Some(usage),
            service_tier: self.service_tier.clone(),
        }
    }

    /// Check if usage tracking is enabled
    pub fn is_usage_enabled(&self) -> bool {
        self.options.enable_usage
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
        // SAFETY: Casting from `usize` to `u32` could lead to precision loss after `u32::MAX`,
        // but this will not be an issue until context lengths exceed 4_294_967_295.
        let total_tokens: u32 = delta
            .token_ids
            .len()
            .try_into()
            .expect("token_ids length exceeds u32::MAX");

        let logprobs = self.create_logprobs(
            delta.tokens,
            &delta.token_ids,
            delta.log_probs,
            delta.top_logprobs,
        );

        // Map backend finish reasons to OpenAI's finish reasons.
        let finish_reason = match delta.finish_reason {
            Some(common::FinishReason::EoS) => Some(dynamo_async_openai::types::FinishReason::Stop),
            Some(common::FinishReason::Stop) => {
                Some(dynamo_async_openai::types::FinishReason::Stop)
            }
            Some(common::FinishReason::Length) => {
                Some(dynamo_async_openai::types::FinishReason::Length)
            }
            Some(common::FinishReason::Cancelled) => {
                Some(dynamo_async_openai::types::FinishReason::Stop)
            }
            Some(common::FinishReason::ContentFilter) => {
                Some(dynamo_async_openai::types::FinishReason::ContentFilter)
            }
            Some(common::FinishReason::Error(err_msg)) => {
                return Err(anyhow::anyhow!(err_msg));
            }
            None => None,
        };

        // Handle reasoning parsing if enabled, otherwise treat all text as normal
        let (normal_text, reasoning_content) =
            match self.create_reasoning_content(&delta.text, &delta.token_ids) {
                Some(reasoning_parser_result) => (
                    reasoning_parser_result.get_some_normal_text(),
                    reasoning_parser_result.get_some_reasoning(),
                ),
                None => (delta.text, None),
            };

        // Enhanced token counting with reasoning support
        if self.options.enable_usage && total_tokens > 0 {
            let reasoning_text = reasoning_content.as_deref().unwrap_or("");
            let normal_text_str = normal_text.as_deref().unwrap_or("");

            if !reasoning_text.is_empty() {
                // Split tokens between reasoning and normal content
                let (reasoning_count, normal_count) =
                    self.count_reasoning_tokens(reasoning_text, normal_text_str, total_tokens);

                self.reasoning_tokens += reasoning_count;
                self.usage.completion_tokens += normal_count;

                tracing::trace!(
                    reasoning_tokens = reasoning_count,
                    normal_tokens = normal_count,
                    total_tokens,
                    "Token counting completed with reasoning split"
                );
            } else {
                // No reasoning content, all tokens are normal completion tokens
                self.usage.completion_tokens += total_tokens;

                tracing::trace!(
                    completion_tokens = total_tokens,
                    "Token counting completed - all tokens as completion"
                );
            }
        }

        // Create the streaming response.
        let index = 0;
        let stream_response = self.create_choice(
            index,
            normal_text,
            reasoning_content,
            finish_reason,
            logprobs,
        );

        Ok(stream_response)
    }

    fn get_isl(&self) -> Option<u32> {
        Some(self.usage.prompt_tokens)
    }

    fn create_usage_chunk(&self) -> NvCreateChatCompletionStreamResponse {
        DeltaGenerator::create_usage_chunk(self)
    }

    fn is_usage_enabled(&self) -> bool {
        DeltaGenerator::is_usage_enabled(self)
    }
}
