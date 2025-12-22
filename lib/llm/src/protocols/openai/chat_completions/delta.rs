// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse};
use crate::{
    local_model::runtime_config::ModelRuntimeConfig,
    protocols::{
        common::{self, timing::RequestTracker},
        openai::nvext::{NvExtProvider, NvExtResponse, TimingInfo},
    },
    types::TokenIdType,
};

/// Provides a method for generating a [`DeltaGenerator`] from a chat completion request.
impl NvCreateChatCompletionRequest {
    /// Enables usage tracking for non-streaming requests to comply with OpenAI API specification.
    ///
    /// According to OpenAI API spec, non-streaming chat completion responses (stream=false)
    /// must always include usage statistics. This method ensures `stream_options.include_usage`
    /// is set to `true` for non-streaming requests.
    ///
    /// # Arguments
    /// * `original_stream_flag` - The original value of the `stream` field before any internal processing
    pub fn enable_usage_for_nonstreaming(&mut self, original_stream_flag: bool) {
        if !original_stream_flag {
            // For non-streaming requests (stream=false), enable usage by default
            if self.inner.stream_options.is_none() {
                self.inner.stream_options =
                    Some(dynamo_async_openai::types::ChatCompletionStreamOptions {
                        include_usage: true,
                    });
            } else if let Some(ref mut opts) = self.inner.stream_options {
                // If stream_options exists, ensure include_usage is true for non-streaming
                opts.include_usage = true;
            }
        }
    }

    /// Creates a [`DeltaGenerator`] instance based on the chat completion request.
    ///
    /// # Arguments
    /// * `request_id` - The request ID to use for the chat completion response ID.
    ///
    /// # Returns
    /// * [`DeltaGenerator`] configured with model name and response options.
    pub fn response_generator(&self, request_id: String) -> DeltaGenerator {
        // Enable tracking if:
        // 1. Client requested timing in extra_fields, OR
        // 2. query_instance_id annotation is present (needs worker_id tracking for response)
        let enable_tracking = self
            .nvext()
            .map(|nv| {
                nv.extra_fields
                    .as_ref()
                    .is_some_and(|fields| fields.iter().any(|f| f == "timing"))
                    || nv.annotations.as_ref().is_some_and(|annots| {
                        annots.iter().any(|a| a.starts_with("query_instance_id"))
                    })
            })
            .unwrap_or(false);

        let options = DeltaGeneratorOptions {
            enable_usage: self
                .inner
                .stream_options
                .as_ref()
                .map(|opts| opts.include_usage)
                .unwrap_or(false),
            enable_logprobs: self.inner.logprobs.unwrap_or(false)
                || self.inner.top_logprobs.unwrap_or(0) > 0,
            enable_tracking,
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
    /// Determines whether request tracking (timing, KV hit rate) should be enabled.
    pub enable_tracking: bool,

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
    /// Optional request tracker for per-request metrics (shared with PreprocessedRequest).
    tracker: Option<Arc<RequestTracker>>,
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

        let chatcmpl_id = format!("chatcmpl-{request_id}");

        // Create request tracker if tracking is enabled
        let tracker = if options.enable_tracking {
            Some(Arc::new(RequestTracker::new()))
        } else {
            None
        };

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
            tracker,
        }
    }

    /// Returns the request tracker if tracking is enabled, for sharing with PreprocessedRequest.
    pub fn tracker(&self) -> Option<Arc<RequestTracker>> {
        self.tracker.clone()
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

    /// Creates a choice within a chat completion response.
    ///
    /// # Arguments
    /// * `index` - The index of the choice in the completion response.
    /// * `text` - The text content for the response.
    /// * `finish_reason` - The reason why the response finished (e.g., stop, length, etc.).
    /// * `logprobs` - Optional log probabilities of the generated tokens.
    /// * `stop_reason` - Optional stop string or token that triggered the stop.
    ///
    /// # Returns
    /// * An [`dynamo_async_openai::types::CreateChatCompletionStreamResponse`] instance representing the choice.
    #[allow(deprecated)]
    pub fn create_choice(
        &mut self,
        index: u32,
        text: Option<String>,
        finish_reason: Option<dynamo_async_openai::types::FinishReason>,
        logprobs: Option<dynamo_async_openai::types::ChatChoiceLogprobs>,
        stop_reason: Option<dynamo_async_openai::types::StopReason>,
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
            reasoning_content: None,
        };

        let choice = dynamo_async_openai::types::ChatChoiceStream {
            index,
            delta,
            finish_reason,
            stop_reason,
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
            nvext: None, // Will be populated by router layer if needed
        }
    }

    /// Creates a final usage-only chunk for OpenAI compliance.
    /// This should be sent after the last content chunk when stream_options.include_usage is true.
    ///
    /// # Returns
    /// * A [`CreateChatCompletionStreamResponse`] with empty choices and usage stats.
    pub fn create_usage_chunk(&self) -> NvCreateChatCompletionStreamResponse {
        let usage = self.get_usage();

        dynamo_async_openai::types::CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![], // Empty choices for usage-only chunk
            usage: Some(usage),
            service_tier: self.service_tier.clone(),
            nvext: None,
        }
    }

    /// Check if usage tracking is enabled
    pub fn is_usage_enabled(&self) -> bool {
        self.options.enable_usage
    }

    pub fn get_usage(&self) -> dynamo_async_openai::types::CompletionUsage {
        let mut usage = self.usage.clone();
        usage.total_tokens = usage.prompt_tokens.saturating_add(usage.completion_tokens);
        usage
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
        // Aggregate token usage even if usage tracking is disabled for metrics tracking
        // SAFETY: Casting from `usize` to `u32` could lead to precision loss after `u32::MAX`,
        // but this will not be an issue until context lengths exceed 4_294_967_295.
        let token_length: u32 = delta
            .token_ids
            .len()
            .try_into()
            .expect("token_ids length exceeds u32::MAX");

        self.usage.completion_tokens += token_length;

        // If backend provides completion_usage with prompt token details,
        // propagate the entire details struct to usage tracking
        if let Some(prompt_details) = delta
            .completion_usage
            .as_ref()
            .and_then(|usage| usage.prompt_tokens_details.as_ref())
        {
            self.usage.prompt_tokens_details = Some(prompt_details.clone());
        }

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

        // Create the streaming response.
        let index = 0;
        let mut stream_response = self.create_choice(
            index,
            delta.text,
            finish_reason,
            logprobs,
            delta.stop_reason,
        );

        // Record first token time (only succeeds on first call due to OnceLock)
        if let Some(ref tracker) = self.tracker {
            tracker.record_first_token();
        }

        // Get worker_id info from tracker (set by KvPushRouter based on phase)
        let worker_id_info = self.tracker.as_ref().and_then(|t| t.get_worker_info());

        let token_ids = delta
            .disaggregated_params
            .as_ref()
            .and_then(|params| params.get("token_ids"))
            .and_then(|v| serde_json::from_value::<Vec<u32>>(v.clone()).ok());

        // Get timing info if this is the final response (has finish_reason)
        let timing_info: Option<TimingInfo> = if finish_reason.is_some() {
            self.tracker.as_ref().map(|tracker| {
                tracker.record_finish();
                tracker.get_timing_info()
            })
        } else {
            None
        };

        // Inject nvext if we have worker_id, token_ids, or timing
        if worker_id_info.is_some() || token_ids.is_some() || timing_info.is_some() {
            let nvext_response = NvExtResponse {
                worker_id: worker_id_info.clone(),
                timing: timing_info,
                token_ids: token_ids.clone(),
            };

            if let Ok(nvext_json) = serde_json::to_value(&nvext_response) {
                stream_response.nvext = Some(nvext_json);
                if let Some(ref info) = worker_id_info {
                    tracing::debug!(
                        "Injected worker_id into chat completion nvext: prefill={:?}, decode={:?}",
                        info.prefill_worker_id,
                        info.decode_worker_id
                    );
                }
                if let Some(ref tokens) = token_ids {
                    tracing::debug!(
                        "Injected token_ids into chat completion nvext: {} tokens",
                        tokens.len()
                    );
                }
            }
        }

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

    fn get_usage(&self) -> dynamo_async_openai::types::CompletionUsage {
        DeltaGenerator::get_usage(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_async_openai::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
    };

    fn create_test_request() -> NvCreateChatCompletionRequest {
        let messages = vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text("test".to_string()),
                name: None,
            },
        )];

        NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages,
                stream: Some(false),
                stream_options: None,
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            unsupported_fields: Default::default(),
        }
    }

    #[test]
    fn test_enable_usage_for_nonstreaming_enables_usage() {
        // Test that non-streaming requests get usage enabled
        let mut request = create_test_request();
        assert!(request.inner.stream_options.is_none());

        request.enable_usage_for_nonstreaming(false); // false = non-streaming

        assert!(
            request.inner.stream_options.is_some(),
            "Non-streaming request should have stream_options created"
        );
        assert!(
            request.inner.stream_options.unwrap().include_usage,
            "Non-streaming request should have include_usage=true for OpenAI compliance"
        );
    }

    #[test]
    fn test_enable_usage_for_nonstreaming_ignores_streaming() {
        // Test that streaming requests are not modified
        let mut request = create_test_request();
        assert!(request.inner.stream_options.is_none());

        request.enable_usage_for_nonstreaming(true); // true = streaming

        assert!(
            request.inner.stream_options.is_none(),
            "Streaming request should not have stream_options modified"
        );
    }
}
