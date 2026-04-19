// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse};
use crate::{
    local_model::runtime_config::ModelRuntimeConfig,
    protocols::{
        common::{self, timing::RequestTracker},
        openai::{
            convert_backend_top_logprobs,
            nvext::{NvExtProvider, NvExtResponseFieldSelection},
            token_to_utf8_bytes,
        },
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
                    Some(dynamo_protocols::types::ChatCompletionStreamOptions {
                        include_usage: true,
                        continuous_usage_stats: false,
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
        let response_fields = NvExtResponseFieldSelection::from_nvext(self.nvext());

        let options = DeltaGeneratorOptions {
            enable_usage: self
                .inner
                .stream_options
                .as_ref()
                .map(|opts| opts.include_usage)
                .unwrap_or(false),
            continuous_usage_stats: self
                .inner
                .stream_options
                .as_ref()
                .map(|opts| opts.continuous_usage_stats)
                .unwrap_or(false),
            enable_logprobs: self.inner.logprobs.unwrap_or(false)
                || self.inner.top_logprobs.unwrap_or(0) > 0,
            response_fields,
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
    /// Determines whether continuous usage statistics should be included in the response.
    pub continuous_usage_stats: bool,
    /// Determines whether log probabilities should be included in the response.
    pub enable_logprobs: bool,
    /// Determines which nvext response fields may be emitted for this request.
    pub response_fields: NvExtResponseFieldSelection,

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
    service_tier: Option<dynamo_protocols::types::ServiceTierResponse>,
    /// Tracks token usage for the completion request.
    usage: dynamo_protocols::types::CompletionUsage,
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

        let usage = dynamo_protocols::types::CompletionUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        };

        let chatcmpl_id = format!("chatcmpl-{request_id}");

        // Always create request tracker for per-worker metrics (TTFT, ITL per worker_id).
        // `response_fields` only controls which nvext fields are returned to the client;
        // the tracker still records timing/ITL internally for metrics.
        let tracker = Some(Arc::new(RequestTracker::new()));

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
    ) -> Option<dynamo_protocols::types::ChatChoiceLogprobs> {
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
                    let converted = convert_backend_top_logprobs(&top_lps, t, *tid, lp);
                    dynamo_protocols::types::ChatCompletionTokenLogprob {
                        token: t.clone(),
                        logprob: lp,
                        bytes: token_to_utf8_bytes(t),
                        top_logprobs: converted,
                    }
                })
                .collect()
        });

        Some(dynamo_protocols::types::ChatChoiceLogprobs {
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
    /// * An [`dynamo_protocols::types::CreateChatCompletionStreamResponse`] instance representing the choice.
    #[allow(deprecated)]
    pub fn create_choice(
        &mut self,
        index: u32,
        text: Option<String>,
        finish_reason: Option<dynamo_protocols::types::FinishReason>,
        logprobs: Option<dynamo_protocols::types::ChatChoiceLogprobs>,
        stop_reason: Option<dynamo_protocols::types::StopReason>,
    ) -> NvCreateChatCompletionStreamResponse {
        let delta = dynamo_protocols::types::ChatCompletionStreamResponseDelta {
            content: text.map(dynamo_protocols::types::ChatCompletionMessageContent::Text),
            function_call: None,
            tool_calls: None,
            role: if self.msg_counter == 0 {
                Some(dynamo_protocols::types::Role::Assistant)
            } else {
                None
            },
            refusal: None,
            reasoning_content: None,
        };

        let choice = dynamo_protocols::types::ChatChoiceStream {
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
        NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: self.id.clone(),
                object: self.object.clone(),
                created: self.created,
                model: self.model.clone(),
                system_fingerprint: self.system_fingerprint.clone(),
                choices,
                usage: if self.options.enable_usage && self.options.continuous_usage_stats {
                    Some(self.get_usage())
                } else {
                    None
                },
                service_tier: self.service_tier.clone(),
            },
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

        NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: self.id.clone(),
                object: self.object.clone(),
                created: self.created,
                model: self.model.clone(),
                system_fingerprint: self.system_fingerprint.clone(),
                choices: vec![], // Empty choices for usage-only chunk
                usage: Some(usage),
                service_tier: self.service_tier.clone(),
            },
            nvext: None,
        }
    }

    /// Check if usage tracking is enabled
    pub fn is_usage_enabled(&self) -> bool {
        self.options.enable_usage
    }

    /// Check if continuous usage tracking is enabled
    pub fn is_continuous_usage_enabled(&self) -> bool {
        self.options.continuous_usage_stats
    }

    pub fn get_usage(&self) -> dynamo_protocols::types::CompletionUsage {
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

        // If backend provides completion_usage, use it to update usage stats
        // This is critical for prompt embeddings where prompt_tokens comes from
        // the embedding sequence length computed by the worker
        if let Some(completion_usage) = delta.completion_usage.as_ref() {
            // Update prompt_tokens from worker if provided (e.g., for embeddings)
            self.usage.prompt_tokens = completion_usage.prompt_tokens;

            // Propagate prompt token details if provided
            if let Some(prompt_details) = completion_usage.prompt_tokens_details.as_ref() {
                self.usage.prompt_tokens_details = Some(prompt_details.clone());
            }
        }

        let logprobs = self.create_logprobs(
            delta.tokens,
            &delta.token_ids,
            delta.log_probs,
            delta.top_logprobs,
        );

        // Map backend finish reasons to OpenAI's finish reasons.
        let finish_reason = match delta.finish_reason {
            Some(common::FinishReason::EoS) => Some(dynamo_protocols::types::FinishReason::Stop),
            Some(common::FinishReason::Stop) => Some(dynamo_protocols::types::FinishReason::Stop),
            Some(common::FinishReason::Length) => {
                Some(dynamo_protocols::types::FinishReason::Length)
            }
            Some(common::FinishReason::Cancelled) => {
                Some(dynamo_protocols::types::FinishReason::Stop)
            }
            Some(common::FinishReason::ContentFilter) => {
                Some(dynamo_protocols::types::FinishReason::ContentFilter)
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

        // Record finish for timing/ITL accounting even when timing is not returned to the client.
        // Kept at call site because it's a side effect on the tracker — not a gating decision.
        if finish_reason.is_some()
            && let Some(ref tracker) = self.tracker
        {
            tracker.record_finish();
        }

        // Build the nvext response payload via the shared gating helper on
        // `NvExtResponseFieldSelection` (see `nvext.rs`). Both chat and
        // completions delta generators go through the same helper so the gating
        // rules stay in one place.
        if let Some(nvext_response) = self.options.response_fields.build_response_nvext(
            self.tracker.as_ref(),
            delta.disaggregated_params.as_ref(),
            finish_reason.is_some(),
        ) && let Ok(nvext_json) = serde_json::to_value(&nvext_response)
        {
            stream_response.nvext = Some(nvext_json);
            if let Some(ref info) = nvext_response.worker_id {
                tracing::debug!(
                    "Injected worker_id into chat completion nvext: prefill={:?}, decode={:?}",
                    info.prefill_worker_id,
                    info.decode_worker_id
                );
            }
            if let Some(ref tokens) = nvext_response.token_ids {
                tracing::debug!(
                    "Injected token_ids into chat completion nvext: {} tokens",
                    tokens.len()
                );
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

    fn is_continuous_usage_enabled(&self) -> bool {
        DeltaGenerator::is_continuous_usage_enabled(self)
    }

    fn get_usage(&self) -> dynamo_protocols::types::CompletionUsage {
        DeltaGenerator::get_usage(self)
    }

    fn tracker(&self) -> Option<std::sync::Arc<crate::protocols::common::timing::RequestTracker>> {
        self.tracker.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::{self, llm_backend::BackendOutput, timing::WORKER_TYPE_PREFILL};
    use crate::protocols::openai::DeltaGeneratorExt;
    use dynamo_protocols::types::{
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
        assert!(
            !request.inner.stream_options.unwrap().continuous_usage_stats,
            "Non-streaming request should have continuous_usage_stats=false for OpenAI compliance"
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

    fn make_request_with_nvext(
        nvext: crate::protocols::openai::nvext::NvExt,
    ) -> NvCreateChatCompletionRequest {
        let mut request = create_test_request();
        request.nvext = Some(nvext);
        request
    }

    fn final_backend_output() -> BackendOutput {
        BackendOutput {
            token_ids: vec![1],
            tokens: vec![Some("hello".to_string())],
            text: Some("hello".to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(common::FinishReason::Stop),
            stop_reason: None,
            index: Some(0),
            completion_usage: None,
            disaggregated_params: Some(serde_json::json!({
                "token_ids": [11, 22, 33],
                "routed_experts": {"layer_0": [1, 3]}
            })),
        }
    }

    #[test]
    fn test_plain_request_without_extra_fields_omits_nvext() {
        let request = create_test_request();
        let mut generator = request.response_generator("req-no-nvext".to_string());
        let tracker = generator.tracker().expect("tracker");
        tracker.record_worker(42, Some(0), WORKER_TYPE_PREFILL);

        let response = generator
            .choice_from_postprocessor(final_backend_output())
            .expect("choice generation");

        assert!(response.nvext.is_none());
    }

    #[test]
    fn test_timing_extra_field_emits_timing_on_final_chunk() {
        use crate::protocols::openai::nvext::NvExt;
        let nvext = NvExt::builder()
            .extra_fields(vec!["timing".to_string()])
            .build()
            .unwrap();
        let mut generator =
            make_request_with_nvext(nvext).response_generator("req-timing".to_string());

        let response = generator
            .choice_from_postprocessor(final_backend_output())
            .expect("choice generation");

        let nvext_json = response.nvext.expect("nvext present for timing request");
        assert!(
            nvext_json.get("timing").is_some(),
            "timing should be emitted when extra_fields=[\"timing\"]"
        );
        assert!(nvext_json.get("worker_id").is_none());
        assert!(nvext_json.get("token_ids").is_none());
        assert!(nvext_json.get("routed_experts").is_none());
    }

    #[test]
    fn test_query_instance_id_emits_worker_id_and_token_ids() {
        use crate::protocols::openai::nvext::NvExt;
        let nvext = NvExt::builder()
            .annotations(vec!["query_instance_id:abc".to_string()])
            .build()
            .unwrap();
        let mut generator =
            make_request_with_nvext(nvext).response_generator("req-qid".to_string());
        let tracker = generator.tracker().expect("tracker");
        tracker.record_worker(42, Some(0), WORKER_TYPE_PREFILL);

        let response = generator
            .choice_from_postprocessor(final_backend_output())
            .expect("choice generation");

        let nvext_json = response
            .nvext
            .expect("nvext present for query_instance_id flow");
        assert!(nvext_json.get("worker_id").is_some());
        assert_eq!(
            nvext_json.get("token_ids"),
            Some(&serde_json::json!([11, 22, 33]))
        );
        // timing is NOT auto-enabled for query_instance_id — it is gated by `extra_fields: ["timing"]`.
        assert!(nvext_json.get("timing").is_none());
        assert!(nvext_json.get("routed_experts").is_none());
    }

    #[test]
    fn test_routed_experts_extra_field_emits_routed_experts() {
        use crate::protocols::openai::nvext::NvExt;
        let nvext = NvExt::builder()
            .extra_fields(vec!["routed_experts".to_string()])
            .build()
            .unwrap();
        let mut generator =
            make_request_with_nvext(nvext).response_generator("req-experts".to_string());

        let response = generator
            .choice_from_postprocessor(final_backend_output())
            .expect("choice generation");

        let nvext_json = response
            .nvext
            .expect("nvext present for routed_experts request");
        assert_eq!(
            nvext_json.get("routed_experts"),
            Some(&serde_json::json!({"layer_0": [1, 3]}))
        );
        assert!(nvext_json.get("worker_id").is_none());
        assert!(nvext_json.get("timing").is_none());
        assert!(nvext_json.get("token_ids").is_none());
    }
}
