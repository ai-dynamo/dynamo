// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use dynamo_protocols::types::{ChatCompletionStreamOptions, CompletionUsage};

use crate::protocols::common::{
    extensions::{NvExt, NvExtResponseFieldSelection},
    llm_backend::BackendOutput,
    timing::RequestTracker,
};

/// Configuration options for the [`DeltaGenerator`], controlling response behavior.
#[derive(Debug, Clone, Default)]
pub struct DeltaGeneratorOptions {
    /// Determines whether token usage statistics should be included in the response.
    pub enable_usage: bool,
    /// Determines whether continuous usage statistics should be included in the response.
    pub continuous_usage_stats: bool,
    /// Determines whether log probabilities should be included in the response.
    pub enable_logprobs: bool,
    /// When true, logprob token fields use "token_id:<id>" format instead of decoded text.
    pub return_tokens_as_token_ids: bool,
    /// Determines which nvext response fields may be emitted for this request.
    pub response_fields: NvExtResponseFieldSelection,
}

impl DeltaGeneratorOptions {
    pub fn new(
        stream_options: Option<&ChatCompletionStreamOptions>,
        return_tokens_as_token_ids: Option<bool>,
        enable_logprobs: bool,
        nvext: Option<&NvExt>,
    ) -> Self {
        let response_fields = NvExtResponseFieldSelection::from_nvext(nvext);
        DeltaGeneratorOptions {
            enable_usage: stream_options.is_some_and(|opts| opts.include_usage),
            continuous_usage_stats: stream_options.is_some_and(|opts| opts.continuous_usage_stats),
            enable_logprobs,
            response_fields,
            return_tokens_as_token_ids: return_tokens_as_token_ids.unwrap_or(false),
        }
    }
}

/// State and lifecycle behavior shared by the chat and text completion delta generators.
pub(crate) struct DeltaGeneratorState {
    id: String,
    object: String,
    created: u32,
    model: String,
    system_fingerprint: Option<String>,
    usage: CompletionUsage,
    /// Trailing generation-stub tokens excluded from reported prompt usage.
    /// `usage.prompt_tokens` keeps the physical prompt length (also what
    /// `get_isl` reports for metrics); the subtraction happens in `get_usage`.
    pending_prompt_tokens: u32,
    options: DeltaGeneratorOptions,
    tracker: Arc<RequestTracker>,
}

impl DeltaGeneratorState {
    pub(crate) fn new(
        id: String,
        object: String,
        model: String,
        options: DeltaGeneratorOptions,
    ) -> Self {
        let now_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap() // cannot fail because UNIX_EPOCH is in the past
            .as_secs();
        // Casting from `u64` to `u32` could lead to precision loss after `u32::MAX`,
        // but this will not be an issue until 2106.
        let created: u32 = now_time.try_into().expect("timestamp exceeds u32::MAX");

        let usage = CompletionUsage {
            completion_tokens: 0,
            prompt_tokens: 0,
            total_tokens: 0,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        };

        // Always create request tracker for per-worker metrics (TTFT, ITL per worker_id).
        // `response_fields` only controls which nvext fields are returned to the client;
        // the tracker still records timing/ITL internally for metrics.
        let tracker = Arc::new(RequestTracker::new());

        Self {
            id,
            object,
            created,
            model,
            system_fingerprint: None,
            usage,
            pending_prompt_tokens: 0,
            options,
            tracker,
        }
    }

    pub(crate) fn id(&self) -> &str {
        &self.id
    }

    pub(crate) fn object(&self) -> &str {
        &self.object
    }

    pub(crate) fn created(&self) -> u32 {
        self.created
    }

    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    pub(crate) fn system_fingerprint(&self) -> Option<&String> {
        self.system_fingerprint.as_ref()
    }

    pub(crate) fn options(&self) -> &DeltaGeneratorOptions {
        &self.options
    }

    pub(crate) fn tracker(&self) -> Arc<RequestTracker> {
        self.tracker.clone()
    }

    pub(crate) fn tracker_ref(&self) -> &Arc<RequestTracker> {
        &self.tracker
    }

    pub(crate) fn update_isl(&mut self, isl: u32) {
        self.usage.prompt_tokens = isl;
    }

    /// Record how many trailing prompt tokens are the renderer's generation
    /// stub so reported usage follows the model's reference API, which feeds
    /// those tokens to the model but does not bill them as prompt tokens.
    pub(crate) fn set_pending_prompt_tokens(&mut self, pending_prompt_tokens: u32) {
        self.pending_prompt_tokens = pending_prompt_tokens;
    }

    pub(crate) fn update_usage_from_backend_output(&mut self, output: &BackendOutput) {
        // Aggregate token usage even if usage tracking is disabled for metrics tracking.
        // SAFETY: Casting from `usize` to `u32` could lead to precision loss after `u32::MAX`,
        // but this will not be an issue until context lengths exceed 4_294_967_295.
        let token_length: u32 = output
            .token_ids
            .len()
            .try_into()
            .expect("token_ids length exceeds u32::MAX");

        self.usage.completion_tokens += token_length;

        // If the backend provides completion_usage, use it to update usage stats.
        // This is critical for prompt embeddings where prompt_tokens comes from
        // the embedding sequence length computed by the worker.
        if let Some(completion_usage) = output.completion_usage.as_ref() {
            self.usage.prompt_tokens = completion_usage.prompt_tokens;

            if let Some(prompt_details) = completion_usage.prompt_tokens_details.as_ref() {
                self.usage.prompt_tokens_details = Some(prompt_details.clone());
            }

            if let Some(completion_details) = completion_usage.completion_tokens_details.as_ref() {
                self.usage.completion_tokens_details = Some(completion_details.clone());
            }
        }
    }

    pub(crate) fn get_isl(&self) -> u32 {
        self.usage.prompt_tokens
    }

    pub(crate) fn get_usage(&self) -> CompletionUsage {
        let mut usage = self.usage.clone();
        if self.pending_prompt_tokens > 0 {
            // Applies equally to the frontend's own count and to a
            // backend-reported prompt_tokens: both measure the physical prompt.
            usage.prompt_tokens = usage
                .prompt_tokens
                .saturating_sub(self.pending_prompt_tokens);
            // The engine's prefix-cache hit covers the physical prompt, so a
            // fully cached prompt would otherwise report more cached than
            // prompt tokens.
            if let Some(cached) = usage
                .prompt_tokens_details
                .as_mut()
                .and_then(|details| details.cached_tokens.as_mut())
            {
                *cached = (*cached).min(usage.prompt_tokens);
            }
        }
        usage.total_tokens = usage.prompt_tokens.saturating_add(usage.completion_tokens);
        usage
    }

    pub(crate) fn is_usage_enabled(&self) -> bool {
        self.options.enable_usage
    }

    pub(crate) fn is_continuous_usage_enabled(&self) -> bool {
        self.options.continuous_usage_stats
    }
}

/// Enables usage tracking for non-streaming requests to comply with OpenAI API specification.
///
/// According to OpenAI API spec, non-streaming chat completion responses (stream=false)
/// must always include usage statistics. This method ensures `stream_options.include_usage`
/// is set to `true` for non-streaming requests.
pub(crate) fn enable_usage_for_nonstreaming(
    stream_options: &mut Option<ChatCompletionStreamOptions>,
    original_stream_flag: bool,
) {
    if original_stream_flag {
        return;
    }
    // For non-streaming requests (stream=false), enable usage
    stream_options
        .get_or_insert_with(|| ChatCompletionStreamOptions {
            include_usage: true,
            continuous_usage_stats: false,
        })
        .include_usage = true;
}

/// Enables usage statistics regardless of the request's `include_usage` value.
pub(crate) fn force_include_usage(stream_options: &mut Option<ChatCompletionStreamOptions>) {
    stream_options
        .get_or_insert_with(|| ChatCompletionStreamOptions {
            include_usage: true,
            continuous_usage_stats: false,
        })
        .include_usage = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_include_usage_inserts_missing_options() {
        let mut options = None;

        force_include_usage(&mut options);

        let options = options.expect("stream options should be inserted");
        assert!(options.include_usage);
        assert!(!options.continuous_usage_stats);
    }

    #[test]
    fn force_include_usage_overrides_false_and_preserves_siblings() {
        let mut options = Some(ChatCompletionStreamOptions {
            include_usage: false,
            continuous_usage_stats: true,
        });

        force_include_usage(&mut options);

        let options = options.expect("stream options should remain present");
        assert!(options.include_usage);
        assert!(options.continuous_usage_stats);
    }

    fn state_with_pending(isl: u32, pending: u32) -> DeltaGeneratorState {
        let mut state = DeltaGeneratorState::new(
            "id".to_string(),
            "chat.completion.chunk".to_string(),
            "model".to_string(),
            DeltaGeneratorOptions::default(),
        );
        state.update_isl(isl);
        state.set_pending_prompt_tokens(pending);
        state
    }

    #[test]
    fn pending_prompt_tokens_are_excluded_from_reported_usage_only() {
        let state = state_with_pending(39, 3);
        let usage = state.get_usage();
        assert_eq!(usage.prompt_tokens, 36);
        assert_eq!(usage.total_tokens, 36);
        assert_eq!(
            state.get_isl(),
            39,
            "metrics keep the physical prompt length"
        );
    }

    #[test]
    fn zero_pending_prompt_tokens_leave_usage_unchanged() {
        let usage = state_with_pending(39, 0).get_usage();
        assert_eq!(usage.prompt_tokens, 39);
    }

    #[test]
    fn backend_reported_prompt_tokens_are_adjusted_and_cached_tokens_clamped() {
        use crate::protocols::common::llm_backend::BackendOutput;
        use dynamo_protocols::types::PromptTokensDetails;

        let mut state = state_with_pending(39, 3);
        let output = BackendOutput {
            token_ids: vec![7, 8],
            tokens: vec![Some("a".to_string()), Some("b".to_string())],
            text: Some("ab".to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: None,
            stop_reason: None,
            index: Some(0),
            completion_usage: Some(CompletionUsage {
                prompt_tokens: 39,
                completion_tokens: 2,
                total_tokens: 41,
                prompt_tokens_details: Some(PromptTokensDetails {
                    // Engine reports a prefix-cache hit over the whole physical prompt.
                    cached_tokens: Some(38),
                    ..Default::default()
                }),
                completion_tokens_details: None,
            }),
            disaggregated_params: None,
            worker_trace_link: None,
            engine_data: None,
            encoder_result: None,
            routing_data: None,
            jailed_text: None,
        };
        state.update_usage_from_backend_output(&output);

        let usage = state.get_usage();
        assert_eq!(usage.prompt_tokens, 36);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(usage.total_tokens, 38);
        assert_eq!(
            usage
                .prompt_tokens_details
                .and_then(|details| details.cached_tokens),
            Some(36)
        );
    }
}
