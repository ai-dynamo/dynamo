// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use dynamo_protocols::types::{ChatCompletionStreamOptions, CompletionUsage};

use crate::protocols::{
    common::timing::RequestTracker, openai::nvext::NvExt,
    openai::nvext::NvExtResponseFieldSelection,
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

/// Initial state for DeltaGenerator
pub(crate) fn initial_state() -> (u32, CompletionUsage, Arc<RequestTracker>) {
    let now_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap() // cannot fail because UNIX_EPOCH is in the past
        .as_secs();
    // Casting from `u64` to `u32` could lead to precision loss after `u32::MAX`,
    // but this will not be an issue until 2106.
    let now: u32 = now_time.try_into().expect("timestamp exceeds u32::MAX");

    let usage = dynamo_protocols::types::CompletionUsage {
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

    (now, usage, tracker)
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
