// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{NvCreateCompletionRequest, NvCreateCompletionResponse};
use crate::{
    protocols::{
        common::{self, timing::RequestTimingTracker},
        openai::nvext::{NvExtProvider, NvExtResponse, TimingInfo, WorkerIdInfo},
    },
    types::TokenIdType,
};

impl NvCreateCompletionRequest {
    /// Enables usage tracking for non-streaming requests to comply with OpenAI API specification.
    ///
    /// According to OpenAI API spec, non-streaming completion responses (stream=false)
    /// must always include usage statistics. This method ensures `stream_options.include_usage`
    /// is set to `true` for non-streaming requests.
    ///
    /// Reference: https://platform.openai.com/docs/api-reference/completions/create
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

    // put this method on the request
    // inspect the request to extract options
    pub fn response_generator(&self, request_id: String) -> DeltaGenerator {
        // Check if client requested timing in extra_fields
        let enable_timing = self
            .nvext()
            .and_then(|nv| nv.extra_fields.as_ref())
            .is_some_and(|fields| fields.iter().any(|f| f == "timing"));

        let options = DeltaGeneratorOptions {
            enable_usage: self
                .inner
                .stream_options
                .as_ref()
                .map(|opts| opts.include_usage)
                .unwrap_or(false),
            enable_logprobs: self.inner.logprobs.unwrap_or(0) > 0,
            enable_timing,
        };

        DeltaGenerator::new(self.inner.model.clone(), options, request_id)
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeltaGeneratorOptions {
    pub enable_usage: bool,
    pub enable_logprobs: bool,
    pub enable_timing: bool,
}

pub struct DeltaGenerator {
    id: String,
    object: String,
    created: u32,
    model: String,
    system_fingerprint: Option<String>,
    usage: dynamo_async_openai::types::CompletionUsage,
    options: DeltaGeneratorOptions,
    timing_tracker: Option<RequestTimingTracker>,
}

impl DeltaGenerator {
    pub fn new(model: String, options: DeltaGeneratorOptions, request_id: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // SAFETY: Casting from `u64` to `u32` could lead to precision loss after `u32::MAX`,
        // but this will not be an issue until 2106.
        let now: u32 = now.try_into().expect("timestamp exceeds u32::MAX");

        // Previously, our home-rolled CompletionUsage impl'd Default
        // PR !387 - https://github.com/64bit/async-openai/pull/387
        let usage = dynamo_async_openai::types::CompletionUsage {
            completion_tokens: 0,
            prompt_tokens: 0,
            total_tokens: 0,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        };

        let completion_id = format!("cmpl-{request_id}");

        // Create timing tracker if timing is enabled
        let timing_tracker = if options.enable_timing {
            Some(RequestTimingTracker::new())
        } else {
            None
        };

        Self {
            id: completion_id,
            object: "text_completion".to_string(),
            created: now,
            model,
            system_fingerprint: None,
            usage,
            options,
            timing_tracker,
        }
    }

    pub fn update_isl(&mut self, isl: u32) {
        self.usage.prompt_tokens = isl;
    }

    pub fn create_logprobs(
        &self,
        tokens: Vec<common::llm_backend::TokenType>,
        token_ids: Vec<TokenIdType>,
        logprobs: Option<common::llm_backend::LogProbs>,
        top_logprobs: Option<common::llm_backend::TopLogprobs>,
    ) -> Option<dynamo_async_openai::types::Logprobs> {
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

        let top_lps = top_logprobs.map_or(vec![], |top_logprobs| {
            toks.iter()
                .zip(tok_lps.iter())
                .zip(top_logprobs.iter())
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
                            logprob: *lp,
                            bytes: None,
                        });
                    }
                    serde_json::to_value(converted_top_lps).unwrap()
                })
                .collect()
        });

        Some(dynamo_async_openai::types::Logprobs {
            tokens: toks.iter().map(|(t, _)| t.clone()).collect(),
            token_logprobs: tok_lps.into_iter().map(Some).collect(),
            text_offset: vec![],
            top_logprobs: top_lps,
        })
    }

    pub fn create_choice(
        &self,
        index: u32,
        text: Option<String>,
        finish_reason: Option<dynamo_async_openai::types::CompletionFinishReason>,
        logprobs: Option<dynamo_async_openai::types::Logprobs>,
    ) -> NvCreateCompletionResponse {
        // todo - update for tool calling

        // According to OpenAI spec: when stream_options.include_usage is true,
        // all intermediate chunks should have usage: null
        // The final usage chunk will be sent separately with empty choices
        let inner = dynamo_async_openai::types::CreateCompletionResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![dynamo_async_openai::types::Choice {
                text: text.unwrap_or_default(),
                index,
                finish_reason,
                logprobs,
            }],
            usage: None, // Always None for chunks with content/choices
            nvext: None, // Will be populated by router layer if needed
        };

        NvCreateCompletionResponse { inner }
    }

    /// Creates a final usage-only chunk for OpenAI compliance.
    /// This should be sent after the last content chunk when stream_options.include_usage is true.
    ///
    /// # Returns
    /// * A [`NvCreateCompletionResponse`] with empty choices and usage stats.
    pub fn create_usage_chunk(&self) -> NvCreateCompletionResponse {
        let mut usage = self.usage.clone();
        usage.total_tokens = usage.prompt_tokens.saturating_add(usage.completion_tokens);

        let inner = dynamo_async_openai::types::CreateCompletionResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![], // Empty choices for usage-only chunk
            usage: Some(usage),
            nvext: None, // Will be populated by router layer if needed
        };

        NvCreateCompletionResponse { inner }
    }

    /// Check if usage tracking is enabled
    pub fn is_usage_enabled(&self) -> bool {
        self.options.enable_usage
    }
}

impl crate::protocols::openai::DeltaGeneratorExt<NvCreateCompletionResponse> for DeltaGenerator {
    fn choice_from_postprocessor(
        &mut self,
        delta: common::llm_backend::BackendOutput,
    ) -> anyhow::Result<NvCreateCompletionResponse> {
        // aggregate usage
        if self.options.enable_usage {
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
        }

        let logprobs = self.create_logprobs(
            delta.tokens,
            delta.token_ids,
            delta.log_probs,
            delta.top_logprobs,
        );

        let finish_reason = delta.finish_reason.map(Into::into);

        // create choice
        let index = delta.index.unwrap_or(0);
        let mut response = self.create_choice(index, delta.text.clone(), finish_reason, logprobs);

        // Record first token time (only succeeds on first call due to OnceLock)
        if let Some(ref tracker) = self.timing_tracker {
            tracker.record_first_token();
        }

        // Extract worker_id from disaggregated_params
        let worker_id_info = delta
            .disaggregated_params
            .as_ref()
            .and_then(|params| params.get("worker_id"))
            .and_then(|v| serde_json::from_value::<WorkerIdInfo>(v.clone()).ok());

        // Get timing info if this is the final response (has finish_reason)
        let timing_info: Option<TimingInfo> = if finish_reason.is_some() {
            self.timing_tracker.as_ref().map(|tracker| {
                tracker.record_finish();
                tracker.get_timing_info()
            })
        } else {
            None
        };

        // Inject nvext if we have worker_id or timing
        if worker_id_info.is_some() || timing_info.is_some() {
            let nvext_response = NvExtResponse {
                worker_id: worker_id_info.clone(),
                timing: timing_info,
            };

            if let Ok(nvext_json) = serde_json::to_value(&nvext_response) {
                response.inner.nvext = Some(nvext_json);
                if let Some(ref info) = worker_id_info {
                    tracing::debug!(
                        "Injected worker_id into completions nvext: prefill={:?}, decode={:?}",
                        info.prefill_worker_id,
                        info.decode_worker_id
                    );
                }
            }
        }

        Ok(response)
    }

    fn get_isl(&self) -> Option<u32> {
        Some(self.usage.prompt_tokens)
    }

    fn create_usage_chunk(&self) -> NvCreateCompletionResponse {
        DeltaGenerator::create_usage_chunk(self)
    }

    fn is_usage_enabled(&self) -> bool {
        DeltaGenerator::is_usage_enabled(self)
    }
}
