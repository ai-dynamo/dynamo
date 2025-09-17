// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{NvCreateCompletionRequest, NvCreateCompletionResponse};
use crate::{protocols::common, types::TokenIdType};

impl NvCreateCompletionRequest {
    // put this method on the request
    // inspect the request to extract options
    pub fn response_generator(&self, request_id: String) -> DeltaGenerator {
        let options = DeltaGeneratorOptions {
            enable_usage: self
                .inner
                .stream_options
                .as_ref()
                .map(|opts| opts.include_usage)
                .unwrap_or(false),
            enable_logprobs: self.inner.logprobs.unwrap_or(0) > 0,
        };

        DeltaGenerator::new(self.inner.model.clone(), options, request_id)
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeltaGeneratorOptions {
    pub enable_usage: bool,
    pub enable_logprobs: bool,
}

#[derive(Debug, Clone)]
pub struct DeltaGenerator {
    id: String,
    object: String,
    created: u32,
    model: String,
    system_fingerprint: Option<String>,
    usage: dynamo_async_openai::types::CompletionUsage,
    options: DeltaGeneratorOptions,
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

        Self {
            id: completion_id,
            object: "text_completion".to_string(),
            created: now,
            model,
            system_fingerprint: None,
            usage,
            options,
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
        let response = self.create_choice(index, delta.text.clone(), finish_reason, logprobs);
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
