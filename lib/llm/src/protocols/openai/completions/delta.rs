// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::{NvCreateCompletionRequest, NvCreateCompletionResponse};
use crate::{
    protocols::{
        common::{self, timing::RequestTracker},
        openai::{
            convert_backend_top_logprobs,
            nvext::{NvExtProvider, NvExtResponseFieldSelection},
        },
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

    // put this method on the request
    // inspect the request to extract options
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
            enable_logprobs: self.inner.logprobs.unwrap_or(0) > 0,
            response_fields,
        };

        DeltaGenerator::new(self.inner.model.clone(), options, request_id)
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeltaGeneratorOptions {
    pub enable_usage: bool,
    pub continuous_usage_stats: bool,
    pub enable_logprobs: bool,
    pub response_fields: NvExtResponseFieldSelection,
}

pub struct DeltaGenerator {
    id: String,
    object: String,
    created: u32,
    model: String,
    system_fingerprint: Option<String>,
    usage: dynamo_protocols::types::CompletionUsage,
    options: DeltaGeneratorOptions,
    tracker: Option<Arc<RequestTracker>>,
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
        let usage = dynamo_protocols::types::CompletionUsage {
            completion_tokens: 0,
            prompt_tokens: 0,
            total_tokens: 0,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        };

        let completion_id = format!("cmpl-{request_id}");

        // Always create request tracker for per-worker metrics (TTFT, ITL per worker_id).
        // `response_fields` only controls which nvext fields are returned to the client;
        // the tracker still records timing/ITL internally for metrics.
        let tracker = Some(Arc::new(RequestTracker::new()));

        Self {
            id: completion_id,
            object: "text_completion".to_string(),
            created: now,
            model,
            system_fingerprint: None,
            usage,
            options,
            tracker,
        }
    }

    /// Returns the request tracker if tracking is enabled, for sharing with PreprocessedRequest.
    pub fn tracker(&self) -> Option<Arc<RequestTracker>> {
        self.tracker.clone()
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
    ) -> Option<dynamo_protocols::types::Logprobs> {
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
                    let converted = convert_backend_top_logprobs(top_lps, t, *tid, *lp);
                    serde_json::to_value(converted).unwrap()
                })
                .collect()
        });

        Some(dynamo_protocols::types::Logprobs {
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
        finish_reason: Option<dynamo_protocols::types::CompletionFinishReason>,
        logprobs: Option<dynamo_protocols::types::Logprobs>,
    ) -> NvCreateCompletionResponse {
        // todo - update for tool calling

        // According to OpenAI spec: when stream_options.include_usage is true,
        // all intermediate chunks should have usage: null
        // The final usage chunk will be sent separately with empty choices
        let inner = dynamo_protocols::types::CreateCompletionResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![dynamo_protocols::types::Choice {
                text: text.unwrap_or_default(),
                index,
                finish_reason,
                logprobs,
            }],
            usage: if self.options.enable_usage && self.options.continuous_usage_stats {
                Some(self.get_usage())
            } else {
                None
            },
        };

        NvCreateCompletionResponse { inner, nvext: None }
    }

    /// Creates a final usage-only chunk for OpenAI compliance.
    /// This should be sent after the last content chunk when stream_options.include_usage is true.
    ///
    /// # Returns
    /// * A [`NvCreateCompletionResponse`] with empty choices and usage stats.
    pub fn create_usage_chunk(&self) -> NvCreateCompletionResponse {
        let usage = self.get_usage();

        let inner = dynamo_protocols::types::CreateCompletionResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![], // Empty choices for usage-only chunk
            usage: Some(usage),
        };

        NvCreateCompletionResponse { inner, nvext: None }
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

impl crate::protocols::openai::DeltaGeneratorExt<NvCreateCompletionResponse> for DeltaGenerator {
    fn choice_from_postprocessor(
        &mut self,
        delta: common::llm_backend::BackendOutput,
    ) -> anyhow::Result<NvCreateCompletionResponse> {
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

            // Propagate completion token details if provided
            if let Some(completion_details) = completion_usage.completion_tokens_details.as_ref() {
                self.usage.completion_tokens_details = Some(completion_details.clone());
            }

            // Propagate prompt token details if provided
            if let Some(prompt_details) = completion_usage.prompt_tokens_details.as_ref() {
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
            delta.engine_data,
        ) && let Ok(nvext_json) = serde_json::to_value(&nvext_response)
        {
            response.nvext = Some(nvext_json);
            if let Some(ref info) = nvext_response.worker_id {
                tracing::debug!(
                    "Injected worker_id into completions nvext: prefill={:?}, decode={:?}",
                    info.prefill_worker_id,
                    info.decode_worker_id
                );
            }
            if let Some(ref tokens) = nvext_response.token_ids {
                tracing::debug!(
                    "Injected token_ids into completions nvext: {} tokens",
                    tokens.len()
                );
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
    use dynamo_protocols::types::{CreateCompletionRequestArgs, Prompt};

    fn create_test_request() -> NvCreateCompletionRequest {
        let inner = CreateCompletionRequestArgs::default()
            .model("test-model")
            .prompt(Prompt::String("test".to_string()))
            .build()
            .expect("completion request");

        NvCreateCompletionRequest {
            inner,
            common: Default::default(),
            nvext: None,
            metadata: None,
            unsupported_fields: Default::default(),
        }
    }

    fn make_request_with_nvext(
        nvext: crate::protocols::openai::nvext::NvExt,
    ) -> NvCreateCompletionRequest {
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
            engine_data: None,
        }
    }

    fn create_test_request_with_extra_fields(fields: Vec<String>) -> NvCreateCompletionRequest {
        let inner = CreateCompletionRequestArgs::default()
            .model("test-model")
            .prompt(Prompt::String("test".to_string()))
            .build()
            .expect("completion request");

        NvCreateCompletionRequest {
            inner,
            common: Default::default(),
            nvext: Some(
                crate::protocols::openai::nvext::NvExt::builder()
                    .extra_fields(fields)
                    .build()
                    .unwrap(),
            ),
            metadata: None,
            unsupported_fields: Default::default(),
        }
    }

    fn make_backend_output_with_engine_data() -> BackendOutput {
        BackendOutput {
            token_ids: vec![42],
            tokens: vec![Some("hello".to_string())],
            text: Some("hello".to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(common::FinishReason::Stop),
            stop_reason: None,
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
            engine_data: Some(serde_json::json!({
                "kv_transfer_time_ms": 12.3,
                "disaggregated_kv_transfer_time_ms": 8.1,
                "prefill_compute_time_ms": 45.6
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

    #[test]
    fn test_engine_data_included_when_requested_via_extra_fields() {
        let request = create_test_request_with_extra_fields(vec!["engine_data".to_string()]);
        let mut generator = request.response_generator("req-engine-1".to_string());

        let backend_output = make_backend_output_with_engine_data();
        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("should produce a response");

        let nvext = response.nvext.expect("nvext should be present");
        let engine_data = nvext
            .get("engine_data")
            .expect("engine_data should be present");
        assert_eq!(engine_data["kv_transfer_time_ms"], 12.3);
        assert_eq!(engine_data["prefill_compute_time_ms"], 45.6);
    }

    #[test]
    fn test_engine_data_excluded_when_not_requested() {
        let request = create_test_request();
        let mut generator = request.response_generator("req-engine-2".to_string());

        let backend_output = make_backend_output_with_engine_data();
        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("should produce a response");

        // nvext may or may not be present (tracker may inject worker_id),
        // but engine_data specifically must be absent
        if let Some(nvext) = &response.nvext {
            assert!(
                nvext.get("engine_data").is_none() || nvext.get("engine_data").unwrap().is_null(),
                "engine_data should not be present when not requested"
            );
        }
    }

    #[test]
    fn test_engine_data_excluded_when_other_extra_fields_requested() {
        let request = create_test_request_with_extra_fields(vec!["timing".to_string()]);
        let mut generator = request.response_generator("req-engine-3".to_string());

        let backend_output = make_backend_output_with_engine_data();
        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("should produce a response");

        if let Some(nvext) = &response.nvext {
            assert!(
                nvext.get("engine_data").is_none() || nvext.get("engine_data").unwrap().is_null(),
                "engine_data should not be present when only timing is requested"
            );
        }
    }

    #[test]
    fn test_engine_data_none_from_backend_no_nvext_noise() {
        let request = create_test_request_with_extra_fields(vec!["engine_data".to_string()]);
        let mut generator = request.response_generator("req-engine-4".to_string());

        let backend_output = BackendOutput {
            token_ids: vec![42],
            tokens: vec![Some("hello".to_string())],
            text: Some("hello".to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(common::FinishReason::Stop),
            stop_reason: None,
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
            engine_data: None, // engine didn't provide any data
        };

        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("should produce a response");

        // engine_data is None from backend, so nvext.engine_data should be absent
        if let Some(nvext) = &response.nvext {
            assert!(
                nvext.get("engine_data").is_none() || nvext.get("engine_data").unwrap().is_null(),
                "engine_data should not appear when backend provides None"
            );
        }
    }
}
