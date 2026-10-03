// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use super::{NvCreateCompletionRequest, NvCreateCompletionResponse};
use crate::{
    protocols::{
        common::{
            self,
            extensions::{NvExtProvider, NvExtResponseInput},
            timing::RequestTracker,
        },
        openai::{
            compatibility::{profile::Endpoint, telemetry},
            delta_common::{self, DeltaGeneratorOptions, DeltaGeneratorState},
        },
    },
    types::TokenIdType,
};

impl NvCreateCompletionRequest {
    pub fn enable_usage_for_nonstreaming(&mut self, original_stream_flag: bool) {
        delta_common::enable_usage_for_nonstreaming(
            &mut self.inner.stream_options,
            original_stream_flag,
        );
    }

    // put this method on the request
    // inspect the request to extract options
    pub fn response_generator(&self, request_id: String) -> DeltaGenerator {
        let options = DeltaGeneratorOptions::new(
            self.inner.stream_options.as_ref(),
            self.return_tokens_as_token_ids,
            self.inner.logprobs.is_some(),
            self.nvext(),
        );
        let mut generator = DeltaGenerator::new(self.inner.model.clone(), options, request_id);
        generator.capture_prompt_logprobs = self.common.prompt_logprobs.is_some();
        // Native vLLM returns the sampled token plus the requested top-k.
        // Explicit token selection replaces top-k, so preserve all returned IDs.
        if !self
            .unsupported_fields
            .get("logprob_token_ids")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|ids| !ids.is_empty())
        {
            generator.top_logprobs_limit = self.inner.logprobs.map(|count| count as usize + 1);
        }
        generator
    }
}

pub struct DeltaGenerator {
    state: DeltaGeneratorState,
    top_logprobs_limit: Option<usize>,
    /// Streaming offsets are relative to the text already emitted for each choice.
    text_offsets: HashMap<u32, u32>,
    capture_prompt_logprobs: bool,
}

impl DeltaGenerator {
    pub fn new(model: String, options: DeltaGeneratorOptions, request_id: String) -> Self {
        Self {
            state: DeltaGeneratorState::new(
                format!("cmpl-{request_id}"),
                "text_completion".to_string(),
                model,
                options,
            ),
            top_logprobs_limit: None,
            text_offsets: HashMap::new(),
            capture_prompt_logprobs: false,
        }
    }

    /// Returns the request tracker. Tracking is always enabled. For sharing with PreprocessedRequest.
    pub fn tracker(&self) -> Arc<RequestTracker> {
        self.state.tracker()
    }

    pub fn update_isl(&mut self, isl: u32) {
        self.state.update_isl(isl);
    }

    pub fn create_logprobs(
        &self,
        tokens: Vec<common::llm_backend::TokenType>,
        token_ids: Vec<TokenIdType>,
        logprobs: Option<common::llm_backend::LogProbs>,
        top_logprobs: Option<common::llm_backend::TopLogprobs>,
    ) -> Option<dynamo_protocols::types::Logprobs> {
        if !self.state.options().enable_logprobs || logprobs.is_none() {
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
            .map(|(_, lp)| (lp as f32).max(-9999.0))
            .collect::<Vec<f32>>();

        let return_as_ids = self.state.options().return_tokens_as_token_ids;
        let top_lps = toks
            .iter()
            .zip(tok_lps.iter())
            .enumerate()
            .map(|(position, ((t, tid), lp))| {
                let entries = top_logprobs
                    .as_ref()
                    .and_then(|positions| positions.get(position))
                    .map(Vec::as_slice)
                    .unwrap_or_default();
                let selected = if return_as_ids {
                    format!("token_id:{tid}")
                } else {
                    t.clone()
                };
                // Completion top_logprobs is a map, unlike chat's token-object list.
                // Preserve backend order, including last-value-wins when two IDs
                // decode to the same string. Supply the sampled token when an
                // older worker sends only alternatives; it counts in top-k + 1.
                let mut result = serde_json::Map::new();
                let missing_selected = !entries.iter().any(|entry| entry.token_id == *tid);
                if missing_selected {
                    result.insert(selected, serde_json::json!(lp));
                }
                let limit = self
                    .top_logprobs_limit
                    .unwrap_or(usize::MAX)
                    .saturating_sub(usize::from(missing_selected));
                for entry in entries.iter().take(limit) {
                    let token = if return_as_ids {
                        format!("token_id:{}", entry.token_id)
                    } else {
                        entry.token.clone().unwrap_or_default()
                    };
                    result.insert(
                        token,
                        serde_json::json!((entry.logprob as f32).max(-9999.0)),
                    );
                }
                serde_json::Value::Object(result)
            })
            .collect();

        let tokens_out: Vec<String> = toks
            .iter()
            .map(|(t, tid)| {
                if return_as_ids {
                    format!("token_id:{}", tid)
                } else {
                    t.clone()
                }
            })
            .collect();

        let mut offset = 0;
        let text_offset = tokens_out
            .iter()
            .map(|token| {
                let current = offset;
                offset += token.chars().count() as u32;
                current
            })
            .collect();
        Some(dynamo_protocols::types::Logprobs {
            tokens: tokens_out,
            token_logprobs: tok_lps.into_iter().map(Some).collect(),
            text_offset,
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
            id: self.state.id().to_string(),
            object: self.state.object().to_string(),
            created: self.state.created(),
            model: self.state.model().to_string(),
            system_fingerprint: self.state.system_fingerprint().cloned(),
            choices: vec![dynamo_protocols::types::Choice {
                text: text.unwrap_or_default(),
                index,
                finish_reason,
                logprobs,
            }],
            usage: if self.state.is_usage_enabled() && self.state.is_continuous_usage_enabled() {
                Some(self.get_usage())
            } else {
                None
            },
        };

        NvCreateCompletionResponse {
            inner: inner.into(),
            nvext: None,
        }
    }

    /// Creates a final usage-only chunk for OpenAI compliance.
    /// This should be sent after the last content chunk when stream_options.include_usage is true.
    ///
    /// # Returns
    /// * A [`NvCreateCompletionResponse`] with empty choices and usage stats.
    pub fn create_usage_chunk(&self) -> NvCreateCompletionResponse {
        let usage = self.get_usage();

        let inner = dynamo_protocols::types::CreateCompletionResponse {
            id: self.state.id().to_string(),
            object: self.state.object().to_string(),
            created: self.state.created(),
            model: self.state.model().to_string(),
            system_fingerprint: self.state.system_fingerprint().cloned(),
            choices: vec![], // Empty choices for usage-only chunk
            usage: Some(usage),
        };

        NvCreateCompletionResponse {
            inner: inner.into(),
            nvext: None,
        }
    }

    /// Check if usage tracking is enabled
    pub fn is_usage_enabled(&self) -> bool {
        self.state.is_usage_enabled()
    }

    /// Check if continuous usage tracking is enabled
    pub fn is_continuous_usage_enabled(&self) -> bool {
        self.state.is_continuous_usage_enabled()
    }

    pub fn get_usage(&self) -> dynamo_protocols::types::CompletionUsage {
        self.state.get_usage()
    }
}

impl crate::protocols::openai::DeltaGeneratorExt<NvCreateCompletionResponse> for DeltaGenerator {
    fn choice_from_postprocessor(
        &mut self,
        delta: common::llm_backend::BackendOutput,
    ) -> anyhow::Result<NvCreateCompletionResponse> {
        self.state.update_usage_from_backend_output(&delta);

        // Keep token IDs available for optional nvext emission only when requested.
        let completion_token_ids_for_nvext =
            if self.state.options().response_fields.completion_token_ids {
                Some(delta.token_ids.clone())
            } else {
                None
            };
        let index = delta.index.unwrap_or(0);
        let mut logprobs = self.create_logprobs(
            delta.tokens,
            delta.token_ids,
            delta.log_probs,
            delta.top_logprobs,
        );
        if self.state.options().enable_logprobs {
            let offset = self.text_offsets.entry(index).or_default();
            if let Some(logprobs) = logprobs.as_mut() {
                for token_offset in &mut logprobs.text_offset {
                    *token_offset += *offset;
                }
            }
            *offset += delta.text.as_deref().unwrap_or_default().chars().count() as u32;
        }

        // Backend errors are response errors, not successful OpenAI stop reasons.
        // Keep completions aligned with the chat-completions delta generator.
        let finish_reason = match delta.finish_reason.as_ref() {
            Some(common::FinishReason::Error(err_msg)) => {
                self.state.tracker_ref().record_finish();
                return Err(anyhow::anyhow!(err_msg.clone()));
            }
            Some(reason) => Some(reason.clone().into()),
            None => None,
        };
        let stop_reason = delta.stop_reason.clone();

        // create choice
        let mut response = self.create_choice(index, delta.text.clone(), finish_reason, logprobs);

        // Record finish for timing/ITL accounting even when timing is not returned to the client.
        // Kept at call site because it's a side effect on the tracker — not a gating decision.
        if finish_reason.is_some() {
            self.state.tracker_ref().record_finish();
        }

        // Build the nvext response payload via the shared gating helper on
        // `NvExtResponseFieldSelection` (see `nvext.rs`). Both chat and
        // completions delta generators go through the same helper so the gating
        // rules stay in one place.
        let prompt_logprobs_payload = if self.capture_prompt_logprobs
            || self.state.options().response_fields.prompt_logprobs
        {
            telemetry::decode_prompt_logprobs(Endpoint::Completion, delta.engine_data.as_ref())?
        } else {
            None
        };
        if self.capture_prompt_logprobs {
            response.inner.choices[0].internal_prompt_logprobs = prompt_logprobs_payload.clone();
        }
        if let Some(nvext_response) =
            self.state
                .options()
                .response_fields
                .build_response_nvext(NvExtResponseInput {
                    tracker: Some(self.state.tracker_ref()),
                    finish_reason: delta.finish_reason.as_ref(),
                    engine_data: delta.engine_data,
                    stop_reason,
                    completion_token_ids: completion_token_ids_for_nvext.as_deref(),
                    prompt_logprobs: prompt_logprobs_payload,
                })
            && let Ok(nvext_json) = serde_json::to_value(&nvext_response)
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
            if let Some(ref tokens) = nvext_response.completion_token_ids {
                tracing::debug!(
                    "Injected completion_token_ids into completions nvext: {} tokens",
                    tokens.len()
                );
            }
        }

        Ok(response)
    }

    fn get_isl(&self) -> Option<u32> {
        Some(self.state.get_isl())
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

    fn tracker(&self) -> Option<Arc<RequestTracker>> {
        Some(self.state.tracker())
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
            return_tokens_as_token_ids: None,
            unsupported_fields: Default::default(),
        }
    }

    fn make_request_with_nvext(
        nvext: crate::protocols::common::extensions::NvExt,
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
            disaggregated_params: None,
            worker_trace_link: None,
            // routed_experts rides the engine's opaque passthrough.
            engine_data: Some(serde_json::json!({
                "routed_experts": {"layer_0": [1, 3]}
            })),
            encoder_result: None,
            routing_data: None,
            jailed_text: None,
        }
    }

    #[test]
    fn test_response_identity_matches_completion_protocol() {
        let request = create_test_request();
        let generator = request.response_generator("request-id".to_string());

        let response = generator.create_choice(0, None, None, None);

        assert_eq!(response.inner.id, "cmpl-request-id");
        assert_eq!(response.inner.object, "text_completion");
        assert_eq!(response.inner.model, "test-model");
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
                crate::protocols::common::extensions::NvExt::builder()
                    .extra_fields(fields)
                    .build()
                    .unwrap(),
            ),
            metadata: None,
            return_tokens_as_token_ids: None,
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
            encoder_result: None,
            worker_trace_link: None,
            engine_data: Some(serde_json::json!({
                "kv_transfer_time_ms": 12.3,
                "disaggregated_kv_transfer_time_ms": 8.1,
                "prefill_compute_time_ms": 45.6
            })),
            routing_data: None,
            jailed_text: None,
        }
    }

    #[test]
    fn test_plain_request_without_extra_fields_omits_nvext() {
        let request = create_test_request();
        let mut generator = request.response_generator("req-no-nvext".to_string());
        generator
            .tracker()
            .record_worker(42, Some(0), WORKER_TYPE_PREFILL);

        let response = generator
            .choice_from_postprocessor(final_backend_output())
            .expect("choice generation");

        assert!(response.nvext.is_none());
    }

    #[test]
    fn test_backend_error_is_not_converted_to_stop() {
        let request = create_test_request();
        let mut generator = request.response_generator("req-backend-error".to_string());
        let mut output = final_backend_output();
        output.finish_reason = Some(common::FinishReason::Error(
            "invalid prompt embeddings".to_string(),
        ));
        assert!(generator.tracker().total_time_ms().is_none());

        let error = generator
            .choice_from_postprocessor(output)
            .expect_err("backend error must fail the response");

        assert_eq!(error.to_string(), "invalid prompt embeddings");
        assert!(generator.tracker().total_time_ms().is_some());
    }

    #[test]
    fn test_stop_reason_is_suppressed_without_nvext_extra_field() {
        let request = create_test_request();
        let mut generator = request.response_generator("req-stop-reason".to_string());
        let mut output = final_backend_output();
        output.stop_reason = Some(dynamo_protocols::types::StopReason::String(
            "END".to_string(),
        ));

        let response = generator
            .choice_from_postprocessor(output)
            .expect("choice generation");

        let response_json = serde_json::to_value(&response).expect("serialize response");
        assert!(response_json["choices"][0].get("stop_reason").is_none());
        assert!(response_json.get("nvext").is_none());
    }

    #[test]
    fn test_stop_reason_emits_in_nvext_when_requested() {
        let request = create_test_request_with_extra_fields(vec!["stop_reason".to_string()]);
        let mut generator = request.response_generator("req-stop-reason-nvext".to_string());
        let mut output = final_backend_output();
        output.stop_reason = Some(dynamo_protocols::types::StopReason::String(
            "END".to_string(),
        ));

        let response = generator
            .choice_from_postprocessor(output)
            .expect("choice generation");

        let response_json = serde_json::to_value(&response).expect("serialize response");
        assert!(response_json["choices"][0].get("stop_reason").is_none());
        assert_eq!(response_json["nvext"]["stop_reason"], "END");
    }

    #[test]
    fn test_cancelled_detailed_finish_reason_preserves_openai_finish_reason() {
        let request =
            create_test_request_with_extra_fields(vec!["detailed_finish_reason".to_string()]);
        let mut generator = request.response_generator("req-cancelled-nvext".to_string());
        let mut output = final_backend_output();
        output.finish_reason = Some(common::FinishReason::Cancelled);

        let response = generator
            .choice_from_postprocessor(output)
            .expect("choice generation");
        let response_json = serde_json::to_value(response).expect("serialize response");

        assert_eq!(response_json["choices"][0]["finish_reason"], "stop");
        assert_eq!(
            response_json["nvext"]["detailed_finish_reason"],
            "cancelled"
        );
    }

    #[test]
    fn test_logprobs_zero_emits_chosen_token_logprob() {
        let mut request = create_test_request();
        request.inner.logprobs = Some(0);
        let mut generator = request.response_generator("req-logprobs-zero".to_string());
        let mut output = final_backend_output();
        output.log_probs = Some(vec![-0.5]);

        let response = generator
            .choice_from_postprocessor(output)
            .expect("choice generation");
        let logprobs = response.inner.choices[0]
            .logprobs
            .as_ref()
            .expect("logprobs");

        assert_eq!(logprobs.tokens, vec!["hello"]);
        assert_eq!(logprobs.token_logprobs, vec![Some(-0.5)]);
        assert_eq!(
            logprobs.top_logprobs,
            vec![serde_json::json!({"hello": -0.5})]
        );
        assert_eq!(logprobs.text_offset, vec![0]);
    }

    #[test]
    fn test_return_token_ids_formats_selected_top_logprob_fallback() {
        let mut request = create_test_request();
        request.inner.logprobs = Some(1);
        request.return_tokens_as_token_ids = Some(true);
        let generator = request.response_generator("req-token-id-logprobs".to_string());

        let logprobs = generator
            .create_logprobs(
                vec![Some("hello".to_string())],
                vec![123],
                Some(vec![-0.5]),
                Some(vec![vec![common::llm_backend::TopLogprob {
                    rank: 1,
                    token_id: 999,
                    token: Some("other".to_string()),
                    logprob: -1.0,
                    bytes: None,
                }]]),
            )
            .expect("logprobs");

        assert_eq!(logprobs.tokens, vec!["token_id:123"]);
        assert_eq!(
            logprobs.top_logprobs[0],
            serde_json::json!({
                "token_id:999": -1.0, "token_id:123": -0.5
            })
        );
        assert_eq!(logprobs.text_offset, vec![0]);
    }

    #[test]
    fn test_completion_maps_preserve_count_selection_and_duplicate_decoding() {
        for selected_ids in [false, true] {
            let mut request = create_test_request();
            request.inner.logprobs = Some(0);
            if selected_ids {
                request
                    .unsupported_fields
                    .insert("logprob_token_ids".into(), serde_json::json!([2, 3]));
            }
            let generator = request.response_generator("maps".into());
            let entries = [
                (1, "é", -0.5),
                (2, "other", f64::NEG_INFINITY),
                (3, "é", -2.0),
            ]
            .into_iter()
            .map(
                |(token_id, token, logprob)| common::llm_backend::TopLogprob {
                    token_id,
                    token: Some(token.into()),
                    logprob,
                    rank: 1,
                    bytes: None,
                },
            )
            .collect();
            let result = generator
                .create_logprobs(
                    vec![Some("é".into())],
                    vec![1],
                    Some(vec![-0.5]),
                    Some(vec![entries]),
                )
                .unwrap();
            let expected = if selected_ids {
                serde_json::json!({"é": -2.0, "other": -9999.0})
            } else {
                serde_json::json!({"é": -0.5})
            };
            assert_eq!(result.top_logprobs, vec![expected]);
        }
    }

    #[tokio::test]
    async fn test_completion_unicode_offsets_across_choices_chunks_and_aggregation() {
        use crate::protocols::{Annotated, openai::ParsingOptions};
        use futures::stream;

        for return_ids in [false, true] {
            let mut request = create_test_request();
            request.inner.logprobs = Some(0);
            request.return_tokens_as_token_ids = Some(return_ids);
            let mut generator = request.response_generator("offsets".into());
            let mut chunks = Vec::new();
            // Choice 1 must not inherit choice 0's offset; Unicode is counted
            // in code points, not UTF-8 bytes or UTF-16 code units.
            for (index, tokens, ids, text, expected_offsets) in [
                (
                    0,
                    vec!["é", "🦀"],
                    vec![1, 22],
                    "é🦀",
                    if return_ids { vec![0, 10] } else { vec![0, 1] },
                ),
                (1, vec!["x"], vec![3], "x", vec![0]),
                (0, vec!["z"], vec![4], "z", vec![2]),
            ] {
                let mut output = final_backend_output();
                output.index = Some(index);
                output.text = Some(text.into());
                output.log_probs = Some(vec![-0.5; tokens.len()]);
                output.tokens = tokens.into_iter().map(|token| Some(token.into())).collect();
                output.token_ids = ids;
                let response = generator.choice_from_postprocessor(output).unwrap();
                assert_eq!(
                    response.inner.choices[0]
                        .logprobs
                        .as_ref()
                        .unwrap()
                        .text_offset,
                    expected_offsets
                );
                chunks.push(Annotated {
                    data: Some(response),
                    id: None,
                    event: None,
                    comment: None,
                    error: None,
                });
            }
            let response = NvCreateCompletionResponse::from_annotated_stream(
                stream::iter(chunks),
                ParsingOptions::default(),
            )
            .await
            .unwrap();
            let logprobs = response.inner.choices[0].logprobs.as_ref().unwrap();
            assert_eq!(
                logprobs.text_offset,
                if return_ids {
                    vec![0, 10, 21]
                } else {
                    vec![0, 1, 2]
                }
            );
            assert_eq!(
                response.inner.choices[1]
                    .logprobs
                    .as_ref()
                    .unwrap()
                    .text_offset,
                vec![0]
            );
        }
    }

    #[test]
    fn test_timing_extra_field_emits_timing_on_final_chunk() {
        use crate::protocols::common::extensions::NvExt;
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
        use crate::protocols::common::extensions::NvExt;
        let nvext = NvExt::builder()
            .annotations(vec!["query_instance_id:abc".to_string()])
            .build()
            .unwrap();
        let mut generator =
            make_request_with_nvext(nvext).response_generator("req-qid".to_string());
        generator
            .tracker()
            .record_worker(42, Some(0), WORKER_TYPE_PREFILL);
        // The query-only tokenized prompt reaches the delta generator via the tracker,
        // mirroring the standalone-router round-trip the preprocessor drains.
        generator
            .tracker()
            .set_external_query_token_ids(vec![11, 22, 33]);

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
        use crate::protocols::common::extensions::NvExt;
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
            encoder_result: None,
            worker_trace_link: None,
            engine_data: None, // engine didn't provide any data
            routing_data: None,
            jailed_text: None,
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
