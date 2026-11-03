// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::{CompletionEcho, NvCreateCompletionRequest, NvCreateCompletionResponse};
use crate::{
    protocols::{
        common::{
            self,
            extensions::{NvExtProvider, NvExtResponseInput},
            llm_backend::PromptLogprobs,
            timing::RequestTracker,
        },
        openai::{
            convert_backend_top_logprobs,
            delta_common::{self, DeltaGeneratorOptions, DeltaGeneratorState},
        },
    },
    tokenizers::traits::Tokenizer,
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
        DeltaGenerator::new(self.inner.model.clone(), options, request_id)
    }
}

pub struct DeltaGenerator {
    state: DeltaGeneratorState,
    echo: Option<EchoState>,
}

/// What `echo: true` needs to rebuild the prompt side of a choice: the prompt the
/// preprocessor sent to the engine and a tokenizer for its per-token text.
struct EchoState {
    prompt_token_ids: Arc<Vec<TokenIdType>>,
    /// The client's prompt verbatim when it was text; token prompts are decoded.
    prompt_text: Option<String>,
    tokenizer: Arc<dyn Tokenizer>,
    skip_special_tokens: bool,
    /// Choices whose prompt text already went out (once per choice).
    text_emitted: HashSet<u32>,
    /// Choices whose prompt logprobs already went out, or were given up on.
    logprobs_emitted: HashSet<u32>,
    /// Generated text and logprobs held back per choice while its prompt logprobs are
    /// still to come, so the prompt positions precede the generated ones on the wire.
    held: HashMap<u32, (String, Option<dynamo_protocols::types::Logprobs>)>,
}

fn append_logprobs(
    into: &mut Option<dynamo_protocols::types::Logprobs>,
    more: dynamo_protocols::types::Logprobs,
) {
    match into {
        None => *into = Some(more),
        Some(existing) => {
            existing.tokens.extend(more.tokens);
            existing.token_logprobs.extend(more.token_logprobs);
            existing.top_logprobs.extend(more.top_logprobs);
            existing.text_offset.extend(more.text_offset);
        }
    }
}

impl EchoState {
    fn token_text(&self, token_id: TokenIdType, return_as_ids: bool) -> String {
        if return_as_ids {
            return format!("token_id:{token_id}");
        }
        // Single-token decode on purpose: BPE merge and leading-space rules differ
        // between single-token and sequence decode (see `Backend`'s top-logprob fill).
        self.tokenizer
            .decode(&[token_id], self.skip_special_tokens)
            .map(String::from)
            .unwrap_or_default()
    }

    fn prompt_text(&self) -> String {
        if let Some(text) = &self.prompt_text {
            return text.clone();
        }
        self.tokenizer
            .decode(&self.prompt_token_ids, self.skip_special_tokens)
            .map(String::from)
            .unwrap_or_default()
    }

    /// One `logprobs` entry per prompt token from the engine's per-position maps,
    /// in the completions layout: the token's own logprob (none at position 0) and a
    /// token-to-logprob object of the alternatives the engine reported.
    fn logprobs(
        &self,
        prompt_logprobs: &PromptLogprobs,
        return_as_ids: bool,
    ) -> dynamo_protocols::types::Logprobs {
        if prompt_logprobs.len() != self.prompt_token_ids.len() {
            tracing::warn!(
                positions = prompt_logprobs.len(),
                prompt_tokens = self.prompt_token_ids.len(),
                "completions echo: the engine's prompt logprobs do not cover the prompt one-to-one"
            );
        }
        let n = self.prompt_token_ids.len();
        let mut tokens = Vec::with_capacity(n);
        let mut token_logprobs = Vec::with_capacity(n);
        let mut top_logprobs = Vec::with_capacity(n);
        for (position, &token_id) in self.prompt_token_ids.iter().enumerate() {
            let text = self.token_text(token_id, return_as_ids);
            let entries = prompt_logprobs.get(position).and_then(Option::as_ref);
            let selected = entries
                .and_then(|entries| entries.get(&token_id))
                .map(|entry| entry.logprob);
            top_logprobs.push(match entries {
                Some(entries) => {
                    let mut values = serde_json::Map::new();
                    for (&alternative, entry) in entries {
                        let key = if return_as_ids {
                            format!("token_id:{alternative}")
                        } else {
                            entry
                                .decoded_token
                                .clone()
                                .unwrap_or_else(|| self.token_text(alternative, false))
                        };
                        values.insert(key, serde_json::json!(entry.logprob));
                    }
                    // Distinct ids can decode to the same string; the prompt token's
                    // own probability wins, as on the completion side.
                    if let Some(logprob) = selected {
                        values.insert(text.clone(), serde_json::json!(logprob));
                    }
                    serde_json::Value::Object(values)
                }
                None => serde_json::Value::Null,
            });
            token_logprobs.push(selected);
            tokens.push(text);
        }
        dynamo_protocols::types::Logprobs {
            tokens,
            token_logprobs,
            top_logprobs,
            text_offset: vec![],
        }
    }
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
            echo: None,
        }
    }

    /// Echo the prompt back on every choice (`echo: true`). `prompt_text` is the
    /// client's prompt when it was text; token prompts are decoded with `tokenizer`.
    /// The prompt text rides the first chunk of each choice, so a streaming client
    /// sees it before any generated text. When `logprobs` was requested, the prompt
    /// logprobs ride the chunk that carries the engine's payload (vLLM and SGLang
    /// attach it to the final chunk), and the choice's generated text and logprobs
    /// are held back until then, so the wire order stays prompt, then completion.
    pub fn enable_echo(
        &mut self,
        prompt_token_ids: Arc<Vec<TokenIdType>>,
        prompt_text: Option<String>,
        tokenizer: Arc<dyn Tokenizer>,
        skip_special_tokens: bool,
    ) {
        self.echo = Some(EchoState {
            prompt_token_ids,
            prompt_text,
            tokenizer,
            skip_special_tokens,
            text_emitted: HashSet::new(),
            logprobs_emitted: HashSet::new(),
            held: HashMap::new(),
        });
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
            .map(|(_, lp)| lp as f32)
            .collect::<Vec<f32>>();

        let return_as_ids = self.state.options().return_tokens_as_token_ids;
        let top_lps = top_logprobs.map_or(vec![], |top_logprobs| {
            toks.iter()
                .zip(tok_lps.iter())
                .zip(top_logprobs.iter())
                .map(|(((t, tid), lp), top_lps)| {
                    let converted =
                        convert_backend_top_logprobs(top_lps, t, *tid, *lp, return_as_ids);
                    // Completions uses a token-to-logprob object, unlike chat's
                    // array of {token, logprob, bytes} records. Keep the selected
                    // token even when it lies outside the requested top-k.
                    let mut values: serde_json::Map<String, serde_json::Value> = converted
                        .into_iter()
                        .map(|item| (item.token, serde_json::json!(item.logprob)))
                        .collect();
                    let selected = if return_as_ids {
                        format!("token_id:{}", tid)
                    } else {
                        t.clone()
                    };
                    // Distinct IDs can decode to the same string; the map must
                    // preserve the chosen token's probability in that case.
                    values.insert(selected, serde_json::json!(lp));
                    serde_json::Value::Object(values)
                })
                .collect()
        });

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

        Some(dynamo_protocols::types::Logprobs {
            tokens: tokens_out,
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
            inner,
            nvext: None,
            echo: None,
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
            inner,
            nvext: None,
            echo: None,
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
        if let Err(error) = self
            .state
            .options()
            .response_fields
            .validate_metadata_upload_result(
                delta.finish_reason.as_ref(),
                delta.engine_data.as_ref(),
            )
        {
            self.state.tracker_ref().record_finish();
            return Err(error);
        }
        self.state.update_usage_from_backend_output(&delta);

        // Keep token IDs available for optional nvext emission only when requested.
        let completion_token_ids_for_nvext =
            if self.state.options().response_fields.completion_token_ids {
                Some(delta.token_ids.clone())
            } else {
                None
            };
        let logprobs = self.create_logprobs(
            delta.tokens,
            delta.token_ids,
            delta.log_probs,
            delta.top_logprobs,
        );

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
        let index = delta.index.unwrap_or(0);
        let mut response = self.create_choice(index, delta.text.clone(), finish_reason, logprobs);

        // Record finish for timing/ITL accounting even when timing is not returned to the client.
        // Kept at call site because it's a side effect on the tracker — not a gating decision.
        if finish_reason.is_some() {
            self.state.tracker_ref().record_finish();
        }

        // The engine's prompt logprobs have two readers: the nvext field when the
        // client selected it, and `echo`, which rebuilds the prompt side of the choice
        // from them.
        let enable_logprobs = self.state.options().enable_logprobs;
        let (wants_echo_text, wants_echo_logprobs) = match &self.echo {
            Some(echo) => (
                !echo.text_emitted.contains(&index),
                enable_logprobs && !echo.logprobs_emitted.contains(&index),
            ),
            None => (false, false),
        };
        let prompt_logprobs =
            if self.state.options().response_fields.prompt_logprobs || wants_echo_logprobs {
                common::llm_backend::prompt_logprobs_from_engine_data(delta.engine_data.as_ref())?
            } else {
                None
            };
        if wants_echo_logprobs
            && let Some(echo) = self.echo.as_mut()
            && let Some(choice) = response
                .inner
                .choices
                .iter_mut()
                .find(|choice| choice.index == index)
        {
            if prompt_logprobs.is_none() && finish_reason.is_none() {
                // Hold this chunk's generated side until the prompt logprobs arrive.
                let held = echo.held.entry(index).or_default();
                held.0.push_str(&std::mem::take(&mut choice.text));
                if let Some(logprobs) = choice.logprobs.take() {
                    append_logprobs(&mut held.1, logprobs);
                }
            } else if let Some((text, logprobs)) = echo.held.remove(&index) {
                // Release what was held ahead of this chunk's own generated side.
                choice.text.insert_str(0, &text);
                let mut released = logprobs;
                if let Some(current) = choice.logprobs.take() {
                    append_logprobs(&mut released, current);
                }
                choice.logprobs = released;
            }
        }
        if wants_echo_text || wants_echo_logprobs {
            let return_as_ids = self.state.options().return_tokens_as_token_ids;
            let echo = self.echo.as_mut().expect("checked above");
            let text = if wants_echo_text {
                echo.text_emitted.insert(index);
                echo.prompt_text()
            } else {
                String::new()
            };
            let logprobs = match &prompt_logprobs {
                Some(payload) if wants_echo_logprobs => {
                    echo.logprobs_emitted.insert(index);
                    Some(echo.logprobs(payload, return_as_ids))
                }
                None if wants_echo_logprobs && finish_reason.is_some() => {
                    tracing::warn!(
                        "completions echo: the engine reported no prompt logprobs; the prompt was echoed as text only"
                    );
                    echo.logprobs_emitted.insert(index);
                    None
                }
                _ => None,
            };
            if !text.is_empty() || logprobs.is_some() {
                response.echo = Some(CompletionEcho {
                    choice_index: index,
                    text,
                    logprobs,
                });
            }
        }

        // Build the nvext response payload via the shared gating helper on
        // `NvExtResponseFieldSelection` (see `nvext.rs`). Both chat and
        // completions delta generators go through the same helper so the gating
        // rules stay in one place.
        let prompt_logprobs_payload = if self.state.options().response_fields.prompt_logprobs {
            prompt_logprobs
        } else {
            None
        };
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
            no_stop_trim: None,
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

    #[test]
    fn metadata_upload_completion_result_is_opaque_and_required() {
        let nvext = serde_json::from_value(serde_json::json!({"metadata_upload": {"kind": "generation_artifact", "contents": [], "mode": "inline_base64"}})).unwrap();
        let request = make_request_with_nvext(nvext);
        let mut generator = request.response_generator("artifact-result".into());
        assert!(generator.tracker().total_time_ms().is_none());
        assert!(
            generator
                .choice_from_postprocessor(final_backend_output())
                .is_err()
        );
        assert!(generator.tracker().total_time_ms().is_some());
        let mut output = final_backend_output();
        let result = serde_json::json!({"kind": "generation_artifact", "mode": "inline_base64", "inline": {"encoding": "base64", "data": "AQI="}, "backend_specific": {"a": [1, 2]}});
        output.engine_data = Some(serde_json::json!({"metadata_upload": result}));
        let response = generator.choice_from_postprocessor(output).unwrap();
        assert_eq!(response.nvext.unwrap()["metadata_upload"], result);

        let mut malformed = final_backend_output();
        malformed.engine_data = Some(
            serde_json::json!({"metadata_upload": {"kind": "generation_artifact", "mode": {"inline_base64": null}}}),
        );
        assert!(
            request
                .response_generator("artifact-malformed".into())
                .choice_from_postprocessor(malformed)
                .is_err()
        );

        let mut cancelled = final_backend_output();
        cancelled.finish_reason = Some(common::FinishReason::Cancelled);
        cancelled.engine_data = None;
        let response = request
            .response_generator("artifact-cancelled".into())
            .choice_from_postprocessor(cancelled)
            .expect("cancellation must not require an artifact receipt");
        let wire = serde_json::to_value(response).unwrap();
        assert_eq!(wire["choices"][0]["finish_reason"], "stop");
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
            no_stop_trim: None,
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
    fn test_malformed_prompt_logprobs_errors_only_when_requested() {
        let mut invalid_output = final_backend_output();
        invalid_output.engine_data = Some(serde_json::json!({"prompt_logprobs": "invalid"}));

        create_test_request()
            .response_generator("req-prompt-logprobs-unrequested".to_string())
            .choice_from_postprocessor(invalid_output.clone())
            .expect("unrequested prompt logprobs are ignored");

        let error = create_test_request_with_extra_fields(vec!["prompt_logprobs".to_string()])
            .response_generator("req-prompt-logprobs-invalid".to_string())
            .choice_from_postprocessor(invalid_output)
            .expect_err("requested malformed prompt logprobs must fail");
        assert!(
            error
                .to_string()
                .contains("invalid prompt_logprobs payload")
        );
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
        assert!(logprobs.top_logprobs.is_empty());
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
    }

    #[test]
    fn test_completion_top_logprobs_map_preserves_selected_utf8_collision() {
        let mut request = create_test_request();
        request.inner.logprobs = Some(2);
        let generator = request.response_generator("req-utf8-map".to_string());
        let logprobs = generator
            .create_logprobs(
                vec![Some("é".to_string())],
                vec![123],
                Some(vec![-0.5]),
                Some(vec![vec![
                    common::llm_backend::TopLogprob {
                        rank: 1,
                        token_id: 123,
                        token: Some("é".to_string()),
                        logprob: -0.5,
                        bytes: None,
                    },
                    common::llm_backend::TopLogprob {
                        rank: 2,
                        token_id: 999,
                        token: Some("é".to_string()),
                        logprob: -1.0,
                        bytes: None,
                    },
                ]]),
            )
            .expect("logprobs");
        assert_eq!(logprobs.top_logprobs[0], serde_json::json!({"é": -0.5}));
    }

    #[tokio::test]
    async fn test_completion_top_logprobs_schema_streaming_and_aggregated() {
        use crate::protocols::{
            Annotated,
            openai::{ParsingOptions, completions::aggregator::DeltaAggregator},
        };
        use serde_json::json;

        let mut request = create_test_request();
        request.inner.logprobs = Some(1);
        let mut generator = request.response_generator("req-logprobs-map".to_string());
        let mut chunks = Vec::new();
        for (i, text) in ["A", "é"].into_iter().enumerate() {
            let mut output = final_backend_output();
            output.token_ids = vec![i as u32 + 1];
            output.tokens = vec![Some(text.to_string())];
            output.text = Some(text.to_string());
            output.log_probs = Some(vec![-0.5]);
            // The selected token is outside the backend's top-k.
            output.top_logprobs = Some(vec![vec![common::llm_backend::TopLogprob {
                rank: 1,
                token_id: 999,
                token: Some("other".to_string()),
                logprob: -0.25,
                bytes: None,
            }]]);
            output.finish_reason = (i == 1).then_some(common::FinishReason::Stop);
            let chunk = generator.choice_from_postprocessor(output).expect("chunk");
            let wire = serde_json::to_value(&chunk).expect("serialized chunk");
            assert_eq!(
                wire["choices"][0]["logprobs"]["top_logprobs"],
                json!([{text: -0.5, "other": -0.25}])
            );
            chunks.push(Annotated::from_data(chunk));
        }

        let response =
            DeltaAggregator::apply(futures::stream::iter(chunks), ParsingOptions::default())
                .await
                .expect("aggregated response");
        let wire = serde_json::to_value(response).expect("serialized response");
        let choice = &wire["choices"][0];
        assert_eq!(choice["text"], "Aé");
        assert_eq!(choice["finish_reason"], "stop");
        assert_eq!(choice["logprobs"]["tokens"], json!(["A", "é"]));
        assert_eq!(choice["logprobs"]["token_logprobs"], json!([-0.5, -0.5]));
        assert_eq!(
            choice["logprobs"]["top_logprobs"],
            json!([{"A": -0.5, "other": -0.25}, {"é": -0.5, "other": -0.25}])
        );
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

    struct MapTokenizer;

    impl crate::tokenizers::traits::Encoder for MapTokenizer {
        fn encode(&self, _input: &str) -> anyhow::Result<crate::tokenizers::Encoding> {
            anyhow::bail!("echo tests never encode")
        }

        fn encode_batch(
            &self,
            _inputs: &[&str],
        ) -> anyhow::Result<Vec<crate::tokenizers::Encoding>> {
            anyhow::bail!("echo tests never encode")
        }
    }

    impl crate::tokenizers::traits::Decoder for MapTokenizer {
        fn decode(
            &self,
            token_ids: &[TokenIdType],
            skip_special_tokens: bool,
        ) -> anyhow::Result<crate::tokenizers::traits::DecodeResult> {
            let text: String = token_ids
                .iter()
                .map(|id| match id {
                    1 => "<s>",
                    5 => "Hello",
                    7 => " world",
                    9 => " there",
                    _ => "?",
                })
                .filter(|piece| !(skip_special_tokens && *piece == "<s>"))
                .collect();
            Ok(crate::tokenizers::traits::DecodeResult::Complete(text))
        }
    }

    impl crate::tokenizers::traits::Tokenizer for MapTokenizer {}

    fn echo_request(logprobs: Option<u8>) -> NvCreateCompletionRequest {
        let mut request = create_test_request();
        request.inner.echo = Some(true);
        request.inner.logprobs = logprobs;
        request
    }

    /// A chunk with one generated token and the engine's prompt logprobs for the
    /// prompt `[1, 5, 7]`: nothing at position 0, two candidates at position 1.
    fn prompt_logprobs_output(finished: bool) -> BackendOutput {
        let mut output = final_backend_output();
        if !finished {
            output.finish_reason = None;
        }
        output.log_probs = Some(vec![-0.1]);
        output.engine_data = Some(serde_json::json!({
            "prompt_logprobs": [
                null,
                {"5": {"logprob": -0.25}, "9": {"logprob": -1.5, "decoded_token": " there"}},
                {"7": {"logprob": -0.5}}
            ]
        }));
        output
    }

    #[test]
    fn echo_rebuilds_the_prompt_positions_from_the_engine_prompt_logprobs() {
        let request = echo_request(Some(2));
        let mut generator = request.response_generator("req-echo".to_string());
        generator.enable_echo(Arc::new(vec![1, 5, 7]), None, Arc::new(MapTokenizer), false);

        let mut response = generator
            .choice_from_postprocessor(prompt_logprobs_output(false))
            .expect("choice generation");
        let echo = response
            .echo
            .clone()
            .expect("the chunk carrying the prompt logprobs echoes the prompt");
        assert_eq!(echo.choice_index, 0);
        assert_eq!(echo.text, "<s>Hello world");
        let logprobs = echo.logprobs.expect("logprobs were requested");
        assert_eq!(logprobs.tokens, vec!["<s>", "Hello", " world"]);
        assert_eq!(logprobs.token_logprobs, vec![None, Some(-0.25), Some(-0.5)]);
        assert_eq!(logprobs.top_logprobs[0], serde_json::Value::Null);
        assert_eq!(
            logprobs.top_logprobs[1],
            serde_json::json!({"Hello": -0.25, " there": -1.5})
        );
        assert_eq!(
            logprobs.top_logprobs[2],
            serde_json::json!({" world": -0.5})
        );
        assert_eq!(
            serde_json::to_value(&response)
                .expect("serialize chunk")
                .get("echo"),
            None,
            "the echo must not leak as a field of its own"
        );

        response.fold_echo();
        assert!(response.echo.is_none());
        let choice = &response.inner.choices[0];
        assert_eq!(choice.text, "<s>Hello worldhello");
        let folded = choice
            .logprobs
            .as_ref()
            .expect("prompt positions ahead of the completion's");
        assert_eq!(folded.tokens, vec!["<s>", "Hello", " world", "hello"]);
        assert_eq!(
            folded.token_logprobs,
            vec![None, Some(-0.25), Some(-0.5), Some(-0.1)]
        );

        let next = generator
            .choice_from_postprocessor(prompt_logprobs_output(true))
            .expect("choice generation");
        assert!(next.echo.is_none(), "a choice echoes its prompt once");
    }

    #[test]
    fn echo_falls_back_to_the_prompt_text_when_the_engine_reports_no_prompt_logprobs() {
        let request = echo_request(Some(1));
        let mut generator = request.response_generator("req-echo-text".to_string());
        generator.enable_echo(
            Arc::new(vec![5, 7]),
            Some("Hello world".to_string()),
            Arc::new(MapTokenizer),
            true,
        );

        let mut partial = final_backend_output();
        partial.finish_reason = None;
        partial.engine_data = None;
        let response = generator
            .choice_from_postprocessor(partial)
            .expect("choice generation");
        let echo = response
            .echo
            .expect("the choice's first chunk echoes the prompt, ahead of any generated text");
        assert_eq!(echo.text, "Hello world", "a text prompt is echoed verbatim");
        assert!(echo.logprobs.is_none(), "no prompt logprobs yet");

        let mut last = final_backend_output();
        last.engine_data = None;
        let response = generator
            .choice_from_postprocessor(last)
            .expect("choice generation");
        assert!(
            response.echo.is_none(),
            "the engine never reported prompt logprobs: nothing more to echo"
        );
    }

    #[test]
    fn echo_logprobs_ride_the_final_chunk_after_the_text() {
        let request = echo_request(Some(2));
        let mut generator = request.response_generator("req-echo-split".to_string());
        generator.enable_echo(Arc::new(vec![1, 5, 7]), None, Arc::new(MapTokenizer), false);

        // First chunk: no prompt logprobs yet (vLLM and SGLang attach them to the
        // final chunk), so only the text goes out.
        let mut partial = final_backend_output();
        partial.finish_reason = None;
        partial.engine_data = None;
        let response = generator
            .choice_from_postprocessor(partial)
            .expect("choice generation");
        let echo = response
            .echo
            .expect("the first chunk echoes the prompt text");
        assert_eq!(echo.text, "<s>Hello world");
        assert!(echo.logprobs.is_none());
        assert_eq!(
            response.inner.choices[0].text, "",
            "the generated side is held until the prompt logprobs arrive"
        );

        // Final chunk: the prompt logprobs, the held text and this chunk's own, in order.
        let mut response = generator
            .choice_from_postprocessor(prompt_logprobs_output(true))
            .expect("choice generation");
        let echo = response
            .echo
            .clone()
            .expect("the chunk carrying the prompt logprobs echoes them");
        assert_eq!(echo.text, "", "the prompt text already went out");
        let logprobs = echo.logprobs.expect("logprobs were requested");
        assert_eq!(logprobs.tokens, vec!["<s>", "Hello", " world"]);
        response.fold_echo();
        let choice = &response.inner.choices[0];
        assert_eq!(
            choice.text, "hellohello",
            "held text first, then this chunk's"
        );
        let folded = choice.logprobs.as_ref().expect("prompt positions first");
        assert_eq!(folded.tokens, vec!["<s>", "Hello", " world", "hello"]);
        assert_eq!(folded.token_logprobs.last(), Some(&Some(-0.1)));
    }

    #[test]
    fn echo_is_not_built_unless_requested() {
        let request = create_test_request();
        let mut generator = request.response_generator("req-no-echo".to_string());
        let response = generator
            .choice_from_postprocessor(prompt_logprobs_output(true))
            .expect("choice generation");
        assert!(response.echo.is_none());
    }
}
