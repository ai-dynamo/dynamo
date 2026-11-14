// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Continuing a vLLM `/inference/v1/generate` request on another worker.
//!
//! The routed request carries its prompt and budgets in canonical fields, which
//! migration already rebases: the retry's prompt is the client's prompt followed
//! by the delivered tokens, with `max_tokens`/`min_tokens` reduced by them. This
//! format only refuses requests whose engine state a retry cannot rebuild, stops
//! re-requesting prompt logprobs the client already has, and finishes a request
//! whose budget is spent.

use dynamo_llm::protocols::common::FinishReason;
use dynamo_llm::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use serde_json::Value;

use super::GenerateRequest;

/// How later attempts continue one request.
pub(crate) struct Continuation {
    max_tokens: Option<u32>,
}

impl Continuation {
    /// `Err` says why a failed attempt cannot continue on another worker.
    pub(crate) fn new(request: &GenerateRequest) -> Result<Self, String> {
        let sampling = &request.sampling_params;
        if sampling.seed.is_some() {
            return Err("`sampling_params.seed` keeps RNG state a retry cannot rebuild".into());
        }
        // vLLM counts these over generated tokens only; a retry moves the
        // delivered tokens into its prompt.
        for (field, value) in [
            ("frequency_penalty", sampling.frequency_penalty),
            ("presence_penalty", sampling.presence_penalty),
        ] {
            if value.is_some_and(|value| value != 0.0) {
                return Err(format!(
                    "`sampling_params.{field}` counts generated tokens a retry moves into the prompt"
                ));
            }
        }
        // Bad words match across the delivered/new boundary; extra args may
        // drive custom logits processors with their own state.
        if sampling
            .bad_words
            .as_ref()
            .is_some_and(|words| !words.is_empty())
        {
            return Err("`sampling_params.bad_words` matches across the retry boundary".into());
        }
        if sampling
            .vllm_xargs
            .as_ref()
            .is_some_and(|args| !args.is_empty())
        {
            return Err("`sampling_params.vllm_xargs` may carry engine state".into());
        }
        if request.kv_transfer_params.is_some() {
            return Err("`kv_transfer_params` binds the request to one KV transfer".into());
        }
        Ok(Self {
            max_tokens: sampling.max_tokens(),
        })
    }

    /// Prepare the attempt after `delivered` tokens; `Some` ends the request.
    pub(crate) fn retry(
        &self,
        delivered: usize,
        prompt_delivered: bool,
        request: &mut PreprocessedRequest,
    ) -> Option<LLMEngineOutput> {
        if self
            .max_tokens
            .is_some_and(|max| u32::try_from(delivered).unwrap_or(u32::MAX) >= max)
        {
            return Some(LLMEngineOutput {
                finish_reason: Some(FinishReason::Length),
                ..Default::default()
            });
        }
        if prompt_delivered {
            request.output_options.prompt_logprobs = None;
        }
        // Keep the worker envelope in step with the canonical fields.
        if let Some(sampling) = request
            .extra_args
            .as_mut()
            .and_then(|extra| extra.pointer_mut("/vllm_tito/sampling_params"))
            .and_then(Value::as_object_mut)
        {
            for (key, value) in [
                ("max_tokens", request.stop_conditions.max_tokens),
                ("min_tokens", request.stop_conditions.min_tokens),
            ] {
                if let Some(value) = value {
                    sampling.insert(key.into(), value.into());
                }
            }
            if prompt_delivered {
                sampling.remove("prompt_logprobs");
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn continuation(sampling: Value) -> Result<Continuation, String> {
        let request: GenerateRequest =
            serde_json::from_value(json!({"token_ids": [1], "sampling_params": sampling})).unwrap();
        Continuation::new(&request)
    }

    #[test]
    fn stateful_requests_cannot_migrate() {
        for sampling in [
            json!({"seed": 7}),
            json!({"frequency_penalty": 0.5}),
            json!({"presence_penalty": -0.5}),
            json!({"bad_words": ["x"]}),
            json!({"vllm_xargs": {"processor": 1}}),
        ] {
            assert!(continuation(sampling).is_err());
        }
        let stateless = json!({"frequency_penalty": 0, "presence_penalty": 0,
            "repetition_penalty": 1.2, "logprobs": 1, "prompt_logprobs": 1, "min_tokens": 2});
        assert!(continuation(stateless).is_ok());
        let pinned: GenerateRequest = serde_json::from_value(json!({
            "token_ids": [1], "sampling_params": {}, "kv_transfer_params": {"do_remote_prefill": true},
        }))
        .unwrap();
        assert!(Continuation::new(&pinned).is_err());
    }

    #[test]
    fn retries_follow_the_rebased_budget_and_drop_delivered_prompt_outputs() {
        let continuation = continuation(json!({"max_tokens": 5})).unwrap();
        let mut request = PreprocessedRequest::builder()
            .model("m".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(dynamo_llm::protocols::common::StopConditions {
                max_tokens: Some(3),
                min_tokens: Some(0),
                ..Default::default()
            })
            .sampling_options(Default::default())
            .output_options(dynamo_llm::protocols::common::OutputOptions {
                prompt_logprobs: Some(1),
                ..Default::default()
            })
            .extra_args(Some(json!({"vllm_tito": {"sampling_params": {
                "max_tokens": 5, "min_tokens": 2, "prompt_logprobs": 1,
            }}})))
            .build()
            .unwrap();
        assert!(continuation.retry(2, true, &mut request).is_none());
        assert_eq!(request.output_options.prompt_logprobs, None);
        assert_eq!(
            request.extra_args.unwrap()["vllm_tito"]["sampling_params"],
            json!({"max_tokens": 3, "min_tokens": 0})
        );
        let mut request = PreprocessedRequest::builder()
            .model("m".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        let spent = continuation.retry(5, true, &mut request).unwrap();
        assert_eq!(spent.finish_reason, Some(FinishReason::Length));
        assert!(spent.token_ids.is_empty());
    }
}
