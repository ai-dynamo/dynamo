// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Continuing a vLLM `/inference/v1/generate` request on another worker.
//!
//! The routed request carries its prompt and budgets in canonical fields, which
//! migration already rebases: the retry's prompt is the client's prompt followed
//! by the delivered tokens, with `max_tokens`/`min_tokens` reduced by them. This
//! format only refuses requests whose engine state a retry cannot rebuild, copies
//! the rebased budgets into the worker envelope, and stops re-requesting prompt
//! logprobs the client already has.

use dynamo_llm::protocols::common::llm_backend::PreprocessedRequest;
use serde_json::Value;

use super::GenerateRequest;

/// How later attempts continue one request.
pub(crate) struct Continuation;

impl Continuation {
    /// `Err` says why a failed attempt cannot continue on another worker.
    pub(crate) fn new(request: &GenerateRequest) -> Result<Self, String> {
        let sampling = &request.sampling_params;
        // Without the client's limit, the worker resolves a default the
        // frontend cannot see, and a retry would get the full default again.
        if sampling.max_tokens.is_none() {
            return Err(
                "without `sampling_params.max_tokens`, the effective token budget is unknown"
                    .into(),
            );
        }
        let raw = |field| sampling.raw.get(field).filter(|value| !value.is_null());
        // Canonical replay holds one choice.
        if raw("n").is_some_and(|n| n.as_u64() != Some(1)) {
            return Err(
                "`sampling_params.n` other than 1 has per-choice state a retry cannot rebuild"
                    .into(),
            );
        }
        // The grammar state lives in the engine; the canonical guided-decoding
        // guard cannot see it in the native body.
        if raw("structured_outputs").is_some() {
            return Err(
                "`sampling_params.structured_outputs` keeps grammar state a retry cannot rebuild"
                    .into(),
            );
        }
        // A stop string can start in the delivered text and end in the retry.
        if raw("stop").is_some_and(|stop| stop.as_array().is_none_or(|stop| !stop.is_empty())) {
            return Err("`sampling_params.stop` matches across the retry boundary".into());
        }
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
        Ok(Self)
    }

    /// Prepare the next attempt of `request`, whose canonical prompt and
    /// budgets migration has already rebased.
    pub(crate) fn retry(
        &self,
        prompt_delivered: bool,
        request: &mut PreprocessedRequest,
    ) -> Result<(), String> {
        // vLLM rejects `max_tokens: 0`.
        if request.stop_conditions.max_tokens == Some(0) {
            return Err("the request has no token budget left".into());
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
        Ok(())
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
            json!({}),
            json!({"max_tokens": 4, "seed": 7}),
            json!({"max_tokens": 4, "n": 2}),
            json!({"max_tokens": 4, "structured_outputs": {"json": {}}}),
            json!({"max_tokens": 4, "stop": "x"}),
            json!({"max_tokens": 4, "stop": ["x"]}),
            json!({"frequency_penalty": 0.5}),
            json!({"presence_penalty": -0.5}),
            json!({"bad_words": ["x"]}),
            json!({"vllm_xargs": {"processor": 1}}),
        ] {
            assert!(continuation(sampling).is_err());
        }
        let stateless = json!({"max_tokens": 4, "n": 1, "stop": [], "frequency_penalty": 0, "presence_penalty": 0,
            "repetition_penalty": 1.2, "logprobs": 1, "prompt_logprobs": 1, "min_tokens": 2});
        assert!(continuation(stateless).is_ok());
        let pinned: GenerateRequest = serde_json::from_value(json!({
            "token_ids": [1], "sampling_params": {"max_tokens": 4}, "kv_transfer_params": {"do_remote_prefill": true},
        }))
        .unwrap();
        assert!(Continuation::new(&pinned).is_err());
    }

    #[test]
    fn retries_copy_the_rebased_budget_and_drop_delivered_prompt_outputs() {
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
        continuation.retry(true, &mut request).unwrap();
        assert_eq!(request.output_options.prompt_logprobs, None);
        assert_eq!(
            request.extra_args.unwrap()["vllm_tito"]["sampling_params"],
            json!({"max_tokens": 3, "min_tokens": 0})
        );
        let mut request = PreprocessedRequest::builder()
            .model("m".to_string())
            .token_ids(vec![1])
            .stop_conditions(dynamo_llm::protocols::common::StopConditions {
                max_tokens: Some(0),
                ..Default::default()
            })
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        assert!(continuation.retry(true, &mut request).is_err());
    }
}
