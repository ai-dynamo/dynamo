// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve only the cache-read semantics explicitly declared by a worker.

use std::sync::Arc;

use dynamo_kv_router::protocols::PrefixCacheReadResolver;
use serde_json::Value;

use crate::preprocessor::PreprocessedRequest;
use crate::protocols::common::timing::RequestPhase;

pub(super) fn resolver(
    request: &PreprocessedRequest,
    phase: RequestPhase,
) -> Option<PrefixCacheReadResolver> {
    if phase != RequestPhase::Aggregated
        || request.prompt_embeds.is_some()
        || request.multi_modal_data.is_some()
        || request.mm_routing_info.is_some()
    {
        return None;
    }
    let extra = request.extra_args.as_ref();
    // Native Generate has its own SamplingParams envelope and conversion rules.
    if extra.and_then(|extra| extra.get("vllm_tito")).is_some() {
        return None;
    }
    let prompt_logprobs = request.output_options.prompt_logprobs.is_some();
    // PreprocessedRequest's typed SamplingOptions has no cache-read field.
    // Backend-specific sampling controls survive in this nested passthrough.
    let nested = extra
        .and_then(|extra| extra.get("sampling_options"))
        .and_then(|sampling| sampling.get("skip_reading_prefix_cache"));
    let direct = extra.and_then(|extra| extra.get("skip_reading_prefix_cache"));
    // Do not infer execution from malformed controls; backend validation owns them.
    if nested.is_some_and(|value| !value.is_null() && !value.is_boolean())
        || (prompt_logprobs
            && nested.is_none()
            && direct.is_some_and(|value| !value.is_null() && !value.is_boolean()))
    {
        return None;
    }
    let nested_present = nested.is_some();
    let nested = nested.and_then(Value::as_bool);
    let direct = direct.and_then(Value::as_bool);
    let skip = if prompt_logprobs {
        (if nested_present { nested } else { direct }).unwrap_or(true)
    } else {
        // Ordinary requests and model defaults need no new per-request resolver.
        nested?
    };
    Some(Arc::new(move |policy| {
        (policy == Some("vllm_python_v1")).then_some(!skip)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(prompt: Option<u32>, extra: Option<Value>) -> PreprocessedRequest {
        let mut request = PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        request.output_options.prompt_logprobs = prompt;
        request.extra_args = extra;
        request
    }

    #[rstest::rstest]
    #[case(None, None, None)]
    #[case(Some(0), None, Some(false))]
    #[case(Some(1), None, Some(false))]
    #[case(Some(1), Some(serde_json::json!({"sampling_options": {"skip_reading_prefix_cache": false}, "skip_reading_prefix_cache": true})), Some(true))]
    #[case(Some(1), Some(serde_json::json!({"sampling_options": {"skip_reading_prefix_cache": null}, "skip_reading_prefix_cache": false})), Some(false))]
    #[case(Some(1), Some(serde_json::json!({"skip_reading_prefix_cache": false})), Some(true))]
    #[case(None, Some(serde_json::json!({"skip_reading_prefix_cache": true})), None)]
    #[case(None, Some(serde_json::json!({"sampling_options": {"skip_reading_prefix_cache": true}})), Some(false))]
    #[case(None, Some(serde_json::json!({"sampling_options": {"skip_reading_prefix_cache": false}})), Some(true))]
    #[case(Some(1), Some(serde_json::json!({"sampling_options": {"skip_reading_prefix_cache": "invalid"}})), None)]
    #[case(Some(1), Some(serde_json::json!({"vllm_tito": {}})), None)]
    fn controls_match_the_declared_adapter(
        #[case] prompt: Option<u32>,
        #[case] extra: Option<Value>,
        #[case] expected: Option<bool>,
    ) {
        let request = request(prompt, extra);
        let resolved = resolver(&request, RequestPhase::Aggregated);
        assert_eq!(
            resolved
                .as_ref()
                .and_then(|resolve| resolve(Some("vllm_python_v1"))),
            expected
        );
        if let Some(resolve) = resolved {
            assert_eq!(resolve(None), None);
            assert_eq!(resolve(Some("unknown-adapter")), None);
        }
    }

    #[test]
    fn remote_prefill_and_decode_have_no_inferred_policy() {
        let request = request(Some(1), None);
        assert!(resolver(&request, RequestPhase::Prefill).is_none());
        assert!(resolver(&request, RequestPhase::Decode).is_none());
    }
}
