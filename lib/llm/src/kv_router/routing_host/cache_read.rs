// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{preprocessor::PreprocessedRequest, protocols::common::timing::RequestPhase};

use super::RoutingHost;

impl RoutingHost {
    pub(super) fn normalize_prefix_cache_read_control(
        &self,
        request: &mut PreprocessedRequest,
        phase: RequestPhase,
    ) {
        if !supports_request(request, phase) {
            request.skip_reading_prefix_cache = None;
            return;
        }
        let already_resolved = request.skip_reading_prefix_cache;
        let Some(skip) = already_resolved.or_else(|| cache_read_control(request)) else {
            return;
        };
        if !can_shadow_control(request) {
            request.skip_reading_prefix_cache = None;
            return;
        }
        let supports_control = self.kv_router_if_enabled().is_some_and(|router| {
            let workers = router.workers_with_configs.borrow();
            // Mixed/legacy pools retain their existing translation. Request-wide
            // precision starts only once every worker advertises this contract.
            !workers.is_empty()
                && workers
                    .values()
                    .all(|config| config.supports_prefix_cache_read_control())
        });
        if supports_control || already_resolved.is_some() {
            // N-2 workers ignore the typed field. Keep its decision authoritative
            // if one joins while this request waits in the scheduler queue.
            // TODO(v1.9): Remove this shadow when v1.6 leaves the N-2 window.
            let extra = request
                .extra_args
                .get_or_insert_with(|| serde_json::json!({}));
            if extra.is_null() {
                *extra = serde_json::json!({});
            }
            if let Some(sampling) = extra.as_object_mut().and_then(|extra| {
                let sampling = extra
                    .entry("sampling_options")
                    .or_insert_with(|| serde_json::json!({}));
                if sampling.is_null() {
                    *sampling = serde_json::json!({});
                }
                sampling.as_object_mut()
            }) {
                sampling.insert("skip_reading_prefix_cache".to_string(), skip.into());
            }
        }
        request.skip_reading_prefix_cache = supports_control.then_some(skip);
    }
}

fn supports_request(request: &PreprocessedRequest, phase: RequestPhase) -> bool {
    if phase != RequestPhase::Aggregated
        || request.prompt_embeds.is_some()
        || request.multi_modal_data.is_some()
        || request.mm_routing_info.is_some()
    {
        return false;
    }
    let extra = request.extra_args.as_ref();
    if extra.and_then(|extra| extra.get("vllm_tito")).is_some() {
        return false;
    }
    true
}

fn can_shadow_control(request: &PreprocessedRequest) -> bool {
    request.extra_args.as_ref().is_none_or(|extra| {
        extra.is_null()
            || extra.as_object().is_some_and(|extra| {
                extra
                    .get("sampling_options")
                    .is_none_or(|sampling| sampling.is_null() || sampling.is_object())
            })
    })
}

fn cache_read_control(request: &PreprocessedRequest) -> Option<bool> {
    let extra = request.extra_args.as_ref();
    let nested = extra
        .and_then(|extra| extra.get("sampling_options"))
        .and_then(|sampling| sampling.get("skip_reading_prefix_cache"));
    let prompt_logprobs = request.output_options.prompt_logprobs.is_some();
    if !prompt_logprobs {
        // Direct extra_args controls are consulted only for prompt logprobs.
        return nested.and_then(serde_json::Value::as_bool);
    }
    // A present nested null masks the direct override, matching the adapter.
    let value = nested.or_else(|| extra.and_then(|extra| extra.get("skip_reading_prefix_cache")));
    match value {
        None | Some(serde_json::Value::Null) => Some(true),
        Some(serde_json::Value::Bool(skip)) => Some(*skip),
        // Leave malformed backend controls to the adapter's validation.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn request(prompt: Option<u32>, extra: Value) -> PreprocessedRequest {
        let mut request = PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        request.output_options.prompt_logprobs = prompt;
        request.extra_args = Some(extra);
        request
    }

    #[test]
    fn prompt_logprobs_and_overrides_match_worker_precedence() {
        for (prompt, extra, expected) in [
            (None, json!({}), None),
            (Some(0), json!({}), Some(true)),
            (
                Some(1),
                json!({"skip_reading_prefix_cache": false}),
                Some(false),
            ),
            (
                Some(1),
                json!({"sampling_options": {"skip_reading_prefix_cache": false}, "skip_reading_prefix_cache": true}),
                Some(false),
            ),
            (
                Some(1),
                json!({"sampling_options": {"skip_reading_prefix_cache": null}, "skip_reading_prefix_cache": false}),
                Some(true),
            ),
            (None, json!({"skip_reading_prefix_cache": true}), None),
            (
                None,
                json!({"sampling_options": {"skip_reading_prefix_cache": true}}),
                Some(true),
            ),
            (
                None,
                json!({"sampling_options": {"skip_reading_prefix_cache": false}}),
                Some(false),
            ),
            (
                None,
                json!({"sampling_options": {"skip_reading_prefix_cache": null}}),
                None,
            ),
            (
                Some(1),
                json!({"sampling_options": {"skip_reading_prefix_cache": "invalid"}}),
                None,
            ),
            (
                Some(1),
                json!({"skip_reading_prefix_cache": "invalid"}),
                None,
            ),
        ] {
            let request = request(prompt, extra);
            assert_eq!(cache_read_control(&request), expected);
        }
    }

    #[test]
    fn excluded_paths_have_no_inferred_cache_read_control() {
        let mut request = request(Some(0), json!({}));
        for phase in [RequestPhase::Prefill, RequestPhase::Decode] {
            assert!(!supports_request(&request, phase));
        }
        request.extra_args = Some(json!({"vllm_tito": {}}));
        assert!(!supports_request(&request, RequestPhase::Aggregated));
        request.extra_args = None;
        request.prompt_embeds = Some("embedding".to_string());
        assert!(!supports_request(&request, RequestPhase::Aggregated));
        request.prompt_embeds = None;
        request.multi_modal_data = Some(Default::default());
        assert!(!supports_request(&request, RequestPhase::Aggregated));
        request.multi_modal_data = None;
        request.mm_routing_info = Some(Default::default());
        assert!(!supports_request(&request, RequestPhase::Aggregated));
    }
}
