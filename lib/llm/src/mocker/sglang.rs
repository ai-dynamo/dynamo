// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native SGLang `/generate` response shaping for the mocker engine.

use dynamo_mocker::sglang::{LogprobOptions, ResponseMetadata};
use dynamo_runtime::error::{DynamoError, ErrorType};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::protocols::common::FinishReason;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};

const PAYLOAD_KEY: &str = "sglang_tito";

/// The part of the opaque native payload the mocker has to reproduce. Unknown
/// fields are ignored, so a newer SGLang control set still parses.
#[derive(Deserialize)]
struct NativeControls {
    rid: Option<String>,
    return_logprob: Option<bool>,
    top_logprobs_num: Option<i64>,
    logprob_start_len: Option<i64>,
    token_ids_logprob: Option<Vec<u32>>,
}

/// Response metadata for a native SGLang request, or `None` for every other
/// request, which the mocker answers with its canonical stream.
pub(super) fn response_metadata(
    request: &PreprocessedRequest,
    fallback_request_id: &str,
) -> Result<Option<ResponseMetadata>, DynamoError> {
    let Some(payload) = request
        .extra_args
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|extra| extra.get(PAYLOAD_KEY))
    else {
        return Ok(None);
    };
    let controls = NativeControls::deserialize(payload)
        .map_err(|error| invalid_argument(format!("invalid extra_args.{PAYLOAD_KEY}: {error}")))?;
    let logprobs = LogprobOptions::new(
        controls.return_logprob.unwrap_or(false),
        controls.top_logprobs_num.unwrap_or(0),
        controls.logprob_start_len.unwrap_or(-1),
    )
    .map_err(invalid_argument)?;
    let request_id = controls
        .rid
        .unwrap_or_else(|| fallback_request_id.to_string());
    let metadata = ResponseMetadata::new(request_id, &request.token_ids, logprobs);
    Ok(Some(match controls.token_ids_logprob {
        Some(token_ids) => metadata.with_candidate_token_ids(token_ids),
        None => metadata,
    }))
}

/// Wrap one canonical chunk in the native response the frontend unwraps.
pub(super) fn adapt(
    metadata: &ResponseMetadata,
    output: &mut LLMEngineOutput,
    completion_tokens: usize,
) {
    let mut response = metadata.response(
        &output.token_ids,
        completion_tokens,
        output.finish_reason.as_ref().map(native_finish_reason),
    );
    if let Some(cached_tokens) = output
        .completion_usage
        .as_ref()
        .and_then(|usage| usage.prompt_tokens_details.as_ref())
        .and_then(|details| details.cached_tokens)
    {
        response["meta_info"]["cached_tokens"] = json!(cached_tokens);
    }
    output.engine_data = Some(json!({"sglang_response": response}));
}

fn native_finish_reason(reason: &FinishReason) -> Value {
    match reason {
        FinishReason::Length => json!({"type": "length"}),
        FinishReason::EoS | FinishReason::Stop => json!({"type": "stop"}),
        FinishReason::Cancelled => json!({
            "type": "abort",
            "message": "request was cancelled",
        }),
        FinishReason::Error(message) => json!({
            "type": "abort",
            "message": message,
        }),
        FinishReason::ContentFilter => json!({
            "type": "abort",
            "message": "generation stopped by content filter",
        }),
    }
}

fn invalid_argument(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message.into())
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};

    fn request(extra_args: Option<Value>) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("mock".to_string())
            .token_ids(vec![11, 12, 13])
            .stop_conditions(StopConditions {
                max_tokens: Some(2),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .extra_args(extra_args)
            .build()
            .unwrap()
    }

    fn native_metadata(payload: Value) -> ResponseMetadata {
        response_metadata(&request(Some(json!({"sglang_tito": payload}))), "fallback")
            .unwrap()
            .unwrap()
    }

    #[test]
    fn only_requests_carrying_the_native_payload_are_adapted() {
        assert!(
            response_metadata(&request(None), "fallback")
                .unwrap()
                .is_none()
        );
        assert!(
            response_metadata(&request(Some(json!({"other_engine": {}}))), "fallback")
                .unwrap()
                .is_none()
        );
        assert!(response_metadata(&request(Some(json!({"sglang_tito": 7}))), "fallback").is_err());
    }

    #[test]
    fn resolves_the_request_id_and_ignores_unknown_controls() {
        let metadata =
            native_metadata(json!({"rid": "resolved-id", "future_field": {"opaque": true}}));
        assert_eq!(metadata.request_id(), "resolved-id");

        // Without an `rid` the mocker replies under the context request ID.
        assert_eq!(native_metadata(json!({})).request_id(), "fallback");
    }

    #[test]
    fn null_logprob_controls_use_sglang_defaults() {
        let metadata = native_metadata(json!({
            "return_logprob": null,
            "top_logprobs_num": null,
            "logprob_start_len": null,
        }));
        let response = metadata.response(&[107], 1, Some(json!({"type": "length"})));
        assert!(response["meta_info"].get("output_token_logprobs").is_none());

        let metadata = native_metadata(json!({
            "return_logprob": true,
            "top_logprobs_num": null,
            "logprob_start_len": null,
        }));
        let response = metadata.response(&[107], 1, Some(json!({"type": "length"})));
        let meta = &response["meta_info"];
        assert!(meta.get("output_token_logprobs").is_some());
        assert!(meta.get("output_top_logprobs").is_none());
        assert!(meta.get("input_token_logprobs").is_none());
    }

    #[test]
    fn maps_cancellation_and_errors_to_abort() {
        let metadata = native_metadata(json!({"rid": "terminal"}));

        for (mut output, expected_message) in [
            (
                LLMEngineOutput::error("backend failed".to_string()),
                "backend failed",
            ),
            (LLMEngineOutput::cancelled(), "request was cancelled"),
        ] {
            adapt(&metadata, &mut output, 0);
            let finish = &output.engine_data.as_ref().unwrap()["sglang_response"]["meta_info"]["finish_reason"];
            assert_eq!(finish["type"], "abort");
            assert_eq!(finish["message"], expected_message);
        }
    }

    #[test]
    fn forwards_measured_cache_usage_without_inventing_unknown_counts() {
        let metadata = native_metadata(json!({}));
        for count in [None, Some(0), Some(2)] {
            let mut output = LLMEngineOutput::length();
            output.completion_usage =
                count.map(|cached| super::super::usage_with_cached_tokens(3, 0, cached));
            adapt(&metadata, &mut output, 0);
            let meta = &output.engine_data.as_ref().unwrap()["sglang_response"]["meta_info"];
            match count {
                Some(cached) => assert_eq!(meta["cached_tokens"], cached),
                None => assert!(meta.get("cached_tokens").is_none()),
            }
        }
    }

    #[test]
    fn preserves_requested_candidate_ids() {
        let metadata = native_metadata(json!({
            "return_logprob": true,
            "token_ids_logprob": [17, 4]
        }));
        let response = metadata.response(&[42], 1, Some(json!({"type": "length"})));
        assert_eq!(
            response["meta_info"]["output_token_ids_logprobs"],
            json!([[[-10.8, 17, null], [-10.5, 4, null]]])
        );
    }
}
