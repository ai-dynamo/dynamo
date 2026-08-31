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
    #[serde(default)]
    return_logprob: bool,
    #[serde(default)]
    top_logprobs_num: i64,
    #[serde(default = "no_prompt_logprobs")]
    logprob_start_len: i64,
}

fn no_prompt_logprobs() -> i64 {
    -1
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
        controls.return_logprob,
        controls.top_logprobs_num,
        controls.logprob_start_len,
    )
    .map_err(invalid_argument)?;
    let request_id = controls
        .rid
        .unwrap_or_else(|| fallback_request_id.to_string());
    Ok(Some(ResponseMetadata::new(
        request_id,
        &request.token_ids,
        logprobs,
    )))
}

/// Wrap one canonical chunk in the native response the frontend unwraps.
pub(super) fn adapt(
    metadata: &ResponseMetadata,
    output: &mut LLMEngineOutput,
    completion_tokens: usize,
) {
    let response = metadata.response(
        &output.token_ids,
        completion_tokens,
        output.finish_reason.as_ref().map(native_finish_reason),
    );
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
    fn wraps_each_chunk_and_its_terminal_finish_reason() {
        let metadata = native_metadata(json!({"rid": "request-1"}));
        let mut token = LLMEngineOutput {
            token_ids: vec![101],
            ..Default::default()
        };
        adapt(&metadata, &mut token, 1);
        let response = &token.engine_data.as_ref().unwrap()["sglang_response"];
        assert_eq!(response["output_ids"], json!([101]));
        assert_eq!(response["meta_info"]["id"], "request-1");
        assert!(response["meta_info"]["finish_reason"].is_null());

        let mut terminal = LLMEngineOutput::length();
        adapt(&metadata, &mut terminal, 1);
        let response = &terminal.engine_data.as_ref().unwrap()["sglang_response"];
        assert_eq!(response["output_ids"], json!([]));
        assert_eq!(
            response["meta_info"]["finish_reason"],
            json!({"type": "length"})
        );
    }

    #[test]
    fn logprob_controls_reach_the_response_builder() {
        let metadata = native_metadata(json!({"return_logprob": true, "top_logprobs_num": 2}));
        let mut token = LLMEngineOutput {
            token_ids: vec![107],
            ..Default::default()
        };
        adapt(&metadata, &mut token, 1);
        let meta = &token.engine_data.as_ref().unwrap()["sglang_response"]["meta_info"];
        assert_eq!(meta["output_token_logprobs"][0][1], 107);
        assert_eq!(meta["output_top_logprobs"][0].as_array().unwrap().len(), 2);
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
}
