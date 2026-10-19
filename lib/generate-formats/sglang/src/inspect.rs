// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a worker reads from one SGLang `/generate` response.

use dynamo_protocols::types::{CompletionUsage, PromptTokensDetails};
use dynamo_runtime::error::{BackendError, DynamoError, ErrorType};
use serde::Deserialize;
use serde_json::{Value, json};

use super::RESPONSE_KEY;
/// A native SGLang response and the transport signals derived from it.
#[derive(Debug)]
pub struct Inspected {
    pub output: LLMEngineOutput,
    pub terminal: bool,
    pub has_output: bool,
}
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;

fn backend(kind: BackendError, message: String) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}

/// The router counts decode output from `token_ids`; clients get the opaque
/// response in `engine_data`. Prefill reports no tokens and fails on backend
/// errors, since decode would never start.
pub fn inspect(response: Value, is_prefill: bool) -> Result<Inspected, DynamoError> {
    let prefill = is_prefill;
    let token_ids = match response.get("output_ids") {
        Some(ids) if !prefill => Vec::<u32>::deserialize(ids).map_err(|error| {
            backend(
                BackendError::Unknown,
                format!("invalid SGLang output_ids: {error}"),
            )
        })?,
        _ => Vec::new(),
    };
    let has_output = [response.get("output_ids"), response.get("text")]
        .into_iter()
        .flatten()
        .any(|value| match value {
            Value::Array(values) => !values.is_empty(),
            Value::String(value) => !value.is_empty(),
            _ => false,
        });
    let meta = &response["meta_info"];
    let finish = meta.get("finish_reason").filter(|reason| !reason.is_null());
    let message = |value: &Value, default: &str| {
        value
            .get("message")
            .and_then(Value::as_str)
            .filter(|message| !message.is_empty())
            .unwrap_or(default)
            .to_string()
    };
    let mut output = match (response.get("error"), finish) {
        (Some(error), _) => LLMEngineOutput::error(message(error, "SGLang generation failed")),
        (None, Some(finish)) if prefill => match finish.get("type").and_then(Value::as_str) {
            Some(kind @ ("abort" | "error" | "cancelled")) => {
                let status = finish.get("status_code").and_then(Value::as_i64);
                let detail = format!(
                    "SGLang generation {kind}: {} (status_code={}, err_type={})",
                    message(finish, "SGLang generation failed"),
                    status.map_or_else(|| "unknown".to_string(), |s| s.to_string()),
                    finish
                        .get("err_type")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown"),
                );
                let kind = if matches!(status, Some(400..=499)) {
                    BackendError::InvalidArgument
                } else {
                    BackendError::Unknown
                };
                return Err(backend(kind, detail));
            }
            _ => LLMEngineOutput::stop(),
        },
        (None, Some(finish)) => match finish.get("type").and_then(Value::as_str) {
            Some("length") => LLMEngineOutput::length(),
            // SGLang also aborts ordinary cancellations, without error metadata.
            Some("abort")
                if ["status_code", "err_type"]
                    .iter()
                    .any(|key| finish.get(key).is_some_and(|value| !value.is_null())) =>
            {
                LLMEngineOutput::error(message(finish, "SGLang aborted the request with an error"))
            }
            Some("abort") => LLMEngineOutput::cancelled(),
            _ => LLMEngineOutput::stop(),
        },
        (None, None) => LLMEngineOutput::default(),
    };
    let terminal = output.finish_reason.is_some();
    let count = |key| {
        meta.get(key)
            .and_then(Value::as_u64)
            .and_then(|count| u32::try_from(count).ok())
    };
    if terminal
        && !prefill
        && let (Some(prompt_tokens), Some(completion_tokens)) =
            (count("prompt_tokens"), count("completion_tokens"))
    {
        output.completion_usage = Some(CompletionUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            prompt_tokens_details: count("cached_tokens").filter(|&cached| cached > 0).map(
                |cached| PromptTokensDetails {
                    cached_tokens: Some(cached),
                    ..Default::default()
                },
            ),
            completion_tokens_details: None,
        });
    }
    output.token_ids = token_ids;
    output.engine_data = Some(json!({ RESPONSE_KEY: response }));
    Ok(Inspected {
        output,
        terminal,
        has_output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_llm::protocols::common::FinishReason;

    #[test]
    fn decode_terminals_report_typed_finish_and_usage() {
        for (finish, expected) in [
            (json!({"type": "length"}), FinishReason::Length),
            (json!({"type": "stop"}), FinishReason::Stop),
            (json!({"type": "abort"}), FinishReason::Cancelled),
            (
                json!({"type": "abort", "status_code": 400, "message": "bad"}),
                FinishReason::Error("bad".to_string()),
            ),
        ] {
            let response = json!({"output_ids": [7], "meta_info": {
                "finish_reason": finish, "prompt_tokens": 3, "completion_tokens": 1,
                "cached_tokens": 2}});
            let inspected = inspect(response, false).unwrap();
            assert!(inspected.terminal && inspected.has_output);
            let output = inspected.output;
            assert_eq!(output.finish_reason, Some(expected));
            assert_eq!(output.token_ids, [7]);
            let usage = output.completion_usage.unwrap();
            assert_eq!((usage.prompt_tokens, usage.completion_tokens), (3, 1));
            assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, Some(2));
        }
    }

    #[test]
    fn prefill_reports_no_tokens_and_fails_on_backend_errors() {
        let done = json!({"output_ids": [7], "meta_info": {"finish_reason": {"type": "length"}}});
        let inspected = inspect(done, true).unwrap();
        assert!(inspected.output.token_ids.is_empty() && inspected.terminal);
        let failed = json!({"meta_info": {"finish_reason": {"type": "abort", "status_code": 422}}});
        let error = inspect(failed, true).unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(inspect(json!({"output_ids": [-1]}), false).is_err());
    }
}
