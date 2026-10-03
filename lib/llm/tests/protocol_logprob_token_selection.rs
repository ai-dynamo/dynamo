// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::engines::ValidateRequest;
use dynamo_llm::protocols::openai::{
    chat_completions::NvCreateChatCompletionRequest, completions::NvCreateCompletionRequest,
};
use serde_json::json;

#[test]
fn reviewed_root_extensions_survive_typed_request_reserialization() {
    // The Python frontend sees this serialized form, not the original HTTP
    // bytes. Exercise falsy values as well as the nonempty sampling directives.
    for fields in [
        json!({"allowed_token_ids": [], "bad_words_token_ids": [], "logprob_token_ids": [],
               "stop_token_ids": [], "detokenize": false, "cache_salt": ""}),
        json!({"allowed_token_ids": [0], "bad_words_token_ids": [[0]], "logprob_token_ids": [0],
               "stop_token_ids": [1], "detokenize": true, "cache_salt": "isolation-key"}),
    ] {
        for chat in [true, false] {
            let mut body = fields.clone();
            body["model"] = json!("contract-probe");
            // This unknown key may be ignored by migration admission, but it
            // must never be promoted into the Python/backend contract.
            body["unreviewed_extension"] = json!("not-forwarded");
            let serialized = if chat {
                body["messages"] = json!([{"role": "user", "content": "Hello"}]);
                let request: NvCreateChatCompletionRequest = serde_json::from_value(body).unwrap();
                serde_json::to_value(request).unwrap()
            } else {
                body["prompt"] = json!("Hello");
                let request: NvCreateCompletionRequest = serde_json::from_value(body).unwrap();
                serde_json::to_value(request).unwrap()
            };
            for (field, value) in fields.as_object().unwrap() {
                assert_eq!(serialized.get(field), Some(value), "{field}, chat={chat}");
            }
            assert!(serialized.get("unreviewed_extension").is_none());
        }
    }
}

#[test]
fn chat_explicit_token_selection_requires_logprobs_true() {
    for selection in [json!(null), json!([]), json!([0]), json!([1, 2])] {
        for enabled in [json!(null), json!(false), json!(true)] {
            let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
                "model": "contract-probe",
                "messages": [{"role": "user", "content": "Hello"}],
                "logprob_token_ids": selection,
                "logprobs": enabled,
            }))
            .unwrap();
            let needs_logprobs = selection.as_array().is_some_and(|ids| !ids.is_empty());
            let result = ValidateRequest::validate(&request);
            assert_eq!(result.is_ok(), !needs_logprobs || enabled == json!(true));
            if let Err(error) = result {
                assert!(error.to_string().contains("logprob_token_ids"));
            }
        }
    }
}

#[test]
fn completion_explicit_token_selection_accepts_logprobs_zero() {
    for selection in [json!(null), json!([]), json!([0]), json!([1, 2])] {
        for count in [json!(null), json!(0), json!(1)] {
            let request: NvCreateCompletionRequest = serde_json::from_value(json!({
                "model": "contract-probe",
                "prompt": "Hello",
                "logprob_token_ids": selection,
                "logprobs": count,
            }))
            .unwrap();
            let needs_logprobs = selection.as_array().is_some_and(|ids| !ids.is_empty());
            let result = ValidateRequest::validate(&request);
            assert_eq!(result.is_ok(), !needs_logprobs || !count.is_null());
            if let Err(error) = result {
                assert!(error.to_string().contains("logprob_token_ids"));
            }
        }
    }
}
