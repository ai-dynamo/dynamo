// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::systemone::{build_native_score_request, parse_candidate_scores};
use serde_json::json;

#[test]
fn parses_exact_native_candidate_scores_in_requested_order() {
    let response = json!({
        "output_ids": [],
        "meta_info": {
            "finish_reason": {"type": "length"},
            "completion_tokens": 0,
            "output_token_ids_logprobs": [[
                [-0.2, 17, null],
                [-1.3, 4, "token"]
            ]]
        }
    });

    assert_eq!(
        parse_candidate_scores(&response, &[17, 4]).unwrap(),
        vec![-0.2, -1.3]
    );
}

#[test]
fn rejects_decoded_tokens_in_prefill_only_scoring() {
    let response = json!({
        "output_ids": [42],
        "meta_info": {
            "finish_reason": {"type": "length"},
            "completion_tokens": 1,
            "output_token_ids_logprobs": [[[-0.2, 17, null], [-1.3, 4, null]]]
        }
    });
    assert!(parse_candidate_scores(&response, &[17, 4]).is_err());
    let mut response = response;
    response["output_ids"] = json!([]);
    assert!(parse_candidate_scores(&response, &[17, 4]).is_err());
}

#[test]
fn requires_explicit_zero_completion_accounting() {
    let response = json!({"output_ids":[], "meta_info":{
        "finish_reason":{"type":"length"},
        "output_token_ids_logprobs":[[[-0.2,17,null],[-1.3,4,null]]]
    }});
    assert!(parse_candidate_scores(&response, &[17, 4]).is_err());
}

#[test]
fn rejects_partial_reordered_or_aborted_native_candidate_scores() {
    for response in [
        json!({"meta_info": {"finish_reason": null, "output_token_ids_logprobs": [[[-0.2, 17, null]]]}}),
        json!({"meta_info": {"finish_reason": {"type": "abort"}, "output_token_ids_logprobs": [[[-0.2, 17, null]]]}}),
        json!({"meta_info": {"finish_reason": {"type": "length"}, "output_token_ids_logprobs": [[[-0.2, 4, null], [-1.3, 17, null]]]}}),
        json!({"meta_info": {"finish_reason": {"type": "length"}, "output_token_ids_logprobs": [[[-0.2, 17, null]]]}}),
        json!({"meta_info": {"finish_reason": {"type": "error"}, "output_token_ids_logprobs": [[[-0.2, 17, null]]]}}),
        json!({"meta_info": {"finish_reason": {}, "output_token_ids_logprobs": [[[-0.2, 17, null]]]}}),
    ] {
        assert!(parse_candidate_scores(&response, &[17, 4]).is_err());
    }
}

#[test]
fn builds_zero_decode_native_sglang_score_request() {
    let request = build_native_score_request(&[1, 2, 3], &[17, 4], "request-salt").unwrap();
    assert_eq!(request["input_ids"], json!([1, 2, 3]));
    assert_eq!(
        request["sampling_params"],
        json!({
            "max_new_tokens": 0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "n": 1
        })
    );
    assert_eq!(request["token_ids_logprob"], json!([17, 4]));
    assert_eq!(request["return_logprob"], true);
    assert_eq!(request["return_text_in_logprobs"], false);
    assert_eq!(request["cache_salt"], "request-salt");
    assert_eq!(request["stream"], true);

    assert!(build_native_score_request(&[], &[17], "salt").is_err());
    assert!(build_native_score_request(&[1], &[17, 17], "salt").is_err());
    assert!(build_native_score_request(&[1], &[17], "").is_err());
}
