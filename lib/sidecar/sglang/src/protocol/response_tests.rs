// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_backend_common::{BackendError, ErrorType, FinishReason};
use serde_json::json;

fn assert_protocol_error(error: DynamoError) {
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::Unknown)
    );
}

#[test]
fn malformed_terminal_is_rejected() {
    assert!(terminal_from_meta(&HashMap::new(), 4, 0, &StopConditions::default()).is_err());
    let meta = HashMap::from([(
        "finish_reason".to_string(),
        json!({"type": "mystery"}).to_string(),
    )]);
    assert!(terminal_from_meta(&meta, 4, 0, &StopConditions::default()).is_err());
}

#[test]
fn terminal_engine_data_handles_prompt_logprob_encodings() {
    let meta = HashMap::from([
        (
            "input_token_logprobs".to_string(),
            json!([[null, 10, null], [-0.2, 11, "b"]]).to_string(),
        ),
        (
            "input_top_logprobs".to_string(),
            json!([null, [[-0.3, 12, "c"]]]).to_string(),
        ),
        ("routed_experts".to_string(), json!([1, 2]).to_string()),
    ]);
    let data = engine_data_from_meta(&meta, true).unwrap().unwrap();
    let prompt = data["prompt_logprobs"].as_array().unwrap();
    assert!(prompt[0].is_null());
    assert_eq!(prompt[1]["11"]["logprob"], json!(-0.2));
    assert_eq!(prompt[1]["12"]["decoded_token"], json!("c"));
    assert_eq!(data["routed_experts"], json!([1, 2]));

    let legacy = HashMap::from([
        (
            "input_token_logprobs".to_string(),
            json!([[-0.1, 10, "a"], [-0.2, 11, "b"]]).to_string(),
        ),
        (
            "input_top_logprobs".to_string(),
            json!([[[-0.3, 12, "c"]], []]).to_string(),
        ),
    ]);
    let data = engine_data_from_meta(&legacy, true).unwrap().unwrap();
    let prompt = data["prompt_logprobs"].as_array().unwrap();
    assert!(prompt[0].is_null());
    assert_eq!(prompt[1]["10"]["logprob"], json!(-0.1));
    assert_eq!(prompt[1]["12"]["decoded_token"], json!("c"));

    let mismatched = HashMap::from([
        (
            "input_token_logprobs".to_string(),
            json!([[null, 10, null], [-0.2, 11, "b"]]).to_string(),
        ),
        (
            "input_top_logprobs".to_string(),
            json!([[[-0.3, 12, "c"]], []]).to_string(),
        ),
    ]);
    assert!(engine_data_from_meta(&mismatched, true).is_err());
}

#[test]
fn prompt_logprobs_are_terminal_only() {
    let meta = HashMap::from([(
        "input_token_logprobs".to_string(),
        json!([[-0.1, 10, "a"]]).to_string(),
    )]);
    assert!(engine_data_from_meta(&meta, false).unwrap().is_none());
}

#[test]
fn output_ids_preserve_order_and_reject_negative_tokens() {
    assert_eq!(output_ids_to_u32(&[]).unwrap(), Vec::<u32>::new());
    assert_eq!(
        output_ids_to_u32(&[0, 11, i32::MAX]).unwrap(),
        vec![0, 11, i32::MAX as u32]
    );
    let error = output_ids_to_u32(&[11, -1, 12]).unwrap_err();
    assert!(error.to_string().contains("negative token id: -1"));
    assert_protocol_error(error);
}

#[test]
fn token_count_metadata_accepts_only_unsigned_in_range_integers() {
    assert_eq!(meta_u32(&HashMap::new(), "prompt_tokens"), None);
    for (raw, expected) in [
        ("0", Some(0)),
        ("4294967295", Some(u32::MAX)),
        ("4294967296", None),
        ("-1", None),
        ("1.5", None),
        ("\"3\"", None),
        ("null", None),
        ("invalid", None),
    ] {
        let meta = HashMap::from([("prompt_tokens".to_string(), raw.to_string())]);
        assert_eq!(meta_u32(&meta, "prompt_tokens"), expected, "{raw}");
    }
}

#[test]
fn object_and_scalar_finish_reasons_preserve_usage_and_user_string() {
    for (finish, expected, stop) in [
        (json!("stop"), FinishReason::Stop, None),
        (
            json!({"type": "stop", "matched": "end"}),
            FinishReason::Stop,
            Some(StopReason::String("end".to_string())),
        ),
        (json!("length"), FinishReason::Length, None),
        (json!({"type": "length"}), FinishReason::Length, None),
        (json!({"type": "cancelled"}), FinishReason::Cancelled, None),
    ] {
        let meta = HashMap::from([("finish_reason".to_string(), finish.to_string())]);
        let terminal = terminal_from_meta(&meta, 4, 3, &StopConditions::default()).unwrap();
        assert_eq!(terminal.finish_reason, Some(expected));
        assert_eq!(terminal.stop_reason, stop);
        assert!(terminal.token_ids.is_empty());
        let usage = terminal.completion_usage.unwrap();
        assert_eq!(usage.prompt_tokens, 4);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, 7);
    }
}

#[test]
fn terminal_errors_preserve_backend_category_and_details() {
    for (status, expected) in [
        (Some(400), BackendError::InvalidArgument),
        (Some(499), BackendError::InvalidArgument),
        (Some(500), BackendError::Unknown),
        (None, BackendError::Unknown),
    ] {
        for kind in ["abort", "error"] {
            let meta = HashMap::from([(
                "finish_reason".to_string(),
                json!({
                    "type": kind, "message": "generation failed", "status_code": status,
                    "err_type": "NativeError",
                })
                .to_string(),
            )]);
            let error = terminal_from_meta(&meta, 4, 3, &StopConditions::default()).unwrap_err();
            assert_eq!(error.error_type(), ErrorType::Backend(expected));
            let detail = error.to_string();
            assert!(detail.contains(kind));
            assert!(detail.contains("generation failed"));
            assert!(detail.contains("NativeError"));
            assert!(detail.contains(&format!(
                "status_code={}",
                status.map_or("unknown".to_string(), |value| value.to_string())
            )));
        }
    }
    let error = terminal_failure("abort", &json!({}));
    assert!(error.to_string().contains("SGLang generation failed"));
    assert!(error.to_string().contains("err_type=unknown"));
    assert_protocol_error(error);
}

#[test]
fn malformed_terminal_metadata_has_protocol_error_category() {
    for raw in ["invalid", "null", "{}", "{\"type\":12}", "\"unknown\""] {
        let meta = HashMap::from([("finish_reason".to_string(), raw.to_string())]);
        assert_protocol_error(
            terminal_from_meta(&meta, 4, 0, &StopConditions::default()).unwrap_err(),
        );
    }
}

#[test]
fn selected_and_top_logprobs_preserve_positions_ranks_and_token_rendering() {
    let meta = HashMap::from([
        (
            "output_token_logprobs".to_string(),
            json!([[-0.1, 10, "a"], [-0.2, 11, null]]).to_string(),
        ),
        (
            "output_top_logprobs".to_string(),
            json!([[[-0.1, 10, "a"], [-0.3, 12, "c"]], [[-0.2, 11, null]]]).to_string(),
        ),
    ]);
    for as_ids in [false, true] {
        let (selected, top) = extract_logprobs(&meta, as_ids).unwrap();
        assert_eq!(selected.unwrap(), vec![-0.1, -0.2]);
        let top = top.unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].len(), 2);
        assert_eq!(top[1].len(), 1);
        for (entry, rank, id, logprob, text) in [
            (&top[0][0], 1, 10, -0.1, Some("a")),
            (&top[0][1], 2, 12, -0.3, Some("c")),
            (&top[1][0], 1, 11, -0.2, None),
        ] {
            assert_eq!(entry.rank, rank);
            assert_eq!(entry.token_id, id);
            assert_eq!(entry.logprob, logprob);
            assert_eq!(
                entry.token,
                if as_ids {
                    Some(format!("token_id:{id}"))
                } else {
                    text.map(str::to_string)
                }
            );
            assert!(entry.bytes.is_none());
        }
    }
}

#[test]
fn absent_and_selected_only_logprobs_do_not_synthesize_candidates() {
    let (selected, top) = extract_logprobs(&HashMap::new(), false).unwrap();
    assert!(selected.is_none());
    assert!(top.is_none());
    let meta = HashMap::from([(
        "output_token_logprobs".to_string(),
        json!([[-0.1, 10, "a"]]).to_string(),
    )]);
    let (selected, top) = extract_logprobs(&meta, false).unwrap();
    assert_eq!(selected.unwrap(), vec![-0.1]);
    assert!(top.is_none());
}

#[test]
fn malformed_selected_and_top_logprob_entries_are_rejected() {
    for selected in [
        json!([null]),
        json!([[]]),
        json!([[null, 10, "a"]]),
        json!([["bad", 10, "a"]]),
    ] {
        let meta = HashMap::from([("output_token_logprobs".to_string(), selected.to_string())]);
        assert_protocol_error(extract_logprobs(&meta, false).unwrap_err());
    }
    for top in [
        json!([[null]]),
        json!([[[]]]),
        json!([[[null, 10, "a"]]]),
        json!([[[-0.1]]]),
        json!([[[-0.1, -1, "a"]]]),
        json!([[[-0.1, u64::from(u32::MAX) + 1, "a"]]]),
    ] {
        let meta = HashMap::from([
            (
                "output_token_logprobs".to_string(),
                json!([[-0.1, 10, "a"]]).to_string(),
            ),
            ("output_top_logprobs".to_string(), top.to_string()),
        ]);
        assert_protocol_error(extract_logprobs(&meta, false).unwrap_err());
    }
}

#[test]
fn prompt_selected_token_wins_duplicate_candidate_and_accepts_null_sentinel() {
    let meta = HashMap::from([
        (
            "input_token_logprobs".to_string(),
            json!([null, [-0.2, 11, "b"]]).to_string(),
        ),
        (
            "input_top_logprobs".to_string(),
            json!([null, [[-0.9, 11, "wrong"], [-0.3, 12, "c"]]]).to_string(),
        ),
    ]);
    assert_eq!(
        engine_data_from_meta(&meta, true).unwrap(),
        Some(json!({
            "prompt_logprobs": [null, {
                "11": {"logprob": -0.2, "decoded_token": "b"},
                "12": {"logprob": -0.3, "decoded_token": "c"},
            }],
        }))
    );
}

#[test]
fn prompt_metadata_rejects_misaligned_or_malformed_positions() {
    for (selected, top) in [
        (json!([null, [-0.2, 11, "b"]]), json!([null])),
        (json!([null, []]), json!([null, []])),
        (json!([null, [-0.2]]), json!([null, []])),
        (json!([null, [-0.2, 11, "b"]]), json!([null, [null]])),
    ] {
        let meta = HashMap::from([
            ("input_token_logprobs".to_string(), selected.to_string()),
            ("input_top_logprobs".to_string(), top.to_string()),
        ]);
        assert_protocol_error(engine_data_from_meta(&meta, true).unwrap_err());
    }
}

#[test]
fn prompt_opt_in_does_not_suppress_independent_routed_experts() {
    let mut meta = HashMap::from([
        ("input_token_logprobs".to_string(), json!([[]]).to_string()),
        ("routed_experts".to_string(), json!([[1, 2]]).to_string()),
    ]);
    assert_eq!(
        engine_data_from_meta(&meta, false).unwrap(),
        Some(json!({"routed_experts": [[1, 2]]}))
    );
    assert!(engine_data_from_meta(&meta, true).is_err());
    meta.insert("input_token_logprobs".to_string(), "[]".to_string());
    assert_eq!(
        engine_data_from_meta(&meta, true).unwrap(),
        Some(json!({"routed_experts": [[1, 2]]}))
    );
    assert!(
        engine_data_from_meta(&HashMap::new(), true)
            .unwrap()
            .is_none()
    );
}

#[test]
fn user_stop_tokens_are_preserved_and_system_stop_tokens_are_hidden() {
    let meta = HashMap::from([(
        "finish_reason".to_string(),
        json!({"type": "stop", "matched": 128001}).to_string(),
    )]);
    for (user_ids, hidden_ids, expected) in [
        (None, None, None),
        (Some(vec![576]), None, None),
        (None, Some(vec![128001]), None),
        (Some(vec![128001]), None, Some(StopReason::Int(128001))),
        (
            Some(vec![128001]),
            Some(vec![128001]),
            Some(StopReason::Int(128001)),
        ),
    ] {
        let stop_conditions = StopConditions {
            stop_token_ids: user_ids,
            stop_token_ids_hidden: hidden_ids,
            ..Default::default()
        };
        let terminal = terminal_from_meta(&meta, 4, 3, &stop_conditions).unwrap();
        assert_eq!(terminal.stop_reason, expected);
        assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    }
}
