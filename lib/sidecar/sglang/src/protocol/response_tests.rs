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

fn entry(logprob: Option<f64>, token_id: i64, text: Option<&str>) -> pb::LogprobEntry {
    pb::LogprobEntry {
        logprob,
        token_id,
        text: text.map(str::to_string),
    }
}

fn entries(entries: Vec<pb::LogprobEntry>) -> pb::LogprobEntries {
    pb::LogprobEntries { entries }
}

fn row(entries: Option<Vec<pb::LogprobEntry>>) -> pb::NullableTopLogprobs {
    pb::NullableTopLogprobs {
        row: entries.map(|entries| pb::TopLogprobRow { entries }),
    }
}

fn meta_with_finish(kind: FinishKind) -> pb::GenerateMetaInfo {
    pb::GenerateMetaInfo {
        finish_reason: Some(pb::FinishReason { kind: Some(kind) }),
        ..Default::default()
    }
}

fn unknown(raw: &str) -> pb::GenerateMetaInfo {
    meta_with_finish(FinishKind::Unknown(raw.to_string()))
}

fn frame(tokens: &[i64], meta: pb::GenerateMetaInfo) -> pb::GenerateResponse {
    pb::GenerateResponse {
        output_ids: (!tokens.is_empty()).then(|| pb::TokenIds {
            ids: tokens.to_vec(),
        }),
        meta_info: Some(meta),
        ..Default::default()
    }
}

fn length() -> FinishKind {
    FinishKind::Length(pb::FinishLength { length: None })
}

fn stream_error(code: u32, message: &str) -> pb::GenerateStreamError {
    pb::GenerateStreamError {
        error: Some(pb::ErrorBody {
            message: message.to_string(),
            code,
        }),
        index: None,
    }
}

fn converter() -> ResponseConverter {
    ResponseConverter::new(false, false, StopConditions::default(), 4)
}

#[test]
fn malformed_terminal_is_rejected() {
    let stop = StopConditions::default();
    let error = terminal_from_meta(&pb::GenerateMetaInfo::default(), 4, 0, &stop).unwrap_err();
    assert!(error.to_string().contains("missing finish_reason"));
    let missing_kind = pb::GenerateMetaInfo {
        finish_reason: Some(pb::FinishReason { kind: None }),
        ..Default::default()
    };
    let error = terminal_from_meta(&missing_kind, 4, 0, &stop).unwrap_err();
    assert!(error.to_string().contains("missing a type"));
    for raw in ["invalid", "{}", r#"{"type": "mystery"}"#] {
        assert_protocol_error(terminal_from_meta(&unknown(raw), 4, 0, &stop).unwrap_err());
    }
}

#[test]
fn typed_finish_reasons_preserve_usage_and_user_string() {
    for (meta, expected, stop) in [
        (
            meta_with_finish(FinishKind::Stop(pb::FinishStop { matched: None })),
            FinishReason::Stop,
            None,
        ),
        (
            meta_with_finish(FinishKind::Stop(pb::FinishStop {
                matched: Some(pb::Matched {
                    value: Some(pb::matched::Value::Str("end".to_string())),
                }),
            })),
            FinishReason::Stop,
            Some(StopReason::String("end".to_string())),
        ),
        (
            meta_with_finish(FinishKind::Stop(pb::FinishStop {
                matched: Some(pb::Matched {
                    value: Some(pb::matched::Value::Tokens(pb::MatchedTokens {
                        ids: vec![1, 2],
                    })),
                }),
            })),
            FinishReason::Stop,
            None,
        ),
        (meta_with_finish(length()), FinishReason::Length, None),
        (
            unknown(r#"{"type": "cancelled"}"#),
            FinishReason::Cancelled,
            None,
        ),
    ] {
        let terminal = terminal_from_meta(&meta, 4, 3, &StopConditions::default()).unwrap();
        assert_eq!(terminal.finish_reason, Some(expected));
        assert_eq!(terminal.stop_reason, stop);
        assert!(terminal.token_ids.is_empty());
        let usage = terminal.completion_usage.unwrap();
        assert_eq!(usage.prompt_tokens, 4);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, 7);
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, Some(0));
    }
}

#[test]
fn user_stop_tokens_are_preserved_and_system_stop_tokens_are_hidden() {
    let meta = meta_with_finish(FinishKind::Stop(pb::FinishStop {
        matched: Some(pb::Matched {
            value: Some(pb::matched::Value::Token(128001)),
        }),
    }));
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

#[test]
fn terminal_errors_preserve_backend_category_and_details() {
    for (status, expected) in [
        (Some(400), BackendError::InvalidArgument),
        (Some(499), BackendError::InvalidArgument),
        (Some(500), BackendError::Unknown),
        (None, BackendError::Unknown),
    ] {
        let abort = meta_with_finish(FinishKind::Abort(pb::FinishAbort {
            message: Some("generation failed".into()),
            status_code: status,
            err_type: Some("NativeError".into()),
        }));
        let error_type = unknown(
            &json!({
                "type": "error", "message": "generation failed", "status_code": status,
                "err_type": "NativeError",
            })
            .to_string(),
        );
        for (kind, meta) in [("abort", abort), ("error", error_type)] {
            let error = terminal_from_meta(&meta, 4, 3, &StopConditions::default()).unwrap_err();
            assert_eq!(error.error_type(), ErrorType::Backend(expected));
            let detail = error.to_string();
            assert!(
                detail.contains(&format!("SGLang generation {kind}")),
                "{detail}"
            );
            assert!(detail.contains("generation failed"));
            assert!(detail.contains("NativeError"));
            assert!(detail.contains(&format!(
                "status_code={}",
                status.map_or("unknown".to_string(), |value| value.to_string())
            )));
        }
    }
    let error = terminal_from_meta(
        &meta_with_finish(FinishKind::Abort(pb::FinishAbort::default())),
        4,
        0,
        &StopConditions::default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("SGLang generation failed"));
    assert!(error.to_string().contains("err_type=unknown"));
    assert_protocol_error(error);
}

#[test]
fn output_ids_preserve_order_and_reject_out_of_range_tokens() {
    assert_eq!(output_ids_to_u32(&[]).unwrap(), Vec::<u32>::new());
    assert_eq!(
        output_ids_to_u32(&[0, 11, i64::from(u32::MAX)]).unwrap(),
        vec![0, 11, u32::MAX]
    );
    let error = output_ids_to_u32(&[11, -1, 12]).unwrap_err();
    assert!(error.to_string().contains("negative token id: -1"));
    assert_protocol_error(error);
    let error = output_ids_to_u32(&[i64::from(u32::MAX) + 1]).unwrap_err();
    assert!(error.to_string().contains("does not fit u32"));
    assert_protocol_error(error);
}

#[test]
fn frames_without_output_ids_are_skipped_until_the_terminal() {
    let mut converter = converter();
    assert!(matches!(
        converter
            .frame(frame(&[], pb::GenerateMetaInfo::default()))
            .unwrap(),
        FrameOutput::Skip
    ));
    let FrameOutput::Delta(delta) = converter
        .frame(frame(&[7, 8], pb::GenerateMetaInfo::default()))
        .unwrap()
    else {
        panic!("expected a delta");
    };
    assert_eq!(delta.token_ids, [7, 8]);
    assert!(delta.engine_data.is_none());
    let FrameOutput::Terminal(terminal) = converter
        .frame(frame(&[], meta_with_finish(length())))
        .unwrap()
    else {
        panic!("expected a terminal");
    };
    assert!(terminal.token_ids.is_empty());
    assert!(terminal.engine_data.is_none());
    let usage = terminal.completion_usage.unwrap();
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (4, 2));
}

#[test]
fn reported_prompt_tokens_replace_the_request_count() {
    let mut converter = converter();
    // Zero is proto3's unset value; the request's own count stands.
    converter
        .frame(frame(&[7], pb::GenerateMetaInfo::default()))
        .unwrap();
    assert_eq!(converter.usage().prompt_tokens, 4);
    // SGLang may truncate the prompt and reports the length it used.
    let meta = pb::GenerateMetaInfo {
        prompt_tokens: 2,
        cached_tokens: 9,
        ..meta_with_finish(length())
    };
    let FrameOutput::Terminal(terminal) = converter.frame(frame(&[8], meta)).unwrap() else {
        panic!("expected a terminal");
    };
    let usage = terminal.completion_usage.unwrap();
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (2, 2));
    assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, Some(2));
}

#[test]
fn prompt_logprobs_from_the_first_frame_attach_to_the_terminal_only() {
    // Rust incremental streams send input logprobs once, on the first frame,
    // and empty lists afterwards.
    let mut converter = ResponseConverter::new(false, false, StopConditions::default(), 3);
    let first = pb::GenerateMetaInfo {
        prompt_tokens: 3,
        output_token_logprobs: Some(entries(vec![entry(Some(-0.5), 101, None)])),
        input_token_logprobs: Some(entries(vec![
            entry(None, 1, None),
            entry(Some(-0.1), 2, None),
            entry(Some(-0.2), 3, None),
        ])),
        input_top_logprobs: vec![
            row(None),
            row(Some(vec![entry(Some(-0.4), 9, None)])),
            row(Some(vec![])),
        ],
        ..Default::default()
    };
    let FrameOutput::Delta(delta) = converter.frame(frame(&[101], first)).unwrap() else {
        panic!("expected a delta");
    };
    assert!(delta.engine_data.is_none());

    let later = pb::GenerateMetaInfo {
        output_token_logprobs: Some(entries(vec![entry(Some(-0.6), 102, None)])),
        input_token_logprobs: Some(entries(Vec::new())),
        ..Default::default()
    };
    let FrameOutput::Delta(delta) = converter.frame(frame(&[102], later)).unwrap() else {
        panic!("expected a delta");
    };
    assert!(delta.engine_data.is_none());

    let terminal = pb::GenerateMetaInfo {
        output_token_logprobs: Some(entries(vec![entry(Some(-0.7), 103, None)])),
        input_token_logprobs: Some(entries(Vec::new())),
        ..meta_with_finish(length())
    };
    let FrameOutput::Terminal(terminal) = converter.frame(frame(&[103], terminal)).unwrap() else {
        panic!("expected a terminal");
    };
    assert_eq!(terminal.token_ids, [103]);
    assert_eq!(terminal.log_probs, Some(vec![-0.7]));
    assert_eq!(
        terminal.engine_data,
        Some(json!({"prompt_logprobs": [
            null,
            {"2": {"logprob": -0.1}, "9": {"logprob": -0.4}},
            {"3": {"logprob": -0.2}},
        ]}))
    );
    let usage = terminal.completion_usage.unwrap();
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (3, 3));
}

#[test]
fn terminal_engine_data_handles_prompt_logprob_encodings() {
    let native = prompt_logprobs_value(
        &[entry(None, 10, None), entry(Some(-0.2), 11, Some("b"))],
        &[row(None), row(Some(vec![entry(Some(-0.3), 12, Some("c"))]))],
    )
    .unwrap()
    .unwrap();
    let prompt = native.as_array().unwrap();
    assert_eq!(prompt.len(), 2);
    assert!(prompt[0].is_null());
    assert_eq!(prompt[1]["11"]["logprob"], json!(-0.2));
    assert_eq!(prompt[1]["12"]["decoded_token"], json!("c"));

    // Older releases omitted the first prompt position.
    let legacy = prompt_logprobs_value(
        &[
            entry(Some(-0.1), 10, Some("a")),
            entry(Some(-0.2), 11, Some("b")),
        ],
        &[
            row(Some(vec![entry(Some(-0.3), 12, Some("c"))])),
            row(Some(vec![])),
        ],
    )
    .unwrap()
    .unwrap();
    let prompt = legacy.as_array().unwrap();
    assert_eq!(prompt.len(), 3);
    assert!(prompt[0].is_null());
    assert_eq!(prompt[1]["10"]["logprob"], json!(-0.1));
    assert_eq!(prompt[1]["12"]["decoded_token"], json!("c"));
    assert!(prompt_logprobs_value(&[], &[]).unwrap().is_none());

    // The selected token wins a duplicate candidate.
    assert_eq!(
        prompt_logprobs_value(
            &[entry(None, 10, None), entry(Some(-0.2), 11, Some("b"))],
            &[
                row(None),
                row(Some(vec![
                    entry(Some(-0.9), 11, Some("wrong")),
                    entry(Some(-0.3), 12, Some("c")),
                ])),
            ],
        )
        .unwrap(),
        Some(json!([null, {
            "11": {"logprob": -0.2, "decoded_token": "b"},
            "12": {"logprob": -0.3, "decoded_token": "c"},
        }]))
    );

    for (selected, top) in [
        (
            vec![entry(None, 10, None), entry(Some(-0.2), 11, Some("b"))],
            vec![row(None)],
        ),
        (
            vec![entry(None, 10, None), entry(Some(-0.2), 11, Some("b"))],
            vec![
                row(Some(vec![entry(Some(-0.3), 12, Some("c"))])),
                row(Some(vec![])),
            ],
        ),
        (
            vec![entry(None, 10, None), entry(None, 11, Some("b"))],
            vec![row(None), row(Some(vec![]))],
        ),
        (
            vec![entry(None, 10, None), entry(Some(-0.2), 11, Some("b"))],
            vec![row(None), row(Some(vec![entry(None, 12, None)]))],
        ),
    ] {
        assert_protocol_error(prompt_logprobs_value(&selected, &top).unwrap_err());
    }
}

#[test]
fn prefill_reports_only_its_terminal_without_tokens() {
    let mut converter = ResponseConverter::new(true, false, StopConditions::default(), 4);
    assert!(matches!(
        converter
            .frame(frame(&[7], pb::GenerateMetaInfo::default()))
            .unwrap(),
        FrameOutput::Skip
    ));
    let FrameOutput::Terminal(terminal) = converter
        .frame(frame(&[8], meta_with_finish(length())))
        .unwrap()
    else {
        panic!("expected a terminal");
    };
    assert!(terminal.token_ids.is_empty());
    assert!(terminal.engine_data.is_none());
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
    let usage = terminal.completion_usage.unwrap();
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (4, 0));
}

#[test]
fn stream_error_items_map_http_codes_to_backend_errors() {
    for (code, expected) in [
        (400, BackendError::InvalidArgument),
        // A deliberate engine rejection, not a transport failure to retry.
        (429, BackendError::InvalidArgument),
        (499, BackendError::Cancelled),
        (503, BackendError::Unknown),
    ] {
        let error = stream_error_to_dynamo(&stream_error(code, "native failure"));
        assert_eq!(error.error_type(), ErrorType::Backend(expected), "{code}");
        let detail = error.to_string();
        assert!(detail.contains("Generate: native failure"), "{detail}");
        assert!(detail.contains(&format!("code {code}")), "{detail}");
        // Engine messages are not echoed to the client.
        assert_eq!(error.public_message(), None);
    }
    assert_protocol_error(stream_error_to_dynamo(&pb::GenerateStreamError::default()));
}

#[test]
fn requested_cancellation_ends_a_499_stream_as_cancelled() {
    let mut converter = converter();
    converter
        .frame(frame(&[7, 8], pb::GenerateMetaInfo::default()))
        .unwrap();
    let cancelled = stream_error(499, "client disconnected");
    let output = converter.stream_error(&cancelled, true).unwrap();
    assert_eq!(output.finish_reason, Some(FinishReason::Cancelled));
    let usage = output.completion_usage.unwrap();
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (4, 2));

    // An abort the sidecar did not request is an error, as is any other code
    // after a cancellation.
    let error = converter.stream_error(&cancelled, false).unwrap_err();
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::Cancelled)
    );
    let error = converter
        .stream_error(&stream_error(503, "overloaded"), true)
        .unwrap_err();
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::Unknown)
    );
}

#[test]
fn selected_and_top_logprobs_preserve_positions_ranks_and_token_rendering() {
    let meta = pb::GenerateMetaInfo {
        output_token_logprobs: Some(entries(vec![
            entry(Some(-0.1), 10, Some("a")),
            entry(Some(-0.2), 11, None),
        ])),
        output_top_logprobs: vec![
            row(Some(vec![
                entry(Some(-0.1), 10, Some("a")),
                entry(Some(-0.3), 12, Some("c")),
            ])),
            row(Some(vec![entry(Some(-0.2), 11, None)])),
            row(None),
        ],
        ..Default::default()
    };
    for as_ids in [false, true] {
        let (selected, top) = extract_logprobs(&meta, as_ids).unwrap();
        assert_eq!(selected.unwrap(), vec![-0.1, -0.2]);
        let top = top.unwrap();
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].len(), 2);
        assert_eq!(top[1].len(), 1);
        // A position with no alternatives is an empty list.
        assert!(top[2].is_empty());
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
    let (selected, top) = extract_logprobs(&pb::GenerateMetaInfo::default(), false).unwrap();
    assert!(selected.is_none());
    assert!(top.is_none());
    // A request without top-k gets an empty alternatives list on every frame.
    let meta = pb::GenerateMetaInfo {
        output_token_logprobs: Some(entries(vec![entry(Some(-0.1), 10, Some("a"))])),
        ..Default::default()
    };
    let (selected, top) = extract_logprobs(&meta, false).unwrap();
    assert_eq!(selected.unwrap(), vec![-0.1]);
    assert!(top.is_none());
}

#[test]
fn malformed_selected_and_top_logprob_entries_are_rejected() {
    let meta = pb::GenerateMetaInfo {
        output_token_logprobs: Some(entries(vec![entry(None, 10, Some("a"))])),
        ..Default::default()
    };
    assert_protocol_error(extract_logprobs(&meta, false).unwrap_err());
    for top in [
        entry(None, 10, Some("a")),
        entry(Some(-0.1), -1, Some("a")),
        entry(Some(-0.1), i64::from(u32::MAX) + 1, Some("a")),
    ] {
        let meta = pb::GenerateMetaInfo {
            output_token_logprobs: Some(entries(vec![entry(Some(-0.1), 10, Some("a"))])),
            output_top_logprobs: vec![row(Some(vec![top]))],
            ..Default::default()
        };
        assert_protocol_error(extract_logprobs(&meta, false).unwrap_err());
    }
}
