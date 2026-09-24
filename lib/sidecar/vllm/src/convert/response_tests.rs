// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::unit_fixtures::minimal_request;
use crate::unit_vllm_fixtures::*;
use dynamo_backend_common::{FinishReason, StopReason};
use serde_json::json;

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn encode_response_enforces_terminal_contract() {
        let request = epd_image_request();
        let ec_transfer_params = || json_to_struct(encoder_handoff()).expect("encoder handoff");

        let mut length = encode_response(Some(ec_transfer_params()));
        length
            .outputs
            .as_mut()
            .and_then(|output| output.finish_info.as_mut())
            .expect("finish info")
            .finish_reason = pb::finish_info::FinishReason::Length as i32;
        let error = ResponseState::new(&request, DisaggregationMode::Encode)
            .convert(length)
            .expect_err("Length must not become a successful encoder handoff");
        assert!(error.to_string().contains("invalid finish reason"));

        let mut token_producing = encode_response(Some(ec_transfer_params()));
        let output = token_producing.outputs.as_mut().expect("sequence output");
        output.text = "unexpected".to_string();
        output.num_tokens = 1;
        output.token_ids = vec![42];
        output
            .finish_info
            .as_mut()
            .expect("finish info")
            .num_output_tokens = 1;
        let error = ResponseState::new(&request, DisaggregationMode::Encode)
            .convert(token_producing)
            .expect_err("Encode must remain tokenless");
        assert!(error.to_string().contains("produced output tokens"));

        let mut cancelled = encode_response(None);
        cancelled
            .outputs
            .as_mut()
            .and_then(|output| output.finish_info.as_mut())
            .expect("finish info")
            .finish_reason = pb::finish_info::FinishReason::Aborted as i32;
        let terminal = ResponseState::new(&request, DisaggregationMode::Encode)
            .convert(cancelled)
            .expect("cancelled response")
            .expect("cancelled terminal");
        assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
        assert!(terminal.encoder_result.is_none());
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn early_prompt_frames_retain_exact_native_metadata() {
        for (has_prompt_logprobs, has_output) in [(false, false), (true, false), (true, true)] {
            let mut request = request();
            request.output_options.prompt_logprobs = has_prompt_logprobs.then_some(1);
            let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
            let prompt = pb::PromptInfo {
                num_prompt_tokens: 3,
                token_ids: vec![11, 22, 33],
                logprobs: vec![f32::NEG_INFINITY, -0.25, -0.5],
                ranks: vec![0, 1, 2],
                candidate_tokens: vec![
                    pb::CandidateTokenInfo::default(),
                    pb::CandidateTokenInfo {
                        tokens: vec![pb::candidate_token_info::TokenInfo {
                            id: 23,
                            logprob: -0.75,
                            rank: 2,
                        }],
                    },
                    pb::CandidateTokenInfo::default(),
                ],
            };
            let first = state
                .convert(pb::GenerateResponse {
                    prompt_info: Some(prompt),
                    outputs: has_output
                        .then(|| sequence_response(false, true, None).outputs.unwrap()),
                })
                .unwrap();
            if has_output {
                let first = first.unwrap();
                assert!(first.finish_reason.is_none());
                assert!(first.engine_data.is_none());
            } else {
                assert!(first.is_none());
            }
            let mut response = sequence_response(true, true, None);
            response
                .outputs
                .as_mut()
                .unwrap()
                .finish_info
                .as_mut()
                .unwrap()
                .num_output_tokens = 1 + u32::from(has_output);
            let result = state.convert(response).unwrap().unwrap();
            assert!(result.finish_reason.is_some());
            assert_eq!(result.engine_data, has_prompt_logprobs.then(|| json!({"prompt_logprobs": [null, {"22": {"logprob": -0.25, "rank": 1}, "23": {"logprob": -0.75, "rank": 2}}, {"33": {"logprob": -0.5, "rank": 2}}]})));
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn negative_infinity_logprobs_are_normalized() {
        let request = request();
        let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
        let mut response = sequence_response(true, true, None);
        response.prompt_info = Some(pb::PromptInfo {
            num_prompt_tokens: 3,
            token_ids: vec![11, 22, 33],
            logprobs: vec![0.0, f32::NEG_INFINITY, -0.3],
            ranks: vec![0, 1, 2],
            candidate_tokens: vec![
                pb::CandidateTokenInfo::default(),
                pb::CandidateTokenInfo {
                    tokens: vec![pb::candidate_token_info::TokenInfo {
                        id: 23,
                        logprob: f32::NEG_INFINITY,
                        rank: 2,
                    }],
                },
                pb::CandidateTokenInfo::default(),
            ],
        });
        let output = response.outputs.as_mut().unwrap();
        output.logprobs[0] = f32::NEG_INFINITY;
        output.candidate_tokens[0].tokens[0].logprob = f32::NEG_INFINITY;

        let mapped = state
            .convert(response)
            .expect("convert response")
            .expect("terminal output");
        assert_eq!(mapped.log_probs.as_deref(), Some(&[-9999.0][..]));
        assert!(
            mapped.top_logprobs.as_ref().unwrap()[0]
                .iter()
                .all(|entry| entry.logprob == -9999.0)
        );
        let prompt = &mapped.engine_data.as_ref().unwrap()["prompt_logprobs"][1];
        assert_eq!(prompt["22"]["logprob"], json!(-9999.0));
        assert_eq!(prompt["23"]["logprob"], json!(-9999.0));
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn missing_outputs_and_buffered_text_preserve_native_semantics() {
        let mut request = request();
        request.output_options = Default::default();
        let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
        assert!(
            state
                .convert(pb::GenerateResponse::default())
                .unwrap()
                .is_none()
        );
        let first = state
            .convert(sequence_response(false, false, None))
            .unwrap()
            .unwrap();
        assert_eq!(first.text.as_deref(), Some(" token"));
        let mut response = sequence_response(true, false, None);
        let output = response.outputs.as_mut().unwrap();
        output.token_ids = vec![43, 44];
        output.num_tokens = 2;
        output.text = String::new();
        output.finish_info.as_mut().unwrap().num_output_tokens = 3;
        let terminal = state.convert(response).unwrap().unwrap();
        assert_eq!(terminal.text.as_deref(), Some(""));
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn malformed_responses_return_typed_protocol_errors() {
        type ResponseMutation = fn(&mut pb::GenerateResponse);
        let cases: &[(&str, ResponseMutation)] = &[
            ("sequence index", |r| r.outputs.as_mut().unwrap().index = 1),
            ("num_tokens", |r| r.outputs.as_mut().unwrap().num_tokens = 2),
            ("terminal num_output_tokens", |r| {
                r.outputs
                    .as_mut()
                    .unwrap()
                    .finish_info
                    .as_mut()
                    .unwrap()
                    .num_output_tokens = 2
            }),
            ("unknown finish reason", |r| {
                r.outputs
                    .as_mut()
                    .unwrap()
                    .finish_info
                    .as_mut()
                    .unwrap()
                    .finish_reason = 42
            }),
            ("NOT_FINISHED", |r| {
                r.outputs
                    .as_mut()
                    .unwrap()
                    .finish_info
                    .as_mut()
                    .unwrap()
                    .finish_reason = pb::finish_info::FinishReason::NotFinished as i32
            }),
            ("output logprob array lengths", |r| {
                r.outputs.as_mut().unwrap().logprobs.clear()
            }),
            ("output logprob array lengths", |r| {
                r.outputs.as_mut().unwrap().ranks.clear()
            }),
            ("output logprob array lengths", |r| {
                r.outputs.as_mut().unwrap().candidate_tokens.clear()
            }),
            ("prompt token count", |r| {
                r.prompt_info = Some(pb::PromptInfo {
                    num_prompt_tokens: 4,
                    ..Default::default()
                })
            }),
            ("prompt logprob array lengths", |r| {
                r.prompt_info = Some(pb::PromptInfo {
                    num_prompt_tokens: 3,
                    token_ids: vec![11, 22, 33],
                    logprobs: vec![0.0; 3],
                    ranks: vec![0; 3],
                    candidate_tokens: vec![],
                })
            }),
        ];
        for (message, mutate) in cases {
            let mut response = sequence_response(true, true, None);
            mutate(&mut response);
            let error = ResponseState::new(&request(), DisaggregationMode::Aggregated)
                .convert(response)
                .expect_err(message);
            assert_eq!(
                error.error_type(),
                dynamo_backend_common::ErrorType::Backend(
                    dynamo_backend_common::BackendError::Unknown
                )
            );
            assert!(error.to_string().contains(message), "{message}: {error}");
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn output_logprobs_preserve_native_ranks_and_token_metadata() {
        let mut request = request();
        request.output_options.logprobs = Some(2);
        let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
        let mut response = sequence_response(true, true, None);
        let output = response.outputs.as_mut().unwrap();
        output.token_ids = vec![42, 50];
        output.num_tokens = 2;
        output.logprobs = vec![-0.25, -0.75];
        output.ranks = vec![1, 3];
        output.candidate_tokens.push(pb::CandidateTokenInfo {
            tokens: vec![pb::candidate_token_info::TokenInfo {
                id: 51,
                logprob: -0.5,
                rank: 1,
            }],
        });
        output.finish_info.as_mut().unwrap().num_output_tokens = 2;
        let result = state.convert(response).unwrap().unwrap();
        assert_eq!(
            result
                .top_logprobs
                .unwrap()
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| {
                            assert!(entry.token.is_none());
                            assert!(entry.bytes.is_none());
                            (entry.token_id, entry.rank, entry.logprob)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
            vec![
                vec![(42, 1, -0.25), (43, 2, -0.5)],
                vec![(50, 3, -0.75), (51, 1, -0.5)],
            ]
        );
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn nonfinite_and_underflowing_logprobs_keep_associations() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -10000.0] {
            let mut response = sequence_response(true, true, None);
            let output = response.outputs.as_mut().unwrap();
            output.logprobs[0] = value;
            output.candidate_tokens[0].tokens[0].logprob = value;
            let result = ResponseState::new(&request(), DisaggregationMode::Aggregated)
                .convert(response)
                .unwrap()
                .unwrap();
            assert_eq!(result.log_probs, Some(vec![-9999.0]));
            let top = &result.top_logprobs.unwrap()[0];
            assert_eq!(
                top.iter()
                    .map(|t| (t.token_id, t.rank, t.logprob))
                    .collect::<Vec<_>>(),
                vec![(42, 1, -9999.0), (43, 2, -9999.0)]
            );
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn prefill_handoff_is_required_only_for_successful_native_terminals() {
        for reason in [
            pb::finish_info::FinishReason::Stop,
            pb::finish_info::FinishReason::Length,
            pb::finish_info::FinishReason::Aborted,
        ] {
            for include_handoff in [false, true] {
                let handoff = json!({"remote_port": 5600, "nested": {"ids": [1, 2], "ok": true}});
                let mut response = sequence_response(
                    true,
                    false,
                    include_handoff.then(|| json_to_struct(handoff.clone()).unwrap()),
                );
                response
                    .outputs
                    .as_mut()
                    .unwrap()
                    .finish_info
                    .as_mut()
                    .unwrap()
                    .finish_reason = reason as i32;
                let result =
                    ResponseState::new(&request(), DisaggregationMode::Prefill).convert(response);
                if reason != pb::finish_info::FinishReason::Aborted && !include_handoff {
                    let error = result.unwrap_err();
                    assert!(error.to_string().contains("missing kv_transfer_params"));
                    continue;
                }
                let terminal = result.unwrap().unwrap();
                if reason == pb::finish_info::FinishReason::Aborted {
                    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
                    assert!(terminal.disaggregated_params.is_none());
                    if !include_handoff {
                        assert!(terminal.token_ids.is_empty());
                        assert!(terminal.text.is_none());
                        let usage = terminal.completion_usage.unwrap();
                        assert_eq!(
                            (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens),
                            (3, 0, 3)
                        );
                    }
                } else {
                    assert_eq!(terminal.disaggregated_params, Some(handoff));
                }
            }
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn empty_chunks_and_delta_tokens_preserve_terminal_usage() {
        let request = minimal_request();
        let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
        assert!(state.convert(pb::GenerateResponse {
            outputs: Some(pb::SequenceOutput::default()),
            ..Default::default()
        }).unwrap().is_none());
        let first = state.convert(sequence_response(false, false, None)).unwrap().unwrap();
        assert_eq!(first.token_ids, vec![42]);
        assert!(first.completion_usage.is_none());
        let mut response = sequence_response(true, false, None);
        let output = response.outputs.as_mut().unwrap();
        output.token_ids = vec![43, 44];
        output.num_tokens = 2;
        output.finish_info.as_mut().unwrap().num_output_tokens = 3;
        let terminal = state.convert(response).unwrap().unwrap();
        assert_eq!(terminal.token_ids, vec![43, 44]);
        assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
        assert!(terminal.disaggregated_params.is_none());
        let usage = terminal.completion_usage.unwrap();
        assert_eq!(
            (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens),
            (3, 3, 6)
        );

        let decode = ResponseState::new(&request, DisaggregationMode::Decode)
            .convert(sequence_response(true, false, None))
            .unwrap()
            .unwrap();
        assert!(decode.disaggregated_params.is_none());
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn output_logprobs_preserve_opt_in_alignment_and_values() {
        for requested in [None, Some(0), Some(2)] {
            let mut request = minimal_request();
            request.output_options.logprobs = requested;
            let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
            let first = state.convert(sequence_response(false, true, None)).unwrap().unwrap();
            assert_eq!(first.token_ids, vec![42]);
            assert_eq!(first.log_probs, requested.map(|_| vec![-0.25]));
            let mut response = sequence_response(true, true, None);
            let output = response.outputs.as_mut().unwrap();
            output.token_ids = vec![42, 50];
            output.num_tokens = 2;
            output.logprobs = vec![-0.25, -0.75];
            output.ranks = vec![1, 1];
            output.candidate_tokens.push(pb::CandidateTokenInfo {
                tokens: vec![pb::candidate_token_info::TokenInfo {
                    id: 51,
                    logprob: -0.5,
                    rank: 2,
                }],
            });
            output.finish_info.as_mut().unwrap().num_output_tokens = 3;
            let result = state.convert(response).unwrap().unwrap();
            assert_eq!(result.log_probs, requested.map(|_| vec![-0.25, -0.75]));
            let expected = (requested == Some(2)).then_some(vec![
                vec![(42, -0.25), (43, -0.5)],
                vec![(50, -0.75), (51, -0.5)],
            ]);
            assert_eq!(
                result.top_logprobs.map(|rows| rows
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .map(|entry| (entry.token_id, entry.logprob))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()),
                expected
            );
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn user_stops_are_preserved_and_system_stops_are_hidden() {
        for (stop, explicit, expected) in [
            (
                pb::finish_info::StopReason::StopString("done".into()),
                vec![],
                Some(StopReason::String("done".into())),
            ),
            (pb::finish_info::StopReason::StopTokenId(42), vec![42], Some(StopReason::Int(42))),
            (pb::finish_info::StopReason::StopTokenId(2), vec![], None),
            (pb::finish_info::StopReason::EosTokenId(2), vec![], None),
            (pb::finish_info::StopReason::EosTokenId(2), vec![2], Some(StopReason::Int(2))),
            (pb::finish_info::StopReason::StopTokenId(2), vec![2], Some(StopReason::Int(2))),
        ] {
            let mut request = minimal_request();
            request.stop_conditions.stop_token_ids = Some(explicit);
            request.stop_conditions.stop_token_ids_hidden = Some(vec![2]);
            let mut response = sequence_response(true, false, None);
            response.outputs.as_mut().unwrap().finish_info.as_mut().unwrap().stop_reason = Some(stop);
            let result = ResponseState::new(&request, DisaggregationMode::Aggregated)
                .convert(response)
                .unwrap()
                .unwrap();
            assert_eq!(result.stop_reason, expected);
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn prefill_terminal_has_zero_completion_usage() {
        for reason in [
            FinishReason::Stop,
            FinishReason::Length,
            FinishReason::Cancelled,
        ] {
            let mut response = terminal_response(reason.clone());
            response.outputs.as_mut().unwrap().finish_info.as_mut().unwrap().kv_transfer_params = Some(
                json_to_struct(json!({"remote_port": 5600, "nested": {"ids": [1, 2], "ok": true}})).unwrap()
            );
            let terminal = ResponseState::new(&minimal_request(), DisaggregationMode::Prefill)
                .convert(response)
                .unwrap()
                .unwrap();
            assert!(terminal.token_ids.is_empty());
            assert!(terminal.text.is_none());
            let usage = terminal.completion_usage.unwrap();
            assert_eq!(
                (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens),
                (3, 0, 3)
            );
            assert_eq!(terminal.finish_reason, Some(reason.clone()));
            if reason == FinishReason::Cancelled {
                assert!(terminal.disaggregated_params.is_none());
            }
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn prompt_metadata_preserves_opt_in_positions_and_values() {
        for has_prompt_logprobs in [false, true] {
            let mut request = minimal_request();
            request.output_options.prompt_logprobs = has_prompt_logprobs.then_some(1);
            let response = prompt_logprob_response(
                &request.token_ids,
                &[0.0, -0.25, -0.5],
                &[vec![], vec![(23, -0.75)], vec![]],
            );
            let result = ResponseState::new(&request, DisaggregationMode::Aggregated)
                .convert(response)
                .unwrap()
                .expect("terminal output")
                .engine_data;
            if !has_prompt_logprobs {
                assert!(result.is_none());
                continue;
            }
            let data = result.unwrap();
            let positions = data["prompt_logprobs"].as_array().unwrap();
            assert_eq!(positions.len(), 3);
            assert!(positions[0].is_null());
            for (position, expected) in [
                (1, vec![("22", -0.25), ("23", -0.75)]),
                (2, vec![("33", -0.5)]),
            ] {
                let entries = positions[position].as_object().unwrap();
                assert_eq!(entries.len(), expected.len());
                for (token, probability) in expected {
                    assert_eq!(entries[token]["logprob"].as_f64(), Some(probability));
                }
            }
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn terminal_reasons_are_preserved() {
        for expected in [
            FinishReason::Stop,
            FinishReason::Length,
            FinishReason::Cancelled,
        ] {
            let result = ResponseState::new(&minimal_request(), DisaggregationMode::Aggregated)
                .convert(terminal_response(expected.clone()))
                .unwrap()
                .expect("terminal output");
            assert_eq!(result.finish_reason, Some(expected));
        }
    }
}
