// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::unit_vllm_fixtures::*;
use dynamo_backend_common::FinishReason;
use serde_json::json;

#[path = "shared.rs"]
mod shared;

struct ResponseAdapter {
    state: ResponseState,
    prompt_token_ids: Vec<u32>,
}

impl ResponseAdapter {
    fn new(request: &PreprocessedRequest, mode: DisaggregationMode) -> Self {
        Self {
            state: ResponseState::new(request, mode),
            prompt_token_ids: request.token_ids.to_vec(),
        }
    }

    fn convert(
        &mut self,
        chunk: shared::ResponseChunk,
    ) -> Result<Option<LLMEngineOutput>, DynamoError> {
        let prompt_info = chunk.prompt_logprobs.map(|prompt| pb::PromptInfo {
            num_prompt_tokens: self.prompt_token_ids.len() as u32,
            token_ids: self.prompt_token_ids.clone(),
            ranks: (0..prompt.selected.len() as u32).collect(),
            logprobs: prompt.selected,
            candidate_tokens: candidate_tokens(prompt.candidates),
        });
        let finish_info = chunk
            .terminal
            .map(|(reason, num_output_tokens)| pb::FinishInfo {
                num_output_tokens,
                finish_reason: match reason {
                    FinishReason::Stop => pb::finish_info::FinishReason::Stop,
                    FinishReason::Length => pb::finish_info::FinishReason::Length,
                    FinishReason::Cancelled => pb::finish_info::FinishReason::Aborted,
                    other => panic!("unsupported response fixture finish reason: {other:?}"),
                } as i32,
                stop_reason: chunk.stop.map(|stop| match stop {
                    shared::StopEvent::String(value) => {
                        pb::finish_info::StopReason::StopString(value)
                    }
                    shared::StopEvent::Token(id) => pb::finish_info::StopReason::StopTokenId(id),
                    shared::StopEvent::Eos(id) => pb::finish_info::StopReason::EosTokenId(id),
                }),
                kv_transfer_params: chunk.handoff.map(|value| json_to_struct(value).unwrap()),
                ec_transfer_params: None,
            });
        let logprobs = chunk.logprobs.unwrap_or_default();
        self.state.convert(pb::GenerateResponse {
            prompt_info,
            outputs: Some(pb::SequenceOutput {
                num_tokens: chunk.token_ids.len() as u32,
                token_ids: chunk.token_ids,
                ranks: vec![1; logprobs.selected.len()],
                logprobs: logprobs.selected,
                candidate_tokens: candidate_tokens(logprobs.candidates),
                finish_info,
                ..Default::default()
            }),
        })
    }

    fn prefill_handoff() -> serde_json::Value {
        json!({"remote_port": 5600, "nested": {"ids": [1, 2], "ok": true}})
    }
}

fn candidate_tokens(positions: Vec<Vec<(u32, f32)>>) -> Vec<pb::CandidateTokenInfo> {
    positions
        .into_iter()
        .map(|entries| pb::CandidateTokenInfo {
            tokens: entries
                .into_iter()
                .enumerate()
                .map(
                    |(index, (id, logprob))| pb::candidate_token_info::TokenInfo {
                        id,
                        logprob,
                        rank: index as u32 + 2,
                    },
                )
                .collect(),
        })
        .collect()
}

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
