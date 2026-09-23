// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::ResponseAdapter;
use crate::unit_fixtures::minimal_request;
use dynamo_backend_common::{DisaggregationMode, FinishReason, StopReason};
use serde_json::Value;

#[derive(Default)]
pub(super) struct Logprobs {
    pub(super) selected: Vec<f32>,
    pub(super) candidates: Vec<Vec<(u32, f32)>>,
}

pub(super) enum StopEvent {
    String(String),
    Token(u32),
    Eos(u32),
}

#[derive(Default)]
pub(super) struct ResponseChunk {
    pub(super) token_ids: Vec<u32>,
    pub(super) terminal: Option<(FinishReason, u32)>,
    pub(super) logprobs: Option<Logprobs>,
    pub(super) prompt_logprobs: Option<Logprobs>,
    pub(super) stop: Option<StopEvent>,
    pub(super) handoff: Option<Value>,
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn empty_chunks_and_delta_tokens_preserve_terminal_usage() {
        let request = minimal_request();
        let mut state = ResponseAdapter::new(&request, DisaggregationMode::Aggregated);
        assert!(state.convert(ResponseChunk::default()).unwrap().is_none());
        let first = state
            .convert(ResponseChunk {
                token_ids: vec![42],
                ..Default::default()
            })
            .unwrap()
            .unwrap();
        assert_eq!(first.token_ids, vec![42]);
        assert!(first.completion_usage.is_none());
        let terminal = state
            .convert(ResponseChunk {
                token_ids: vec![43, 44],
                terminal: Some((FinishReason::Stop, 3)),
                ..Default::default()
            })
            .unwrap()
            .unwrap();
        assert_eq!(terminal.token_ids, vec![43, 44]);
        assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
        assert!(terminal.disaggregated_params.is_none());
        let usage = terminal.completion_usage.unwrap();
        assert_eq!(
            (
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens
            ),
            (3, 3, 6)
        );

        let decode = ResponseAdapter::new(&request, DisaggregationMode::Decode)
            .convert(ResponseChunk {
                token_ids: vec![42],
                terminal: Some((FinishReason::Stop, 1)),
                ..Default::default()
            })
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
            let mut state = ResponseAdapter::new(&request, DisaggregationMode::Aggregated);
            let first = state
                .convert(ResponseChunk {
                    token_ids: vec![42],
                    logprobs: Some(Logprobs {
                        selected: vec![-0.25],
                        candidates: vec![vec![(43, -0.5)]],
                    }),
                    ..Default::default()
                })
                .unwrap()
                .unwrap();
            assert_eq!(first.token_ids, vec![42]);
            assert_eq!(first.log_probs, requested.map(|_| vec![-0.25]));
            let result = state
                .convert(ResponseChunk {
                    token_ids: vec![42, 50],
                    terminal: Some((FinishReason::Stop, 3)),
                    logprobs: Some(Logprobs {
                        selected: vec![-0.25, -0.75],
                        candidates: vec![vec![(43, -0.5)], vec![(51, -0.5)]],
                    }),
                    ..Default::default()
                })
                .unwrap()
                .unwrap();
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
    fn prompt_metadata_preserves_opt_in_positions_and_values() {
        for has_prompt_logprobs in [false, true] {
            let mut request = minimal_request();
            request.output_options.prompt_logprobs = has_prompt_logprobs.then_some(1);
            let result = ResponseAdapter::new(&request, DisaggregationMode::Aggregated)
                .convert(ResponseChunk {
                    token_ids: vec![42],
                    terminal: Some((FinishReason::Stop, 1)),
                    prompt_logprobs: Some(Logprobs {
                        selected: vec![0.0, -0.25, -0.5],
                        candidates: vec![vec![], vec![(23, -0.75)], vec![]],
                    }),
                    ..Default::default()
                })
                .unwrap()
                .unwrap();
            if !has_prompt_logprobs {
                assert!(result.engine_data.is_none());
                continue;
            }
            let data = result.engine_data.unwrap();
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
            let result = ResponseAdapter::new(&minimal_request(), DisaggregationMode::Aggregated)
                .convert(ResponseChunk {
                    token_ids: vec![42],
                    terminal: Some((expected.clone(), 1)),
                    ..Default::default()
                })
                .unwrap()
                .unwrap();
            assert_eq!(result.finish_reason, Some(expected));
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn user_stops_are_preserved_and_system_stops_are_hidden() {
        for (stop, explicit, expected) in [
            (
                StopEvent::String("done".into()),
                vec![],
                Some(StopReason::String("done".into())),
            ),
            (StopEvent::Token(42), vec![42], Some(StopReason::Int(42))),
            (StopEvent::Token(2), vec![], None),
            (StopEvent::Eos(2), vec![], None),
            (StopEvent::Eos(2), vec![2], Some(StopReason::Int(2))),
            (StopEvent::Token(2), vec![2], Some(StopReason::Int(2))),
        ] {
            let mut request = minimal_request();
            request.stop_conditions.stop_token_ids = Some(explicit);
            request.stop_conditions.stop_token_ids_hidden = Some(vec![2]);
            let result = ResponseAdapter::new(&request, DisaggregationMode::Aggregated)
                .convert(ResponseChunk {
                    token_ids: vec![42],
                    terminal: Some((FinishReason::Stop, 1)),
                    stop: Some(stop),
                    ..Default::default()
                })
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
            let terminal = ResponseAdapter::new(&minimal_request(), DisaggregationMode::Prefill)
                .convert(ResponseChunk {
                    token_ids: vec![42],
                    terminal: Some((reason.clone(), 1)),
                    handoff: Some(ResponseAdapter::prefill_handoff()),
                    ..Default::default()
                })
                .unwrap()
                .unwrap();
            assert!(terminal.token_ids.is_empty());
            assert!(terminal.text.is_none());
            let usage = terminal.completion_usage.unwrap();
            assert_eq!(
                (
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens
                ),
                (3, 0, 3)
            );
            assert_eq!(terminal.finish_reason, Some(reason.clone()));
            if reason == FinishReason::Cancelled {
                assert!(terminal.disaggregated_params.is_none());
            }
        }
    }
}
