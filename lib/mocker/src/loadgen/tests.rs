// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::{
    BlockHashOptions, compute_block_hash_for_seq, compute_seq_hash_for_block,
};
use tempfile::NamedTempFile;
use uuid::Uuid;

use super::*;

fn write_trace(lines: &[serde_json::Value]) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    for line in lines {
        use std::io::Write;
        writeln!(file, "{}", serde_json::to_string(line).unwrap()).unwrap();
    }
    file
}

#[test]
fn test_from_mooncake_single_turn_preserves_fields() {
    let file = write_trace(&[serde_json::json!({
        "timestamp": 123.0,
        "input_length": 8,
        "output_length": 4,
        "hash_ids": [7, 8],
    })]);

    let trace = Trace::from_mooncake(file.path(), 4).unwrap();
    assert_eq!(trace.sessions.len(), 1);
    let session = &trace.sessions[0];
    assert_eq!(session.first_arrival_timestamp_ms, Some(123.0));
    assert_eq!(session.turns.len(), 1);
    assert_eq!(session.turns[0].input_length, 8);
    assert_eq!(session.turns[0].max_output_tokens, 4);
    assert_eq!(session.turns[0].hash_ids, vec![7, 8]);
}

#[test]
fn test_from_mooncake_multi_turn_uses_session_id_and_delay() {
    let file = write_trace(&[
        serde_json::json!({
            "session_id": "a",
            "timestamp": 10.0,
            "input_length": 4,
            "output_length": 1,
            "hash_ids": [1],
        }),
        serde_json::json!({
            "session_id": "a",
            "delay": 25.0,
            "input_length": 8,
            "output_length": 2,
            "hash_ids": [1, 2],
        }),
        serde_json::json!({
            "session_id": "b",
            "timestamp": 20.0,
            "input_length": 4,
            "output_length": 1,
            "hash_ids": [3],
        }),
    ]);

    let trace = Trace::from_mooncake(file.path(), 4).unwrap();
    assert_eq!(trace.sessions.len(), 2);
    assert_eq!(trace.sessions[0].session_id, "a");
    assert_eq!(trace.sessions[0].turns.len(), 2);
    assert_eq!(trace.sessions[0].turns[1].delay_after_previous_ms, 25.0);
    assert_eq!(trace.sessions[1].session_id, "b");
}

#[test]
fn test_from_mooncake_defaults_missing_input_length_from_hash_capacity() {
    let file = write_trace(&[serde_json::json!({
        "timestamp": 7.0,
        "output_length": 3,
        "hash_ids": [5, 6],
    })]);

    let trace = Trace::from_mooncake(file.path(), 4).unwrap();
    assert_eq!(trace.sessions.len(), 1);
    assert_eq!(trace.sessions[0].turns[0].input_length, 8);
}

#[test]
fn test_turn_to_direct_request_repeats_hash_ids_by_block_size() {
    let turn = TurnTrace {
        input_length: 6,
        max_output_tokens: 3,
        hash_ids: vec![1, 2],
        delay_after_previous_ms: 0.0,
    };

    let request = turn
        .to_direct_request(4, Uuid::from_u128(1), Some(5.0))
        .unwrap();
    assert_eq!(request.tokens, vec![1, 1, 1, 1, 2, 2]);
    assert_eq!(request.arrival_timestamp_ms, Some(5.0));
}

#[test]
fn test_turn_replay_hashes_match_full_blocks_only() {
    let turn = TurnTrace {
        input_length: 6,
        max_output_tokens: 3,
        hash_ids: vec![1, 2],
        delay_after_previous_ms: 0.0,
    };

    let request = turn
        .to_direct_request(4, Uuid::from_u128(1), Some(5.0))
        .unwrap();
    let replay_hashes = turn.to_replay_hashes(4, 4).unwrap();
    let expected_local =
        compute_block_hash_for_seq(&request.tokens, 4, BlockHashOptions::default());

    assert_eq!(replay_hashes.local_block_hashes, expected_local);
    assert_eq!(
        replay_hashes.sequence_hashes,
        compute_seq_hash_for_block(&expected_local)
    );
    assert_eq!(replay_hashes.local_block_hashes.len(), 1);
}

#[test]
fn test_turn_replay_hashes_support_distinct_trace_and_engine_block_sizes() {
    let turn = TurnTrace {
        input_length: 6,
        max_output_tokens: 3,
        hash_ids: vec![1, 2],
        delay_after_previous_ms: 0.0,
    };

    let request = turn
        .to_direct_request(4, Uuid::from_u128(2), Some(5.0))
        .unwrap();
    let replay_hashes = turn.to_replay_hashes(4, 2).unwrap();
    let expected_local =
        compute_block_hash_for_seq(&request.tokens, 2, BlockHashOptions::default());

    assert_eq!(replay_hashes.local_block_hashes, expected_local);
    assert_eq!(
        replay_hashes.sequence_hashes,
        compute_seq_hash_for_block(&expected_local)
    );
    assert_eq!(replay_hashes.local_block_hashes.len(), 3);
}

#[test]
fn test_partition_by_session_round_robin_keeps_sessions_intact() {
    let trace = Trace::synthetic(SyntheticTraceSpec {
        block_size: 4,
        num_sessions: 4,
        turns_per_session: 2,
        input_tokens: LengthSpec {
            mean: 8,
            stddev: 0.0,
        },
        output_tokens: LengthSpec {
            mean: 2,
            stddev: 0.0,
        },
        shared_prefix_ratio: 0.5,
        num_prefix_groups: 2,
        first_turn_arrivals: ArrivalSpec::Burst,
        inter_turn_delays: DelaySpec::ConstantMs(5.0),
        seed: 7,
    })
    .unwrap();

    let partitions =
        trace.partition_by_session(SessionPartitionSpec::RoundRobin { num_partitions: 2 });
    assert_eq!(partitions.len(), 2);
    assert_eq!(partitions[0].sessions.len(), 2);
    assert_eq!(partitions[1].sessions.len(), 2);
    assert!(
        partitions
            .iter()
            .flat_map(|partition| partition.sessions.iter())
            .all(|session| session.turns.len() == 2)
    );
}

#[test]
fn test_synthetic_prefix_groups_share_prefixes_within_group() {
    let trace = Trace::synthetic(SyntheticTraceSpec {
        block_size: 4,
        num_sessions: 6,
        turns_per_session: 1,
        input_tokens: LengthSpec {
            mean: 16,
            stddev: 0.0,
        },
        output_tokens: LengthSpec {
            mean: 2,
            stddev: 0.0,
        },
        shared_prefix_ratio: 0.5,
        num_prefix_groups: 2,
        first_turn_arrivals: ArrivalSpec::Burst,
        inter_turn_delays: DelaySpec::None,
        seed: 42,
    })
    .unwrap();

    let prefix_len = 2;
    let prefixes = trace
        .sessions
        .iter()
        .map(|session| session.turns[0].hash_ids[..prefix_len].to_vec())
        .collect::<Vec<_>>();
    assert!(prefixes.windows(2).any(|window| window[0] == window[1]));
}

#[test]
fn test_expand_hash_prefix_depth_scales_hashes_and_input_length() {
    let trace = Trace {
        block_size: 4,
        sessions: vec![SessionTrace {
            session_id: "session".to_string(),
            first_arrival_timestamp_ms: Some(10.0),
            turns: vec![TurnTrace {
                input_length: 6,
                max_output_tokens: 2,
                hash_ids: vec![7, 8],
                delay_after_previous_ms: 0.0,
            }],
        }],
    }
    .expand_hash_prefix_depth(3);

    let turn = &trace.sessions[0].turns[0];
    assert_eq!(turn.input_length, 18);
    assert_eq!(turn.hash_ids, vec![21, 22, 23, 24, 25, 26]);

    let request = turn
        .to_direct_request(trace.block_size, Uuid::from_u128(2), Some(10.0))
        .unwrap();
    assert_eq!(request.tokens.len(), 18);
}

#[test]
fn test_rescale_ready_span_scales_session_starts_and_inter_turn_delays() {
    let trace = Trace {
        block_size: 4,
        sessions: vec![
            SessionTrace {
                session_id: "a".to_string(),
                first_arrival_timestamp_ms: Some(10.0),
                turns: vec![
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 1,
                        hash_ids: vec![1],
                        delay_after_previous_ms: 0.0,
                    },
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 1,
                        hash_ids: vec![2],
                        delay_after_previous_ms: 20.0,
                    },
                ],
            },
            SessionTrace {
                session_id: "b".to_string(),
                first_arrival_timestamp_ms: Some(30.0),
                turns: vec![TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![3],
                    delay_after_previous_ms: 0.0,
                }],
            },
        ],
    }
    .rescale_ready_span(100)
    .unwrap();

    assert_eq!(trace.sessions[0].first_arrival_timestamp_ms, Some(0.0));
    assert_eq!(trace.sessions[1].first_arrival_timestamp_ms, Some(100.0));
    assert_eq!(trace.sessions[0].turns[1].delay_after_previous_ms, 100.0);
}

#[test]
fn test_driver_requires_completion_before_follow_up_turn() {
    let trace = Trace {
        block_size: 4,
        sessions: vec![SessionTrace {
            session_id: "s".to_string(),
            first_arrival_timestamp_ms: Some(0.0),
            turns: vec![
                TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![1],
                    delay_after_previous_ms: 0.0,
                },
                TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![2],
                    delay_after_previous_ms: 10.0,
                },
            ],
        }],
    };

    let mut driver = trace.into_trace_driver().unwrap();
    let first = driver.pop_ready(0.0, 1);
    assert_eq!(first.len(), 1);
    assert!(driver.pop_ready(100.0, 1).is_empty());

    driver.on_complete(first[0].request_uuid, 5.0).unwrap();
    assert!(driver.pop_ready(14.0, 1).is_empty());
    let second = driver.pop_ready(15.0, 1);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].turn_index, 1);
}

#[test]
fn test_driver_next_ready_time_tracks_earliest_pending_turn() {
    let trace = Trace {
        block_size: 4,
        sessions: vec![
            SessionTrace {
                session_id: "a".to_string(),
                first_arrival_timestamp_ms: Some(10.0),
                turns: vec![
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 1,
                        hash_ids: vec![1],
                        delay_after_previous_ms: 0.0,
                    },
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 1,
                        hash_ids: vec![2],
                        delay_after_previous_ms: 5.0,
                    },
                ],
            },
            SessionTrace {
                session_id: "b".to_string(),
                first_arrival_timestamp_ms: Some(20.0),
                turns: vec![TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![3],
                    delay_after_previous_ms: 0.0,
                }],
            },
        ],
    };

    let mut driver = trace.into_trace_driver().unwrap();
    assert_eq!(driver.next_ready_time_ms(), Some(10.0));

    let first = driver.pop_ready(10.0, 1);
    assert_eq!(first.len(), 1);
    assert_eq!(driver.next_ready_time_ms(), Some(20.0));

    driver.on_complete(first[0].request_uuid, 25.0).unwrap();
    assert_eq!(driver.next_ready_time_ms(), Some(20.0));

    let second = driver.pop_ready(20.0, 1);
    assert_eq!(second.len(), 1);
    assert_eq!(driver.next_ready_time_ms(), Some(30.0));
}

#[test]
fn test_trace_driver_round_trips_turn_semantics_into_ready_requests() {
    let trace = Trace {
        block_size: 2,
        sessions: vec![
            SessionTrace {
                session_id: "session-a".to_string(),
                first_arrival_timestamp_ms: Some(10.0),
                turns: vec![
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 2,
                        hash_ids: vec![1, 2],
                        delay_after_previous_ms: 0.0,
                    },
                    TurnTrace {
                        input_length: 2,
                        max_output_tokens: 3,
                        hash_ids: vec![3],
                        delay_after_previous_ms: 5.0,
                    },
                ],
            },
            SessionTrace {
                session_id: "session-b".to_string(),
                first_arrival_timestamp_ms: Some(12.0),
                turns: vec![TurnTrace {
                    input_length: 2,
                    max_output_tokens: 1,
                    hash_ids: vec![4],
                    delay_after_previous_ms: 0.0,
                }],
            },
        ],
    };
    let expected = trace.clone();
    let mut driver = trace.into_trace_driver().unwrap();

    assert!(driver.pop_ready(9.0, usize::MAX).is_empty());

    let first = driver.pop_ready(10.0, usize::MAX);
    assert_eq!(first.len(), 1);
    let first = &first[0];
    assert_eq!(first.session_id, "session-a");
    assert_eq!(first.turn_index, 0);
    assert_eq!(first.scheduled_ready_at_ms, 10.0);
    assert_eq!(
        first.request.tokens.len(),
        expected.sessions[0].turns[0].input_length
    );
    assert_eq!(
        first.request.max_output_tokens,
        expected.sessions[0].turns[0].max_output_tokens
    );
    assert_eq!(first.request.arrival_timestamp_ms, Some(10.0));
    assert_eq!(
        first.replay_hashes.as_ref(),
        Some(
            &expected.sessions[0].turns[0]
                .to_replay_hashes(expected.block_size, expected.block_size)
                .unwrap()
        )
    );
    let expected_first_request = expected.sessions[0].turns[0]
        .to_direct_request(expected.block_size, first.request_uuid, Some(10.0))
        .unwrap();
    assert_eq!(first.request.tokens, expected_first_request.tokens);
    assert_eq!(
        first.request.max_output_tokens,
        expected_first_request.max_output_tokens
    );
    assert_eq!(first.request.uuid, expected_first_request.uuid);
    assert_eq!(
        first.request.arrival_timestamp_ms,
        expected_first_request.arrival_timestamp_ms
    );

    let second = driver.pop_ready(12.0, usize::MAX);
    assert_eq!(second.len(), 1);
    let second = &second[0];
    assert_eq!(second.session_id, "session-b");
    assert_eq!(second.turn_index, 0);
    assert_eq!(second.scheduled_ready_at_ms, 12.0);
    assert_eq!(
        second.request.tokens.len(),
        expected.sessions[1].turns[0].input_length
    );
    assert_eq!(
        second.request.max_output_tokens,
        expected.sessions[1].turns[0].max_output_tokens
    );
    assert_eq!(second.request.arrival_timestamp_ms, Some(12.0));
    assert_eq!(
        second.replay_hashes.as_ref(),
        Some(
            &expected.sessions[1].turns[0]
                .to_replay_hashes(expected.block_size, expected.block_size)
                .unwrap()
        )
    );

    driver.on_complete(first.request_uuid, 20.0).unwrap();
    assert!(driver.pop_ready(24.0, usize::MAX).is_empty());

    let third = driver.pop_ready(25.0, usize::MAX);
    assert_eq!(third.len(), 1);
    let third = &third[0];
    assert_eq!(third.session_id, "session-a");
    assert_eq!(third.turn_index, 1);
    assert_eq!(third.scheduled_ready_at_ms, 25.0);
    assert_eq!(
        third.request.tokens.len(),
        expected.sessions[0].turns[1].input_length
    );
    assert_eq!(
        third.request.max_output_tokens,
        expected.sessions[0].turns[1].max_output_tokens
    );
    assert_eq!(third.request.arrival_timestamp_ms, Some(25.0));
    assert_eq!(
        third.replay_hashes.as_ref(),
        Some(
            &expected.sessions[0].turns[1]
                .to_replay_hashes(expected.block_size, expected.block_size)
                .unwrap()
        )
    );
    let expected_third_request = expected.sessions[0].turns[1]
        .to_direct_request(expected.block_size, third.request_uuid, Some(25.0))
        .unwrap();
    assert_eq!(third.request.tokens, expected_third_request.tokens);
    assert_eq!(
        third.request.max_output_tokens,
        expected_third_request.max_output_tokens
    );
    assert_eq!(third.request.uuid, expected_third_request.uuid);
    assert_eq!(
        third.request.arrival_timestamp_ms,
        expected_third_request.arrival_timestamp_ms
    );
}

#[test]
fn test_trace_driver_rechunks_trace_blocks_into_engine_blocks() {
    let trace = Trace {
        block_size: 4,
        sessions: vec![SessionTrace {
            session_id: "session-a".to_string(),
            first_arrival_timestamp_ms: Some(10.0),
            turns: vec![TurnTrace {
                input_length: 6,
                max_output_tokens: 2,
                hash_ids: vec![1, 2],
                delay_after_previous_ms: 0.0,
            }],
        }],
    };
    let mut driver = trace.into_trace_driver_with_block_size(2).unwrap();

    let ready = driver.pop_ready(10.0, usize::MAX);
    assert_eq!(ready.len(), 1);
    let ready = &ready[0];
    assert_eq!(ready.request.tokens, vec![1, 1, 1, 1, 2, 2]);
    assert_eq!(
        ready.replay_hashes.as_ref(),
        Some(
            &TurnTrace {
                input_length: 6,
                max_output_tokens: 2,
                hash_ids: vec![1, 2],
                delay_after_previous_ms: 0.0,
            }
            .to_replay_hashes(4, 2)
            .unwrap()
        )
    );
}
