// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
