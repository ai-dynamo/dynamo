// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::common::protocols::EngineType;

fn args(engine_type: EngineType) -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(engine_type)
        .block_size(4)
        .num_gpu_blocks(128)
        .max_num_seqs(Some(8))
        .max_num_batched_tokens(Some(64))
        .speedup_ratio(1000.0)
        .dp_size(1)
        .build()
        .unwrap()
}

#[tokio::test]
async fn streams_planned_tokens_to_the_owning_request() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let engine = LiveEngine::start(args(engine_type), 0).unwrap();
        let uuid = Uuid::from_u128(1);
        let mut request = engine
            .submit(DirectRequest {
                tokens: vec![1, 2, 3],
                max_output_tokens: 3,
                output_token_ids: Some(vec![41, 42, 43]),
                uuid: Some(uuid),
                ..Default::default()
            })
            .await
            .unwrap();

        let mut outputs = Vec::new();
        while let Some(signal) = request.recv().await {
            outputs.push((signal.uuid, signal.token_id, signal.completed));
            if signal.completed {
                break;
            }
        }
        assert_eq!(
            outputs,
            vec![
                (uuid, Some(41), false),
                (uuid, Some(42), false),
                (uuid, Some(43), true),
            ]
        );
        assert!(request.recv().await.is_none());
        assert_eq!(engine.active_request_count(), 0);
    }
}

#[tokio::test]
async fn dropping_engine_closes_outstanding_request_streams() {
    let engine = LiveEngine::start(args(EngineType::Vllm), 0).unwrap();
    let mut request = engine
        .submit(DirectRequest {
            tokens: vec![1; 256],
            max_output_tokens: 10_000,
            uuid: Some(Uuid::from_u128(6)),
            ..Default::default()
        })
        .await
        .unwrap();

    drop(engine);
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while request.recv().await.is_some() {}
    })
    .await
    .expect("engine shutdown should close every outstanding output route");
}

#[tokio::test]
async fn duplicate_request_id_does_not_replace_the_original_stream() {
    let engine = LiveEngine::start(args(EngineType::Vllm), 0).unwrap();
    let uuid = Uuid::from_u128(3);
    let original = engine
        .submit(DirectRequest {
            tokens: vec![1, 2, 3],
            max_output_tokens: 1_000,
            uuid: Some(uuid),
            ..Default::default()
        })
        .await
        .unwrap();
    let duplicate = engine
        .submit(DirectRequest {
            tokens: vec![4, 5, 6],
            max_output_tokens: 1,
            uuid: Some(uuid),
            ..Default::default()
        })
        .await;
    let error = match duplicate {
        Ok(_) => panic!("duplicate request ID must be rejected"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("already active"));
    assert_eq!(engine.active_request_count(), 1);
    original.cancel().await.unwrap();
    assert_eq!(engine.active_request_count(), 0);
}

#[tokio::test]
async fn cancelled_pass_output_does_not_reach_reused_request_id() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let mut timed_args = args(engine_type);
        timed_args.speedup_ratio = 0.1;
        let engine = LiveEngine::start(timed_args, 0).unwrap();
        let uuid = Uuid::from_u128(8);
        let old = engine
            .submit(DirectRequest {
                tokens: vec![1],
                max_output_tokens: 100,
                output_token_ids: Some(vec![11; 100]),
                uuid: Some(uuid),
                ..Default::default()
            })
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), engine.cancel(uuid))
                .await
                .expect("old request cancellation should be observed during the pass")
                .unwrap()
        );
        drop(old);

        let mut replacement = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            engine.submit(DirectRequest {
                tokens: vec![2],
                max_output_tokens: 1,
                output_token_ids: Some(vec![22]),
                uuid: Some(uuid),
                ..Default::default()
            }),
        )
        .await
        .expect("replacement should be admitted after the pending pass boundary")
        .unwrap();
        let output = tokio::time::timeout(std::time::Duration::from_secs(3), replacement.recv())
            .await
            .expect("replacement should produce its planned token")
            .unwrap();
        assert_eq!(output.token_id, Some(22));
        assert!(output.completed);
        assert!(replacement.recv().await.is_none());
        assert_eq!(engine.active_request_count(), 0);
    }
}

#[tokio::test]
async fn slow_reader_does_not_stall_an_unrelated_request() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let engine = LiveEngine::start_with_output_gate(args(engine_type), 0, None, 4).unwrap();
        let mut slow = engine
            .submit(DirectRequest {
                tokens: vec![1],
                // Explicit plans are authoritative in both scheduler cores;
                // the live adapter must reserve their effective length.
                max_output_tokens: 1,
                output_token_ids: Some(vec![7; 3]),
                uuid: Some(Uuid::new_v4()),
                ..Default::default()
            })
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let mut fast = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            engine.submit(DirectRequest {
                tokens: vec![2],
                max_output_tokens: 1,
                output_token_ids: Some(vec![22]),
                uuid: Some(Uuid::new_v4()),
                ..Default::default()
            }),
        )
        .await
        .expect("an unrelated request should use the remaining output budget")
        .unwrap();
        let fast_output = tokio::time::timeout(std::time::Duration::from_secs(1), fast.recv())
            .await
            .expect("unrelated request should not wait for the slow reader")
            .unwrap();
        assert_eq!(fast_output.token_id, Some(22));
        assert!(fast_output.completed);

        let received = tokio::time::timeout(std::time::Duration::from_secs(3), async {
            let mut received = 0;
            while let Some(signal) = slow.recv().await {
                received += 1;
                if signal.completed {
                    break;
                }
            }
            received
        })
        .await
        .expect("slow reader should resume and receive the full response");
        assert_eq!(received, 3);
        assert!(slow.recv().await.is_none());
        assert_eq!(engine.active_request_count(), 0);

        let metrics = engine.metrics_receiver().borrow().clone();
        assert_eq!(metrics.running_requests, 0);
        assert_eq!(metrics.waiting_requests, 0);
    }
}

#[tokio::test]
async fn empty_effective_output_is_rejected_before_route_registration() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let engine = LiveEngine::start(args(engine_type), 0).unwrap();
        let error = engine
            .submit(DirectRequest {
                tokens: vec![1],
                max_output_tokens: 4,
                output_token_ids: Some(Vec::new()),
                uuid: Some(Uuid::new_v4()),
                ..Default::default()
            })
            .await
            .err()
            .expect("empty explicit output plan should be rejected");
        assert!(error.to_string().contains("at least one output token"));
        assert_eq!(engine.active_request_count(), 0);
    }
}

#[tokio::test]
async fn output_budget_bounds_slow_reader_memory_until_its_stream_is_released() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let engine = LiveEngine::start_with_output_gate(args(engine_type), 0, None, 4).unwrap();
        let slow = engine
            .submit(DirectRequest {
                tokens: vec![1],
                max_output_tokens: 4,
                output_token_ids: Some(vec![7; 4]),
                uuid: Some(Uuid::new_v4()),
                ..Default::default()
            })
            .await
            .unwrap();

        let pending = engine.submit(DirectRequest {
            tokens: vec![2],
            max_output_tokens: 1,
            output_token_ids: Some(vec![22]),
            uuid: Some(Uuid::new_v4()),
            ..Default::default()
        });
        tokio::pin!(pending);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), &mut pending)
                .await
                .is_err(),
            "submission should wait rather than exceed the global output budget"
        );

        drop(slow);
        let mut admitted = tokio::time::timeout(std::time::Duration::from_secs(1), pending)
            .await
            .expect("dropping the buffered stream should release its output budget")
            .unwrap();
        let output = tokio::time::timeout(std::time::Duration::from_secs(1), admitted.recv())
            .await
            .expect("request admitted after budget release should make progress")
            .unwrap();
        assert_eq!(output.token_id, Some(22));
        assert!(output.completed);
    }
}
