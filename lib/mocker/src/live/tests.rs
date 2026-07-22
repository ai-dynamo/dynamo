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

async fn wait_for_idle(engine: &LiveEngine) {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            let metrics = engine.metrics_receiver().borrow().clone();
            if engine.active_request_count() == 0
                && metrics.running_requests == 0
                && metrics.waiting_requests == 0
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("live request state should return to idle");
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
async fn queued_output_does_not_reach_a_reused_request_id() {
    let (gate_tx, gate_rx) = watch::channel(false);
    let engine =
        LiveEngine::start_with_output_gate(args(EngineType::Vllm), 0, Some(gate_rx), 2).unwrap();
    let mut metrics = engine.metrics_receiver();
    let uuid = Uuid::from_u128(8);
    let old = engine
        .submit(DirectRequest {
            tokens: vec![1],
            max_output_tokens: 1,
            output_token_ids: Some(vec![11]),
            uuid: Some(uuid),
            ..Default::default()
        })
        .await
        .unwrap();

    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            metrics.changed().await.unwrap();
            let metrics = metrics.borrow();
            if metrics.running_requests == 0 && metrics.waiting_requests == 0 {
                break;
            }
        }
    })
    .await
    .expect("old terminal output should be queued before ID reuse");
    assert!(!engine.cancel(uuid).await.unwrap());
    drop(old);

    let mut replacement = engine
        .submit(DirectRequest {
            tokens: vec![2],
            max_output_tokens: 1,
            output_token_ids: Some(vec![22]),
            uuid: Some(uuid),
            ..Default::default()
        })
        .await
        .unwrap();
    gate_tx.send(true).unwrap();

    let output = tokio::time::timeout(std::time::Duration::from_secs(3), replacement.recv())
        .await
        .expect("replacement should produce its planned token")
        .unwrap();
    assert_eq!(output.token_id, Some(22));
    assert!(output.completed);
    assert!(replacement.recv().await.is_none());
}

#[tokio::test]
async fn full_output_stream_is_cancelled_without_stalling_an_unrelated_request() {
    let engine = LiveEngine::start_with_output_gate(args(EngineType::Vllm), 0, None, 1).unwrap();
    let mut slow = engine
        .submit(DirectRequest {
            tokens: vec![1],
            max_output_tokens: 3,
            output_token_ids: Some(vec![7; 3]),
            uuid: Some(Uuid::new_v4()),
            ..Default::default()
        })
        .await
        .unwrap();
    let mut fast = engine
        .submit(DirectRequest {
            tokens: vec![2],
            max_output_tokens: 1,
            output_token_ids: Some(vec![22]),
            uuid: Some(Uuid::new_v4()),
            ..Default::default()
        })
        .await
        .unwrap();

    let fast_output = tokio::time::timeout(std::time::Duration::from_secs(1), fast.recv())
        .await
        .expect("unrelated request should not wait for the slow reader")
        .unwrap();
    assert_eq!(fast_output.token_id, Some(22));
    assert!(fast_output.completed);
    assert_eq!(slow.recv().await.unwrap().token_id, Some(7));
    assert!(slow.recv().await.is_none());
    wait_for_idle(&engine).await;
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
async fn dropping_an_active_request_cleans_up_and_allows_id_reuse() {
    let (gate_tx, gate_rx) = watch::channel(false);
    let mut timed_args = args(EngineType::Vllm);
    timed_args.speedup_ratio = 0.1;
    let engine = LiveEngine::start_with_output_gate(timed_args, 0, Some(gate_rx), 1).unwrap();
    let uuid = Uuid::from_u128(9);
    let request = engine
        .submit(DirectRequest {
            tokens: vec![1],
            max_output_tokens: 100,
            output_token_ids: Some(vec![7; 100]),
            uuid: Some(uuid),
            ..Default::default()
        })
        .await
        .unwrap();

    drop(request);
    wait_for_idle(&engine).await;

    let mut replacement = engine
        .submit(DirectRequest {
            tokens: vec![2],
            max_output_tokens: 1,
            output_token_ids: Some(vec![22]),
            uuid: Some(uuid),
            ..Default::default()
        })
        .await
        .unwrap();
    gate_tx.send(true).unwrap();
    let output = replacement.recv().await.unwrap();
    assert_eq!(output.token_id, Some(22));
    assert!(output.completed);
}

#[tokio::test]
async fn aborting_a_deferred_submit_cleans_up_after_admission() {
    let mut timed_args = args(EngineType::Vllm);
    timed_args.speedup_ratio = 0.1;
    let engine = LiveEngine::start(timed_args, 0).unwrap();
    let first = engine
        .submit(DirectRequest {
            tokens: vec![1],
            max_output_tokens: 100,
            output_token_ids: Some(vec![7; 100]),
            uuid: Some(Uuid::from_u128(10)),
            ..Default::default()
        })
        .await
        .unwrap();
    let submit_engine = engine.clone();
    let pending = tokio::spawn(async move {
        submit_engine
            .submit(DirectRequest {
                tokens: vec![2],
                max_output_tokens: 100,
                output_token_ids: Some(vec![8; 100]),
                uuid: Some(Uuid::from_u128(11)),
                ..Default::default()
            })
            .await
    });

    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while engine.active_request_count() != 2 || pending.is_finished() {
            assert!(
                !pending.is_finished(),
                "submit was not deferred to the pass boundary"
            );
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("deferred submit should register its route before admission");
    pending.abort();
    let join_error = match pending.await {
        Err(error) => error,
        Ok(_) => panic!("aborted submit task unexpectedly completed"),
    };
    assert!(join_error.is_cancelled());
    first.cancel().await.unwrap();
    wait_for_idle(&engine).await;
}

#[tokio::test]
async fn dispatcher_exit_shuts_down_the_engine_and_closes_streams() {
    let (gate_tx, gate_rx) = watch::channel(false);
    let engine = LiveEngine::start_with_output_gate(
        args(EngineType::Vllm),
        0,
        Some(gate_rx),
        DEFAULT_REQUEST_OUTPUT_CAPACITY,
    )
    .unwrap();
    let mut request = engine
        .submit(DirectRequest {
            tokens: vec![1],
            max_output_tokens: 3,
            output_token_ids: Some(vec![7; 3]),
            uuid: Some(Uuid::from_u128(12)),
            ..Default::default()
        })
        .await
        .unwrap();

    drop(gate_tx);
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(1), request.recv())
            .await
            .expect("dispatcher failure should close request streams")
            .is_none()
    );
    let error = engine
        .submit(DirectRequest {
            tokens: vec![2],
            max_output_tokens: 1,
            output_token_ids: Some(vec![22]),
            uuid: Some(Uuid::from_u128(13)),
            ..Default::default()
        })
        .await
        .err()
        .expect("dispatcher failure should stop new submissions");
    assert!(error.to_string().contains("not running"));
    assert_eq!(engine.active_request_count(), 0);
}
