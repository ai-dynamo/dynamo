// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use super::*;
use crate::common::protocols::EngineType;
use dynamo_kv_router::protocols::StorageTier;

struct NoopKvSink;

impl crate::common::protocols::KvCacheEventSink for NoopKvSink {
    fn publish(&self, _event: dynamo_kv_router::protocols::KvCacheEvent) -> anyhow::Result<()> {
        Ok(())
    }

    fn publish_with_storage_tier(
        &self,
        _event: dynamo_kv_router::protocols::KvCacheEvent,
        _storage_tier: StorageTier,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

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

async fn submit_and_finish(engine: &LiveEngine, tokens: Vec<u32>, uuid: Uuid) {
    let mut request = engine
        .submit(DirectRequest {
            tokens,
            max_output_tokens: 4,
            uuid: Some(uuid),
            ..Default::default()
        })
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(signal) = request.recv().await {
            if signal.completed {
                return;
            }
        }
        panic!("request output closed before completion");
    })
    .await
    .expect("request should complete");
    wait_for_idle(engine).await;
}

#[tokio::test]
async fn sglang_live_metrics_retain_the_last_prefill_cache_observation() {
    let engine = LiveEngine::start(args(EngineType::Sglang), 0).unwrap();
    let repeated_prompt = (1..=8).collect::<Vec<_>>();

    submit_and_finish(&engine, repeated_prompt.clone(), Uuid::from_u128(30)).await;
    submit_and_finish(&engine, repeated_prompt, Uuid::from_u128(31)).await;

    let hit = engine.metrics_receiver().borrow().clone();
    assert!(hit.sglang_cache_hit_tokens > 0);
    assert!(hit.sglang_cache_total_tokens >= hit.sglang_cache_hit_tokens);

    submit_and_finish(&engine, (101..=108).collect(), Uuid::from_u128(32)).await;

    let miss = engine.metrics_receiver().borrow().clone();
    assert_eq!(miss.sglang_cache_hit_tokens, 0);
    assert!(miss.sglang_cache_total_tokens > 0);
    engine.shutdown().await.unwrap();
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

#[tokio::test]
async fn ordered_lane_forwards_admission_before_releasing_output() {
    let (gate_tx, gate_rx) = watch::channel(false);
    let (admission_tx, mut admission_rx) = mpsc::unbounded_channel();
    let engine = LiveEngine::start_internal(
        args(EngineType::Vllm),
        0,
        LiveEngineOptions {
            admission_tx: Some(admission_tx),
            ..LiveEngineOptions::default()
        },
        Some(gate_rx),
    )
    .unwrap();
    let uuid = Uuid::from_u128(20);
    let mut request = engine
        .submit(DirectRequest {
            tokens: vec![1, 2, 3],
            max_output_tokens: 1,
            output_token_ids: Some(vec![9]),
            uuid: Some(uuid),
            ..Default::default()
        })
        .await
        .unwrap();

    let admission = admission_rx.recv().await.unwrap();
    assert_eq!(admission.event.uuid, uuid);
    gate_tx.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while request.rx.is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("output should reach its request stream");
    let after_dispatch = tokio::time::Instant::now();
    let observed = request.recv_observed().await.unwrap();
    assert!(observed.observed_at <= after_dispatch);
    assert_eq!(observed.event.uuid, uuid);
    assert!(observed.event.completed);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn replay_options_allow_zero_output_and_full_response_buffering() {
    let zero_engine = LiveEngine::start_with_options(
        args(EngineType::Sglang),
        0,
        LiveEngineOptions {
            request_output_capacity: None,
            allow_zero_output: true,
            ..LiveEngineOptions::default()
        },
    )
    .unwrap();
    let mut zero = zero_engine
        .submit(DirectRequest {
            tokens: vec![1, 2, 3],
            max_output_tokens: 0,
            uuid: Some(Uuid::from_u128(21)),
            ..Default::default()
        })
        .await
        .unwrap();
    let terminal = zero.recv().await.unwrap();
    assert!(terminal.completed);
    assert_eq!(terminal.token_id, None);
    zero_engine.shutdown().await.unwrap();

    let buffered_engine = LiveEngine::start_with_options(
        args(EngineType::Vllm),
        0,
        LiveEngineOptions {
            request_output_capacity: None,
            allow_zero_output: true,
            ..LiveEngineOptions::default()
        },
    )
    .unwrap();
    let mut buffered = buffered_engine
        .submit(DirectRequest {
            tokens: vec![4, 5, 6],
            max_output_tokens: 32,
            output_token_ids: Some(vec![7; 32]),
            uuid: Some(Uuid::from_u128(22)),
            ..Default::default()
        })
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while buffered_engine.active_request_count() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the full response should buffer without a receiver draining it");
    let mut output_count = 0;
    let mut saw_terminal = false;
    while let Some(output) = buffered.recv().await {
        output_count += usize::from(output.token_id.is_some());
        if output.completed {
            saw_terminal = true;
            break;
        }
    }
    assert_eq!(output_count, 32);
    assert!(saw_terminal);
    assert_eq!(buffered_engine.active_request_count(), 0);
    buffered_engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn shutdown_waits_for_scheduler_owned_publishers_to_drop() {
    let sink: Arc<dyn crate::common::protocols::KvCacheEventSink> = Arc::new(NoopKvSink);
    let sink_weak = Arc::downgrade(&sink);
    let engine = LiveEngine::start_with_options(
        args(EngineType::Vllm),
        0,
        LiveEngineOptions {
            kv_event_publishers: KvEventPublishers::new(Some(sink), None),
            ..LiveEngineOptions::default()
        },
    )
    .unwrap();

    engine.shutdown().await.unwrap();
    assert!(
        sink_weak.upgrade().is_none(),
        "scheduler publisher must be destroyed before shutdown resolves"
    );
}

#[tokio::test]
async fn shutdown_surfaces_admission_forwarding_failure() {
    let (admission_tx, admission_rx) = mpsc::unbounded_channel();
    drop(admission_rx);
    let engine = LiveEngine::start_with_options(
        args(EngineType::Vllm),
        0,
        LiveEngineOptions {
            admission_tx: Some(admission_tx),
            ..LiveEngineOptions::default()
        },
    )
    .unwrap();
    let submitted = engine
        .submit(DirectRequest {
            tokens: vec![1, 2, 3],
            max_output_tokens: 2,
            uuid: Some(Uuid::from_u128(23)),
            ..Default::default()
        })
        .await;
    if let Ok(mut request) = submitted {
        while request.recv().await.is_some() {}
    }

    let error = engine.shutdown().await.unwrap_err();
    assert!(
        format!("{error:#}").contains("admission receiver closed"),
        "{error:#}"
    );
}
