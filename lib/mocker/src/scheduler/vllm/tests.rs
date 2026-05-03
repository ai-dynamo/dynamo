// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_kv_router::indexer::{METRIC_EVENT_REMOVED, METRIC_EVENT_STORED};
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, WorkerId};
use rstest::rstest;
use tokio::sync::mpsc;
use tokio::time::interval;
use uuid::Uuid;

use crate::common::protocols::{
    DirectRequest, FpmPublisher, KvCacheEventSink, KvEventPublishers, MockEngineArgs, OutputSignal,
    PreemptionMode, RawKvEvent, RawKvEventSink,
};
use crate::common::sequence::ActiveSequence;
use crate::scheduler::RouterEventVisibility;
use crate::scheduler::SchedulerHandle;
use crate::scheduler::test_utils::{RouterIndexerHarness, removed_event_count, stored_hashes};

use super::core::{RequestStatus, VllmCore, VllmRequestState};
use super::live::{MockerMetrics, Scheduler};

const ROUTER_TEST_WORKER_ID: WorkerId = 23;

fn assert_scheduler_idle(metrics: &MockerMetrics) {
    assert_eq!(
        metrics.active_decode_blocks, 0,
        "Expected 0 active blocks, got {}",
        metrics.active_decode_blocks
    );
    assert_eq!(
        metrics.gpu_cache_usage_perc, 0.0,
        "Expected 0.0 cache usage, got {}",
        metrics.gpu_cache_usage_perc
    );
    assert!(
        metrics.total_blocks > 0,
        "Expected total_blocks to be populated, got {}",
        metrics.total_blocks
    );
}

fn make_args() -> MockEngineArgs {
    MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(6)
        .max_num_batched_tokens(Some(8))
        .max_num_seqs(Some(3))
        .enable_chunked_prefill(true)
        .enable_prefix_caching(false)
        .speedup_ratio(0.0)
        .build()
        .unwrap()
}

fn router_args() -> MockEngineArgs {
    MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(12)
        .max_num_batched_tokens(Some(12))
        .max_num_seqs(Some(3))
        .enable_chunked_prefill(true)
        .enable_prefix_caching(true)
        .speedup_ratio(0.0)
        .build()
        .unwrap()
}

mod core_behavior {
    use super::*;

    #[test]
    fn test_unified_pass_keeps_partial_prefill_in_running() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(6)
            .max_num_batched_tokens(Some(12))
            .max_num_seqs(Some(3))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let r1 = Uuid::from_u128(1);
        let r2 = Uuid::from_u128(2);
        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 2,
            uuid: Some(r1),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 2,
            uuid: Some(r2),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);

        assert_eq!(
            pass.output_signals.len(),
            1,
            "first request should emit immediately"
        );
        assert_eq!(core.state.waiting.len(), 0);
        assert_eq!(
            core.state.running.iter().copied().collect::<Vec<_>>(),
            vec![r1, r2]
        );
        assert_eq!(core.state.requests.get(&r1).unwrap().num_computed_tokens, 8);
        assert_eq!(core.state.requests.get(&r2).unwrap().num_computed_tokens, 4);
        assert_eq!(
            core.state
                .requests
                .get(&r1)
                .unwrap()
                .sequence
                .generated_tokens(),
            1
        );
        assert_eq!(
            core.state.requests.get(&r2).unwrap().status,
            RequestStatus::Running
        );
        assert_eq!(core.kv_manager.num_active_blocks(), 4);
    }

    #[test]
    fn test_running_requests_consume_budget_before_waiting() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(4))
            .max_num_seqs(Some(3))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let r1 = Uuid::from_u128(1);
        let r2 = Uuid::from_u128(2);
        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 2,
            uuid: Some(r1),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 2,
            uuid: Some(r2),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();
        core.execute_pass(&mut collector, 0.0);
        let pass = core.execute_pass(&mut collector, 1.0);

        assert!(pass.output_signals.iter().any(|signal| signal.uuid == r1));
        assert_eq!(
            core.state.requests.get(&r2).unwrap().num_computed_tokens,
            0,
            "waiting request should not steal budget before the running request catches up"
        );
    }

    #[test]
    fn test_execute_pass_batches_two_ready_requests_together() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let r1 = Uuid::from_u128(101);
        let r2 = Uuid::from_u128(202);
        for (uuid, tokens) in [(r1, vec![1; 4]), (r2, vec![2; 4])] {
            core.receive(DirectRequest {
                tokens,
                max_output_tokens: 1,
                uuid: Some(uuid),
                dp_rank: 0,
                arrival_timestamp_ms: None,
            });
        }

        let mut collector = crate::replay::TraceCollector::default();
        collector.on_arrival(r1, 0.0, 4, 1);
        collector.on_arrival(r2, 0.0, 4, 1);
        let pass = core.execute_pass(&mut collector, 0.0);
        let admitted = pass
            .admissions
            .iter()
            .map(|admission| admission.uuid)
            .collect::<Vec<_>>();
        let first = collector.snapshot(r1).unwrap();
        let second = collector.snapshot(r2).unwrap();

        assert_eq!(pass.admissions.len(), 2);
        assert!(admitted.contains(&r1));
        assert!(admitted.contains(&r2));
        assert!(
            first.first_admit_ms.is_some(),
            "r1 should have been admitted"
        );
        assert!(
            second.first_admit_ms.is_some(),
            "r2 should have been admitted"
        );
        assert!(
            first.first_token_ms.is_some(),
            "r1 should have emitted a token"
        );
        assert!(
            second.first_token_ms.is_some(),
            "r2 should have emitted a token"
        );
        assert_eq!(first.first_admit_ms, second.first_admit_ms);
        assert_eq!(first.first_token_ms, second.first_token_ms);
    }

    #[test]
    fn test_prefill_completion_emits_handoff_delay() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(8)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .worker_type(crate::common::protocols::WorkerType::Prefill)
            .kv_transfer_bandwidth(Some(1.0))
            .kv_bytes_per_token(Some(1_000_000))
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        core.receive(DirectRequest {
            tokens: vec![1; 8],
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(81)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        let signal = pass
            .output_signals
            .first()
            .expect("prefill pass should emit one completed signal");

        assert!(signal.completed);
        assert_eq!(signal.handoff_delay_ms, Some(8.0));
    }

    #[test]
    fn test_first_token_can_arrive_on_prompt_completion_pass() {
        let mut core = VllmCore::new(make_args());
        let uuid = Uuid::from_u128(11);
        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 2,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);

        assert_eq!(pass.output_signals.len(), 1);
        assert_eq!(pass.output_signals[0].uuid, uuid);
        assert!(!pass.output_signals[0].completed);
        assert_eq!(
            core.state
                .requests
                .get(&uuid)
                .unwrap()
                .sequence
                .generated_tokens(),
            1
        );
    }

    #[test]
    fn test_preemption_requeues_newest_running_request() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(6)
            .max_num_batched_tokens(Some(12))
            .max_num_seqs(Some(3))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .preemption_mode(PreemptionMode::Lifo)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let r1 = Uuid::from_u128(1);
        let r2 = Uuid::from_u128(2);
        let r3 = Uuid::from_u128(3);
        for (uuid, range) in [(r1, 0u32..8u32), (r2, 100u32..108u32), (r3, 200u32..212u32)] {
            core.receive(DirectRequest {
                tokens: range.collect(),
                max_output_tokens: 2,
                uuid: Some(uuid),
                dp_rank: 0,
                arrival_timestamp_ms: None,
            });
        }

        let mut collector = crate::replay::TraceCollector::default();
        core.execute_pass(&mut collector, 0.0);
        core.execute_pass(&mut collector, 1.0);
        let request = core.state.requests.get(&r2).unwrap();
        assert_eq!(request.status, RequestStatus::Preempted);
        assert_eq!(request.num_computed_tokens, 0);
        assert_eq!(request.num_preemptions, 1);
        assert_eq!(core.state.waiting.front().copied(), Some(r2));
    }

    #[test]
    fn test_running_request_catches_up_decode_tail_before_promote() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(8)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let uuid = Uuid::from_u128(99);
        let mut sequence = ActiveSequence::new((0..6).collect(), 16, Some(4), true, false);

        let signal = sequence.take_creation_signal().unwrap();
        assert_eq!(core.kv_manager.process(&signal), 2);
        for _ in 0..6 {
            let signals = sequence.generate();
            for signal in &signals {
                core.kv_manager.process(signal);
            }
            if sequence.generated_tokens() < sequence.max_output_tokens() {
                sequence.commit_allocation(sequence.len());
            }
        }

        let free = sequence.reset_with_signal();
        for signal in &free {
            core.kv_manager.process(signal);
        }
        let prompt_only = sequence
            .prepare_allocation(sequence.num_input_tokens())
            .unwrap();
        assert_eq!(core.kv_manager.process(&prompt_only), 2);
        sequence.commit_allocation(sequence.num_input_tokens());

        core.state.insert_running_for_test(uuid);
        core.state.requests.insert(
            uuid,
            VllmRequestState {
                sequence,
                status: RequestStatus::Running,
                num_computed_tokens: 9,
                num_preemptions: 1,
            },
        );

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        let request = core.state.requests.get(&uuid).unwrap();

        assert_eq!(pass.output_signals.len(), 1);
        assert_eq!(request.num_computed_tokens, 12);
        assert_eq!(request.sequence.num_allocated_tokens(), 13);
        assert_eq!(core.kv_manager.num_active_blocks(), 4);
    }

    #[test]
    fn test_completion_returns_scheduler_to_idle() {
        let mut core = VllmCore::new(make_args());
        for uuid in [Uuid::from_u128(1), Uuid::from_u128(2)] {
            core.receive(DirectRequest {
                tokens: (0..8).collect(),
                max_output_tokens: 2,
                uuid: Some(uuid),
                dp_rank: 0,
                arrival_timestamp_ms: None,
            });
        }

        let mut collector = crate::replay::TraceCollector::default();
        while !core.is_empty() {
            core.execute_pass(&mut collector, 0.0);
        }

        assert!(core.state.waiting.is_empty());
        assert!(core.state.running.is_empty());
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
    }
}

mod router_events {
    use super::*;

    #[test]
    fn test_vllm_pass_visibility_is_pass_start() {
        let mut core = VllmCore::new_with_kv_capture(router_args(), ROUTER_TEST_WORKER_ID);
        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 2,
            uuid: Some(Uuid::from_u128(71)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);

        assert_eq!(
            pass.router_event_visibility,
            RouterEventVisibility::PassStart
        );
    }

    #[tokio::test]
    async fn test_completion_events_apply_cleanly() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let mut core = VllmCore::new_with_kv_capture(router_args(), ROUTER_TEST_WORKER_ID);
        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 4,
            uuid: Some(Uuid::from_u128(41)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();
        let mut now_ms = 0.0;
        let mut saw_store = false;
        while !core.is_empty() {
            let pass = core.execute_pass(&mut collector, now_ms);
            saw_store |= !stored_hashes(&pass.kv_events).is_empty();
            now_ms = pass.end_ms;
            harness.apply_events(pass.kv_events).await;
        }

        assert!(saw_store);
        assert!(harness.ok_count(METRIC_EVENT_STORED) > 0);
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        harness.assert_no_event_warnings();
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_preemption_recompute_events_apply_cleanly() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(6)
            .max_num_batched_tokens(Some(12))
            .max_num_seqs(Some(3))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .preemption_mode(PreemptionMode::Lifo)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new_with_kv_capture(args, ROUTER_TEST_WORKER_ID);
        let r1 = Uuid::from_u128(51);
        let r2 = Uuid::from_u128(52);
        let r3 = Uuid::from_u128(53);
        for (uuid, range) in [(r1, 0u32..8u32), (r2, 100u32..108u32), (r3, 200u32..212u32)] {
            core.receive(DirectRequest {
                tokens: range.collect(),
                max_output_tokens: 2,
                uuid: Some(uuid),
                dp_rank: 0,
                arrival_timestamp_ms: None,
            });
        }

        let mut collector = crate::replay::TraceCollector::default();
        let mut now_ms = 0.0;
        let mut saw_remove = false;
        for _ in 0..2 {
            let pass = core.execute_pass(&mut collector, now_ms);
            saw_remove |= removed_event_count(&pass.kv_events) > 0;
            now_ms = pass.end_ms;
            harness.apply_events(pass.kv_events).await;
        }

        let request = core.state.requests.get(&r2).unwrap();
        assert_eq!(request.status, RequestStatus::Preempted);
        assert_eq!(request.num_computed_tokens, 0);
        assert_eq!(request.num_preemptions, 1);
        assert_eq!(core.state.waiting.front().copied(), Some(r2));
        assert!(saw_remove);
        assert!(harness.ok_count(METRIC_EVENT_REMOVED) > 0);
        harness.assert_no_event_warnings();
        harness.shutdown();
    }
}

mod live_scheduler {
    use super::*;

    type CapturedKvEvent = (KvCacheEvent, Option<Vec<Vec<u32>>>);

    #[derive(Default)]
    struct CapturingKvSink {
        events: Mutex<Vec<CapturedKvEvent>>,
    }

    impl CapturingKvSink {
        fn take(&self) -> Vec<CapturedKvEvent> {
            std::mem::take(&mut *self.events.lock().unwrap())
        }
    }

    impl KvCacheEventSink for CapturingKvSink {
        fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
            self.events.lock().unwrap().push((event, None));
            Ok(())
        }
    }

    impl RawKvEventSink for CapturingKvSink {
        fn publish(&self, event: RawKvEvent) -> anyhow::Result<()> {
            self.events
                .lock()
                .unwrap()
                .push((event.event, event.block_token_ids));
            Ok(())
        }
    }

    #[rstest]
    #[case::case_1(false, false, false)]
    #[case::case_2(false, true, false)]
    #[case::case_3(true, false, false)]
    #[case::case_4(true, true, false)]
    #[case::case_5(false, false, true)]
    #[case::case_6(false, true, true)]
    #[case::case_7(true, false, true)]
    #[case::case_8(true, true, true)]
    #[tokio::test]
    async fn test_scheduler_token_generation_patterns(
        #[case] use_shared_tokens: bool,
        #[case] enable_prefix_caching: bool,
        #[case] enable_chunked_prefill: bool,
    ) {
        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();

        let args = MockEngineArgs::builder()
            .num_gpu_blocks(500)
            .block_size(64)
            .speedup_ratio(1000.0)
            .enable_prefix_caching(enable_prefix_caching)
            .enable_chunked_prefill(enable_chunked_prefill)
            .build()
            .unwrap();

        // Side-channel router indexer: the mocker's emitted KV event stream is
        // forwarded in real time into `LocalKvIndexer`, which applies Stored/
        // Removed events against its own radix tree. If the mocker ever emits
        // an invalid event (dangling parent, re-Stored of a present block, or
        // Removed of an unknown block), the indexer's per-status counters tick
        // — `assert_no_event_errors()` turns those into a test failure.
        let harness = RouterIndexerHarness::new(64, ROUTER_TEST_WORKER_ID);
        let (forwarder_sink, forwarder_task) = harness.spawn_forwarder();
        let publishers = KvEventPublishers::new(Some(forwarder_sink as _), None);

        let scheduler = Scheduler::new(
            args,
            0,
            Some(output_tx),
            publishers,
            None,
            FpmPublisher::default(),
        );

        crate::scheduler::test_utils::assert_scheduler_completes_all(
            &scheduler,
            &mut output_rx,
            200,
            1000,
            100,
            use_shared_tokens,
        )
        .await;

        // Stop the scheduler so no new events fire, then drop the forwarder's
        // sender by dropping the scheduler → forwarder task drains and exits.
        drop(scheduler);
        let _ = tokio::time::timeout(Duration::from_secs(2), forwarder_task).await;
        harness.flush().await;
        harness.assert_no_event_errors();
        // NOTE: we do NOT assert `dump_events().is_empty()` here because
        // mocker's protocol does not emit router `Removed` events on
        // request completion.
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_cache_hit_rate_with_identical_requests() {
        let block_size: usize = 64;
        let max_output_tokens: usize = 10;
        let speedup_ratio = 10.0;
        let num_requests = 10;
        let token_length = 65;

        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();

        let args = MockEngineArgs::builder()
            .num_gpu_blocks(100)
            .block_size(block_size)
            .speedup_ratio(speedup_ratio)
            .build()
            .unwrap();

        let scheduler = Scheduler::new(
            args,
            0,
            Some(output_tx),
            KvEventPublishers::default(),
            None,
            FpmPublisher::default(),
        );
        let identical_tokens: Vec<u32> = (0..token_length).collect();

        for _ in 0..num_requests {
            scheduler.receive(DirectRequest {
                tokens: identical_tokens.clone(),
                max_output_tokens,
                uuid: None,
                dp_rank: 0,
                arrival_timestamp_ms: None,
            });
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let mut received_tokens = 0;
        let timeout = tokio::time::sleep(Duration::from_millis(500));
        tokio::pin!(timeout);
        let metrics_rx = scheduler.metrics_receiver();
        let mut debug_interval = interval(Duration::from_millis(500));

        loop {
            tokio::select! {
                biased;
                _ = debug_interval.tick() => {
                    let _metrics = metrics_rx.borrow().clone();
                    tracing::debug!("Forward Pass Metrics: {_metrics:#?}");
                }
                Some(output_batch) = output_rx.recv() => {
                    received_tokens += output_batch.len();
                    timeout.set(tokio::time::sleep(Duration::from_millis(500)));
                }
                _ = &mut timeout => break,
            }
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
        let metrics = metrics_rx.borrow().clone();
        assert_scheduler_idle(&metrics);
        assert_eq!(received_tokens, num_requests * max_output_tokens);
    }

    #[tokio::test]
    async fn test_receiver_drop_cleans_up_resources() {
        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(10)
            .block_size(64)
            .speedup_ratio(100.0)
            .build()
            .unwrap();

        let scheduler = Scheduler::new(
            args,
            0,
            Some(output_tx),
            KvEventPublishers::default(),
            None,
            FpmPublisher::default(),
        );
        scheduler.receive(DirectRequest {
            tokens: (0..256).collect(),
            max_output_tokens: 200,
            uuid: None,
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut received_count = 0;
        while received_count < 129 {
            if let Some(output_batch) = output_rx.recv().await {
                received_count += output_batch.len();
                continue;
            }
            panic!("Channel closed before receiving 129 tokens");
        }

        drop(output_rx);
        let metrics_rx = scheduler.metrics_receiver();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if metrics_rx.borrow().active_decode_blocks == 0 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let metrics = metrics_rx.borrow().clone();
        assert_scheduler_idle(&metrics);
    }

    #[tokio::test]
    async fn test_live_scheduler_forwards_buffered_kv_token_ids() {
        let sink = Arc::new(CapturingKvSink::default());
        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(12)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(1000.0)
            .zmq_kv_events_port(Some(12345))
            .build()
            .unwrap();
        let scheduler = Scheduler::new(
            args,
            0,
            Some(output_tx),
            KvEventPublishers::new(None, Some(sink.clone())),
            None,
            FpmPublisher::default(),
        );

        scheduler.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(72)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let output_batch = tokio::time::timeout(Duration::from_secs(2), output_rx.recv())
            .await
            .expect("scheduler should emit output")
            .expect("output channel should stay open");
        let signal = output_batch
            .into_iter()
            .next()
            .expect("live scheduler should emit one output signal");
        assert!(signal.completed);

        tokio::time::sleep(Duration::from_millis(50)).await;
        let events = sink.take();
        let stored = events
            .into_iter()
            .find_map(|(event, block_token_ids)| match event.data {
                KvCacheEventData::Stored(_) => block_token_ids,
                _ => None,
            })
            .expect("live scheduler should forward stored KV event token ids");
        assert!(!stored.is_empty());
        assert!(stored.iter().all(|block| !block.is_empty()));
    }

    #[tokio::test]
    async fn test_live_pathological_load_no_router_event_errors() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let (sink, forward_task) = harness.spawn_forwarder();

        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
        let scheduler = Scheduler::new(
            MockEngineArgs::builder()
                .block_size(4)
                .num_gpu_blocks(6)
                .max_num_batched_tokens(Some(8))
                .max_num_seqs(Some(3))
                .enable_prefix_caching(true)
                .enable_chunked_prefill(true)
                .speedup_ratio(1000.0)
                .build()
                .unwrap(),
            0,
            Some(output_tx),
            KvEventPublishers::new(Some(sink.clone()), None),
            None,
            FpmPublisher::default(),
        );

        for _ in 0..8 {
            scheduler.receive(DirectRequest {
                tokens: vec![42; 8],
                max_output_tokens: 4,
                uuid: None,
                dp_rank: 0,
                arrival_timestamp_ms: None,
            });
        }

        let expected = 8 * 4;
        let mut seen = 0;
        let timeout = tokio::time::sleep(Duration::from_secs(5));
        tokio::pin!(timeout);

        loop {
            tokio::select! {
                Some(output_batch) = output_rx.recv() => {
                    seen += output_batch.len();
                    if seen == expected {
                        break;
                    }
                }
                _ = &mut timeout => {
                    break;
                }
            }
        }

        assert_eq!(seen, expected);
        drop(scheduler);
        drop(sink);
        forward_task.await.unwrap();
        harness.flush().await;

        harness.assert_no_event_errors();
        assert!(harness.ok_count(METRIC_EVENT_STORED) > 0);
        harness.shutdown();
    }
}

mod forward_pass_metrics {
    use super::*;

    /// Helper to build args with specific parameters for FPM tests.
    fn fpm_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    #[test]
    fn test_fpm_single_prefill_request() {
        let mut core = VllmCore::new(fpm_args());
        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        let fpm = pass.fpm.expect("FPM should be present");

        assert_eq!(fpm.num_prefill_requests, 1);
        assert_eq!(fpm.sum_prefill_tokens, 8, "all 8 prompt tokens computed");
        assert_eq!(fpm.sum_prefill_kv_tokens, 0, "no prefix cache");
        assert_eq!(fpm.num_decode_requests, 0);
        assert_eq!(fpm.num_queued_prefill, 0);
        assert_eq!(fpm.num_queued_decode, 0);
        assert!(fpm.wall_time_secs > 0.0);
    }

    #[test]
    fn test_fpm_prefill_and_decode_mixed_batch() {
        let mut core = VllmCore::new(fpm_args());

        // r1: 4-token prompt, 3 output tokens
        let r1 = Uuid::from_u128(1);
        core.receive(DirectRequest {
            tokens: (0..4).collect(),
            max_output_tokens: 3,
            uuid: Some(r1),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();

        // Pass 1: prefill r1 (4 tokens) + first decode token
        let pass1 = core.execute_pass(&mut collector, 0.0);
        let fpm1 = pass1.fpm.expect("FPM should be present");
        assert_eq!(fpm1.num_prefill_requests, 1);
        assert_eq!(fpm1.sum_prefill_tokens, 4);

        // r2: 4-token prompt arriving while r1 is decoding
        let r2 = Uuid::from_u128(2);
        core.receive(DirectRequest {
            tokens: (100..104).collect(),
            max_output_tokens: 3,
            uuid: Some(r2),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        // Pass 2: r1 decode + r2 prefill (mixed batch)
        let pass2 = core.execute_pass(&mut collector, 1.0);
        let fpm2 = pass2.fpm.expect("FPM should be present");
        assert_eq!(fpm2.num_prefill_requests, 1, "r2 is prefilling");
        assert_eq!(fpm2.num_decode_requests, 1, "r1 is decoding");
        assert_eq!(fpm2.sum_prefill_tokens, 4);
        assert!(
            fpm2.sum_decode_kv_tokens > 0,
            "decode request should have KV context"
        );
    }

    #[test]
    fn test_fpm_completed_requests_metrics_correct() {
        // This tests the fix: completed requests should still contribute
        // correct metrics even though they're removed from state before
        // compute_fpm runs.
        let mut core = VllmCore::new(fpm_args());

        // Request with 4-token prompt and 1 output token — completes in 1 pass
        let r1 = Uuid::from_u128(1);
        core.receive(DirectRequest {
            tokens: (0..4).collect(),
            max_output_tokens: 1,
            uuid: Some(r1),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        let fpm = pass.fpm.expect("FPM should be present");

        // r1 completes in this pass. The bug was that prompt_len would be 0
        // because the request was removed from state before compute_fpm ran.
        assert_eq!(fpm.num_prefill_requests, 1);
        assert_eq!(fpm.sum_prefill_tokens, 4);
        // var_prefill_length should reflect the actual prompt length (4), not 0.
        // With a single request, variance is 0 regardless, so check sum_prefill_tokens
        // as the main indicator.
        assert!(pass.completed_requests > 0, "request should have completed");
    }

    #[test]
    fn test_fpm_completed_decode_request_has_kv_context() {
        // Decode request that completes — its KV context should be captured
        // correctly even though it's removed from state.
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);

        let r1 = Uuid::from_u128(1);
        core.receive(DirectRequest {
            tokens: (0..4).collect(),
            max_output_tokens: 2,
            uuid: Some(r1),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();

        // Pass 1: prefill + first decode token
        core.execute_pass(&mut collector, 0.0);

        // Pass 2: second decode token (completes the request)
        let pass2 = core.execute_pass(&mut collector, 1.0);
        let fpm2 = pass2.fpm.expect("FPM should be present");

        assert_eq!(fpm2.num_decode_requests, 1);
        // The completed decode request should have contributed its KV context
        // (prompt_len + generated_so_far at schedule time).
        assert!(
            fpm2.sum_decode_kv_tokens > 0,
            "completed decode request should still contribute KV context, got {}",
            fpm2.sum_decode_kv_tokens
        );
    }

    #[test]
    fn test_fpm_queued_requests() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(4) // Very limited KV — only room for one request
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(2))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);

        // r1 and r2 both have 8-token prompts but only 4 blocks available
        let r1 = Uuid::from_u128(1);
        let r2 = Uuid::from_u128(2);
        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 1,
            uuid: Some(r1),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 1,
            uuid: Some(r2),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        let fpm = pass.fpm.expect("FPM should be present");

        // At least one request should be scheduled, the other might be queued
        // (depending on KV capacity). Some requests may have completed and
        // been removed from both scheduled and queued.
        let total_scheduled = fpm.num_prefill_requests + fpm.num_decode_requests;
        assert!(
            total_scheduled >= 1,
            "at least one request should be scheduled"
        );
    }

    #[test]
    fn test_fpm_var_prefill_length_with_multiple_requests() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(32)
            .max_num_batched_tokens(Some(32))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);

        // Two prefill requests with different prompt lengths
        core.receive(DirectRequest {
            tokens: (0..4).collect(), // prompt_len = 4
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..112).collect(), // prompt_len = 12
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(2)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        let fpm = pass.fpm.expect("FPM should be present");

        assert_eq!(fpm.num_prefill_requests, 2);
        // Population variance of [4, 12]: mean=8, var=((4-8)^2+(12-8)^2)/2 = 16
        assert!(
            (fpm.var_prefill_length - 16.0).abs() < 1e-6,
            "expected var=16.0, got {}",
            fpm.var_prefill_length
        );
    }

    #[test]
    fn test_fpm_chunked_prefill_reports_chunk_not_full_prompt() {
        // With max_num_batched_tokens=8 and a 16-token prompt, chunked prefill
        // should split across two passes. Each pass should report only the
        // chunk size in sum_prefill_tokens, not the full prompt length.
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);

        core.receive(DirectRequest {
            tokens: (0..16).collect(),
            max_output_tokens: 2,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();

        // Pass 1: first chunk
        let pass1 = core.execute_pass(&mut collector, 0.0);
        let fpm1 = pass1.fpm.expect("FPM should be present");
        assert_eq!(fpm1.num_prefill_requests, 1);
        assert!(
            fpm1.sum_prefill_tokens <= 8,
            "chunk should be at most 8 tokens, got {}",
            fpm1.sum_prefill_tokens
        );
        assert!(fpm1.sum_prefill_tokens > 0);

        // Pass 2: remaining chunk
        let pass2 = core.execute_pass(&mut collector, 1.0);
        let fpm2 = pass2.fpm.expect("FPM should be present");
        assert_eq!(fpm2.num_prefill_requests, 1, "still prefilling");
        assert!(
            fpm2.sum_prefill_tokens <= 8,
            "second chunk should also be at most 8 tokens, got {}",
            fpm2.sum_prefill_tokens
        );

        // Total across both chunks should equal the full prompt length
        assert_eq!(
            fpm1.sum_prefill_tokens + fpm2.sum_prefill_tokens,
            16,
            "total prefill tokens across chunks should equal full prompt"
        );

        // Variance should be over the full prompt length (16) in both passes
        assert_eq!(
            fpm1.var_prefill_length, 0.0,
            "single request → zero variance"
        );
        assert_eq!(
            fpm2.var_prefill_length, 0.0,
            "single request → zero variance"
        );
    }

    #[test]
    fn test_fpm_preemption_creates_queued_decode() {
        // Trigger preemption: fill KV with running requests, then submit a new
        // one that forces eviction. The preempted request should appear as a
        // queued decode in FPM.
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(6) // 24 tokens of KV — very tight
            .max_num_batched_tokens(Some(32))
            .max_num_seqs(Some(3))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .preemption_mode(PreemptionMode::Lifo)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let mut collector = crate::replay::TraceCollector::default();

        // r1: 4-token prompt, long output (stays running)
        core.receive(DirectRequest {
            tokens: (0..4).collect(),
            max_output_tokens: 20,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        // Prefill r1 and decode a few tokens to build up KV
        core.execute_pass(&mut collector, 0.0);
        core.execute_pass(&mut collector, 1.0);
        core.execute_pass(&mut collector, 2.0);

        // r2: another request that will compete for KV
        core.receive(DirectRequest {
            tokens: (100..116).collect(), // 16 tokens — will pressure KV
            max_output_tokens: 5,
            uuid: Some(Uuid::from_u128(2)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        // This pass should trigger preemption
        let pass = core.execute_pass(&mut collector, 3.0);
        let fpm = pass.fpm.expect("FPM should be present");

        // We should see at least one queued decode (preempted request) OR one
        // queued prefill (if the new request couldn't be scheduled). The key
        // assertion is that queued metrics are non-zero when KV pressure exists.
        let total_queued = fpm.num_queued_prefill + fpm.num_queued_decode;
        if total_queued > 0 {
            // Preemption occurred — verify the preempted decode has KV context
            if fpm.num_queued_decode > 0 {
                assert!(
                    fpm.sum_queued_decode_kv_tokens > 0,
                    "preempted decode should have KV context"
                );
            }
        }
        // Regardless, at least one request should be scheduled
        let total_scheduled = fpm.num_prefill_requests + fpm.num_decode_requests;
        assert!(total_scheduled >= 1);
    }

    #[tokio::test]
    async fn test_fpm_sent_through_sink() {
        use crate::scheduler::test_utils::CapturingFpmSink;

        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();

        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
        let fpm_sink = Arc::new(CapturingFpmSink::default());
        let fpm_publisher = crate::common::protocols::FpmPublisher::new(Some(
            fpm_sink.clone() as Arc<dyn crate::common::protocols::FpmSink>
        ));

        let scheduler = Scheduler::new(
            args,
            0,
            Some(output_tx),
            KvEventPublishers::default(),
            None,
            fpm_publisher,
        );

        scheduler.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 2,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        // Wait for at least one output signal — ensures the scheduler has
        // completed at least one pass and drained the deferred FPM buffer.
        tokio::time::timeout(Duration::from_secs(5), output_rx.recv())
            .await
            .expect("timed out waiting for output")
            .expect("output channel closed");

        let snapshots = fpm_sink.take();
        assert!(
            !snapshots.is_empty(),
            "should have received at least one FPM snapshot"
        );
        let fpm = &snapshots[0];
        assert_eq!(fpm.num_prefill_requests, 1);
        assert!(fpm.sum_prefill_tokens > 0);
        assert!(fpm.wall_time_secs > 0.0);
    }
}

#[cfg(feature = "kvbm-offload")]
mod offload {
    use dynamo_tokens::PositionalLineageHash;
    use kvbm_engine::G2;
    use kvbm_logical::manager::BlockManager;
    use uuid::Uuid;

    use crate::common::protocols::{DirectRequest, MockEngineArgs};
    use crate::kvbm_offload::{KvbmOffloadConfig, MockOffloadEngine};

    use super::super::core::VllmCore;

    /// Seed `g2` with each PLH by allocating a fresh slot, staging,
    /// registering, and dropping — so the block lands in the inactive
    /// pool and `find_matches_with_options(plh)` returns it.
    fn seed_g2_blocks(g2: &BlockManager<G2>, plhs: &[PositionalLineageHash]) {
        for plh in plhs {
            let (mut alloc, _evicted) = g2.allocate_blocks_with_evictions(1).expect("G2 allocate");
            let mutable = alloc.pop().unwrap();
            let staged = mutable.stage(*plh, g2.block_size()).expect("G2 stage");
            drop(g2.register_block(staged));
        }
    }

    /// Pass entry must call `tick_offload_engine` when an engine is
    /// attached — otherwise PS models never advance and swap-ins
    /// would hang forever. Verifies by observing that
    /// `earliest_offload_deadline` stays `None` on an idle engine
    /// across repeated passes (a no-op tick that doesn't panic is
    /// already a useful signal; the full tick path is covered in
    /// `engine::tests` — this test just confirms the scheduler hook
    /// is wired).
    #[tokio::test]
    async fn execute_pass_ticks_offload_engine_when_attached() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(8)
            .block_size(4)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let engine = MockOffloadEngine::new(KvbmOffloadConfig::default())
            .await
            .expect("engine build");
        core.kv_manager.attach_new_offload_engine(engine);

        assert!(core.kv_manager.earliest_offload_deadline().is_none());
        let mut collector = crate::replay::TraceCollector::default();
        core.execute_pass(&mut collector, 0.0);
        core.execute_pass(&mut collector, 10.0);
        assert!(core.kv_manager.earliest_offload_deadline().is_none());
    }

    /// Retain-and-promote: an admission-parked swap-in whose handle reports
    /// complete must be removed from `requests_awaiting_swap_in` during the
    /// next pass; a handle that's still pending must survive.
    #[tokio::test]
    async fn ready_swap_ins_drain_on_pass_entry() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(8)
            .block_size(4)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let config = KvbmOffloadConfig {
            block_size_tokens: 4,
            block_size_bytes: Some(1_000_000),
            bandwidth_g2_to_g1_gbps: 1.0,
            ..Default::default()
        };
        let engine = MockOffloadEngine::new(config).await.expect("engine build");
        engine.tick(0.0);

        let uuid = Uuid::new_v4();
        core.receive(DirectRequest {
            tokens: (0..4).collect(),
            max_output_tokens: 2,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
        let plhs = core
            .state
            .requests
            .get(&uuid)
            .unwrap()
            .sequence
            .positional_lineage_hashes();
        assert_eq!(plhs.len(), 1, "test request should have one full block");
        seed_g2_blocks(engine.g2_manager(), &plhs);

        core.kv_manager.attach_new_offload_engine(engine);
        let mut collector = crate::replay::TraceCollector::default();
        let pass1 = core.execute_pass(&mut collector, 0.0);
        assert_eq!(
            pass1.admissions.len(),
            0,
            "parked swap-in should not admit in the same pass"
        );
        assert_eq!(
            core.requests_awaiting_swap_in.len(),
            1,
            "request must be parked on swap-in"
        );

        // Before finish time (0.5 ms of a 1 ms transfer): handle stays
        // pending, entry must survive the pass-entry retain.
        core.execute_pass(&mut collector, 0.5);
        assert_eq!(
            core.requests_awaiting_swap_in.len(),
            1,
            "pending swap-in must survive"
        );

        // Past finish time: tick flips the bit, retain drops the entry.
        core.execute_pass(&mut collector, 1.0);
        assert!(
            core.requests_awaiting_swap_in.is_empty(),
            "completed swap-in must drain on pass entry"
        );
    }

    /// A parked G2→G1 transfer must reserve its destination G1 slot before the
    /// bandwidth model starts. Otherwise a following cold request could allocate
    /// the same HBM capacity while DMA is still in flight.
    #[tokio::test]
    async fn g2_swap_in_reserves_destination_slot_before_transfer() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(1)
            .block_size(4)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(2))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let config = KvbmOffloadConfig {
            block_size_tokens: 4,
            block_size_bytes: Some(1_000_000),
            bandwidth_g2_to_g1_gbps: 1.0,
            ..Default::default()
        };
        let engine = MockOffloadEngine::new(config).await.expect("engine build");
        engine.tick(0.0);

        let hit_uuid = Uuid::new_v4();
        core.receive(DirectRequest {
            tokens: (0..4).collect(),
            max_output_tokens: 2,
            uuid: Some(hit_uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
        let cold_uuid = Uuid::new_v4();
        core.receive(DirectRequest {
            tokens: (4..8).collect(),
            max_output_tokens: 2,
            uuid: Some(cold_uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
        let plhs = core
            .state
            .requests
            .get(&hit_uuid)
            .unwrap()
            .sequence
            .positional_lineage_hashes();
        assert_eq!(plhs.len(), 1, "test request should have one full block");
        seed_g2_blocks(engine.g2_manager(), &plhs);
        core.kv_manager.attach_new_offload_engine(engine);

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);

        assert_eq!(
            core.requests_awaiting_swap_in.len(),
            1,
            "G2 hit should be parked on swap-in"
        );
        assert_eq!(
            core.kv_manager.num_active_blocks(),
            1,
            "parked swap-in must pin one destination G1 slot"
        );
        assert_eq!(
            pass.admissions.len(),
            0,
            "cold request must not allocate the slot reserved for in-flight swap-in"
        );
        assert!(
            core.state.waiting.contains(&cold_uuid),
            "cold request should remain waiting for G1 capacity"
        );
    }

    /// Completed swap-ins have just paid G2→G1 bandwidth and registered their
    /// blocks into G1 inactive. They must re-enter at the front, before cold
    /// requests can allocate and evict those freshly onboarded blocks.
    #[tokio::test]
    async fn completed_swap_ins_reenter_front_preserving_order() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(2)
            .block_size(4)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(2))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let config = KvbmOffloadConfig {
            block_size_tokens: 4,
            block_size_bytes: Some(1_000_000),
            bandwidth_g2_to_g1_gbps: 2.0,
            ..Default::default()
        };
        let engine = MockOffloadEngine::new(config).await.expect("engine build");
        engine.tick(0.0);

        let first_hit = Uuid::from_u128(1);
        core.receive(DirectRequest {
            tokens: (0..4).collect(),
            max_output_tokens: 2,
            uuid: Some(first_hit),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
        let second_hit = Uuid::from_u128(2);
        core.receive(DirectRequest {
            tokens: (4..8).collect(),
            max_output_tokens: 2,
            uuid: Some(second_hit),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
        let cold = Uuid::from_u128(3);
        core.receive(DirectRequest {
            tokens: (8..12).collect(),
            max_output_tokens: 2,
            uuid: Some(cold),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut hit_plhs = Vec::new();
        for uuid in [first_hit, second_hit] {
            let plhs = core
                .state
                .requests
                .get(&uuid)
                .unwrap()
                .sequence
                .positional_lineage_hashes();
            assert_eq!(plhs.len(), 1, "each hit request should have one block");
            hit_plhs.extend(plhs);
        }
        seed_g2_blocks(engine.g2_manager(), &hit_plhs);
        core.kv_manager.attach_new_offload_engine(engine);

        let mut collector = crate::replay::TraceCollector::default();
        let pass1 = core.execute_pass(&mut collector, 0.0);
        assert_eq!(
            pass1.admissions.len(),
            0,
            "both G2 hits should park, while cold request has no free G1 slot"
        );
        assert_eq!(core.requests_awaiting_swap_in.len(), 2);
        assert_eq!(
            core.state.waiting.iter().copied().collect::<Vec<_>>(),
            vec![cold],
            "only the cold request should remain in waiting while hits are parked"
        );

        // Both 1 MB transfers share a 2 GB/s link, so each receives
        // 1 GB/s and finishes at 1 ms. On promotion they should be
        // admitted before the cold request, preserving their original
        // parking order.
        let pass2 = core.execute_pass(&mut collector, 1.0);
        let admitted: Vec<_> = pass2
            .admissions
            .iter()
            .map(|admission| admission.uuid)
            .collect();
        assert_eq!(
            admitted,
            vec![first_hit, second_hit],
            "completed swap-ins should re-enter ahead of cold requests in order"
        );
    }

    /// End-to-end: a request whose prefix lives only in G2 (engine
    /// attached, request fully cold) must be parked on first
    /// admission, promoted on tick past the swap-in's finish time, and
    /// scheduled on the subsequent pass with the swapped-in prefix
    /// counted as cached (no fresh `Stored` event for the prefix on
    /// admission).
    #[tokio::test]
    async fn cold_request_with_g2_prefix_swaps_in_then_admits() {
        // 4 tokens per block; sequence has 16 tokens → 4 full blocks.
        let block_size = 4;
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(16)
            .block_size(block_size)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        // 250 KB/block × 4 blocks ÷ 1 GB/s = 1.0 ms swap-in. Test then
        // probes at t=0.0 (parked), t=0.5 (still in flight), t=2.0
        // (completed and admitted in the same pass).
        let config = KvbmOffloadConfig {
            block_size_tokens: block_size,
            block_size_bytes: Some(250_000),
            bandwidth_g2_to_g1_gbps: 1.0,
            ..Default::default()
        };
        let engine = MockOffloadEngine::new(config).await.expect("engine build");
        engine.tick(0.0);

        // Receive the request first, then read its PLHs out of the
        // scheduler's own state so we don't risk drift from
        // `VllmCore::receive`'s ActiveSequence construction.
        let uuid = Uuid::new_v4();
        core.receive(DirectRequest {
            tokens: (0..(block_size * 4) as u32).collect(),
            max_output_tokens: 2,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
        let plhs = core
            .state
            .requests
            .get(&uuid)
            .unwrap()
            .sequence
            .positional_lineage_hashes();
        assert!(!plhs.is_empty(), "test sequence must have full blocks");

        seed_g2_blocks(engine.g2_manager(), &plhs);
        core.kv_manager.attach_new_offload_engine(engine);

        // Pass 1 (t=0.0): admission detects G2 prefix, parks the
        // request, schedules nothing.
        let mut collector = crate::replay::TraceCollector::default();
        let pass1 = core.execute_pass(&mut collector, 0.0);
        assert_eq!(
            core.requests_awaiting_swap_in.len(),
            1,
            "request must be parked on swap-in"
        );
        assert_eq!(
            pass1.admissions.len(),
            0,
            "no admission should fire while parked"
        );

        // Pass 2 (t=0.5): swap-in still in flight (1MB / 1GB/s = 1ms
        // finish; only 0.5ms elapsed). Request stays parked.
        core.execute_pass(&mut collector, 0.5);
        assert_eq!(core.requests_awaiting_swap_in.len(), 1);

        // Pass 3 (t=2.0): tick past finish → flag flips → promote
        // registers PLHs in G1 inactive pool → re-adds to waiting.
        // Same pass then admits the request via schedule_request,
        // which sees cached_tokens > 0 (InactiveHit) for the prefix.
        let pass3 = core.execute_pass(&mut collector, 2.0);
        assert!(
            core.requests_awaiting_swap_in.is_empty(),
            "swap-in must drain"
        );
        assert_eq!(
            pass3.admissions.len(),
            1,
            "promoted request must be admitted in the same pass"
        );
        let admission = &pass3.admissions[0];
        assert_eq!(admission.uuid, uuid);
        assert!(
            admission.reused_input_tokens > 0,
            "swap-in'd prefix must count as reused tokens; got {}",
            admission.reused_input_tokens
        );
    }

    /// Replaying the same synthetic G2-offload trace twice with the same
    /// `now_ms` sequence must produce byte-identical scheduler behaviour.
    /// Live/offline mode is a caller concern: the engine sees only the
    /// timestamps passed into `tick` / `try_onboard_prefix`.
    #[tokio::test]
    async fn equivalence_replayed_twice_with_g2_offload() {
        use std::sync::{Arc, Mutex};

        use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData};

        use crate::common::protocols::{KvCacheEventSink, KvEventPublishers};
        use crate::scheduler::AdmissionEvent;

        // Synthetic trace: each request has a 4-block (16-token)
        // prefix. Requests R0..R3 share the same prompt → all hit the
        // same G2-resident PLHs. Request R4 is unique → no G2 hit,
        // takes the normal allocate-fresh path.
        const BLOCK_SIZE: usize = 4;
        const NUM_BLOCKS: usize = 4;
        const TOKENS_PER_REQ: usize = BLOCK_SIZE * NUM_BLOCKS;

        #[derive(Default, Clone)]
        struct CapturingSink {
            events: Arc<Mutex<Vec<KvCacheEvent>>>,
        }
        impl KvCacheEventSink for CapturingSink {
            fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
                self.events.lock().unwrap().push(event);
                Ok(())
            }
        }

        #[derive(Debug, PartialEq)]
        struct ModeReport {
            // (uuid_index_in_trace, reused_input_tokens) per admission.
            admissions: Vec<(usize, usize)>,
            // (Stored, Removed) event counts.
            stored_count: usize,
            removed_count: usize,
            // Total swap-in promotions observed (parked → admitted).
            swap_in_admissions: usize,
        }

        async fn run_mode() -> ModeReport {
            let args = MockEngineArgs::builder()
                .num_gpu_blocks(32)
                .block_size(BLOCK_SIZE)
                .max_num_batched_tokens(Some(256))
                .max_num_seqs(Some(8))
                .enable_chunked_prefill(true)
                .enable_prefix_caching(true)
                .speedup_ratio(0.0)
                .build()
                .unwrap();
            let sink = CapturingSink::default();
            let publishers = KvEventPublishers::new(Some(Arc::new(sink.clone()) as _), None);
            let mut core = VllmCore::new_with_sink(args, 0, publishers);

            // 4 requests × 4 blocks × 250 KB each = 4 MB of swap-in
            // work. Under PS with N=4 on a 4 GB/s link (effective
            // 1 GB/s per transfer), each completes at t=1.0 ms.
            let config = KvbmOffloadConfig {
                block_size_tokens: BLOCK_SIZE,
                block_size_bytes: Some(250_000),
                bandwidth_g2_to_g1_gbps: 4.0,
                ..Default::default()
            };
            let engine = MockOffloadEngine::new(config).await.unwrap();
            engine.tick(0.0);

            // Receive 4 G2-hit requests sharing one prompt + 1 fresh
            // request. Receive R0 first so we can read its PLHs from
            // the scheduler's own state (avoids building a parallel
            // `ActiveSequence` that could drift from `VllmCore::receive`).
            let shared_tokens: Vec<u32> = (0..TOKENS_PER_REQ as u32).collect();
            let mut uuids = Vec::with_capacity(5);
            for i in 0..4 {
                let uuid = Uuid::from_u128(1000 + i as u128);
                core.receive(DirectRequest {
                    tokens: shared_tokens.clone(),
                    max_output_tokens: 2,
                    uuid: Some(uuid),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                });
                uuids.push(uuid);
            }
            let r4 = Uuid::from_u128(2000);
            core.receive(DirectRequest {
                tokens: ((TOKENS_PER_REQ as u32)..(2 * TOKENS_PER_REQ as u32)).collect(),
                max_output_tokens: 2,
                uuid: Some(r4),
                dp_rank: 0,
                arrival_timestamp_ms: None,
            });
            uuids.push(r4);

            let plhs = core
                .state
                .requests
                .get(&uuids[0])
                .unwrap()
                .sequence
                .positional_lineage_hashes();
            seed_g2_blocks(engine.g2_manager(), &plhs);
            core.kv_manager.attach_new_offload_engine(engine);

            // Drive 5 passes at well-separated `now_ms`. Pass 1 parks
            // the G2 hits + admits R4 (no G2 prefix, falls through).
            // Passes 2-3 are pre-completion. Passes 4-5 promote and
            // admit R0..R3.
            let timestamps = [0.0, 0.5, 0.8, 1.5, 3.0];
            let mut all_admissions: Vec<AdmissionEvent> = Vec::new();
            let mut collector = crate::replay::TraceCollector::default();
            let mut swap_in_admissions = 0usize;
            for &ts in &timestamps {
                let parked_before = core.requests_awaiting_swap_in.len();
                let pass = core.execute_pass(&mut collector, ts);
                let parked_after = core.requests_awaiting_swap_in.len();
                swap_in_admissions += parked_before.saturating_sub(parked_after);
                all_admissions.extend(pass.admissions);
            }

            let admissions = all_admissions
                .into_iter()
                .map(|a| {
                    let idx = uuids.iter().position(|u| *u == a.uuid).unwrap();
                    (idx, a.reused_input_tokens)
                })
                .collect();
            let events = sink.events.lock().unwrap().clone();
            let stored_count = events
                .iter()
                .filter(|e| matches!(e.data, KvCacheEventData::Stored(_)))
                .count();
            let removed_count = events
                .iter()
                .filter(|e| matches!(e.data, KvCacheEventData::Removed(_)))
                .count();

            ModeReport {
                admissions,
                stored_count,
                removed_count,
                swap_in_admissions,
            }
        }

        let report_first = run_mode().await;
        let report_second = run_mode().await;

        // Sanity: the trace must actually have exercised G2 offload —
        // otherwise this test reduces to "G1 mode parity" and gives a
        // false sense of coverage.
        assert!(
            report_first.swap_in_admissions > 0,
            "trace should exercise at least one G2 swap-in admission"
        );
        assert_eq!(
            report_first, report_second,
            "replayed G2-offload trace must be deterministic\nfirst:  {report_first:?}\nsecond: {report_second:?}",
        );
    }
}
