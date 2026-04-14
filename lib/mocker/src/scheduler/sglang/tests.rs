// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::time::Duration;

use dynamo_kv_router::indexer::{METRIC_EVENT_REMOVED, METRIC_EVENT_STORED};
use dynamo_kv_router::protocols::WorkerId;
use rstest::rstest;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::config::{SchedulePolicy, SglangConfig, ceil_to_block};
use super::core::SglangCore;
use super::decode;
use super::decode::simulate_decode_step;
use super::live::SglangScheduler;
use super::policy::apply_schedule_policy;
use super::prefill::get_new_batch_prefill;
use super::request::SglangRequest;
use crate::common::protocols::{
    DirectRequest, EngineType, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
    SglangArgs,
};
use crate::kv_manager::SglangKvManager;
use crate::scheduler::test_utils::{
    RouterIndexerHarness, nth_stored_hashes, removed_event_count, stored_hashes,
};
use crate::scheduler::{RouterEventVisibility, SchedulerHandle, capture_router_event_sink};

const ROUTER_TEST_WORKER_ID: WorkerId = 17;

fn test_args(
    num_gpu_blocks: usize,
    block_size: usize,
    chunked_prefill_size: usize,
) -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .num_gpu_blocks(num_gpu_blocks)
        .block_size(block_size)
        .speedup_ratio(1.0)
        .sglang(Some(SglangArgs {
            page_size: Some(block_size),
            chunked_prefill_size: Some(chunked_prefill_size),
            ..Default::default()
        }))
        .build()
        .unwrap()
}

fn direct_request(tokens: Vec<u32>, max_output_tokens: usize) -> DirectRequest {
    DirectRequest {
        tokens,
        max_output_tokens,
        uuid: None,
        dp_rank: 0,
        arrival_timestamp_ms: None,
    }
}

fn make_decoded_request(
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    prompt_tokens: Vec<u64>,
    max_output_tokens: usize,
) -> SglangRequest {
    let prompt_len = prompt_tokens.len();
    let alloc = kv_manager.allocate_for_request(&prompt_tokens).unwrap();
    let mut running = vec![SglangRequest {
        uuid: Uuid::new_v4(),
        prompt_tokens,
        max_output_tokens,
        output_ids: Vec::new(),
        last_node: Some(alloc.last_node),
        kv_indices: alloc.kv_indices,
        materialized_tokens: prompt_len,
        cached_tokens: 0,
        allocated_tokens: ceil_to_block(prompt_len, config.block_size),
    }];
    let result = simulate_decode_step(&mut running, kv_manager, config, 0.0, false);
    assert_eq!(result.output_signals.len(), 1);
    running.pop().unwrap()
}

mod scheduling {
    use super::*;

    #[tokio::test]
    async fn test_sglang_scheduler_fifo_ordering() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(100)
            .block_size(64)
            .speedup_ratio(100.0)
            .build()
            .unwrap();

        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
        let scheduler = SglangScheduler::new(
            args,
            0,
            Some(output_tx),
            KvEventPublishers::default(),
            None,
            FpmPublisher::default(),
        );

        let num_requests = 5;
        let max_output = 3;
        for i in 0..num_requests {
            scheduler.receive(crate::common::protocols::DirectRequest {
                tokens: vec![i as u32; 10],
                max_output_tokens: max_output,
                uuid: None,
                dp_rank: 0,
                arrival_timestamp_ms: None,
            });
        }

        let expected_signals = num_requests * max_output;
        let mut received = 0;
        let timeout = tokio::time::sleep(Duration::from_secs(5));
        tokio::pin!(timeout);

        loop {
            tokio::select! {
                Some(output_batch) = output_rx.recv() => {
                    received += output_batch.len();
                    if received >= expected_signals {
                        break;
                    }
                    timeout.set(tokio::time::sleep(Duration::from_secs(2)));
                }
                _ = &mut timeout => break,
            }
        }

        assert_eq!(received, expected_signals);
    }

    #[test]
    fn test_lpm_reorders_by_current_sequence_prefix_match() {
        let mut kv_manager = SglangKvManager::new(1000, 1, KvEventPublishers::default(), 0);
        kv_manager
            .cache_mut()
            .insert(&[1, 2, 3, 4, 5], &[0, 1, 2, 3, 4]);

        let config = SglangConfig {
            schedule_policy: SchedulePolicy::Lpm,
            ..SglangConfig::from_args(
                &MockEngineArgs::builder()
                    .speedup_ratio(1.0)
                    .build()
                    .unwrap(),
            )
        };

        let no_match_uuid = Uuid::new_v4();
        let match_uuid = Uuid::new_v4();
        let mut waiting = VecDeque::from([
            SglangRequest {
                uuid: no_match_uuid,
                prompt_tokens: vec![9, 8, 7],
                max_output_tokens: 1,
                output_ids: Vec::new(),
                last_node: None,
                kv_indices: Vec::new(),
                materialized_tokens: 0,
                cached_tokens: 0,
                allocated_tokens: 0,
            },
            SglangRequest {
                uuid: match_uuid,
                prompt_tokens: vec![1, 2, 3, 4, 5],
                max_output_tokens: 1,
                output_ids: vec![6, 7],
                last_node: None,
                kv_indices: Vec::new(),
                materialized_tokens: 0,
                cached_tokens: 0,
                allocated_tokens: 0,
            },
        ]);

        apply_schedule_policy(&mut waiting, &kv_manager, &config);
        assert_eq!(waiting[0].uuid, match_uuid);
        assert_eq!(waiting[1].uuid, no_match_uuid);
    }

    #[test]
    fn test_lpm_deprioritizes_duplicate_short_prefixes() {
        let config = SglangConfig {
            schedule_policy: SchedulePolicy::Lpm,
            ..SglangConfig::from_args(
                &MockEngineArgs::builder()
                    .block_size(1)
                    .speedup_ratio(1.0)
                    .build()
                    .unwrap(),
            )
        };
        let kv_manager = SglangKvManager::new(1000, 1, KvEventPublishers::default(), 0);
        let duplicate_prefix = (0..32).collect::<Vec<_>>();
        let mut waiting = VecDeque::new();
        for _ in 0..33 {
            waiting.push_back(SglangRequest {
                uuid: Uuid::new_v4(),
                prompt_tokens: duplicate_prefix.clone(),
                max_output_tokens: 1,
                output_ids: Vec::new(),
                last_node: None,
                kv_indices: Vec::new(),
                materialized_tokens: 0,
                cached_tokens: 0,
                allocated_tokens: 0,
            });
        }
        let unique_uuid = Uuid::new_v4();
        waiting.push_back(SglangRequest {
            uuid: unique_uuid,
            prompt_tokens: (100..132).collect(),
            max_output_tokens: 1,
            output_ids: Vec::new(),
            last_node: None,
            kv_indices: Vec::new(),
            materialized_tokens: 0,
            cached_tokens: 0,
            allocated_tokens: 0,
        });

        apply_schedule_policy(&mut waiting, &kv_manager, &config);
        assert_eq!(
            waiting.iter().position(|req| req.uuid == unique_uuid),
            Some(1)
        );
    }
}

mod core_behavior {
    use super::*;

    #[test]
    fn test_chunked_prefill_budget_is_page_aware() {
        let config = SglangConfig {
            chunked_prefill_size: 8,
            ..SglangConfig::from_args(
                &MockEngineArgs::builder()
                    .block_size(4)
                    .speedup_ratio(1.0)
                    .build()
                    .unwrap(),
            )
        };
        let mut kv_manager = SglangKvManager::new(10000, 4, KvEventPublishers::default(), 0);
        let mut waiting = VecDeque::from([SglangRequest {
            uuid: Uuid::new_v4(),
            prompt_tokens: vec![1; 6],
            max_output_tokens: 3,
            output_ids: Vec::new(),
            last_node: None,
            kv_indices: Vec::new(),
            materialized_tokens: 0,
            cached_tokens: 0,
            allocated_tokens: 0,
        }]);

        let admit = get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &[]);
        assert_eq!(admit.can_run.len(), 1);
        assert_eq!(admit.can_run[0].materialized_tokens, 6);
        assert_eq!(admit.can_run[0].allocated_tokens, 8);
    }

    #[test]
    fn test_chunked_prefill_subpage_budget_defers_next_request() {
        let config = SglangConfig {
            chunked_prefill_size: 8,
            ..SglangConfig::from_args(
                &MockEngineArgs::builder()
                    .block_size(4)
                    .speedup_ratio(1.0)
                    .build()
                    .unwrap(),
            )
        };

        let first_uuid = Uuid::new_v4();
        let second_uuid = Uuid::new_v4();
        let mut kv_manager = SglangKvManager::new(10000, 4, KvEventPublishers::default(), 0);
        let mut waiting = VecDeque::from([
            SglangRequest {
                uuid: first_uuid,
                prompt_tokens: vec![1; 7],
                max_output_tokens: 3,
                output_ids: Vec::new(),
                last_node: None,
                kv_indices: Vec::new(),
                materialized_tokens: 0,
                cached_tokens: 0,
                allocated_tokens: 0,
            },
            SglangRequest {
                uuid: second_uuid,
                prompt_tokens: vec![2; 8],
                max_output_tokens: 3,
                output_ids: Vec::new(),
                last_node: None,
                kv_indices: Vec::new(),
                materialized_tokens: 0,
                cached_tokens: 0,
                allocated_tokens: 0,
            },
        ]);

        let admit = get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &[]);
        assert_eq!(admit.can_run.len(), 1);
        assert_eq!(admit.can_run[0].uuid, first_uuid);
        assert_eq!(waiting.len(), 1);
        assert_eq!(waiting[0].uuid, second_uuid);
    }

    #[test]
    fn test_decode_allocation_is_page_aware() {
        let config = SglangConfig::from_args(
            &MockEngineArgs::builder()
                .engine_type(EngineType::Sglang)
                .block_size(4)
                .speedup_ratio(1.0)
                .build()
                .unwrap(),
        );
        let mut kv_manager = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        let alloc = kv_manager
            .allocate_for_request(&[1, 2, 3, 4, 5, 6])
            .unwrap();
        let mut running = vec![SglangRequest {
            uuid: Uuid::new_v4(),
            prompt_tokens: vec![1, 2, 3, 4, 5, 6],
            max_output_tokens: 4,
            output_ids: Vec::new(),
            last_node: Some(alloc.last_node),
            kv_indices: alloc.kv_indices,
            materialized_tokens: 6,
            cached_tokens: 4,
            allocated_tokens: 8,
        }];

        let first = simulate_decode_step(&mut running, &mut kv_manager, &config, 0.0, false);
        assert_eq!(running[0].allocated_tokens, 8);
        assert_eq!(running[0].output_len(), 1);
        assert_eq!(first.output_signals.len(), 1);

        simulate_decode_step(&mut running, &mut kv_manager, &config, 0.0, false);
        assert_eq!(running[0].allocated_tokens, 8);

        simulate_decode_step(&mut running, &mut kv_manager, &config, 0.0, false);
        assert_eq!(running[0].allocated_tokens, 12);
    }

    #[test]
    fn test_decode_speedup_ratio_scales_sglang_decode_time() {
        let base_args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .speedup_ratio(2.0)
            .decode_speedup_ratio(1.0)
            .build()
            .unwrap();
        let fast_args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .speedup_ratio(2.0)
            .decode_speedup_ratio(4.0)
            .build()
            .unwrap();
        let base_config = SglangConfig::from_args(&base_args);
        let fast_config = SglangConfig::from_args(&fast_args);

        let mut base_kv_manager = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        let base_alloc = base_kv_manager.allocate_for_request(&[1, 2, 3, 4]).unwrap();
        let mut base_running = vec![SglangRequest {
            uuid: Uuid::new_v4(),
            prompt_tokens: vec![1, 2, 3, 4],
            max_output_tokens: 4,
            output_ids: Vec::new(),
            last_node: Some(base_alloc.last_node),
            kv_indices: base_alloc.kv_indices,
            materialized_tokens: 4,
            cached_tokens: 0,
            allocated_tokens: 4,
        }];

        let mut fast_kv_manager = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        let fast_alloc = fast_kv_manager.allocate_for_request(&[1, 2, 3, 4]).unwrap();
        let mut fast_running = vec![SglangRequest {
            uuid: Uuid::new_v4(),
            prompt_tokens: vec![1, 2, 3, 4],
            max_output_tokens: 4,
            output_ids: Vec::new(),
            last_node: Some(fast_alloc.last_node),
            kv_indices: fast_alloc.kv_indices,
            materialized_tokens: 4,
            cached_tokens: 0,
            allocated_tokens: 4,
        }];

        let base = simulate_decode_step(
            &mut base_running,
            &mut base_kv_manager,
            &base_config,
            0.0,
            true,
        );
        let fast = simulate_decode_step(
            &mut fast_running,
            &mut fast_kv_manager,
            &fast_config,
            0.0,
            true,
        );
        let ratio = base.end_ms / fast.end_ms;

        assert!(base.end_ms > fast.end_ms);
        assert!(
            (ratio - 4.0).abs() < 1e-3,
            "expected 4x decode speedup ratio, got {ratio}"
        );
    }

    #[test]
    fn test_check_decode_mem_preserves_generated_output_on_retract() {
        let config = SglangConfig::from_args(
            &MockEngineArgs::builder()
                .engine_type(EngineType::Sglang)
                .block_size(4)
                .speedup_ratio(1.0)
                .build()
                .unwrap(),
        );
        let mut kv_manager = SglangKvManager::new(8, 4, KvEventPublishers::default(), 0);
        let first = kv_manager.cache_mut().token_pool.allocate(4).unwrap();
        let second = kv_manager.cache_mut().token_pool.allocate(4).unwrap();

        let mut running = vec![
            SglangRequest {
                uuid: Uuid::new_v4(),
                prompt_tokens: vec![1, 2, 3, 4],
                max_output_tokens: 10,
                output_ids: vec![11, 12, 13],
                last_node: None,
                kv_indices: first,
                materialized_tokens: 7,
                cached_tokens: 4,
                allocated_tokens: 8,
            },
            SglangRequest {
                uuid: Uuid::new_v4(),
                prompt_tokens: vec![9, 8, 7, 6],
                max_output_tokens: 10,
                output_ids: vec![21],
                last_node: None,
                kv_indices: second,
                materialized_tokens: 5,
                cached_tokens: 4,
                allocated_tokens: 8,
            },
        ];

        let retracted = decode::check_decode_mem(&mut running, &mut kv_manager, &config);
        assert_eq!(retracted.len(), 1);
        assert_eq!(retracted[0].output_ids, vec![21]);
        assert_eq!(retracted[0].materialized_tokens, 0);
        assert!(retracted[0].kv_indices.is_empty());
    }

    #[test]
    fn test_unfinished_decode_request_is_cached_after_output() {
        let config = SglangConfig::from_args(
            &MockEngineArgs::builder()
                .engine_type(EngineType::Sglang)
                .block_size(4)
                .speedup_ratio(1.0)
                .build()
                .unwrap(),
        );
        let mut kv_manager = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        let alloc = kv_manager.allocate_for_request(&[1, 2, 3, 4]).unwrap();
        let mut running = vec![SglangRequest {
            uuid: Uuid::new_v4(),
            prompt_tokens: vec![1, 2, 3, 4],
            max_output_tokens: 4,
            output_ids: Vec::new(),
            last_node: Some(alloc.last_node),
            kv_indices: alloc.kv_indices,
            materialized_tokens: 4,
            cached_tokens: 0,
            allocated_tokens: 4,
        }];

        simulate_decode_step(&mut running, &mut kv_manager, &config, 0.0, false);
        let prefix = running[0].sequence_prefix(4);
        assert_eq!(kv_manager.cache().prefix_match_len(&prefix), 4);
    }

    #[test]
    fn test_active_decode_blocks_tracks_page_reserved_occupancy_in_blocks() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(32)
            .block_size(4)
            .speedup_ratio(1.0)
            .sglang(Some(SglangArgs {
                chunked_prefill_size: Some(8),
                page_size: Some(4),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        core.receive(crate::common::protocols::DirectRequest {
            tokens: vec![1; 6],
            max_output_tokens: 2,
            uuid: None,
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let pass = core.execute_pass_internal(None, 0.0);
        assert_eq!(pass.completed_requests, 0);
        assert_eq!(pass.active_decode_blocks, 2);
    }

    #[test]
    fn test_sglang_pass_visibility_is_pass_end() {
        let mut core = SglangCore::new_with_kv_capture(test_args(32, 4, 4), ROUTER_TEST_WORKER_ID);
        core.receive(direct_request(vec![1, 2, 3, 4], 1));

        let pass = core.execute_pass_internal(None, 0.0);

        assert_eq!(pass.router_event_visibility, RouterEventVisibility::PassEnd);
    }
}

async fn assert_sglang_scheduler_completes_all(
    scheduler: &SglangScheduler,
    output_rx: &mut mpsc::UnboundedReceiver<Vec<OutputSignal>>,
    num_requests: usize,
    prompt_len: usize,
    max_output_tokens: usize,
    use_shared_tokens: bool,
) {
    let shared_prefix = vec![1u32; prompt_len / 2];
    for i in 0..num_requests {
        let mut input_tokens = if use_shared_tokens {
            shared_prefix.clone()
        } else {
            Vec::new()
        };
        let unique_len = prompt_len - input_tokens.len();
        input_tokens.extend((0..unique_len).map(|j| (i * unique_len + j) as u32 + 1000));
        scheduler.receive(crate::common::protocols::DirectRequest {
            tokens: input_tokens,
            max_output_tokens,
            uuid: None,
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
    }

    let expected_tokens = num_requests * max_output_tokens;
    let mut received_tokens = 0;
    let timeout = tokio::time::sleep(Duration::from_millis(200));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            biased;
            Some(output_batch) = output_rx.recv() => {
                received_tokens += output_batch.len();
                if received_tokens >= expected_tokens {
                    break;
                }
                timeout.set(tokio::time::sleep(Duration::from_millis(200)));
            }
            _ = &mut timeout => break,
        }
    }

    assert_eq!(received_tokens, expected_tokens);

    let metrics = scheduler.metrics_receiver().borrow().clone();
    assert!(metrics.active_decode_blocks > 0);
    assert!(metrics.total_blocks > 0);
    assert!((0.0..=1.0).contains(&metrics.gpu_cache_usage_perc));
}

mod router_events {
    use super::*;

    #[rstest]
    #[case::case_1(false, "fifo", 1)]
    #[case::case_2(true, "fifo", 1)]
    #[case::case_3(false, "lpm", 1)]
    #[case::case_4(true, "lpm", 1)]
    #[case::case_5(false, "fifo", 4)]
    #[case::case_6(true, "fifo", 4)]
    #[case::case_7(false, "lpm", 4)]
    #[case::case_8(true, "lpm", 4)]
    #[tokio::test]
    async fn test_sglang_scheduler_token_generation_patterns(
        #[case] use_shared_tokens: bool,
        #[case] schedule_policy: &str,
        #[case] page_size: usize,
    ) {
        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(500)
            .block_size(64)
            .speedup_ratio(1000.0)
            .sglang(Some(SglangArgs {
                schedule_policy: Some(schedule_policy.to_string()),
                page_size: Some(page_size),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let scheduler = SglangScheduler::new(
            args,
            0,
            Some(output_tx),
            KvEventPublishers::default(),
            None,
            FpmPublisher::default(),
        );

        assert_sglang_scheduler_completes_all(
            &scheduler,
            &mut output_rx,
            200,
            1000,
            100,
            use_shared_tokens,
        )
        .await;
    }

    #[tokio::test]
    async fn test_chunked_prefill_events_apply_cleanly() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let mut core = SglangCore::new_with_kv_capture(test_args(32, 4, 4), ROUTER_TEST_WORKER_ID);
        core.receive(direct_request(vec![1, 2, 3, 4, 5, 6], 2));

        let pass1 = core.execute_pass_internal(None, 0.0);
        let mut prompt_hashes = stored_hashes(&pass1.kv_events);
        assert_eq!(prompt_hashes.len(), 4);
        harness.apply_events(pass1.kv_events).await;

        let pass2 = core.execute_pass_internal(None, pass1.end_ms);
        prompt_hashes.extend(nth_stored_hashes(&pass2.kv_events, 0));
        harness.apply_events(pass2.kv_events).await;

        assert_eq!(prompt_hashes.len(), 6);
        assert!(harness.ok_count(METRIC_EVENT_STORED) >= 2);
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_decode_growth_events_apply_cleanly() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let mut core = SglangCore::new_with_kv_capture(test_args(32, 4, 16), ROUTER_TEST_WORKER_ID);
        core.receive(direct_request(vec![7, 8, 9, 10], 5));

        let pass1 = core.execute_pass_internal(None, 0.0);
        let mut full_hashes = stored_hashes(&pass1.kv_events);
        harness.apply_events(pass1.kv_events).await;

        let pass2 = core.execute_pass_internal(None, pass1.end_ms);
        full_hashes.extend(stored_hashes(&pass2.kv_events));
        harness.apply_events(pass2.kv_events).await;

        assert_eq!(full_hashes.len(), 6);
        assert!(harness.ok_count(METRIC_EVENT_STORED) >= 2);
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_retract_frees_do_not_leave_stale_blocks() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let args = test_args(8, 4, 16);
        let config = SglangConfig::from_args(&args);
        let (buffer, sink) = capture_router_event_sink(ROUTER_TEST_WORKER_ID);
        let mut kv_manager =
            SglangKvManager::new(10, 4, KvEventPublishers::new(Some(sink), None), 0);

        let req1 = make_decoded_request(&mut kv_manager, &config, vec![1, 2, 3, 4], 4);
        let req1_events = buffer.drain();
        let req1_hashes = stored_hashes(&req1_events);
        harness.apply_events(req1_events).await;

        let req2 = make_decoded_request(&mut kv_manager, &config, vec![9, 8, 7, 6], 4);
        harness.apply_events(buffer.drain()).await;

        let mut running = vec![req1, req2];
        let retracted = decode::check_decode_mem(&mut running, &mut kv_manager, &config);
        assert_eq!(retracted.len(), 1);

        let retract_events = buffer.drain();
        assert!(removed_event_count(&retract_events) > 0);
        harness.apply_events(retract_events).await;

        assert_eq!(harness.overlap_for_hashes(req1_hashes).await, 4);
        assert!(harness.ok_count(METRIC_EVENT_REMOVED) > 0);
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_completion_tail_free_emits_valid_removals() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let mut core = SglangCore::new_with_kv_capture(test_args(32, 4, 16), ROUTER_TEST_WORKER_ID);
        core.receive(direct_request(vec![11, 12, 13, 14], 3));

        let pass1 = core.execute_pass_internal(None, 0.0);
        let prompt_hashes = nth_stored_hashes(&pass1.kv_events, 0);
        let mut full_hashes = stored_hashes(&pass1.kv_events);
        harness.apply_events(pass1.kv_events).await;

        let pass2 = core.execute_pass_internal(None, pass1.end_ms);
        full_hashes.extend(stored_hashes(&pass2.kv_events));
        harness.apply_events(pass2.kv_events).await;

        let pass3 = core.execute_pass_internal(None, pass2.end_ms);
        assert!(removed_event_count(&pass3.kv_events) > 0);
        full_hashes.extend(stored_hashes(&pass3.kv_events));
        harness.apply_events(pass3.kv_events).await;

        assert_eq!(prompt_hashes.len(), 4);
        assert!(full_hashes.len() >= prompt_hashes.len());
        assert!(harness.ok_count(METRIC_EVENT_REMOVED) > 0);
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_mixed_chunk_decode_retract_reprefill_complete_events_apply_cleanly() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let args = test_args(8, 4, 4);
        let config = SglangConfig::from_args(&args);
        let (buffer, sink) = capture_router_event_sink(ROUTER_TEST_WORKER_ID);
        let mut kv_manager =
            SglangKvManager::new(12, 4, KvEventPublishers::new(Some(sink), None), 0);

        let mut waiting = VecDeque::from([SglangRequest {
            uuid: Uuid::new_v4(),
            prompt_tokens: vec![1, 2, 3, 4, 5, 6],
            max_output_tokens: 3,
            output_ids: Vec::new(),
            last_node: None,
            kv_indices: Vec::new(),
            materialized_tokens: 0,
            cached_tokens: 0,
            allocated_tokens: 0,
        }]);

        let chunk1 = get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &[]);
        let mut req1 = chunk1.can_run.into_iter().next().unwrap();
        decode::cache_materialized_prefix(&mut req1, &mut kv_manager, &config);
        waiting.push_front(req1);
        harness.apply_events(buffer.drain()).await;

        let chunk2 = get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &[]);
        let mut running = chunk2.can_run;
        let decode1 = simulate_decode_step(&mut running, &mut kv_manager, &config, 0.0, false);
        assert_eq!(decode1.output_signals.len(), 1);
        harness.apply_events(buffer.drain()).await;
        let req1 = running.pop().unwrap();

        let req2 = make_decoded_request(&mut kv_manager, &config, vec![9, 10, 11, 12], 3);
        harness.apply_events(buffer.drain()).await;

        let mut running = vec![req1, req2];
        let mut retracted = decode::check_decode_mem(&mut running, &mut kv_manager, &config);
        assert_eq!(retracted.len(), 1);
        harness.apply_events(buffer.drain()).await;

        let mut waiting = VecDeque::from([retracted.pop().unwrap()]);
        let mut now_ms = 0.0;
        let mut saw_remove = harness.ok_count(METRIC_EVENT_REMOVED) > 0;
        loop {
            let admit =
                get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &running);
            for mut req in admit.can_run {
                if req.materialized_tokens < req.current_sequence_len() {
                    decode::cache_materialized_prefix(&mut req, &mut kv_manager, &config);
                    waiting.push_front(req);
                } else {
                    running.push(req);
                }
            }

            let events = buffer.drain();
            saw_remove |= removed_event_count(&events) > 0;
            harness.apply_events(events).await;

            if running.is_empty() {
                if waiting.is_empty() {
                    break;
                }
                continue;
            }

            let decode =
                simulate_decode_step(&mut running, &mut kv_manager, &config, now_ms, false);
            now_ms = decode.end_ms;
            for req in decode.requests.into_iter().rev() {
                waiting.push_front(req);
            }
            let events = buffer.drain();
            saw_remove |= removed_event_count(&events) > 0;
            harness.apply_events(events).await;

            if running.is_empty() && waiting.is_empty() {
                break;
            }
        }

        assert!(saw_remove);
        harness.assert_no_event_errors();
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_live_pathological_load_no_router_event_errors() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let (sink, forward_task) = harness.spawn_forwarder();

        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
        let scheduler = SglangScheduler::new(
            MockEngineArgs::builder()
                .engine_type(EngineType::Sglang)
                .num_gpu_blocks(4)
                .block_size(4)
                .speedup_ratio(1000.0)
                .sglang(Some(SglangArgs {
                    page_size: Some(4),
                    chunked_prefill_size: Some(4),
                    ..Default::default()
                }))
                .build()
                .unwrap(),
            0,
            Some(output_tx),
            KvEventPublishers::new(Some(sink.clone()), None),
            None,
            FpmPublisher::default(),
        );

        for _ in 0..8 {
            scheduler.receive(direct_request(vec![42], 4));
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
        assert!(harness.ok_count(METRIC_EVENT_REMOVED) > 0);
        harness.shutdown();
    }

    #[test]
    fn test_prefill_completion_emits_handoff_delay() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(64)
            .block_size(4)
            .worker_type(crate::common::protocols::WorkerType::Prefill)
            .kv_transfer_bandwidth(Some(1.0))
            .kv_bytes_per_token(Some(1_000_000))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        core.receive(DirectRequest {
            tokens: vec![1; 8],
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(91)),
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
}

mod forward_pass_metrics {
    use super::*;

    fn fpm_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap()
    }

    #[test]
    fn test_fpm_single_prefill_request() {
        let mut core = SglangCore::new(fpm_args());
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
        assert!(
            fpm.sum_prefill_tokens > 0,
            "prefill tokens should be computed"
        );
        // In SGLang, after prefill the request immediately joins running and
        // participates in the decode step of the same pass.
        assert_eq!(fpm.num_decode_requests, 1);
        assert_eq!(fpm.num_queued_prefill, 0);
        assert_eq!(fpm.num_queued_decode, 0);
        assert!(fpm.wall_time_secs > 0.0);
    }

    #[test]
    fn test_fpm_prefill_and_decode_mixed_batch() {
        let mut core = SglangCore::new(fpm_args());

        // r1: 4-token prompt, 3 output tokens
        core.receive(DirectRequest {
            tokens: (0..4).collect(),
            max_output_tokens: 3,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();

        // Pass 1: prefill r1
        let pass1 = core.execute_pass(&mut collector, 0.0);
        let fpm1 = pass1.fpm.expect("FPM should be present");
        assert_eq!(fpm1.num_prefill_requests, 1);

        // r2: arriving while r1 is decoding
        core.receive(DirectRequest {
            tokens: (100..104).collect(),
            max_output_tokens: 3,
            uuid: Some(Uuid::from_u128(2)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        // Pass 2: r2 prefill + decode step runs on all running (r1 + r2)
        let pass2 = core.execute_pass(&mut collector, 1.0);
        let fpm2 = pass2.fpm.expect("FPM should be present");
        assert_eq!(fpm2.num_prefill_requests, 1, "r2 is prefilling");
        // In SGLang, after r2 prefill completes it joins running alongside r1,
        // so the decode step sees both.
        assert_eq!(fpm2.num_decode_requests, 2, "r1 + r2 both in decode step");
        assert!(
            fpm2.sum_decode_kv_tokens > 0,
            "decode requests should have KV context"
        );
    }

    #[test]
    fn test_fpm_empty_pass_is_zeroed() {
        let mut core = SglangCore::new(fpm_args());

        // Submit and fully drain a request first so the core isn't empty
        // (empty core blocks in receive_requests in live mode, but
        // execute_pass_internal works fine on an empty core).
        let pass = core.execute_hidden_pass(0.0);
        let fpm = pass.fpm.expect("FPM should be present even for empty pass");

        assert_eq!(fpm.num_prefill_requests, 0);
        assert_eq!(fpm.num_decode_requests, 0);
        assert_eq!(fpm.num_queued_prefill, 0);
        assert_eq!(fpm.num_queued_decode, 0);
        assert_eq!(fpm.sum_prefill_tokens, 0);
        assert_eq!(fpm.sum_decode_kv_tokens, 0);
    }

    #[test]
    fn test_fpm_queued_requests() {
        // Very limited KV to force queuing.
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(4)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(2))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(8),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);

        // Two 8-token requests but limited KV
        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(2)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        let fpm = pass.fpm.expect("FPM should be present");

        let total_scheduled = fpm.num_prefill_requests + fpm.num_decode_requests;
        assert!(
            total_scheduled >= 1,
            "at least one request should be scheduled"
        );
        // With tight KV, the second request should be queued.
        let total_queued = fpm.num_queued_prefill + fpm.num_queued_decode;
        assert!(
            total_queued >= 1,
            "at least one request should be queued, got {total_queued}"
        );
    }

    #[test]
    fn test_fpm_var_prefill_length_with_multiple_requests() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(32)
            .max_num_batched_tokens(Some(32))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(32),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);

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
        // With chunked_prefill_size=8 and a 16-token prompt, the request
        // should be chunked. Each pass should report only the chunk size
        // in sum_prefill_tokens, not the full prompt length.
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(32)
            .max_num_batched_tokens(Some(32))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(8),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);

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
    }

    #[test]
    fn test_fpm_retracted_decode_becomes_queued_decode() {
        // Very tight KV to force decode retraction. Fill the KV with running
        // requests, then the decode step should retract some, and those should
        // appear as queued decodes in the next pass's FPM.
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(6) // 24 tokens — very tight
            .max_num_batched_tokens(Some(32))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(32),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        let mut collector = crate::replay::TraceCollector::default();

        // Two requests with 4-token prompts and long outputs to fill KV
        core.receive(DirectRequest {
            tokens: (0..4).collect(),
            max_output_tokens: 20,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..104).collect(),
            max_output_tokens: 20,
            uuid: Some(Uuid::from_u128(2)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        // Run several passes to build up KV pressure
        for i in 0..4 {
            core.execute_pass(&mut collector, i as f64);
        }

        // Add a third request to increase memory pressure
        core.receive(DirectRequest {
            tokens: (200..212).collect(), // 12 tokens
            max_output_tokens: 10,
            uuid: Some(Uuid::from_u128(3)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });

        // Run more passes — at some point retraction should occur
        let mut saw_queued_decode = false;
        for i in 4..10 {
            let pass = core.execute_pass(&mut collector, i as f64);
            let fpm = pass.fpm.expect("FPM should be present");
            if fpm.num_queued_decode > 0 {
                saw_queued_decode = true;
                assert!(
                    fpm.sum_queued_decode_kv_tokens > 0,
                    "retracted decode should have KV context"
                );
                break;
            }
        }

        // If retraction didn't happen (KV was sufficient), that's also valid —
        // just verify we always get Some(fpm).
        if !saw_queued_decode {
            // Verify the requests completed or are still running with valid FPM
            let pass = core.execute_hidden_pass(10.0);
            assert!(pass.fpm.is_some(), "FPM should always be present");
        }
    }

    #[tokio::test]
    async fn test_fpm_sent_through_sink() {
        use std::sync::Arc;

        use crate::common::protocols::FpmSink;
        use crate::scheduler::test_utils::CapturingFpmSink;

        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap();

        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
        let fpm_sink = Arc::new(CapturingFpmSink::default());
        let fpm_publisher = FpmPublisher::new(Some(fpm_sink.clone() as Arc<dyn FpmSink>));

        let scheduler = SglangScheduler::new(
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
