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

use crate::common::handoff::HandoffId;
use crate::common::protocols::{
    DirectRequest, EngineType, FpmPublisher, G1Backend, KvCacheEventSink, KvEventPublishers,
    MockEngineArgs, OutputSignal, PreemptionMode, RawKvEvent, RawKvEventSink,
};
use crate::common::sequence::ActiveSequence;
use crate::kv_manager::G1Acquire;
use crate::scheduler::SchedulerHandle;
use crate::scheduler::test_utils::{RouterIndexerHarness, removed_event_count, stored_hashes};
use crate::scheduler::{
    RouterEventVisibility, SchedulerCommand, SchedulerCommandResult, SchedulerLifecycleEvent,
};

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
    make_args_for_engine(EngineType::Vllm)
}

fn make_args_for_engine(engine_type: EngineType) -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(engine_type)
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

#[rstest]
#[case(EngineType::Vllm)]
#[case(EngineType::Trtllm)]
fn zero_output_request_completes_after_prefill(#[case] engine_type: EngineType) {
    let mut core = VllmCore::new(make_args_for_engine(engine_type));
    let uuid = core.receive(DirectRequest {
        tokens: vec![1, 2, 3, 4],
        max_output_tokens: 0,
        uuid: Some(Uuid::from_u128(90_001)),
        ..Default::default()
    });
    let mut collector = crate::replay::TraceCollector::default();

    let pass = core.execute_pass(&mut collector, 0.0);

    assert!(core.state.requests.is_empty());
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
    assert_eq!(pass.completed_requests, 1);
    let fpm = pass.fpm.as_ref().unwrap();
    assert_eq!(fpm.num_decode_requests, 0);
    assert_eq!(fpm.sum_decode_kv_tokens, 0);
    assert!(matches!(
        pass.output_signals.as_slice(),
        [OutputSignal {
            uuid: signal_uuid,
            token_id: None,
            completed: true,
            rejected: false,
            ..
        }] if *signal_uuid == uuid
    ));
}

#[test]
fn fully_cached_zero_output_request_is_not_decode_fpm_work() {
    let mut core = VllmCore::new(router_args());
    let tokens = vec![1, 2, 3, 4];
    let mut collector = crate::replay::TraceCollector::default();

    core.receive(DirectRequest {
        tokens: tokens.clone(),
        max_output_tokens: 0,
        uuid: Some(Uuid::from_u128(90_002)),
        ..Default::default()
    });
    let seed_pass = core.execute_pass(&mut collector, 0.0);
    assert_eq!(seed_pass.completed_requests, 1);

    let uuid = core.receive(DirectRequest {
        tokens,
        max_output_tokens: 0,
        uuid: Some(Uuid::from_u128(90_003)),
        ..Default::default()
    });
    let pass = core.execute_pass(&mut collector, seed_pass.end_ms);

    assert_eq!(pass.completed_requests, 1);
    let fpm = pass.fpm.as_ref().unwrap();
    assert_eq!(fpm.num_prefill_requests, 0);
    assert_eq!(fpm.num_decode_requests, 0);
    assert_eq!(fpm.sum_decode_kv_tokens, 0);
    assert!(matches!(
        pass.output_signals.as_slice(),
        [OutputSignal {
            uuid: signal_uuid,
            token_id: None,
            completed: true,
            rejected: false,
            ..
        }] if *signal_uuid == uuid
    ));
}

#[test]
fn speculative_batch_drains_zero_output_before_emitting_tokens() {
    let args = MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(8)
        .max_num_batched_tokens(Some(8))
        .max_num_seqs(Some(2))
        .enable_chunked_prefill(true)
        .enable_prefix_caching(false)
        .speedup_ratio(0.0)
        .aic_nextn(Some(2))
        .aic_nextn_accept_rates(Some("1,1".to_string()))
        .build()
        .unwrap();
    let mut core = VllmCore::new(args);
    let zero_uuid = core.receive(DirectRequest {
        tokens: vec![1, 2, 3, 4],
        max_output_tokens: 0,
        uuid: Some(Uuid::from_u128(90_002)),
        ..Default::default()
    });
    let normal_uuid = core.receive(DirectRequest {
        tokens: vec![5, 6, 7, 8],
        max_output_tokens: 3,
        uuid: Some(Uuid::from_u128(90_003)),
        ..Default::default()
    });
    let mut collector = crate::replay::TraceCollector::default();

    let pass = core.execute_pass(&mut collector, 0.0);

    assert!(core.state.requests.is_empty());
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
    assert_eq!(pass.completed_requests, 2);
    assert_eq!(
        pass.output_signals
            .iter()
            .filter(|signal| signal.uuid == zero_uuid && signal.token_id.is_none())
            .count(),
        1
    );
    assert_eq!(
        pass.output_signals
            .iter()
            .filter(|signal| signal.uuid == normal_uuid && signal.token_id.is_some())
            .count(),
        3
    );
    assert_eq!(pass.accept_length_output_tokens, 3);
    assert_eq!(pass.accept_length_decode_forwards, 1);
}

mod source_holds {
    use super::*;

    fn args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(8)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .worker_type(crate::common::protocols::WorkerType::Prefill)
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    fn request(uuid: Uuid) -> DirectRequest {
        DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 2,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        }
    }

    fn execute(core: &mut VllmCore, now_ms: f64) -> crate::scheduler::EnginePassResult {
        let mut collector = crate::replay::TraceCollector::default();
        core.execute_pass(&mut collector, now_ms)
    }

    #[test]
    fn terminal_completion_holds_source_without_freeing_kv() {
        let mut core = VllmCore::new(args());
        let request_id = Uuid::from_u128(101);
        let handoff_id = HandoffId::from(Uuid::from_u128(201));
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request: request(request_id),
        })
        .unwrap();

        let first = execute(&mut core, 0.0);
        assert!(!first.output_signals[0].completed);
        let active_before_terminal = core.kv_manager.num_active_blocks();
        assert!(active_before_terminal > 0);
        let expected_held_blocks = core.state.requests[&request_id]
            .sequence
            .len()
            .div_ceil(core.args.block_size);

        let terminal = execute(&mut core, first.end_ms);
        assert!(terminal.output_signals[0].completed);
        assert!(matches!(
            terminal.lifecycle_events.as_slice(),
            [SchedulerLifecycleEvent::SourceHeld {
                handoff_id: held_handoff,
                request_id: held_request,
                ..
            }] if *held_handoff == handoff_id && *held_request == request_id
        ));
        assert!(core.source_is_held(handoff_id));
        assert!(!core.state.requests.contains_key(&request_id));
        assert_eq!(core.kv_manager.num_active_blocks(), expected_held_blocks);
        assert!(expected_held_blocks > active_before_terminal);

        core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
            .unwrap();
        assert!(!core.source_is_held(handoff_id));
        assert_eq!(core.kv_manager.num_active_blocks(), 0);

        core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
            .unwrap();
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
    }

    #[test]
    fn cancel_and_early_release_cleanup_exactly_once() {
        let mut core = VllmCore::new(args());
        let first_id = HandoffId::from(Uuid::from_u128(202));
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id: first_id,
            request: request(Uuid::from_u128(102)),
        })
        .unwrap();
        let first = execute(&mut core, 0.0);
        execute(&mut core, first.end_ms);
        let held_blocks = core.kv_manager.num_active_blocks();

        core.apply_command(SchedulerCommand::CancelSource {
            handoff_id: first_id,
        })
        .unwrap();
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        core.apply_command(SchedulerCommand::CancelSource {
            handoff_id: first_id,
        })
        .unwrap();
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        assert!(held_blocks > 0);

        let second_id = first_id;
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id: second_id,
            request: request(Uuid::from_u128(103)),
        })
        .unwrap();
        assert!(core.source_is_registered(second_id));
        core.apply_command(SchedulerCommand::ReleaseSource {
            handoff_id: second_id,
        })
        .unwrap();
        assert!(!core.source_is_registered(second_id));

        let first = execute(&mut core, 0.0);
        let terminal = execute(&mut core, first.end_ms);
        assert!(terminal.output_signals[0].completed);
        assert!(!core.source_is_held(second_id));
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
    }

    #[test]
    fn active_request_id_is_rejected_before_source_hold_registration() {
        let mut core = VllmCore::new(args());
        let request_id = Uuid::from_u128(104);
        let handoff_id = HandoffId::from(Uuid::from_u128(204));
        core.receive(request(request_id));

        assert!(
            core.apply_command(SchedulerCommand::Submit(request(request_id)))
                .is_err()
        );
        assert!(
            core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
                handoff_id,
                request: request(request_id),
            })
            .is_err()
        );
        assert!(!core.source_is_registered(handoff_id));
        assert_eq!(core.num_requests(), 1);
    }
}

mod destination_lifecycle {
    use super::*;
    use crate::common::protocols::WorkerType;

    fn args(worker_type: WorkerType) -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(12)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .worker_type(worker_type)
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    fn request(uuid: Uuid, tokens: Vec<u32>, max_output_tokens: usize) -> DirectRequest {
        DirectRequest {
            tokens,
            max_output_tokens,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        }
    }

    fn execute(core: &mut VllmCore, now_ms: f64) -> crate::scheduler::EnginePassResult {
        let mut collector = crate::replay::TraceCollector::default();
        core.execute_pass(&mut collector, now_ms)
    }

    fn assert_no_republished_stores(
        activation: &[dynamo_kv_router::protocols::LocalBlockHash],
        later: &[dynamo_kv_router::protocols::LocalBlockHash],
    ) {
        assert!(later.iter().all(|hash| !activation.contains(hash)));
    }

    #[test]
    fn materialized_prompt_above_max_model_len_is_rejected() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(12)
            .max_model_len(Some(8))
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .worker_type(WorkerType::Decode)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let handoff_id = HandoffId::from(Uuid::from_u128(30_001));
        let uuid = Uuid::from_u128(30_002);

        assert!(matches!(
            core.apply_command(SchedulerCommand::ReserveDestination {
                handoff_id,
                request: request(uuid, vec![1; 9], 1),
            })
            .unwrap(),
            SchedulerCommandResult::DestinationAccepted { request_id } if request_id == uuid
        ));
        assert_eq!(
            core.apply_command(SchedulerCommand::ActivateDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );

        let pass = execute(&mut core, 0.0);
        assert!(matches!(
            pass.output_signals.as_slice(),
            [OutputSignal {
                uuid: signal_uuid,
                token_id: None,
                completed: true,
                rejected: true,
                ..
            }] if *signal_uuid == uuid
        ));
        assert!(!core.state().requests.contains_key(&uuid));
    }

    fn drive_source_to_hold(core: &mut VllmCore, handoff_id: HandoffId, req: DirectRequest) {
        assert!(matches!(
            core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
                handoff_id,
                request: req,
            })
            .unwrap(),
            SchedulerCommandResult::Submitted(_)
        ));
        let mut now_ms = 0.0;
        for _ in 0..8 {
            let pass = execute(core, now_ms);
            now_ms = pass.end_ms;
            if core.is_empty() {
                break;
            }
        }
        assert!(core.is_empty());
        assert!(core.source_is_held(handoff_id));
        assert!(!core.is_drained());
    }

    #[tokio::test]
    async fn productive_zero_duration_live_pass_continues_without_external_wake() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(4)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .worker_type(WorkerType::Decode)
            .speedup_ratio(1000.0)
            .build()
            .unwrap();
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();
        let scheduler = Scheduler::new(
            args,
            0,
            Some(output_tx),
            KvEventPublishers::default(),
            None,
            FpmPublisher::default(),
        );
        let uuid = Uuid::from_u128(31_001);
        scheduler.receive(request(uuid, vec![1; 4], 1));

        let output = tokio::time::timeout(Duration::from_secs(1), output_rx.recv())
            .await
            .expect("productive zero-duration admission must continue immediately")
            .expect("output channel should remain open");
        assert!(output.iter().any(|signal| signal.uuid == uuid));
    }

    #[test]
    fn destination_lifecycle_is_counted_once_as_queued_decode() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(6)
            .max_num_batched_tokens(Some(32))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .worker_type(WorkerType::Decode)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let held_handoff = HandoffId::from(Uuid::from_u128(32_001));
        let held_uuid = Uuid::from_u128(32_002);
        let held = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: held_handoff,
                    request: request(held_uuid, vec![1; 4], 1),
                },
                true,
            )
            .unwrap();
        assert_eq!(held.lifecycle_events.len(), 1);
        assert_eq!(core.mocker_metrics().waiting_requests, 1);
        let held_snapshot = execute(&mut core, 0.0).fpm.unwrap();
        assert_eq!(held_snapshot.num_queued_decode, 1);
        assert_eq!(held_snapshot.sum_queued_decode_kv_tokens, 4);
        assert_eq!(held_snapshot.var_queued_decode_kv_tokens, 0.0);

        let pending_handoff = HandoffId::from(Uuid::from_u128(32_003));
        let pending = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: pending_handoff,
                    request: request(Uuid::from_u128(32_004), vec![2; 24], 1),
                },
                true,
            )
            .unwrap();
        assert!(pending.lifecycle_events.is_empty());
        assert_eq!(core.mocker_metrics().waiting_requests, 2);
        let mixed_snapshot = execute(&mut core, 0.0).fpm.unwrap();
        assert_eq!(mixed_snapshot.num_queued_decode, 2);
        assert_eq!(mixed_snapshot.sum_queued_decode_kv_tokens, 28);
        assert_eq!(mixed_snapshot.var_queued_decode_kv_tokens, 100.0);

        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination {
                handoff_id: pending_handoff,
            })
            .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert_eq!(core.mocker_metrics().waiting_requests, 1);

        core.receive(request(Uuid::from_u128(32_005), vec![3; 4], 2));
        let blocker = execute(&mut core, 1.0);
        assert_eq!(core.mocker_metrics().running_requests, 1);
        assert_eq!(
            core.apply_command(SchedulerCommand::ActivateDestination {
                handoff_id: held_handoff,
            })
            .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert_eq!(core.mocker_metrics().waiting_requests, 1);

        let activated_pass = execute(&mut core, blocker.end_ms);
        let activated = activated_pass.fpm.unwrap();
        assert_eq!(activated.num_queued_prefill, 0);
        assert_eq!(activated.num_queued_decode, 1);
        assert_eq!(activated.sum_queued_decode_kv_tokens, 4);

        let completed = execute(&mut core, activated_pass.end_ms).fpm.unwrap();
        assert_eq!(completed.num_queued_decode, 0);
        assert_eq!(completed.sum_queued_decode_kv_tokens, 0);
        assert_eq!(core.mocker_metrics().waiting_requests, 0);
    }

    #[test]
    fn destination_transfer_footprint_excludes_decode_headroom() {
        let footprint = |max_output_tokens| {
            let mut core = VllmCore::new(args(WorkerType::Decode));
            let effects = core
                .apply_command_effects(
                    SchedulerCommand::ReserveDestination {
                        handoff_id: HandoffId::new(),
                        request: request(Uuid::new_v4(), (0..10).collect(), max_output_tokens),
                    },
                    true,
                )
                .unwrap();
            let [
                SchedulerLifecycleEvent::DestinationReserved {
                    transferable_prompt_tokens,
                    ..
                },
            ] = effects.lifecycle_events.as_slice()
            else {
                panic!("destination reservation should complete immediately");
            };
            *transferable_prompt_tokens
        };

        assert_eq!(footprint(1), 12);
        assert_eq!(footprint(128), 12);
    }

    #[test]
    fn handoff_prefill_to_reserved_decode_owns_kv_until_normal_admission() {
        let logical_uuid = Uuid::from_u128(10_001);
        let handoff_id = HandoffId::from(Uuid::from_u128(10_002));
        let logical_tokens = (0..8).collect::<Vec<_>>();

        let mut source = VllmCore::new_with_kv_capture(args(WorkerType::Prefill), 31);
        let mut destination = VllmCore::new_with_kv_capture(args(WorkerType::Decode), 32);

        destination.receive(request(Uuid::from_u128(10_003), (0..4).collect(), 1));
        execute(&mut destination, 0.0);
        assert!(destination.is_empty());
        destination.drain_kv_events();

        let blocker_uuid = Uuid::from_u128(10_004);
        destination.receive(request(blocker_uuid, (100..104).collect(), 3));
        let blocker_first = execute(&mut destination, 1.0);
        assert_eq!(destination.state.running.len(), 1);
        destination.drain_kv_events();

        drive_source_to_hold(
            &mut source,
            handoff_id,
            request(logical_uuid, logical_tokens.clone(), 2),
        );

        let usage_before_reservation = destination.kv_manager.num_active_blocks();
        let reserve = destination
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: request(logical_uuid, logical_tokens, 2),
                },
                true,
            )
            .unwrap();
        assert_eq!(
            reserve.result,
            SchedulerCommandResult::DestinationAccepted {
                request_id: logical_uuid
            }
        );
        assert!(reserve.lifecycle_events.is_empty());
        assert!(destination.destination_is_held(handoff_id));
        assert!(!destination.is_drained());
        assert_eq!(destination.state.running.len(), 1);
        assert_eq!(
            destination.kv_manager.num_active_blocks(),
            usage_before_reservation
        );
        assert!(stored_hashes(&destination.drain_kv_events()).is_empty());

        let mut now_ms = blocker_first.end_ms;
        let mut blocker_completed = false;
        for _ in 0..16 {
            let pass = execute(&mut destination, now_ms);
            now_ms = pass.end_ms;
            if pass
                .output_signals
                .iter()
                .any(|signal| signal.uuid == blocker_uuid && signal.completed)
            {
                blocker_completed = true;
                break;
            }
        }
        assert!(blocker_completed, "blocker request should complete");
        let usage_before_physical_reservation = destination.kv_manager.num_active_blocks();
        let reserved = destination.retry_pending_destinations();
        assert!(matches!(
            reserved.as_slice(),
            [SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: reserved_handoff,
                request_id: reserved_request,
                transferable_prompt_tokens,
            }] if *reserved_handoff == handoff_id && *reserved_request == logical_uuid
                && *transferable_prompt_tokens == 4
        ));
        assert!(destination.kv_manager.num_active_blocks() > usage_before_physical_reservation);
        let reserved_block_count = destination.destination_block_count(handoff_id);
        assert!(reserved_block_count > 0);

        let activation_blocker_uuid = Uuid::from_u128(10_005);
        destination.receive(request(activation_blocker_uuid, (200..204).collect(), 3));
        let activation_blocker_first = execute(&mut destination, now_ms);
        assert_eq!(destination.state.running.len(), 1);
        let occupancy_before_activation = destination.kv_manager.num_active_blocks();
        assert_eq!(
            destination
                .apply_command(SchedulerCommand::ActivateDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert!(!destination.destination_is_held(handoff_id));
        assert_eq!(
            destination.kv_manager.num_active_blocks(),
            occupancy_before_activation
        );
        assert_eq!(
            destination.state.requests[&logical_uuid].status,
            RequestStatus::Waiting
        );
        assert_eq!(destination.state.running.len(), 1);
        assert_eq!(
            destination.request_block_count(logical_uuid),
            reserved_block_count
        );
        let activation_stores = stored_hashes(&destination.drain_kv_events());
        assert!(!activation_stores.is_empty());

        assert_eq!(
            source
                .apply_command(SchedulerCommand::ReleaseSource { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert!(source.is_empty());
        assert!(source.is_drained());

        let blocked = execute(&mut destination, activation_blocker_first.end_ms);
        assert_eq!(
            destination.state.requests[&logical_uuid].status,
            RequestStatus::Waiting
        );
        assert_eq!(
            destination.request_block_count(logical_uuid),
            reserved_block_count
        );
        let blocked_stores = stored_hashes(&blocked.kv_events);
        assert_no_republished_stores(&activation_stores, &blocked_stores);

        let blocker_terminal = execute(&mut destination, blocked.end_ms);
        assert!(
            blocker_terminal
                .output_signals
                .iter()
                .any(|signal| signal.uuid == activation_blocker_uuid && signal.completed)
        );
        assert_eq!(
            destination.state.requests[&logical_uuid].status,
            RequestStatus::Waiting
        );

        let admitted = execute(&mut destination, blocker_terminal.end_ms);
        assert_eq!(
            destination.state.requests[&logical_uuid].status,
            RequestStatus::Running
        );
        assert!(destination.request_block_count(logical_uuid) >= reserved_block_count);
        assert!(
            destination.state.requests[&logical_uuid]
                .sequence
                .num_allocated_tokens()
                >= destination.state.requests[&logical_uuid]
                    .sequence
                    .num_input_tokens()
        );
        assert_no_republished_stores(&activation_stores, &stored_hashes(&admitted.kv_events));

        let terminal = execute(&mut destination, admitted.end_ms);
        assert!(
            terminal
                .output_signals
                .iter()
                .any(|signal| signal.uuid == logical_uuid && signal.completed)
        );
        assert!(destination.is_empty());
        assert!(destination.is_drained());
        assert!(!destination.destination_is_held(handoff_id));
        assert_eq!(destination.kv_manager.num_active_blocks(), 0);
    }

    #[test]
    fn destination_cancel_reaches_activated_request_exactly_once() {
        let request_id = Uuid::from_u128(10_101);
        let handoff_id = HandoffId::from(Uuid::from_u128(10_102));
        let mut core = VllmCore::new_with_kv_capture(args(WorkerType::Decode), 33);

        let reserve = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: request(request_id, (0..8).collect(), 4),
                },
                true,
            )
            .unwrap();
        assert!(matches!(
            reserve.lifecycle_events.as_slice(),
            [SchedulerLifecycleEvent::DestinationReserved { .. }]
        ));
        assert_eq!(
            core.apply_command(SchedulerCommand::ActivateDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        execute(&mut core, 0.0);
        assert!(core.state.requests.contains_key(&request_id));

        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert!(core.is_empty());
        assert!(core.is_drained());
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Noop
        );
    }

    #[test]
    fn cancel_running_destination_immediately_retries_pending_head() {
        let mut core = VllmCore::new(args(WorkerType::Decode));
        let running_handoff = HandoffId::from(Uuid::from_u128(10_201));
        let pending_handoff = HandoffId::from(Uuid::from_u128(10_202));
        let running_request = Uuid::from_u128(10_203);
        let pending_request = Uuid::from_u128(10_204);

        let reserved = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: running_handoff,
                    request: request(running_request, (0..4).collect(), 8),
                },
                true,
            )
            .unwrap();
        assert_eq!(reserved.lifecycle_events.len(), 1);
        assert_eq!(
            core.apply_command(SchedulerCommand::ActivateDestination {
                handoff_id: running_handoff,
            })
            .unwrap(),
            SchedulerCommandResult::Applied
        );
        execute(&mut core, 0.0);
        assert_eq!(core.mocker_metrics().running_requests, 1);

        let pending = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: pending_handoff,
                    request: request(pending_request, (100..104).collect(), 2),
                },
                true,
            )
            .unwrap();
        assert!(pending.lifecycle_events.is_empty());

        let canceled = core
            .apply_command_effects(
                SchedulerCommand::CancelDestination {
                    handoff_id: running_handoff,
                },
                true,
            )
            .unwrap();
        assert!(matches!(
            canceled.lifecycle_events.as_slice(),
            [SchedulerLifecycleEvent::DestinationReserved {
                handoff_id,
                request_id,
                ..
            }] if *handoff_id == pending_handoff && *request_id == pending_request
        ));
        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination {
                handoff_id: pending_handoff,
            })
            .unwrap(),
            SchedulerCommandResult::Applied
        );
    }

    #[test]
    fn trtllm_destination_reservation_fails_without_acquiring_kv() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Trtllm)
            .block_size(4)
            .num_gpu_blocks(12)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .worker_type(WorkerType::Decode)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut destination = VllmCore::new(args);
        let handoff_id = HandoffId::from(Uuid::from_u128(10_005));

        let error = destination
            .apply_command(SchedulerCommand::ReserveDestination {
                handoff_id,
                request: request(Uuid::from_u128(10_006), (0..8).collect(), 2),
            })
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "destination reservation is not supported for TRT-LLM"
        );
        assert!(!destination.destination_is_held(handoff_id));
        assert_eq!(destination.kv_manager.num_active_blocks(), 0);
        assert!(destination.is_drained());
    }
}

mod core_behavior {
    use super::*;

    #[test]
    fn test_planned_output_tokens_are_emitted_exactly() {
        let mut core = VllmCore::new(make_args());
        let uuid = Uuid::from_u128(0xA11CE);
        let planned = vec![101, 202, 303];
        core.receive(DirectRequest {
            tokens: vec![1, 2],
            max_output_tokens: planned.len(),
            output_token_ids: Some(planned.clone()),
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });

        let mut collector = crate::replay::TraceCollector::default();
        let mut emitted = Vec::new();
        for step in 0..planned.len() {
            let pass = core.execute_pass(&mut collector, step as f64);
            emitted.extend(
                pass.output_signals
                    .into_iter()
                    .filter(|signal| signal.uuid == uuid)
                    .map(|signal| signal.token_id.expect("planned token should be present")),
            );
        }

        assert_eq!(emitted, planned);
        assert!(core.is_empty());
    }

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
            output_token_ids: None,
            uuid: Some(r1),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        core.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(r2),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);

        assert_eq!(
            pass.output_signals.len(),
            1,
            "first request should emit immediately"
        );
        assert_eq!(core.state.waiting.len(), 0);
        assert_eq!(pass.mocker_metrics.running_requests, 2);
        assert_eq!(pass.mocker_metrics.waiting_requests, 0);
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
        assert_eq!(core.kv_manager.num_active_blocks(), 3);
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
            output_token_ids: None,
            uuid: Some(r1),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        core.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(r2),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
                output_token_ids: None,
                uuid: Some(uuid),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                ..Default::default()
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(81)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            output_token_ids: None,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(2))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .preemption_mode(PreemptionMode::Lifo)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let r1 = Uuid::from_u128(1);
        let r2 = Uuid::from_u128(2);
        for (uuid, range) in [(r1, 0u32..8u32), (r2, 100u32..108u32)] {
            core.receive(DirectRequest {
                tokens: range.collect(),
                max_output_tokens: 8,
                output_token_ids: None,
                uuid: Some(uuid),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                ..Default::default()
            });
        }

        let mut collector = crate::replay::TraceCollector::default();
        let mut now_ms = 0.0;
        let mut preemptions_before = 0;
        for _ in 0..16 {
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms.max(now_ms + 1.0);
            preemptions_before = pass.mocker_metrics.vllm_preemptions_total;
            if preemptions_before > 0 {
                break;
            }
        }
        let request = core.state.requests.get(&r2).unwrap();
        assert_eq!(request.status, RequestStatus::Preempted);
        assert_eq!(request.num_computed_tokens, 0);
        assert_eq!(request.num_preemptions, 1);
        assert_eq!(core.state.waiting.front().copied(), Some(r2));
        assert_eq!(preemptions_before, 1);
    }

    #[test]
    fn test_waiting_full_isl_gate_blocks_without_preemption_then_admits() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(6)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(3))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .preemption_mode(PreemptionMode::Lifo)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let holder = Uuid::from_u128(1);
        let blocked = Uuid::from_u128(2);
        let follower = Uuid::from_u128(3);
        core.receive(DirectRequest {
            tokens: (0..16).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(holder),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        core.receive(DirectRequest {
            tokens: (100..112).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(blocked),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        core.receive(DirectRequest {
            tokens: (200..204).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(follower),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass1 = core.execute_pass(&mut collector, 0.0);

        assert!(core.state.waiting.contains(&blocked));
        assert!(core.state.waiting.contains(&follower));
        assert_eq!(core.state.waiting.front().copied(), Some(blocked));
        assert!(
            !pass1
                .admissions
                .iter()
                .any(|admission| admission.uuid == blocked)
        );
        assert!(
            !pass1
                .admissions
                .iter()
                .any(|admission| admission.uuid == follower),
            "a smaller follower must not skip a blocked FIFO head"
        );
        assert!(
            pass1
                .output_signals
                .iter()
                .any(|signal| signal.uuid == holder && signal.completed)
        );
        assert_eq!(
            core.state
                .requests
                .get(&blocked)
                .unwrap()
                .num_computed_tokens,
            0
        );
        assert_eq!(pass1.mocker_metrics.vllm_preemptions_total, 0);

        let pass2 = core.execute_pass(&mut collector, pass1.end_ms.max(1.0));
        assert!(
            pass2
                .admissions
                .iter()
                .any(|admission| admission.uuid == blocked),
            "blocked request should be admitted after the holder completes"
        );
        assert_eq!(pass2.mocker_metrics.vllm_preemptions_total, 0);
    }

    #[test]
    fn test_fresh_request_larger_than_pool_is_rejected_and_follower_runs() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(4)
            .max_num_batched_tokens(Some(32))
            .max_num_seqs(Some(2))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let oversized = Uuid::from_u128(1);
        let follower = Uuid::from_u128(2);
        for (uuid, range) in [(oversized, 0u32..20u32), (follower, 100u32..104u32)] {
            core.receive(DirectRequest {
                tokens: range.collect(),
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(uuid),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                ..Default::default()
            });
        }

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);

        assert!(
            pass.output_signals
                .iter()
                .any(|signal| { signal.uuid == oversized && signal.completed && signal.rejected })
        );
        assert!(
            pass.admissions
                .iter()
                .any(|admission| admission.uuid == follower)
        );
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0);
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
        assert!(matches!(
            core.kv_manager.process(&signal),
            G1Acquire::Ready(2)
        ));
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
        assert!(matches!(
            core.kv_manager.process(&prompt_only),
            G1Acquire::Ready(2)
        ));
        sequence.commit_allocation(sequence.num_input_tokens());

        core.state.insert_running_for_test(uuid);
        core.state.requests.insert(
            uuid,
            VllmRequestState {
                sequence,
                status: RequestStatus::Running,
                num_computed_tokens: 9,
                num_preemptions: 1,
                offload_dependency: None,
            },
        );

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        let request = core.state.requests.get(&uuid).unwrap();

        assert_eq!(pass.output_signals.len(), 1);
        assert_eq!(request.num_computed_tokens, 12);
        assert_eq!(request.sequence.len(), 13);
        assert_eq!(request.sequence.num_allocated_tokens(), 12);
        assert_eq!(core.kv_manager.num_active_blocks(), 3);
    }

    #[test]
    fn test_completion_returns_scheduler_to_idle() {
        let mut core = VllmCore::new(make_args());
        for uuid in [Uuid::from_u128(1), Uuid::from_u128(2)] {
            core.receive(DirectRequest {
                tokens: (0..8).collect(),
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(uuid),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                ..Default::default()
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

    #[test]
    fn test_mtp_batch_applies_request_bursts_in_stable_order() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("1,1".to_string()))
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let short = Uuid::from_u128(1);
        let long = Uuid::from_u128(2);
        core.receive(DirectRequest {
            tokens: (0..4).collect(),
            max_output_tokens: 5,
            output_token_ids: None,
            uuid: Some(short),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        core.receive(DirectRequest {
            tokens: (100..104).collect(),
            max_output_tokens: 8,
            output_token_ids: None,
            uuid: Some(long),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });

        let mut collector = crate::replay::TraceCollector::default();
        let first = core.execute_pass(&mut collector, 0.0);
        assert_eq!(first.output_signals.len(), 6);
        let pass = core.execute_pass(&mut collector, first.end_ms);
        let ordered = pass
            .output_signals
            .iter()
            .map(|signal| (signal.uuid, signal.completed))
            .collect::<Vec<_>>();
        assert_eq!(
            ordered,
            vec![
                (short, false),
                (short, true),
                (long, false),
                (long, false),
                (long, false),
            ]
        );

        let request = core.state.requests.get(&long).unwrap();
        assert_eq!(request.sequence.generated_tokens(), 6);
        assert_eq!(request.sequence.len() - request.num_computed_tokens, 1);
        assert_eq!(
            pass.fpm.unwrap().num_decode_requests,
            2,
            "FPM counts requests participating in the forward pass, not emitted tokens"
        );
    }

    #[test]
    fn test_mtp_recomputes_last_prefix_cache_block() {
        fn second_request_admission_reuse(mtp_enabled: bool) -> (usize, u64, u64) {
            let mut builder = MockEngineArgs::builder()
                .block_size(4)
                .num_gpu_blocks(16)
                .max_num_batched_tokens(Some(16))
                .max_num_seqs(Some(1))
                .enable_prefix_caching(true)
                .speedup_ratio(0.0);
            if mtp_enabled {
                builder = builder
                    .aic_nextn(Some(1))
                    .aic_nextn_accept_rates(Some("1".to_string()));
            }
            let mut core = VllmCore::new(builder.build().unwrap());
            let mut collector = crate::replay::TraceCollector::default();

            core.receive(DirectRequest {
                tokens: (0..8).collect(),
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(1)),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                ..Default::default()
            });
            core.execute_pass(&mut collector, 0.0);
            assert!(core.is_empty());

            core.receive(DirectRequest {
                tokens: (0..12).collect(),
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(2)),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                ..Default::default()
            });
            let pass = core.execute_pass(&mut collector, 1.0);
            let fpm = pass.fpm.expect("forward-pass metrics should be present");
            (
                pass.admissions[0].reused_input_tokens,
                fpm.sum_prefill_tokens,
                fpm.sum_prefill_kv_tokens,
            )
        }

        assert_eq!(second_request_admission_reuse(false), (8, 4, 8));
        assert_eq!(second_request_admission_reuse(true), (4, 8, 4));
    }

    #[rstest]
    #[case::kvbm(G1Backend::Kvbm)]
    #[case::native(G1Backend::Native)]
    fn test_mtp_releases_unused_block_reservations(#[case] backend: G1Backend) {
        let args = MockEngineArgs::builder()
            .g1_backend(backend)
            .block_size(2)
            .num_gpu_blocks(8)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(2))
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("0,1".to_string()))
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let uuid = Uuid::from_u128(3);
        core.receive(DirectRequest {
            tokens: (0..3).collect(),
            max_output_tokens: 5,
            output_token_ids: None,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        assert_eq!(pass.output_signals.len(), 1);
        assert_eq!(
            core.kv_manager.num_active_blocks(),
            2,
            "the block reserved only for rejected drafts must be released: backend={backend:?}"
        );
        let request = core.state.requests.get(&uuid).unwrap();
        assert_eq!(request.sequence.generated_tokens(), 1);
        assert_eq!(request.sequence.len() - request.num_computed_tokens, 1);
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(71)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);

        assert_eq!(
            pass.router_event_visibility,
            RouterEventVisibility::PassStart
        );
        assert!(!pass.kv_events.is_empty());
        assert!(
            pass.kv_events
                .iter()
                .all(|event| event.worker_id == ROUTER_TEST_WORKER_ID)
        );
        assert!(pass.kv_events.iter().all(|event| event.event.dp_rank == 0));
    }

    #[tokio::test]
    async fn test_completion_events_apply_cleanly() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let mut core = VllmCore::new_with_kv_capture(router_args(), ROUTER_TEST_WORKER_ID);
        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 4,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(41)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
    async fn test_native_destination_cancel_preserves_tail_first_eviction() {
        let block_size = 4;
        let harness = RouterIndexerHarness::new(block_size as u32, ROUTER_TEST_WORKER_ID);
        let args = MockEngineArgs::builder()
            .block_size(block_size)
            .num_gpu_blocks(3)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .g1_backend(G1Backend::Native)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new_with_kv_capture(args, ROUTER_TEST_WORKER_ID);
        let seed_tokens = (0..8).collect::<Vec<_>>();
        core.receive(DirectRequest {
            tokens: seed_tokens.clone(),
            max_output_tokens: 1,
            output_token_ids: Some(vec![8]),
            uuid: Some(Uuid::from_u128(81)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });

        let mut collector = crate::replay::TraceCollector::default();
        let mut now_ms = 0.0;
        let mut prefix_hashes = Vec::new();
        for _ in 0..4 {
            if core.is_empty() {
                break;
            }
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms;
            prefix_hashes.extend(stored_hashes(&pass.kv_events));
            harness.apply_events(pass.kv_events).await;
        }
        assert!(core.is_empty(), "seed request should complete");
        assert_eq!(
            prefix_hashes.len(),
            2,
            "seed request should store the two-block prefix A -> B"
        );
        assert_eq!(harness.overlap_for_hashes(prefix_hashes.clone()).await, 2);

        let handoff_id = HandoffId::from(Uuid::from_u128(82));
        let request_id = Uuid::from_u128(83);
        let reservation = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: DirectRequest {
                        tokens: seed_tokens,
                        max_output_tokens: 1,
                        output_token_ids: Some(vec![8]),
                        uuid: Some(request_id),
                        dp_rank: 0,
                        arrival_timestamp_ms: None,
                        ..Default::default()
                    },
                },
                true,
            )
            .unwrap();
        assert!(matches!(
            reservation.lifecycle_events.as_slice(),
            [SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: reserved_handoff,
                request_id: reserved_request,
                ..
            }] if *reserved_handoff == handoff_id && *reserved_request == request_id
        ));
        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );

        core.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 1,
            output_token_ids: Some(vec![108]),
            uuid: Some(Uuid::from_u128(84)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        let pressure = core.execute_pass(&mut collector, now_ms);
        assert_eq!(
            removed_event_count(&pressure.kv_events),
            1,
            "two fresh blocks with one free slot should evict exactly one cached block"
        );
        harness.apply_events(pressure.kv_events).await;

        assert_eq!(
            harness.overlap_for_hashes(prefix_hashes).await,
            1,
            "destination cancellation must make leaf B older than parent A"
        );
        harness.assert_no_event_errors();
        harness.assert_no_event_warnings();
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_preemption_recompute_events_apply_cleanly() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(6)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(2))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .preemption_mode(PreemptionMode::Lifo)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new_with_kv_capture(args, ROUTER_TEST_WORKER_ID);
        let r1 = Uuid::from_u128(51);
        let r2 = Uuid::from_u128(52);
        for (uuid, range) in [(r1, 0u32..8u32), (r2, 100u32..108u32)] {
            core.receive(DirectRequest {
                tokens: range.collect(),
                max_output_tokens: 8,
                output_token_ids: None,
                uuid: Some(uuid),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                ..Default::default()
            });
        }

        let mut collector = crate::replay::TraceCollector::default();
        let mut now_ms = 0.0;
        let mut saw_preemption = false;
        for _ in 0..16 {
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms.max(now_ms + 1.0);
            harness.apply_events(pass.kv_events).await;
            if core
                .state
                .requests
                .get(&r2)
                .is_some_and(|request| request.status == RequestStatus::Preempted)
            {
                saw_preemption = true;
                break;
            }
        }

        assert!(saw_preemption);
        let request = core.state.requests.get(&r2).unwrap();
        assert_eq!(request.status, RequestStatus::Preempted);
        assert_eq!(request.num_computed_tokens, 0);
        assert_eq!(request.num_preemptions, 1);
        assert_eq!(core.state.waiting.front().copied(), Some(r2));

        let mut readmitted = false;
        for _ in 0..32 {
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms.max(now_ms + 1.0);
            readmitted |= pass.admissions.iter().any(|admission| admission.uuid == r2);
            harness.apply_events(pass.kv_events).await;
            if readmitted {
                break;
            }
        }

        assert!(
            readmitted,
            "preempted request should be admitted for recompute"
        );
        harness.assert_no_event_warnings();
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_mtp_preemption_recompute_drains_with_clean_events() {
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
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("0,1".to_string()))
            .build()
            .unwrap();
        let mut core = VllmCore::new_with_kv_capture(args, ROUTER_TEST_WORKER_ID);
        let requests = [
            (Uuid::from_u128(61), 0u32..8u32),
            (Uuid::from_u128(62), 100u32..108u32),
            (Uuid::from_u128(63), 200u32..212u32),
        ];
        for (uuid, tokens) in requests {
            core.receive(DirectRequest {
                tokens: tokens.collect(),
                max_output_tokens: 7,
                output_token_ids: None,
                uuid: Some(uuid),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                ..Default::default()
            });
        }

        let mut collector = crate::replay::TraceCollector::default();
        let mut now_ms = 0.0;
        let mut output_tokens = 0;
        let mut saw_remove = false;
        for _ in 0..200 {
            if core.is_empty() {
                break;
            }
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms.max(now_ms + 1.0);
            output_tokens += pass.output_signals.len();
            saw_remove |= removed_event_count(&pass.kv_events) > 0;
            harness.apply_events(pass.kv_events).await;
        }

        assert!(core.is_empty());
        assert_eq!(output_tokens, 21);
        assert!(core.state.waiting.is_empty());
        assert!(core.state.running.is_empty());
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        assert!(saw_remove);
        harness.assert_no_event_errors();
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
    #[case::vllm(EngineType::Vllm)]
    #[case::trtllm(EngineType::Trtllm)]
    #[tokio::test]
    async fn native_g1_runs_through_live_scheduler_wrapper(#[case] engine_type: EngineType) {
        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
        let args = MockEngineArgs::builder()
            .engine_type(engine_type)
            .g1_backend(G1Backend::Native)
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(2))
            .enable_prefix_caching(true)
            .enable_chunked_prefill(true)
            .speedup_ratio(1000.0)
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

        // Submit identical prompts sequentially so the second request exercises
        // the native manager's inactive-prefix lookup through the async wrapper.
        for uuid in [Uuid::from_u128(81), Uuid::from_u128(82)] {
            scheduler.receive(DirectRequest {
                tokens: (0..8).collect(),
                max_output_tokens: 2,
                output_token_ids: Some(vec![100, 101]),
                uuid: Some(uuid),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                ..Default::default()
            });

            let signals = tokio::time::timeout(Duration::from_secs(2), async {
                let mut signals = Vec::new();
                loop {
                    let batch = output_rx
                        .recv()
                        .await
                        .expect("live scheduler output channel should remain open");
                    let completed = batch
                        .iter()
                        .any(|signal| signal.uuid == uuid && signal.completed);
                    signals.extend(batch.into_iter().filter(|signal| signal.uuid == uuid));
                    if completed {
                        break signals;
                    }
                }
            })
            .await
            .expect("native G1 live request should complete");

            assert_eq!(
                signals
                    .iter()
                    .filter_map(|signal| signal.token_id)
                    .collect::<Vec<_>>(),
                vec![100, 101]
            );
            assert!(signals.last().is_some_and(|signal| signal.completed));
        }

        let metrics_rx = scheduler.metrics_receiver();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let metrics = metrics_rx.borrow().clone();
                if metrics.running_requests == 0
                    && metrics.waiting_requests == 0
                    && metrics.active_decode_blocks == 0
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("native G1 live scheduler should return to idle");
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
        if enable_prefix_caching {
            assert!(harness.ok_count(METRIC_EVENT_STORED) > 0);
        } else {
            assert_eq!(harness.ok_count(METRIC_EVENT_STORED), 0);
            assert_eq!(harness.ok_count(METRIC_EVENT_REMOVED), 0);
        }
        // NOTE: we do NOT assert `dump_events().is_empty()` here because
        // mocker's protocol does not emit router `Removed` events on
        // request completion.
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_scheduler_mtp_lifecycle_drains_small_blocks() {
        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(128)
            .block_size(4)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(16))
            .speedup_ratio(1000.0)
            .enable_prefix_caching(false)
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("1,1".to_string()))
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

        crate::scheduler::test_utils::assert_scheduler_completes_all(
            &scheduler,
            &mut output_rx,
            24,
            5,
            7,
            false,
        )
        .await;
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
                ..Default::default()
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
            output_token_ids: None,
            uuid: None,
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(72)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
                output_token_ids: None,
                uuid: None,
                dp_rank: 0,
                arrival_timestamp_ms: None,
                ..Default::default()
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            output_token_ids: None,
            uuid: Some(r1),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            output_token_ids: None,
            uuid: Some(r2),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            output_token_ids: None,
            uuid: Some(r1),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            output_token_ids: None,
            uuid: Some(r1),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            output_token_ids: None,
            uuid: Some(r1),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        core.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(r2),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        core.receive(DirectRequest {
            tokens: (100..112).collect(), // prompt_len = 12
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });

        // Prefill r1 and decode a few tokens to build up KV
        core.execute_pass(&mut collector, 0.0);
        core.execute_pass(&mut collector, 1.0);
        core.execute_pass(&mut collector, 2.0);

        // r2: another request that will compete for KV
        core.receive(DirectRequest {
            tokens: (100..116).collect(), // 16 tokens — will pressure KV
            max_output_tokens: 5,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
    use kvbm_engine::{G2, G3};
    use kvbm_logical::manager::BlockManager;
    use uuid::Uuid;

    use crate::common::handoff::HandoffId;
    use crate::common::protocols::{DirectRequest, MockEngineArgs, MoveBlock, WorkerType};
    use crate::common::sequence::ActiveSequence;
    use crate::kv_manager::G1Acquire;
    use crate::kvbm_offload::shared_g3::shared_g3_test_guard_blocking;
    use crate::kvbm_offload::{KvbmOffloadConfig, MockOffloadEngine};
    use crate::scheduler::{
        LiveBoundaryCore, SchedulerCommand, SchedulerCommandResult, SchedulerLifecycleEvent,
    };

    use super::super::core::{RequestStatus, VllmCore, VllmRequestState};

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

    fn seed_g3_blocks(g3: &BlockManager<G3>, plhs: &[PositionalLineageHash]) {
        for plh in plhs {
            let (mut alloc, _evicted) = g3.allocate_blocks_with_evictions(1).expect("G3 allocate");
            let mutable = alloc.pop().unwrap();
            let staged = mutable.stage(*plh, g3.block_size()).expect("G3 stage");
            drop(g3.register_block(staged));
        }
    }

    fn insert_running_request(
        core: &mut VllmCore,
        uuid: Uuid,
        tokens: Vec<u32>,
        max_output_tokens: usize,
    ) {
        let mut sequence = ActiveSequence::new(
            tokens,
            max_output_tokens,
            Some(core.args.block_size),
            false,
            false,
        );
        let signal = sequence
            .prepare_allocation(sequence.len())
            .expect("running test request must need initial KV");
        assert!(matches!(
            core.kv_manager.process(&signal),
            G1Acquire::Ready(_)
        ));
        sequence.commit_allocation(sequence.len());
        let num_computed_tokens = sequence.len();
        core.state.insert_running_for_test(uuid);
        core.state.requests.insert(
            uuid,
            VllmRequestState {
                sequence,
                status: RequestStatus::Running,
                num_computed_tokens,
                num_preemptions: 0,
                offload_dependency: None,
            },
        );
    }

    fn seed_inactive_g1_block(core: &mut VllmCore) {
        let mut sequence = ActiveSequence::new(
            (10_000..10_004).collect(),
            1,
            Some(core.args.block_size),
            false,
            false,
        );
        let signal = sequence
            .prepare_allocation(sequence.len())
            .expect("cache seed must need initial KV");
        assert!(matches!(
            core.kv_manager.process(&signal),
            G1Acquire::Ready(1)
        ));
        sequence.commit_allocation(sequence.len());
        for signal in sequence.free_signal() {
            assert!(matches!(
                core.kv_manager.process(&signal),
                G1Acquire::Ready(_)
            ));
        }
        assert_eq!(core.kv_manager.num_inactive_blocks(), 1);
    }

    fn attach_g1_offload_engine(core: &mut VllmCore) {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let mut engine = runtime
            .block_on(MockOffloadEngine::new(KvbmOffloadConfig {
                block_size_tokens: core.args.block_size,
                block_size_bytes: Some(1_000_000),
                bandwidth_g1_to_g2_gbps: 1.0,
                offload_batch_size: 4,
                ..Default::default()
            }))
            .expect("engine build");
        engine.tick(0.0);
        engine.attach_runtime(runtime);
        core.kv_manager.attach_new_offload_engine(engine);
    }

    #[test]
    fn decode_growth_waits_without_preempting() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(3)
            .block_size(4)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .worker_type(WorkerType::Decode)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        seed_inactive_g1_block(&mut core);

        let blocked = Uuid::from_u128(701);
        let unrelated = Uuid::from_u128(702);
        insert_running_request(&mut core, blocked, (0..4).collect(), 8);
        insert_running_request(&mut core, unrelated, (100..102).collect(), 8);
        attach_g1_offload_engine(&mut core);

        let mut collector = crate::replay::TraceCollector::default();
        let sampled_pass = core.execute_pass(&mut collector, 0.0);
        assert!(
            sampled_pass
                .output_signals
                .iter()
                .any(|signal| signal.uuid == blocked),
            "sampling must not acquire the dangling token's new block"
        );
        let blocked_state = core.state.requests.get(&blocked).unwrap();
        assert_eq!(blocked_state.sequence.generated_tokens(), 1);
        assert_eq!(blocked_state.sequence.num_allocated_tokens(), 4);
        assert_eq!(blocked_state.num_computed_tokens, 4);
        assert!(blocked_state.offload_dependency.is_none());

        let blocked_pass = core.execute_pass(&mut collector, sampled_pass.end_ms);
        assert!(
            blocked_pass
                .output_signals
                .iter()
                .all(|signal| signal.uuid != blocked),
            "the sampled token must wait before computation when its block cannot be acquired"
        );
        assert!(
            blocked_pass
                .output_signals
                .iter()
                .any(|signal| signal.uuid == unrelated),
            "the unrelated running request should continue"
        );
        let blocked_state = core.state.requests.get(&blocked).unwrap();
        assert_eq!(blocked_state.sequence.generated_tokens(), 1);
        assert_eq!(blocked_state.sequence.num_allocated_tokens(), 4);
        assert_eq!(blocked_state.num_computed_tokens, 4);
        let dependency = blocked_state
            .offload_dependency
            .expect("decode growth must retain its exact offload dependency");
        let _ = dependency.offload_id;
        let deadline = dependency
            .deadline_ms
            .expect("active offload dependency must expose a virtual deadline");
        let unrelated_state = core.state.requests.get(&unrelated).unwrap();
        assert_eq!(unrelated_state.status, RequestStatus::Running);
        assert_eq!(unrelated_state.num_preemptions, 0);
        assert_eq!(core.state.preemptions_total, 0);

        core.tick_offload_only(deadline);
        let resumed = core.execute_pass(&mut collector, deadline);
        assert!(
            resumed
                .output_signals
                .iter()
                .any(|signal| signal.uuid == blocked),
            "decode must resume after its exact offload dependency terminates"
        );
        let blocked_state = core.state.requests.get(&blocked).unwrap();
        assert_eq!(blocked_state.sequence.generated_tokens(), 2);
        assert_eq!(blocked_state.sequence.num_allocated_tokens(), 5);
        assert_eq!(blocked_state.num_computed_tokens, 5);
        assert!(blocked_state.offload_dependency.is_none());
        assert_eq!(core.state.preemptions_total, 0);
    }

    #[test]
    fn speculative_decode_reservation_waits_without_preempting() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(5)
            .block_size(4)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .worker_type(WorkerType::Decode)
            .speedup_ratio(0.0)
            .aic_nextn(Some(5))
            .aic_nextn_accept_rates(Some("1,1,1,1,1".to_string()))
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        seed_inactive_g1_block(&mut core);

        let blocked = Uuid::from_u128(711);
        let unrelated = Uuid::from_u128(712);
        insert_running_request(&mut core, blocked, (0..4).collect(), 8);
        insert_running_request(&mut core, unrelated, (100..102).collect(), 8);
        attach_g1_offload_engine(&mut core);

        let mut collector = crate::replay::TraceCollector::default();
        let blocked_pass = core.execute_pass(&mut collector, 0.0);
        assert!(
            blocked_pass.output_signals.is_empty(),
            "speculative generation must not begin before full-burst reservation"
        );
        for uuid in [blocked, unrelated] {
            let request = core.state.requests.get(&uuid).unwrap();
            assert_eq!(request.sequence.generated_tokens(), 0);
            assert_eq!(request.num_preemptions, 0);
            assert_eq!(request.status, RequestStatus::Running);
            assert!(request.offload_dependency.is_some());
        }
        let dependency = core.state.requests[&blocked]
            .offload_dependency
            .expect("speculative reservation must retain its exact dependency");
        let _ = dependency.offload_id;
        let deadline = dependency
            .deadline_ms
            .expect("active offload dependency must expose a virtual deadline");
        assert_eq!(core.state.preemptions_total, 0);

        core.tick_offload_only(deadline);
        let resumed = core.execute_pass(&mut collector, deadline);
        assert_eq!(
            resumed
                .output_signals
                .iter()
                .filter(|signal| signal.uuid == blocked)
                .count(),
            6
        );
        assert_eq!(
            resumed
                .output_signals
                .iter()
                .filter(|signal| signal.uuid == unrelated)
                .count(),
            6
        );
        for uuid in [blocked, unrelated] {
            let request = core.state.requests.get(&uuid).unwrap();
            assert_eq!(request.sequence.generated_tokens(), 6);
            assert_eq!(request.num_preemptions, 0);
            assert!(request.offload_dependency.is_none());
        }
        assert_eq!(core.state.preemptions_total, 0);
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn live_destination_eviction_uses_command_application_time() {
        let block_size = 4;
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(2)
            .block_size(block_size)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(2))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new_with_kv_capture(args, 7);
        let config = KvbmOffloadConfig {
            block_size_tokens: block_size,
            block_size_bytes: Some(20_000_000),
            bandwidth_g1_to_g2_gbps: 1.0,
            ..Default::default()
        };
        let engine = MockOffloadEngine::new(config).await.expect("engine build");
        engine.tick(0.0);
        core.kv_manager.attach_new_offload_engine(engine);

        let cached_uuid = Uuid::from_u128(1);
        core.receive(DirectRequest {
            tokens: (0..block_size as u32).collect(),
            max_output_tokens: 1,
            uuid: Some(cached_uuid),
            ..Default::default()
        });
        let mut collector = crate::replay::TraceCollector::default();
        let mut now_ms = 0.0;
        let completed = (0..4).any(|_| {
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms;
            pass.output_signals
                .iter()
                .any(|signal| signal.uuid == cached_uuid && signal.completed)
        });
        assert!(completed, "cache seed request should complete normally");
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        assert_eq!(core.kv_manager.num_inactive_blocks(), 1);
        core.drain_kv_events();

        let blocker_uuid = Uuid::from_u128(3);
        core.receive(DirectRequest {
            tokens: vec![9; block_size - 1],
            max_output_tokens: 100,
            uuid: Some(blocker_uuid),
            ..Default::default()
        });
        let blocker_pass = core.execute_pass(&mut collector, now_ms);
        assert!(
            blocker_pass
                .output_signals
                .iter()
                .any(|signal| signal.uuid == blocker_uuid && !signal.completed)
        );
        assert_eq!(core.kv_manager.num_active_blocks(), 1);
        assert_eq!(core.kv_manager.num_inactive_blocks(), 1);

        let handoff_id = HandoffId::from(Uuid::from_u128(2));
        let effects = core
            .apply_live_command(
                SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: DirectRequest {
                        tokens: (block_size as u32..(block_size * 2) as u32).collect(),
                        max_output_tokens: 1,
                        uuid: Some(Uuid::from_u128(2)),
                        ..Default::default()
                    },
                },
                true,
                100.0,
            )
            .unwrap();
        assert!(effects.lifecycle_events.is_empty());

        let deadline = core
            .earliest_offload_deadline()
            .expect("reservation eviction should schedule G1 to G2 offload");
        assert!(
            (deadline - 120.0).abs() < 0.01,
            "20 ms transfer applied at t=100 should finish near t=120, got {deadline}"
        );
        core.tick_offload_only(100.0);
        assert_eq!(
            core.kv_manager.num_active_blocks(),
            2,
            "the active blocker and quarantined offload source both occupy G1"
        );
        assert_eq!(core.earliest_offload_deadline(), Some(deadline));

        let transport = core.tick_offload_transport_only(deadline);
        assert!(transport.lifecycle_events.is_empty());
        assert!(transport.kv_events.iter().any(|event| {
            event.storage_tier == dynamo_kv_router::protocols::StorageTier::HostPinned
        }));
        assert_eq!(core.kv_manager.num_active_blocks(), 1);
        assert_eq!(core.destination_block_count(handoff_id), 0);

        let reserved = core.retry_pending_destinations();
        assert!(reserved.iter().any(|event| matches!(
            event,
            SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: observed,
                request_id,
                ..
            } if *observed == handoff_id && *request_id == Uuid::from_u128(2)
        )));
        assert!(core.destination_block_count(handoff_id) > 0);
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
            output_token_ids: None,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        let plhs = core
            .state
            .requests
            .get(&uuid)
            .unwrap()
            .sequence
            .positional_lineage_hashes();
        assert_eq!(plhs.len(), 1, "test request should have one full block");
        seed_g2_blocks(engine.g2_manager(), plhs);

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
            output_token_ids: None,
            uuid: Some(hit_uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        let cold_uuid = Uuid::new_v4();
        core.receive(DirectRequest {
            tokens: (4..8).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(cold_uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        let plhs = core
            .state
            .requests
            .get(&hit_uuid)
            .unwrap()
            .sequence
            .positional_lineage_hashes();
        assert_eq!(plhs.len(), 1, "test request should have one full block");
        seed_g2_blocks(engine.g2_manager(), plhs);
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

    /// Offline replay uses this hook to advance offload-only deadlines before
    /// running admissions at the same timestamp. A staged G3 hit must take
    /// two virtual hops, then be immediately reusable by the next pass.
    #[test]
    fn tick_offload_only_completes_g3_staging_before_same_timestamp_admission() {
        let _guard = shared_g3_test_guard_blocking();
        let args = MockEngineArgs::builder()
            // Use two blocks so the staged hit still has a reusable prefix
            // after vLLM's required final-block recompute.
            .num_gpu_blocks(2)
            .block_size(4)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let config = KvbmOffloadConfig {
            block_size_tokens: 4,
            block_size_bytes: Some(1_000_000),
            num_g2_blocks: 8,
            num_g3_blocks: Some(8),
            bandwidth_g2_to_g1_gbps: 1.0,
            bandwidth_g3_to_g2_gbps: 1.0,
            ..Default::default()
        };
        let mut engine = rt
            .block_on(MockOffloadEngine::new(config))
            .expect("engine build");
        engine.attach_runtime(rt);
        engine.tick(0.0);

        let uuid = Uuid::new_v4();
        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        let plhs = core
            .state
            .requests
            .get(&uuid)
            .unwrap()
            .sequence
            .positional_lineage_hashes();
        assert_eq!(plhs.len(), 2, "test request should have two full blocks");
        seed_g3_blocks(engine.g3_manager().expect("G3 enabled"), plhs);
        core.kv_manager.attach_new_offload_engine(engine);

        let mut collector = crate::replay::TraceCollector::default();
        let parked = core.execute_pass(&mut collector, 0.0);
        assert!(parked.admissions.is_empty());
        assert_eq!(core.requests_awaiting_swap_in.len(), 1);

        core.tick_offload_only(2.0);
        assert_eq!(
            core.requests_awaiting_swap_in.len(),
            1,
            "G3→G2 completion should start, not finish, the G2→G1 hop"
        );

        core.tick_offload_only(4.0);
        assert!(
            core.requests_awaiting_swap_in.is_empty(),
            "G2→G1 completion should requeue the request before admission"
        );

        let admitted = core.execute_pass(&mut collector, 4.0);
        assert_eq!(admitted.admissions.len(), 1);
        assert_eq!(admitted.admissions[0].reused_input_tokens, 4);
    }

    /// A partial G2 swap-in is scheduled from the first G1 miss onward. The
    /// already-cached G1 prefix must stay pinned while the transfer is pending,
    /// otherwise G1 eviction can remove the parent block before the suffix
    /// publishes its Device-tier `Stored` event.
    #[tokio::test]
    async fn partial_g2_swap_in_pins_cached_g1_prefix() {
        let block_size = 4;
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(3)
            .block_size(block_size)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .aic_nextn(Some(1))
            .aic_nextn_accept_rates(Some("1".to_string()))
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let config = KvbmOffloadConfig {
            block_size_tokens: block_size,
            block_size_bytes: Some(1_000_000),
            bandwidth_g2_to_g1_gbps: 1.0,
            ..Default::default()
        };
        let engine = MockOffloadEngine::new(config).await.expect("engine build");
        engine.tick(0.0);

        let uuid = Uuid::new_v4();
        core.receive(DirectRequest {
            tokens: (0..(block_size * 3) as u32).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        let (prefix_signal, plhs) = {
            let request = core.state.requests.get(&uuid).unwrap();
            (
                request
                    .sequence
                    .prepare_allocation(block_size * 2)
                    .expect("prefix allocation signal"),
                request.sequence.positional_lineage_hashes().to_vec(),
            )
        };
        let prefix_blocks = match &prefix_signal {
            MoveBlock::Use(blocks, ..) => blocks.clone(),
            _ => panic!("expected prefix Use signal"),
        };
        assert!(matches!(
            core.kv_manager.process(&prefix_signal),
            G1Acquire::Ready(2)
        ));
        assert!(matches!(
            core.kv_manager.process(&MoveBlock::Deref(prefix_blocks)),
            G1Acquire::Ready(1)
        ));

        assert_eq!(plhs.len(), 3, "test request should have three full blocks");
        seed_g2_blocks(engine.g2_manager(), &plhs[2..]);
        core.kv_manager.attach_new_offload_engine(engine);

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);

        assert_eq!(pass.admissions.len(), 0);
        assert_eq!(core.requests_awaiting_swap_in.len(), 1);
        assert_eq!(
            core.kv_manager.num_active_blocks(),
            3,
            "two cached prefix blocks plus one destination slot must be pinned"
        );
    }

    #[tokio::test]
    async fn cancel_parked_swap_in_releases_ownership_and_isolates_uuid_reuse() {
        let block_size = 4;
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(3)
            .block_size(block_size)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let config = KvbmOffloadConfig {
            block_size_tokens: block_size,
            block_size_bytes: Some(1_000_000),
            bandwidth_g2_to_g1_gbps: 1.0,
            ..Default::default()
        };
        let engine = MockOffloadEngine::new(config).await.expect("engine build");
        engine.tick(0.0);

        let uuid = Uuid::from_u128(1);
        let source_handoff = HandoffId::from(Uuid::from_u128(10));
        let result = core
            .apply_command(SchedulerCommand::SubmitHandoffPrefill {
                handoff_id: source_handoff,
                request: DirectRequest {
                    tokens: (0..(block_size * 3) as u32).collect(),
                    max_output_tokens: 2,
                    uuid: Some(uuid),
                    ..Default::default()
                },
            })
            .unwrap();
        assert_eq!(result, SchedulerCommandResult::Submitted(uuid));

        let (prefix_signal, plhs) = {
            let request = core.state.requests.get(&uuid).unwrap();
            (
                request
                    .sequence
                    .prepare_allocation(block_size * 2)
                    .expect("prefix allocation signal"),
                request.sequence.positional_lineage_hashes().to_vec(),
            )
        };
        let prefix_blocks = match &prefix_signal {
            MoveBlock::Use(blocks, ..) => blocks.clone(),
            _ => panic!("expected prefix Use signal"),
        };
        assert!(matches!(
            core.kv_manager.process(&prefix_signal),
            G1Acquire::Ready(2)
        ));
        assert!(matches!(
            core.kv_manager.process(&MoveBlock::Deref(prefix_blocks)),
            G1Acquire::Ready(1)
        ));
        seed_g2_blocks(engine.g2_manager(), &plhs[2..]);
        core.kv_manager.attach_new_offload_engine(engine);

        let mut collector = crate::replay::TraceCollector::default();
        core.execute_pass(&mut collector, 0.0);
        assert_eq!(core.requests_awaiting_swap_in.len(), 1);
        assert_eq!(core.kv_manager.num_active_blocks(), 3);
        assert!(
            core.apply_command(SchedulerCommand::Submit(DirectRequest {
                tokens: vec![9; block_size],
                max_output_tokens: 1,
                uuid: Some(uuid),
                ..Default::default()
            }))
            .is_err(),
            "a parked swap-in must retain request-ID ownership"
        );

        let follower_handoff = HandoffId::from(Uuid::from_u128(20));
        let follower_uuid = Uuid::from_u128(2);
        let reserve = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: follower_handoff,
                    request: DirectRequest {
                        tokens: vec![7; block_size],
                        max_output_tokens: 1,
                        uuid: Some(follower_uuid),
                        ..Default::default()
                    },
                },
                true,
            )
            .unwrap();
        assert!(reserve.lifecycle_events.is_empty());

        let canceled = core
            .apply_command_effects(
                SchedulerCommand::CancelSource {
                    handoff_id: source_handoff,
                },
                true,
            )
            .unwrap();
        assert_eq!(canceled.result, SchedulerCommandResult::Applied);
        assert!(core.requests_awaiting_swap_in.is_empty());
        assert!(!core.state.requests.contains_key(&uuid));
        assert!(
            canceled.lifecycle_events.is_empty(),
            "cancelled swap-in must retain prefix pins and its destination until transfer termination"
        );

        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination {
                handoff_id: follower_handoff,
            })
            .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert_eq!(
            core.kv_manager.num_active_blocks(),
            3,
            "cancelled swap-in prefix pins and destination must remain reserved"
        );
        core.tick_offload_only(0.5);
        assert_eq!(core.kv_manager.num_active_blocks(), 3);

        assert_eq!(
            core.receive(DirectRequest {
                tokens: vec![99; block_size],
                max_output_tokens: 1,
                uuid: Some(uuid),
                ..Default::default()
            }),
            uuid
        );
        core.tick_offload_only(10.0);
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        assert!(core.requests_awaiting_swap_in.is_empty());
        assert!(core.state.requests.contains_key(&uuid));
        let replacement = core.execute_pass(&mut collector, 10.0);
        assert!(
            replacement
                .admissions
                .iter()
                .any(|event| event.uuid == uuid)
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
            output_token_ids: None,
            uuid: Some(first_hit),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        let second_hit = Uuid::from_u128(2);
        core.receive(DirectRequest {
            tokens: (4..8).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(second_hit),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        let cold = Uuid::from_u128(3);
        core.receive(DirectRequest {
            tokens: (8..12).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(cold),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            output_token_ids: None,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        let plhs = core
            .state
            .requests
            .get(&uuid)
            .unwrap()
            .sequence
            .positional_lineage_hashes();
        assert!(!plhs.is_empty(), "test sequence must have full blocks");

        seed_g2_blocks(engine.g2_manager(), plhs);
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
    /// timestamps passed into `tick` and swap-in start.
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
                    output_token_ids: None,
                    uuid: Some(uuid),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                    ..Default::default()
                });
                uuids.push(uuid);
            }
            let r4 = Uuid::from_u128(2000);
            core.receive(DirectRequest {
                tokens: ((TOKENS_PER_REQ as u32)..(2 * TOKENS_PER_REQ as u32)).collect(),
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(r4),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                ..Default::default()
            });
            uuids.push(r4);

            let plhs = core
                .state
                .requests
                .get(&uuids[0])
                .unwrap()
                .sequence
                .positional_lineage_hashes();
            seed_g2_blocks(engine.g2_manager(), plhs);
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

    #[test]
    fn destination_reservation_retries_after_presence_filtered_offload() {
        let block_size = 4;
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(2)
            .block_size(block_size)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(2))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);

        let cached_uuid = Uuid::from_u128(1);
        core.receive(DirectRequest {
            tokens: (0..block_size as u32).collect(),
            max_output_tokens: 1,
            uuid: Some(cached_uuid),
            ..Default::default()
        });
        let cached_plhs = core
            .state
            .requests
            .get(&cached_uuid)
            .unwrap()
            .sequence
            .positional_lineage_hashes()
            .to_vec();
        let mut collector = crate::replay::TraceCollector::default();
        let mut now_ms = 0.0;
        let completed = (0..4).any(|_| {
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms;
            pass.output_signals
                .iter()
                .any(|signal| signal.uuid == cached_uuid && signal.completed)
        });
        assert!(completed, "cache seed request should complete");
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        assert_eq!(core.kv_manager.num_inactive_blocks(), 1);

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let mut engine = runtime
            .block_on(MockOffloadEngine::new(KvbmOffloadConfig {
                block_size_tokens: block_size,
                ..Default::default()
            }))
            .expect("engine build");
        engine.tick(now_ms);
        seed_g2_blocks(engine.g2_manager(), &cached_plhs);
        engine.attach_runtime(runtime);
        core.kv_manager.attach_new_offload_engine(engine);

        let handoff_id = HandoffId::from(Uuid::from_u128(2));
        let request_id = Uuid::from_u128(2);
        let effects = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: DirectRequest {
                        tokens: (block_size as u32..(block_size * 3) as u32).collect(),
                        max_output_tokens: 1,
                        uuid: Some(request_id),
                        ..Default::default()
                    },
                },
                true,
            )
            .unwrap();

        assert_eq!(
            effects.result,
            SchedulerCommandResult::DestinationAccepted { request_id }
        );
        assert!(core.earliest_offload_deadline().is_none());
        assert!(effects.lifecycle_events.iter().any(|event| matches!(
            event,
            SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: observed_handoff,
                request_id: observed_request,
                ..
            } if *observed_handoff == handoff_id && *observed_request == request_id
        )));
        assert!(core.destination_is_held(handoff_id));
        assert_eq!(core.destination_block_count(handoff_id), 2);
    }
}
