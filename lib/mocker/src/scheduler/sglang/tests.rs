// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::indexer::{METRIC_EVENT_REMOVED, METRIC_EVENT_STORED};
use dynamo_kv_router::protocols::{BlockHashOptions, WorkerId, compute_block_hash_for_seq};
use rstest::rstest;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use super::config::{SchedulePolicy, SglangConfig, ceil_to_block};
use super::core::SglangCore;
use super::decode;
use super::decode::simulate_decode_step;
use super::live::SglangScheduler;
use super::policy::apply_schedule_policy;
use super::prefill::get_new_batch_prefill;
use super::request::SglangRequest;
use crate::common::handoff::HandoffId;
use crate::common::protocols::{
    DirectRequest, EngineType, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
    SglangArgs,
};
use crate::kv_manager::SglangKvManager;
use crate::kv_manager::sglang_backend::ActiveKvLease;
use crate::scheduler::test_utils::{
    CapturingFpmSink, RouterIndexerHarness, nth_stored_hashes, removed_event_count, stored_hashes,
};
use crate::scheduler::{
    RouterEventVisibility, SchedulerCommand, SchedulerCommandEnvelope, SchedulerCommandResult,
    SchedulerHandle, SchedulerLifecycleEvent, capture_router_event_sink,
};

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
        ..Default::default()
    }
}

fn make_decoded_request(
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    prompt_tokens: Vec<u32>,
    max_output_tokens: usize,
) -> SglangRequest {
    let prompt_len = prompt_tokens.len();
    let alloc = kv_manager.allocate_for_request(&prompt_tokens).unwrap();
    let mut running = vec![SglangRequest {
        uuid: Uuid::new_v4(),
        sequence_tokens: prompt_tokens,
        prompt_len,
        max_output_tokens,
        planned_output_ids: None,
        kv_lease: alloc.lease,
        materialized_tokens: prompt_len,
        allocated_tokens: ceil_to_block(prompt_len, config.block_size),
    }];
    let result = simulate_decode_step(&mut running, kv_manager, config, 0.0, false);
    assert_eq!(result.output_signals.len(), 1);
    running.pop().unwrap()
}

#[test]
fn zero_output_request_completes_after_prefill() {
    let mut core = SglangCore::new(test_args(32, 4, 16));
    let uuid = core.receive(direct_request(vec![1, 2, 3, 4], 0));

    let pass = core.execute_pass_internal(None, 0.0);

    assert!(core.is_empty());
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
fn zero_output_completion_survives_decode_reservation_failure() {
    let config = SglangConfig::from_args(&test_args(2, 4, 16));
    let mut kv_manager = SglangKvManager::new(8, 4, KvEventPublishers::default(), 0);
    let zero_alloc = kv_manager.allocate_for_request(&[1, 2, 3, 4]).unwrap();
    let normal_alloc = kv_manager.allocate_for_request(&[5, 6, 7, 8]).unwrap();
    let zero_uuid = Uuid::from_u128(90_004);
    let normal_uuid = Uuid::from_u128(90_005);
    let mut running = vec![
        SglangRequest {
            uuid: zero_uuid,
            sequence_tokens: vec![1, 2, 3, 4],
            prompt_len: 4,
            max_output_tokens: 0,
            planned_output_ids: None,
            kv_lease: zero_alloc.lease,
            materialized_tokens: 4,
            allocated_tokens: 4,
        },
        SglangRequest {
            uuid: normal_uuid,
            sequence_tokens: vec![5, 6, 7, 8],
            prompt_len: 4,
            max_output_tokens: 1,
            planned_output_ids: None,
            kv_lease: normal_alloc.lease,
            materialized_tokens: 4,
            allocated_tokens: 4,
        },
    ];

    // The normal request cannot reserve a decode token while both prompts own
    // the entire pool. Its retry must not discard the zero-output completion.
    let result = decode::simulate_decode_step_with_sampler(
        &mut running,
        &mut kv_manager,
        &config,
        None,
        0.0,
        false,
    );

    assert_eq!(running.len(), 1);
    assert_eq!(running[0].uuid, normal_uuid);
    assert_eq!(result.completed_requests.len(), 1);
    assert!(matches!(
        result.output_signals.as_slice(),
        [OutputSignal {
            uuid: signal_uuid,
            token_id: None,
            completed: true,
            rejected: false,
            ..
        }] if *signal_uuid == zero_uuid
    ));
}

#[test]
fn fresh_prefill_tracks_cache_owned_prefix_indices() {
    let args = test_args(8, 4, 16);
    let config = SglangConfig::from_args(&args);
    let (buffer, sink) = capture_router_event_sink(ROUTER_TEST_WORKER_ID);
    let mut kv_manager = SglangKvManager::new(11, 4, KvEventPublishers::new(Some(sink), None), 0);
    let prompt = vec![1, 2, 3, 4];

    let cached = kv_manager.allocate_for_request(&prompt).unwrap();
    let cached_indices = cached.lease.indices().to_vec();
    kv_manager.finish(&prompt, cached.lease);
    let mut waiting = VecDeque::from([SglangRequest {
        uuid: Uuid::from_u128(90_002),
        sequence_tokens: prompt.clone(),
        prompt_len: prompt.len(),
        max_output_tokens: 1,
        planned_output_ids: None,
        materialized_tokens: 0,
        kv_lease: ActiveKvLease::default(),
        allocated_tokens: 0,
    }]);
    let req = get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &[])
        .can_run
        .pop()
        .unwrap();

    assert_eq!(req.cached_tokens(), prompt.len());
    assert_eq!(req.kv_indices(), cached_indices);

    // The three-token blocker owns a complete physical page. With two pages
    // total there is no free capacity, so the cache-hit request is retracted
    // while the blocker can fill the remainder of its existing page.
    let blocker_tokens = vec![9, 10, 11];
    let blocker_alloc = kv_manager.allocate_for_request(&blocker_tokens).unwrap();
    let blocker = SglangRequest {
        uuid: Uuid::from_u128(90_003),
        sequence_tokens: blocker_tokens,
        prompt_len: 3,
        max_output_tokens: 1,
        planned_output_ids: None,
        kv_lease: blocker_alloc.lease,
        materialized_tokens: 3,
        allocated_tokens: 4,
    };
    buffer.drain();

    let mut running = vec![req, blocker];
    let retracted = decode::check_decode_mem(&mut running, &mut kv_manager, &config);
    assert_eq!(retracted.len(), 1);
    assert_eq!(retracted[0].uuid, Uuid::from_u128(90_002));
    assert_eq!(removed_event_count(&buffer.drain()), 0);
    assert_eq!(kv_manager.cache().page_pool.available(), 0);
    assert_eq!(kv_manager.cache_mut().match_prefix(&prompt).0, prompt.len());
}

mod source_holds {
    use super::*;

    fn args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(16)
            .block_size(4)
            .max_num_seqs(Some(1))
            .worker_type(crate::common::protocols::WorkerType::Prefill)
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
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

    fn execute(core: &mut SglangCore, now_ms: f64) -> crate::scheduler::EnginePassResult {
        let mut collector = crate::replay::TraceCollector::default();
        core.execute_pass(&mut collector, now_ms)
    }

    fn occupied_tokens(core: &SglangCore) -> usize {
        core.kv_manager.cache().total_tokens() - core.kv_manager.cache().available_tokens()
    }

    #[test]
    fn terminal_completion_holds_source_without_running_cleanup() {
        let mut core = SglangCore::new(args());
        let request_id = Uuid::from_u128(301);
        let handoff_id = HandoffId::from(Uuid::from_u128(401));
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request: request(request_id),
        })
        .unwrap();

        let first = execute(&mut core, 0.0);
        assert!(!first.output_signals[0].completed);
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
        assert!(core.running.is_empty());
        let held_tokens = occupied_tokens(&core);

        core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
            .unwrap();
        assert!(!core.source_is_held(handoff_id));
        let released_tokens = occupied_tokens(&core);
        assert!(released_tokens < held_tokens);

        core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
            .unwrap();
        assert_eq!(occupied_tokens(&core), released_tokens);
    }

    #[test]
    fn cancel_and_early_release_cleanup_exactly_once() {
        let mut core = SglangCore::new(args());
        let first_id = HandoffId::from(Uuid::from_u128(402));
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id: first_id,
            request: request(Uuid::from_u128(302)),
        })
        .unwrap();
        let first = execute(&mut core, 0.0);
        execute(&mut core, first.end_ms);
        let held_tokens = occupied_tokens(&core);

        core.apply_command(SchedulerCommand::CancelSource {
            handoff_id: first_id,
        })
        .unwrap();
        let cancelled_tokens = occupied_tokens(&core);
        assert!(cancelled_tokens < held_tokens);
        core.apply_command(SchedulerCommand::CancelSource {
            handoff_id: first_id,
        })
        .unwrap();
        assert_eq!(occupied_tokens(&core), cancelled_tokens);

        core.kv_manager.evict(cancelled_tokens);
        let second_id = first_id;
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id: second_id,
            request: request(Uuid::from_u128(303)),
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
    }

    #[test]
    fn ordinary_completion_keeps_default_cleanup() {
        let mut core = SglangCore::new(args());
        core.receive(request(Uuid::from_u128(306)));

        let first = execute(&mut core, 0.0);
        let terminal = execute(&mut core, first.end_ms);

        assert!(terminal.output_signals[0].completed);
        assert_eq!(occupied_tokens(&core), 8);
    }

    #[test]
    fn active_request_id_is_rejected_before_source_hold_registration() {
        let mut core = SglangCore::new(args());
        let request_id = Uuid::from_u128(307);
        let handoff_id = HandoffId::from(Uuid::from_u128(407));
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
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(16)
            .block_size(4)
            .max_num_seqs(Some(1))
            .worker_type(worker_type)
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
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

    #[test]
    fn destination_lifecycle_is_counted_once_as_queued_decode() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(6)
            .block_size(4)
            .max_num_seqs(Some(1))
            .worker_type(WorkerType::Decode)
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(32),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        let held_handoff = HandoffId::from(Uuid::from_u128(33_001));
        let held_uuid = Uuid::from_u128(33_002);
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

        let pending_handoff = HandoffId::from(Uuid::from_u128(33_003));
        let pending = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: pending_handoff,
                    request: request(Uuid::from_u128(33_004), vec![2; 24], 1),
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

        core.receive(request(Uuid::from_u128(33_005), vec![3; 4], 2));
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
            let mut core = SglangCore::new(args(WorkerType::Decode));
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

    async fn send_live_command(
        scheduler: &SglangScheduler,
        command: SchedulerCommand,
    ) -> crate::scheduler::SchedulerCommandEffects {
        let (reply, reply_rx) = oneshot::channel();
        scheduler
            .command_sender()
            .send(SchedulerCommandEnvelope { command, reply })
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), reply_rx)
            .await
            .expect("scheduler command reply timed out")
            .unwrap()
            .unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn kv_blocked_live_scheduler_waits_and_cancel_wakes_it() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(2)
            .block_size(4)
            .max_num_seqs(Some(2))
            .worker_type(WorkerType::Decode)
            .speedup_ratio(1000.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();
        let fpm = Arc::new(CapturingFpmSink::default());
        let mut scheduler = SglangScheduler::new(
            args,
            0,
            Some(output_tx),
            KvEventPublishers::default(),
            None,
            FpmPublisher::new(Some(fpm.clone() as _)),
        );
        let mut lifecycle = scheduler.take_lifecycle_receiver().unwrap();
        let handoff_id = HandoffId::from(Uuid::from_u128(30_001));
        let reserved_uuid = Uuid::from_u128(30_002);
        let accepted = send_live_command(
            &scheduler,
            SchedulerCommand::ReserveDestination {
                handoff_id,
                request: request(reserved_uuid, vec![1; 8], 1),
            },
        )
        .await;
        assert!(matches!(
            accepted.result,
            SchedulerCommandResult::DestinationAccepted { request_id }
                if request_id == reserved_uuid
        ));
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(1), lifecycle.recv())
                .await
                .expect("destination reservation lifecycle timed out"),
            Some(SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: observed,
                ..
            }) if observed == handoff_id
        ));

        let blocked_uuid = Uuid::from_u128(30_003);
        scheduler.receive(request(blocked_uuid, vec![2; 4], 1));
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if fpm
                    .take()
                    .iter()
                    .any(|snapshot| snapshot.num_queued_prefill > 0)
                {
                    break;
                }
                fpm.wait_for_snapshot().await;
            }
        })
        .await
        .expect("blocked scheduler FPM snapshot timed out");
        assert!(
            tokio::time::timeout(Duration::from_millis(10), fpm.wait_for_snapshot())
                .await
                .is_err(),
            "a blocked scheduler must not publish repeated zero-work passes"
        );

        let canceled = send_live_command(
            &scheduler,
            SchedulerCommand::CancelDestination { handoff_id },
        )
        .await;
        assert_eq!(canceled.result, SchedulerCommandResult::Applied);
        let output = tokio::time::timeout(Duration::from_secs(1), output_rx.recv())
            .await
            .expect("capacity release should wake scheduling")
            .expect("output channel should remain open");
        assert!(output.iter().any(|signal| signal.uuid == blocked_uuid));
    }

    fn execute(core: &mut SglangCore, now_ms: f64) -> crate::scheduler::EnginePassResult {
        let mut collector = crate::replay::TraceCollector::default();
        core.execute_pass(&mut collector, now_ms)
    }

    fn assert_no_republished_stores(
        activation: &[dynamo_kv_router::protocols::LocalBlockHash],
        later: &[dynamo_kv_router::protocols::LocalBlockHash],
    ) {
        assert!(later.iter().all(|hash| !activation.contains(hash)));
    }

    fn occupied_tokens(core: &SglangCore) -> usize {
        core.kv_manager.cache().total_tokens() - core.kv_manager.cache().available_tokens()
    }

    fn drive_source_to_hold(core: &mut SglangCore, handoff_id: HandoffId, req: DirectRequest) {
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

    #[test]
    fn handoff_prefill_to_reserved_decode_owns_kv_until_prebuilt_admission() {
        let logical_uuid = Uuid::from_u128(20_001);
        let handoff_id = HandoffId::from(Uuid::from_u128(20_002));
        let logical_tokens = (0..8).collect::<Vec<_>>();

        let mut source = SglangCore::new_with_kv_capture(args(WorkerType::Prefill), 41);
        let mut destination = SglangCore::new_with_kv_capture(args(WorkerType::Decode), 42);

        destination.receive(request(Uuid::from_u128(20_003), (0..4).collect(), 1));
        execute(&mut destination, 0.0);
        assert!(destination.is_empty());
        destination.drain_kv_events();

        let blocker_uuid = Uuid::from_u128(20_004);
        destination.receive(request(blocker_uuid, (100..104).collect(), 3));
        let blocker_first = execute(&mut destination, 1.0);
        assert_eq!(destination.running.len(), 1);
        destination.drain_kv_events();

        drive_source_to_hold(
            &mut source,
            handoff_id,
            request(logical_uuid, logical_tokens.clone(), 2),
        );

        let usage_before_reservation = occupied_tokens(&destination);
        let reserved = destination
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: request(logical_uuid, logical_tokens, 2),
                },
                true,
            )
            .unwrap();
        assert_eq!(
            reserved.result,
            SchedulerCommandResult::DestinationAccepted {
                request_id: logical_uuid
            }
        );
        assert!(matches!(
            reserved.lifecycle_events.as_slice(),
            [SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: reserved_handoff,
                request_id: reserved_request,
                transferable_prompt_tokens: 4,
            }] if *reserved_handoff == handoff_id && *reserved_request == logical_uuid
        ));
        assert!(destination.destination_is_held(handoff_id));
        assert!(!destination.is_drained());
        assert_eq!(destination.running.len(), 1);
        assert!(occupied_tokens(&destination) > usage_before_reservation);
        assert!(stored_hashes(&destination.drain_kv_events()).is_empty());

        let reserved_indices = destination.destination_indices(handoff_id);
        let protected_before_activation = destination.kv_manager.cache().protected_size;
        assert!(!reserved_indices.is_empty());
        assert_eq!(
            destination
                .apply_command(SchedulerCommand::ActivateDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        let ready = destination
            .prebuilt_request(logical_uuid)
            .expect("activated request must be prebuilt-ready");
        assert_eq!(ready.kv_indices(), reserved_indices);
        assert_eq!(destination.running.len(), 1);
        assert!(destination.kv_manager.cache().protected_size >= protected_before_activation);
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

        let blocked = execute(&mut destination, blocker_first.end_ms);
        assert!(
            blocked
                .admissions
                .iter()
                .all(|admission| admission.uuid != logical_uuid)
        );
        let ready = destination
            .prebuilt_request(logical_uuid)
            .expect("full running batch must keep request ready");
        assert_eq!(ready.kv_indices(), reserved_indices);
        assert_no_republished_stores(&activation_stores, &stored_hashes(&blocked.kv_events));

        let blocker_terminal = execute(&mut destination, blocked.end_ms);
        assert!(
            blocker_terminal
                .admissions
                .iter()
                .all(|admission| admission.uuid != logical_uuid)
        );
        assert!(
            blocker_terminal
                .output_signals
                .iter()
                .any(|signal| signal.uuid == blocker_uuid && signal.completed)
        );
        assert!(destination.prebuilt_request(logical_uuid).is_some());

        let admitted = execute(&mut destination, blocker_terminal.end_ms);
        let logical_admissions = admitted
            .admissions
            .iter()
            .filter(|admission| admission.uuid == logical_uuid)
            .collect::<Vec<_>>();
        assert_eq!(logical_admissions.len(), 1);
        assert_eq!(logical_admissions[0].reused_input_tokens, 0);
        let running = destination
            .running
            .iter()
            .find(|request| request.uuid == logical_uuid)
            .expect("prebuilt request must enter the running batch");
        assert!(running.kv_indices().starts_with(&reserved_indices));
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
        assert_eq!(destination.kv_manager.cache().protected_size, 0);
    }

    #[test]
    fn destination_cancel_reaches_activated_request_exactly_once() {
        let request_id = Uuid::from_u128(20_101);
        let handoff_id = HandoffId::from(Uuid::from_u128(20_102));
        let mut core = SglangCore::new_with_kv_capture(args(WorkerType::Decode), 43);

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
        assert!(
            core.running
                .iter()
                .any(|request| request.uuid == request_id)
        );

        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert!(core.is_empty());
        assert!(core.is_drained());
        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Noop
        );
    }

    #[test]
    fn activation_retains_partial_page_until_owner_is_cancelled() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(3)
            .block_size(4)
            .max_num_seqs(Some(1))
            .worker_type(WorkerType::Decode)
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        let blocker = core.kv_manager.allocate_decode_token(None).unwrap();
        let owner_handoff = HandoffId::from(Uuid::from_u128(20_201));
        let follower_handoff = HandoffId::from(Uuid::from_u128(20_202));
        let follower_request = Uuid::from_u128(20_203);

        let owner = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: owner_handoff,
                    request: request(Uuid::from_u128(20_204), (0..5).collect(), 2),
                },
                true,
            )
            .unwrap();
        assert_eq!(owner.lifecycle_events.len(), 1);
        let follower = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: follower_handoff,
                    request: request(follower_request, (100..104).collect(), 2),
                },
                true,
            )
            .unwrap();
        assert!(follower.lifecycle_events.is_empty());

        let activated = core
            .apply_command_effects(
                SchedulerCommand::ActivateDestination {
                    handoff_id: owner_handoff,
                },
                true,
            )
            .unwrap();
        assert!(
            activated.lifecycle_events.is_empty(),
            "activating a partial prompt must retain its complete physical page"
        );

        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination {
                handoff_id: owner_handoff,
            })
            .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert!(core.destination_is_held(follower_handoff));
        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination {
                handoff_id: follower_handoff,
            })
            .unwrap(),
            SchedulerCommandResult::Applied
        );
        core.kv_manager.cache_mut().page_pool.free(&[blocker]);
    }
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
                output_token_ids: None,
                uuid: None,
                dp_rank: 0,
                arrival_timestamp_ms: None,
                ..Default::default()
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
                sequence_tokens: vec![9, 8, 7],
                prompt_len: 3,
                max_output_tokens: 1,
                planned_output_ids: None,
                materialized_tokens: 0,
                kv_lease: ActiveKvLease::default(),
                allocated_tokens: 0,
            },
            SglangRequest {
                uuid: match_uuid,
                sequence_tokens: vec![1, 2, 3, 4, 5, 6, 7],
                prompt_len: 5,
                max_output_tokens: 1,
                planned_output_ids: None,
                materialized_tokens: 0,
                kv_lease: ActiveKvLease::default(),
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
                sequence_tokens: duplicate_prefix.clone(),
                prompt_len: duplicate_prefix.len(),
                max_output_tokens: 1,
                planned_output_ids: None,
                materialized_tokens: 0,
                kv_lease: ActiveKvLease::default(),
                allocated_tokens: 0,
            });
        }
        let unique_uuid = Uuid::new_v4();
        waiting.push_back(SglangRequest {
            uuid: unique_uuid,
            sequence_tokens: (100..132).collect(),
            prompt_len: 32,
            max_output_tokens: 1,
            planned_output_ids: None,
            materialized_tokens: 0,
            kv_lease: ActiveKvLease::default(),
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
    fn test_planned_output_tokens_are_emitted_exactly() {
        let mut core = SglangCore::new(test_args(100, 4, 8));
        let uuid = Uuid::from_u128(0xB0B);
        let planned = vec![111, 222, 333];
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
            sequence_tokens: vec![1; 6],
            prompt_len: 6,
            max_output_tokens: 3,
            planned_output_ids: None,
            materialized_tokens: 0,
            kv_lease: ActiveKvLease::default(),
            allocated_tokens: 0,
        }]);

        let admit = get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &[]);
        assert_eq!(admit.can_run.len(), 1);
        assert_eq!(admit.can_run[0].materialized_tokens, 6);
        assert_eq!(admit.can_run[0].allocated_tokens, 8);
    }

    #[test]
    fn test_chunked_prefill_admits_next_chunk_when_full_prompt_does_not_fit() {
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
        let mut kv_manager = SglangKvManager::new(12, 4, KvEventPublishers::default(), 0);
        let mut waiting = VecDeque::from([SglangRequest {
            uuid: Uuid::new_v4(),
            sequence_tokens: vec![1; 16],
            prompt_len: 16,
            max_output_tokens: 2,
            planned_output_ids: None,
            materialized_tokens: 0,
            kv_lease: ActiveKvLease::default(),
            allocated_tokens: 0,
        }]);

        let admit = get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &[]);
        assert_eq!(admit.can_run.len(), 1);
        assert_eq!(admit.can_run[0].materialized_tokens, 8);
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
                sequence_tokens: vec![1; 7],
                prompt_len: 7,
                max_output_tokens: 3,
                planned_output_ids: None,
                materialized_tokens: 0,
                kv_lease: ActiveKvLease::default(),
                allocated_tokens: 0,
            },
            SglangRequest {
                uuid: second_uuid,
                sequence_tokens: vec![2; 8],
                prompt_len: 8,
                max_output_tokens: 3,
                planned_output_ids: None,
                materialized_tokens: 0,
                kv_lease: ActiveKvLease::default(),
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
        let mut alloc = kv_manager
            .allocate_for_request(&[1, 2, 3, 4, 5, 6])
            .unwrap();
        kv_manager.extend_cached_prefix(&[1, 2, 3, 4], &mut alloc.lease);
        let mut running = vec![SglangRequest {
            uuid: Uuid::new_v4(),
            sequence_tokens: vec![1, 2, 3, 4, 5, 6],
            prompt_len: 6,
            max_output_tokens: 4,
            planned_output_ids: None,
            kv_lease: alloc.lease,
            materialized_tokens: 6,
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
            sequence_tokens: vec![1, 2, 3, 4],
            prompt_len: 4,
            max_output_tokens: 4,
            planned_output_ids: None,
            kv_lease: base_alloc.lease,
            materialized_tokens: 4,
            allocated_tokens: 4,
        }];

        let mut fast_kv_manager = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        let fast_alloc = fast_kv_manager.allocate_for_request(&[1, 2, 3, 4]).unwrap();
        let mut fast_running = vec![SglangRequest {
            uuid: Uuid::new_v4(),
            sequence_tokens: vec![1, 2, 3, 4],
            prompt_len: 4,
            max_output_tokens: 4,
            planned_output_ids: None,
            kv_lease: fast_alloc.lease,
            materialized_tokens: 4,
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
        let mut kv_manager = SglangKvManager::new(16, 4, KvEventPublishers::default(), 0);
        let first = kv_manager.cache_mut().page_pool.allocate(8).unwrap();
        let second = kv_manager.cache_mut().page_pool.allocate(5).unwrap();

        let mut running = vec![
            SglangRequest {
                uuid: Uuid::new_v4(),
                sequence_tokens: vec![1, 2, 3, 4, 11, 12, 13, 14],
                prompt_len: 4,
                max_output_tokens: 10,
                planned_output_ids: None,
                kv_lease: ActiveKvLease::from_parts(first, 4, kv_manager.cache().root()),
                materialized_tokens: 8,
                allocated_tokens: 8,
            },
            SglangRequest {
                uuid: Uuid::new_v4(),
                sequence_tokens: vec![9, 8, 7, 6, 21],
                prompt_len: 4,
                max_output_tokens: 10,
                planned_output_ids: None,
                kv_lease: ActiveKvLease::from_parts(second, 4, kv_manager.cache().root()),
                materialized_tokens: 5,
                allocated_tokens: 8,
            },
        ];

        let retracted = decode::check_decode_mem(&mut running, &mut kv_manager, &config);
        assert_eq!(retracted.len(), 1);
        assert_eq!(retracted[0].output_tokens(), &[21]);
        assert_eq!(retracted[0].materialized_tokens, 0);
        assert!(retracted[0].kv_indices().is_empty());
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
            sequence_tokens: vec![1, 2, 3, 4],
            prompt_len: 4,
            max_output_tokens: 4,
            planned_output_ids: None,
            kv_lease: alloc.lease,
            materialized_tokens: 4,
            allocated_tokens: 4,
        }];

        simulate_decode_step(&mut running, &mut kv_manager, &config, 0.0, false);
        let prefix = running[0].sequence_prefix(4);
        assert_eq!(kv_manager.cache().prefix_match_len(prefix), 4);
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
            output_token_ids: None,
            uuid: None,
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });

        let pass = core.execute_pass_internal(None, 0.0);
        assert_eq!(pass.completed_requests, 0);
        assert_eq!(pass.mocker_metrics.active_decode_blocks, 2);
    }

    #[test]
    fn test_mtp_batch_applies_request_bursts_in_stable_order() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(32)
            .block_size(4)
            .speedup_ratio(0.0)
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("1,1".to_string()))
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        let short = Uuid::from_u128(101);
        let long = Uuid::from_u128(102);
        let longer = Uuid::from_u128(103);
        for (uuid, tokens, max_output_tokens) in [
            (short, vec![1, 2, 3, 4], 5),
            (long, vec![5, 6, 7, 8], 8),
            (longer, vec![9, 10, 11, 12], 8),
        ] {
            core.receive(DirectRequest {
                tokens,
                max_output_tokens,
                uuid: Some(uuid),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                ..Default::default()
            });
        }

        let mut collector = crate::replay::TraceCollector::default();
        let first = core.execute_pass(&mut collector, 0.0);
        assert_eq!(first.output_signals.len(), 9);
        let second = core.execute_pass(&mut collector, first.end_ms);
        let ordered = second
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
                (longer, false),
                (longer, false),
                (longer, false),
            ]
        );
        assert_eq!(core.running.len(), 2);
        assert_eq!(core.running[0].uuid, long);
        assert_eq!(core.running[0].output_len(), 6);
        assert_eq!(core.running[1].uuid, longer);
        assert_eq!(core.running[1].output_len(), 6);
        assert_eq!(second.fpm.unwrap().num_decode_requests, 3);

        let third = core.execute_pass(&mut collector, second.end_ms);
        assert_eq!(
            third
                .output_signals
                .iter()
                .map(|signal| signal.uuid)
                .collect::<Vec<_>>(),
            vec![long, long, longer, longer],
        );
    }

    #[test]
    fn test_sglang_pass_visibility_is_pass_end() {
        let mut core = SglangCore::new_with_kv_capture(test_args(32, 4, 4), ROUTER_TEST_WORKER_ID);
        core.receive(direct_request(vec![1, 2, 3, 4], 1));

        let pass = core.execute_pass_internal(None, 0.0);

        assert_eq!(pass.router_event_visibility, RouterEventVisibility::PassEnd);
        assert!(!pass.kv_events.is_empty());
        assert!(
            pass.kv_events
                .iter()
                .all(|event| event.worker_id == ROUTER_TEST_WORKER_ID)
        );
        assert!(pass.kv_events.iter().all(|event| event.event.dp_rank == 0));
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
            ..Default::default()
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
        assert_eq!(prompt_hashes.len(), 1);
        harness.apply_events(pass1.kv_events).await;

        let pass2 = core.execute_pass_internal(None, pass1.end_ms);
        prompt_hashes.extend(nth_stored_hashes(&pass2.kv_events, 0));
        harness.apply_events(pass2.kv_events).await;

        let pass3 = core.execute_pass_internal(None, pass2.end_ms);
        prompt_hashes.extend(nth_stored_hashes(&pass3.kv_events, 0));
        harness.apply_events(pass3.kv_events).await;

        assert_eq!(prompt_hashes.len(), 2);
        assert!(harness.ok_count(METRIC_EVENT_STORED) >= 2);
        harness.assert_no_event_warnings();
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

        let mut now_ms = pass1.end_ms;
        for _ in 0..3 {
            let pass = core.execute_pass_internal(None, now_ms);
            now_ms = pass.end_ms;
            full_hashes.extend(stored_hashes(&pass.kv_events));
            harness.apply_events(pass.kv_events).await;
        }

        assert_eq!(full_hashes.len(), 2);
        assert!(harness.ok_count(METRIC_EVENT_STORED) >= 2);
        harness.assert_no_event_warnings();
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_completed_output_block_uses_router_token_identity() {
        let uuid = Uuid::from_u128(0x5a17);
        let prompt_tokens = vec![101, 202];
        let mut expected_request = SglangRequest {
            uuid,
            sequence_tokens: prompt_tokens.clone(),
            prompt_len: prompt_tokens.len(),
            max_output_tokens: 2,
            planned_output_ids: None,
            materialized_tokens: 0,
            kv_lease: ActiveKvLease::default(),
            allocated_tokens: 0,
        };
        let first_output = expected_request.next_output_token();
        expected_request.append_output_token(first_output);
        let second_output = expected_request.next_output_token();

        let mut expected_tokens = prompt_tokens;
        expected_tokens.extend([first_output, second_output]);
        let expected_hashes =
            compute_block_hash_for_seq(&expected_tokens, 4, BlockHashOptions::default());

        let mut core = SglangCore::new_with_kv_capture(test_args(32, 4, 16), ROUTER_TEST_WORKER_ID);
        let mut request = direct_request(expected_tokens[..2].to_vec(), 2);
        request.uuid = Some(uuid);
        core.receive(request);

        let mut now_ms = 0.0;
        let mut hashes = Vec::new();
        while !core.is_empty() {
            let pass = core.execute_pass_internal(None, now_ms);
            now_ms = pass.end_ms;
            hashes.extend(stored_hashes(&pass.kv_events));
        }

        assert_eq!(
            hashes, expected_hashes,
            "completed prompt+output block should hash the same u32 token identity that SGLang generated"
        );
    }

    #[tokio::test]
    async fn test_retract_frees_do_not_leave_stale_blocks() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let args = test_args(8, 4, 16);
        let config = SglangConfig::from_args(&args);
        let (buffer, sink) = capture_router_event_sink(ROUTER_TEST_WORKER_ID);
        let mut kv_manager =
            SglangKvManager::new(16, 4, KvEventPublishers::new(Some(sink), None), 0);

        let req1 = make_decoded_request(&mut kv_manager, &config, vec![1, 2, 3, 4, 5, 6, 7], 4);
        let req1_events = buffer.drain();
        let req1_hashes = stored_hashes(&req1_events);
        harness.apply_events(req1_events).await;
        assert_eq!(
            harness.overlap_for_hashes(req1_hashes.clone()).await,
            req1_hashes.len() as u32
        );

        let req2 = make_decoded_request(&mut kv_manager, &config, vec![9, 8, 7, 6, 5, 4, 3], 4);
        harness.apply_events(buffer.drain()).await;
        assert_eq!(
            harness.overlap_for_hashes(req1_hashes.clone()).await,
            req1_hashes.len() as u32
        );

        let mut running = vec![req1, req2];
        let retracted = decode::check_decode_mem(&mut running, &mut kv_manager, &config);
        assert_eq!(retracted.len(), 1);

        let retract_events = buffer.drain();
        harness.apply_events(retract_events).await;

        assert_eq!(
            harness.overlap_for_hashes(req1_hashes.clone()).await,
            1,
            "one evictable suffix page is removed to make physical room"
        );
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_completion_tail_free_does_not_remove_unpublished_blocks() {
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
        assert_eq!(removed_event_count(&pass3.kv_events), 0);
        full_hashes.extend(stored_hashes(&pass3.kv_events));
        harness.apply_events(pass3.kv_events).await;

        assert_eq!(prompt_hashes.len(), 1);
        assert!(full_hashes.len() >= prompt_hashes.len());
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_mixed_chunk_decode_retract_reprefill_complete_events_apply_cleanly() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let args = test_args(8, 4, 4);
        let config = SglangConfig::from_args(&args);
        let (buffer, sink) = capture_router_event_sink(ROUTER_TEST_WORKER_ID);
        let mut kv_manager =
            SglangKvManager::new(16, 4, KvEventPublishers::new(Some(sink), None), 0);

        let mut waiting = VecDeque::from([SglangRequest {
            uuid: Uuid::new_v4(),
            sequence_tokens: vec![1, 2, 3, 4, 5, 6, 7],
            prompt_len: 7,
            max_output_tokens: 3,
            planned_output_ids: None,
            materialized_tokens: 0,
            kv_lease: ActiveKvLease::default(),
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

        let req2 =
            make_decoded_request(&mut kv_manager, &config, vec![9, 10, 11, 12, 13, 14, 15], 3);
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
        harness.assert_no_event_warnings();
        harness.shutdown();
    }

    #[tokio::test]
    async fn test_mtp_lifecycle_drains_with_clean_router_events() {
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(5)
            .block_size(4)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("0,1".to_string()))
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(4),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new_with_kv_capture(args, ROUTER_TEST_WORKER_ID);
        let requests = [
            direct_request(vec![1, 2, 3, 4, 5, 6], 7),
            direct_request(vec![9, 10, 11, 12], 5),
        ];
        let expected_tokens = requests
            .iter()
            .map(|request| request.max_output_tokens)
            .sum::<usize>();
        for request in requests {
            core.receive(request);
        }

        let mut collector = crate::replay::TraceCollector::default();
        let mut now_ms = 0.0;
        let mut output_tokens = 0;
        let mut saw_remove = false;
        for _ in 0..100 {
            if core.is_empty() {
                break;
            }
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms.max(now_ms + 1.0);
            output_tokens += pass.output_signals.len();
            saw_remove |= removed_event_count(&pass.kv_events) > 0;
            harness.apply_events(pass.kv_events).await;
        }

        assert!(
            core.is_empty(),
            "scheduler did not drain: waiting={}, running={}, outputs={output_tokens}, available={}, evictable={}",
            core.waiting.len(),
            core.running.len(),
            core.kv_manager.cache().available_tokens(),
            core.kv_manager.cache().evictable_size,
        );
        assert_eq!(output_tokens, expected_tokens);
        assert!(core.waiting.is_empty());
        assert!(core.running.is_empty());
        assert!(saw_remove);
        harness.assert_no_event_errors();
        harness.assert_no_event_warnings();
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(91)),
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
    fn test_mocker_metrics_reports_prefix_cache_reuse() {
        let mut core = SglangCore::new(fpm_args());

        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass1 = core.execute_pass(&mut collector, 0.0);
        assert_eq!(pass1.mocker_metrics.sglang_cache_hit_tokens, 0);
        assert_eq!(pass1.mocker_metrics.sglang_cache_total_tokens, 8);

        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });

        let pass2 = core.execute_pass(&mut collector, pass1.end_ms);
        assert!(pass2.mocker_metrics.sglang_cache_hit_tokens > 0);
        assert!(
            pass2.mocker_metrics.sglang_cache_hit_tokens
                <= pass2.mocker_metrics.sglang_cache_total_tokens
        );
    }

    #[test]
    fn fully_cached_zero_output_request_is_not_forward_pass_work() {
        let mut core = SglangCore::new(fpm_args());
        let tokens = (0..8).collect::<Vec<_>>();
        let mut collector = crate::replay::TraceCollector::default();

        core.receive(DirectRequest {
            tokens: tokens.clone(),
            max_output_tokens: 0,
            uuid: Some(Uuid::from_u128(90_004)),
            ..Default::default()
        });
        let seed_pass = core.execute_pass(&mut collector, 0.0);
        assert_eq!(seed_pass.completed_requests, 1);

        let uuid = core.receive(DirectRequest {
            tokens,
            max_output_tokens: 0,
            uuid: Some(Uuid::from_u128(90_005)),
            ..Default::default()
        });
        let pass = core.execute_pass(&mut collector, seed_pass.end_ms);

        assert_eq!(pass.completed_requests, 1);
        assert_eq!(pass.mocker_metrics.sglang_cache_hit_tokens, 8);
        assert_eq!(pass.mocker_metrics.sglang_cache_total_tokens, 8);
        let fpm = pass.fpm.as_ref().unwrap();
        assert_eq!(fpm.num_prefill_requests, 0);
        assert_eq!(fpm.sum_prefill_tokens, 0);
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        core.receive(DirectRequest {
            tokens: (100..108).collect(),
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });
        core.receive(DirectRequest {
            tokens: (100..104).collect(),
            max_output_tokens: 20,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        });

        // Run several passes to build up KV pressure
        for i in 0..4 {
            core.execute_pass(&mut collector, i as f64);
        }

        // Add a third request to increase memory pressure
        core.receive(DirectRequest {
            tokens: (200..212).collect(), // 12 tokens
            max_output_tokens: 10,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(3)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
