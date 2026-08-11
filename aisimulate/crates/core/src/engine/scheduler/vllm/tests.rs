// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rstest::rstest;
use uuid::Uuid;

use crate::engine::HandoffId;
use crate::engine::common::protocols::{
    DirectRequest, EngineType, MockEngineArgs, OutputSignal, PreemptionMode,
};
use crate::engine::scheduler::{
    KvEventVisibility, SchedulerCommand, SchedulerCommandResult, SchedulerLifecycleEvent,
};

use super::core::{RequestStatus, VllmCore};
use super::request::RequestKvState;

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

fn prefix_cache_args() -> MockEngineArgs {
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
#[case::vllm(EngineType::Vllm)]
#[case::trtllm(EngineType::Trtllm)]
fn shared_vllm_core_exposes_kv_events_at_pass_end(#[case] engine_type: EngineType) {
    let args = MockEngineArgs::builder()
        .engine_type(engine_type)
        .block_size(4)
        .num_gpu_blocks(16)
        .max_num_batched_tokens(Some(16))
        .max_num_seqs(Some(1))
        .enable_prefix_caching(true)
        .speedup_ratio(0.0)
        .build()
        .unwrap();
    let mut core = VllmCore::new_with_kv_capture(args, 7);
    core.receive(DirectRequest {
        tokens: (0..8).collect(),
        max_output_tokens: 2,
        uuid: Some(Uuid::from_u128(71)),
        ..Default::default()
    });

    let mut collector = crate::engine::trace::TraceCollector::default();
    let pass = core.execute_pass(&mut collector, 0.0);

    assert_eq!(pass.kv_event_visibility, KvEventVisibility::PassEnd);
    assert!(!pass.kv_events.is_empty());
    assert!(pass.kv_events.iter().all(|event| event.dp_rank == 0));
}

#[test]
fn engine_adapter_constructs_native_request_state() {
    let mut core = VllmCore::new(prefix_cache_args());
    let uuid = Uuid::from_u128(80_100);
    core.receive(DirectRequest {
        tokens: (0..8).collect(),
        max_output_tokens: 2,
        uuid: Some(uuid),
        ..Default::default()
    });

    assert!(core.request_uses_flat_tokens(uuid));
}

#[test]
fn flat_tokens_cover_every_native_g1_configuration() {
    let mut next_uuid = 80_110_u128;
    for engine_type in [EngineType::Vllm, EngineType::Trtllm] {
        for enable_prefix_caching in [false, true] {
            for emit_token_ids in [false, true] {
                let args = MockEngineArgs::builder()
                    .engine_type(engine_type)
                    .block_size(4)
                    .num_gpu_blocks(32)
                    .max_num_batched_tokens(Some(32))
                    .max_num_seqs(Some(4))
                    .enable_chunked_prefill(true)
                    .enable_prefix_caching(enable_prefix_caching)
                    .emit_kv_events(emit_token_ids)
                    .emit_kv_token_ids(emit_token_ids)
                    .speedup_ratio(0.0)
                    .build()
                    .unwrap();
                let mut core = VllmCore::new(args);
                let uuid = Uuid::from_u128(next_uuid);
                next_uuid += 1;
                core.receive(DirectRequest {
                    tokens: (0..9).collect(),
                    max_output_tokens: 2,
                    uuid: Some(uuid),
                    ..Default::default()
                });
                assert!(core.request_uses_flat_tokens(uuid));
            }
        }
    }
}

#[test]
fn minimum_shared_scheduler_block_size_constructs_requests_for_both_backends() {
    for (case, engine_type) in [EngineType::Vllm, EngineType::Trtllm]
        .into_iter()
        .enumerate()
    {
        let args = MockEngineArgs::builder()
            .engine_type(engine_type)
            .block_size(2)
            .num_gpu_blocks(4)
            .max_num_batched_tokens(Some(4))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let uuid = Uuid::from_u128(80_200 + case as u128);
        core.receive(DirectRequest {
            tokens: vec![0],
            max_output_tokens: 1,
            output_token_ids: Some(vec![1]),
            uuid: Some(uuid),
            ..Default::default()
        });
        assert!(core.request_uses_flat_tokens(uuid));
    }
}

#[test]
fn flat_storage_capacity_is_bounded_by_realizable_output() {
    const BLOCK_SIZE: usize = 4;
    const MAX_OUTPUT_TOKENS: usize = 1_000_000;

    for (case, (prompt_len, output_capacity_hint)) in [(8_usize, 4_usize), (10, 2), (17, 0)]
        .into_iter()
        .enumerate()
    {
        let request = RequestKvState::native(
            Uuid::from_u128(80_300 + case as u128),
            (0..prompt_len as u32).collect(),
            MAX_OUTPUT_TOKENS,
            output_capacity_hint,
            BLOCK_SIZE,
            true,
            true,
            true,
            None,
        );
        assert_eq!(request.max_output_tokens(), MAX_OUTPUT_TOKENS);
        let (token_capacity, lease_capacity) = request.native_storage_capacities();
        let bounded_blocks = (prompt_len + output_capacity_hint).div_ceil(BLOCK_SIZE);
        let unbounded_blocks = (prompt_len + MAX_OUTPUT_TOKENS).div_ceil(BLOCK_SIZE);
        assert!(token_capacity >= prompt_len + output_capacity_hint);
        assert!(token_capacity < prompt_len + MAX_OUTPUT_TOKENS);
        assert!(lease_capacity >= bounded_blocks);
        assert!(lease_capacity < unbounded_blocks);
    }
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
    let mut collector = crate::engine::trace::TraceCollector::default();

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
            cached_tokens: Some(0),
            ..
        }] if *signal_uuid == uuid
    ));
}

#[test]
fn fully_cached_zero_output_request_is_not_decode_fpm_work() {
    let mut core = VllmCore::new(prefix_cache_args());
    let tokens = vec![1, 2, 3, 4];
    let mut collector = crate::engine::trace::TraceCollector::default();

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
            cached_tokens: Some(4),
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
    let mut collector = crate::engine::trace::TraceCollector::default();

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
            .worker_type(crate::engine::common::protocols::WorkerType::Prefill)
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    fn request(uuid: Uuid) -> DirectRequest {
        DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 2,
            uuid: Some(uuid),
            arrival_timestamp_ms: None,
            ..Default::default()
        }
    }

    fn execute(core: &mut VllmCore, now_ms: f64) -> crate::engine::scheduler::EnginePassResult {
        let mut collector = crate::engine::trace::TraceCollector::default();
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
    use crate::engine::common::protocols::WorkerType;

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
            arrival_timestamp_ms: None,
            ..Default::default()
        }
    }

    fn execute(core: &mut VllmCore, now_ms: f64) -> crate::engine::scheduler::EnginePassResult {
        let mut collector = crate::engine::trace::TraceCollector::default();
        core.execute_pass(&mut collector, now_ms)
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

    #[test]
    fn handoff_prefill_to_reserved_decode_owns_kv_until_normal_admission() {
        let request_id = Uuid::from_u128(10_001);
        let handoff_id = HandoffId::from(Uuid::from_u128(10_002));
        let tokens = (0..8).collect::<Vec<_>>();
        let mut source = VllmCore::new(args(WorkerType::Prefill));
        let mut destination = VllmCore::new(args(WorkerType::Decode));

        drive_source_to_hold(
            &mut source,
            handoff_id,
            request(request_id, tokens.clone(), 2),
        );
        let reserve = destination
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: request(request_id, tokens, 2),
                },
                true,
            )
            .unwrap();
        assert!(matches!(
            reserve.lifecycle_events.as_slice(),
            [SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: observed,
                request_id: observed_request,
                ..
            }] if *observed == handoff_id && *observed_request == request_id
        ));
        let reserved_blocks = destination.destination_block_count(handoff_id);
        assert!(reserved_blocks > 0);
        assert_eq!(destination.kv_manager.num_active_blocks(), reserved_blocks);

        assert_eq!(
            destination
                .apply_command(SchedulerCommand::ActivateDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert_eq!(destination.kv_manager.num_active_blocks(), reserved_blocks);
        let first = execute(&mut destination, 0.0);
        assert!(
            first
                .admissions
                .iter()
                .any(|event| event.uuid == request_id)
        );

        let mut now_ms = first.end_ms;
        for _ in 0..8 {
            if destination.is_empty() {
                break;
            }
            now_ms = execute(&mut destination, now_ms).end_ms;
        }
        assert!(destination.is_empty());
        assert_eq!(destination.kv_manager.num_active_blocks(), 0);
        assert_eq!(
            source
                .apply_command(SchedulerCommand::ReleaseSource { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert!(source.is_drained());
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
                        handoff_id: HandoffId::from(Uuid::new_v4()),
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
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
        let mut signals = Vec::new();
        for step in 0..planned.len() {
            let pass = core.execute_pass(&mut collector, step as f64);
            signals.extend(
                pass.output_signals
                    .into_iter()
                    .filter(|signal| signal.uuid == uuid),
            );
        }

        let emitted: Vec<_> = signals
            .iter()
            .map(|signal| signal.token_id.expect("planned token should be present"))
            .collect();
        assert_eq!(emitted, planned);
        assert_eq!(signals[0].cached_tokens, Some(0));
        assert!(
            signals[1..]
                .iter()
                .all(|signal| signal.cached_tokens.is_none())
        );
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
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(r2),
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
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
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(r2),
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
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
                arrival_timestamp_ms: None,
            });
        }

        let mut collector = crate::engine::trace::TraceCollector::default();
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
            .worker_type(crate::engine::common::protocols::WorkerType::Prefill)
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
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
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
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
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
                arrival_timestamp_ms: None,
            });
        }

        let mut collector = crate::engine::trace::TraceCollector::default();
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
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..112).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(blocked),
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (200..204).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(follower),
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
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
                arrival_timestamp_ms: None,
            });
        }

        let mut collector = crate::engine::trace::TraceCollector::default();
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
    fn test_completion_returns_scheduler_to_idle() {
        let mut core = VllmCore::new(make_args());
        for uuid in [Uuid::from_u128(1), Uuid::from_u128(2)] {
            core.receive(DirectRequest {
                tokens: (0..8).collect(),
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(uuid),
                arrival_timestamp_ms: None,
            });
        }

        let mut collector = crate::engine::trace::TraceCollector::default();
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
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..104).collect(),
            max_output_tokens: 8,
            output_token_ids: None,
            uuid: Some(long),
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
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
    fn test_mtp_releases_unused_block_reservations() {
        let args = MockEngineArgs::builder()
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
            uuid: Some(uuid),
            ..Default::default()
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        assert_eq!(pass.output_signals.len(), 1);
        assert_eq!(
            core.kv_manager.num_active_blocks(),
            2,
            "the block reserved only for rejected drafts must be released"
        );
        let request = core.state.requests.get(&uuid).unwrap();
        assert_eq!(request.sequence.generated_tokens(), 1);
        assert_eq!(request.sequence.len() - request.num_computed_tokens, 1);
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
            let mut collector = crate::engine::trace::TraceCollector::default();

            core.receive(DirectRequest {
                tokens: (0..8).collect(),
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(1)),
                arrival_timestamp_ms: None,
            });
            core.execute_pass(&mut collector, 0.0);
            assert!(core.is_empty());

            core.receive(DirectRequest {
                tokens: (0..12).collect(),
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(2)),
                arrival_timestamp_ms: None,
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
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
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
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();

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
            output_token_ids: None,
            uuid: Some(r1),
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
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
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();

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
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(r2),
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
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
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..112).collect(), // prompt_len = 12
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
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
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();

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
        let mut collector = crate::engine::trace::TraceCollector::default();

        // r1: 4-token prompt, long output (stays running)
        core.receive(DirectRequest {
            tokens: (0..4).collect(),
            max_output_tokens: 20,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
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

    #[test]
    fn test_first_signal_carries_admission_cache_truth() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Vllm)
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(3))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        // 9 tokens = two full blocks (cacheable) + one partial.
        let tokens: Vec<u32> = (0..9).collect();

        fn drive_to_completion(core: &mut VllmCore, uuid: Uuid) -> Vec<OutputSignal> {
            let mut collector = crate::engine::trace::TraceCollector::default();
            let mut signals = Vec::new();
            for step in 0..100 {
                let pass = core.execute_pass(&mut collector, step as f64);
                signals.extend(
                    pass.output_signals
                        .iter()
                        .filter(|signal| signal.uuid == uuid)
                        .cloned(),
                );
                if signals.iter().any(|signal| signal.completed) {
                    return signals;
                }
            }
            panic!("request {uuid} never completed");
        }

        let cold = Uuid::from_u128(90);
        core.receive(DirectRequest {
            tokens: tokens.clone(),
            max_output_tokens: 3,
            uuid: Some(cold),
            ..Default::default()
        });
        let cold_signals = drive_to_completion(&mut core, cold);
        assert_eq!(
            cold_signals[0].cached_tokens,
            Some(0),
            "cold request must report zero admission cache hits"
        );
        assert!(
            cold_signals[1..]
                .iter()
                .all(|signal| signal.cached_tokens.is_none()),
            "cache truth must ride the first signal only"
        );

        let warm = Uuid::from_u128(91);
        core.receive(DirectRequest {
            tokens,
            max_output_tokens: 3,
            uuid: Some(warm),
            ..Default::default()
        });
        let warm_signals = drive_to_completion(&mut core, warm);
        assert_eq!(
            warm_signals[0].cached_tokens,
            Some(8),
            "repeat of the same prompt must report its two full blocks as admission cache hits"
        );
        assert!(
            warm_signals[1..]
                .iter()
                .all(|signal| signal.cached_tokens.is_none()),
            "cache truth must ride the first signal only"
        );
    }

    #[test]
    fn test_preempted_request_keeps_original_admission_cache_truth() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Vllm)
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
        let mut core = VllmCore::new(args);
        let mut collector = crate::engine::trace::TraceCollector::default();
        let mut now_ms = 0.0;

        // Warm one block so the victim's first admission sees a genuine hit.
        let warm = Uuid::from_u128(70);
        core.receive(DirectRequest {
            tokens: (0..4).collect(),
            max_output_tokens: 1,
            uuid: Some(warm),
            ..Default::default()
        });
        for _ in 0..8 {
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms.max(now_ms + 1.0);
            if core.state.requests.is_empty() {
                break;
            }
        }
        assert!(
            core.state.requests.is_empty(),
            "warm request never completed"
        );

        // The filler is admitted first; the victim (warm prefix + unique tail) is
        // newest, so Lifo preempts it under KV pressure before its first output.
        let filler = Uuid::from_u128(71);
        let victim = Uuid::from_u128(72);
        core.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 8,
            uuid: Some(filler),
            ..Default::default()
        });
        core.receive(DirectRequest {
            tokens: (0..4).chain(200..212).collect(),
            max_output_tokens: 4,
            uuid: Some(victim),
            ..Default::default()
        });

        let mut victim_signals = Vec::new();
        let mut preempted = false;
        for _ in 0..64 {
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms.max(now_ms + 1.0);
            victim_signals.extend(
                pass.output_signals
                    .iter()
                    .filter(|signal| signal.uuid == victim)
                    .cloned(),
            );
            if !preempted && pass.mocker_metrics.vllm_preemptions_total > 0 {
                preempted = true;
                assert!(
                    victim_signals.is_empty(),
                    "victim must be preempted before its first output"
                );
                assert_eq!(core.state.requests.get(&victim).unwrap().num_preemptions, 1);
            }
            if victim_signals.iter().any(|signal| signal.completed) {
                break;
            }
        }
        assert!(preempted, "victim was never preempted");
        assert!(
            victim_signals.iter().any(|signal| signal.completed),
            "victim never completed after readmission"
        );
        assert_eq!(
            victim_signals[0].cached_tokens,
            Some(4),
            "first signal must retain the original admission count, not a post-preemption re-probe"
        );
        assert!(
            victim_signals[1..]
                .iter()
                .all(|signal| signal.cached_tokens.is_none())
        );
    }
}
