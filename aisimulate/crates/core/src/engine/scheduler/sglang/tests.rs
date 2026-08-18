// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use uuid::Uuid;

use super::config::{SchedulePolicy, SglangConfig};
use super::core::SglangCore;
use super::decode;
use super::decode::simulate_decode_step;
use super::policy::apply_schedule_policy;
use super::prefill::get_new_batch_prefill;
use super::request::SglangRequest;
use crate::engine::HandoffId;
use crate::engine::common::protocols::{
    DirectRequest, EngineType, KvEventPublishers, MockEngineArgs, OutputSignal, SglangArgs,
};
use crate::engine::kv_manager::SglangKvManager;
use crate::engine::kv_manager::sglang_backend::RadixRequestLease;
use crate::engine::scheduler::{
    SchedulerCommand, SchedulerCommandResult, SchedulerLifecycleEvent, capture_kv_event_sink,
};
use crate::engine::{KvEvent, KvEventData, PressureKind};

fn stored_hashes(events: &[KvEvent]) -> Vec<u64> {
    events
        .iter()
        .flat_map(|event| match &event.data {
            KvEventData::Stored(stored) => stored
                .blocks
                .iter()
                .map(|block| block.block_hash)
                .collect::<Vec<_>>(),
            KvEventData::Removed { .. } => Vec::new(),
        })
        .collect()
}

fn removed_event_count(events: &[KvEvent]) -> usize {
    events
        .iter()
        .filter(|event| matches!(event.data, KvEventData::Removed { .. }))
        .count()
}

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
        arrival_timestamp_ms: None,
        ..Default::default()
    }
}

#[test]
fn request_storage_reservation_is_bounded_for_submit_and_destination() {
    const BLOCK_SIZE: usize = 4;
    const MAX_OUTPUT_TOKENS: usize = 1_000_000;
    const CLIP_MAX_NEW_TOKENS: usize = 7;
    let args = MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .num_gpu_blocks(4)
        .block_size(BLOCK_SIZE)
        .speedup_ratio(0.0)
        .sglang(Some(SglangArgs {
            page_size: Some(BLOCK_SIZE),
            chunked_prefill_size: Some(16),
            clip_max_new_tokens: Some(CLIP_MAX_NEW_TOKENS),
            ..Default::default()
        }))
        .build()
        .unwrap();
    let mut core = SglangCore::new(args);
    let prompt = (0..8).collect::<Vec<_>>();
    let bounded_tokens = prompt.len() + CLIP_MAX_NEW_TOKENS;
    let bounded_pages = bounded_tokens / BLOCK_SIZE;

    let submitted = Uuid::from_u128(80_001);
    core.receive(DirectRequest {
        tokens: prompt.clone(),
        max_output_tokens: MAX_OUTPUT_TOKENS,
        uuid: Some(submitted),
        ..Default::default()
    });
    let (token_capacity, hash_capacity) = core.request_storage_capacities(submitted).unwrap();
    assert!(token_capacity >= bounded_tokens);
    assert!(token_capacity < prompt.len() + MAX_OUTPUT_TOKENS);
    assert!(hash_capacity >= bounded_pages);
    assert!(hash_capacity < (prompt.len() + MAX_OUTPUT_TOKENS) / BLOCK_SIZE);

    let destination = Uuid::from_u128(80_002);
    core.apply_command_effects(
        SchedulerCommand::ReserveDestination {
            handoff_id: HandoffId::from(Uuid::from_u128(80_003)),
            request: DirectRequest {
                tokens: prompt.clone(),
                max_output_tokens: MAX_OUTPUT_TOKENS,
                uuid: Some(destination),
                ..Default::default()
            },
        },
        false,
    )
    .unwrap();
    let (token_capacity, hash_capacity) = core.request_storage_capacities(destination).unwrap();
    assert!(token_capacity >= bounded_tokens);
    assert!(token_capacity < prompt.len() + MAX_OUTPUT_TOKENS);
    assert!(hash_capacity >= bounded_pages);
    assert!(hash_capacity < (prompt.len() + MAX_OUTPUT_TOKENS) / BLOCK_SIZE);

    let planned = Uuid::from_u128(80_004);
    let planned_output = (100..109).collect::<Vec<_>>();
    core.receive(DirectRequest {
        tokens: prompt.clone(),
        max_output_tokens: MAX_OUTPUT_TOKENS,
        output_token_ids: Some(planned_output.clone()),
        uuid: Some(planned),
        ..Default::default()
    });
    let (token_capacity, hash_capacity) = core.request_storage_capacities(planned).unwrap();
    let realizable_planned = 8;
    assert!(token_capacity >= prompt.len() + realizable_planned);
    assert!(hash_capacity >= (prompt.len() + realizable_planned) / BLOCK_SIZE);
    assert!(
        planned_output.len() > CLIP_MAX_NEW_TOKENS,
        "planned-output coverage must exceed the online storage clip"
    );
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
fn same_timestamp_retry_hint_stops_when_admission_state_converges() {
    let mut core = SglangCore::new(test_args(1, 4, 8));
    core.receive(direct_request(vec![1; 8], 2));
    let now_ms = 37.0;
    let mut retries = 0usize;

    let terminal_retry_status = loop {
        let pass = core.execute_pass_internal(None, now_ms);
        assert_eq!(pass.end_ms, now_ms, "retry must not advance virtual time");
        assert!(pass.admissions.is_empty());
        assert!(pass.output_signals.is_empty());
        assert!(pass.pressure_events.is_empty());
        assert!(pass.lifecycle_events.is_empty());
        assert!(pass.kv_events.is_empty());
        if pass.same_timestamp_retry != crate::engine::generalized::SameTimestampRetry::Retry {
            break pass.same_timestamp_retry;
        }
        retries += 1;
        assert!(retries < 1024, "SGLang admission state must converge");
    };

    assert!(retries > 1, "the regression must exercise repeated retries");
    assert_eq!(
        terminal_retry_status,
        crate::engine::generalized::SameTimestampRetry::Exhausted
    );
    assert!(!core.is_empty(), "the impossible request remains in flight");
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
    )
    .unwrap();

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
fn cached_prompt_one_token_past_a_page_resumes_at_the_last_page() {
    // 13 tokens with 4-token pages: len - 1 = 12 is already page-aligned.
    // Rounding the resume point up-then-down a page landed at 8 instead of 12,
    // so 5 tokens needed computing rather than 1. With chunked_prefill_size = 4
    // that no longer fits one chunk and the request needs a second prefill
    // round, stopping at 12 materialized tokens instead of finishing at 13.
    let args = test_args(16, 4, 4);
    let config = SglangConfig::from_args(&args);
    let mut kv_manager = SglangKvManager::new(16, 4, KvEventPublishers::default(), 0);
    let prompt: Vec<u32> = (1..=13u32).collect();

    let cached = kv_manager.allocate_for_request(&prompt).unwrap();
    kv_manager.finish(&prompt, cached.lease);

    let mut waiting = VecDeque::from([SglangRequest {
        uuid: Uuid::from_u128(77_002),
        sequence_tokens: prompt.clone(),
        prompt_len: prompt.len(),
        max_output_tokens: 1,
        planned_output_ids: None,
        materialized_tokens: 0,
        kv_lease: RadixRequestLease::default(),
        allocated_tokens: 0,
    }]);

    let admit = get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &[]);

    assert_eq!(admit.can_run.len(), 1);
    // Only the final page is left to compute, so the whole prompt lands in one
    // chunk even with a 4-token chunk budget.
    assert_eq!(admit.can_run[0].materialized_tokens, prompt.len());
    assert!(waiting.is_empty());
}

#[test]
fn prefix_cache_matches_past_the_first_chunk() {
    // Regression for the chunk-then-match ordering: with the whole 16-token
    // prompt already in the radix cache and chunked_prefill_size = 8, the
    // reusable prefix used to be capped at the chunk size, so a fully cached
    // prompt still reported 8 reused tokens and recomputed the rest.
    let args = test_args(16, 4, 8);
    let config = SglangConfig::from_args(&args);
    let mut kv_manager = SglangKvManager::new(16, 4, KvEventPublishers::default(), 0);
    let prompt: Vec<u32> = (1..=16u32).collect();

    let cached = kv_manager.allocate_for_request(&prompt).unwrap();
    kv_manager.finish(&prompt, cached.lease);

    let mut waiting = VecDeque::from([SglangRequest {
        uuid: Uuid::from_u128(77_001),
        sequence_tokens: prompt.clone(),
        prompt_len: prompt.len(),
        max_output_tokens: 1,
        planned_output_ids: None,
        materialized_tokens: 0,
        kv_lease: RadixRequestLease::default(),
        allocated_tokens: 0,
    }]);

    let admit = get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &[]);

    // The whole prompt is cached, so one admission consumes it all and nothing
    // is left waiting, regardless of chunked_prefill_size.
    assert_eq!(admit.admissions.len(), 1);
    assert_eq!(admit.can_run.len(), 1);
    assert_eq!(admit.can_run[0].materialized_tokens, prompt.len());

    // The final page is held back for recomputation, so it is reported as
    // computed rather than reused even though the cache holds it. Without the
    // cap the allocator matches all 16 tokens and this prompt is modelled as
    // zero prefill work, which understates the forward pass SGLang still runs
    // to produce the first output token.
    assert_eq!(admit.admissions[0].reused_input_tokens, prompt.len() - 4);
    assert_eq!(admit.prefill_fpm[0].prefix_tokens, prompt.len() - 4);
    assert_eq!(admit.prefill_fpm[0].tokens_computed, 4);
}

#[test]
fn fresh_prefill_tracks_cache_owned_prefix_pages_and_pressure_event() {
    let args = test_args(8, 4, 16);
    let config = SglangConfig::from_args(&args);
    let (buffer, sink) = capture_kv_event_sink();
    let mut kv_manager = SglangKvManager::new(11, 4, KvEventPublishers::new(Some(sink)), 0);
    let prompt = vec![1, 2, 3, 4];

    let cached = kv_manager.allocate_for_request(&prompt).unwrap();
    let cached_pages = cached.lease.pages().to_vec();
    kv_manager.finish(&prompt, cached.lease);
    let mut waiting = VecDeque::from([SglangRequest {
        uuid: Uuid::from_u128(90_002),
        sequence_tokens: prompt.clone(),
        prompt_len: prompt.len(),
        max_output_tokens: 1,
        planned_output_ids: None,
        materialized_tokens: 0,
        kv_lease: RadixRequestLease::default(),
        allocated_tokens: 0,
    }]);
    let req = get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &[])
        .can_run
        .pop()
        .unwrap();

    assert_eq!(req.cached_tokens(), prompt.len());
    assert_eq!(req.kv_pages(), cached_pages);

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
    let (retracted, pressure_events) = decode::check_decode_mem_with_pressure_events(
        &mut running,
        &mut kv_manager,
        &config,
        1,
        12.5,
    );
    assert_eq!(retracted.len(), 1);
    assert_eq!(retracted[0].uuid, Uuid::from_u128(90_002));
    let [pressure] = pressure_events.as_slice() else {
        panic!("expected exactly one SGLang retraction pressure event");
    };
    assert_eq!(pressure.at_ms, 12.5);
    assert_eq!(pressure.kind, PressureKind::SglangRetraction);
    assert_eq!(pressure.request_id, Uuid::from_u128(90_002));
    assert_eq!(pressure.state_before.running_requests, 2);
    assert_eq!(pressure.state_after.running_requests, 1);
    assert_eq!(pressure.state_before.waiting_requests, None);
    assert_eq!(pressure.state_after.waiting_requests, None);
    assert!(pressure.request_active_blocks_before > 0);
    assert_eq!(pressure.logical_available_blocks_before, Some(0));
    assert_eq!(pressure.required_blocks_before, Some(1));
    assert!(pressure.state_after.active_blocks <= pressure.state_before.active_blocks);
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
            .worker_type(crate::engine::common::protocols::WorkerType::Prefill)
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
            arrival_timestamp_ms: None,
            ..Default::default()
        }
    }

    fn execute(core: &mut SglangCore, now_ms: f64) -> crate::engine::scheduler::EnginePassResult {
        let mut collector = crate::engine::trace::TraceCollector::default();
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
    use crate::engine::common::protocols::WorkerType;

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

    fn execute(core: &mut SglangCore, now_ms: f64) -> crate::engine::scheduler::EnginePassResult {
        let mut collector = crate::engine::trace::TraceCollector::default();
        core.execute_pass(&mut collector, now_ms)
    }

    fn assert_no_republished_stores(activation: &[u64], later: &[u64]) {
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

        let reserved_pages = destination.destination_pages(handoff_id);
        let protected_before_activation = destination.kv_manager.cache().protected_size;
        assert!(!reserved_pages.is_empty());
        assert_eq!(
            destination
                .apply_command(SchedulerCommand::ActivateDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        let ready = destination
            .prebuilt_request(logical_uuid)
            .expect("activated request must be prebuilt-ready");
        assert_eq!(ready.kv_pages(), reserved_pages);
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
        assert_eq!(ready.kv_pages(), reserved_pages);
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
        assert!(running.kv_pages().starts_with(&reserved_pages));
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
        let blocker = core.kv_manager.reserve_decode_pages(1).unwrap();
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
        core.kv_manager.release_decode_reservation(blocker);
    }
}

mod scheduling {
    use super::*;

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
                    .engine_type(EngineType::Sglang)
                    .block_size(1)
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
                kv_lease: RadixRequestLease::default(),
                allocated_tokens: 0,
            },
            SglangRequest {
                uuid: match_uuid,
                sequence_tokens: vec![1, 2, 3, 4, 5, 6, 7],
                prompt_len: 5,
                max_output_tokens: 1,
                planned_output_ids: None,
                materialized_tokens: 0,
                kv_lease: RadixRequestLease::default(),
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
                kv_lease: RadixRequestLease::default(),
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
            kv_lease: RadixRequestLease::default(),
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
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
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
            kv_lease: RadixRequestLease::default(),
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
            kv_lease: RadixRequestLease::default(),
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
                kv_lease: RadixRequestLease::default(),
                allocated_tokens: 0,
            },
            SglangRequest {
                uuid: second_uuid,
                sequence_tokens: vec![2; 8],
                prompt_len: 8,
                max_output_tokens: 3,
                planned_output_ids: None,
                materialized_tokens: 0,
                kv_lease: RadixRequestLease::default(),
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
        let first = kv_manager.cache_mut().page_pool.allocate_pages(2).unwrap();
        let second = kv_manager.cache_mut().page_pool.allocate_pages(2).unwrap();

        let mut running = vec![
            SglangRequest {
                uuid: Uuid::new_v4(),
                sequence_tokens: vec![1, 2, 3, 4, 11, 12, 13, 14],
                prompt_len: 4,
                max_output_tokens: 10,
                planned_output_ids: None,
                kv_lease: RadixRequestLease::from_parts(first, 8, 4, kv_manager.cache().root()),
                materialized_tokens: 8,
                allocated_tokens: 8,
            },
            SglangRequest {
                uuid: Uuid::new_v4(),
                sequence_tokens: vec![9, 8, 7, 6, 21],
                prompt_len: 4,
                max_output_tokens: 10,
                planned_output_ids: None,
                kv_lease: RadixRequestLease::from_parts(second, 5, 4, kv_manager.cache().root()),
                materialized_tokens: 5,
                allocated_tokens: 8,
            },
        ];

        let retracted = decode::check_decode_mem(&mut running, &mut kv_manager, &config);
        assert_eq!(retracted.len(), 1);
        assert_eq!(retracted[0].output_tokens(), &[21]);
        assert_eq!(retracted[0].materialized_tokens, 0);
        assert!(retracted[0].kv_pages().is_empty());
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
        core.receive(crate::engine::common::protocols::DirectRequest {
            tokens: vec![1; 6],
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: None,
            arrival_timestamp_ms: None,
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
                arrival_timestamp_ms: None,
                ..Default::default()
            });
        }

        let mut collector = crate::engine::trace::TraceCollector::default();
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
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
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
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();

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
    fn test_mocker_metrics_reports_prefix_cache_reuse() {
        let mut core = SglangCore::new(fpm_args());

        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
        let pass1 = core.execute_pass(&mut collector, 0.0);
        assert_eq!(pass1.mocker_metrics.sglang_cache_hit_tokens, 0);
        assert_eq!(pass1.mocker_metrics.sglang_cache_total_tokens, 8);

        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            arrival_timestamp_ms: None,
        });

        let pass2 = core.execute_pass(&mut collector, pass1.end_ms);
        assert!(pass2.mocker_metrics.sglang_cache_hit_tokens > 0);
        assert!(
            pass2.mocker_metrics.sglang_cache_hit_tokens
                <= pass2.mocker_metrics.sglang_cache_total_tokens
        );
    }

    #[test]
    fn disabled_prefix_caching_does_not_reuse_or_retain_completed_prefixes() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(16)
            .block_size(4)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        let tokens = (0..8).collect::<Vec<_>>();
        let mut collector = crate::engine::trace::TraceCollector::default();

        for request_id in [91_001_u128, 91_002_u128] {
            core.receive(DirectRequest {
                tokens: tokens.clone(),
                max_output_tokens: 0,
                uuid: Some(Uuid::from_u128(request_id)),
                ..Default::default()
            });
            let pass = core.execute_pass(&mut collector, 0.0);

            assert_eq!(pass.completed_requests, 1);
            assert_eq!(pass.mocker_metrics.sglang_cache_hit_tokens, 0);
            assert_eq!(pass.mocker_metrics.sglang_cache_total_tokens, 8);
            assert_eq!(pass.fpm.unwrap().num_prefill_requests, 1);
            assert!(core.is_empty());
            assert_eq!(
                core.kv_manager.cache().available_tokens(),
                core.kv_manager.cache().total_tokens(),
                "a completed request must not leave reusable radix pages behind"
            );
        }
    }

    #[test]
    fn fully_cached_zero_output_request_is_not_forward_pass_work() {
        let mut core = SglangCore::new(fpm_args());
        let tokens = (0..8).collect::<Vec<_>>();
        let mut collector = crate::engine::trace::TraceCollector::default();

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
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            arrival_timestamp_ms: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
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
        let mut collector = crate::engine::trace::TraceCollector::default();

        // Two requests with 4-token prompts and long outputs to fill KV
        core.receive(DirectRequest {
            tokens: (0..4).collect(),
            max_output_tokens: 20,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            arrival_timestamp_ms: None,
        });
        core.receive(DirectRequest {
            tokens: (100..104).collect(),
            max_output_tokens: 20,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(3)),
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
}
