// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

use std::{cell::RefCell, time::Duration};

use dynamo_kv_router::plugins::prelude::*;
use dynamo_scheduler::host::{WorkerSelectionInput, WorkerSelector};
use dynamo_session_affinity::host::{
    AcquireStep, SessionAffinity, SessionAffinityConfig, SessionAffinityMode,
    subagent_group_affinity_id,
};
use dynamo_slot_tracker::host::{
    DecodeProgressUpdate, PrefillLoadHint, SchedulerBookingDescriptor, SequenceRequest,
};
use support::*;
use tokio::time::Instant;

fn book_prefill(
    slots: &NativeSlots,
    id: &str,
    worker: WorkerWithDpRank,
) -> SchedulerBookingDescriptor {
    let request_id = id.to_owned();
    let attempt_id = slots
        .add_request_if_registered_admitted(
            SequenceRequest {
                request_id: request_id.clone(),
                token_sequence: None,
                expected_output_tokens: None,
                track_prefill_tokens: true,
                prefill_load_hint: Some(PrefillLoadHint {
                    initial_effective_prefill_tokens: 16_385,
                    expected_prefill_duration: None,
                }),
                worker,
                lora_name: None,
            },
            Instant::now(),
        )
        .unwrap();
    SchedulerBookingDescriptor {
        request_id,
        worker,
        attempt_id,
    }
}

fn recorded_select(
    policy: &WorkerSelectionPolicy,
    slots: &NativeSlots,
    request: &SchedulingRequest,
) -> (WorkerWithDpRank, Vec<WorkerWithDpRank>) {
    let reads = RefCell::new(Vec::new());
    let provider = |worker| {
        reads.borrow_mut().push(worker);
        slots.decode_progress_for(worker)
    };
    let selected = select_with_provider(policy, &workers(), request, Some(&provider));
    (selected, reads.into_inner())
}

#[tokio::test(start_paused = true)]
async fn unbound_group_moves_and_keeps_its_accepted_dispatch_target() {
    let (slots, owners) = backgrounds(8, 128, 8);
    let policy = policy();
    let affinity = SessionAffinity::with_config(
        SessionAffinityConfig::new(Duration::from_secs(60)).with_mode(SessionAffinityMode::Soft),
    )
    .unwrap();
    let parent = "parent";
    let group = subagent_group_affinity_id(parent);

    let AcquireStep::Held(first_hold) = affinity.try_acquire(&group, None).unwrap() else {
        panic!("a new group should acquire immediately");
    };
    let mut first = request();
    first.mode = ScheduleMode::Tracked {
        request_id: "first-child".into(),
    };
    first.session_context = Some(SessionContext::new(
        "first-child".into(),
        Some(parent.into()),
        None,
        None,
    ));
    first.affinity_target = first_hold.target();
    assert_eq!(first.affinity_target, None);
    first.worker_loads = slots.project_worker_loads(None, Instant::now());
    let (selected, reads) = recorded_select(&policy, &slots, &first);
    assert_eq!((selected, reads), (ALTERNATIVE, vec![SOURCE]));

    // Model an accepted dispatch by booking the pick and committing its real affinity hold.
    // Keep the lease live while the first child's prefill remains booked.
    book_prefill(&slots, "first-child", selected);
    let _first_lease = affinity.commit(first_hold, selected.into()).unwrap();

    let finished = &owners[0];
    slots
        .free_if_booking(
            &finished.request_id,
            finished.worker,
            finished.attempt_id,
            Instant::now(),
        )
        .unwrap();
    tokio::time::advance(Duration::from_millis(101)).await;

    let AcquireStep::Held(second_hold) = affinity.try_acquire(&group, None).unwrap() else {
        panic!("the accepted dispatch should have bound the group");
    };
    let mut second = request();
    second.mode = ScheduleMode::Tracked {
        request_id: "second-child".into(),
    };
    second.session_context = Some(SessionContext::new(
        "second-child".into(),
        Some(parent.into()),
        None,
        None,
    ));
    second.affinity_target = second_hold.target();
    assert_eq!(second.affinity_target, Some(ALTERNATIVE.into()));
    second.worker_loads = slots.project_worker_loads(None, Instant::now());
    assert_eq!(second.worker_loads[&SOURCE].active_requests, 7);
    assert_eq!(second.worker_loads[&ALTERNATIVE].active_requests, 1);
    assert!(second.worker_loads[&ALTERNATIVE].active_prefill_tokens > 0);
    assert_eq!(
        recorded_select(&policy, &slots, &second),
        (ALTERNATIVE, Vec::new())
    );
}

#[tokio::test(start_paused = true)]
async fn first_move_uses_count_floor_and_strict_budget_threshold() {
    let policy = policy();
    for (count, limit, expected, expected_reads) in [
        (7, 128, SOURCE, 0),
        (8, 16, SOURCE, 1),
        (8, 40, SOURCE, 1),
        (8, 41, ALTERNATIVE, 1),
        (9, 128, ALTERNATIVE, 1),
    ] {
        let (slots, _) = backgrounds(count, limit, 8);
        let mut request = request();
        request.worker_loads = slots.project_worker_loads(None, Instant::now());
        assert_eq!(
            recorded_select(&policy, &slots, &request),
            (expected, vec![SOURCE; expected_reads]),
            "count={count}, limit={limit}",
        );
    }
}

#[tokio::test(start_paused = true)]
async fn rejected_hotspots_match_the_builtin_two_tier_fallback() {
    #[derive(Clone, Copy, Debug)]
    enum Case {
        NotSibling,
        MissingProgress,
        CountMismatch,
        Stale,
        PinnedAlternative,
        SourceOnly,
        LoadBalance,
        PartialPrefix,
        BusyAlternative,
        ShrunkSource,
        PartialAlternative,
        WarmAlternative,
        UnknownTarget,
    }

    let policy = policy();
    let mut registry = dynamo_router_policies::default_registry();
    dynamo_router_policies::register(&mut registry).unwrap();
    let fallback_factory = registry.resolve_type("dynamo-two-tier-cost-fn").unwrap();
    let fallback = fallback_factory(
        &SelectionConfig::default(),
        WorkerType::Aggregated,
        RoutingPartitionRef::new("test", "default"),
    );
    for (case, expected_reads) in [
        (Case::NotSibling, 0),
        (Case::MissingProgress, 1),
        (Case::CountMismatch, 1),
        (Case::Stale, 1),
        (Case::PinnedAlternative, 0),
        (Case::SourceOnly, 0),
        (Case::LoadBalance, 0),
        (Case::PartialPrefix, 0),
        (Case::BusyAlternative, 0),
        (Case::ShrunkSource, 0),
        (Case::PartialAlternative, 0),
        (Case::WarmAlternative, 0),
        (Case::UnknownTarget, 0),
    ] {
        let limit = if matches!(case, Case::PinnedAlternative) {
            16
        } else {
            128
        };
        let (slots, owners) = backgrounds(8, limit, 8);
        let mut request = request();
        match case {
            Case::NotSibling => request.session_context = None,
            Case::MissingProgress => {
                let owner = &owners[0];
                slots
                    .update_output_if_booking(
                        &owner.request_id,
                        owner.worker,
                        owner.attempt_id,
                        0,
                        None,
                        Some(DecodeProgressUpdate::unavailable()),
                    )
                    .unwrap();
            }
            Case::Stale => tokio::time::advance(Duration::from_millis(101)).await,
            Case::PinnedAlternative => request.pinned_worker = Some(ALTERNATIVE),
            Case::SourceOnly => {
                request.allowed_worker_ids = Some([SOURCE.worker_id].into_iter().collect());
            }
            Case::LoadBalance => {
                for index in 0..40 {
                    book_prefill(&slots, &format!("extra-{index}"), SOURCE);
                }
            }
            Case::PartialPrefix => {
                request
                    .overlap
                    .tier_overlap_blocks
                    .device
                    .insert(SOURCE, 128);
            }
            Case::BusyAlternative
            | Case::ShrunkSource
            | Case::PartialAlternative
            | Case::WarmAlternative => {
                book_prefill(&slots, "first-child", ALTERNATIVE);
                request.affinity_target = Some(ALTERNATIVE.into());
                match case {
                    Case::BusyAlternative => {
                        book_prefill(&slots, "unrelated", ALTERNATIVE);
                    }
                    Case::ShrunkSource => {
                        request
                            .overlap
                            .tier_overlap_blocks
                            .device
                            .insert(SOURCE, 192);
                    }
                    Case::PartialAlternative => {
                        request
                            .overlap
                            .tier_overlap_blocks
                            .device
                            .insert(ALTERNATIVE, 128);
                    }
                    Case::WarmAlternative => {
                        request
                            .overlap
                            .tier_overlap_blocks
                            .device
                            .insert(ALTERNATIVE, 256);
                    }
                    _ => unreachable!(),
                }
            }
            Case::UnknownTarget => {
                request.affinity_target = Some(WorkerWithDpRank::new(901, 0).into());
            }
            Case::CountMismatch => {}
        }
        request.worker_loads = slots.project_worker_loads(None, Instant::now());
        if matches!(case, Case::CountMismatch) {
            // Preserve the earlier load projection while the native set loses one owner.
            let owner = &owners[0];
            slots
                .free_if_booking(
                    &owner.request_id,
                    owner.worker,
                    owner.attempt_id,
                    Instant::now(),
                )
                .unwrap();
        }
        let expected = select_with_provider(&fallback, &workers(), &request, None);
        if matches!(
            case,
            Case::BusyAlternative | Case::ShrunkSource | Case::PartialAlternative
        ) {
            assert_eq!(
                expected, SOURCE,
                "{case:?} must distinguish fallback from retention"
            );
        }
        let (actual, reads) = recorded_select(&policy, &slots, &request);
        assert_eq!(actual, expected, "{case:?}");
        assert_eq!(reads, vec![SOURCE; expected_reads], "{case:?}");
    }
}

#[test]
fn rejects_unsupported_worker_role() {
    let unsupported = policy_for_role(WorkerType::Prefill);
    assert_eq!(unsupported.required_worker_inputs(), WorkerInputs::NONE);
    let workers = workers();
    let request = request();
    let result = unsupported.select_worker(WorkerSelectionInput::configured(
        &workers,
        &request,
        request.eligibility(),
        BLOCK_SIZE,
    ));
    assert!(matches!(result, Err(error) if error.to_string().contains("aggregated")));
}
