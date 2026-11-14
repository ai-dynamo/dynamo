// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[allow(dead_code)]
mod support;

use std::{cell::RefCell, time::Duration};

use dynamo_kv_router::plugins::prelude::*;
use dynamo_scheduler::host::{WorkerSelectionInput, WorkerSelector};
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
    supply_progress: bool,
) -> (WorkerWithDpRank, Vec<WorkerWithDpRank>) {
    let reads = RefCell::new(Vec::new());
    let provider = |worker| {
        reads.borrow_mut().push(worker);
        slots.decode_progress_for(worker)
    };
    let selected = select_with_provider(
        policy,
        &workers(),
        request,
        supply_progress.then_some(&provider),
    );
    (selected, reads.into_inner())
}

#[tokio::test(start_paused = true)]
async fn long_decode_moves_then_keeps_the_dispatched_target() {
    let (slots, owners) = backgrounds(128, 8);
    let policy = policy();
    let mut request = request();
    request.worker_loads = slots.project_worker_loads(None, Instant::now());
    assert_eq!(
        recorded_select(&policy, &slots, &request, true),
        (ALTERNATIVE, vec![SOURCE])
    );

    let child = book_prefill(&slots, "first-child", ALTERNATIVE);
    for owner in &owners {
        slots
            .update_output_if_booking(
                &owner.request_id,
                owner.worker,
                owner.attempt_id,
                0,
                None,
                Some(DecodeProgressUpdate::observed(128, 96)),
            )
            .unwrap();
    }
    request.affinity_target = Some(ALTERNATIVE.into());
    request.worker_loads = slots.project_worker_loads(None, Instant::now());
    assert_eq!(
        recorded_select(&policy, &slots, &request, true),
        (ALTERNATIVE, vec![SOURCE])
    );

    slots
        .mark_prefill_completed_if_booking(
            &child.request_id,
            child.worker,
            child.attempt_id,
            Instant::now(),
        )
        .unwrap();
    request
        .overlap
        .tier_overlap_blocks
        .device
        .insert(ALTERNATIVE, 256);
    request.worker_loads = slots.project_worker_loads(None, Instant::now());
    assert_eq!(
        recorded_select(&policy, &slots, &request, true),
        (ALTERNATIVE, Vec::new())
    );
}

#[tokio::test(start_paused = true)]
async fn short_decode_and_equal_remaining_budget_keep_the_warm_source() {
    let policy = policy();
    for limit in [16, 40] {
        let (slots, _) = backgrounds(limit, 8);
        assert_eq!(select(&policy, &slots, &mut request()), SOURCE, "{limit}");
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
        (Case::MissingProgress, 0),
        (Case::CountMismatch, 1),
        (Case::Stale, 1),
        (Case::PinnedAlternative, 0),
        (Case::SourceOnly, 0),
        (Case::LoadBalance, 0),
        (Case::PartialPrefix, 0),
    ] {
        let limit = if matches!(case, Case::PinnedAlternative) {
            16
        } else {
            128
        };
        let (slots, owners) = backgrounds(limit, 8);
        let mut request = request();
        match case {
            Case::NotSibling => request.session_context = None,
            Case::MissingProgress => {
                book_prefill(&slots, "first-child", ALTERNATIVE);
                request.affinity_target = Some(ALTERNATIVE.into());
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
        let (actual, reads) = recorded_select(
            &policy,
            &slots,
            &request,
            !matches!(case, Case::MissingProgress),
        );
        assert_eq!(actual, expected, "{case:?}");
        assert_eq!(reads, vec![SOURCE; expected_reads], "{case:?}");
    }

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
