// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_kv_router::plugins::prelude::*;
use dynamo_router_policies::{default_registry, power_of_two_policy_with_rng, register};
use dynamo_scheduler::host::WorkerSelector;
use parking_lot::Mutex;
use support::*;

const PICKS: usize = 2000;

/// `count` workers of two ranks each, every rank at a distinct load, so every cost is distinct
/// and rank (0, 0) is the single cheapest.
fn load_only_fixture(count: usize) -> (HashMap<u64, TestWorker>, SchedulingRequest) {
    let (workers, mut request) = fixture(count, 64);
    for (worker, load) in &mut request.worker_loads {
        load.active_requests = 0;
        load.active_prefill_tokens = 0;
        load.active_decode_blocks = worker.worker_id as usize * 2 + worker.dp_rank as usize;
        load.additional_active_blocks = 0;
    }
    (workers, request)
}

fn load_only_config() -> KvRouterConfig {
    KvRouterConfig {
        overlap_score_credit: 0.0,
        ..Default::default()
    }
}

fn pick_counts(
    policy: &WorkerSelectionPolicy,
    workers: &HashMap<u64, TestWorker>,
    request: &SchedulingRequest,
) -> HashMap<WorkerWithDpRank, usize> {
    let mut counts = HashMap::new();
    for _ in 0..PICKS {
        let selected = policy
            .select_worker(selection_input(workers, request, 16))
            .unwrap();
        *counts.entry(selected.worker).or_default() += 1;
    }
    counts
}

fn seeded(choices: usize, seed: u64) -> WorkerSelectionPolicy {
    power_of_two_policy_with_rng(
        SelectionConfig::from(&load_only_config()),
        "decode",
        choices,
        Some(Arc::new(Mutex::new(fastrand::Rng::with_seed(seed)))),
    )
}

/// A policy file that names the type for decode only: decode picks spread over the workers,
/// while prefill keeps the default and takes the exact minimum every time.
#[test]
fn decode_only_selection_spreads_decode_and_leaves_prefill_on_the_exact_minimum() {
    let policy_file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        policy_file.path(),
        "worker_selection:\n  decode:\n    type: dynamo-power-of-two-cost-fn\n    parameters: {choices: 2}\n",
    )
    .unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(policy_file.path().display().to_string()),
        ..load_only_config()
    };
    let mut registry = default_registry();
    register(&mut registry).unwrap();
    let factory = registry.resolve(&config).unwrap().unwrap();
    let policy = |role| {
        factory(
            &SelectionConfig::from(&config),
            role,
            RoutingPartitionRef::new("model", "default"),
        )
    };
    let (workers, mut request) = load_only_fixture(8);
    let cheapest = WorkerWithDpRank::new(0, 0);
    let costliest = WorkerWithDpRank::new(7, 1);

    let prefill = pick_counts(&policy(WorkerType::Prefill), &workers, &request);
    assert_eq!(prefill, HashMap::from([(cheapest, PICKS)]));

    // The disaggregated decode stage routes without prompt-side load.
    request.track_prefill_tokens = false;
    let decode = pick_counts(&policy(WorkerType::Decode), &workers, &request);
    // The cheapest of 16 ranks wins only the picks that draw it: 2 / 16.
    let share = decode[&cheapest] as f64 / PICKS as f64;
    assert!((0.07..0.19).contains(&share), "cheapest share {share}");
    // The costliest rank loses to whatever it is drawn with; every other rank is reachable.
    assert!(!decode.contains_key(&costliest));
    assert_eq!(decode.len(), 15);
}

#[test]
fn fewer_candidates_than_choices_take_the_exact_minimum() {
    let (workers, mut request) = load_only_fixture(1);
    request.track_prefill_tokens = false;
    let counts = pick_counts(&seeded(4, 3), &workers, &request);
    assert_eq!(
        counts,
        HashMap::from([(WorkerWithDpRank::new(0, 0), PICKS)])
    );
}

#[test]
fn seeded_selection_is_reproducible() {
    let (workers, mut request) = load_only_fixture(8);
    request.track_prefill_tokens = false;
    let picks = |policy: WorkerSelectionPolicy| -> Vec<WorkerWithDpRank> {
        (0..64)
            .map(|_| {
                policy
                    .select_worker(selection_input(&workers, &request, 16))
                    .unwrap()
                    .worker
            })
            .collect()
    };
    let first = picks(seeded(2, 42));
    assert_eq!(first, picks(seeded(2, 42)));
    assert_ne!(first, picks(seeded(2, 43)));
}

/// A mandatory pin is returned as it is and draws nothing from the seeded stream.
#[test]
fn mandatory_pin_is_preserved_and_does_not_advance_the_seeded_stream() {
    let (workers, mut request) = load_only_fixture(8);
    request.track_prefill_tokens = false;
    let (pinning, reference) = (seeded(2, 42), seeded(2, 42));
    for pinned in [true, false, true, true, false, false] {
        request.pinned_worker = pinned.then_some(WorkerWithDpRank::new(3, 1));
        let selected = pinning
            .select_worker(selection_input(&workers, &request, 16))
            .unwrap()
            .worker;
        if pinned {
            assert_eq!(Some(selected), request.pinned_worker);
        } else {
            let expected = reference
                .select_worker(selection_input(&workers, &request, 16))
                .unwrap()
                .worker;
            assert_eq!(selected, expected);
        }
    }
}
