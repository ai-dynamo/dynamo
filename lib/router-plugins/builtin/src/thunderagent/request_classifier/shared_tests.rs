// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn state() -> State {
    State::new(ThunderAgentConfig {
        shared_prefix_accounting: true,
        buffer_per_program: 16,
        ..Default::default()
    })
    .with_block_size(16)
}

fn worker(rank: u32) -> WorkerWithDpRank {
    WorkerWithDpRank::new(1, rank)
}

fn capacity(tokens: usize) -> WorkerCapacitySnapshot {
    WorkerCapacitySnapshot::new([(worker(0), tokens)])
}

fn request(id: &str, tokens: usize, hashes: &[u64]) -> RequestRegistration {
    RequestRegistration::new(
        id.into(),
        id.into(),
        tokens,
        RequestProgress::new(tokens).0,
        false,
    )
    .with_prefix_hashes(Arc::from(hashes))
}

fn released(state: &State, id: &str) -> bool {
    matches!(
        state.wait_status(id, &state.requests[id].notify),
        WaitStatus::Released(_)
    )
}

fn complete(state: &mut State, id: &str, tokens: usize, caps: &WorkerCapacitySnapshot) {
    state.on_event(
        ClassifyEvent::Completed {
            request_id: id.into(),
            worker: worker(0),
            context_tokens: Some(tokens),
        },
        caps,
        Instant::now(),
    );
}

#[test]
fn shared_prefix_charges_partial_tails_generation_and_each_growth_reserve() {
    let mut state = state();
    let caps = capacity(1_000);
    let hashes: Vec<_> = (0..44).collect();
    // 44 complete blocks (704 tokens), plus a private one-token tail.
    state
        .register(request("a", 705, &hashes), &caps, Instant::now())
        .unwrap();
    state
        .register(request("b", 705, &hashes), &caps, Instant::now())
        .unwrap();
    assert!(released(&state, "b"));
    assert_eq!(state.worker_used(worker(0)), 704 + 2 * (1 + 16));
    let (progress, updater) = RequestProgress::new(705);
    state.update_program("a", |p| p.request_progress = Some(progress));
    updater.update_context_tokens(805);
    state.reconcile(&caps, Instant::now());
    assert_eq!(state.worker_used(worker(0)), 838);
    complete(&mut state, "a", 805, &caps);
    state.remove_program("a");
    assert_eq!(state.worker_used(worker(0)), 721);
    state.remove_program("b");
    assert_eq!(state.worker_used(worker(0)), 0);
    assert_eq!(state.prefix_accounting.savings(worker(0)), 0);
}

#[test]
fn missing_hashes_and_partial_blocks_never_receive_credit() {
    let mut state = state();
    let caps = capacity(1_000);
    // Supplying an extra hash cannot discount the partial trailing block.
    state
        .register(request("a", 31, &[1, 2]), &caps, Instant::now())
        .unwrap();
    state
        .register(request("b", 31, &[1, 2]), &caps, Instant::now())
        .unwrap();
    state
        .register(request("c", 31, &[]), &caps, Instant::now())
        .unwrap();
    assert_eq!(state.worker_used(worker(0)), 3 * (31 + 16) - 16);
    state.config.shared_prefix_accounting = false;
    state
        .register(request("d", 31, &[1, 2]), &caps, Instant::now())
        .unwrap();
    assert!(state.programs["d"].prefix_hashes.is_empty());
}

#[test]
fn worker_rank_migration_and_abort_restore_prefix_ownership() {
    let mut state = state();
    let caps = WorkerCapacitySnapshot::new([(worker(0), 1_000), (worker(1), 1_000)]);
    for id in ["a", "b"] {
        state
            .register(
                request(id, 640, &(0..40).collect::<Vec<_>>()).with_pinned_worker(Some(worker(0))),
                &caps,
                Instant::now(),
            )
            .unwrap();
    }
    assert_eq!(state.worker_used(worker(0)), 672);
    state.on_event(
        ClassifyEvent::Sent {
            request_id: "b".into(),
            worker: worker(1),
        },
        &caps,
        Instant::now(),
    );
    assert_eq!(state.worker_used(worker(0)), 656);
    assert_eq!(state.worker_used(worker(1)), 656);
    complete(&mut state, "a", 640, &caps);
    let mut next = request("a-next", 320, &(100..120).collect::<Vec<_>>());
    next.session_id = "a".into();
    state.register(next, &caps, Instant::now()).unwrap();
    assert_eq!(state.worker_used(worker(0)), 336);
    state.cancel_request("a-next", &caps, Instant::now());
    assert_eq!(state.worker_used(worker(0)), 656);
    state.set_assignment("b", Some(worker(0)));
    assert_eq!(state.worker_used(worker(0)), 672);
    assert_eq!(state.worker_used(worker(1)), 0);
    let final_request = RequestRegistration::new(
        "final".into(),
        "a".into(),
        0,
        RequestProgress::new(0).0,
        true,
    );
    state
        .register(final_request, &caps, Instant::now())
        .unwrap();
    assert_eq!(state.worker_used(worker(0)), 656);
}

#[test]
fn one_reconcile_resumes_shared_waiters_without_overbooking() {
    let mut state = state();
    let small = capacity(100);
    let caps = capacity(1_000);
    let hashes: Vec<_> = (0..40).collect();
    for id in ["a", "b", "c", "d"] {
        state
            .register(request(id, 705, &hashes), &small, Instant::now())
            .unwrap();
    }
    state.reconcile(&caps, Instant::now());
    // 640 common + 81 private per request. Three exceed the 850 resume limit.
    assert_eq!(
        state
            .requests
            .values()
            .filter(|r| r.phase == RequestPhase::Released)
            .count(),
        2
    );
    assert_eq!(state.worker_used(worker(0)), 802);
    assert_eq!(state.paused_programs.len(), 2);
}

#[test]
fn idle_reclaim_counts_last_owner_and_preserves_inflight_and_oversized_guards() {
    let mut state = state();
    let caps = capacity(1_000);
    let hashes: Vec<_> = (0..40).collect();
    for id in ["a", "b"] {
        state
            .register(request(id, 705, &hashes), &caps, Instant::now())
            .unwrap();
        complete(&mut state, id, 705, &caps);
    }
    // The in-flight request remains protected while both idle shared owners can be reclaimed.
    state
        .register(request("running", 100, &[]), &caps, Instant::now())
        .unwrap();
    state
        .register(request("too-large", 1_100, &[]), &caps, Instant::now())
        .unwrap();
    state.reconcile(&caps, Instant::now());
    assert_eq!(state.programs["a"].lifecycle, ProgramLifecycle::Active);
    assert_eq!(state.programs["b"].lifecycle, ProgramLifecycle::Active);
    state
        .register(
            request("waiting", 700, &(100..140).collect::<Vec<_>>()),
            &caps,
            Instant::now(),
        )
        .unwrap();
    state.reconcile(&caps, Instant::now());
    assert!(released(&state, "waiting"));
    assert!(released(&state, "running"));
    assert!(!released(&state, "too-large"));
    assert_eq!(state.programs["a"].lifecycle, ProgramLifecycle::Paused);
    assert_eq!(state.programs["b"].lifecycle, ProgramLifecycle::Paused);
    assert_eq!(state.worker_used(worker(0)), 832);
}

#[test]
fn hard_pin_cannot_borrow_shared_credit_from_another_rank() {
    let mut state = state();
    let caps = WorkerCapacitySnapshot::new([(worker(0), 1_000), (worker(1), 500)]);
    let hashes: Vec<_> = (0..40).collect();
    state
        .register(
            request("a", 640, &hashes).with_pinned_worker(Some(worker(0))),
            &caps,
            Instant::now(),
        )
        .unwrap();
    state
        .register(
            request("b", 640, &hashes).with_pinned_worker(Some(worker(1))),
            &caps,
            Instant::now(),
        )
        .unwrap();
    state.reconcile(&caps, Instant::now());
    assert!(!released(&state, "b"));
    assert_eq!(state.worker_used(worker(1)), 0);
    assert_eq!(state.worker_used(worker(0)), 656);
}
