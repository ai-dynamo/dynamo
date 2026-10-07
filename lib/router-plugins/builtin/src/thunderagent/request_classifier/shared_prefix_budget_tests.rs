// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod shared_prefix_budget_tests {
    use super::*;

    const INPUT: usize = 16_384;
    const FULL_CHARGE: usize = INPUT + 100 + 256;

    fn state(enabled: bool) -> State {
        State::new(ThunderAgentConfig {
            shared_prefix_budget: enabled,
            ..Default::default()
        })
    }

    fn capacities(workers: &[WorkerWithDpRank], tokens: usize) -> WorkerCapacitySnapshot {
        WorkerCapacitySnapshot::new(workers.iter().map(|&worker| (worker, tokens)))
            .with_live_workers(workers.iter().copied())
    }

    fn hashes() -> Vec<u64> {
        (1..=256).collect()
    }

    fn main_request(
        request: &str,
        session: &str,
        worker: Option<WorkerWithDpRank>,
        hashes: Option<Vec<u64>>,
    ) -> RequestRegistration {
        RequestRegistration::new(
            request.into(),
            session.into(),
            INPUT,
            RequestProgress::new(INPUT).0,
            false,
        )
        .with_pinned_worker(worker)
        .with_sequence_hashes(hashes)
    }

    fn complete(
        state: &mut State,
        capacities: &WorkerCapacitySnapshot,
        request: &str,
        worker: WorkerWithDpRank,
        now: Instant,
    ) {
        state.on_event(
            ClassifyEvent::Sent {
                request_id: request.into(),
                worker,
            },
            capacities,
            now,
        );
        state.on_event(
            ClassifyEvent::Completed {
                request_id: request.into(),
                worker,
                context_tokens: Some(INPUT),
            },
            capacities,
            now,
        );
        assert!(!state.requests.contains_key(request));
    }

    #[test]
    fn pending_programs_are_full_cost_until_both_actual_sent_events() {
        let w0 = WorkerWithDpRank::new(7, 0);
        let capacities = capacities(&[w0], 64_000);
        let now = Instant::now();
        let mut state = state(true);
        for index in 0..2 {
            let id = format!("request-{index}");
            state
                .register(
                    main_request(&id, &format!("session-{index}"), Some(w0), Some(hashes())),
                    &capacities,
                    now,
                )
                .unwrap();
            assert_eq!(state.request_status(&id), WaitStatus::Released(Some(w0)));
        }
        assert_eq!(state.normal_usage[&w0], FULL_CHARGE * 2);
        state.on_event(
            ClassifyEvent::Sent {
                request_id: "request-0".into(),
                worker: w0,
            },
            &capacities,
            now,
        );
        assert_eq!(state.normal_usage[&w0], FULL_CHARGE * 2);
        state.on_event(
            ClassifyEvent::Sent {
                request_id: "request-1".into(),
                worker: w0,
            },
            &capacities,
            now,
        );
        assert_eq!(state.normal_usage[&w0], INPUT + 2 * (100 + 256));
        state.cancel_request("request-0", &capacities, now);
        assert_eq!(state.normal_usage[&w0], FULL_CHARGE);
        state.cancel_request("request-1", &capacities, now);
        assert!(state.normal_usage.is_empty());
    }

    #[test]
    fn confirmed_duplicate_prefix_frees_budget_for_a_full_cost_newcomer() {
        let w0 = WorkerWithDpRank::new(7, 0);
        let capacities = capacities(&[w0], 40_000);
        let now = Instant::now();
        for enabled in [false, true] {
            let mut state = state(enabled);
            for index in 0..2 {
                let id = format!("old-{index}");
                state
                    .register(
                        main_request(&id, &id, Some(w0), Some(hashes())),
                        &capacities,
                        now,
                    )
                    .unwrap();
                complete(&mut state, &capacities, &id, w0, now);
            }
            state
                .register(main_request("new", "new", Some(w0), None), &capacities, now)
                .unwrap();
            assert_eq!(
                state.request_status("new"),
                if enabled {
                    WaitStatus::Released(Some(w0))
                } else {
                    WaitStatus::Waiting
                }
            );
            if enabled {
                assert_eq!(
                    state.normal_usage[&w0],
                    INPUT + 2 * (100 + 256) + FULL_CHARGE
                );
            }
        }
    }

    #[test]
    fn equal_hashes_do_not_share_budget_across_workers_or_ranks() {
        let workers = [
            WorkerWithDpRank::new(7, 0),
            WorkerWithDpRank::new(8, 0),
            WorkerWithDpRank::new(7, 1),
        ];
        let capacities = capacities(&workers, 64_000);
        let now = Instant::now();
        let mut state = state(true);
        for (index, &worker) in workers.iter().enumerate() {
            let id = format!("session-{index}");
            state
                .register(
                    main_request(&id, &id, Some(worker), Some(hashes())),
                    &capacities,
                    now,
                )
                .unwrap();
            complete(&mut state, &capacities, &id, worker, now);
        }
        for worker in workers {
            assert_eq!(state.normal_usage[&worker], FULL_CHARGE);
        }
    }

    #[test]
    fn absent_or_different_hashes_receive_no_prefix_credit() {
        let w0 = WorkerWithDpRank::new(7, 0);
        let capacities = capacities(&[w0], 64_000);
        let now = Instant::now();
        for second in [None, Some((2..=257).collect())] {
            let mut state = state(true);
            for (index, chain) in [Some(hashes()), second].into_iter().enumerate() {
                let id = format!("session-{index}");
                state
                    .register(main_request(&id, &id, Some(w0), chain), &capacities, now)
                    .unwrap();
                complete(&mut state, &capacities, &id, w0, now);
            }
            assert_eq!(state.normal_usage[&w0], 2 * FULL_CHARGE);
        }
    }

    #[test]
    fn unpinned_main_or_continuation_is_rejected_before_budget_mutation() {
        let w0 = WorkerWithDpRank::new(7, 0);
        let capacities = capacities(&[w0], 64_000);
        let now = Instant::now();
        let mut state = state(true);
        assert!(matches!(
            state.register(
                main_request("missing", "missing", None, Some(hashes())),
                &capacities,
                now
            ),
            Err(ThunderAgentError::NonFinalRequiresHardPin)
        ));
        assert!(
            state.programs.is_empty() && state.requests.is_empty() && state.normal_usage.is_empty()
        );
        state
            .register(
                main_request("old", "old", Some(w0), Some(hashes())),
                &capacities,
                now,
            )
            .unwrap();
        complete(&mut state, &capacities, "old", w0, now);
        assert!(matches!(
            state.register(
                main_request("next", "old", None, Some(hashes())),
                &capacities,
                now
            ),
            Err(ThunderAgentError::NonFinalRequiresHardPin)
        ));
        assert_eq!(state.programs.len(), 1);
        assert!(state.requests.is_empty());
        assert_eq!(state.normal_usage[&w0], FULL_CHARGE);
    }

    #[test]
    fn existing_acting_weights_apply_once_to_confirmed_shared_prefixes() {
        let w0 = WorkerWithDpRank::new(7, 0);
        let capacities = capacities(&[w0], 64_000);
        let now = Instant::now();
        for weight in [0.5, 0.3, 1.0, 1.5] {
            let mut state = State::new(ThunderAgentConfig {
                shared_prefix_budget: true,
                acting_token_weight: weight,
                ..Default::default()
            });
            for index in 0..2 {
                let id = format!("acting-{index}");
                state
                    .register(
                        main_request(&id, &id, Some(w0), Some(hashes())),
                        &capacities,
                        now,
                    )
                    .unwrap();
                if index == 0 {
                    complete(&mut state, &capacities, &id, w0, now);
                } else {
                    state.on_event(
                        ClassifyEvent::Sent {
                            request_id: id.clone(),
                            worker: w0,
                        },
                        &capacities,
                        now,
                    );
                    assert_eq!(
                        state.normal_usage[&w0],
                        scale_tokens(INPUT, weight.max(1.0)) + 2 * (100 + 256),
                        "one reasoning and one acting member, weight {weight}"
                    );
                    state.on_event(
                        ClassifyEvent::Completed {
                            request_id: id,
                            worker: w0,
                            context_tokens: Some(INPUT),
                        },
                        &capacities,
                        now,
                    );
                }
            }
            assert_eq!(
                state.normal_usage[&w0],
                scale_tokens(INPUT, weight) + 2 * (100 + 256),
                "shared prefix with acting weight {weight}"
            );
        }
    }
    #[test]
    fn final_rechecks_live_target_progress_after_worker_loss() {
        let w0 = WorkerWithDpRank::new(7, 0);
        let w1 = WorkerWithDpRank::new(8, 0);
        let now = Instant::now();
        let both = capacities(&[w0, w1], 20_356);
        let mut state = state(true);
        state.register(main_request("owner", "owner", Some(w0), Some(hashes())), &both, now).unwrap();
        complete(&mut state, &both, "owner", w0, now);
        let (progress, updater) = RequestProgress::new(20_000);
        state.register(RequestRegistration::new("background".into(), "background".into(),
            20_000, progress, false).with_pinned_worker(Some(w1)), &both, now).unwrap();
        state.on_event(ClassifyEvent::Sent { request_id: "background".into(), worker: w1 }, &both, now);
        let lost = capacities(&[w1], 20_356);
        state.register(RequestRegistration::new("final".into(), "owner".into(),
            1, RequestProgress::new(1).0, true).with_pinned_worker(Some(w1)), &lost, now).unwrap();
        assert_eq!(state.request_status("final"), WaitStatus::Waiting);
        assert_eq!(state.programs["owner"].assigned_worker, None);
        // Model progress arriving after the event's initial accounting snapshot.
        updater.update_context_tokens(20_100);
        let grown = capacities(&[w1], 20_756);
        state.admit_request("final", &grown, now);
        assert_eq!(state.request_status("final"), WaitStatus::Waiting);
        assert_eq!(state.normal_usage[&w1], 20_456);
        assert!(state.programs.contains_key("owner"));
        assert!(state.final_reservations.is_empty());
    }

}
