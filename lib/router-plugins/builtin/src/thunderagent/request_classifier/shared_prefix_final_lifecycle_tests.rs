// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod shared_prefix_final_lifecycle_tests {
    use super::*;

    fn owned_after_loss() -> (State, WorkerCapacitySnapshot, WorkerWithDpRank, Instant) {
        let w0 = WorkerWithDpRank::new(7, 0);
        let w1 = WorkerWithDpRank::new(9, 0);
        let both =
            WorkerCapacitySnapshot::new([(w0, 64_000), (w1, 64_000)]).with_live_workers([w0, w1]);
        let w1_only = WorkerCapacitySnapshot::new([(w1, 64_000)]).with_live_workers([w1]);
        let now = Instant::now();
        let mut state = State::new(ThunderAgentConfig {
            shared_prefix_budget: true,
            ..Default::default()
        });
        state
            .register(
                RequestRegistration::new(
                    "old-main".into(),
                    "old-session".into(),
                    1,
                    RequestProgress::new(1).0,
                    false,
                )
                .with_pinned_worker(Some(w0)),
                &both,
                now,
            )
            .unwrap();
        assert_eq!(
            state.request_status("old-main"),
            WaitStatus::Released(Some(w0))
        );
        assert!(state.on_event(
            ClassifyEvent::Sent {
                request_id: "old-main".into(),
                worker: w0,
            },
            &both,
            now
        ));
        assert!(state.on_event(
            ClassifyEvent::Completed {
                request_id: "old-main".into(),
                worker: w0,
                context_tokens: Some(1),
            },
            &both,
            now
        ));
        state.reconcile(&w1_only, now + Duration::from_secs(1));
        assert_eq!(state.programs["old-session"].assigned_worker, None);
        (state, w1_only, w1, now)
    }

    fn final_request(id: &str, session: &str, pin: WorkerWithDpRank) -> RequestRegistration {
        RequestRegistration::new(
            id.into(),
            session.into(),
            1,
            RequestProgress::new(1).0,
            true,
        )
        .with_pinned_worker(Some(pin))
    }

    #[test]
    fn b_final_cancel_after_release_frees_reservation_once() {
        let (mut state, w1_only, w1, now) = owned_after_loss();
        state
            .register(
                final_request("final", "old-session", w1),
                &w1_only,
                now + Duration::from_secs(2),
            )
            .unwrap();
        assert_eq!(
            state.request_status("final"),
            WaitStatus::Released(Some(w1))
        );
        assert_eq!(state.normal_usage[&w1], 357);
        assert!(state.cancel_request("final", &w1_only, now + Duration::from_secs(3)));
        assert!(!state.cancel_request("final", &w1_only, now + Duration::from_secs(4)));
        assert!(
            state.programs.is_empty()
                && state.requests.is_empty()
                && state.normal_usage.is_empty()
                && state.final_reservations.is_empty()
        );
    }

    #[test]
    fn b_final_waits_for_pinned_headroom_then_releases() {
        let (mut state, w1_only, w1, now) = owned_after_loss();
        for (index, input_tokens) in [16_385, 16_385, 16_385, 13_121].iter().enumerate() {
            let request_id = format!("background-main-{index}");
            state
                .register(
                    RequestRegistration::new(
                        request_id.clone(),
                        format!("background-{index}"),
                        *input_tokens,
                        RequestProgress::new(*input_tokens).0,
                        false,
                    )
                    .with_pinned_worker(Some(w1)),
                    &w1_only,
                    now + Duration::from_secs(2),
                )
                .unwrap();
            assert_eq!(
                state.request_status(&request_id),
                WaitStatus::Released(Some(w1))
            );
            assert!(state.on_event(
                ClassifyEvent::Sent {
                    request_id: request_id.clone(),
                    worker: w1,
                },
                &w1_only,
                now + Duration::from_secs(2)
            ));
            assert!(state.on_event(
                ClassifyEvent::Completed {
                    request_id,
                    worker: w1,
                    context_tokens: Some(*input_tokens),
                },
                &w1_only,
                now + Duration::from_secs(2)
            ));
        }
        assert_eq!(state.normal_usage[&w1], 63_700);
        state
            .register(
                final_request("old-final", "old-session", w1),
                &w1_only,
                now + Duration::from_secs(3),
            )
            .unwrap();
        assert_eq!(state.request_status("old-final"), WaitStatus::Waiting);
        assert!(state.programs.contains_key("old-session"));
        assert!(state.final_reservations.is_empty());

        state
            .register(
                final_request("free-final", "background-0", w1),
                &w1_only,
                now + Duration::from_secs(3),
            )
            .unwrap();
        assert_eq!(
            state.request_status("free-final"),
            WaitStatus::Released(Some(w1))
        );
        assert!(state.on_event(
            ClassifyEvent::Sent {
                request_id: "free-final".into(),
                worker: w1,
            },
            &w1_only,
            now + Duration::from_secs(3)
        ));
        assert!(state.on_event(
            ClassifyEvent::Completed {
                request_id: "free-final".into(),
                worker: w1,
                context_tokens: Some(1),
            },
            &w1_only,
            now + Duration::from_secs(3)
        ));
        assert_eq!(state.normal_usage[&w1], 46_959);
        state.reconcile(&w1_only, now + Duration::from_secs(4));
        assert_eq!(
            state.request_status("old-final"),
            WaitStatus::Released(Some(w1))
        );
        assert_eq!(state.normal_usage[&w1], 47_316);
        assert!(state.on_event(
            ClassifyEvent::Sent {
                request_id: "old-final".into(),
                worker: w1,
            },
            &w1_only,
            now + Duration::from_secs(4)
        ));
        assert!(state.on_event(
            ClassifyEvent::Completed {
                request_id: "old-final".into(),
                worker: w1,
                context_tokens: Some(1),
            },
            &w1_only,
            now + Duration::from_secs(4)
        ));
        assert_eq!(state.normal_usage[&w1], 46_959);
        assert!(state.final_reservations.is_empty());
    }

    #[test]
    fn b_final_worker_loss_then_abort_clears_held_charge() {
        let (mut state, w1_only, w1, now) = owned_after_loss();
        state
            .register(
                final_request("final", "old-session", w1),
                &w1_only,
                now + Duration::from_secs(2),
            )
            .unwrap();
        assert!(state.on_event(
            ClassifyEvent::Sent {
                request_id: "final".into(),
                worker: w1,
            },
            &w1_only,
            now + Duration::from_secs(2)
        ));
        let no_live = WorkerCapacitySnapshot::new([(w1, 64_000)]).with_live_workers([]);
        state.reconcile(&no_live, now + Duration::from_secs(3));
        assert_eq!(state.normal_usage[&w1], 357);
        assert!(state.on_event(
            ClassifyEvent::Aborted {
                request_id: "final".into(),
                worker: Some(w1),
                error: None,
            },
            &no_live,
            now + Duration::from_secs(3)
        ));
        assert!(
            state.programs.is_empty()
                && state.requests.is_empty()
                && state.normal_usage.is_empty()
                && state.final_reservations.is_empty()
        );
    }

    #[test]
    fn b_final_long_input_rejected_before_request_registration() {
        let (mut state, w1_only, w1, now) = owned_after_loss();
        assert!(
            state
                .register(
                    RequestRegistration::new(
                        "long-final".into(),
                        "old-session".into(),
                        2,
                        RequestProgress::new(2).0,
                        true,
                    )
                    .with_pinned_worker(Some(w1)),
                    &w1_only,
                    now + Duration::from_secs(2),
                )
                .is_err()
        );
        assert!(!state.requests.contains_key("long-final"));
        assert!(state.programs.contains_key("old-session"));
        assert!(state.final_reservations.is_empty());
    }
}
