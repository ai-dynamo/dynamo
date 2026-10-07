// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod shared_prefix_final_contract_tests {
    use super::*;

    fn setup_old() -> (
        State,
        WorkerCapacitySnapshot,
        WorkerCapacitySnapshot,
        WorkerWithDpRank,
        WorkerWithDpRank,
        Instant,
    ) {
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
        (state, both, w1_only, w0, w1, now)
    }

    fn final_request(pin: Option<WorkerWithDpRank>) -> RequestRegistration {
        RequestRegistration::new(
            "final".into(),
            "old-session".into(),
            1,
            RequestProgress::new(1).0,
            true,
        )
        .with_pinned_worker(pin)
    }

    #[test]
    fn first_scope_final_contract_missing_pin_rejected() {
        let (mut state, _, w1_only, _, _, now) = setup_old();
        state.reconcile(&w1_only, now + Duration::from_secs(1));
        assert!(state.programs.contains_key("old-session"));
        assert!(
            state
                .register(final_request(None), &w1_only, now + Duration::from_secs(2))
                .is_err(),
            "proposed contract rejects an unpinned model-serving final"
        );
    }

    #[test]
    fn first_scope_final_contract_unknown_session_rejected() {
        let (_, both, _, _, w1, now) = setup_old();
        let mut state = State::new(ThunderAgentConfig {
            shared_prefix_budget: true,
            ..Default::default()
        });
        assert!(
            state.register(final_request(Some(w1)), &both, now).is_err(),
            "proposed contract rejects a final without an old program"
        );
    }

    #[test]
    fn first_scope_final_contract_after_loss_keeps_target_charge_until_terminal() {
        let (mut state, _, w1_only, _, w1, now) = setup_old();
        state.reconcile(&w1_only, now + Duration::from_secs(1));
        state
            .register(
                final_request(Some(w1)),
                &w1_only,
                now + Duration::from_secs(2),
            )
            .unwrap();
        assert_eq!(
            state.request_status("final"),
            WaitStatus::Released(Some(w1))
        );
        assert!(
            state.normal_usage.get(&w1).copied().unwrap_or(0) >= 357,
            "proposed contract holds the final charge on W1 at release"
        );
        assert!(state.on_event(
            ClassifyEvent::Sent {
                request_id: "final".into(),
                worker: w1,
            },
            &w1_only,
            now + Duration::from_secs(2)
        ));
        assert!(
            state.normal_usage.get(&w1).copied().unwrap_or(0) >= 357,
            "proposed contract keeps the charge through Sent"
        );
        assert!(state.on_event(
            ClassifyEvent::Completed {
                request_id: "final".into(),
                worker: w1,
                context_tokens: Some(1),
            },
            &w1_only,
            now + Duration::from_secs(2)
        ));
        assert!(
            state.programs.is_empty() && state.requests.is_empty() && state.normal_usage.is_empty()
        );
    }

    #[test]
    fn first_scope_final_contract_before_loss_keeps_target_charge() {
        let (mut state, both, _, _, w1, now) = setup_old();
        state
            .register(final_request(Some(w1)), &both, now + Duration::from_secs(1))
            .unwrap();
        assert_eq!(
            state.request_status("final"),
            WaitStatus::Released(Some(w1))
        );
        assert!(
            state.normal_usage.get(&w1).copied().unwrap_or(0) >= 357,
            "proposed contract charges a pre-loss final on its W1 target"
        );
    }

    #[test]
    fn first_scope_final_contract_full_target_does_not_release_final() {
        let (mut state, _, w1_only, _, w1, now) = setup_old();
        state.reconcile(&w1_only, now + Duration::from_secs(1));
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
        let result = state.register(
            final_request(Some(w1)),
            &w1_only,
            now + Duration::from_secs(3),
        );
        assert!(
            result.is_err() || state.request_status("final") != WaitStatus::Released(Some(w1)),
            "proposed contract must hold or reject a final when W1 has only300 free tokens"
        );
    }
}
