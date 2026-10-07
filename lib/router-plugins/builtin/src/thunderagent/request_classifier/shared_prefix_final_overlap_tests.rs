// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod shared_prefix_final_overlap_tests {
    use super::*;

    fn state() -> State {
        State::new(ThunderAgentConfig {
            shared_prefix_budget: true,
            ..Default::default()
        })
    }

    fn final_request(id: &str, worker: WorkerWithDpRank) -> RequestRegistration {
        RequestRegistration::new(
            id.into(),
            "session".into(),
            1,
            RequestProgress::new(1).0,
            true,
        )
        .with_pinned_worker(Some(worker))
    }

    #[test]
    fn b_duplicate_waiting_final_is_rejected() {
        let w0 = WorkerWithDpRank::new(7, 0);
        let w1 = WorkerWithDpRank::new(9, 0);
        let only_w0 = WorkerCapacitySnapshot::new([(w0, 64_000)]).with_live_workers([w0]);
        let now = Instant::now();
        let mut state = state();
        state
            .register(
                RequestRegistration::new(
                    "main".into(),
                    "session".into(),
                    1,
                    RequestProgress::new(1).0,
                    false,
                )
                .with_pinned_worker(Some(w0)),
                &only_w0,
                now,
            )
            .unwrap();
        assert!(state.on_event(
            ClassifyEvent::Sent {
                request_id: "main".into(),
                worker: w0,
            },
            &only_w0,
            now
        ));
        assert!(state.on_event(
            ClassifyEvent::Completed {
                request_id: "main".into(),
                worker: w0,
                context_tokens: Some(1),
            },
            &only_w0,
            now
        ));
        state
            .register(
                final_request("final-first", w1),
                &only_w0,
                now + Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(state.request_status("final-first"), WaitStatus::Waiting);
        assert!(
            state
                .register(
                    final_request("final-second", w1),
                    &only_w0,
                    now + Duration::from_secs(1)
                )
                .is_err(),
            "a second final must not queue behind a waiting first final"
        );
        assert_eq!(state.requests.len(), 1);
    }

    #[test]
    fn b_final_queued_behind_aborted_new_main_does_not_wait_forever() {
        let w0 = WorkerWithDpRank::new(7, 0);
        let capacity = WorkerCapacitySnapshot::new([(w0, 64_000)]).with_live_workers([w0]);
        let now = Instant::now();
        let mut state = state();
        state
            .register(
                RequestRegistration::new(
                    "main".into(),
                    "session".into(),
                    1,
                    RequestProgress::new(1).0,
                    false,
                )
                .with_pinned_worker(Some(w0)),
                &capacity,
                now,
            )
            .unwrap();
        assert_eq!(state.request_status("main"), WaitStatus::Released(Some(w0)));
        state
            .register(final_request("final", w0), &capacity, now)
            .unwrap();
        assert_eq!(state.request_status("final"), WaitStatus::Waiting);
        assert!(state.on_event(
            ClassifyEvent::Aborted {
                request_id: "main".into(),
                worker: None,
                error: None,
            },
            &capacity,
            now
        ));
        assert!(!state.programs.contains_key("session"));
        assert_eq!(
            state.request_status("final"),
            WaitStatus::Missing,
            "queued final must terminate when its old program disappears"
        );
        assert!(state.requests.is_empty() && state.normal_usage.is_empty());
    }
}
