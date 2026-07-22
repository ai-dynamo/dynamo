// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn handoff_id() -> HandoffId {
    HandoffId::from(Uuid::from_u128(1))
}

fn one(actions: Vec<IssuedHandoffAction>) -> IssuedHandoffAction {
    assert_eq!(actions.len(), 1);
    actions.into_iter().next().unwrap()
}

fn acknowledge(
    coordinator: &mut HandoffCoordinatorCore,
    action: IssuedHandoffAction,
    outcome: HandoffActionOutcome,
) -> Vec<IssuedHandoffAction> {
    coordinator.on_action_outcome(action.id, outcome).unwrap()
}

fn transfer_timing(delay_ms: Option<f64>) -> HandoffTransferTiming {
    HandoffTransferTiming {
        mode: KvTransferTimingMode::FullPrompt,
        full_prompt_tokens: 1,
        kv_bytes_per_token: delay_ms.map(|delay_ms| (delay_ms * 1_000_000.0) as usize),
        bandwidth_gb_s: delay_ms.map(|_| 1.0),
    }
}

fn destination_reserved(handoff_id: HandoffId) -> HandoffFact {
    HandoffFact::DestinationReserved {
        handoff_id,
        transferable_prompt_tokens: 1,
    }
}

fn drive_success(order: HandoffOrder) {
    let id = handoff_id();
    let mut coordinator = HandoffCoordinatorCore::new(id, order);
    let first = one(coordinator.start().unwrap());

    let reserve = match order {
        HandoffOrder::SourceFirst => {
            assert!(matches!(first.action, HandoffAction::SubmitPrefill { .. }));
            assert!(
                acknowledge(&mut coordinator, first, HandoffActionOutcome::Submitted).is_empty()
            );
            one(coordinator
                .on_fact(HandoffFact::SourceHeld {
                    handoff_id: id,
                    transfer_timing: transfer_timing(Some(2.5)),
                })
                .unwrap())
        }
        HandoffOrder::DestinationFirst => {
            assert!(matches!(
                first.action,
                HandoffAction::ReserveDestination { .. }
            ));
            first
        }
    };
    assert!(acknowledge(&mut coordinator, reserve, HandoffActionOutcome::Accepted).is_empty());
    let after_reserved = coordinator.on_fact(destination_reserved(id)).unwrap();

    let transfer = match order {
        HandoffOrder::SourceFirst => one(after_reserved),
        HandoffOrder::DestinationFirst => {
            let submit = one(after_reserved);
            assert!(matches!(submit.action, HandoffAction::SubmitPrefill { .. }));
            assert!(
                acknowledge(&mut coordinator, submit, HandoffActionOutcome::Submitted).is_empty()
            );
            one(coordinator
                .on_fact(HandoffFact::SourceHeld {
                    handoff_id: id,
                    transfer_timing: transfer_timing(Some(2.5)),
                })
                .unwrap())
        }
    };
    assert!(matches!(
        transfer.action,
        HandoffAction::StartTransfer { delay_ms: 2.5, .. }
    ));
    assert!(acknowledge(&mut coordinator, transfer, HandoffActionOutcome::Scheduled).is_empty());
    let activate = one(coordinator
        .on_fact(HandoffFact::TransferCompleted { handoff_id: id })
        .unwrap());
    let release = one(acknowledge(
        &mut coordinator,
        activate,
        HandoffActionOutcome::Applied,
    ));
    let complete = one(acknowledge(
        &mut coordinator,
        release,
        HandoffActionOutcome::Applied,
    ));
    assert!(matches!(complete.action, HandoffAction::Complete { .. }));
    assert!(coordinator.is_complete());
    assert!(
        coordinator
            .on_action_outcome(release.id, HandoffActionOutcome::Applied)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn backend_orders_reach_the_same_safe_completion_sequence() {
    drive_success(HandoffOrder::SourceFirst);
    drive_success(HandoffOrder::DestinationFirst);
}

#[test]
fn timing_mode_selects_full_or_destination_missing_footprint() {
    for (mode, expected_delay_ms) in [
        (KvTransferTimingMode::FullPrompt, 8.0),
        (KvTransferTimingMode::DestinationMissing, 4.0),
    ] {
        let id = handoff_id();
        let mut coordinator = HandoffCoordinatorCore::new(id, HandoffOrder::SourceFirst);
        let submit = one(coordinator.start().unwrap());
        acknowledge(&mut coordinator, submit, HandoffActionOutcome::Submitted);
        let reserve = one(coordinator
            .on_fact(HandoffFact::SourceHeld {
                handoff_id: id,
                transfer_timing: HandoffTransferTiming {
                    mode,
                    full_prompt_tokens: 8,
                    kv_bytes_per_token: Some(1_000_000),
                    bandwidth_gb_s: Some(1.0),
                },
            })
            .unwrap());
        acknowledge(&mut coordinator, reserve, HandoffActionOutcome::Accepted);
        let transfer = one(coordinator
            .on_fact(HandoffFact::DestinationReserved {
                handoff_id: id,
                transferable_prompt_tokens: 4,
            })
            .unwrap());
        assert!(matches!(
            transfer.action,
            HandoffAction::StartTransfer { delay_ms, .. }
                if delay_ms == expected_delay_ms
        ));
    }
}

#[test]
fn full_destination_hit_keeps_zero_delay_transfer_transition() {
    let id = handoff_id();
    let mut coordinator = HandoffCoordinatorCore::new(id, HandoffOrder::SourceFirst);
    let submit = one(coordinator.start().unwrap());
    acknowledge(&mut coordinator, submit, HandoffActionOutcome::Submitted);
    let reserve = one(coordinator
        .on_fact(HandoffFact::SourceHeld {
            handoff_id: id,
            transfer_timing: HandoffTransferTiming {
                mode: KvTransferTimingMode::DestinationMissing,
                full_prompt_tokens: 8,
                kv_bytes_per_token: Some(1_000_000),
                bandwidth_gb_s: Some(1.0),
            },
        })
        .unwrap());
    acknowledge(&mut coordinator, reserve, HandoffActionOutcome::Accepted);
    let transfer = one(coordinator
        .on_fact(HandoffFact::DestinationReserved {
            handoff_id: id,
            transferable_prompt_tokens: 0,
        })
        .unwrap());
    assert!(matches!(
        transfer.action,
        HandoffAction::StartTransfer { delay_ms: 0.0, .. }
    ));
}

#[test]
fn cancellation_cleans_actions_that_may_have_been_applied() {
    let id = handoff_id();
    let mut coordinator = HandoffCoordinatorCore::new(id, HandoffOrder::SourceFirst);
    let submit = one(coordinator.start().unwrap());
    acknowledge(&mut coordinator, submit, HandoffActionOutcome::Submitted);
    let reserve = one(coordinator
        .on_fact(HandoffFact::SourceHeld {
            handoff_id: id,
            transfer_timing: transfer_timing(None),
        })
        .unwrap());

    let cleanup = coordinator
        .on_fact(HandoffFact::Canceled { handoff_id: id })
        .unwrap();
    assert_eq!(cleanup.len(), 2);
    assert!(
        cleanup
            .iter()
            .any(|action| matches!(action.action, HandoffAction::CancelSource { .. }))
    );
    assert!(
        cleanup
            .iter()
            .any(|action| matches!(action.action, HandoffAction::CancelDestination { .. }))
    );

    assert!(acknowledge(&mut coordinator, reserve, HandoffActionOutcome::Accepted).is_empty());
    let mut completed = false;
    for action in cleanup.iter().copied() {
        let next = acknowledge(&mut coordinator, action, HandoffActionOutcome::Noop);
        completed |= next
            .iter()
            .any(|action| matches!(action.action, HandoffAction::Complete { .. }));
    }
    assert!(completed);
    assert!(coordinator.is_complete());
    for action in cleanup {
        assert!(
            coordinator
                .on_action_outcome(action.id, HandoffActionOutcome::Noop)
                .unwrap()
                .is_empty()
        );
    }
}

fn drive_to_activation(order: HandoffOrder) -> (HandoffCoordinatorCore, IssuedHandoffAction) {
    let id = handoff_id();
    let mut coordinator = HandoffCoordinatorCore::new(id, order);
    let first = one(coordinator.start().unwrap());
    let reserve = match order {
        HandoffOrder::SourceFirst => {
            acknowledge(&mut coordinator, first, HandoffActionOutcome::Submitted);
            one(coordinator
                .on_fact(HandoffFact::SourceHeld {
                    handoff_id: id,
                    transfer_timing: transfer_timing(None),
                })
                .unwrap())
        }
        HandoffOrder::DestinationFirst => first,
    };
    acknowledge(&mut coordinator, reserve, HandoffActionOutcome::Accepted);
    let after_reserved = coordinator.on_fact(destination_reserved(id)).unwrap();
    let transfer = match order {
        HandoffOrder::SourceFirst => one(after_reserved),
        HandoffOrder::DestinationFirst => {
            let submit = one(after_reserved);
            acknowledge(&mut coordinator, submit, HandoffActionOutcome::Submitted);
            one(coordinator
                .on_fact(HandoffFact::SourceHeld {
                    handoff_id: id,
                    transfer_timing: transfer_timing(None),
                })
                .unwrap())
        }
    };
    acknowledge(&mut coordinator, transfer, HandoffActionOutcome::Scheduled);
    let activation = one(coordinator
        .on_fact(HandoffFact::TransferCompleted { handoff_id: id })
        .unwrap());
    (coordinator, activation)
}

#[test]
fn lost_activation_ack_is_cleaned_on_failure_or_timeout() {
    for (order, fact) in [
        (
            HandoffOrder::SourceFirst,
            HandoffFact::TimedOut {
                handoff_id: handoff_id(),
            },
        ),
        (
            HandoffOrder::DestinationFirst,
            HandoffFact::Failed {
                handoff_id: handoff_id(),
            },
        ),
    ] {
        let (mut coordinator, activation) = drive_to_activation(order);
        let cleanup = coordinator.on_fact(fact).unwrap();
        assert!(matches!(
            activation.action,
            HandoffAction::ActivateDestination { .. }
        ));
        assert!(
            cleanup
                .iter()
                .any(|issued| matches!(issued.action, HandoffAction::CancelSource { .. }))
        );
        assert!(
            cleanup
                .iter()
                .any(|issued| matches!(issued.action, HandoffAction::CancelDestination { .. }))
        );
        assert!(
            coordinator
                .on_action_outcome(activation.id, HandoffActionOutcome::Applied)
                .unwrap()
                .is_empty()
        );
        for action in cleanup {
            acknowledge(&mut coordinator, action, HandoffActionOutcome::Noop);
        }
        assert_eq!(coordinator.completion(), Some(HandoffCompletion::Canceled));
    }
}

#[test]
fn invalid_same_handoff_ordering_is_rejected() {
    let id = handoff_id();
    let mut coordinator = HandoffCoordinatorCore::new(id, HandoffOrder::SourceFirst);
    let _submit = one(coordinator.start().unwrap());
    assert!(
        coordinator
            .on_fact(HandoffFact::SourceHeld {
                handoff_id: id,
                transfer_timing: transfer_timing(None),
            })
            .is_err()
    );

    let mut coordinator = HandoffCoordinatorCore::new(id, HandoffOrder::DestinationFirst);
    let reserve = one(coordinator.start().unwrap());
    acknowledge(&mut coordinator, reserve, HandoffActionOutcome::Accepted);
    assert!(
        coordinator
            .on_fact(HandoffFact::TransferCompleted { handoff_id: id })
            .is_err()
    );
}

#[test]
fn invalid_transfer_timing_is_rejected_before_scheduling() {
    for bandwidth_gb_s in [-1.0, f64::NAN, f64::INFINITY] {
        let id = handoff_id();
        let mut coordinator = HandoffCoordinatorCore::new(id, HandoffOrder::SourceFirst);
        let submit = one(coordinator.start().unwrap());
        acknowledge(&mut coordinator, submit, HandoffActionOutcome::Submitted);

        let error = coordinator
            .on_fact(HandoffFact::SourceHeld {
                handoff_id: id,
                transfer_timing: HandoffTransferTiming {
                    bandwidth_gb_s: Some(bandwidth_gb_s),
                    ..transfer_timing(Some(1.0))
                },
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid handoff transfer bandwidth")
        );
    }
}

#[test]
fn timeout_before_initial_ack_cleans_only_the_issued_owner() {
    for order in [HandoffOrder::SourceFirst, HandoffOrder::DestinationFirst] {
        let id = handoff_id();
        let mut coordinator = HandoffCoordinatorCore::new(id, order);
        let initial = one(coordinator.start().unwrap());
        let cleanup = coordinator
            .on_fact(HandoffFact::TimedOut { handoff_id: id })
            .unwrap();
        assert_eq!(cleanup.len(), 1);
        match order {
            HandoffOrder::SourceFirst => assert!(matches!(
                cleanup[0].action,
                HandoffAction::CancelSource { .. }
            )),
            HandoffOrder::DestinationFirst => assert!(matches!(
                cleanup[0].action,
                HandoffAction::CancelDestination { .. }
            )),
        }
        let initial_outcome = match initial.action {
            HandoffAction::SubmitPrefill { .. } => HandoffActionOutcome::Submitted,
            HandoffAction::ReserveDestination { .. } => HandoffActionOutcome::Accepted,
            _ => unreachable!(),
        };
        assert!(
            coordinator
                .on_action_outcome(initial.id, initial_outcome)
                .unwrap()
                .is_empty()
        );
        let complete = one(acknowledge(
            &mut coordinator,
            cleanup[0],
            HandoffActionOutcome::Noop,
        ));
        assert!(matches!(complete.action, HandoffAction::Complete { .. }));
        assert_eq!(coordinator.completion(), Some(HandoffCompletion::Canceled));
    }
}

#[test]
fn destination_first_unacknowledged_prefill_is_cleaned_as_ambiguous_ownership() {
    let id = handoff_id();
    let mut coordinator = HandoffCoordinatorCore::new(id, HandoffOrder::DestinationFirst);
    let reserve = one(coordinator.start().unwrap());
    acknowledge(&mut coordinator, reserve, HandoffActionOutcome::Accepted);
    let submit = one(coordinator.on_fact(destination_reserved(id)).unwrap());
    assert!(matches!(submit.action, HandoffAction::SubmitPrefill { .. }));

    let cleanup = coordinator
        .on_fact(HandoffFact::Canceled { handoff_id: id })
        .unwrap();
    assert_eq!(cleanup.len(), 2);
    assert!(
        cleanup
            .iter()
            .any(|action| matches!(action.action, HandoffAction::CancelSource { .. }))
    );
    assert!(
        cleanup
            .iter()
            .any(|action| matches!(action.action, HandoffAction::CancelDestination { .. }))
    );
    assert!(
        coordinator
            .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
            .unwrap()
            .is_empty()
    );
    for action in cleanup {
        acknowledge(&mut coordinator, action, HandoffActionOutcome::Noop);
    }
    assert_eq!(coordinator.completion(), Some(HandoffCompletion::Canceled));
}

#[test]
fn timeout_after_transfer_is_scheduled_cleans_both_owners() {
    for order in [HandoffOrder::SourceFirst, HandoffOrder::DestinationFirst] {
        let id = handoff_id();
        let mut coordinator = HandoffCoordinatorCore::new(id, order);
        let first = one(coordinator.start().unwrap());
        let reserve = match order {
            HandoffOrder::SourceFirst => {
                acknowledge(&mut coordinator, first, HandoffActionOutcome::Submitted);
                one(coordinator
                    .on_fact(HandoffFact::SourceHeld {
                        handoff_id: id,
                        transfer_timing: transfer_timing(None),
                    })
                    .unwrap())
            }
            HandoffOrder::DestinationFirst => first,
        };
        acknowledge(&mut coordinator, reserve, HandoffActionOutcome::Accepted);
        let after_reserved = coordinator.on_fact(destination_reserved(id)).unwrap();
        let transfer = match order {
            HandoffOrder::SourceFirst => one(after_reserved),
            HandoffOrder::DestinationFirst => {
                let submit = one(after_reserved);
                acknowledge(&mut coordinator, submit, HandoffActionOutcome::Submitted);
                one(coordinator
                    .on_fact(HandoffFact::SourceHeld {
                        handoff_id: id,
                        transfer_timing: transfer_timing(None),
                    })
                    .unwrap())
            }
        };
        acknowledge(&mut coordinator, transfer, HandoffActionOutcome::Scheduled);

        let cleanup = coordinator
            .on_fact(HandoffFact::TimedOut { handoff_id: id })
            .unwrap();
        assert_eq!(cleanup.len(), 2);
        for action in cleanup {
            acknowledge(&mut coordinator, action, HandoffActionOutcome::Noop);
        }
        assert_eq!(coordinator.completion(), Some(HandoffCompletion::Canceled));
    }
}

#[test]
fn lost_source_release_ack_still_runs_ambiguous_cleanup() {
    for order in [HandoffOrder::SourceFirst, HandoffOrder::DestinationFirst] {
        let id = handoff_id();
        let (mut coordinator, activation) = drive_to_activation(order);
        let release = one(acknowledge(
            &mut coordinator,
            activation,
            HandoffActionOutcome::Applied,
        ));
        let cleanup = coordinator
            .on_fact(HandoffFact::Canceled { handoff_id: id })
            .unwrap();
        assert!(
            cleanup
                .iter()
                .any(|issued| matches!(issued.action, HandoffAction::CancelSource { .. }))
        );
        assert!(
            cleanup
                .iter()
                .any(|issued| matches!(issued.action, HandoffAction::CancelDestination { .. }))
        );
        assert!(
            coordinator
                .on_action_outcome(release.id, HandoffActionOutcome::Applied)
                .unwrap()
                .is_empty()
        );
        for action in cleanup {
            acknowledge(&mut coordinator, action, HandoffActionOutcome::Noop);
        }
        assert_eq!(coordinator.completion(), Some(HandoffCompletion::Canceled));
    }
}

#[test]
fn duplicates_are_idempotent_but_conflicts_are_rejected() {
    let id = handoff_id();
    let mut coordinator = HandoffCoordinatorCore::new(id, HandoffOrder::SourceFirst);
    let submit = one(coordinator.start().unwrap());
    assert!(coordinator.start().unwrap().is_empty());
    assert!(acknowledge(&mut coordinator, submit, HandoffActionOutcome::Submitted).is_empty());
    assert!(acknowledge(&mut coordinator, submit, HandoffActionOutcome::Submitted).is_empty());
    assert!(
        coordinator
            .on_action_outcome(submit.id, HandoffActionOutcome::Applied)
            .is_err()
    );

    let fact = HandoffFact::SourceHeld {
        handoff_id: id,
        transfer_timing: transfer_timing(None),
    };
    assert_eq!(coordinator.on_fact(fact.clone()).unwrap().len(), 1);
    assert!(coordinator.on_fact(fact).unwrap().is_empty());
    assert!(
        coordinator
            .on_fact(HandoffFact::DestinationReserved {
                handoff_id: HandoffId::from(Uuid::from_u128(99)),
                transferable_prompt_tokens: 1,
            })
            .is_err()
    );
}
