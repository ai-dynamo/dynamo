// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runs `Controller::tick` every `interval` on the tokio runtime that hosts the EPP.
//! If no runtime is available when the factory runs (tests, tools), the loop is simply not started
//! and the policy keeps baseline weights, which is today's behavior.
//! The task holds only a weak reference, so it ends once the policy that owns the state is dropped.

use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use tokio::task::JoinHandle;

use crate::controller::{Controller, Transition};
use crate::state::SharedState;

/// Spawns the loop and returns its handle, or `None` when there is no tokio runtime.
pub fn start(
    shared: Arc<SharedState>,
    mut controller: Controller,
    interval: Duration,
) -> Option<JoinHandle<()>> {
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        tracing::warn!(
            "adaptive-imbalance: no tokio runtime at policy creation, loop not started; weights stay at baseline"
        );
        return None;
    };
    let weak: Weak<SharedState> = Arc::downgrade(&shared);
    drop(shared);
    Some(handle.spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let Some(shared) = weak.upgrade() else {
                tracing::debug!("adaptive-imbalance: policy dropped, stopping the control loop");
                break;
            };
            let snapshot = shared.load_snapshot();
            let now = Instant::now();
            let Some(update) = controller.tick(&snapshot, now) else {
                continue;
            };
            let next = update.weights;
            // Age of the load sample the loop acted on; a growing age means the request path has gone quiet.
            let snapshot_age_ms = snapshot
                .taken_at
                .map(|t| now.saturating_duration_since(t).as_millis() as u64);
            macro_rules! log_update {
                ($level:expr, $message:literal) => {
                    tracing::event!(
                        $level,
                        imbalance = next.imbalance,
                        split = next.split,
                        overlap_score_credit = next.set.overlap_score_credit,
                        prefill_load_scale = next.set.prefill_load_scale,
                        decode_active_request_weight = next.set.decode_active_request_weight,
                        temperature = next.temperature,
                        snapshot_age_ms = ?snapshot_age_ms,
                        $message
                    )
                };
            }
            // Transitions are rare (hysteresis and dwell bound them), so they go to info; the
            // per-tick weight changes while engaged go to debug.
            match update.transition {
                Some(Transition::Engaged) => {
                    log_update!(tracing::Level::INFO, "adaptive-imbalance: engaged")
                }
                Some(Transition::Disengaged) => {
                    log_update!(tracing::Level::INFO, "adaptive-imbalance: disengaged")
                }
                None => log_update!(tracing::Level::DEBUG, "adaptive-imbalance: weights updated"),
            }
            shared.publish_weights(next);
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Parameters;
    use crate::math::WeightSet;
    use crate::state::Weights;

    fn baseline() -> Weights {
        Weights {
            set: WeightSet {
                overlap_score_credit: 1.0,
                prefill_load_scale: 1.0,
                decode_active_request_weight: 0.5,
            },
            temperature: 0.0,
            imbalance: 0.0,
            split: 0.0,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn loop_publishes_after_hot_load_is_observed() {
        let shared = Arc::new(SharedState::new(baseline()));
        let params = Parameters {
            dwell_cycles: 1,
            ..Parameters::default()
        };
        start(
            Arc::clone(&shared),
            Controller::new(&params, baseline()),
            Duration::from_millis(250),
        );

        shared.publish_load(vec![30.0, 4.0, 2.0, 5.0]);
        for _ in 0..10 {
            tokio::time::advance(Duration::from_millis(250)).await;
            tokio::task::yield_now().await;
        }
        let w = shared.weights();
        assert!(w.split > 0.9, "split={}", w.split);
        assert!(w.set.overlap_score_credit < 0.1);
    }

    #[tokio::test(start_paused = true)]
    async fn loop_exits_once_the_shared_state_is_dropped() {
        let shared = Arc::new(SharedState::new(baseline()));
        let handle = start(
            Arc::clone(&shared),
            Controller::new(&Parameters::default(), baseline()),
            Duration::from_millis(250),
        )
        .expect("a runtime is present");

        tokio::time::advance(Duration::from_millis(250)).await;
        tokio::task::yield_now().await;
        assert!(
            !handle.is_finished(),
            "the loop runs while the policy holds the state"
        );
        assert_eq!(
            Arc::strong_count(&shared),
            1,
            "between ticks the task holds no strong reference"
        );

        drop(shared);
        tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("the loop must stop after its state is dropped")
            .expect("the loop must not panic");
    }

    #[test]
    fn without_a_runtime_start_is_a_no_op() {
        let shared = Arc::new(SharedState::new(baseline()));
        let handle = start(
            Arc::clone(&shared),
            Controller::new(&Parameters::default(), baseline()),
            Duration::from_millis(250),
        );
        assert!(handle.is_none());
        assert_eq!(shared.weights().set, baseline().set);
    }
}
