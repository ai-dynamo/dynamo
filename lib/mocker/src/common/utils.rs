// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
use std::sync::Once;

use aisimulate_core::engine::{WorkerType as EngineWorkerType, prefill_handoff_delay_ms};

use crate::common::handoff::HandoffTransferTiming;
use crate::common::protocols::{KvTransferTimingMode, MockEngineArgs, WorkerType};

pub fn prefill_handoff_transfer_timing(
    num_input_tokens: usize,
    kv_transfer_bandwidth: Option<f64>,
    kv_bytes_per_token: Option<usize>,
    mode: KvTransferTimingMode,
) -> HandoffTransferTiming {
    HandoffTransferTiming {
        mode,
        full_prompt_tokens: num_input_tokens,
        kv_bytes_per_token,
        bandwidth_gb_s: kv_transfer_bandwidth,
    }
}

/// Compute the modeled handoff delay after a prefill worker emits its terminal token.
///
/// NOTE: this intentionally does not model the internal prefill TTFT itself accurately, and the
/// exact prefill/decode boundary is backend dependent. For now we only care about decode-visible
/// TTFT, which is what the client observes, so modeling the delay as prefill-to-decode handoff is
/// good enough.
pub fn compute_prefill_handoff_delay_ms(
    worker_type: WorkerType,
    completed: bool,
    num_input_tokens: usize,
    kv_transfer_bandwidth: Option<f64>,
    kv_bytes_per_token: Option<usize>,
) -> Option<f64> {
    let worker_type = match worker_type {
        WorkerType::Aggregated => EngineWorkerType::Aggregated,
        WorkerType::Prefill => EngineWorkerType::Prefill,
        WorkerType::Decode => EngineWorkerType::Decode,
    };
    let delay_ms = prefill_handoff_delay_ms(
        worker_type,
        completed,
        num_input_tokens,
        kv_transfer_bandwidth,
        kv_bytes_per_token,
    );
    if let Some(delay_ms) = delay_ms {
        tracing::debug!(
            num_input_tokens,
            bandwidth_gb_s = kv_transfer_bandwidth,
            delay_ms = format!("{delay_ms:.2}"),
            "KV handoff delay for prefill completion"
        );
    }
    delay_ms
}

/// Compute the KV transfer delay duration for a given number of input tokens.
///
/// Returns `None` if KV transfer simulation is disabled (bandwidth is 0 or not configured).
pub fn compute_kv_transfer_delay(
    args: &MockEngineArgs,
    num_input_tokens: usize,
) -> Option<Duration> {
    compute_prefill_handoff_delay_ms(
        args.worker_type,
        true,
        num_input_tokens,
        args.kv_transfer_bandwidth,
        args.kv_bytes_per_token,
    )
    .map(|delay_ms| Duration::from_secs_f64(delay_ms / 1000.0))
}

/// Sleep for the specified duration using timerfd on Linux for precision.
pub async fn sleep_precise(duration: Duration) {
    sleep_until_precise(Instant::now() + duration).await;
}

#[cfg(target_os = "linux")]
enum PreciseTimerState {
    Uninitialized,
    Ready(tokio_timerfd::Delay),
    // A timerfd error permanently selects Tokio for this timer instance.
    Disabled,
}

#[cfg(target_os = "linux")]
static TIMERFD_FALLBACK_WARNING: Once = Once::new();

#[cfg(all(test, target_os = "linux"))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum TimerTestMode {
    Tokio,
    TimerFd,
    FailCreation,
}

pub(crate) struct ReusablePreciseTimer {
    #[cfg(target_os = "linux")]
    state: PreciseTimerState,
    #[cfg(all(test, target_os = "linux"))]
    test_mode: TimerTestMode,
    #[cfg(all(test, target_os = "linux"))]
    timerfd_create_attempts: usize,
}

impl Default for ReusablePreciseTimer {
    fn default() -> Self {
        Self {
            #[cfg(target_os = "linux")]
            state: PreciseTimerState::Uninitialized,
            #[cfg(all(test, target_os = "linux"))]
            test_mode: TimerTestMode::Tokio,
            #[cfg(all(test, target_os = "linux"))]
            timerfd_create_attempts: 0,
        }
    }
}

impl ReusablePreciseTimer {
    pub(crate) async fn sleep_until(&mut self, deadline: Instant) {
        #[cfg(all(test, target_os = "linux"))]
        if self.test_mode == TimerTestMode::Tokio {
            sleep_until_tokio(deadline).await;
            return;
        }

        if deadline <= Instant::now() {
            tokio::task::yield_now().await;
            return;
        }

        #[cfg(target_os = "linux")]
        {
            match self.arm_timerfd(deadline) {
                Ok(true) => {}
                Ok(false) => {
                    sleep_until_tokio(deadline).await;
                    return;
                }
                Err(error) => {
                    self.disable_timerfd(&error);
                    sleep_until_tokio(deadline).await;
                    return;
                }
            }

            let result = match &mut self.state {
                PreciseTimerState::Ready(delay) => Some(delay.await),
                PreciseTimerState::Uninitialized | PreciseTimerState::Disabled => None,
            };
            match result {
                Some(Ok(())) => {}
                Some(Err(error)) => {
                    self.disable_timerfd(&error);
                    sleep_until_tokio(deadline).await;
                }
                None => sleep_until_tokio(deadline).await,
            }
        }
        #[cfg(not(target_os = "linux"))]
        sleep_until_tokio(deadline).await;
    }

    #[cfg(target_os = "linux")]
    fn arm_timerfd(&mut self, deadline: Instant) -> std::io::Result<bool> {
        match &mut self.state {
            PreciseTimerState::Uninitialized => {
                #[cfg(test)]
                {
                    self.timerfd_create_attempts += 1;
                    if self.test_mode == TimerTestMode::FailCreation {
                        return Err(std::io::Error::other("injected timerfd creation failure"));
                    }
                }
                self.state = PreciseTimerState::Ready(tokio_timerfd::Delay::new(deadline)?);
                Ok(true)
            }
            PreciseTimerState::Ready(delay) => {
                delay.reset(deadline);
                Ok(true)
            }
            PreciseTimerState::Disabled => Ok(false),
        }
    }

    #[cfg(target_os = "linux")]
    fn disable_timerfd(&mut self, error: &std::io::Error) {
        self.state = PreciseTimerState::Disabled;
        TIMERFD_FALLBACK_WARNING.call_once(|| {
            tracing::warn!(
                error = %error,
                "precise timerfd failed; using Tokio for this timer"
            );
        });
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(crate) fn with_timerfd_for_test() -> Self {
        Self {
            test_mode: TimerTestMode::TimerFd,
            ..Self::default()
        }
    }

    #[cfg(all(test, target_os = "linux"))]
    fn with_timerfd_creation_failure() -> Self {
        Self {
            test_mode: TimerTestMode::FailCreation,
            ..Self::default()
        }
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(crate) fn timerfd_create_attempts(&self) -> usize {
        self.timerfd_create_attempts
    }
}

async fn sleep_until_tokio(deadline: Instant) {
    if deadline <= Instant::now() {
        tokio::task::yield_now().await;
    } else {
        tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
    }
}

/// Sleep until the specified deadline using timerfd on Linux for precision.
///
/// Unlike `sleep_precise`, this accounts for time already elapsed since the
/// deadline's reference point, making it suitable for simulation loops where
/// computation time should be subtracted from the sleep.
pub async fn sleep_until_precise(deadline: Instant) {
    ReusablePreciseTimer::default().sleep_until(deadline).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::Poll;

    #[tokio::test(flavor = "current_thread")]
    async fn test_expired_precise_sleep_yields_to_runtime() {
        let sleep = sleep_until_precise(Instant::now());
        tokio::pin!(sleep);

        let first_poll = futures::poll!(sleep.as_mut());

        assert!(matches!(first_poll, Poll::Pending));
        sleep.await;
    }

    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "current_thread")]
    async fn reusable_precise_timer_reuses_its_timerfd() {
        let mut timer = ReusablePreciseTimer::with_timerfd_for_test();

        for _ in 0..2 {
            let started = Instant::now();
            let deadline = started + Duration::from_millis(5);
            tokio::time::timeout(Duration::from_secs(1), timer.sleep_until(deadline))
                .await
                .expect("reusable precise timer did not complete");
            assert!(started.elapsed() >= Duration::from_millis(1));
        }

        assert_eq!(timer.timerfd_create_attempts(), 1);
    }

    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "current_thread")]
    async fn reusable_precise_timer_rearms_after_cancelled_wait_expires() {
        let mut timer = ReusablePreciseTimer::with_timerfd_for_test();
        let cancelled_deadline = Instant::now() + Duration::from_millis(50);
        {
            let wait = timer.sleep_until(cancelled_deadline);
            tokio::pin!(wait);
            tokio::select! {
                biased;
                _ = tokio::time::sleep(Duration::from_millis(2)) => {}
                _ = &mut wait => panic!("the cancelled wait completed early"),
            }
        }

        tokio::time::sleep_until(tokio::time::Instant::from_std(
            cancelled_deadline + Duration::from_millis(5),
        ))
        .await;

        let rearmed_at = Instant::now();
        let rearmed_deadline = rearmed_at + Duration::from_millis(30);
        tokio::time::timeout(Duration::from_secs(1), timer.sleep_until(rearmed_deadline))
            .await
            .expect("rearmed precise timer did not complete");

        assert!(
            rearmed_at.elapsed() >= Duration::from_millis(20),
            "the old unread expiration completed the rearmed wait early"
        );
        assert_eq!(timer.timerfd_create_attempts(), 1);
    }

    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "current_thread")]
    async fn reusable_precise_timer_latches_creation_failure() {
        let mut timer = ReusablePreciseTimer::with_timerfd_creation_failure();

        for _ in 0..2 {
            let started = Instant::now();
            timer.sleep_until(started + Duration::from_millis(2)).await;
            assert!(started.elapsed() >= Duration::from_millis(1));
        }

        assert_eq!(timer.timerfd_create_attempts(), 1);
    }

    #[test]
    fn test_prefill_handoff_delay_only_applies_to_completed_prefill() {
        let delay_ms = compute_prefill_handoff_delay_ms(
            WorkerType::Prefill,
            true,
            128,
            Some(1.0),
            Some(1_000_000),
        )
        .expect("prefill completion should produce a handoff delay");
        assert!((delay_ms - 128.0).abs() < 1e-9);

        assert!(
            compute_prefill_handoff_delay_ms(
                WorkerType::Prefill,
                false,
                128,
                Some(1.0),
                Some(1_000_000),
            )
            .is_none()
        );
        assert!(
            compute_prefill_handoff_delay_ms(
                WorkerType::Decode,
                true,
                128,
                Some(1.0),
                Some(1_000_000),
            )
            .is_none()
        );
    }
}
