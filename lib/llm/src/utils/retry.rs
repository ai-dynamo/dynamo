// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Retry pacing and failure logging shared by background reconnect loops.

use std::time::Duration;

/// Capped exponential backoff: `initial`, doubling up to `max`.
#[derive(Debug, Clone)]
pub(crate) struct Backoff {
    initial: Duration,
    max: Duration,
    next: Duration,
}

impl Backoff {
    pub(crate) const fn new(initial: Duration, max: Duration) -> Self {
        Self {
            initial,
            max,
            next: initial,
        }
    }

    /// The delay before the next attempt; each call doubles the one after it.
    pub(crate) fn next_delay(&mut self) -> Duration {
        let delay = self.next;
        self.next = self.next.saturating_mul(2).min(self.max);
        delay
    }

    pub(crate) fn reset(&mut self) {
        self.next = self.initial;
    }
}

/// Consecutive failures of one repeating operation, so it can warn once per
/// streak, log repeats at debug, and log recovery at info.
#[derive(Debug, Default)]
pub(crate) struct FailureStreak {
    failures: u64,
}

impl FailureStreak {
    /// Counts a failure. True when it starts a streak and should be warned about.
    pub(crate) fn fail(&mut self) -> bool {
        self.failures = self.failures.saturating_add(1);
        self.failures == 1
    }

    /// Ends the streak. Returns its length when one was in progress, so the
    /// caller can report recovery.
    pub(crate) fn recover(&mut self) -> Option<u64> {
        let failures = std::mem::take(&mut self.failures);
        (failures > 0).then_some(failures)
    }

    pub(crate) fn failures(&self) -> u64 {
        self.failures
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_to_its_cap_and_resets() {
        let mut backoff = Backoff::new(Duration::from_secs(1), Duration::from_secs(5));
        let delays = [(); 5].map(|()| backoff.next_delay().as_secs());
        assert_eq!(delays, [1, 2, 4, 5, 5]);
        backoff.reset();
        assert_eq!(backoff.next_delay(), Duration::from_secs(1));
    }

    #[test]
    fn failure_streak_starts_once_and_reports_its_length_on_recovery() {
        let mut streak = FailureStreak::default();
        assert_eq!(streak.recover(), None);
        assert!(streak.fail());
        assert!(!streak.fail());
        assert_eq!(streak.failures(), 2);
        assert_eq!(streak.recover(), Some(2));
        assert!(streak.fail());
    }
}
