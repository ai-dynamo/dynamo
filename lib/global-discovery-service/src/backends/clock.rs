// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wall-clock time for backends that store leases where other processes read
//! them, and so cannot use Tokio's monotonic clock. Tests drive expiry with a
//! [`ManualClock`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The current time as Unix milliseconds.
pub trait Clock: Send + Sync {
    fn now_ms(&self) -> u64;
}

/// The system clock. A time before 1970 reads as 0.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, millis)
    }
}

/// A clock that moves only when it is set or advanced. It starts at the time
/// it is given, or at 0 by default.
#[derive(Debug, Default)]
pub struct ManualClock {
    now_ms: AtomicU64,
}

impl ManualClock {
    pub fn new(now_ms: u64) -> Self {
        Self {
            now_ms: AtomicU64::new(now_ms),
        }
    }

    pub fn set(&self, now_ms: u64) {
        self.now_ms.store(now_ms, Ordering::SeqCst);
    }

    /// Moves the clock forward by `by`, rounded down to milliseconds.
    pub fn advance(&self, by: Duration) {
        let by = millis(by);
        // `fetch_update` with `Some` never fails.
        let _ = self
            .now_ms
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |now| {
                Some(now.saturating_add(by))
            });
    }
}

impl Clock for ManualClock {
    fn now_ms(&self) -> u64 {
        self.now_ms.load(Ordering::SeqCst)
    }
}

/// Whole milliseconds, saturating at `u64::MAX`.
pub(crate) fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_manual_clock_moves_only_when_told() {
        let clock = ManualClock::new(1_000);
        assert_eq!(clock.now_ms(), 1_000);
        clock.advance(Duration::from_millis(1_500));
        assert_eq!(clock.now_ms(), 2_500);
        clock.set(10);
        assert_eq!(clock.now_ms(), 10);
        clock.advance(Duration::MAX);
        assert_eq!(clock.now_ms(), u64::MAX);
        assert_eq!(ManualClock::default().now_ms(), 0);
    }

    #[test]
    fn the_system_clock_is_after_2026() {
        // 2026-01-01T00:00:00Z.
        assert!(SystemClock.now_ms() > 1_767_225_600_000);
    }
}
