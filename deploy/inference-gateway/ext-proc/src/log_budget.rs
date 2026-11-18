// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A per-second cap on diagnostic log lines.
//!
//! The EPP's diagnostic lines (slow-stream summaries, runtime stalls) are most
//! useful exactly when the process is struggling, which is also when an
//! unbounded burst of them would add to the struggle: every line is a
//! synchronous write to stdout from the thread that emits it.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Allows up to `per_second` lines in each one-second window and counts the
/// rest. The limit is approximate at window boundaries, which is all a log
/// budget needs.
pub(crate) struct LogBudget {
    per_second: u32,
    window: AtomicU64,
    used: AtomicU32,
    suppressed: AtomicU64,
}

impl LogBudget {
    pub(crate) const fn new(per_second: u32) -> Self {
        Self {
            per_second,
            window: AtomicU64::new(0),
            used: AtomicU32::new(0),
            suppressed: AtomicU64::new(0),
        }
    }

    /// Spend one line of the current window's budget. Returns the number of
    /// lines suppressed since the last line that was allowed, or `None` when
    /// the budget is spent and this line must be dropped.
    pub(crate) fn take(&self) -> Option<u64> {
        self.take_at(EPOCH.elapsed().as_secs())
    }

    fn take_at(&self, second: u64) -> Option<u64> {
        // Only a later second opens a new window: a caller that read the
        // clock just before another moved the window must not move it back
        // and refill the budget.
        let window = self.window.load(Ordering::Relaxed);
        if second > window
            && self
                .window
                .compare_exchange(window, second, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            self.used.store(0, Ordering::Relaxed);
        }
        if self.used.fetch_add(1, Ordering::Relaxed) < self.per_second {
            Some(self.suppressed.swap(0, Ordering::Relaxed))
        } else {
            self.suppressed.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_the_budget_then_counts_what_it_drops() {
        let budget = LogBudget::new(2);
        assert_eq!(budget.take_at(7), Some(0));
        assert_eq!(budget.take_at(7), Some(0));
        assert_eq!(budget.take_at(7), None);
        assert_eq!(budget.take_at(7), None);

        // A new window refills the budget, and its first line reports the
        // lines the previous window dropped.
        assert_eq!(budget.take_at(8), Some(2));
        assert_eq!(budget.take_at(8), Some(0));
        assert_eq!(budget.take_at(8), None);

        // A caller still holding the previous second spends from the current
        // window instead of refilling it.
        assert_eq!(budget.take_at(7), None);
    }
}
