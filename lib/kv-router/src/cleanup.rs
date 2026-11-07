// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared throttle for periodic stale-state cleanup.
//!
//! [`CleanupState`] and [`CleanupGuard`] allow at most one cleanup in flight
//! and space completed runs at least [`CLEANUP_INTERVAL_MS`] apart. The
//! sequence-side
//! [`PromptMembershipTrie`](super::sequences::prompt_membership_trie::PromptMembershipTrie)
//! uses them to throttle its stale-child sweep, and the session prefix index
//! reuses [`CLEANUP_INTERVAL_MS`] as its cleanup interval.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

pub const CLEANUP_INTERVAL_MS: u64 = 5 * 60 * 1000;

pub struct CleanupState {
    clock_origin: Instant,
    last_cleanup_elapsed_ms: AtomicU64,
    scheduled: AtomicBool,
}

impl CleanupState {
    pub fn new() -> Self {
        Self {
            clock_origin: Instant::now(),
            last_cleanup_elapsed_ms: AtomicU64::new(0),
            scheduled: AtomicBool::new(false),
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.clock_origin.elapsed().as_millis() as u64
    }

    pub fn try_schedule(&self) -> bool {
        let now_ms = self.elapsed_ms();
        let last_ms = self.last_cleanup_elapsed_ms.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last_ms) < CLEANUP_INTERVAL_MS {
            return false;
        }

        self.scheduled
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }
}

impl Default for CleanupState {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CleanupGuard<'a> {
    state: &'a CleanupState,
    completed_elapsed_ms: Option<u64>,
}

impl<'a> CleanupGuard<'a> {
    pub fn new(state: &'a CleanupState) -> Self {
        Self {
            state,
            completed_elapsed_ms: None,
        }
    }

    pub fn mark_completed(&mut self) {
        self.completed_elapsed_ms = Some(self.state.elapsed_ms());
    }
}

impl Drop for CleanupGuard<'_> {
    fn drop(&mut self) {
        if let Some(elapsed_ms) = self.completed_elapsed_ms {
            self.state
                .last_cleanup_elapsed_ms
                .store(elapsed_ms, Ordering::Relaxed);
        }
        self.state.scheduled.store(false, Ordering::Release);
    }
}
