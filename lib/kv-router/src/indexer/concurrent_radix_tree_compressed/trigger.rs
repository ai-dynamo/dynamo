// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The volume trigger's shared state: block estimates that lanes flush into and sweeps
//! overwrite, and the timing of the last sweep. It only reads and writes atomics, with no
//! lock, so it is written against `loom`'s atomics under `cfg(loom)`; the models in
//! `lib/kv-router/loom` include this file. Time is passed in, so the models stay
//! deterministic.
//!
//! The scheduling flag that keeps one sweep in flight is `CleanupState`'s; the models
//! compose it with this state the way `try_schedule_cleanup` does.

#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

/// A volume-triggered sweep also waits this many times the previous sweep's duration.
pub(crate) const SWEEP_DUTY_FACTOR: u64 = 10;
/// Dead blocks must reach `live / DEAD_FRACTION` before a volume-triggered sweep.
pub(crate) const DEAD_FRACTION: u64 = 8;

pub(crate) struct VolumeTrigger {
    dead: AtomicI64,
    linked: AtomicI64,
    last_finish_us: AtomicU64,
    last_duration_us: AtomicU64,
    enabled: AtomicBool,
    dead_floor: AtomicU64,
    min_gap_us: AtomicU64,
}

impl VolumeTrigger {
    pub(crate) fn new(enabled: bool, dead_floor: u64, min_gap_us: u64) -> Self {
        let trigger = Self {
            dead: AtomicI64::new(0),
            linked: AtomicI64::new(0),
            last_finish_us: AtomicU64::new(0),
            last_duration_us: AtomicU64::new(0),
            enabled: AtomicBool::new(false),
            dead_floor: AtomicU64::new(0),
            min_gap_us: AtomicU64::new(0),
        };
        trigger.configure(enabled, dead_floor, min_gap_us);
        trigger
    }

    pub(crate) fn configure(&self, enabled: bool, dead_floor: u64, min_gap_us: u64) {
        self.enabled.store(enabled, Ordering::Relaxed);
        self.dead_floor.store(dead_floor, Ordering::Relaxed);
        self.min_gap_us.store(min_gap_us, Ordering::Relaxed);
    }

    /// Adds a lane's deltas. Relaxed: the estimates only steer when to sweep, and a
    /// sweep's recount replaces them anyway.
    #[inline]
    pub(crate) fn add(&self, dead: i64, linked: i64) {
        if dead != 0 {
            self.dead.fetch_add(dead, Ordering::Relaxed);
        }
        if linked != 0 {
            self.linked.fetch_add(linked, Ordering::Relaxed);
        }
    }

    /// `(dead, linked)` block estimates, clamped at zero: flushes racing a sweep's
    /// overwrite can drive them below their true values.
    #[inline]
    pub(crate) fn estimates(&self) -> (u64, u64) {
        let read = |value: &AtomicI64| value.load(Ordering::Relaxed).max(0) as u64;
        (read(&self.dead), read(&self.linked))
    }

    /// Whether the dead volume calls for a sweep at `now_us`: dead blocks reach an eighth
    /// of the live ones and the floor, and the last sweep ended at least
    /// `max(min_gap, 10 x its duration)` ago. `now_us` is evaluated only once the volume
    /// condition holds.
    #[inline]
    pub(crate) fn due(&self, now_us: impl FnOnce() -> u64) -> bool {
        if !self.enabled.load(Ordering::Relaxed) {
            return false;
        }
        let (dead, linked) = self.estimates();
        let live = linked.saturating_sub(dead);
        let threshold = (live / DEAD_FRACTION).max(self.dead_floor.load(Ordering::Relaxed));
        if dead == 0 || dead < threshold {
            return false;
        }
        let gap = self
            .min_gap_us
            .load(Ordering::Relaxed)
            .max(SWEEP_DUTY_FACTOR.saturating_mul(self.last_duration_us.load(Ordering::Relaxed)));
        now_us().saturating_sub(self.last_finish_us.load(Ordering::Relaxed)) >= gap
    }

    /// Records a sweep that ended at `now_us` after `duration_us`, overwriting the
    /// estimates with what it left linked and dead.
    pub(crate) fn finish(&self, now_us: u64, duration_us: u64, linked: u64, dead: u64) {
        self.linked
            .store(i64::try_from(linked).unwrap_or(i64::MAX), Ordering::Relaxed);
        self.dead
            .store(i64::try_from(dead).unwrap_or(i64::MAX), Ordering::Relaxed);
        self.last_duration_us.store(duration_us, Ordering::Relaxed);
        self.last_finish_us.store(now_us, Ordering::Relaxed);
    }
}
