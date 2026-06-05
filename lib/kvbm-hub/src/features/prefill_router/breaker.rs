// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-owned CD prefill-overload circuit breaker (P1: state machine only).
//!
//! The breaker is a three-tier hysteresis state machine
//! ([`BreakerTier::Calm`] → [`BreakerTier::Warm`] → [`BreakerTier::Hot`]) that
//! senses prefill-fleet pressure and (in P2) is pushed to decode workers so
//! they stop disaggregating into a saturated fleet. It NARROWS disaggregation
//! only (more Local, never more Remote) and never touches the
//! recompute/release/mark_failed/onboarding/session lifecycles.
//!
//! ## P1 scope
//!
//! This file ships the pure machinery: the [`CircuitBreaker`] struct
//! (atomic tier + monotonic epoch + immutable config) and its
//! [`CircuitBreaker::evaluate`] hysteresis step. There is no push, no decode
//! consumption, no GNMT logic — those are P2. The breaker is constructed and
//! ticked only when the operator opts in (`cd_breaker_enabled = true`); with
//! the default-OFF config the hub never constructs it and the entire path is a
//! runtime no-op.
//!
//! ## CRITICAL: tick-driven, not event-driven
//!
//! [`CircuitBreaker::evaluate`] is driven by a DEDICATED breaker-tick task
//! (spawned alongside the manager, ~200–500ms fixed interval) — NOT
//! event-driven on dispatch. Event-driven evaluation LATCHES in HOT: in HOT
//! decode stops enqueuing → no dispatch events → the tier never recomputes →
//! stuck HOT forever. The tick task decouples compute from traffic and is
//! where the recovery (clear) logic runs. See the manager for the spawn.
//!
//! ## Pressure axis & watermark direction
//!
//! The primary axis is the router's FREE-CAPACITY FRACTION `free_frac ∈
//! [0.0, 1.0]` (`Selector::available_permits / total_permits`): `1.0` = fully
//! idle fleet, `0.0` = fully saturated. The breaker trips toward MORE pressure
//! (LOWER free fraction) and clears toward LESS pressure (HIGHER free
//! fraction). The config knobs are PRESSURE-oriented:
//! - `warm_high` (default `0.5`): trip CALM→WARM when `free_frac <= warm_high`.
//! - `hot_high` (default `0.15`): trip to HOT when `free_frac <= hot_high`.
//! - `clear_low` (default `0.7`): descend one tier only once `free_frac >=
//!   clear_low`, sustained over `clear_debounce_ticks`.
//!
//! Trip is IMMEDIATE (one tick over a HIGH watermark). Clear is DEBOUNCED
//! (must hold for N consecutive ticks) and descends one tier at a time
//! (HOT→WARM→CALM). The watermark GAP (`clear_low > warm_high`) plus the
//! debounce damp flapping.
//!
//! The secondary axis is the router CD-queue depth. In P1 there is no
//! queue-depth accessor on the hub queue backend, so the queue axis is
//! DISABLED (the `queue_depth_*` thresholds default to `0`, the disabled
//! sentinel, and `evaluate` ignores the depth when both are `0`).

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use kvbm_protocols::disagg::BreakerTier;

/// Immutable breaker tuning. Mirrors the `cd_breaker_*` knobs on
/// `kvbm_config::DisaggConfig`; the hub constructs one of these only when
/// `cd_breaker_enabled` is true.
#[derive(Debug, Clone, Copy)]
pub struct BreakerConfig {
    /// Free-capacity fraction at/below which the breaker trips to WARM.
    pub warm_high: f64,
    /// Free-capacity fraction at/below which the breaker trips to HOT.
    /// Must be `< warm_high`.
    pub hot_high: f64,
    /// Free-capacity fraction at/above which the breaker may descend a tier
    /// (debounced). Should be `> warm_high` to give a no-flap gap.
    pub clear_low: f64,
    /// CD-queue depth at/above which the breaker trips to WARM. `0` disables
    /// the queue axis.
    pub queue_depth_warm: usize,
    /// CD-queue depth at/above which the breaker trips to HOT. `0` disables
    /// the queue axis.
    pub queue_depth_hot: usize,
    /// Consecutive ticks the clear condition must hold before descending one
    /// tier. Trip is immediate; clear is debounced.
    pub clear_debounce_ticks: u32,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        // Mirrors the kvbm-config serde defaults so the two surfaces can't
        // drift silently.
        Self {
            warm_high: 0.5,
            hot_high: 0.15,
            clear_low: 0.7,
            queue_depth_warm: 0,
            queue_depth_hot: 0,
            clear_debounce_ticks: 3,
        }
    }
}

impl BreakerConfig {
    /// True iff the queue-depth trip axis is active (either threshold set).
    fn queue_axis_enabled(&self) -> bool {
        self.queue_depth_warm > 0 || self.queue_depth_hot > 0
    }
}

fn tier_to_u8(t: BreakerTier) -> u8 {
    match t {
        BreakerTier::Calm => 0,
        BreakerTier::Warm => 1,
        BreakerTier::Hot => 2,
    }
}

fn tier_from_u8(v: u8) -> BreakerTier {
    match v {
        0 => BreakerTier::Calm,
        1 => BreakerTier::Warm,
        _ => BreakerTier::Hot,
    }
}

/// Router-owned circuit breaker. Holds the current [`BreakerTier`] and a
/// monotonic `epoch` (bumped only when the tier changes). The tier/epoch are
/// read with a single relaxed atomic load on the (future, P2) push path; the
/// breaker-tick task is the sole writer via [`Self::evaluate`].
///
/// The epoch is seeded from a hub-boot-monotonic source so a restarted hub's
/// epochs strictly exceed any value a decode may have cached from a prior hub
/// instance (the MUST-FIX from the design). P1 does not push the epoch yet,
/// but seeding it now keeps the invariant in one place.
#[derive(Debug)]
pub struct CircuitBreaker {
    config: BreakerConfig,
    /// Current tier as a `u8` (see [`tier_to_u8`]).
    tier: AtomicU8,
    /// Monotonic epoch; bumped ONLY when [`Self::evaluate`] changes the tier.
    epoch: AtomicU64,
    /// Count of consecutive ticks the clear condition has held. Owned by the
    /// single tick task (no concurrent writers), stored atomically only so the
    /// breaker stays `Sync` behind a shared `Arc`.
    clear_streak: AtomicU64,
}

impl CircuitBreaker {
    /// Construct a breaker starting in [`BreakerTier::Calm`] with the epoch
    /// seeded from `boot_epoch_seed` (a hub-boot-monotonic value; see the
    /// struct docs).
    pub fn new(config: BreakerConfig, boot_epoch_seed: u64) -> Self {
        Self {
            config,
            tier: AtomicU8::new(tier_to_u8(BreakerTier::Calm)),
            epoch: AtomicU64::new(boot_epoch_seed),
            clear_streak: AtomicU64::new(0),
        }
    }

    /// Current tier (single relaxed load).
    pub fn tier(&self) -> BreakerTier {
        tier_from_u8(self.tier.load(Ordering::Relaxed))
    }

    /// Current epoch (single relaxed load).
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    /// Immutable config.
    pub fn config(&self) -> &BreakerConfig {
        &self.config
    }

    /// The tier the pressure signals alone would dictate THIS tick (the "trip
    /// target"), ignoring hysteresis/debounce. Pure helper for [`Self::evaluate`]
    /// and the tests. Higher of the capacity-axis and queue-axis verdicts.
    fn trip_target(&self, free_frac: f64, queue_depth: usize) -> BreakerTier {
        // Capacity axis. `free_frac` is clamped defensively; a NaN (e.g. a
        // 0/0 if a caller forgot to guard total_permits==0) is treated as
        // fully saturated (most conservative: trip).
        let ff = if free_frac.is_nan() {
            0.0
        } else {
            free_frac.clamp(0.0, 1.0)
        };
        let cap_tier = if ff <= self.config.hot_high {
            BreakerTier::Hot
        } else if ff <= self.config.warm_high {
            BreakerTier::Warm
        } else {
            BreakerTier::Calm
        };

        // Queue axis (only if enabled). Trip target is whichever axis is hotter.
        let q_tier = if self.config.queue_axis_enabled() {
            if self.config.queue_depth_hot > 0 && queue_depth >= self.config.queue_depth_hot {
                BreakerTier::Hot
            } else if self.config.queue_depth_warm > 0
                && queue_depth >= self.config.queue_depth_warm
            {
                BreakerTier::Warm
            } else {
                BreakerTier::Calm
            }
        } else {
            BreakerTier::Calm
        };

        cap_tier.max(q_tier)
    }

    /// One hysteresis step. Returns `Some(new_tier)` iff the tier CHANGED this
    /// tick (and bumps the epoch in that case); `None` if it held.
    ///
    /// Semantics:
    /// - **Trip is immediate**: if the pressure signals dictate a HIGHER tier
    ///   than the current one, jump straight to it this tick and reset the
    ///   clear streak.
    /// - **Clear is debounced and single-step**: the tier descends one level
    ///   (HOT→WARM→CALM) only after the clear condition — `free_frac >=
    ///   clear_low` AND the trip target is no hotter than the tier we'd
    ///   descend TO — has held for `clear_debounce_ticks` consecutive ticks.
    ///
    /// `free_frac` is the router free-capacity fraction in `[0.0, 1.0]`;
    /// callers MUST guard `total_permits == 0` upstream (pass `0.0` to trip, or
    /// skip the tick). `queue_depth` is the CD-queue depth (ignored unless a
    /// `queue_depth_*` threshold is set).
    pub fn evaluate(&self, free_frac: f64, queue_depth: usize) -> Option<BreakerTier> {
        let current = self.tier();
        let target = self.trip_target(free_frac, queue_depth);

        // Trip: jump immediately to a hotter tier.
        if target > current {
            self.clear_streak.store(0, Ordering::Relaxed);
            return self.set_tier(target);
        }

        // Already CALM and not tripping: nothing to clear.
        if current == BreakerTier::Calm {
            self.clear_streak.store(0, Ordering::Relaxed);
            return None;
        }

        // Clear path: descend one tier at a time. The tier we'd step down TO.
        let next_down = match current {
            BreakerTier::Hot => BreakerTier::Warm,
            BreakerTier::Warm => BreakerTier::Calm,
            BreakerTier::Calm => unreachable!("handled above"),
        };

        // Clear condition: free fraction recovered above the clear watermark
        // AND the pressure target is no hotter than where we'd land. (If the
        // target is still at/above `current`, the trip branch already handled
        // it; here we require it to be <= next_down so we don't descend into a
        // tier the signals would immediately re-trip out of.)
        let ff = if free_frac.is_nan() {
            0.0
        } else {
            free_frac
        };
        let clear_ok = ff >= self.config.clear_low && target <= next_down;

        if clear_ok {
            let streak = self.clear_streak.fetch_add(1, Ordering::Relaxed) + 1;
            if streak >= self.config.clear_debounce_ticks as u64 {
                self.clear_streak.store(0, Ordering::Relaxed);
                return self.set_tier(next_down);
            }
            None
        } else {
            // Clear condition broke — reset the debounce.
            self.clear_streak.store(0, Ordering::Relaxed);
            None
        }
    }

    /// Store `new` as the tier and bump the epoch IFF it differs from the
    /// current tier. Returns `Some(new)` on a change, `None` otherwise.
    fn set_tier(&self, new: BreakerTier) -> Option<BreakerTier> {
        let prev = self.tier();
        if prev == new {
            return None;
        }
        self.tier.store(tier_to_u8(new), Ordering::Relaxed);
        self.epoch.fetch_add(1, Ordering::Relaxed);
        Some(new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BreakerConfig {
        BreakerConfig {
            warm_high: 0.5,
            hot_high: 0.15,
            clear_low: 0.7,
            queue_depth_warm: 0,
            queue_depth_hot: 0,
            clear_debounce_ticks: 3,
        }
    }

    fn breaker() -> CircuitBreaker {
        CircuitBreaker::new(cfg(), 0)
    }

    #[test]
    fn starts_calm() {
        let b = breaker();
        assert_eq!(b.tier(), BreakerTier::Calm);
        assert_eq!(b.epoch(), 0);
    }

    #[test]
    fn boot_epoch_seed_is_honored() {
        let b = CircuitBreaker::new(cfg(), 12_345);
        assert_eq!(b.epoch(), 12_345);
        // First real change increments from the seed, never resets to 0.
        assert_eq!(b.evaluate(0.4, 0), Some(BreakerTier::Warm));
        assert_eq!(b.epoch(), 12_346);
    }

    #[test]
    fn trip_to_warm_is_immediate_over_warm_high() {
        let b = breaker();
        // free_frac just at the warm watermark trips (<=, inclusive).
        assert_eq!(b.evaluate(0.5, 0), Some(BreakerTier::Warm));
        assert_eq!(b.tier(), BreakerTier::Warm);
        assert_eq!(b.epoch(), 1);
    }

    #[test]
    fn trip_to_hot_is_immediate_over_hot_high() {
        let b = breaker();
        // Deep saturation jumps straight to HOT in one tick (skips WARM).
        assert_eq!(b.evaluate(0.1, 0), Some(BreakerTier::Hot));
        assert_eq!(b.tier(), BreakerTier::Hot);
        assert_eq!(b.epoch(), 1);
    }

    #[test]
    fn calm_holds_above_warm_high() {
        let b = breaker();
        // Plenty of free capacity: no change, no epoch bump.
        assert_eq!(b.evaluate(0.9, 0), None);
        assert_eq!(b.tier(), BreakerTier::Calm);
        assert_eq!(b.epoch(), 0);
    }

    #[test]
    fn clear_is_debounced_n_ticks() {
        let b = breaker();
        // Trip to WARM.
        assert_eq!(b.evaluate(0.4, 0), Some(BreakerTier::Warm));
        // Recovered above clear_low, but must hold debounce ticks (3).
        assert_eq!(b.evaluate(0.8, 0), None); // streak 1
        assert_eq!(b.evaluate(0.8, 0), None); // streak 2
        assert_eq!(b.tier(), BreakerTier::Warm);
        assert_eq!(b.evaluate(0.8, 0), Some(BreakerTier::Calm)); // streak 3 → descend
        assert_eq!(b.tier(), BreakerTier::Calm);
    }

    #[test]
    fn clear_descends_one_tier_at_a_time() {
        let b = breaker();
        // Trip straight to HOT.
        assert_eq!(b.evaluate(0.05, 0), Some(BreakerTier::Hot));
        // Recover: HOT→WARM after debounce.
        assert_eq!(b.evaluate(0.9, 0), None);
        assert_eq!(b.evaluate(0.9, 0), None);
        assert_eq!(b.evaluate(0.9, 0), Some(BreakerTier::Warm));
        assert_eq!(b.tier(), BreakerTier::Warm);
        // Then WARM→CALM after another full debounce window.
        assert_eq!(b.evaluate(0.9, 0), None);
        assert_eq!(b.evaluate(0.9, 0), None);
        assert_eq!(b.evaluate(0.9, 0), Some(BreakerTier::Calm));
        assert_eq!(b.tier(), BreakerTier::Calm);
    }

    #[test]
    fn debounce_resets_when_clear_condition_breaks() {
        let b = breaker();
        assert_eq!(b.evaluate(0.4, 0), Some(BreakerTier::Warm));
        assert_eq!(b.evaluate(0.8, 0), None); // streak 1
        // Drop back below clear_low (but still above warm_high so no re-trip):
        // breaks the streak.
        assert_eq!(b.evaluate(0.6, 0), None); // streak reset
        assert_eq!(b.tier(), BreakerTier::Warm);
        // Now a fresh full window is required.
        assert_eq!(b.evaluate(0.8, 0), None); // 1
        assert_eq!(b.evaluate(0.8, 0), None); // 2
        assert_eq!(b.evaluate(0.8, 0), Some(BreakerTier::Calm)); // 3 → descend
    }

    #[test]
    fn no_flap_in_the_watermark_gap() {
        // free_frac in (warm_high, clear_low) == (0.5, 0.7): neither trips
        // (not <= 0.5) nor clears (not >= 0.7). A breaker sitting in WARM
        // must stay WARM with no epoch churn across many ticks in the gap.
        let b = breaker();
        assert_eq!(b.evaluate(0.4, 0), Some(BreakerTier::Warm));
        let epoch_after_trip = b.epoch();
        for _ in 0..50 {
            assert_eq!(b.evaluate(0.6, 0), None, "gap value must not change tier");
        }
        assert_eq!(b.tier(), BreakerTier::Warm);
        assert_eq!(b.epoch(), epoch_after_trip, "no epoch churn in the gap");
    }

    #[test]
    fn re_trip_during_recovery_cancels_descent() {
        let b = breaker();
        assert_eq!(b.evaluate(0.05, 0), Some(BreakerTier::Hot)); // HOT
        // Two clear ticks toward HOT→WARM...
        assert_eq!(b.evaluate(0.9, 0), None);
        assert_eq!(b.evaluate(0.9, 0), None);
        // ...then pressure spikes again: still HOT (target == current, no
        // change), streak reset.
        assert_eq!(b.evaluate(0.05, 0), None);
        assert_eq!(b.tier(), BreakerTier::Hot);
        // The earlier streak is gone: a full fresh window is needed.
        assert_eq!(b.evaluate(0.9, 0), None);
        assert_eq!(b.evaluate(0.9, 0), None);
        assert_eq!(b.evaluate(0.9, 0), Some(BreakerTier::Warm));
    }

    #[test]
    fn epoch_is_monotonic_and_only_bumps_on_change() {
        let b = breaker();
        let mut last = b.epoch();
        // A long mixed sequence; epoch must be non-decreasing and increment
        // by exactly 1 on each reported change, and not at all otherwise.
        let seq = [0.9, 0.4, 0.4, 0.05, 0.05, 0.9, 0.9, 0.9, 0.6, 0.9, 0.9, 0.9];
        for &ff in &seq {
            let changed = b.evaluate(ff, 0).is_some();
            let now = b.epoch();
            assert!(now >= last, "epoch must be monotonic");
            if changed {
                assert_eq!(now, last + 1, "epoch bumps by exactly 1 on change");
            } else {
                assert_eq!(now, last, "epoch must not bump without a change");
            }
            last = now;
        }
    }

    #[test]
    fn queue_axis_trips_when_enabled() {
        let mut c = cfg();
        c.queue_depth_warm = 10;
        c.queue_depth_hot = 100;
        let b = CircuitBreaker::new(c, 0);
        // Capacity idle (free_frac 1.0) but queue depth crosses WARM.
        assert_eq!(b.evaluate(1.0, 10), Some(BreakerTier::Warm));
        // Queue depth crosses HOT.
        assert_eq!(b.evaluate(1.0, 100), Some(BreakerTier::Hot));
    }

    #[test]
    fn queue_axis_ignored_when_disabled() {
        // Both thresholds 0 (the disabled sentinel): a huge queue_depth must
        // NOT trip — only the capacity axis is live.
        let b = breaker();
        assert_eq!(b.evaluate(1.0, 1_000_000), None);
        assert_eq!(b.tier(), BreakerTier::Calm);
    }

    #[test]
    fn hotter_axis_wins() {
        let mut c = cfg();
        c.queue_depth_warm = 10;
        let b = CircuitBreaker::new(c, 0);
        // Capacity axis says HOT (free 0.1), queue axis says WARM: HOT wins.
        assert_eq!(b.evaluate(0.1, 20), Some(BreakerTier::Hot));
    }

    #[test]
    fn nan_free_frac_treated_as_saturated() {
        // A defensive guard: a 0/0 free fraction (caller forgot total==0)
        // must be treated as fully saturated and trip, never panic.
        let b = breaker();
        assert_eq!(b.evaluate(f64::NAN, 0), Some(BreakerTier::Hot));
    }
}
