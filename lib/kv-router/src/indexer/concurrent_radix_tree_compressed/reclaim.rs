// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Eager reclamation of stale leaves.
//!
//! The design is inspired by the chain index in smg-project/smg #2814, which frees emptied
//! storage as soon as enough of it accumulates. This module applies the idea to CRTC's
//! linked nodes; it shares no code with SMG.
//!
//! - **Volume trigger.** Every event lane counts, in a lane-local [`ReclaimTally`], the
//!   blocks it links into the tree and the blocks it leaves without any holder. Tallies are
//!   folded into the shared estimates in [`ReclaimState`] in batches of
//!   [`TALLY_FLUSH_BLOCKS`] and when the lane goes idle, so no event pays a shared
//!   read-modify-write. Scheduling a sweep reads the two estimates: once dead blocks reach
//!   an eighth of the live ones (and at least the dead floor), a sweep is due, at most one
//!   in flight and no sooner than `max(min_gap, 10 x last sweep)` after the previous one,
//!   which caps the sweep duty cycle near 10% of one lane. The five-minute timer stays as a
//!   backstop. A sweep recounts both values exactly and overwrites the estimates.

#[cfg(any(test, feature = "bench"))]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crossbeam_utils::CachePadded;

use super::trigger::VolumeTrigger;

/// A lane flushes its tally once either delta reaches this many blocks.
pub(super) const TALLY_FLUSH_BLOCKS: i64 = 1024;
/// A sweep is never due for fewer dead blocks than this, by default.
pub const DEFAULT_DEAD_FLOOR: u64 = 65_536;
/// Minimum time between the end of one sweep and a volume-triggered next one, by default.
pub const DEFAULT_MIN_GAP: Duration = Duration::from_millis(100);

/// Settings for stale-leaf reclamation.
#[derive(Clone, Copy, Debug)]
pub struct ReclaimConfig {
    /// Schedule a sweep once dead volume calls for one; otherwise only the timer does.
    pub volume_sweep: bool,
    /// Dead blocks below which no volume-triggered sweep runs.
    pub dead_floor: u64,
    /// Minimum time between a sweep's end and the next volume-triggered sweep.
    pub min_gap: Duration,
}

impl Default for ReclaimConfig {
    fn default() -> Self {
        Self {
            volume_sweep: true,
            dead_floor: DEFAULT_DEAD_FLOOR,
            min_gap: DEFAULT_MIN_GAP,
        }
    }
}

impl ReclaimConfig {
    /// The behavior before volume sweeps, for A/B runs.
    #[cfg(any(test, feature = "bench"))]
    pub fn legacy() -> Self {
        Self {
            volume_sweep: false,
            ..Self::default()
        }
    }
}

/// One event lane's block-volume deltas since its last flush.
#[derive(Debug, Default)]
pub(super) struct ReclaimTally {
    /// Blocks left on nodes no rank covers any more.
    pub(super) dead: i64,
    /// Blocks linked into the tree.
    pub(super) linked: i64,
}

impl ReclaimTally {
    fn is_empty(&self) -> bool {
        self.dead == 0 && self.linked == 0
    }

    fn needs_flush(&self) -> bool {
        self.dead.abs() >= TALLY_FLUSH_BLOCKS || self.linked.abs() >= TALLY_FLUSH_BLOCKS
    }
}

/// What scheduled a sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SweepTrigger {
    Timer,
    Volume,
}

/// The tree's reclamation state: the shared volume trigger, its clock, and sweep stats.
pub(super) struct ReclaimState {
    /// Read on every enqueue, written on tally flushes and by sweeps.
    trigger: CachePadded<VolumeTrigger>,
    origin: Instant,
    /// Whether the sweep in flight was scheduled by the volume trigger.
    volume_scheduled: AtomicBool,
    #[cfg(any(test, feature = "bench"))]
    pub(super) stats: SweepStats,
}

fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

impl ReclaimState {
    pub(super) fn new(config: ReclaimConfig) -> Self {
        Self {
            trigger: CachePadded::new(VolumeTrigger::new(
                config.volume_sweep,
                config.dead_floor,
                micros(config.min_gap),
            )),
            origin: Instant::now(),
            volume_scheduled: AtomicBool::new(false),
            #[cfg(any(test, feature = "bench"))]
            stats: SweepStats::default(),
        }
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn configure(&self, config: ReclaimConfig) {
        self.trigger.configure(
            config.volume_sweep,
            config.dead_floor,
            micros(config.min_gap),
        );
    }

    fn now_us(&self) -> u64 {
        micros(self.origin.elapsed())
    }

    /// Folds `tally` into the shared estimates if either delta is large enough, or
    /// whenever it is non-empty with `force`.
    #[inline]
    pub(super) fn flush(&self, tally: &mut ReclaimTally, force: bool) {
        if tally.is_empty() || !(force || tally.needs_flush()) {
            return;
        }
        let ReclaimTally { dead, linked } = std::mem::take(tally);
        self.trigger.add(dead, linked);
    }

    /// `(dead, linked)` block estimates.
    #[cfg(any(test, feature = "bench"))]
    pub(super) fn estimates(&self) -> (u64, u64) {
        self.trigger.estimates()
    }

    /// Whether the dead volume calls for a sweep now. Reads only values that change on
    /// tally flushes and sweeps, so the check stays read-shared; it reads the clock only
    /// once the volume condition holds.
    #[inline]
    pub(super) fn volume_due(&self) -> bool {
        self.trigger.due(|| self.now_us())
    }

    pub(super) fn note_scheduled(&self, trigger: SweepTrigger) {
        self.volume_scheduled
            .store(trigger == SweepTrigger::Volume, Ordering::Relaxed);
    }

    /// Records a finished sweep: overwrites the estimates with its exact recount minus
    /// what it unlinked, and remembers when it ended and how long it took.
    pub(super) fn finish_sweep(&self, started: Instant, outcome: &SweepOutcome) {
        let duration_us = micros(started.elapsed());
        self.trigger.finish(
            self.now_us(),
            duration_us,
            outcome.linked.saturating_sub(outcome.reclaimed_blocks),
            outcome.dead.saturating_sub(outcome.reclaimed_blocks),
        );
        let volume = self.volume_scheduled.swap(false, Ordering::Relaxed);
        #[cfg(any(test, feature = "bench"))]
        self.stats.record(outcome, duration_us, volume);
        #[cfg(not(any(test, feature = "bench")))]
        let _ = volume;
    }
}

/// What one sweep found and reclaimed.
#[derive(Debug, Default)]
pub(super) struct SweepOutcome {
    /// Nodes visited below the root and the anchors.
    pub(super) nodes: u64,
    /// Blocks on those nodes, before unlinking.
    pub(super) linked: u64,
    /// Blocks on holder-less nodes among them, before unlinking.
    pub(super) dead: u64,
    pub(super) candidates: u64,
    pub(super) reclaimed_nodes: u64,
    pub(super) reclaimed_blocks: u64,
    /// Candidates left linked because something else held a reference.
    pub(super) skipped_held: u64,
    /// Candidates left linked because a writer held their gate or covered them again.
    pub(super) skipped_busy: u64,
}

/// Cumulative sweep counters for tests and benches.
#[cfg(any(test, feature = "bench"))]
#[derive(Default)]
pub(super) struct SweepStats {
    pub(super) sweeps_volume: AtomicU64,
    pub(super) sweeps_other: AtomicU64,
    pub(super) total_us: AtomicU64,
    pub(super) max_us: AtomicU64,
    pub(super) reclaimed_nodes: AtomicU64,
    pub(super) reclaimed_blocks: AtomicU64,
    pub(super) skipped_held: AtomicU64,
    pub(super) skipped_busy: AtomicU64,
    pub(super) last_nodes: AtomicU64,
    pub(super) last_linked: AtomicU64,
    pub(super) last_dead: AtomicU64,
}

#[cfg(any(test, feature = "bench"))]
impl SweepStats {
    fn record(&self, outcome: &SweepOutcome, duration_us: u64, volume: bool) {
        let counter = if volume {
            &self.sweeps_volume
        } else {
            &self.sweeps_other
        };
        counter.fetch_add(1, Ordering::Relaxed);
        self.total_us.fetch_add(duration_us, Ordering::Relaxed);
        self.max_us.fetch_max(duration_us, Ordering::Relaxed);
        self.reclaimed_nodes
            .fetch_add(outcome.reclaimed_nodes, Ordering::Relaxed);
        self.reclaimed_blocks
            .fetch_add(outcome.reclaimed_blocks, Ordering::Relaxed);
        self.skipped_held
            .fetch_add(outcome.skipped_held, Ordering::Relaxed);
        self.skipped_busy
            .fetch_add(outcome.skipped_busy, Ordering::Relaxed);
        self.last_nodes.store(outcome.nodes, Ordering::Relaxed);
        self.last_linked.store(outcome.linked, Ordering::Relaxed);
        self.last_dead.store(outcome.dead, Ordering::Relaxed);
    }
}
