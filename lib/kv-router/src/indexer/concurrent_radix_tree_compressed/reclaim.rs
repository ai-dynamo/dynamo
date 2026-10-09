// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Eager reclamation of stale leaves and capacity discipline for compressed edges.
//!
//! The design is inspired by the chain index in smg-project/smg #2814, which frees emptied
//! storage as soon as enough of it accumulates and keeps its arrays close to their length.
//! This module applies both ideas to CRTC's linked nodes; it shares no code with SMG.
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
//! - **Edge capacity.** A split prefix never grows again, so it is copied into an
//!   exact-size allocation; a leaf that appends past its capacity reserves at most
//!   `need / divisor` (at least `min_slack`) blocks of slack instead of doubling.

#[cfg(any(test, feature = "bench"))]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use crossbeam_utils::CachePadded;

use super::trigger::VolumeTrigger;
use crate::protocols::{ExternalSequenceBlockHash, LocalBlockHash};

/// A lane flushes its tally once either delta reaches this many blocks.
pub(super) const TALLY_FLUSH_BLOCKS: i64 = 1024;
/// A sweep is never due for fewer dead blocks than this, by default.
pub const DEFAULT_DEAD_FLOOR: u64 = 65_536;
/// Minimum time between the end of one sweep and a volume-triggered next one, by default.
pub const DEFAULT_MIN_GAP: Duration = Duration::from_millis(100);
/// A growing leaf reserves `need / DEFAULT_SLACK_DIVISOR` blocks of slack, by default.
pub const DEFAULT_SLACK_DIVISOR: u32 = 8;
/// A growing leaf reserves at least this many blocks of slack, by default.
pub const DEFAULT_LEAF_MIN_SLACK: u32 = 4;

/// Settings for stale-leaf reclamation and edge capacity.
#[derive(Clone, Copy, Debug)]
pub struct ReclaimConfig {
    /// Schedule a sweep once dead volume calls for one; otherwise only the timer does.
    pub volume_sweep: bool,
    /// Dead blocks below which no volume-triggered sweep runs.
    pub dead_floor: u64,
    /// Minimum time between a sweep's end and the next volume-triggered sweep.
    pub min_gap: Duration,
    /// Leaf append slack divisor; `0` keeps `Vec`'s doubling growth.
    pub leaf_slack_divisor: u32,
    /// Minimum leaf append slack in blocks.
    pub leaf_min_slack: u32,
    /// Copy split prefixes into exact-size allocations.
    pub exact_split_prefix: bool,
}

impl Default for ReclaimConfig {
    fn default() -> Self {
        Self {
            volume_sweep: true,
            dead_floor: DEFAULT_DEAD_FLOOR,
            min_gap: DEFAULT_MIN_GAP,
            leaf_slack_divisor: DEFAULT_SLACK_DIVISOR,
            leaf_min_slack: DEFAULT_LEAF_MIN_SLACK,
            exact_split_prefix: true,
        }
    }
}

impl ReclaimConfig {
    /// The behavior before volume sweeps and capacity discipline, for A/B runs.
    #[cfg(any(test, feature = "bench"))]
    pub fn legacy() -> Self {
        Self {
            volume_sweep: false,
            leaf_slack_divisor: 0,
            exact_split_prefix: false,
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
    pub(super) capacity: EdgeCapacity,
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
            capacity: EdgeCapacity::with(config),
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
        self.capacity.configure(config);
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

type EdgeEntry = (LocalBlockHash, ExternalSequenceBlockHash);

/// Capacity rules for compressed edges. Read only when an edge splits or outgrows its
/// allocation.
pub(super) struct EdgeCapacity {
    slack_divisor: AtomicU32,
    min_slack: AtomicU32,
    exact_split_prefix: AtomicBool,
}

impl Default for EdgeCapacity {
    fn default() -> Self {
        let capacity = Self {
            slack_divisor: AtomicU32::new(0),
            min_slack: AtomicU32::new(0),
            exact_split_prefix: AtomicBool::new(false),
        };
        capacity.configure(ReclaimConfig::default());
        capacity
    }
}

impl EdgeCapacity {
    fn configure(&self, config: ReclaimConfig) {
        self.slack_divisor
            .store(config.leaf_slack_divisor, Ordering::Relaxed);
        self.min_slack
            .store(config.leaf_min_slack, Ordering::Relaxed);
        self.exact_split_prefix
            .store(config.exact_split_prefix, Ordering::Relaxed);
    }

    pub(super) fn with(config: ReclaimConfig) -> Self {
        let capacity = Self::default();
        capacity.configure(config);
        capacity
    }

    /// Makes room for `additional` more blocks on a leaf edge. Past its capacity the edge
    /// reserves `need / divisor` blocks of slack, at least `min_slack`, so after any append
    /// `capacity <= len + max(len / divisor, min_slack)` before allocator rounding. A divisor
    /// of zero leaves growth to `Vec`.
    #[inline]
    pub(super) fn reserve_for_append(&self, edge: &mut Vec<EdgeEntry>, additional: usize) {
        let need = edge.len() + additional;
        if edge.capacity() >= need {
            return;
        }
        let divisor = self.slack_divisor.load(Ordering::Relaxed) as usize;
        if divisor == 0 {
            return;
        }
        let slack = (need / divisor).max(self.min_slack.load(Ordering::Relaxed) as usize);
        edge.reserve_exact(need + slack - edge.len());
    }

    /// Moves a split prefix into an exact-size allocation. A split prefix is internal and
    /// never grows again, so its leftover capacity would be slack for the node's lifetime.
    /// Copying instead of `shrink_to_fit` frees the slack on every allocator: an in-place
    /// shrinking `realloc` can keep the whole block.
    #[inline]
    pub(super) fn trim_split_prefix(&self, edge: &mut Vec<EdgeEntry>) {
        if edge.capacity() == edge.len() || !self.exact_split_prefix.load(Ordering::Relaxed) {
            return;
        }
        let mut exact = Vec::with_capacity(edge.len());
        exact.extend_from_slice(edge);
        *edge = exact;
    }
}
