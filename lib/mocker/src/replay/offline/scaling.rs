// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional scaling-policy component for offline replay.
//!
//! A [`ReplayScalingPolicy`] is invoked once per
//! [`ScalingTick`](super::events::SimulationEventKind::ScalingTick) inside the
//! runtime-owned event loop. Placement remains a separate, monomorphized
//! component; this trait only observes a settled snapshot and returns desired
//! non-draining capacity.

use std::collections::BTreeMap;

use super::components::TrafficStats;
use crate::common::protocols::ForwardPassSnapshot;

const IDLE_FPM_INTERVAL_MS: f64 = 1_000.0;

/// Latest FPM snapshot for each logical worker and DP rank.
#[derive(Debug, Default)]
pub(super) struct LatestFpmBuffer {
    snapshots: BTreeMap<(usize, u32), ForwardPassSnapshot>,
    last_publish_ms: BTreeMap<(usize, u32), f64>,
}

impl LatestFpmBuffer {
    pub(super) fn activate_worker(&mut self, worker_id: usize, dp_size: u32, now_ms: f64) {
        for dp_rank in 0..dp_size.max(1) {
            self.last_publish_ms
                .entry((worker_id, dp_rank))
                .or_insert(now_ms);
        }
    }

    pub(super) fn insert(&mut self, worker_id: usize, snapshot: ForwardPassSnapshot, now_ms: f64) {
        let key = (worker_id, snapshot.dp_rank);
        self.snapshots.insert(key, snapshot);
        self.last_publish_ms.insert(key, now_ms);
    }

    pub(super) fn emit_idle_due(&mut self, active_worker_ids: &[usize], dp_size: u32, now_ms: f64) {
        debug_assert!(active_worker_ids.is_sorted());
        let dp_size = dp_size.max(1);
        let is_active = |(worker_id, dp_rank): &(usize, u32)| {
            *dp_rank < dp_size && active_worker_ids.binary_search(worker_id).is_ok()
        };
        self.snapshots.retain(|key, _| is_active(key));
        self.last_publish_ms.retain(|key, _| is_active(key));

        for &worker_id in active_worker_ids {
            for dp_rank in 0..dp_size {
                let key = (worker_id, dp_rank);
                let last_publish_ms = self.last_publish_ms.entry(key).or_insert(now_ms);
                let elapsed_ms = now_ms - *last_publish_ms;
                if elapsed_ms >= IDLE_FPM_INTERVAL_MS {
                    *last_publish_ms +=
                        (elapsed_ms / IDLE_FPM_INTERVAL_MS).floor() * IDLE_FPM_INTERVAL_MS;
                    self.snapshots.insert(
                        key,
                        ForwardPassSnapshot {
                            dp_rank,
                            ..Default::default()
                        },
                    );
                }
            }
        }
    }

    pub(super) fn take(&mut self) -> Vec<(usize, ForwardPassSnapshot)> {
        std::mem::take(&mut self.snapshots)
            .into_iter()
            .map(|((worker_id, _), snapshot)| (worker_id, snapshot))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::LatestFpmBuffer;
    use crate::common::protocols::ForwardPassSnapshot;

    #[test]
    fn latest_fpm_buffer_coalesces_each_worker_rank() {
        let mut buffer = LatestFpmBuffer::default();
        buffer.insert(
            4,
            ForwardPassSnapshot {
                dp_rank: 0,
                wall_time_secs: 1.0,
                ..Default::default()
            },
            100.0,
        );
        buffer.insert(
            4,
            ForwardPassSnapshot {
                dp_rank: 0,
                wall_time_secs: 2.0,
                ..Default::default()
            },
            200.0,
        );
        buffer.insert(
            4,
            ForwardPassSnapshot {
                dp_rank: 1,
                wall_time_secs: 3.0,
                ..Default::default()
            },
            300.0,
        );
        buffer.insert(
            5,
            ForwardPassSnapshot {
                dp_rank: 0,
                wall_time_secs: 4.0,
                ..Default::default()
            },
            400.0,
        );

        let snapshots = buffer.take();

        assert_eq!(snapshots.len(), 3);
        assert_eq!(snapshots[0].0, 4);
        assert_eq!(snapshots[0].1.dp_rank, 0);
        assert_eq!(snapshots[0].1.wall_time_secs, 2.0);
        assert_eq!(snapshots[1].0, 4);
        assert_eq!(snapshots[1].1.dp_rank, 1);
        assert_eq!(snapshots[1].1.wall_time_secs, 3.0);
        assert_eq!(snapshots[2].0, 5);
        assert_eq!(snapshots[2].1.dp_rank, 0);
        assert_eq!(snapshots[2].1.wall_time_secs, 4.0);
        assert!(buffer.take().is_empty());
    }

    #[test]
    fn latest_fpm_buffer_emits_idle_on_simulated_cadence() {
        let mut buffer = LatestFpmBuffer::default();
        buffer.activate_worker(4, 2, 100.0);

        buffer.emit_idle_due(&[4], 2, 1_099.0);
        assert!(buffer.take().is_empty());

        buffer.emit_idle_due(&[4], 2, 1_100.0);
        let snapshots = buffer.take();
        assert_eq!(snapshots.len(), 2);
        assert!(
            snapshots
                .iter()
                .all(|(worker_id, snapshot)| { *worker_id == 4 && snapshot.wall_time_secs == 0.0 })
        );

        buffer.insert(
            4,
            ForwardPassSnapshot {
                dp_rank: 1,
                wall_time_secs: 0.25,
                ..Default::default()
            },
            1_500.0,
        );
        buffer.emit_idle_due(&[4], 2, 2_100.0);
        let snapshots = buffer.take();
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0].1.dp_rank, 0);
        assert_eq!(snapshots[0].1.wall_time_secs, 0.0);
        assert_eq!(snapshots[1].1.dp_rank, 1);
        assert_eq!(snapshots[1].1.wall_time_secs, 0.25);

        let mut delayed_observation = LatestFpmBuffer::default();
        delayed_observation.activate_worker(8, 1, 0.0);
        delayed_observation.emit_idle_due(&[8], 1, 1_500.0);
        assert_eq!(delayed_observation.take().len(), 1);
        delayed_observation.emit_idle_due(&[8], 1, 2_400.0);
        assert_eq!(delayed_observation.take().len(), 1);

        delayed_observation.insert(
            8,
            ForwardPassSnapshot {
                wall_time_secs: 0.1,
                ..Default::default()
            },
            3_000.0,
        );
        delayed_observation.emit_idle_due(&[], 1, 3_500.0);
        assert!(delayed_observation.take().is_empty());
        delayed_observation.emit_idle_due(&[8], 1, 4_000.0);
        assert!(delayed_observation.take().is_empty());
    }
}

/// Snapshot handed to the scaling policy at one tick. The runtime has already advanced the clock to
/// `now_ms` and settled all same-timestamp work before this is built, so the planner observes a
/// consistent post-settlement snapshot (matching the old advance-then-tick ordering).
#[derive(Debug)]
pub struct ReplayScalingSnapshot {
    /// Simulated clock at this tick (equals the scheduled tick time).
    pub now_ms: f64,
    /// Latest prefill FPM snapshot per worker/rank observed since the previous tick (empty in
    /// agg mode, which routes all passes through `decode_fpm`).
    pub prefill_fpm: Vec<(usize, ForwardPassSnapshot)>,
    /// Latest decode (agg: aggregated) FPM snapshot per worker/rank observed since the previous
    /// tick.
    pub decode_fpm: Vec<(usize, ForwardPassSnapshot)>,
    /// Traffic stats over `[previous tick, now]`. Drained every tick; the Python side merges
    /// partial windows across ticks that don't consume traffic.
    pub traffic: TrafficStats,
    /// Active (ready, non-draining) logical worker IDs. Agg reports decode only.
    pub active_prefill_ids: Vec<usize>,
    pub active_decode_ids: Vec<usize>,
    /// Starting workers contribute to desired non-draining capacity.
    pub starting_prefill_ids: Vec<usize>,
    pub starting_decode_ids: Vec<usize>,
    /// Draining workers are irrevocably leaving and do not contribute to target capacity.
    pub draining_prefill_ids: Vec<usize>,
    pub draining_decode_ids: Vec<usize>,
}

/// Scaling decision for one tick. A target is desired non-draining capacity
/// (`active + starting`); a `None` target leaves that capacity unchanged.
/// `next_tick_ms = None` stops the recurring tick (the sim then runs to natural completion).
#[derive(Debug, Default, Clone)]
pub struct ReplayScalingDecision {
    /// Target prefill replica count (agg ignores this).
    pub target_prefill: Option<usize>,
    /// Target decode (agg: total) replica count.
    pub target_decode: Option<usize>,
    /// Absolute simulated time (ms) of the next tick. `None` => do not re-arm.
    pub next_tick_ms: Option<f64>,
}

/// Implemented by a replay scaling component. One call per `ScalingTick`.
///
/// Intentionally NOT `Send`: the simulation runs single-threaded and the PyO3 implementation
/// holds Python state, so the runtime loop holds the GIL rather than crossing threads.
pub trait ReplayScalingPolicy {
    /// First tick time (ms). A non-finite value disables scaling ticks.
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64>;

    /// Drive one tick: decide scaling targets and the next tick time.
    fn on_tick(&mut self, snapshot: ReplayScalingSnapshot)
    -> anyhow::Result<ReplayScalingDecision>;
}
