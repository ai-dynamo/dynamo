// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Processor-sharing bandwidth model for a single-link simulation.
//!
//! `N` active transfers share the link equally: each progresses at `B/N`
//! where `B` is link throughput. When one completes, the remaining `N-1`
//! speed up to `B/(N-1)`. This is a deterministic approximation of link
//! contention for replay timing, not a claim that the real transport
//! scheduler is exactly processor-sharing.
//!
//! ## Id uniqueness
//!
//! Bandwidth sharing models do not mint ids themselves. A [`TransferId`]
//! counter is handed in at construction — when two models share a single
//! counter (e.g. one for offload, one for onboard), ids are globally unique
//! across both links. That is required whenever ids coexist in a shared map:
//! different models dispensing overlapping ids would cause completion signals
//! to cross-fire into the wrong pending transfer.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Opaque handle returned by [`BandwidthSharingModel::start_transfer`]. Callers
/// correlate completions returned by [`advance_to`](BandwidthSharingModel::advance_to)
/// with their own notification state via this id.
pub type TransferId = u64;

/// Processor-sharing (PS) queue for a bandwidth-bounded link.
///
/// ## Contract
/// Callers must drive state by calling
/// [`advance_to`](Self::advance_to) before
/// [`start_transfer`](Self::start_transfer) when time has moved forward while transfers
/// are active.
#[derive(Debug, Clone)]
pub struct BandwidthSharingModel {
    /// Throughput of the link in bytes/ms. `f64::INFINITY` means unbounded
    /// (all transfers complete instantly).
    bandwidth_bytes_per_ms: f64,
    /// Active transfers. Order is not significant; we do linear scans.
    active: Vec<ActiveTransfer>,
    /// Simulation time of the last state update. Monotonically
    /// non-decreasing.
    last_update_ms: f64,
    /// Source of [`TransferId`]s. When shared across models (via
    /// `Arc` clone), ids are globally unique across all of them.
    id_counter: Arc<AtomicU64>,
}

#[derive(Debug, Clone)]
struct ActiveTransfer {
    id: TransferId,
    /// Bytes still to transmit at `BandwidthSharingModel::last_update_ms`.
    remaining_bytes: f64,
}

impl BandwidthSharingModel {
    /// Build a bandwidth sharing model for a link with throughput `gbps` GB/s
    /// (1e9 bytes/s). Non-positive `gbps` means "no bandwidth cap" —
    /// transfers complete instantly.
    ///
    /// Pass an `Arc` clone of the same counter to every model whose
    /// [`TransferId`]s can coexist in shared maps.
    pub fn new(gbps: f64, id_counter: Arc<AtomicU64>) -> Self {
        let bandwidth_bytes_per_ms = if gbps > 0.0 {
            gbps * 1e6
        } else {
            f64::INFINITY
        };
        Self {
            bandwidth_bytes_per_ms,
            active: Vec::new(),
            last_update_ms: 0.0,
            id_counter,
        }
    }

    /// Start a new transfer of `bytes` bytes arriving at `now_ms`. Adds
    /// it to the active set so subsequent arrivals share bandwidth with it.
    /// Returns a [`TransferId`] the caller can match against completions
    /// returned by [`advance_to`](Self::advance_to).
    ///
    /// `now_ms` is clamped up to `last_update_ms` so the model's
    /// simulation time stays monotonic — trace replay's `current_time_ms`
    /// can rewind between passes (jumping to the next arrival timestamp),
    /// but an in-flight transfer can't "un-copy" bytes.
    pub fn start_transfer(&mut self, now_ms: f64, bytes: usize) -> TransferId {
        let now_ms = now_ms.max(self.last_update_ms);
        self.last_update_ms = now_ms;
        let id = self.id_counter.fetch_add(1, Ordering::Relaxed);
        let remaining = if self.bandwidth_bytes_per_ms.is_finite() {
            bytes as f64
        } else {
            // Infinite bandwidth: transfer has zero remaining; it will
            // drain on the next advance_to at any time >= now_ms.
            0.0
        };
        self.active.push(ActiveTransfer {
            id,
            remaining_bytes: remaining,
        });
        id
    }

    /// Advance simulation to `now_ms` under processor-sharing semantics.
    /// Returns IDs of all transfers that completed during
    /// `[last_update_ms, now_ms]`, in completion order.
    ///
    /// When `now_ms < last_update_ms` (trace replay can rewind
    /// `current_time_ms` to the next arrival timestamp, which may lag the
    /// prior pass's `end_ms`), the call is a no-op — the model stays
    /// at `last_update_ms`. Physical bytes already copied can't un-copy.
    pub fn advance_to(&mut self, now_ms: f64) -> Vec<TransferId> {
        let mut completed = Vec::new();
        if now_ms < self.last_update_ms {
            return completed;
        }

        loop {
            if self.active.is_empty() {
                self.last_update_ms = now_ms;
                break;
            }
            let n = self.active.len() as f64;
            let rate_per = self.bandwidth_bytes_per_ms / n;

            // Find the active transfer with smallest remaining bytes — it
            // is the next one to hit zero under PS.
            let (min_idx, min_remaining) = self
                .active
                .iter()
                .enumerate()
                .map(|(i, t)| (i, t.remaining_bytes))
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("active is non-empty");

            // Time until that transfer completes at the current PS rate.
            let time_to_next_completion = if rate_per.is_finite() && rate_per > 0.0 {
                min_remaining / rate_per
            } else {
                // Infinite bandwidth → every active transfer completes now.
                0.0
            };
            let next_event_ms = self.last_update_ms + time_to_next_completion;

            if next_event_ms > now_ms + 1e-9 {
                // Not enough elapsed time for the next completion. Deduct
                // proportional progress from every active transfer and stop.
                let elapsed = now_ms - self.last_update_ms;
                if rate_per.is_finite() && elapsed > 0.0 {
                    let delta_per = elapsed * rate_per;
                    for t in &mut self.active {
                        t.remaining_bytes = (t.remaining_bytes - delta_per).max(0.0);
                    }
                }
                self.last_update_ms = now_ms;
                break;
            }

            // Advance to next_event_ms: transfer at min_idx hits zero and
            // completes. All other active transfers received the same
            // proportional work.
            if rate_per.is_finite() && time_to_next_completion > 0.0 {
                let delta_per = time_to_next_completion * rate_per;
                for t in &mut self.active {
                    t.remaining_bytes = (t.remaining_bytes - delta_per).max(0.0);
                }
            }
            self.last_update_ms = next_event_ms;
            completed.push(self.active.swap_remove(min_idx).id);
            // Loop to check if the next-to-complete now also finishes before
            // now_ms (e.g. multiple transfers at identical remaining).
        }

        completed
    }

    /// Time at which the next completion will fire if no further arrivals
    /// happen. `None` if no transfers are active. Used by the scheduler's
    /// stall-advance to know when to wake up next.
    pub fn earliest_finish(&self) -> Option<f64> {
        if self.active.is_empty() {
            return None;
        }
        let n = self.active.len() as f64;
        let rate_per = self.bandwidth_bytes_per_ms / n;
        let min_remaining = self
            .active
            .iter()
            .map(|t| t.remaining_bytes)
            .fold(f64::INFINITY, f64::min);
        let delta = if rate_per.is_finite() && rate_per > 0.0 {
            min_remaining / rate_per
        } else {
            0.0
        };
        Some(self.last_update_ms + delta)
    }

    /// Number of currently active (in-flight) transfers.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// True if no transfers are currently in flight.
    pub fn is_idle(&self) -> bool {
        self.active.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn make_model(gbps: f64) -> BandwidthSharingModel {
        BandwidthSharingModel::new(gbps, Arc::new(AtomicU64::new(0)))
    }

    fn approx_eq(a: f64, b: f64) {
        assert!(
            (a - b).abs() < EPS,
            "expected {a} ≈ {b} (diff {})",
            (a - b).abs()
        );
    }

    #[test]
    fn single_transfer_finishes_at_duration() {
        // 1 GB/s link, 1 MB transfer under PS with N=1 → 1 ms finish.
        let mut model = make_model(1.0);
        let id = model.start_transfer(0.0, 1_000_000);
        approx_eq(model.earliest_finish().unwrap(), 1.0);
        assert!(model.advance_to(0.5).is_empty());
        approx_eq(model.earliest_finish().unwrap(), 1.0);
        let completed = model.advance_to(1.0);
        assert_eq!(completed, vec![id]);
        assert!(model.is_idle());
    }

    #[test]
    fn two_concurrent_share_bandwidth_under_ps() {
        // Two transfers arriving together on the same link must each
        // receive half the bandwidth and both complete at 2x the
        // single-transfer duration.
        let mut model = make_model(1.0);
        let id1 = model.start_transfer(0.0, 1_000_000);
        let id2 = model.start_transfer(0.0, 1_000_000);
        // Rate/each = 0.5 MB/ms, so each needs 2 ms to finish.
        approx_eq(model.earliest_finish().unwrap(), 2.0);
        assert!(model.advance_to(1.99).is_empty());
        let completed = model.advance_to(2.0);
        assert_eq!(completed.len(), 2);
        assert!(completed.contains(&id1));
        assert!(completed.contains(&id2));
        assert!(model.is_idle());
    }

    #[test]
    fn staggered_arrivals_follow_ps_semantics() {
        // T1 arrives at 0, 1 MB on 1 GB/s link.
        // T2 arrives at 0.5 ms — by then T1 has 0.5 MB remaining.
        // N=2 → each gets 0.5 MB/ms. T1 (0.5 MB) finishes at 0.5 + 1 = 1.5.
        // After T1 finishes, T2 remaining = 0.5 MB with N=1 → 0.5 ms to go.
        // So T2 finishes at 1.5 + 0.5 = 2.0.
        // Total throughput: 2 MB over 2 ms on a 1 GB/s link — correct.
        let mut model = make_model(1.0);
        let id1 = model.start_transfer(0.0, 1_000_000);
        assert!(model.advance_to(0.5).is_empty());
        let id2 = model.start_transfer(0.5, 1_000_000);

        // At now_ms = 1.5, T1 finishes but T2 does not.
        let c1 = model.advance_to(1.5);
        assert_eq!(c1, vec![id1]);
        assert_eq!(model.active_count(), 1);

        // T2 finishes at 2.0.
        let c2 = model.advance_to(2.0);
        assert_eq!(c2, vec![id2]);
        assert!(model.is_idle());
    }

    #[test]
    fn staggered_later_arrival_inherits_remaining_bandwidth() {
        // T1 arrives at 0 and finishes alone at 1.0 ms (rate 1 MB/ms).
        // T2 arrives at 5 ms (long after T1 done). T2 should finish at 6 ms.
        let mut model = make_model(1.0);
        let _id1 = model.start_transfer(0.0, 1_000_000);
        let c1 = model.advance_to(1.0);
        assert_eq!(c1.len(), 1);
        assert!(model.is_idle());

        // Scheduler idles to 5 ms then admits T2.
        assert!(model.advance_to(5.0).is_empty());
        let id2 = model.start_transfer(5.0, 1_000_000);
        approx_eq(model.earliest_finish().unwrap(), 6.0);
        let c2 = model.advance_to(6.0);
        assert_eq!(c2, vec![id2]);
    }

    #[test]
    fn zero_bandwidth_is_infinite_throughput() {
        // Non-positive bandwidth → transfers complete instantly on the
        // next advance_to, regardless of byte count.
        let mut model = make_model(0.0);
        let id = model.start_transfer(10.0, 1_000_000_000);
        approx_eq(model.earliest_finish().unwrap(), 10.0);
        let completed = model.advance_to(10.0);
        assert_eq!(completed, vec![id]);
    }

    #[test]
    fn earliest_finish_is_none_when_idle() {
        let model = make_model(1.0);
        assert!(model.is_idle());
        assert!(model.earliest_finish().is_none());
    }

    #[test]
    fn shared_counter_yields_globally_unique_ids_across_models() {
        // Two models sharing a single `Arc<AtomicU64>` counter must
        // never hand out the same TransferId. This is the invariant
        // `TransferState` relies on when keying `awaiters` and
        // `swap_in_flags` by TransferId — a collision would cause
        // completion signals to cross-fire into the wrong transfer.
        let counter = Arc::new(AtomicU64::new(0));
        let mut a = BandwidthSharingModel::new(1.0, counter.clone());
        let mut b = BandwidthSharingModel::new(1.0, counter);
        let mut ids = Vec::new();
        for _ in 0..5 {
            ids.push(a.start_transfer(0.0, 1));
            ids.push(b.start_transfer(0.0, 1));
        }
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 10, "shared counter must yield 10 distinct ids");
    }

    #[test]
    fn many_simultaneous_arrivals_all_complete_at_nx() {
        // Stress: 10 transfers arriving together each get 1/10 bandwidth.
        // All finish at 10 * per-transfer duration.
        let mut model = make_model(1.0);
        for _ in 0..10 {
            let _id = model.start_transfer(0.0, 1_000_000);
        }
        approx_eq(model.earliest_finish().unwrap(), 10.0);
        let completed = model.advance_to(10.0);
        assert_eq!(completed.len(), 10);
        assert!(model.is_idle());
    }
}
