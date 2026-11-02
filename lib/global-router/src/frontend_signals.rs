// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the Global Frontends themselves count: each one's requests in flight
//! per pool. This process's are exact ([`InFlightTracker`]). Least-in-flight
//! on the core path ranks pools by them, as the current selector does.
//!
//! A pick the core has made but not yet returned to its request is pending:
//! it counts as in flight from the pick until the request takes it over as
//! its own count ([`FrontendSignals::take_pending`]) or gives it back
//! ([`FrontendSignals::release_pending`]). The core runs one model's picks
//! one at a time, so each pick sees the ones before it.

use std::sync::Arc;

use dashmap::DashMap;
use dynamo_global_common::PoolId;

use crate::core::worker_id;
use crate::load::{InFlightGuard, InFlightTracker};

#[derive(Debug)]
pub struct FrontendSignals {
    own: Arc<InFlightTracker>,
    /// Picks not yet taken over by their requests, per pool, keyed by the
    /// pool's core worker id.
    pending: DashMap<u64, u64>,
}

impl FrontendSignals {
    /// `own` must be the tracker the Global Frontend counts its requests in.
    pub fn new(own: Arc<InFlightTracker>) -> Self {
        Self {
            own,
            pending: DashMap::new(),
        }
    }

    /// Counts one more of this process's requests in flight to `pool`
    /// until the guard drops.
    pub fn start(&self, pool: &PoolId) -> InFlightGuard {
        self.own.start(pool)
    }

    /// Requests in flight to `pool` from this replica, pending picks
    /// included.
    pub fn in_flight(&self, pool: &PoolId) -> u64 {
        self.own.get(pool) + self.pending_on(worker_id(pool))
    }

    /// Counts a pick of the pool that is core worker `worker` as pending.
    pub(crate) fn add_pending(&self, worker: u64) {
        *self.pending.entry(worker).or_default() += 1;
    }

    /// Gives back a pending pick of core worker `worker`.
    pub(crate) fn release_pending(&self, worker: u64) {
        if let dashmap::mapref::entry::Entry::Occupied(mut entry) = self.pending.entry(worker) {
            *entry.get_mut() -= 1;
            if *entry.get() == 0 {
                entry.remove();
            }
        }
    }

    /// Turns a pending pick of `pool` into the request's count in flight,
    /// which lasts until the guard drops. The count starts before the pick
    /// is released, so a reader may see the request twice for a moment but
    /// never not at all.
    pub(crate) fn take_pending(&self, pool: &PoolId) -> InFlightGuard {
        let guard = self.own.start(pool);
        self.release_pending(worker_id(pool));
        guard
    }

    /// Pending picks on every pool.
    #[cfg(test)]
    pub(crate) fn pending(&self) -> u64 {
        self.pending.iter().map(|entry| *entry.value()).sum()
    }

    fn pending_on(&self, worker: u64) -> u64 {
        self.pending.get(&worker).map_or(0, |count| *count)
    }
}

#[cfg(test)]
mod tests {
    use crate::selector::tests::pool;

    use super::*;

    #[test]
    fn in_flight_is_this_replica_s_count() {
        let own = Arc::new(InFlightTracker::new());
        let signals = FrontendSignals::new(own.clone());
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;
        let held = signals.start(&blue);
        let _other = own.start(&blue);
        assert_eq!(signals.in_flight(&blue), 2);
        drop(held);
        assert_eq!(signals.in_flight(&blue), 1);
    }

    #[test]
    fn a_pending_pick_counts_until_taken_over_or_released() {
        let own = Arc::new(InFlightTracker::new());
        let signals = FrontendSignals::new(own.clone());
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;

        signals.add_pending(worker_id(&blue));
        signals.add_pending(worker_id(&blue));
        assert_eq!(signals.in_flight(&blue), 2);

        let taken = signals.take_pending(&blue);
        assert_eq!((signals.in_flight(&blue), own.get(&blue)), (2, 1));
        signals.release_pending(worker_id(&blue));
        assert_eq!((signals.in_flight(&blue), signals.pending()), (1, 0));
        drop(taken);
        assert_eq!(signals.in_flight(&blue), 0);

        // Releasing more than was added leaves nothing negative.
        signals.release_pending(worker_id(&blue));
        assert_eq!(signals.pending(), 0);
    }
}
