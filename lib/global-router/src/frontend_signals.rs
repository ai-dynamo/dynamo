// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the Global Frontends themselves count: each one's requests in flight
//! per pool. This process's are exact ([`InFlightTracker`]); each peer
//! replica's arrive over replica sync (`InflightCount`) every interval and
//! count until they go stale. Least-in-flight on the core path ranks pools
//! by them, as the current selector ranks by this process's alone.
//!
//! A pick the core has made but not yet returned to its request is pending:
//! it counts as in flight from the pick until the request takes it over as
//! its own count ([`FrontendSignals::take_pending`]) or gives it back
//! ([`FrontendSignals::release_pending`]). The core runs one model's picks
//! one at a time, so each pick sees the ones before it.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dynamo_global_common::PoolId;
use parking_lot::{Mutex, RwLock};

use crate::core::worker_id;
use crate::load::{InFlightGuard, InFlightTracker};

#[derive(Debug)]
pub struct FrontendSignals {
    own: Arc<InFlightTracker>,
    /// Picks not yet taken over by their requests, per pool, keyed by the
    /// pool's core worker id.
    pending: DashMap<u64, u64>,
    /// Each peer replica's requests in flight per pool, as it last reported
    /// them, by the peer's process id.
    peers: Mutex<HashMap<u64, PeerReport>>,
    /// The peers' counts summed per pool, rebuilt on every report, so a
    /// pick reads them without summing.
    peer_totals: RwLock<PeerTotals>,
}

/// One peer's last report: requests in flight per pool, keyed by the pool's
/// core worker id.
#[derive(Debug)]
struct PeerReport {
    in_flight: HashMap<u64, u64>,
    expires: Instant,
}

#[derive(Debug, Default)]
struct PeerTotals {
    in_flight: Arc<HashMap<u64, u64>>,
    /// When the first of the summed reports goes stale; the totals are
    /// rebuilt without it then.
    expires: Option<Instant>,
}

impl FrontendSignals {
    /// `own` must be the tracker the Global Frontend counts its requests in.
    pub fn new(own: Arc<InFlightTracker>) -> Self {
        Self {
            own,
            pending: DashMap::new(),
            peers: Mutex::default(),
            peer_totals: RwLock::default(),
        }
    }

    /// This process's requests in flight to each pool it has sent to.
    pub fn local(&self) -> Vec<(PoolId, u64)> {
        self.own.snapshot()
    }

    /// Counts one more of this process's requests in flight to `pool`
    /// until the guard drops.
    pub fn start(&self, pool: &PoolId) -> InFlightGuard {
        self.own.start(pool)
    }

    /// Requests in flight to `pool` from every replica: this one's, pending
    /// picks included, plus each peer's last report (from [`Self::peers`]).
    pub fn in_flight(&self, pool: &PoolId) -> u64 {
        let worker = worker_id(pool);
        let peers = self.peers().get(&worker).copied().unwrap_or(0);
        self.own.get(pool) + self.pending_on(worker) + peers
    }
    /// Records a peer replica's requests in flight per pool (keyed by the
    /// pool's core worker id), replacing its previous report. The report
    /// counts until `fresh_for` has passed without a newer one, so a peer
    /// that leaves stops counting without being told.
    pub fn record_peer(&self, peer: u64, in_flight: HashMap<u64, u64>, fresh_for: Duration) {
        let now = Instant::now();
        let mut peers = self.peers.lock();
        peers.insert(
            peer,
            PeerReport {
                in_flight,
                expires: now + fresh_for,
            },
        );
        self.rebuild_totals(&mut peers, now);
    }

    /// The peers' requests in flight summed per pool, keyed by the pool's
    /// core worker id, without reports gone stale.
    pub fn peers(&self) -> Arc<HashMap<u64, u64>> {
        let now = Instant::now();
        {
            let totals = self.peer_totals.read();
            if totals.expires.is_none_or(|expires| now <= expires) {
                return totals.in_flight.clone();
            }
        }
        let mut peers = self.peers.lock();
        self.rebuild_totals(&mut peers, now);
        self.peer_totals.read().in_flight.clone()
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

    /// Drops stale reports from `peers` and sums the rest into the totals.
    fn rebuild_totals(&self, peers: &mut HashMap<u64, PeerReport>, now: Instant) {
        peers.retain(|_, report| now <= report.expires);
        let mut in_flight = HashMap::<u64, u64>::new();
        for report in peers.values() {
            for (worker, count) in &report.in_flight {
                *in_flight.entry(*worker).or_default() += count;
            }
        }
        *self.peer_totals.write() = PeerTotals {
            in_flight: Arc::new(in_flight),
            expires: peers.values().map(|report| report.expires).min(),
        };
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

    #[test]
    fn peers_sum_per_pool_and_a_newer_report_replaces_an_older() {
        let signals = FrontendSignals::new(Arc::new(InFlightTracker::new()));
        let fresh = Duration::from_secs(60);
        assert!(signals.peers().is_empty());

        signals.record_peer(1, HashMap::from([(10, 3), (20, 1)]), fresh);
        signals.record_peer(2, HashMap::from([(10, 2)]), fresh);
        assert_eq!(*signals.peers(), HashMap::from([(10, 5), (20, 1)]));

        // Peer 1's count on pool 10 fell to zero.
        signals.record_peer(1, HashMap::from([(10, 0), (20, 4)]), fresh);
        assert_eq!(*signals.peers(), HashMap::from([(10, 2), (20, 4)]));
    }

    #[test]
    fn a_peer_report_stops_counting_once_stale() {
        let signals = FrontendSignals::new(Arc::new(InFlightTracker::new()));
        let fresh = Duration::from_millis(50);
        signals.record_peer(1, HashMap::from([(10, 3)]), fresh);
        signals.record_peer(2, HashMap::from([(10, 2)]), Duration::from_secs(60));
        assert_eq!(signals.peers().get(&10), Some(&5));

        std::thread::sleep(fresh * 3);
        assert_eq!(signals.peers().get(&10), Some(&2));
    }

    #[test]
    fn in_flight_is_this_replica_s_plus_each_peer_s() {
        let own = Arc::new(InFlightTracker::new());
        let signals = FrontendSignals::new(own.clone());
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;
        let _held = own.start(&blue);
        assert_eq!(signals.in_flight(&blue), 1);
        assert_eq!(signals.local(), vec![(blue.clone(), 1)]);

        signals.record_peer(
            7,
            HashMap::from([(worker_id(&blue), 4)]),
            Duration::from_secs(60),
        );
        assert_eq!(signals.in_flight(&blue), 5);
    }
}
