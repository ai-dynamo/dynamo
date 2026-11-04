// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Frontends' requests in flight per pool: this process's, plus
//! picks the core has made that no request has taken over yet, plus each peer
//! replica's last report over replica sync until it goes stale.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use dynamo_global_common::PoolId;
use dynamo_kv_router::WorkerId;
use parking_lot::{Mutex, RwLock};

use crate::load::{InFlightGuard, InFlightTracker};

#[derive(Debug)]
pub(crate) struct FrontendSignals {
    own: Arc<InFlightTracker>,
    /// Pending picks by worker id.
    pending: DashMap<WorkerId, u64>,
    /// Each peer's last report, by the peer's process id.
    peers: Mutex<HashMap<u64, PeerReport>>,
    /// The peers' counts summed per worker, rebuilt on every report.
    peer_totals: RwLock<PeerTotals>,
}

#[derive(Debug)]
struct PeerReport {
    in_flight: HashMap<WorkerId, u64>,
    expires: Instant,
}

#[derive(Debug, Default)]
struct PeerTotals {
    in_flight: Arc<HashMap<WorkerId, u64>>,
    /// When the first summed report goes stale.
    expires: Option<Instant>,
}

impl FrontendSignals {
    /// `own` must be the tracker the Global Frontend counts its requests in.
    pub(crate) fn new(own: Arc<InFlightTracker>) -> Self {
        Self {
            own,
            pending: DashMap::new(),
            peers: Mutex::default(),
            peer_totals: RwLock::default(),
        }
    }

    /// This process's requests in flight to each pool it has sent to.
    pub(crate) fn local(&self) -> Vec<(PoolId, u64)> {
        self.own.snapshot()
    }

    pub(crate) fn start(&self, pool: &PoolId) -> InFlightGuard {
        self.own.start(pool)
    }

    /// Requests in flight to `pool` (worker `worker`) from every replica,
    /// pending picks included.
    pub(crate) fn in_flight(&self, pool: &PoolId, worker: WorkerId) -> u64 {
        let pending = self.pending.get(&worker).map_or(0, |count| *count);
        let peers = self.peers().get(&worker).copied().unwrap_or(0);
        self.own.get(pool) + pending + peers
    }

    /// Replaces `peer`'s report, which counts until `fresh_for` passes
    /// without a newer one.
    pub(crate) fn record_peer(
        &self,
        peer: u64,
        in_flight: HashMap<WorkerId, u64>,
        fresh_for: Duration,
    ) {
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

    /// The fresh peer reports summed per worker.
    pub(crate) fn peers(&self) -> Arc<HashMap<WorkerId, u64>> {
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

    pub(crate) fn add_pending(&self, worker: WorkerId) {
        *self.pending.entry(worker).or_default() += 1;
    }

    pub(crate) fn release_pending(&self, worker: WorkerId) {
        if let Entry::Occupied(mut entry) = self.pending.entry(worker) {
            *entry.get_mut() -= 1;
            if *entry.get() == 0 {
                entry.remove();
            }
        }
    }

    /// Turns a pending pick into the request's count. Counting first means a
    /// reader may briefly see it twice, never not at all.
    pub(crate) fn take_pending(&self, pool: &PoolId, worker: WorkerId) -> InFlightGuard {
        let guard = self.own.start(pool);
        self.release_pending(worker);
        guard
    }

    #[cfg(test)]
    pub(crate) fn pending(&self) -> u64 {
        self.pending.iter().map(|entry| *entry.value()).sum()
    }

    fn rebuild_totals(&self, peers: &mut HashMap<u64, PeerReport>, now: Instant) {
        peers.retain(|_, report| now <= report.expires);
        let mut in_flight = HashMap::<WorkerId, u64>::new();
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
    use super::*;

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
}
