// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dynamo_global_router_shared::PoolId;

use crate::load::InFlightTracker;

/// A pool's load: requests in flight from every Global Frontend, and the
/// workers serving them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolLoad {
    pub in_flight: u64,
    pub workers: u64,
}

impl PoolLoad {
    /// Requests in flight per worker. Pools differ in size, so loads are
    /// compared per worker.
    pub fn per_worker(&self) -> f64 {
        self.in_flight as f64 / self.workers as f64
    }
}

/// What the router knows about each pool besides the catalog: its load, as
/// its Relay last reported it, and the round trip to it, as this process
/// measures it. Both are written by background tasks and read on every
/// request.
#[derive(Debug)]
pub struct PoolSignals {
    loads: DashMap<PoolId, Report>,
    round_trips: DashMap<PoolId, Duration>,
    in_flight: Arc<InFlightTracker>,
    max_age: Duration,
    fresh_reads: AtomicU64,
    stale_reads: AtomicU64,
}

#[derive(Debug)]
struct Report {
    load: PoolLoad,
    received: Instant,
    /// This process's requests to the pool so far, when the report arrived.
    started: u64,
}

impl PoolSignals {
    /// `in_flight` must be the tracker the forwarding engine counts its
    /// requests in. A load report older than `max_age` is not used.
    pub fn new(in_flight: Arc<InFlightTracker>, max_age: Duration) -> Self {
        Self {
            loads: DashMap::new(),
            round_trips: DashMap::new(),
            in_flight,
            max_age,
            fresh_reads: AtomicU64::new(0),
            stale_reads: AtomicU64::new(0),
        }
    }

    /// Records the load a pool's Relay reported. A pool with no workers has
    /// no load per worker, so its report is dropped.
    pub fn record_load(&self, pool: &PoolId, load: PoolLoad) {
        if load.workers == 0 {
            self.forget_load(pool);
            return;
        }
        let report = Report {
            load,
            received: Instant::now(),
            started: self.in_flight.started(pool),
        };
        self.loads.insert(pool.clone(), report);
    }

    /// Drops a pool's load, for when its Relay stream ends.
    pub fn forget_load(&self, pool: &PoolId) {
        self.loads.remove(pool);
    }

    /// The pool's load, if its last report is fresh. The Relay reports once
    /// a second, so the requests this process has sent to the pool since the
    /// report are added; otherwise a burst would all go to the same pool.
    pub fn load(&self, pool: &PoolId) -> Option<PoolLoad> {
        let load = self
            .loads
            .get(pool)
            .filter(|report| report.received.elapsed() <= self.max_age)
            .map(|report| PoolLoad {
                in_flight: report.load.in_flight
                    + self.in_flight.started(pool).saturating_sub(report.started),
                workers: report.load.workers,
            });
        let reads = if load.is_some() {
            &self.fresh_reads
        } else {
            &self.stale_reads
        };
        reads.fetch_add(1, Ordering::Relaxed);
        load
    }

    pub fn record_round_trip(&self, pool: &PoolId, round_trip: Duration) {
        self.round_trips.insert(pool.clone(), round_trip);
    }

    /// The last measured round trip to the pool.
    pub fn round_trip(&self, pool: &PoolId) -> Option<Duration> {
        self.round_trips.get(pool).map(|round_trip| *round_trip)
    }

    /// How many load reads found a fresh report, and how many did not.
    pub fn reads(&self) -> (u64, u64) {
        (
            self.fresh_reads.load(Ordering::Relaxed),
            self.stale_reads.load(Ordering::Relaxed),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::catalog::tests::pool;

    use super::*;

    #[test]
    fn adds_requests_sent_since_the_report_and_drops_stale_reports() {
        let in_flight = Arc::new(InFlightTracker::new());
        let signals = PoolSignals::new(in_flight.clone(), Duration::from_millis(50));
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;
        let _before = in_flight.start(&blue);

        assert_eq!(signals.load(&blue), None);
        signals.record_load(
            &blue,
            PoolLoad {
                in_flight: 10,
                workers: 4,
            },
        );
        let _after = [in_flight.start(&blue), in_flight.start(&blue)];
        let load = signals.load(&blue).unwrap();
        assert_eq!((load.in_flight, load.workers), (12, 4));
        assert_eq!(load.per_worker(), 3.0);

        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(signals.load(&blue), None);
        assert_eq!(signals.reads(), (1, 2));

        signals.record_load(
            &blue,
            PoolLoad {
                in_flight: 3,
                workers: 0,
            },
        );
        assert_eq!(signals.load(&blue), None);
    }
}
