// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dynamo_global_common::PoolId;
use parking_lot::RwLock;

use crate::load::InFlightTracker;

/// One serving-load report from a pool's Relay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadReport {
    /// Requests at the pool from every Global Frontend, waiting for a first
    /// token or generating.
    pub in_flight: u64,
    /// Requests generating.
    pub generating: u64,
    /// How many requests the pool's workers can run at once, as they report
    /// it. `None` if any worker has not.
    pub max_concurrency: Option<u64>,
}

/// How full a pool is against its own capacity, 1 being full, on each
/// resource that can run out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoolOccupancy {
    /// The fuller of the two.
    pub load: f64,
    /// Requests in flight over the requests the pool can run at once.
    pub slots: f64,
    /// The KV blocks all requests in flight would hold, at the blocks each
    /// running request holds now, over the pool's KV blocks. `None` until
    /// the Relay has a complete KV view of the pool.
    pub kv: Option<f64>,
}

/// What the load-aware selectors know about each pool besides the catalog:
/// its load, as its Relay last reported it, and the round trip to each
/// region, as this process measures it. Background tasks write them; every
/// request reads them.
#[derive(Debug)]
pub struct PoolSignals {
    loads: DashMap<PoolId, Report>,
    kv: DashMap<PoolId, KvReport>,
    regions: RwLock<Arc<HashMap<String, Duration>>>,
    in_flight: Arc<InFlightTracker>,
    max_age: Duration,
}

#[derive(Debug)]
struct Report {
    report: LoadReport,
    received: Instant,
    /// This process's requests to the pool so far, when the report arrived.
    started: u64,
}

#[derive(Debug)]
struct KvReport {
    used: u64,
    total: u64,
    received: Instant,
}

impl PoolSignals {
    /// `in_flight` must be the tracker the forwarding engine counts its
    /// requests in. A report older than `max_age` is not used.
    pub fn new(in_flight: Arc<InFlightTracker>, max_age: Duration) -> Self {
        Self {
            loads: DashMap::new(),
            kv: DashMap::new(),
            regions: RwLock::default(),
            in_flight,
            max_age,
        }
    }

    /// Records what a pool's Relay reported.
    pub fn record_load(&self, pool: &PoolId, report: LoadReport) {
        let current = Report {
            report,
            received: Instant::now(),
            started: self.in_flight.started(pool),
        };
        self.loads.insert(pool.clone(), current);
    }

    /// Drops a pool's load, for when its Relay stream ends or its view is
    /// incomplete.
    pub fn forget_load(&self, pool: &PoolId) {
        self.loads.remove(pool);
    }

    /// Records the KV blocks a pool's workers hold for running requests, and
    /// their total, as the pool's Relay reported.
    pub fn record_kv(&self, pool: &PoolId, used: u64, total: u64) {
        let received = Instant::now();
        self.kv.insert(
            pool.clone(),
            KvReport {
                used,
                total,
                received,
            },
        );
    }

    /// Drops a pool's KV report, for when its Relay stream ends or its view
    /// is incomplete.
    pub fn forget_kv(&self, pool: &PoolId) {
        self.kv.remove(pool);
    }

    /// How full the pool is against its own capacity.
    ///
    /// The Relay reports about once a second, so requests this process has
    /// sent since the last report count as in flight; otherwise a burst
    /// would all go to the same pool. The KV side counts waiting requests
    /// too, at the blocks each running request holds now, so it reaches 1
    /// when the pool's KV cannot hold everything it has, even though the
    /// blocks in use never exceed the total. Workers report KV only when it
    /// changes, so a pool that has had no traffic may have no KV view yet;
    /// its KV is then empty in practice, and its slots alone say how full it
    /// is. `None` without a fresh report, or before the workers report how
    /// many requests they can run.
    pub fn occupancy(&self, pool: &PoolId) -> Option<PoolOccupancy> {
        let current = self
            .loads
            .get(pool)
            .filter(|current| current.received.elapsed() <= self.max_age)?;
        let capacity = current
            .report
            .max_concurrency
            .filter(|capacity| *capacity > 0)?;
        let in_flight = (current.report.in_flight
            + self.in_flight.started(pool).saturating_sub(current.started))
            as f64;
        let generating = current.report.generating as f64;
        let slots = in_flight / capacity as f64;
        let kv = self
            .kv
            .get(pool)
            .filter(|kv| kv.received.elapsed() <= self.max_age && kv.total > 0)
            .map(|kv| {
                let held = if generating > 0.0 {
                    kv.used as f64 * in_flight.max(generating) / generating
                } else {
                    kv.used as f64
                };
                held / kv.total as f64
            });
        Some(PoolOccupancy {
            load: kv.map_or(slots, |kv| slots.max(kv)),
            slots,
            kv,
        })
    }

    /// Replaces the measured round trip to each region.
    pub fn set_region_round_trips(&self, round_trips: HashMap<String, Duration>) {
        *self.regions.write() = Arc::new(round_trips);
    }

    /// The measured round trip to each region, as one consistent set.
    pub fn region_round_trips(&self) -> Arc<HashMap<String, Duration>> {
        self.regions.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::selector::tests::pool;

    use super::*;

    /// 64 requests in flight, 32 of them generating, of 128 slots.
    fn half_full() -> LoadReport {
        LoadReport {
            in_flight: 64,
            generating: 32,
            max_concurrency: Some(128),
        }
    }

    #[test]
    fn occupancy_is_the_fuller_of_slots_and_kv() {
        let in_flight = Arc::new(InFlightTracker::new());
        let signals = PoolSignals::new(in_flight.clone(), Duration::from_secs(60));
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;
        signals.record_load(&blue, half_full());
        // No KV view yet: the slots alone.
        let occupancy = signals.occupancy(&blue).unwrap();
        assert_eq!((occupancy.load, occupancy.kv), (0.5, None));

        // Slot-bound: half the slots, and 1,000 of 16,384 KV blocks for 32
        // running requests (2,000 for all 64).
        signals.record_kv(&blue, 1000, 16384);
        let occupancy = signals.occupancy(&blue).unwrap();
        assert_eq!(occupancy.slots, 0.5);
        assert!((occupancy.kv.unwrap() - 2000.0 / 16384.0).abs() < 1e-12);
        assert_eq!(occupancy.load, 0.5);

        // KV-bound: 8,000 blocks for 32 running requests; all 64 would need
        // 16,000, nearly every block, though half the slots are free.
        signals.record_kv(&blue, 8000, 16384);
        let occupancy = signals.occupancy(&blue).unwrap();
        assert!((occupancy.load - 16000.0 / 16384.0).abs() < 1e-12);

        // Without the workers' capacity, nothing is known.
        signals.record_load(
            &blue,
            LoadReport {
                max_concurrency: None,
                ..half_full()
            },
        );
        assert_eq!(signals.occupancy(&blue), None);
    }

    #[test]
    fn adds_requests_sent_since_the_report_and_drops_stale_reports() {
        let in_flight = Arc::new(InFlightTracker::new());
        let signals = PoolSignals::new(in_flight.clone(), Duration::from_millis(50));
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;
        let _before = in_flight.start(&blue);

        assert_eq!(signals.occupancy(&blue), None);
        signals.record_load(&blue, half_full());
        let _after = [in_flight.start(&blue), in_flight.start(&blue)];
        assert!((signals.occupancy(&blue).unwrap().slots - 66.0 / 128.0).abs() < 1e-12);

        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(signals.occupancy(&blue), None);
    }
}
