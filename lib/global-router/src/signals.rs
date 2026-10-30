// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dynamo_global_common::PoolId;
use parking_lot::RwLock;

use crate::load::InFlightTracker;

/// A report is used for this many of its Relay's report windows, as the Relay
/// judges its own sources, so a late or missed report or a quick reconnect
/// does not make a pool's load unknown.
const FRESH_WINDOWS: u32 = 3;

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
    /// How often the Relay reports.
    pub window: Duration,
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
    reports: DashMap<PoolId, HashMap<RelaySource, SourceReport>>,
    regions: RwLock<Arc<HashMap<String, Duration>>>,
    in_flight: Arc<InFlightTracker>,
}

/// One Relay process and one start of that process.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RelaySource {
    pub runtime_id: u64,
    pub incarnation: u64,
}

#[derive(Debug, Default)]
struct SourceReport {
    load: Option<Report>,
    kv: Option<KvReport>,
}

#[derive(Debug)]
struct Report {
    report: LoadReport,
    received: Instant,
    expires: Instant,
    /// This process's requests in flight to the pool when the report arrived.
    ours: u64,
}

#[derive(Debug)]
struct KvReport {
    used: u64,
    total: u64,
    expires: Instant,
}

impl PoolSignals {
    /// `in_flight` must be the tracker the Global Frontend counts its
    /// requests in.
    pub fn new(in_flight: Arc<InFlightTracker>) -> Self {
        Self {
            reports: DashMap::new(),
            regions: RwLock::default(),
            in_flight,
        }
    }

    /// Records what a pool's Relay reported.
    pub fn record_load(&self, pool: &PoolId, source: RelaySource, report: LoadReport) {
        let received = Instant::now();
        let current = Report {
            expires: received + report.window * FRESH_WINDOWS,
            received,
            report,
            ours: self.in_flight.get(pool),
        };
        let mut sources = self.reports.entry(pool.clone()).or_default();
        prune(&mut sources, received);
        let source_report = sources.entry(source).or_default();
        source_report.load = Some(current);
    }

    /// Records the KV blocks a pool's workers hold for running requests, and
    /// their total, as the pool's Relay reported once a `window`.
    pub fn record_kv(
        &self,
        pool: &PoolId,
        source: RelaySource,
        used: u64,
        total: u64,
        window: Duration,
    ) {
        let now = Instant::now();
        let mut sources = self.reports.entry(pool.clone()).or_default();
        prune(&mut sources, now);
        let source_report = sources.entry(source).or_default();
        source_report.kv = Some(KvReport {
            used,
            total,
            expires: now + window * FRESH_WINDOWS,
        });
    }

    /// Drops what a pool's Relay reported, for when the pool leaves or its
    /// Relay changes.
    pub fn forget(&self, pool: &PoolId) {
        self.reports.remove(pool);
    }

    /// How full the pool is against its own capacity. `None` without a fresh
    /// report, or before the workers report how many requests they can run.
    ///
    /// This process's own requests count as they are now, not as the report
    /// saw them: a burst sent since the report does not all go to one pool,
    /// and requests that finish between reports stop counting. The KV side
    /// counts waiting requests at the blocks each running request holds now,
    /// so it reaches 1 when the pool's KV cannot hold everything it has.
    /// Workers report KV only when it changes, so a pool without traffic may
    /// have no KV view yet; its slots alone then say how full it is.
    pub fn occupancy(&self, pool: &PoolId) -> Option<PoolOccupancy> {
        let now = Instant::now();
        let sources = self.reports.get(pool)?;
        let (source, current) = sources
            .iter()
            .filter_map(|(source, source_report)| {
                source_report
                    .load
                    .as_ref()
                    .filter(|current| {
                        now <= current.expires
                            && current.report.max_concurrency.is_some_and(|n| n > 0)
                    })
                    .map(|current| (source, current))
            })
            .max_by_key(|(source, current)| (current.received, **source))?;
        let capacity = current
            .report
            .max_concurrency
            .filter(|capacity| *capacity > 0)?;
        let in_flight = (current.report.in_flight + self.in_flight.get(pool))
            .saturating_sub(current.ours) as f64;
        let generating = current.report.generating as f64;
        let slots = in_flight / capacity as f64;
        let kv = sources
            .get(source)
            .and_then(|source_report| source_report.kv.as_ref())
            .filter(|kv| now <= kv.expires && kv.total > 0)
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

fn prune(sources: &mut HashMap<RelaySource, SourceReport>, now: Instant) {
    sources.retain(|_, report| {
        report.load.as_ref().is_some_and(|load| now <= load.expires)
            || report.kv.as_ref().is_some_and(|kv| now <= kv.expires)
    });
}

#[cfg(test)]
mod tests {
    use crate::selector::tests::pool;

    use super::*;

    const RELAY: RelaySource = RelaySource {
        runtime_id: 1,
        incarnation: 1,
    };

    /// 64 requests in flight, 32 of them generating, of 128 slots, in a
    /// report fresh for a minute.
    fn half_full() -> LoadReport {
        LoadReport {
            in_flight: 64,
            generating: 32,
            max_concurrency: Some(128),
            window: Duration::from_secs(20),
        }
    }

    #[test]
    fn occupancy_is_the_fuller_of_slots_and_kv() {
        let in_flight = Arc::new(InFlightTracker::new());
        let signals = PoolSignals::new(in_flight.clone());
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;
        signals.record_load(&blue, RELAY, half_full());
        // No KV view yet: the slots alone.
        let occupancy = signals.occupancy(&blue).unwrap();
        assert_eq!((occupancy.load, occupancy.kv), (0.5, None));

        // Slot-bound: half the slots, and 1,000 of 16,384 KV blocks for 32
        // running requests (2,000 for all 64).
        let window = Duration::from_secs(20);
        signals.record_kv(&blue, RELAY, 1000, 16384, window);
        let occupancy = signals.occupancy(&blue).unwrap();
        assert_eq!(occupancy.slots, 0.5);
        assert!((occupancy.kv.unwrap() - 2000.0 / 16384.0).abs() < 1e-12);
        assert_eq!(occupancy.load, 0.5);

        // KV-bound: 8,000 blocks for 32 running requests; all 64 would need
        // 16,000, nearly every block, though half the slots are free.
        signals.record_kv(&blue, RELAY, 8000, 16384, window);
        let occupancy = signals.occupancy(&blue).unwrap();
        assert!((occupancy.load - 16000.0 / 16384.0).abs() < 1e-12);

        // Without the workers' capacity, nothing is known.
        signals.record_load(
            &blue,
            RELAY,
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
        let signals = PoolSignals::new(in_flight.clone());
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;
        let _before = in_flight.start(&blue);

        assert_eq!(signals.occupancy(&blue), None);
        signals.record_load(&blue, RELAY, half_full());
        let _after = [in_flight.start(&blue), in_flight.start(&blue)];
        assert!((signals.occupancy(&blue).unwrap().slots - 66.0 / 128.0).abs() < 1e-12);

        // A report is used for three of its windows.
        let window = Duration::from_millis(50);
        signals.record_load(
            &blue,
            RELAY,
            LoadReport {
                window,
                ..half_full()
            },
        );
        assert!(signals.occupancy(&blue).is_some());
        std::thread::sleep(window * 4);
        assert_eq!(signals.occupancy(&blue), None);
    }

    #[test]
    fn replica_load_is_not_summed_and_kv_comes_from_the_selected_replica() {
        let signals = PoolSignals::new(Arc::new(InFlightTracker::new()));
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;
        let green = pool("green", "qwen", &["http://green:8000"]).id;
        let other = RelaySource {
            runtime_id: 2,
            incarnation: 1,
        };
        let window = Duration::from_secs(20);
        signals.record_load(&blue, RELAY, half_full());
        signals.record_kv(&blue, RELAY, 8000, 16384, window);
        signals.record_load(&green, RELAY, half_full());
        signals.record_load(
            &blue,
            other,
            LoadReport {
                in_flight: 16,
                generating: 8,
                ..half_full()
            },
        );
        signals.record_kv(&blue, other, 1000, 16384, window);

        let occupancy = signals.occupancy(&blue).unwrap();
        assert_eq!(occupancy.slots, 16.0 / 128.0);
        assert_eq!(occupancy.kv, Some(2000.0 / 16384.0));
        assert_eq!(signals.occupancy(&green).unwrap().slots, 0.5);

        signals.record_load(&blue, other, half_full());
        assert_eq!(signals.occupancy(&blue).unwrap().slots, 0.5);

        signals.record_load(
            &blue,
            other,
            LoadReport {
                max_concurrency: None,
                ..half_full()
            },
        );
        assert_eq!(signals.occupancy(&blue).unwrap().slots, 0.5);
        assert_eq!(
            signals.occupancy(&blue).unwrap().kv,
            Some(16000.0 / 16384.0)
        );

        let restarted = RelaySource {
            runtime_id: RELAY.runtime_id,
            incarnation: RELAY.incarnation + 1,
        };
        signals.record_load(&blue, restarted, half_full());
        assert_eq!(signals.occupancy(&blue).unwrap().kv, None);
    }

    #[test]
    fn uses_a_surviving_replica_after_the_latest_report_expires() {
        let signals = PoolSignals::new(Arc::new(InFlightTracker::new()));
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;
        let other = RelaySource {
            runtime_id: 2,
            incarnation: 1,
        };
        signals.record_load(&blue, RELAY, half_full());
        signals.record_load(
            &blue,
            other,
            LoadReport {
                in_flight: 16,
                window: Duration::from_millis(20),
                ..half_full()
            },
        );
        assert_eq!(signals.occupancy(&blue).unwrap().slots, 16.0 / 128.0);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(signals.occupancy(&blue).unwrap().slots, 0.5);
    }

    #[test]
    fn requests_that_finish_between_reports_stop_counting() {
        let in_flight = Arc::new(InFlightTracker::new());
        let signals = PoolSignals::new(in_flight.clone());
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;
        // One of the report's 64 is this process's.
        let counted = in_flight.start(&blue);
        signals.record_load(&blue, RELAY, half_full());

        // Short requests that start and finish before the next report.
        for _ in 0..100 {
            drop(in_flight.start(&blue));
        }
        assert_eq!(signals.occupancy(&blue).unwrap().slots, 0.5);

        drop(counted);
        assert_eq!(signals.occupancy(&blue).unwrap().slots, 63.0 / 128.0);
    }
}
