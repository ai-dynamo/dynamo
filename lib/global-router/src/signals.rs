// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dynamo_global_router_shared::PoolId;

use crate::load::InFlightTracker;

/// How far back a pool's expected wait looks, unless set otherwise.
const WAIT_WINDOW: Duration = Duration::from_secs(10);

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

/// One report from a pool's Relay: the pool's load, and the counts its
/// expected wait is worked out from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadReport {
    pub load: PoolLoad,
    /// Requests waiting for a first token.
    pub awaiting: u64,
    /// Requests the pool has started since its Relay started.
    pub started_total: u64,
    /// When the Relay observed these counts, in milliseconds on its clock.
    pub observed_ms: u64,
}

/// A pool's expected wait for a first token, and what it is worked out
/// from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoolWait {
    pub wait: Duration,
    /// Requests waiting for a first token, averaged over the window.
    pub awaiting: f64,
    /// Requests the pool started per second over the window.
    pub arrivals_per_s: f64,
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
    wait_window: Duration,
    fresh_reads: AtomicU64,
    stale_reads: AtomicU64,
}

#[derive(Debug)]
struct Report {
    load: PoolLoad,
    received: Instant,
    /// This process's requests to the pool so far, when the report arrived.
    started: u64,
    /// The Relay's recent counts, oldest first. The first is the last one
    /// at or before the start of the wait window.
    samples: VecDeque<Sample>,
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    observed_ms: u64,
    awaiting: u64,
    started_total: u64,
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
            wait_window: WAIT_WINDOW,
            fresh_reads: AtomicU64::new(0),
            stale_reads: AtomicU64::new(0),
        }
    }

    /// How far back a pool's expected wait looks.
    pub fn with_wait_window(mut self, window: Duration) -> Self {
        self.wait_window = window;
        self
    }

    /// Records what a pool's Relay reported. A pool with no workers has no
    /// load per worker, so its report is dropped.
    pub fn record_load(&self, pool: &PoolId, report: LoadReport) {
        if report.load.workers == 0 {
            self.forget_load(pool);
            return;
        }
        let sample = Sample {
            observed_ms: report.observed_ms,
            awaiting: report.awaiting,
            started_total: report.started_total,
        };
        let (received, started) = (Instant::now(), self.in_flight.started(pool));
        let mut current = self.loads.entry(pool.clone()).or_insert_with(|| Report {
            load: report.load,
            received,
            started,
            samples: VecDeque::new(),
        });
        current.load = report.load;
        current.received = received;
        current.started = started;
        let samples = &mut current.samples;
        // A Relay that restarted counts from zero again.
        if samples.back().is_some_and(|last| {
            sample.started_total < last.started_total || sample.observed_ms < last.observed_ms
        }) {
            samples.clear();
        }
        if samples
            .back()
            .is_some_and(|last| last.observed_ms == sample.observed_ms)
        {
            samples.pop_back();
        }
        samples.push_back(sample);
        let start = sample
            .observed_ms
            .saturating_sub(self.wait_window.as_millis() as u64);
        while samples.get(1).is_some_and(|next| next.observed_ms <= start) {
            samples.pop_front();
        }
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

    /// The pool's expected wait for a first token, by Little's law: the
    /// requests waiting for one, averaged over the wait window, divided by
    /// the rate the pool starts requests. Like requests per worker, it rises
    /// as a pool fills up, but it also compares pools whose workers differ
    /// in speed or size. Requests this process has sent since the last
    /// report count as waiting, as in [`Self::load`]. The rate counts at
    /// least one request, so a pool that starts none still has a finite
    /// wait. `None` without a fresh report, or until a second report gives
    /// a rate. Unlike [`Self::load`], this is not counted as a read.
    pub fn wait(&self, pool: &PoolId) -> Option<PoolWait> {
        let report = self
            .loads
            .get(pool)
            .filter(|report| report.received.elapsed() <= self.max_age)?;
        let (first, last) = (report.samples.front()?, report.samples.back()?);
        let span_ms = last.observed_ms.saturating_sub(first.observed_ms);
        if span_ms == 0 {
            return None;
        }
        let window = report.samples.iter().skip(1);
        let awaiting = window.map(|sample| sample.awaiting as f64).sum::<f64>()
            / (report.samples.len() - 1) as f64
            + self.in_flight.started(pool).saturating_sub(report.started) as f64;
        let started = (last.started_total - first.started_total).max(1);
        let arrivals_per_s = started as f64 * 1000.0 / span_ms as f64;
        Some(PoolWait {
            wait: Duration::from_secs_f64(awaiting / arrivals_per_s),
            awaiting,
            arrivals_per_s,
        })
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

    /// A report from a pool of 8 workers whose requests are all waiting.
    fn report(observed_ms: u64, awaiting: u64, started_total: u64) -> LoadReport {
        LoadReport {
            load: PoolLoad {
                in_flight: awaiting,
                workers: 8,
            },
            awaiting,
            started_total,
            observed_ms,
        }
    }

    fn wait_secs(signals: &PoolSignals, pool: &PoolId) -> Option<f64> {
        signals.wait(pool).map(|wait| wait.wait.as_secs_f64())
    }

    #[test]
    fn adds_requests_sent_since_the_report_and_drops_stale_reports() {
        let in_flight = Arc::new(InFlightTracker::new());
        let signals = PoolSignals::new(in_flight.clone(), Duration::from_millis(50));
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;
        let _before = in_flight.start(&blue);

        assert_eq!(signals.load(&blue), None);
        let mut first = report(0, 0, 0);
        first.load = PoolLoad {
            in_flight: 10,
            workers: 4,
        };
        signals.record_load(&blue, first);
        let _after = [in_flight.start(&blue), in_flight.start(&blue)];
        let load = signals.load(&blue).unwrap();
        assert_eq!((load.in_flight, load.workers), (12, 4));
        assert_eq!(load.per_worker(), 3.0);

        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(signals.load(&blue), None);
        assert_eq!(signals.reads(), (1, 2));

        let mut idle = report(1000, 3, 0);
        idle.load.workers = 0;
        signals.record_load(&blue, idle);
        assert_eq!(signals.load(&blue), None);
    }

    #[test]
    fn expected_wait_is_waiting_requests_over_the_start_rate() {
        let in_flight = Arc::new(InFlightTracker::new());
        let signals = PoolSignals::new(in_flight.clone(), Duration::from_secs(60));
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;

        // One report gives no rate yet.
        signals.record_load(&blue, report(0, 4, 0));
        assert_eq!(signals.wait(&blue), None);

        // 20 starts in 2 s is 10 per second; 7 waiting on average: 0.7 s.
        signals.record_load(&blue, report(1000, 6, 10));
        signals.record_load(&blue, report(2000, 8, 20));
        let wait = signals.wait(&blue).unwrap();
        assert_eq!((wait.awaiting, wait.arrivals_per_s), (7.0, 10.0));
        assert!((wait.wait.as_secs_f64() - 0.7).abs() < 1e-9);

        // Requests this process sent since the report are waiting too.
        let _sent = [
            in_flight.start(&blue),
            in_flight.start(&blue),
            in_flight.start(&blue),
        ];
        assert!((wait_secs(&signals, &blue).unwrap() - 1.0).abs() < 1e-9);
        // Not counted as load reads.
        assert_eq!(signals.reads(), (0, 0));
    }

    #[test]
    fn expected_wait_looks_back_one_window_and_restarts_with_the_relay() {
        let signals = PoolSignals::new(Arc::new(InFlightTracker::new()), Duration::from_secs(60))
            .with_wait_window(Duration::from_secs(2));
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;

        // A burst of starts early on falls out of the 2 s window: the rate
        // is the last 2 s, 4 per second, with 2 waiting.
        for (at, started) in [(0, 0), (1000, 100), (2000, 104), (3000, 108)] {
            signals.record_load(&blue, report(at, 2, started));
        }
        assert!((wait_secs(&signals, &blue).unwrap() - 0.5).abs() < 1e-9);

        // A pool that starts nothing still has a finite wait.
        signals.record_load(&blue, report(4000, 2, 108));
        signals.record_load(&blue, report(5000, 2, 108));
        assert!((wait_secs(&signals, &blue).unwrap() - 4.0).abs() < 1e-9);

        // The Relay restarted: its counts begin again, so does the history.
        signals.record_load(&blue, report(6000, 2, 3));
        assert_eq!(signals.wait(&blue), None);
        signals.record_load(&blue, report(7000, 2, 7));
        assert!((wait_secs(&signals, &blue).unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn stale_reports_have_no_expected_wait() {
        let signals = PoolSignals::new(Arc::new(InFlightTracker::new()), Duration::from_millis(50));
        let blue = pool("blue", "qwen", &["http://blue:8000"]).id;
        signals.record_load(&blue, report(0, 1, 0));
        signals.record_load(&blue, report(1000, 1, 5));
        assert!(signals.wait(&blue).is_some());
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(signals.wait(&blue), None);
    }
}
