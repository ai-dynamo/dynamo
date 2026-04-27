// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;
use std::time::Instant;

use prometheus::{Histogram, HistogramOpts, HistogramVec, Registry};

static KVBM_PROBES: OnceLock<KvbmPhaseMetrics> = OnceLock::new();

pub struct KvbmPhaseMetrics {
    pub hash_ms: Histogram,
    pub lookup_ms: Histogram,
    pub allocate_ms: HistogramVec,
    pub evict_ms: Histogram,
    pub return_ms: Histogram,
}

fn probe_buckets() -> Vec<f64> {
    prometheus::exponential_buckets(0.001, 2.0, 20).unwrap()
}

impl KvbmPhaseMetrics {
    pub fn register(registry: &Registry) -> Result<(), prometheus::Error> {
        let metrics = KVBM_PROBES.get_or_init(|| {
            let buckets = probe_buckets();
            Self {
                hash_ms: Histogram::with_opts(
                    HistogramOpts::new(
                        "dynamo_compass_kvbm_hash_ms",
                        "KVBM hash computation time in milliseconds",
                    )
                    .buckets(buckets.clone()),
                )
                .expect("kvbm_hash_ms"),
                lookup_ms: Histogram::with_opts(
                    HistogramOpts::new(
                        "dynamo_compass_kvbm_lookup_ms",
                        "KVBM radix tree lookup time in milliseconds",
                    )
                    .buckets(buckets.clone()),
                )
                .expect("kvbm_lookup_ms"),
                allocate_ms: HistogramVec::new(
                    HistogramOpts::new(
                        "dynamo_compass_kvbm_allocate_ms",
                        "KVBM block allocation time in milliseconds",
                    )
                    .buckets(buckets.clone()),
                    &["phase"],
                )
                .expect("kvbm_allocate_ms"),
                evict_ms: Histogram::with_opts(
                    HistogramOpts::new(
                        "dynamo_compass_kvbm_evict_ms",
                        "KVBM block eviction time in milliseconds",
                    )
                    .buckets(buckets.clone()),
                )
                .expect("kvbm_evict_ms"),
                return_ms: Histogram::with_opts(
                    HistogramOpts::new(
                        "dynamo_compass_kvbm_return_ms",
                        "KVBM block return time in milliseconds",
                    )
                    .buckets(buckets),
                )
                .expect("kvbm_return_ms"),
            }
        });
        registry.register(Box::new(metrics.hash_ms.clone()))?;
        registry.register(Box::new(metrics.lookup_ms.clone()))?;
        registry.register(Box::new(metrics.allocate_ms.clone()))?;
        registry.register(Box::new(metrics.evict_ms.clone()))?;
        registry.register(Box::new(metrics.return_ms.clone()))?;
        Ok(())
    }

    pub fn get() -> Option<&'static Self> {
        KVBM_PROBES.get()
    }

    pub fn observe_hash(&self, elapsed_ms: f64) {
        self.hash_ms.observe(elapsed_ms);
    }

    pub fn observe_lookup(&self, elapsed_ms: f64) {
        self.lookup_ms.observe(elapsed_ms);
    }

    pub fn observe_allocate(&self, total_ms: f64, on_cpu_ms: f64, lock_wait_ms: f64) {
        self.allocate_ms
            .with_label_values(&["total"])
            .observe(total_ms);
        self.allocate_ms
            .with_label_values(&["on_cpu"])
            .observe(on_cpu_ms);
        self.allocate_ms
            .with_label_values(&["lock_wait"])
            .observe(lock_wait_ms);
    }

    pub fn observe_evict(&self, elapsed_ms: f64) {
        self.evict_ms.observe(elapsed_ms);
    }

    pub fn observe_return(&self, elapsed_ms: f64) {
        self.return_ms.observe(elapsed_ms);
    }
}

pub struct LockTimingGuard {
    lock_request_time: Instant,
    lock_acquired_time: Option<Instant>,
}

impl LockTimingGuard {
    pub fn new() -> Self {
        Self {
            lock_request_time: Instant::now(),
            lock_acquired_time: None,
        }
    }

    pub fn mark_acquired(&mut self) {
        self.lock_acquired_time = Some(Instant::now());
    }

    pub fn finish(self) -> (f64, f64) {
        let now = Instant::now();
        let acquired = self.lock_acquired_time.unwrap_or(now);
        let lock_wait_ms = acquired
            .duration_since(self.lock_request_time)
            .as_secs_f64()
            * 1000.0;
        let on_cpu_ms = now.duration_since(acquired).as_secs_f64() * 1000.0;
        (on_cpu_ms, lock_wait_ms)
    }
}

impl Default for LockTimingGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_lock_timing_guard() {
        let mut guard = LockTimingGuard::new();
        sleep(Duration::from_millis(5));
        guard.mark_acquired();
        sleep(Duration::from_millis(5));
        let (on_cpu, lock_wait) = guard.finish();
        assert!(lock_wait >= 3.0);
        assert!(on_cpu >= 3.0);
    }

    #[test]
    fn test_probe_registration() {
        let registry = Registry::new();
        KvbmPhaseMetrics::register(&registry).unwrap();
        let probes = KvbmPhaseMetrics::get().unwrap();
        probes.observe_hash(0.12);
        probes.observe_lookup(0.84);
        probes.observe_allocate(3.51, 1.20, 2.31);
        probes.observe_evict(0.43);
        probes.observe_return(0.09);
    }
}
