// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dynamo_global_router::{Pool, PoolSignals, PoolSnapshot};
use dynamo_global_router_shared::PoolId;
use serde_json::{Value, json};

/// How long a session's last pool is remembered for [`RoutingStats`].
const SESSION_MEMORY: Duration = Duration::from_secs(30 * 60);
const SWEEP_INTERVAL: Duration = Duration::from_secs(60);

/// Counts where this process sends requests, whatever the policy, so a
/// benchmark can compare policies: per pool, how many requests it got;
/// overall, how many stayed in this process's region, how many carried a
/// session ID, and how many of a session's later turns went to the same pool
/// as the turn before. The counts only grow; read them twice and subtract.
#[derive(Debug)]
pub struct RoutingStats {
    region: Option<String>,
    requests: AtomicU64,
    in_region: AtomicU64,
    with_session: AtomicU64,
    later_turns: AtomicU64,
    same_pool: AtomicU64,
    routed: DashMap<PoolId, AtomicU64>,
    last_pool: DashMap<String, (PoolId, Instant)>,
    swept: Mutex<Instant>,
}

impl RoutingStats {
    /// `region` is this process's region, if it knows it.
    pub fn new(region: Option<String>) -> Self {
        Self {
            region,
            requests: AtomicU64::new(0),
            in_region: AtomicU64::new(0),
            with_session: AtomicU64::new(0),
            later_turns: AtomicU64::new(0),
            same_pool: AtomicU64::new(0),
            routed: DashMap::new(),
            last_pool: DashMap::new(),
            swept: Mutex::new(Instant::now()),
        }
    }

    /// Counts one request sent to `pool`.
    pub fn record(&self, pool: &Pool, session: Option<&str>) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        if self.region.as_deref() == Some(pool.location.region.as_str()) {
            self.in_region.fetch_add(1, Ordering::Relaxed);
        }
        match self.routed.get(&pool.id) {
            Some(count) => count.fetch_add(1, Ordering::Relaxed),
            None => self
                .routed
                .entry(pool.id.clone())
                .or_default()
                .fetch_add(1, Ordering::Relaxed),
        };
        let Some(session) = session else {
            return;
        };
        self.with_session.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now();
        let previous = self
            .last_pool
            .insert(session.to_owned(), (pool.id.clone(), now));
        if let Some((previous, at)) = previous
            && now.duration_since(at) <= SESSION_MEMORY
        {
            self.later_turns.fetch_add(1, Ordering::Relaxed);
            if previous == pool.id {
                self.same_pool.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.sweep(now);
    }

    fn sweep(&self, now: Instant) {
        let Ok(mut swept) = self.swept.try_lock() else {
            return;
        };
        if now.duration_since(*swept) < SWEEP_INTERVAL {
            return;
        }
        *swept = now;
        drop(swept);
        self.last_pool
            .retain(|_, (_, at)| now.duration_since(*at) <= SESSION_MEMORY);
    }

    /// The counts, with each known pool's current load, timing and round
    /// trip.
    pub fn report(&self, pools: &PoolSnapshot, signals: &PoolSignals) -> Value {
        let count = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        let pools = pools
            .pools()
            .iter()
            .map(|pool| {
                let load = signals.load(&pool.id);
                let timing = signals.timing(&pool.id);
                let occupancy = signals.occupancy(&pool.id);
                json!({
                    "name": pool.name,
                    "region": pool.location.region,
                    "routed": self.routed.get(&pool.id).map_or(0, |routed| count(&routed)),
                    "in_flight": load.map(|load| load.in_flight),
                    "workers": load.map(|load| load.workers),
                    "wait_ms": timing.map(|timing| timing.wait.as_secs_f64() * 1000.0),
                    "token_ms": timing
                        .and_then(|timing| timing.token_time)
                        .map(|token_time| token_time.as_secs_f64() * 1000.0),
                    "served_per_s": timing.map(|timing| timing.served_per_s),
                    "occupancy": occupancy.map(|occupancy| occupancy.load),
                    "slot_occupancy": occupancy.map(|occupancy| occupancy.slots),
                    "kv_occupancy": occupancy.and_then(|occupancy| occupancy.kv),
                    "round_trip_ms": signals
                        .round_trip(&pool.id)
                        .map(|round_trip| round_trip.as_secs_f64() * 1000.0),
                })
            })
            .collect::<Vec<_>>();
        let (fresh, stale) = signals.reads();
        json!({
            "region": self.region,
            "requests": count(&self.requests),
            "in_region": count(&self.in_region),
            "with_session": count(&self.with_session),
            "later_turns": count(&self.later_turns),
            "same_pool": count(&self.same_pool),
            "load_reads": {"fresh": fresh, "stale": stale},
            "pools": pools,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dynamo_global_router::InFlightTracker;
    use dynamo_global_router_shared::{PoolIdDeriver, PoolKey, PoolLocation, V1PoolIdDeriver};

    use super::*;

    fn pool(region: &str, dgd: &str) -> Pool {
        Pool {
            id: V1PoolIdDeriver.derive(&PoolKey::new(region, "dynamo", dgd).unwrap()),
            name: format!("{region}/dynamo/{dgd}"),
            model: "qwen".into(),
            location: PoolLocation {
                region: region.into(),
                availability_zone: None,
                cluster: None,
                datacenter: None,
            },
            frontends: vec![format!("http://{dgd}:8000/").parse().unwrap()],
            relay: None,
        }
    }

    #[test]
    fn counts_regions_sessions_and_repeat_pools() {
        let stats = RoutingStats::new(Some("us-east-2".into()));
        let (ohio, tokyo) = (pool("us-east-2", "a"), pool("ap-northeast-1", "b"));

        stats.record(&ohio, Some("s"));
        stats.record(&ohio, Some("s"));
        stats.record(&tokyo, Some("s"));
        stats.record(&tokyo, None);

        let snapshot = PoolSnapshot::new(vec![ohio, tokyo]);
        let signals = PoolSignals::new(Arc::new(InFlightTracker::new()), Duration::from_secs(1));
        let report = stats.report(&snapshot, &signals);
        assert_eq!(report["requests"], 4);
        assert_eq!(report["in_region"], 2);
        assert_eq!(report["with_session"], 3);
        assert_eq!(report["later_turns"], 2);
        assert_eq!(report["same_pool"], 1);
        assert_eq!(report["pools"][1]["routed"], 2);
    }
}
