// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selectors that keep a session's turns on one pool, so the pool's KV cache
//! for the session is reused, and that prefer the region a request entered.
//! Both read each pool's load from [`PoolSignals`] and leave a pool only when
//! [`Spill`] says to: while it is overloaded, or, with
//! [`SpillRule::Cheapest`], for a pool enough cheaper.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dynamo_global_router_shared::PoolId;
use dynamo_runtime::pipeline::BuiltinRoutePicker;
use xxhash_rust::xxh3::{xxh3_64, xxh3_64_with_seed};

use crate::catalog::Pool;
use crate::placement::{
    Spill, SpillRule, by_distance, cheapest, cost_order, headroom, least_loaded, place,
};
use crate::selector::{PoolSelector, PoolTarget, RoutingRequest, target};
use crate::signals::PoolSignals;

/// How often [`AffinitySelector`] drops pins that have expired.
const SWEEP_INTERVAL: Duration = Duration::from_secs(60);

/// Sends each session to a home pool chosen by hashing, nearest the region
/// the request entered, while that pool is not overloaded.
///
/// Pools are ordered by the distance of their region and, within a region,
/// by a hash of the session ID and the pool ID (rendezvous hashing). Every
/// turn of a session, on any replica, sees the same order without stored
/// state. The request goes to the first pool in that order that is not
/// overloaded, so a session leaves its home pool only while home is
/// overloaded, and returns as soon as it is not. A request without a session
/// ID is hashed on a counter instead, which spreads such requests over the
/// nearest region's pools. With [`SpillRule::Cheapest`], a request leaves
/// home only for a pool enough cheaper.
#[derive(Debug)]
pub struct RendezvousSelector {
    signals: Arc<PoolSignals>,
    region: String,
    spill: Spill,
    unkeyed: AtomicU64,
    frontends: DashMap<PoolId, BuiltinRoutePicker>,
}

impl RendezvousSelector {
    /// `region` is the region of this process, which every request enters.
    pub fn new(signals: Arc<PoolSignals>, region: String, spill: Spill) -> Self {
        Self {
            signals,
            region,
            spill,
            unkeyed: AtomicU64::new(0),
            frontends: DashMap::new(),
        }
    }
}

impl PoolSelector for RendezvousSelector {
    fn select(&self, request: &RoutingRequest<'_>, candidates: &[&Pool]) -> Option<PoolTarget> {
        let mut placed = place(candidates, &self.region, &self.signals, self.spill);
        let key = match request.session {
            Some(session) => xxh3_64(session.as_bytes()),
            None => self.unkeyed.fetch_add(1, Ordering::Relaxed),
        };
        let score = |pool: &Pool| xxh3_64_with_seed(pool.id.as_str().as_bytes(), key);
        placed.sort_by(|left, right| {
            by_distance(left, right).then_with(|| score(right.pool).cmp(&score(left.pool)))
        });
        let choice = match self.spill.rule {
            SpillRule::FleetRatio => placed
                .iter()
                .find(|candidate| !candidate.overloaded)
                .or_else(|| least_loaded(&placed)),
            // Home is the first pool in hash order.
            SpillRule::Cheapest { margin } => cheapest(&placed, placed.first(), margin),
            SpillRule::Headroom { factor } => headroom(&placed, placed.first(), factor),
        }?;
        target(&self.frontends, choice.pool)
    }
}

/// Pins each session to a pool, and moves the pin only when that pool is
/// overloaded.
///
/// A new session goes to the first pool in cost order that is not
/// overloaded: nearest region first, then less busy, which is usually the
/// less busy pool in the region the request entered. Later
/// turns go to the pinned pool while it is not overloaded, even if it is in
/// another region, since the session's cache is there. When the pinned pool
/// is overloaded, the turn goes to the first other pool in cost order that
/// is not, and the pin moves there. With [`SpillRule::Cheapest`], a turn
/// leaves the pinned pool only for a pool enough cheaper, and the pin moves
/// there. Pins belong to this process and expire after `idle` without a
/// request.
#[derive(Debug)]
pub struct AffinitySelector {
    signals: Arc<PoolSignals>,
    region: String,
    spill: Spill,
    idle: Duration,
    pins: DashMap<String, Pin>,
    swept: Mutex<Instant>,
    frontends: DashMap<PoolId, BuiltinRoutePicker>,
}

#[derive(Debug)]
struct Pin {
    pool: PoolId,
    used: Instant,
}

impl AffinitySelector {
    /// `region` is the region of this process, which every request enters.
    pub fn new(signals: Arc<PoolSignals>, region: String, spill: Spill, idle: Duration) -> Self {
        Self {
            signals,
            region,
            spill,
            idle,
            pins: DashMap::new(),
            swept: Mutex::new(Instant::now()),
            frontends: DashMap::new(),
        }
    }

    /// Sessions with a live pin.
    pub fn pinned_sessions(&self) -> usize {
        self.pins.len()
    }

    fn pinned(&self, session: &str) -> Option<PoolId> {
        self.pins
            .get(session)
            .filter(|pin| pin.used.elapsed() <= self.idle)
            .map(|pin| pin.pool.clone())
    }

    fn pin(&self, session: &str, pool: &PoolId) {
        let used = Instant::now();
        match self.pins.get_mut(session) {
            Some(mut pin) => {
                if pin.pool != *pool {
                    pin.pool = pool.clone();
                }
                pin.used = used;
            }
            None => {
                let pool = pool.clone();
                self.pins.insert(session.to_owned(), Pin { pool, used });
            }
        }
    }

    fn sweep(&self) {
        let Ok(mut swept) = self.swept.try_lock() else {
            return;
        };
        if swept.elapsed() < SWEEP_INTERVAL {
            return;
        }
        *swept = Instant::now();
        drop(swept);
        self.pins.retain(|_, pin| pin.used.elapsed() <= self.idle);
    }
}

impl PoolSelector for AffinitySelector {
    fn select(&self, request: &RoutingRequest<'_>, candidates: &[&Pool]) -> Option<PoolTarget> {
        let mut placed = place(candidates, &self.region, &self.signals, self.spill);
        cost_order(&mut placed);
        let pinned = request.session.and_then(|session| self.pinned(session));
        let is_pinned = |pool: &Pool| pinned.as_ref() == Some(&pool.id);
        let choice = match self.spill.rule {
            SpillRule::FleetRatio => placed
                .iter()
                .find(|candidate| is_pinned(candidate.pool) && !candidate.overloaded)
                .or_else(|| {
                    placed
                        .iter()
                        .find(|candidate| !is_pinned(candidate.pool) && !candidate.overloaded)
                })
                .or_else(|| least_loaded(&placed)),
            // Without a pin, the first pool in cost order: the least busy
            // pool in the nearest region.
            SpillRule::Cheapest { margin } => {
                let default = placed
                    .iter()
                    .find(|candidate| is_pinned(candidate.pool))
                    .or_else(|| placed.first());
                cheapest(&placed, default, margin)
            }
            SpillRule::Headroom { factor } => {
                let default = placed
                    .iter()
                    .find(|candidate| is_pinned(candidate.pool))
                    .or_else(|| placed.first());
                headroom(&placed, default, factor)
            }
        }?;
        if let Some(session) = request.session {
            self.pin(session, &choice.pool.id);
        }
        self.sweep();
        target(&self.frontends, choice.pool)
    }
}

#[cfg(test)]
mod tests {
    use crate::placement::tests::{
        OHIO, by_cost, by_headroom, by_latency, fleet, set_loads, set_occupancy, set_waits,
    };

    use super::*;

    fn pick(selector: &dyn PoolSelector, pools: &[Pool], session: Option<&str>) -> String {
        let candidates = pools.iter().collect::<Vec<_>>();
        let request = RoutingRequest::new("qwen").with_session(session);
        let target = selector.select(&request, &candidates).unwrap();
        let pool = pools.iter().find(|pool| pool.id == target.pool).unwrap();
        pool.name.rsplit('/').next().unwrap().to_owned()
    }

    #[test]
    fn rendezvous_keeps_each_session_on_one_local_pool_on_every_replica() {
        let (pools, signals) = fleet();
        set_loads(&pools, &signals, [10, 10, 10, 10]);
        let first = RendezvousSelector::new(signals.clone(), OHIO.into(), Spill::default());
        let second = RendezvousSelector::new(signals, OHIO.into(), Spill::default());

        let homes = (0..64)
            .map(|n| {
                let session = format!("session-{n}");
                let home = pick(&first, &pools, Some(&session));
                assert_eq!(pick(&first, &pools, Some(&session)), home);
                assert_eq!(pick(&second, &pools, Some(&session)), home);
                home
            })
            .collect::<Vec<_>>();
        assert!(homes.iter().all(|home| home.starts_with("ohio-")));
        assert!(homes.contains(&"ohio-a".into()) && homes.contains(&"ohio-b".into()));
    }

    #[test]
    fn rendezvous_spills_to_the_nearest_region_with_room_and_comes_home() {
        let (pools, signals) = fleet();
        let selector = RendezvousSelector::new(signals.clone(), OHIO.into(), Spill::default());
        set_loads(&pools, &signals, [10, 10, 10, 10]);
        let home = pick(&selector, &pools, Some("s"));

        // Both Ohio pools are overloaded (fleet 12.5 per worker); Oregon is
        // nearer than Tokyo and has room.
        set_loads(&pools, &signals, [180, 180, 20, 20]);
        assert_eq!(pick(&selector, &pools, Some("s")), "oregon-a");
        // Oregon too (fleet 18.75, each 25): only Tokyo has room.
        set_loads(&pools, &signals, [200, 200, 200, 0]);
        assert_eq!(pick(&selector, &pools, Some("s")), "tokyo-a");

        set_loads(&pools, &signals, [10, 10, 10, 10]);
        assert_eq!(pick(&selector, &pools, Some("s")), home);
    }

    #[test]
    fn affinity_pins_new_sessions_to_the_less_loaded_local_pool() {
        let (pools, signals) = fleet();
        let selector = AffinitySelector::new(
            signals.clone(),
            OHIO.into(),
            Spill::default(),
            Duration::from_secs(60),
        );
        set_loads(&pools, &signals, [40, 20, 30, 30]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-b");

        // Ohio-a is now less loaded, but Ohio-b is not overloaded: stay.
        set_loads(&pools, &signals, [20, 30, 30, 30]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-b");
        assert_eq!(pick(&selector, &pools, Some("t")), "ohio-a");
        assert_eq!(selector.pinned_sessions(), 2);
        // No session: no pin, the cheapest pool with room.
        assert_eq!(pick(&selector, &pools, None), "ohio-a");
        assert_eq!(selector.pinned_sessions(), 2);
    }

    #[test]
    fn affinity_moves_the_pin_on_overload_and_stays_there() {
        let (pools, signals) = fleet();
        let selector = AffinitySelector::new(
            signals.clone(),
            OHIO.into(),
            Spill::default(),
            Duration::from_secs(60),
        );
        set_loads(&pools, &signals, [10, 20, 20, 20]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-a");

        // Ohio-a overloaded, Ohio-b has room: move within the region.
        set_loads(&pools, &signals, [200, 40, 40, 40]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-b");

        // Both Ohio pools overloaded: move to the nearest region with room.
        set_loads(&pools, &signals, [200, 200, 20, 20]);
        assert_eq!(pick(&selector, &pools, Some("s")), "oregon-a");

        // Ohio recovers. Oregon is busier per worker but under the floor
        // (2.5 against a fleet of 1.56), so the pin stays in Oregon.
        set_loads(&pools, &signals, [10, 10, 20, 10]);
        assert_eq!(pick(&selector, &pools, Some("s")), "oregon-a");

        // Equal loads: nothing is overloaded, so the pin holds.
        set_loads(&pools, &signals, [200, 200, 200, 200]);
        assert_eq!(pick(&selector, &pools, Some("s")), "oregon-a");
    }

    #[test]
    fn affinity_by_latency_pins_new_sessions_to_the_pool_that_answers_sooner() {
        let (pools, signals) = fleet();
        // Both Ohio pools hold 5 requests per worker, but Ohio-a waits 2 s
        // and Ohio-b 0.25 s.
        set_waits(&pools, &signals, [(8, 4), (2, 8), (2, 8), (2, 8)]);
        let idle = Duration::from_secs(60);

        let per_worker =
            AffinitySelector::new(signals.clone(), OHIO.into(), Spill::default(), idle);
        assert_eq!(pick(&per_worker, &pools, Some("s")), "ohio-a");

        let wait = AffinitySelector::new(signals, OHIO.into(), by_latency(), idle);
        assert_eq!(pick(&wait, &pools, Some("s")), "ohio-b");
    }

    // Waits below are (requests waiting, requests served per second): 4 and
    // 8 is 0.5 s. From Ohio, Oregon is 50 ms away and Tokyo 135 ms.

    #[test]
    fn affinity_by_cost_moves_only_for_a_margin_and_does_not_flip_back() {
        let (pools, signals) = fleet();
        let idle = Duration::from_secs(60);
        let selector = AffinitySelector::new(signals.clone(), OHIO.into(), by_cost(), idle);
        set_waits(&pools, &signals, [(4, 8); 4]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-a");

        // Ohio-a 1.2 s against Ohio-b 0.5 s: under the 1 s margin, stay.
        set_waits(&pools, &signals, [(6, 5), (4, 8), (4, 8), (4, 8)]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-a");
        // Ohio-a 2 s: Ohio-b is 1.5 s cheaper, so the pin moves.
        set_waits(&pools, &signals, [(8, 4), (4, 8), (4, 8), (4, 8)]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-b");
        // The gap reverses, but by less than the margin: no flip back.
        set_waits(&pools, &signals, [(4, 8), (6, 5), (4, 8), (4, 8)]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-b");

        // Ohio-b 3 s, Ohio-a 1.5 s: Oregon (0.2 s + 50 ms) is cheapest.
        set_waits(&pools, &signals, [(6, 4), (9, 3), (1, 5), (1, 5)]);
        assert_eq!(pick(&selector, &pools, Some("s")), "oregon-a");
        // Oregon's Relay goes quiet: its cost is unknown, so the turn goes
        // to the cheapest pool whose cost is known, Tokyo.
        signals.forget_load(&pools[2].id);
        assert_eq!(pick(&selector, &pools, Some("s")), "tokyo-a");
    }

    #[test]
    fn affinity_by_cost_starts_sessions_locally_unless_a_pool_is_a_margin_cheaper() {
        let (pools, signals) = fleet();
        let idle = Duration::from_secs(60);
        let selector = AffinitySelector::new(signals.clone(), OHIO.into(), by_cost(), idle);
        // Ohio-b is the cheaper local pool at 1 s; Oregon, at 0.25 s, is
        // under a margin cheaper.
        set_waits(&pools, &signals, [(8, 4), (4, 4), (1, 5), (8, 4)]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-b");
        // Ohio-b at 1.5 s: Oregon is 1.25 s cheaper.
        set_waits(&pools, &signals, [(8, 4), (6, 4), (1, 5), (8, 4)]);
        assert_eq!(pick(&selector, &pools, Some("t")), "oregon-a");
    }

    #[test]
    fn rendezvous_by_cost_leaves_home_for_a_margin_and_comes_back() {
        let (pools, signals) = fleet();
        let selector = RendezvousSelector::new(signals.clone(), OHIO.into(), by_cost());
        set_waits(&pools, &signals, [(4, 8); 4]);
        let home = pick(&selector, &pools, Some("s"));
        let (index, other) = match home.as_str() {
            "ohio-a" => (0, "ohio-b"),
            _ => (1, "ohio-a"),
        };

        // Home waits 3 s, the other Ohio pool 0.5 s.
        let mut waits = [(4, 8); 4];
        waits[index] = (9, 3);
        set_waits(&pools, &signals, waits);
        assert_eq!(pick(&selector, &pools, Some("s")), other);

        set_waits(&pools, &signals, [(4, 8); 4]);
        assert_eq!(pick(&selector, &pools, Some("s")), home);
    }

    // Occupancies below are fractions of each pool's own capacity.

    #[test]
    fn affinity_by_headroom_moves_for_twice_the_room_and_does_not_flip_back() {
        let (pools, signals) = fleet();
        let idle = Duration::from_secs(60);
        let selector = AffinitySelector::new(signals.clone(), OHIO.into(), by_headroom(), idle);
        set_occupancy(&pools, &signals, [0.1, 0.5, 0.5, 0.5]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-a");

        // 10% against 2%: 0.9 spare against 0.98, nothing to gain.
        set_occupancy(&pools, &signals, [0.1, 0.02, 0.02, 0.02]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-a");
        // 50% against 30%: 0.5 spare against 0.7, under twice.
        set_occupancy(&pools, &signals, [0.5, 0.3, 0.3, 0.3]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-a");
        // 80% against 10%: 0.2 against 0.9, so to the nearest such pool.
        set_occupancy(&pools, &signals, [0.8, 0.1, 0.1, 0.1]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-b");
        // Ohio-a recovers to 40% and Ohio-b fills to 60%: 0.4 spare
        // against 0.6 is under twice, so no flip back.
        set_occupancy(&pools, &signals, [0.4, 0.6, 0.6, 0.6]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-b");
    }

    #[test]
    fn affinity_by_headroom_fails_over_from_a_full_pool_or_stays_if_all_are_full() {
        let (pools, signals) = fleet();
        let idle = Duration::from_secs(60);
        let selector = AffinitySelector::new(signals.clone(), OHIO.into(), by_headroom(), idle);
        set_occupancy(&pools, &signals, [0.1, 0.5, 0.5, 0.5]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-a");

        // Ohio-a is full; Ohio-b has a little room, Oregon more, but the
        // nearest pool with any room wins.
        set_occupancy(&pools, &signals, [1.2, 0.95, 0.5, 0.5]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-b");
        // Every pool is full: moving would only cost a cold prefill.
        set_occupancy(&pools, &signals, [1.1, 1.2, 1.1, 1.3]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-b");
    }

    #[test]
    fn affinity_by_headroom_lets_a_quiet_region_take_load_from_busy_ones() {
        let (pools, signals) = fleet();
        let idle = Duration::from_secs(60);
        let selector = AffinitySelector::new(signals.clone(), OHIO.into(), by_headroom(), idle);
        set_occupancy(&pools, &signals, [0.1, 0.2, 0.2, 0.2]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-a");

        // Ohio and Oregon at 90%, Tokyo at 20%: only Tokyo has twice the room.
        set_occupancy(&pools, &signals, [0.9, 0.9, 0.9, 0.2]);
        assert_eq!(pick(&selector, &pools, Some("s")), "tokyo-a");
        // A new session from Ohio goes there too.
        assert_eq!(pick(&selector, &pools, Some("t")), "tokyo-a");
        // With Ohio at 60%, a new session stays local: 0.4 spare against
        // Tokyo's 0.8 is not more than twice.
        set_occupancy(&pools, &signals, [0.6, 0.7, 0.9, 0.2]);
        assert_eq!(pick(&selector, &pools, Some("u")), "ohio-a");
    }

    #[test]
    fn affinity_forgets_idle_pins() {
        let (pools, signals) = fleet();
        let selector = AffinitySelector::new(
            signals.clone(),
            OHIO.into(),
            Spill::default(),
            Duration::from_millis(20),
        );
        set_loads(&pools, &signals, [10, 20, 20, 20]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-a");
        set_loads(&pools, &signals, [20, 10, 20, 20]);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-b");
    }
}
