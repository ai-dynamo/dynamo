// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dynamo_global_common::PoolId;

use crate::Pool;
use crate::placement::{headroom, place};
use crate::selector::{Cursors, PoolSelector, PoolTarget, RoutingRequest, target};
use crate::signals::PoolSignals;

/// Keeps each session's turns on one pool, so the pool's KV cache for the
/// session is reused, and moves a session only when another pool has much
/// more room.
///
/// Pools are compared by how full they are against their own capacity (see
/// [`PoolSignals::occupancy`]), so pools of any speed or size compare. A
/// request's default pool is its session's pinned pool, even in another
/// region, or else the least full pool in the nearest region. It goes
/// elsewhere only when another pool has more than `headroom_factor` times
/// the default's spare capacity, to the nearest such pool; a full default
/// sends it to the nearest pool with any room, and keeps it if none has
/// room. The session's pin moves with it.
///
/// Requests without a session ID are placed the same way and pin nothing.
/// Pins belong to this process and expire after `idle` without a request;
/// call [`Self::sweep`] periodically to free them.
#[derive(Debug)]
pub struct AffinitySelector {
    signals: Arc<PoolSignals>,
    region: String,
    headroom_factor: f64,
    idle: Duration,
    pins: DashMap<String, Pin>,
    frontends: Cursors<PoolId>,
}

#[derive(Debug)]
struct Pin {
    pool: PoolId,
    used: Instant,
}

impl AffinitySelector {
    /// `region` is the region of this process, which every request enters.
    pub fn new(
        signals: Arc<PoolSignals>,
        region: String,
        headroom_factor: f64,
        idle: Duration,
    ) -> Self {
        Self {
            signals,
            region,
            headroom_factor,
            idle,
            pins: DashMap::new(),
            frontends: Cursors::default(),
        }
    }

    /// Sessions with a live pin.
    pub fn pinned_sessions(&self) -> usize {
        self.pins.len()
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

    /// Drops expired pins. It visits every pin, so run it off the request
    /// path.
    pub fn sweep(&self) {
        self.pins.retain(|_, pin| pin.used.elapsed() <= self.idle);
    }
}

impl PoolSelector for AffinitySelector {
    fn select(&self, request: &RoutingRequest<'_>, candidates: &[&Pool]) -> Option<PoolTarget> {
        let placed = place(candidates, &self.region, &self.signals);
        let pinned = request.session.and_then(|session| {
            let pin = self.pins.get(session)?;
            if pin.used.elapsed() > self.idle {
                return None;
            }
            placed
                .iter()
                .find(|candidate| candidate.pool.id == pin.pool)
        });
        let choice = headroom(&placed, pinned.or(placed.first()), self.headroom_factor)?;
        if let Some(session) = request.session {
            self.pin(session, &choice.pool.id);
        }
        target(&self.frontends, choice.pool)
    }
}

#[cfg(test)]
mod tests {
    use crate::placement::tests::{OHIO, fleet, set_occupancy};

    use super::*;

    // Occupancies below are fractions of each pool's own capacity.

    fn selector(signals: &Arc<PoolSignals>, idle: Duration) -> AffinitySelector {
        AffinitySelector::new(signals.clone(), OHIO.into(), 2.0, idle)
    }

    fn pick(selector: &AffinitySelector, pools: &[Pool], session: Option<&str>) -> String {
        let candidates = pools.iter().collect::<Vec<_>>();
        let request = RoutingRequest::new("qwen").with_session(session);
        let target = selector.select(&request, &candidates).unwrap();
        target.frontend.host_str().unwrap().to_owned()
    }

    #[test]
    fn moves_for_twice_the_room_and_does_not_flip_back() {
        let (pools, signals) = fleet();
        let selector = selector(&signals, Duration::from_secs(60));
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
    fn fails_over_from_a_full_pool_or_stays_if_all_are_full() {
        let (pools, signals) = fleet();
        let selector = selector(&signals, Duration::from_secs(60));
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
    fn lets_a_quiet_region_take_load_from_busy_ones() {
        let (pools, signals) = fleet();
        let selector = selector(&signals, Duration::from_secs(60));
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
    fn requests_without_a_session_are_placed_but_not_pinned() {
        let (pools, signals) = fleet();
        let selector = selector(&signals, Duration::from_secs(60));
        set_occupancy(&pools, &signals, [0.5, 0.3, 0.9, 0.0]);
        assert_eq!(pick(&selector, &pools, None), "ohio-b");
        set_occupancy(&pools, &signals, [0.9, 0.9, 0.9, 0.0]);
        assert_eq!(pick(&selector, &pools, None), "tokyo-a");
        assert_eq!(selector.pinned_sessions(), 0);
    }

    #[test]
    fn a_pool_whose_load_is_unknown_counts_as_full() {
        let (pools, signals) = fleet();
        let selector = selector(&signals, Duration::from_secs(60));
        set_occupancy(&pools, &signals, [0.1, 0.5, 0.5, 0.5]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-a");

        // Ohio-a's Relay goes quiet while every other pool is full: the
        // session stays rather than move to a full pool.
        set_occupancy(&pools, &signals, [0.1, 1.0, 1.1, 1.0]);
        signals.forget_load(&pools[0].id);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-a");
        // Once a pool has room, the session goes to the nearest such pool.
        set_occupancy(&pools, &signals, [0.1, 1.0, 0.9, 0.2]);
        signals.forget_load(&pools[0].id);
        assert_eq!(pick(&selector, &pools, Some("s")), "oregon-a");
    }

    #[test]
    fn forgets_idle_pins() {
        let (pools, signals) = fleet();
        let selector = selector(&signals, Duration::from_millis(200));
        set_occupancy(&pools, &signals, [0.1, 0.5, 0.5, 0.5]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-a");
        // 0.5 spare against 0.9 is under twice, so the pin holds...
        set_occupancy(&pools, &signals, [0.5, 0.1, 0.5, 0.5]);
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-a");
        // ...until it expires, and the session starts over on the emptier pool.
        std::thread::sleep(Duration::from_millis(250));
        assert_eq!(pick(&selector, &pools, Some("s")), "ohio-b");
        selector.sweep();
        assert_eq!(selector.pinned_sessions(), 1);
    }

    #[test]
    fn sweep_drops_expired_pins() {
        let (pools, signals) = fleet();
        let selector = selector(&signals, Duration::ZERO);
        set_occupancy(&pools, &signals, [0.1, 0.5, 0.5, 0.5]);
        pick(&selector, &pools, Some("s"));
        assert_eq!(selector.pinned_sessions(), 1);
        std::thread::sleep(Duration::from_millis(1));
        selector.sweep();
        assert_eq!(selector.pinned_sessions(), 0);
    }
}
