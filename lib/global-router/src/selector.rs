// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Borrow;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dashmap::DashMap;
use dynamo_global_common::PoolId;
use url::Url;

use crate::Pool;
use crate::load::InFlightTracker;
use crate::placement::{self, Placed};
use crate::signals::PoolSignals;

/// What a selector may know about a request. Routing inputs such as the
/// client's region become fields here.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct RoutingRequest<'a> {
    pub model: &'a str,
    /// The session the request belongs to, from `X-Dynamo-Session-ID`.
    pub session: Option<&'a str>,
}

impl<'a> RoutingRequest<'a> {
    pub fn new(model: &'a str) -> Self {
        Self {
            model,
            session: None,
        }
    }

    pub fn with_session(mut self, session: Option<&'a str>) -> Self {
        self.session = session;
        self
    }
}

/// Where to send one request: one of the candidates, and one of its
/// frontends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolTarget<'a> {
    pub pool: &'a Pool,
    pub frontend: &'a Url,
}

/// Chooses a pool, and a frontend in it, for each request.
pub trait PoolSelector: Send + Sync {
    /// `candidates` are the pools serving the request's model. `None` means
    /// none of them can take the request.
    fn select<'a>(
        &self,
        request: &RoutingRequest<'_>,
        candidates: &[&'a Pool],
    ) -> Option<PoolTarget<'a>>;
}

/// Rotates through the pools serving each model, then through the chosen
/// pool's frontends, so every pool gets an equal share of a model's requests
/// however many frontends it has.
///
/// Each model and each pool has its own cursor. When the candidates change,
/// the cursor carries on over the new list.
#[derive(Debug, Default)]
pub struct RoundRobinSelector {
    pools: Cursors<String>,
    frontends: Cursors<PoolId>,
}

impl RoundRobinSelector {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PoolSelector for RoundRobinSelector {
    fn select<'a>(
        &self,
        request: &RoutingRequest<'_>,
        candidates: &[&'a Pool],
    ) -> Option<PoolTarget<'a>> {
        let reachable = || candidates.iter().copied().filter(|pool| reachable(pool));
        let position = self.pools.next(request.model, reachable().count())?;
        target(&self.frontends, reachable().nth(position)?)
    }
}

/// Sends each request to the nearest pool with the fewest requests in
/// flight, by affinity's placement rule with load = requests in flight over
/// the pool's capacity ([`PoolSignals::capacity`]). Ties rotate.
///
/// A slower pool holds its requests longer, so it shows more in flight and
/// gets a smaller share. Only this process's requests count: replicas do not
/// share counts.
#[derive(Debug)]
pub struct LeastInFlightSelector {
    in_flight: Arc<InFlightTracker>,
    signals: Arc<PoolSignals>,
    region: String,
    headroom_factor: f64,
    ties: AtomicUsize,
    frontends: Cursors<PoolId>,
}

impl LeastInFlightSelector {
    /// `in_flight` must be the tracker the Global Frontend counts its
    /// requests in; `region` is the region of this process, which every
    /// request enters.
    pub fn new(
        in_flight: Arc<InFlightTracker>,
        signals: Arc<PoolSignals>,
        region: String,
        headroom_factor: f64,
    ) -> Self {
        Self {
            in_flight,
            signals,
            region,
            headroom_factor,
            ties: AtomicUsize::new(0),
            frontends: Cursors::default(),
        }
    }
}

impl PoolSelector for LeastInFlightSelector {
    fn select<'a>(
        &self,
        _request: &RoutingRequest<'_>,
        candidates: &[&'a Pool],
    ) -> Option<PoolTarget<'a>> {
        let round_trips = self.signals.region_round_trips();
        // Read each count once: requests start and finish while this runs.
        let mut placed = candidates
            .iter()
            .copied()
            .filter(|pool| reachable(pool))
            .map(|pool| {
                let in_flight = self.in_flight.get(&pool.id);
                Placed {
                    item: pool,
                    load: in_flight_load(in_flight, self.signals.capacity(&pool.id)),
                    distance: placement::distance(
                        &pool.location.region,
                        &self.region,
                        &round_trips,
                    ),
                    in_flight,
                }
            })
            .collect::<Vec<_>>();
        placement::order(&mut placed);
        let tied = placement::leading_ties(&placed);
        if tied == 0 {
            return None;
        }
        let default = self.ties.fetch_add(1, Ordering::Relaxed) % tied;
        let choice = placed[placement::headroom(&placed, default, self.headroom_factor)].item;
        target(&self.frontends, choice)
    }
}

/// `in_flight` over capacity; `None` without a capacity.
pub(crate) fn in_flight_load(in_flight: u64, capacity: Option<u64>) -> Option<f64> {
    capacity
        .filter(|capacity| *capacity > 0)
        .map(|capacity| in_flight as f64 / capacity as f64)
}

/// Whether the pool has a frontend to send to.
pub(crate) fn reachable(pool: &Pool) -> bool {
    !pool.frontends.is_empty()
}

/// `pool`, with the next of its frontends in rotation.
pub(crate) fn target<'a>(frontends: &Cursors<PoolId>, pool: &'a Pool) -> Option<PoolTarget<'a>> {
    let frontend = &pool.frontends[frontends.next(&pool.id, pool.frontends.len())?];
    Some(PoolTarget { pool, frontend })
}

/// One rotating cursor per key.
#[derive(Debug)]
pub(crate) struct Cursors<K: Eq + Hash>(DashMap<K, AtomicUsize>);

impl<K: Eq + Hash> Default for Cursors<K> {
    fn default() -> Self {
        Self(DashMap::new())
    }
}

impl<K: Eq + Hash> Cursors<K> {
    /// Advances `key`'s cursor over `count` positions and returns the chosen
    /// one, or `None` when there are none.
    pub(crate) fn next<Q>(&self, key: &Q, count: usize) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: ToOwned<Owned = K> + Eq + Hash + ?Sized,
    {
        if count == 0 {
            return None;
        }
        let turn = match self.0.get(key) {
            Some(cursor) => cursor.fetch_add(1, Ordering::Relaxed),
            None => self
                .0
                .entry(key.to_owned())
                .or_default()
                .fetch_add(1, Ordering::Relaxed),
        };
        Some(turn % count)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use dynamo_global_common::PoolKey;

    use crate::signals::{LoadReport, RelaySource};

    use super::*;

    pub(crate) fn pool(dgd: &str, model: &str, frontends: &[&str]) -> Pool {
        regional_pool("us-east-2", dgd, model, frontends)
    }

    pub(crate) fn least_in_flight(
        in_flight: &Arc<InFlightTracker>,
        signals: &Arc<PoolSignals>,
    ) -> LeastInFlightSelector {
        LeastInFlightSelector::new(in_flight.clone(), signals.clone(), "us-east-2".into(), 2.0)
    }

    /// Ohio and Tokyo, 135 ms apart, each able to run 100 requests at once.
    pub(crate) fn two_regions(in_flight: &Arc<InFlightTracker>) -> (Pool, Pool, Arc<PoolSignals>) {
        let ohio = regional_pool("us-east-2", "ohio", "qwen", &["http://ohio:8000"]);
        let tokyo = regional_pool("ap-northeast-1", "tokyo", "qwen", &["http://tokyo:8000"]);
        let signals = Arc::new(PoolSignals::new(in_flight.clone()));
        signals.set_region_round_trips(HashMap::from([(
            "ap-northeast-1".to_owned(),
            Duration::from_millis(135),
        )]));
        for pool in [&ohio, &tokyo] {
            let source = RelaySource {
                runtime_id: 1,
                incarnation: 1,
            };
            let report = LoadReport {
                in_flight: 0,
                generating: 0,
                max_concurrency: Some(100),
                window: Duration::from_secs(60),
            };
            signals.record_load(&pool.id, source, report);
        }
        (ohio, tokyo, signals)
    }

    pub(crate) fn regional_pool(region: &str, dgd: &str, model: &str, frontends: &[&str]) -> Pool {
        let key = PoolKey::new(region, "dynamo", dgd).unwrap();
        let frontends = frontends.iter().map(|url| url.parse().unwrap()).collect();
        Pool::new(&key, model, region, frontends)
    }

    fn picks(selector: &RoundRobinSelector, model: &str, candidates: &[&Pool]) -> String {
        let target = selector
            .select(&RoutingRequest::new(model), candidates)
            .unwrap();
        target.frontend.host_str().unwrap().to_owned()
    }

    #[test]
    fn rotates_pools_then_their_frontends() {
        let blue = pool(
            "blue",
            "qwen",
            &["http://blue-0:8000", "http://blue-1:8000"],
        );
        let green = pool("green", "qwen", &["http://green-0:8000"]);
        let selector = RoundRobinSelector::new();

        let order = (0..6)
            .map(|_| picks(&selector, "qwen", &[&blue, &green]))
            .collect::<Vec<_>>();
        assert_eq!(
            order,
            [
                "blue-0", "green-0", "blue-1", "green-0", "blue-0", "green-0"
            ]
        );
    }

    #[test]
    fn models_rotate_independently() {
        let qwen = [
            pool("qwen-a", "qwen", &["http://qwen-a:8000"]),
            pool("qwen-b", "qwen", &["http://qwen-b:8000"]),
        ];
        let llama = [
            pool("llama-a", "llama", &["http://llama-a:8000"]),
            pool("llama-b", "llama", &["http://llama-b:8000"]),
        ];
        let qwen = qwen.iter().collect::<Vec<_>>();
        let llama = llama.iter().collect::<Vec<_>>();
        let selector = RoundRobinSelector::new();

        assert_eq!(picks(&selector, "qwen", &qwen), "qwen-a");
        assert_eq!(picks(&selector, "llama", &llama), "llama-a");
        assert_eq!(picks(&selector, "qwen", &qwen), "qwen-b");
        assert_eq!(picks(&selector, "llama", &llama), "llama-b");
    }

    #[test]
    fn skips_pools_without_frontends() {
        let empty = pool("draining", "qwen", &[]);
        let live = pool("live", "qwen", &["http://live:8000"]);
        let selector = RoundRobinSelector::new();
        let request = RoutingRequest::new("qwen");

        for _ in 0..3 {
            let target = selector.select(&request, &[&empty, &live]).unwrap();
            assert_eq!(target.pool.id, live.id);
        }
        assert_eq!(selector.select(&request, &[&empty]), None);
        assert_eq!(selector.select(&request, &[]), None);
    }

    #[test]
    fn least_in_flight_picks_the_pool_with_fewest_requests() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let in_flight = Arc::new(InFlightTracker::new());
        let selector = least_in_flight(&in_flight, &Arc::new(PoolSignals::new(in_flight.clone())));
        let request = RoutingRequest::new("qwen");

        let _busy = [in_flight.start(&blue.id), in_flight.start(&blue.id)];
        let _one = in_flight.start(&green.id);
        for _ in 0..10 {
            let target = selector.select(&request, &[&blue, &green]).unwrap();
            assert_eq!(target.pool.id, green.id);
        }
    }

    #[test]
    fn least_in_flight_spreads_ties_and_skips_pools_without_frontends() {
        let empty = pool("draining", "qwen", &[]);
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let in_flight = Arc::new(InFlightTracker::new());
        let selector = least_in_flight(&in_flight, &Arc::new(PoolSignals::new(in_flight.clone())));
        let request = RoutingRequest::new("qwen");

        let chosen = (0..4)
            .map(|_| {
                selector
                    .select(&request, &[&empty, &blue, &green])
                    .unwrap()
                    .pool
                    .id
                    .clone()
            })
            .collect::<Vec<_>>();
        let (blue, green) = (blue.id.clone(), green.id.clone());
        assert_eq!(chosen, [blue.clone(), green.clone(), blue, green]);
        assert_eq!(selector.select(&request, &[&empty]), None);
    }

    /// Stays local until another region has twice the room; unknown capacity counts as full.
    #[test]
    fn least_in_flight_stays_in_its_region_until_another_has_much_more_room() {
        let in_flight = Arc::new(InFlightTracker::new());
        let (ohio, tokyo, signals) = two_regions(&in_flight);
        let selector = least_in_flight(&in_flight, &signals);
        let picked = || {
            let request = RoutingRequest::new("qwen");
            selector
                .select(&request, &[&tokyo, &ohio])
                .unwrap()
                .pool
                .id
                .clone()
        };

        // Ohio 5 of 100, Tokyo idle: 0.95 spare against 1.0.
        let mut held = (0..5)
            .map(|_| in_flight.start(&ohio.id))
            .collect::<Vec<_>>();
        assert_eq!(picked(), ohio.id);
        // Ohio 55 of 100: 0.45 spare, and Tokyo's 1.0 is more than twice it.
        held.extend((0..50).map(|_| in_flight.start(&ohio.id)));
        assert_eq!(picked(), tokyo.id);
        // Tokyo's capacity unknown: it counts as full, so Ohio keeps it.
        signals.forget(&tokyo.id);
        assert_eq!(picked(), ohio.id);
    }
}
