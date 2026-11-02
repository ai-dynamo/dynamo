// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Borrow;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use dashmap::DashMap;
use dynamo_global_common::PoolId;
use url::Url;

use crate::Pool;
use crate::load::{InFlightGuard, InFlightTracker};
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

/// What a selector holds for one request from its pool choice until the
/// request ends. Dropping it ends the request wherever the selector counts
/// it.
pub trait RequestLifecycle: Send {}

/// A pool choice, with whatever the selector holds for the request.
pub struct Selection<'a> {
    pub target: PoolTarget<'a>,
    /// `None` when the selector keeps nothing per request.
    pub lifecycle: Option<Box<dyn RequestLifecycle>>,
    /// The request's count in flight to the pool, when the selector started
    /// it in the pick, so that picks counting the same requests see it at
    /// once; otherwise `None` and the frontend starts it after the pick.
    pub in_flight: Option<InFlightGuard>,
}

impl<'a> From<PoolTarget<'a>> for Selection<'a> {
    fn from(target: PoolTarget<'a>) -> Self {
        Self {
            target,
            lifecycle: None,
            in_flight: None,
        }
    }
}

/// Chooses a pool, and a frontend in it, for each request. Selection may
/// wait, and may hold state for the request: the router core's selector
/// books the request on its pool.
#[async_trait]
pub trait PoolSelector: Send + Sync {
    /// `candidates` are the pools serving the request's model. `None` means
    /// none of them can take the request.
    async fn select<'a>(
        &self,
        request: &RoutingRequest<'_>,
        candidates: &[&'a Pool],
    ) -> Option<Selection<'a>>;
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

#[async_trait]
impl PoolSelector for RoundRobinSelector {
    async fn select<'a>(
        &self,
        request: &RoutingRequest<'_>,
        candidates: &[&'a Pool],
    ) -> Option<Selection<'a>> {
        let reachable = || candidates.iter().copied().filter(|pool| reachable(pool));
        let position = self.pools.next(request.model, reachable().count())?;
        target(&self.frontends, reachable().nth(position)?).map(Selection::from)
    }
}

/// Sends each request to the nearest pool with the fewest requests in
/// flight, with affinity's placement rule ([`crate::AffinitySelector`]) on
/// requests in flight instead of occupancy.
///
/// A pool's load is this process's requests in flight to it over the
/// requests its workers can run at once, as its Relay reports
/// ([`PoolSignals::capacity`]); without a report the load is unknown, which
/// counts as full. Pools are ordered nearest region first, then less loaded,
/// then fewer in flight. The request goes to the first, rotating among
/// pools that tie with it, unless another pool has more than
/// `headroom_factor` times its spare room. Without capacities or round
/// trips the rule is the fewest in flight, ties rotating.
///
/// A slower pool holds its requests longer, so it shows more in flight and
/// gets a smaller share. Only this process's requests count: replicas do not
/// share counts on this path.
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

#[async_trait]
impl PoolSelector for LeastInFlightSelector {
    async fn select<'a>(
        &self,
        _request: &RoutingRequest<'_>,
        candidates: &[&'a Pool],
    ) -> Option<Selection<'a>> {
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
        target(&self.frontends, choice).map(Selection::from)
    }
}

/// Least-in-flight's load for a pool: `in_flight` over the requests its
/// workers can run at once. `None` without a capacity.
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
    use dynamo_global_common::PoolKey;

    use crate::signals::{LoadReport, RelaySource};

    use super::*;

    pub(crate) fn pool(dgd: &str, model: &str, frontends: &[&str]) -> Pool {
        regional_pool("us-east-2", dgd, model, frontends)
    }

    /// Least-in-flight in Ohio, counting in `in_flight`, with no pool's
    /// capacity or round trip known yet.
    pub(crate) fn least_in_flight(in_flight: &Arc<InFlightTracker>) -> LeastInFlightSelector {
        least_in_flight_with(in_flight, &Arc::new(PoolSignals::new(in_flight.clone())))
    }

    pub(crate) fn least_in_flight_with(
        in_flight: &Arc<InFlightTracker>,
        signals: &Arc<PoolSignals>,
    ) -> LeastInFlightSelector {
        LeastInFlightSelector::new(in_flight.clone(), signals.clone(), "us-east-2".into(), 2.0)
    }

    /// Records that `pool`'s workers can run `capacity` requests at once.
    pub(crate) fn set_capacity(signals: &PoolSignals, pool: &Pool, capacity: u64) {
        let source = RelaySource {
            runtime_id: 1,
            incarnation: 1,
        };
        let report = LoadReport {
            in_flight: 0,
            generating: 0,
            max_concurrency: Some(capacity),
            window: std::time::Duration::from_secs(60),
        };
        signals.record_load(&pool.id, source, report);
    }

    pub(crate) fn regional_pool(region: &str, dgd: &str, model: &str, frontends: &[&str]) -> Pool {
        let key = PoolKey::new(region, "dynamo", dgd).unwrap();
        let frontends = frontends.iter().map(|url| url.parse().unwrap()).collect();
        Pool::new(&key, model, region, frontends)
    }

    async fn picks(selector: &RoundRobinSelector, model: &str, candidates: &[&Pool]) -> String {
        let selection = selector
            .select(&RoutingRequest::new(model), candidates)
            .await;
        let target = selection.unwrap().target;
        target.frontend.host_str().unwrap().to_owned()
    }

    #[tokio::test]
    async fn rotates_pools_then_their_frontends() {
        let blue = pool(
            "blue",
            "qwen",
            &["http://blue-0:8000", "http://blue-1:8000"],
        );
        let green = pool("green", "qwen", &["http://green-0:8000"]);
        let selector = RoundRobinSelector::new();

        let mut order = Vec::new();
        for _ in 0..6 {
            order.push(picks(&selector, "qwen", &[&blue, &green]).await);
        }
        assert_eq!(
            order,
            [
                "blue-0", "green-0", "blue-1", "green-0", "blue-0", "green-0"
            ]
        );
    }

    #[tokio::test]
    async fn models_rotate_independently() {
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

        assert_eq!(picks(&selector, "qwen", &qwen).await, "qwen-a");
        assert_eq!(picks(&selector, "llama", &llama).await, "llama-a");
        assert_eq!(picks(&selector, "qwen", &qwen).await, "qwen-b");
        assert_eq!(picks(&selector, "llama", &llama).await, "llama-b");
    }

    #[tokio::test]
    async fn skips_pools_without_frontends() {
        let empty = pool("draining", "qwen", &[]);
        let live = pool("live", "qwen", &["http://live:8000"]);
        let selector = RoundRobinSelector::new();
        let request = RoutingRequest::new("qwen");

        for _ in 0..3 {
            let target = selector
                .select(&request, &[&empty, &live])
                .await
                .unwrap()
                .target;
            assert_eq!(target.pool.id, live.id);
        }
        assert!(selector.select(&request, &[&empty]).await.is_none());
        assert!(selector.select(&request, &[]).await.is_none());
    }

    #[tokio::test]
    async fn least_in_flight_picks_the_pool_with_fewest_requests() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let in_flight = Arc::new(InFlightTracker::new());
        let selector = least_in_flight(&in_flight);
        let request = RoutingRequest::new("qwen");

        let _busy = [in_flight.start(&blue.id), in_flight.start(&blue.id)];
        let _one = in_flight.start(&green.id);
        for _ in 0..10 {
            let target = selector
                .select(&request, &[&blue, &green])
                .await
                .unwrap()
                .target;
            assert_eq!(target.pool.id, green.id);
        }
    }

    #[tokio::test]
    async fn least_in_flight_spreads_ties_and_skips_pools_without_frontends() {
        let empty = pool("draining", "qwen", &[]);
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let selector = least_in_flight(&Arc::new(InFlightTracker::new()));
        let request = RoutingRequest::new("qwen");

        let mut chosen = Vec::new();
        for _ in 0..4 {
            let selection = selector.select(&request, &[&empty, &blue, &green]).await;
            chosen.push(selection.unwrap().target.pool.id.clone());
        }
        let (blue, green) = (blue.id.clone(), green.id.clone());
        assert_eq!(chosen, [blue.clone(), green.clone(), blue, green]);
        assert!(selector.select(&request, &[&empty]).await.is_none());
    }

    /// Ohio and Tokyo, 135 ms apart, each able to run 100 requests at once.
    fn two_regions(in_flight: &Arc<InFlightTracker>) -> (Pool, Pool, Arc<PoolSignals>) {
        let ohio = regional_pool("us-east-2", "ohio", "qwen", &["http://ohio:8000"]);
        let tokyo = regional_pool("ap-northeast-1", "tokyo", "qwen", &["http://tokyo:8000"]);
        let signals = Arc::new(PoolSignals::new(in_flight.clone()));
        signals.set_region_round_trips(std::collections::HashMap::from([(
            "ap-northeast-1".to_owned(),
            std::time::Duration::from_millis(135),
        )]));
        set_capacity(&signals, &ohio, 100);
        set_capacity(&signals, &tokyo, 100);
        (ohio, tokyo, signals)
    }

    async fn picked(selector: &LeastInFlightSelector, candidates: &[&Pool]) -> PoolId {
        let selection = selector
            .select(&RoutingRequest::new("qwen"), candidates)
            .await;
        selection.unwrap().target.pool.id.clone()
    }

    /// The nearest pool, though a farther one has fewer in flight, until the
    /// farther one has more than twice its spare room.
    #[tokio::test]
    async fn least_in_flight_stays_in_its_region_until_another_has_much_more_room() {
        let in_flight = Arc::new(InFlightTracker::new());
        let (ohio, tokyo, signals) = two_regions(&in_flight);
        let selector = least_in_flight_with(&in_flight, &signals);

        // Ohio 5 of 100, Tokyo idle: 0.95 spare against 1.0.
        let mut held = (0..5)
            .map(|_| in_flight.start(&ohio.id))
            .collect::<Vec<_>>();
        assert_eq!(picked(&selector, &[&tokyo, &ohio]).await, ohio.id);
        // Ohio 55 of 100: 0.45 spare, and Tokyo's 1.0 is more than twice it.
        held.extend((0..50).map(|_| in_flight.start(&ohio.id)));
        assert_eq!(picked(&selector, &[&tokyo, &ohio]).await, tokyo.id);
    }

    /// A pool whose capacity is unknown counts as full, as under affinity:
    /// the request leaves it for a pool with known room, and stays if none
    /// has.
    #[tokio::test]
    async fn least_in_flight_counts_an_unknown_capacity_as_full() {
        let in_flight = Arc::new(InFlightTracker::new());
        let (ohio, tokyo, signals) = two_regions(&in_flight);
        signals.forget(&ohio.id);
        let selector = least_in_flight_with(&in_flight, &signals);
        assert_eq!(picked(&selector, &[&ohio, &tokyo]).await, tokyo.id);
        signals.forget(&tokyo.id);
        assert_eq!(picked(&selector, &[&ohio, &tokyo]).await, ohio.id);
    }

    /// Pools in the region with the same load and count rotate.
    #[tokio::test]
    async fn least_in_flight_rotates_among_equal_pools_in_its_region() {
        let in_flight = Arc::new(InFlightTracker::new());
        let (ohio, tokyo, signals) = two_regions(&in_flight);
        let ohio_b = regional_pool("us-east-2", "ohio-b", "qwen", &["http://ohio-b:8000"]);
        set_capacity(&signals, &ohio_b, 100);
        let selector = least_in_flight_with(&in_flight, &signals);
        let candidates = [&tokyo, &ohio, &ohio_b];
        let mut seen = Vec::new();
        for _ in 0..4 {
            seen.push(picked(&selector, &candidates).await);
        }
        assert_eq!(
            seen,
            [ohio.id.clone(), ohio_b.id.clone(), ohio.id, ohio_b.id]
        );
    }
}
