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

/// Sends each request to the pool with the fewest of this process's requests
/// in flight.
///
/// A slower pool holds its requests longer, so it shows more in flight and
/// gets a smaller share. Only this process's requests count: replicas do not
/// share counts, and the pools' own load reports are not read. Ties rotate
/// among the tied pools.
#[derive(Debug)]
pub struct LeastInFlightSelector {
    in_flight: Arc<InFlightTracker>,
    ties: AtomicUsize,
    frontends: Cursors<PoolId>,
}

impl LeastInFlightSelector {
    /// `in_flight` must be the tracker the Global Frontend counts its
    /// requests in.
    pub fn new(in_flight: Arc<InFlightTracker>) -> Self {
        Self {
            in_flight,
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
        // Read each count once: requests start and finish while this runs.
        let counts = candidates
            .iter()
            .copied()
            .filter(|pool| reachable(pool))
            .map(|pool| (pool, self.in_flight.get(&pool.id)))
            .collect::<Vec<_>>();
        let fewest = counts.iter().map(|(_, count)| *count).min()?;
        let tied = counts.iter().filter(|(_, count)| *count == fewest);
        let turn = self.ties.fetch_add(1, Ordering::Relaxed) % tied.clone().count();
        let (pool, _) = tied.clone().nth(turn)?;
        target(&self.frontends, pool)
    }
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

    use super::*;

    pub(crate) fn pool(dgd: &str, model: &str, frontends: &[&str]) -> Pool {
        regional_pool("us-east-2", dgd, model, frontends)
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
        let selector = LeastInFlightSelector::new(in_flight.clone());
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
        let selector = LeastInFlightSelector::new(Arc::new(InFlightTracker::new()));
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
}
