// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Borrow;
use std::hash::Hash;

use dashmap::DashMap;
use dynamo_global_common::PoolId;
use dynamo_runtime::pipeline::BuiltinRoutePicker;
use url::Url;

use dynamo_global_common::Pool;

/// What a selector may know about a request. Routing inputs such as the
/// session ID or the client's region become fields here.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct RoutingRequest<'a> {
    pub model: &'a str,
}

impl<'a> RoutingRequest<'a> {
    pub fn new(model: &'a str) -> Self {
        Self { model }
    }
}

/// Where to send one request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolTarget {
    pub pool: PoolId,
    pub frontend: Url,
}

/// Chooses a pool, and a frontend in it, for each request.
pub trait PoolSelector: Send + Sync {
    /// `candidates` are the pools serving the request's model. `None` means
    /// none of them can take the request.
    fn select(&self, request: &RoutingRequest<'_>, candidates: &[&Pool]) -> Option<PoolTarget>;
}

/// Rotates through the pools serving each model, then through the chosen
/// pool's frontends, so every pool gets an equal share of a model's requests
/// however many frontends it has.
///
/// Each model and each pool has its own cursor, kept by the runtime's
/// round-robin picker. When the candidates change, the cursor carries on over
/// the new list.
#[derive(Debug, Default)]
pub struct RoundRobinSelector {
    pools: DashMap<String, BuiltinRoutePicker>,
    frontends: DashMap<PoolId, BuiltinRoutePicker>,
}

impl RoundRobinSelector {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PoolSelector for RoundRobinSelector {
    fn select(&self, request: &RoutingRequest<'_>, candidates: &[&Pool]) -> Option<PoolTarget> {
        let reachable = candidates
            .iter()
            .copied()
            .filter(|pool| !pool.frontends.is_empty())
            .collect::<Vec<_>>();
        let pool = reachable[rotate(&self.pools, request.model, reachable.len())?];
        let frontend = &pool.frontends[rotate(&self.frontends, &pool.id, pool.frontends.len())?];
        Some(PoolTarget {
            pool: pool.id.clone(),
            frontend: frontend.clone(),
        })
    }
}

/// Advances `key`'s cursor over `count` candidates and returns the chosen
/// position, or `None` when there are no candidates.
fn rotate<K, Q>(pickers: &DashMap<K, BuiltinRoutePicker>, key: &Q, count: usize) -> Option<usize>
where
    K: Borrow<Q> + Eq + Hash,
    Q: ToOwned<Owned = K> + Eq + Hash + ?Sized,
{
    // The picker chooses among worker IDs. Candidate positions stand in for
    // them, since round robin only counts.
    let positions = (0..count as u64).collect::<Vec<_>>();
    let pick = |picker: &BuiltinRoutePicker| picker.select_worker(&positions, |_| 0);
    let position = match pickers.get(key) {
        Some(picker) => pick(&picker),
        None => pick(
            &pickers
                .entry(key.to_owned())
                .or_insert_with(BuiltinRoutePicker::round_robin),
        ),
    };
    position.map(|position| position as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::tests::pool;

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
            assert_eq!(target.pool, live.id);
        }
        assert_eq!(selector.select(&request, &[&empty]), None);
        assert_eq!(selector.select(&request, &[]), None);
    }
}
