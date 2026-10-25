// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_global_router_shared::PoolId;
use url::Url;

use crate::catalog::Pool;

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

/// Sends every request to the first frontend of the first pool that has one.
/// A stand-in until the round-robin and load-aware policies land.
#[derive(Clone, Copy, Debug, Default)]
pub struct FirstPoolSelector;

impl PoolSelector for FirstPoolSelector {
    fn select(&self, _request: &RoutingRequest<'_>, candidates: &[&Pool]) -> Option<PoolTarget> {
        candidates.iter().find_map(|pool| {
            let frontend = pool.frontends.first()?;
            Some(PoolTarget {
                pool: pool.id.clone(),
                frontend: frontend.clone(),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::tests::pool;

    #[test]
    fn first_pool_selector_skips_pools_without_frontends() {
        let empty = pool("draining", "qwen", &[]);
        let live = pool("live", "qwen", &["http://live:8000", "http://live-2:8000"]);
        let request = RoutingRequest::new("qwen");

        let target = FirstPoolSelector
            .select(&request, &[&empty, &live])
            .unwrap();
        assert_eq!(target.pool, live.id);
        assert_eq!(target.frontend.as_str(), "http://live:8000/");
        assert_eq!(FirstPoolSelector.select(&request, &[&empty]), None);
        assert_eq!(FirstPoolSelector.select(&request, &[]), None);
    }
}
