// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use dynamo_global_router::Pool;

/// Every pool the Global Frontend knows about at one point in time. The
/// Global Frontend reads the current snapshot from a `watch` channel, so a
/// source of pools publishes a new one whenever a pool joins, leaves, or
/// changes its frontends.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PoolSnapshot {
    pools: Vec<Pool>,
}

impl PoolSnapshot {
    pub fn new(pools: Vec<Pool>) -> Self {
        Self { pools }
    }

    pub fn pools(&self) -> &[Pool] {
        &self.pools
    }

    /// Pools serving `model`, in catalog order.
    pub fn pools_for<'a>(&'a self, model: &'a str) -> impl Iterator<Item = &'a Pool> {
        self.pools.iter().filter(move |pool| pool.model == model)
    }

    /// Models served by at least one pool.
    pub fn models(&self) -> BTreeSet<&str> {
        self.pools.iter().map(|pool| pool.model.as_str()).collect()
    }
}

/// The router core follows the catalog as a list of pools.
impl AsRef<[Pool]> for PoolSnapshot {
    fn as_ref(&self) -> &[Pool] {
        &self.pools
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use dynamo_global_common::PoolKey;

    use super::*;

    pub(crate) fn pool(dgd: &str, model: &str, frontends: &[&str]) -> Pool {
        let key = PoolKey::new("ohio", "dynamo", dgd).unwrap();
        let frontends = frontends.iter().map(|url| url.parse().unwrap()).collect();
        Pool::new(&key, model, "us-east-2", frontends)
    }

    #[test]
    fn snapshot_groups_pools_by_model() {
        let snapshot = PoolSnapshot::new(vec![
            pool("blue", "qwen", &["http://blue:8000"]),
            pool("green", "qwen", &["http://green:8000"]),
            pool("llama", "llama", &["http://llama:8000"]),
        ]);

        assert_eq!(
            snapshot.models().into_iter().collect::<Vec<_>>(),
            ["llama", "qwen"]
        );
        let qwen = snapshot
            .pools_for("qwen")
            .map(|pool| pool.frontends[0].host_str());
        assert_eq!(qwen.collect::<Vec<_>>(), [Some("blue"), Some("green")]);
        assert_eq!(snapshot.pools_for("missing").count(), 0);
    }
}
