// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use dynamo_global_common::Pool;
use tokio::sync::watch;

/// Every pool a catalog knows about at one point in time.
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

/// A source of pools. An implementation publishes a new snapshot whenever a
/// pool joins, leaves, or changes its frontends.
pub trait PoolCatalog: Send + Sync {
    /// The receiver holds the current snapshot, and its `changed()` waits for
    /// the next one. It returns an error once the catalog is dropped.
    fn subscribe(&self) -> watch::Receiver<Arc<PoolSnapshot>>;
}

/// A catalog that never changes, for pools given on the command line and
/// for tests.
pub struct StaticPoolCatalog {
    snapshot: watch::Sender<Arc<PoolSnapshot>>,
}

impl StaticPoolCatalog {
    pub fn new(pools: Vec<Pool>) -> Self {
        let (snapshot, _) = watch::channel(Arc::new(PoolSnapshot::new(pools)));
        Self { snapshot }
    }
}

impl PoolCatalog for StaticPoolCatalog {
    fn subscribe(&self) -> watch::Receiver<Arc<PoolSnapshot>> {
        self.snapshot.subscribe()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use dynamo_global_common::{PoolIdDeriver, PoolKey, PoolLocation, V1PoolIdDeriver};

    use super::*;

    pub(crate) fn pool(dgd: &str, model: &str, frontends: &[&str]) -> Pool {
        Pool {
            id: V1PoolIdDeriver.derive(&PoolKey::new("ohio", "dynamo", dgd).unwrap()),
            model: model.to_owned(),
            location: PoolLocation {
                region: "us-east-2".to_owned(),
                availability_zone: None,
                cluster: None,
                datacenter: None,
            },
            frontends: frontends.iter().map(|url| url.parse().unwrap()).collect(),
        }
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

    #[test]
    fn static_catalog_serves_its_pools_to_every_subscriber() {
        let catalog = StaticPoolCatalog::new(vec![pool("blue", "qwen", &["http://blue:8000"])]);
        let first = catalog.subscribe();
        let second = catalog.subscribe();

        assert_eq!(first.borrow().pools().len(), 1);
        assert_eq!(*first.borrow(), *second.borrow());
    }
}
