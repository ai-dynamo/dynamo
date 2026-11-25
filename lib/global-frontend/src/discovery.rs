// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use dynamo_global_discovery_client::{GlobalDiscoveryClient, PoolCache};
use dynamo_global_router::Pool;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::PoolSnapshot;

/// Pools from Global Discovery. A listing says where a pool is, not what it
/// serves, so each pool's `model` is empty, and only
/// [`crate::RelayGatedPoolCatalog::for_discovery`] takes them: it gives each
/// pool the models its live Relay views report ready.
#[derive(Clone)]
pub struct DiscoveredPools(pub(crate) watch::Receiver<Arc<PoolSnapshot>>);

impl DiscoveredPools {
    /// The pools as last listed.
    pub fn borrow(&self) -> watch::Ref<'_, Arc<PoolSnapshot>> {
        self.0.borrow()
    }

    /// As [`Self::borrow`], marking the pools seen.
    pub fn borrow_and_update(&mut self) -> watch::Ref<'_, Arc<PoolSnapshot>> {
        self.0.borrow_and_update()
    }

    /// Waits for pools not yet seen. Fails once discovery stops.
    pub async fn changed(&mut self) -> Result<(), watch::error::RecvError> {
        self.0.changed().await
    }
}

/// Candidate pools from GDS. Partial lists add and update pools; only complete
/// lists remove them.
pub fn discover_pools(
    client: GlobalDiscoveryClient,
    interval: Duration,
    cancel: CancellationToken,
) -> DiscoveredPools {
    let cache = PoolCache::spawn(client, interval, cancel.clone());
    let (sender, receiver) = watch::channel(Arc::new(PoolSnapshot::default()));
    tokio::spawn(async move {
        // Own the cache until the adapter stops, including while discovery is
        // unchanged. Dropping it stops its refresh task.
        let mut discovered = cache.subscribe();
        loop {
            let mut pools = discovered
                .borrow_and_update()
                .values()
                .map(|pool| Pool::new(&pool.key, "", &pool.location.region, pool.frontends.clone()))
                .collect::<Vec<_>>();
            // The cache is a map; stable order avoids changing selectors or
            // notifying subscribers when only the map's iteration order differs.
            pools.sort_unstable_by(|a, b| a.name.cmp(&b.name));
            let next = Arc::new(PoolSnapshot::new(pools));
            sender.send_if_modified(|current| {
                if **current == *next {
                    return false;
                }
                *current = next;
                true
            });
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = sender.closed() => break,
                changed = discovered.changed() => {
                    if changed.is_err() {
                        break;
                    }
                },
            }
        }
    });
    DiscoveredPools(receiver)
}
