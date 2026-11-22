// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use dynamo_global_discovery_client::{GlobalDiscoveryClient, PoolCache};
use dynamo_global_router::Pool;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::PoolSnapshot;

/// Candidate pools from GDS. Partial lists add and update pools; only complete
/// lists remove them. Relay liveness independently decides which are routable.
pub fn discover_pools(
    client: GlobalDiscoveryClient,
    interval: Duration,
    cancel: CancellationToken,
) -> watch::Receiver<Arc<PoolSnapshot>> {
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
                .filter_map(|pool| {
                    let model = pool.model.as_deref()?;
                    Some(Pool::new(
                        &pool.key,
                        model,
                        &pool.location.region,
                        pool.frontends.clone(),
                    ))
                })
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
    receiver
}
