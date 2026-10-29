// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pools as the router core's workers.
//!
//! The core selects among workers of one model in one routing group. Each
//! pool with a frontend becomes one worker with one data-parallel rank, and
//! all pools share the routing group [`ROUTING_GROUP`], so the pools of a
//! model form one partition. The worker id is a hash of the pool id, which
//! both sides can compute, so a chosen worker maps back to its pool without
//! a table.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_global_common::PoolId;
use dynamo_kv_router::WorkerId;
use dynamo_kv_router::services::selection::{WorkerCatalogSource, WorkerRequest};
use tokio::sync::watch;
use xxhash_rust::xxh3::xxh3_64;

use crate::Pool;
use crate::selector::reachable;

/// The routing group every pool is registered in.
pub const ROUTING_GROUP: &str = "global";

/// A pool has no KV blocks the core could count, so one token per block
/// keeps the block arithmetic trivial. The core requires some block size.
const BLOCK_SIZE: u32 = 1;

/// The core's id for `pool`.
pub fn worker_id(pool: &PoolId) -> WorkerId {
    xxh3_64(pool.as_str().as_bytes())
}

/// `pool` as one of the core's workers. The pool id doubles as the
/// endpoint, which the core requires and the frontend never dials.
pub fn worker(pool: &Pool) -> WorkerRequest {
    WorkerRequest {
        worker_id: worker_id(&pool.id),
        model_name: pool.model.clone(),
        routing_group: ROUTING_GROUP.to_owned(),
        endpoint: Some(pool.id.to_string()),
        block_size: Some(BLOCK_SIZE),
        data_parallel_start_rank: Some(0),
        data_parallel_size: Some(1),
        ..WorkerRequest::default()
    }
}

/// The pools with a frontend, as the core's desired workers.
pub fn workers(pools: &[Pool]) -> Vec<WorkerRequest> {
    pools
        .iter()
        .filter(|pool| reachable(pool))
        .map(worker)
        .collect()
}

/// Publishes each pool snapshot to the core as a membership snapshot, so
/// the core's workers follow the pools: a pool that joins, leaves, or loses
/// its last frontend is added or removed on the next snapshot.
pub struct PoolCatalogSource<S> {
    pools: watch::Receiver<Arc<S>>,
    started: bool,
}

impl<S> PoolCatalogSource<S> {
    /// `pools` is the frontend's catalog subscription.
    pub fn new(pools: watch::Receiver<Arc<S>>) -> Self {
        Self {
            pools,
            started: false,
        }
    }
}

#[async_trait]
impl<S> WorkerCatalogSource for PoolCatalogSource<S>
where
    S: AsRef<[Pool]> + Send + Sync,
{
    /// The current pools first, then each change; `None` once the catalog
    /// is dropped.
    async fn next_snapshot(&mut self) -> Option<Vec<WorkerRequest>> {
        if std::mem::replace(&mut self.started, true) {
            self.pools.changed().await.ok()?;
        }
        let snapshot = self.pools.borrow_and_update().clone();
        Some(workers((*snapshot).as_ref()))
    }
}

#[cfg(test)]
mod tests {
    use crate::selector::tests::pool;

    use super::*;

    fn ids(workers: &[WorkerRequest]) -> Vec<WorkerId> {
        workers.iter().map(|worker| worker.worker_id).collect()
    }

    #[test]
    fn a_pool_is_one_worker_in_the_global_group() {
        let blue = pool(
            "blue",
            "qwen",
            &["http://blue-0:8000", "http://blue-1:8000"],
        );
        let worker = worker(&blue);

        assert_eq!(worker.worker_id, worker_id(&blue.id));
        assert_eq!(worker.model_name, "qwen");
        assert_eq!(worker.routing_group, ROUTING_GROUP);
        assert_eq!(worker.endpoint.as_deref(), Some(blue.id.as_str()));
        assert_eq!(
            (worker.data_parallel_start_rank, worker.data_parallel_size),
            (Some(0), Some(1))
        );
        assert!(worker.block_size.is_some_and(|size| size > 0));
        assert_ne!(
            worker_id(&blue.id),
            worker_id(&pool("green", "qwen", &[]).id)
        );
    }

    #[test]
    fn pools_without_frontends_are_not_workers() {
        let draining = pool("draining", "qwen", &[]);
        let live = pool("live", "qwen", &["http://live:8000"]);

        assert_eq!(
            ids(&workers(&[draining, live.clone()])),
            [worker_id(&live.id)]
        );
        assert!(workers(&[]).is_empty());
    }

    #[tokio::test]
    async fn snapshots_add_and_remove_workers() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let (catalog, pools) = watch::channel(Arc::new(vec![blue.clone()]));
        let mut source = PoolCatalogSource::new(pools);

        // The first snapshot comes without waiting for a change.
        let first = source.next_snapshot().await.unwrap();
        assert_eq!(ids(&first), [worker_id(&blue.id)]);

        catalog.send_replace(Arc::new(vec![blue.clone(), green.clone()]));
        let mut second = ids(&source.next_snapshot().await.unwrap());
        second.sort_unstable();
        let mut both = [worker_id(&blue.id), worker_id(&green.id)];
        both.sort_unstable();
        assert_eq!(second, both);

        // Losing its last frontend removes a pool.
        let mut blue_down = blue.clone();
        blue_down.frontends.clear();
        catalog.send_replace(Arc::new(vec![blue_down, green.clone()]));
        let third = source.next_snapshot().await.unwrap();
        assert_eq!(ids(&third), [worker_id(&green.id)]);

        drop(catalog);
        assert!(source.next_snapshot().await.is_none());
    }
}
