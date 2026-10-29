// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pools as the router core's workers.
//!
//! The core selects among workers of one model in one routing group. Each
//! pool with a frontend becomes one worker with one data-parallel rank, and
//! all pools share the routing group [`ROUTING_GROUP`], so the pools of a
//! model form one partition. The worker id is a hash of the pool id, which
//! both sides can compute, so a chosen worker maps back to its pool without
//! a table. Pickers that weigh pool signals get the way back through a
//! [`PoolDirectory`] the source keeps in step with its snapshots.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_global_common::PoolId;
use dynamo_kv_router::WorkerId;
use dynamo_kv_router::services::selection::{WorkerCatalogSource, WorkerRequest};
use tokio::sync::{oneshot, watch};
use xxhash_rust::xxh3::xxh3_64;

use super::directory::PoolDirectory;
use crate::Pool;
use crate::selector::reachable;

/// The routing group every pool is registered in.
pub const ROUTING_GROUP: &str = "global";

/// A pool has no KV blocks the core could count, so one token per block
/// keeps the block arithmetic trivial. The core requires some block size.
const BLOCK_SIZE: u32 = 1;

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
    directory: Option<Arc<PoolDirectory>>,
    started: bool,
    first_taken: Option<oneshot::Sender<()>>,
}

impl<S> PoolCatalogSource<S> {
    /// `pools` is the frontend's catalog subscription.
    pub fn new(pools: watch::Receiver<Arc<S>>) -> Self {
        Self {
            pools,
            directory: None,
            started: false,
            first_taken: None,
        }
    }

    /// Like [`Self::new`], with a receiver that resolves once the core has
    /// taken the first snapshot and come back for the next one, so a caller
    /// can hold requests until the configured pools are selectable. It
    /// fails if the source is dropped first.
    pub fn with_ready(pools: watch::Receiver<Arc<S>>) -> (Self, oneshot::Receiver<()>) {
        let (ready, is_ready) = oneshot::channel();
        let mut source = Self::new(pools);
        source.first_taken = Some(ready);
        (source, is_ready)
    }

    /// Also keeps `directory` in step with the snapshots, for pickers that
    /// map workers back to pools.
    pub fn with_directory(mut self, directory: Arc<PoolDirectory>) -> Self {
        self.directory = Some(directory);
        self
    }
}

#[async_trait]
impl<S> WorkerCatalogSource for PoolCatalogSource<S>
where
    S: AsRef<[Pool]> + Send + Sync,
{
    /// The current pools first, then each change; `None` once the catalog
    /// is dropped. The directory is updated before the core sees the
    /// snapshot, so a picker never meets a worker the directory lacks.
    async fn next_snapshot(&mut self) -> Option<Vec<WorkerRequest>> {
        if std::mem::replace(&mut self.started, true) {
            if let Some(ready) = self.first_taken.take() {
                let _ = ready.send(());
            }
            self.pools.changed().await.ok()?;
        }
        let snapshot = self.pools.borrow_and_update().clone();
        let pools: &[Pool] = (*snapshot).as_ref();
        if let Some(directory) = &self.directory {
            directory.replace(pools);
        }
        Some(workers(pools))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

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
        let directory = Arc::new(PoolDirectory::new());
        let mut source = PoolCatalogSource::new(pools).with_directory(directory.clone());

        let first = source.next_snapshot().await.unwrap();
        assert_eq!(ids(&first), [worker_id(&blue.id)]);
        assert_eq!(
            directory.pool(worker_id(&blue.id)).map(|entry| entry.id),
            Some(blue.id.clone())
        );

        catalog.send_replace(Arc::new(vec![blue.clone(), green.clone()]));
        let mut second = ids(&source.next_snapshot().await.unwrap());
        second.sort_unstable();
        let mut both = [worker_id(&blue.id), worker_id(&green.id)];
        both.sort_unstable();
        assert_eq!(second, both);
        assert_eq!(directory.snapshot().len(), 2);

        let mut blue_down = blue.clone();
        blue_down.frontends.clear();
        catalog.send_replace(Arc::new(vec![blue_down, green.clone()]));
        let third = source.next_snapshot().await.unwrap();
        assert_eq!(ids(&third), [worker_id(&green.id)]);

        drop(catalog);
        assert!(source.next_snapshot().await.is_none());
    }

    #[tokio::test]
    async fn ready_resolves_once_the_core_comes_back_for_a_second_snapshot() {
        let blue = pool("blue", "qwen", &["http://blue-0:8000"]);
        let (_catalog, pools) = watch::channel(Arc::new(vec![blue]));
        let (mut source, mut ready) = PoolCatalogSource::with_ready(pools);

        source.next_snapshot().await.unwrap();
        assert!(
            ready.try_recv().is_err(),
            "the first snapshot is not yet taken"
        );

        // Coming back for the next snapshot means the first one is applied
        // (or in retry); nothing changed, so the call waits.
        let waits = tokio::time::timeout(Duration::from_millis(20), source.next_snapshot()).await;
        assert!(waits.is_err());
        assert!(ready.try_recv().is_ok());
    }

    #[tokio::test]
    async fn ready_fails_if_the_source_is_dropped_first() {
        let (_catalog, pools) = watch::channel(Arc::new(Vec::<Pool>::new()));
        let (source, ready) = PoolCatalogSource::with_ready(pools);
        drop(source);
        assert!(ready.await.is_err());
    }
}
