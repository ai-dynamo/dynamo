// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pools as the router core's workers: one worker per reachable pool, keyed
//! by a hash of the pool id.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_global_common::PoolId;
use dynamo_kv_router::WorkerId;
use dynamo_kv_router::services::selection::{WorkerCatalogSource, WorkerRequest};
use tokio::sync::{OwnedRwLockWriteGuard, RwLock, watch};
use xxhash_rust::xxh3::xxh3_64;

use crate::Pool;
use crate::selector::reachable;

pub(super) const ROUTING_GROUP: &str = "global";

/// Required by the core; pools have no KV blocks.
const BLOCK_SIZE: u32 = 1;

pub(super) fn worker_id(pool: &PoolId) -> WorkerId {
    xxh3_64(pool.as_str().as_bytes())
}

/// The endpoint is required by the core but never dialed.
fn worker(pool: &Pool) -> WorkerRequest {
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

fn workers(pools: &[Pool]) -> Vec<WorkerRequest> {
    pools
        .iter()
        .filter(|pool| reachable(pool))
        .map(worker)
        .collect()
}

/// Feeds each pool snapshot to the core's `CatalogReconciler` and publishes
/// it once applied. The reconciler asks for the next snapshot only after it
/// has applied the last one, so the source holds `catalog` for writing from
/// handing a snapshot out until that request: selects that hold it for
/// reading see the published pools and the core's workers agree.
pub(super) struct PoolCatalogSource<S> {
    pools: watch::Receiver<Arc<S>>,
    started: bool,
    catalog: Arc<RwLock<()>>,
    applying: Option<(Arc<S>, OwnedRwLockWriteGuard<()>)>,
    applied: watch::Sender<Arc<S>>,
}

impl<S> PoolCatalogSource<S> {
    /// The receiver holds the pools the core has applied. It changes once the
    /// first snapshot is applied; until then it holds the unapplied pools.
    pub(super) fn new(
        pools: watch::Receiver<Arc<S>>,
        catalog: Arc<RwLock<()>>,
    ) -> (Self, watch::Receiver<Arc<S>>) {
        let (applied, applied_pools) = watch::channel(pools.borrow().clone());
        let source = Self {
            pools,
            started: false,
            catalog,
            applying: None,
            applied,
        };
        (source, applied_pools)
    }
}

#[async_trait]
impl<S> WorkerCatalogSource for PoolCatalogSource<S>
where
    S: AsRef<[Pool]> + Send + Sync,
{
    async fn next_snapshot(&mut self) -> Option<Vec<WorkerRequest>> {
        if let Some((pools, _applying)) = self.applying.take() {
            self.applied.send_replace(pools);
        }
        if std::mem::replace(&mut self.started, true) {
            self.pools.changed().await.ok()?;
        }
        let applying = self.catalog.clone().write_owned().await;
        let snapshot = self.pools.borrow_and_update().clone();
        let workers = workers((*snapshot).as_ref());
        self.applying = Some((snapshot, applying));
        Some(workers)
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
        let (mut source, _applied) = PoolCatalogSource::new(pools, Arc::default());

        let first = source.next_snapshot().await.unwrap();
        assert_eq!(ids(&first), [worker_id(&blue.id)]);

        catalog.send_replace(Arc::new(vec![blue.clone(), green.clone()]));
        let mut second = ids(&source.next_snapshot().await.unwrap());
        second.sort_unstable();
        let mut both = [worker_id(&blue.id), worker_id(&green.id)];
        both.sort_unstable();
        assert_eq!(second, both);

        let mut blue_down = blue.clone();
        blue_down.frontends.clear();
        catalog.send_replace(Arc::new(vec![blue_down, green.clone()]));
        let third = source.next_snapshot().await.unwrap();
        assert_eq!(ids(&third), [worker_id(&green.id)]);

        drop(catalog);
        assert!(source.next_snapshot().await.is_none());
    }

    /// A snapshot is published only when the reconciler asks for the next
    /// one, after it has applied this one.
    #[tokio::test]
    async fn snapshots_are_published_once_applied() {
        let blue = Arc::new(vec![pool("blue", "qwen", &["http://blue:8000"])]);
        let green = Arc::new(vec![pool("green", "qwen", &["http://green:8000"])]);
        let (catalog, pools) = watch::channel(blue.clone());
        let catalog_lock = Arc::new(RwLock::new(()));
        let (mut source, mut applied) = PoolCatalogSource::new(pools, catalog_lock.clone());

        source.next_snapshot().await.unwrap();
        assert!(!applied.has_changed().unwrap());
        assert!(
            catalog_lock.try_read().is_err(),
            "held while the core applies"
        );

        catalog.send_replace(green.clone());
        assert!(Arc::ptr_eq(&applied.borrow(), &blue));
        source.next_snapshot().await.unwrap();
        assert!(Arc::ptr_eq(&applied.borrow_and_update(), &blue));
        assert!(catalog_lock.try_read().is_err());

        drop(catalog);
        assert!(source.next_snapshot().await.is_none());
        assert!(Arc::ptr_eq(&applied.borrow_and_update(), &green));
        assert!(catalog_lock.try_read().is_ok(), "released once applied");
    }
}
