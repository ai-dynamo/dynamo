// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pools as the router core's workers: one worker per reachable pool, keyed
//! by a hash of the pool id.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use dynamo_global_common::PoolId;
use dynamo_kv_router::WorkerId;
use dynamo_kv_router::services::selection::{CatalogReconciler, WorkerRequest};
use tokio::sync::watch;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
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

/// How long a pool that left the catalog stays a core worker after the pools
/// without it are published, so a select that read the earlier pools never
/// names a worker the core has dropped.
pub(super) const RETIRE_AFTER: Duration = Duration::from_secs(5);

/// Delay before a failed pass is applied again, as `CatalogReconciler::run`.
const RETRY_AFTER: Duration = Duration::from_secs(5);

/// Where a core worker stands against the published pools.
enum Membership {
    /// In the latest pools.
    Wanted,
    /// Left the latest pools, which are not published yet.
    Left,
    /// Left the published pools at this instant.
    Retiring(Instant),
}

/// Applies each pool snapshot to the core, then publishes it to `applied`.
/// Pools join the core before the snapshot naming them is published, and
/// leave it [`RETIRE_AFTER`] after one without them is, so the core's workers
/// always cover the published pools. A failed pass publishes nothing and is
/// applied again after [`RETRY_AFTER`] or on the next snapshot. Returns once
/// `pools` closes or `cancel` fires.
pub(super) async fn reconcile<S>(
    mut reconciler: CatalogReconciler,
    mut pools: watch::Receiver<Arc<S>>,
    applied: watch::Sender<Arc<S>>,
    cancel: CancellationToken,
) where
    S: AsRef<[Pool]>,
{
    let mut members: HashMap<WorkerId, (WorkerRequest, Membership)> = HashMap::new();
    let mut snapshot = pools.borrow_and_update().clone();
    loop {
        let wanted = workers((*snapshot).as_ref());
        for (_, membership) in members.values_mut() {
            if matches!(membership, Membership::Wanted) {
                *membership = Membership::Left;
            }
        }
        for worker in &wanted {
            members.insert(worker.worker_id, (worker.clone(), Membership::Wanted));
        }
        let now = Instant::now();
        members.retain(|_, (_, membership)| {
            !matches!(membership, Membership::Retiring(since) if now >= *since + RETIRE_AFTER)
        });
        // From the snapshot itself, so the core still rejects two pools whose
        // ids hash to one worker id.
        let desired = wanted
            .into_iter()
            .chain(
                members
                    .values()
                    .filter(|(_, membership)| !matches!(membership, Membership::Wanted))
                    .map(|(worker, _)| worker.clone()),
            )
            .collect::<Vec<_>>();
        let failed = match reconciler.apply(&desired).await {
            Ok(()) => {
                applied.send_replace(snapshot.clone());
                let published = Instant::now();
                for (_, membership) in members.values_mut() {
                    if matches!(membership, Membership::Left) {
                        *membership = Membership::Retiring(published);
                    }
                }
                false
            }
            Err(error) => {
                tracing::warn!(%error, retry_in = ?RETRY_AFTER, "the router core did not take the pools; keeping the last applied pools");
                true
            }
        };
        let wake = if failed {
            Some(now + RETRY_AFTER)
        } else {
            members
                .values()
                .filter_map(|(_, membership)| match membership {
                    Membership::Retiring(since) => Some(*since + RETIRE_AFTER),
                    _ => None,
                })
                .min()
        };
        tokio::select! {
            _ = cancel.cancelled() => return,
            changed = pools.changed() => {
                if changed.is_err() {
                    return;
                }
                snapshot = pools.borrow_and_update().clone();
            }
            _ = tokio::time::sleep_until(wake.unwrap_or(now)), if wake.is_some() => {}
        }
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
}
