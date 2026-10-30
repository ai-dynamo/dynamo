// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use dynamo_global_common::{PoolId, PoolKey};
use dynamo_global_frontend::{
    PoolSnapshot, RelayGatedPoolCatalog, RelaySourceKey, RelayStreamGuard,
};
use dynamo_global_router::{Pool, PoolSelector, RoundRobinSelector, RoutingRequest};
use dynamo_global_view_broadcaster::{Frame, PoolViewBroadcaster};
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

struct StaticCatalog(watch::Sender<Arc<PoolSnapshot>>);
impl StaticCatalog {
    fn new(pools: Vec<Pool>) -> Self {
        Self(watch::channel(Arc::new(PoolSnapshot::new(pools))).0)
    }
    fn subscribe(&self) -> watch::Receiver<Arc<PoolSnapshot>> {
        self.0.subscribe()
    }
}
fn source(pool: &PoolId, replica: &str) -> RelaySourceKey {
    RelaySourceKey {
        pool: pool.clone(),
        service: "pool-relay".into(),
        replica: replica.into(),
    }
}

fn pool() -> Pool {
    let key = PoolKey::new("east", "dynamo", "blue").unwrap();
    Pool::new(
        &key,
        "qwen",
        "east",
        vec!["http://blue:8000/".parse().unwrap()],
    )
}

async fn wait_for_pool(snapshot: &mut watch::Receiver<Arc<PoolSnapshot>>, expected_count: usize) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if snapshot.borrow().pools().len() == expected_count {
                break;
            }
            snapshot.changed().await.unwrap();
        }
    })
    .await
    .unwrap();
}

async fn receive(mut frames: mpsc::Receiver<Frame<PoolId, ()>>, mut guard: RelayStreamGuard) {
    while let Some(frame) = frames.recv().await {
        match frame {
            Frame::Snapshot { revision, pools } => {
                guard.replace_snapshot(revision, pools.into_keys()).unwrap();
            }
            Frame::Upsert { revision, pool, .. } => guard.upsert(revision, pool).unwrap(),
            Frame::Remove { revision, pool } => guard.remove(revision, &pool).unwrap(),
        }
    }
}

#[tokio::test]
async fn broadcaster_reaches_two_routers_and_cancellation_withdraws_views() {
    let pool = pool();
    let discovery = StaticCatalog::new(vec![pool.clone()]);
    let first = Arc::new(RelayGatedPoolCatalog::new(
        discovery.subscribe(),
        CancellationToken::new(),
    ));
    let second = Arc::new(RelayGatedPoolCatalog::new(
        discovery.subscribe(),
        CancellationToken::new(),
    ));
    let mut first_pools = first.subscribe();
    let mut second_pools = second.subscribe();

    let (first_sender, first_frames) = mpsc::channel(16);
    let (second_sender, second_frames) = mpsc::channel(16);
    let first_task = tokio::spawn(receive(
        first_frames,
        first.open(source(&pool.id, "relay-a")),
    ));
    let second_task = tokio::spawn(receive(
        second_frames,
        second.open(source(&pool.id, "relay-a")),
    ));

    let broadcaster = PoolViewBroadcaster::<PoolId, ()>::new();
    broadcaster.upsert(pool.id.clone(), ());
    let connections = HashMap::from([
        ("router-a".to_owned(), first_sender),
        ("router-b".to_owned(), second_sender),
    ]);
    let (directory, routers) = watch::channel(HashSet::from([
        "router-a".to_owned(),
        "router-b".to_owned(),
    ]));
    let fanout = broadcaster.clone();
    let broadcaster_task = tokio::spawn(async move {
        fanout
            .run(routers, move |router| {
                let sink = connections.get(&router).unwrap().clone();
                async move { Ok::<_, std::io::Error>(sink) }
            })
            .await;
    });

    wait_for_pool(&mut first_pools, 1).await;
    wait_for_pool(&mut second_pools, 1).await;
    broadcaster.remove(&pool.id);
    wait_for_pool(&mut first_pools, 0).await;
    wait_for_pool(&mut second_pools, 0).await;
    broadcaster.upsert(pool.id.clone(), ());
    wait_for_pool(&mut first_pools, 1).await;
    wait_for_pool(&mut second_pools, 1).await;

    first_task.abort();
    assert!(first_task.await.unwrap_err().is_cancelled());
    wait_for_pool(&mut first_pools, 0).await;
    assert_eq!(second_pools.borrow().pools().len(), 1);
    let selector = RoundRobinSelector::new();
    let request = RoutingRequest::new("qwen");
    {
        let current = second_pools.borrow();
        let candidates = current.pools_for("qwen").collect::<Vec<_>>();
        assert!(selector.select(&request, &candidates).is_some());
    }
    second_task.abort();
    assert!(second_task.await.unwrap_err().is_cancelled());
    wait_for_pool(&mut second_pools, 0).await;

    // Discovery still lists the pool, but neither router can route to it.
    assert_eq!(discovery.subscribe().borrow().pools().len(), 1);
    {
        let current = second_pools.borrow();
        let candidates = current.pools_for("qwen").collect::<Vec<_>>();
        assert!(selector.select(&request, &candidates).is_none());
    }
    drop(directory);
    broadcaster_task.await.unwrap();
}

#[tokio::test]
async fn an_old_guard_cannot_withdraw_a_replacement_or_another_relay() {
    let pool = pool();
    let discovery = StaticCatalog::new(vec![pool.clone()]);
    let catalog = RelayGatedPoolCatalog::new(discovery.subscribe(), CancellationToken::new());
    let snapshot = catalog.subscribe();

    let mut old = catalog.open(source(&pool.id, "relay-a"));
    old.replace_snapshot(0, [pool.id.clone()]).unwrap();
    let mut replacement = catalog.open(source(&pool.id, "relay-a"));
    assert!(snapshot.borrow().pools().is_empty());
    replacement.replace_snapshot(0, [pool.id.clone()]).unwrap();
    drop(old);
    assert_eq!(snapshot.borrow().pools().len(), 1);

    let mut other = catalog.open(source(&pool.id, "relay-b"));
    other.replace_snapshot(0, [pool.id.clone()]).unwrap();
    drop(replacement);
    assert_eq!(snapshot.borrow().pools().len(), 1);
    drop(other);
    assert!(snapshot.borrow().pools().is_empty());
}

#[tokio::test]
async fn discovery_changes_still_control_the_candidate_pool_set() {
    let pool = pool();
    let (discovery, listing) = watch::channel(Arc::new(PoolSnapshot::new(vec![pool.clone()])));
    let catalog = RelayGatedPoolCatalog::new(listing, CancellationToken::new());
    let mut snapshot = catalog.subscribe();
    let mut relay = catalog.open(source(&pool.id, "relay-a"));
    relay.replace_snapshot(0, [pool.id.clone()]).unwrap();
    wait_for_pool(&mut snapshot, 1).await;

    discovery.send(Arc::new(PoolSnapshot::default())).unwrap();
    wait_for_pool(&mut snapshot, 0).await;
    discovery
        .send(Arc::new(PoolSnapshot::new(vec![pool])))
        .unwrap();
    wait_for_pool(&mut snapshot, 1).await;
}

#[tokio::test]
async fn missed_revision_withdraws_only_that_relay_until_resnapshot() {
    let pool = pool();
    let discovery = StaticCatalog::new(vec![pool.clone()]);
    let catalog = RelayGatedPoolCatalog::new(discovery.subscribe(), CancellationToken::new());
    let snapshot = catalog.subscribe();
    let mut first = catalog.open(source(&pool.id, "relay-a"));
    let mut second = catalog.open(source(&pool.id, "relay-b"));
    first.replace_snapshot(5, [pool.id.clone()]).unwrap();
    second.replace_snapshot(5, [pool.id.clone()]).unwrap();

    assert!(first.replace_snapshot(5, []).is_err());
    assert_eq!(snapshot.borrow().pools().len(), 1);

    assert!(first.upsert(7, pool.id.clone()).is_err());
    assert_eq!(snapshot.borrow().pools().len(), 1);
    drop(second);
    assert!(snapshot.borrow().pools().is_empty());
    assert!(first.upsert(8, pool.id.clone()).is_err());
    first.replace_snapshot(9, [pool.id]).unwrap();
    assert_eq!(snapshot.borrow().pools().len(), 1);
}

#[tokio::test]
async fn identical_service_and_replica_names_in_different_pools_do_not_replace_each_other() {
    let first = pool();
    let key = PoolKey::new("west", "dynamo", "blue").unwrap();
    let second = Pool::new(
        &key,
        "qwen",
        "west",
        vec!["http://west:8000".parse().unwrap()],
    );
    let discovery = StaticCatalog::new(vec![first.clone(), second.clone()]);
    let catalog = RelayGatedPoolCatalog::new(discovery.subscribe(), CancellationToken::new());
    let snapshot = catalog.subscribe();
    let mut east = catalog.open(source(&first.id, "relay-a"));
    let mut west = catalog.open(source(&second.id, "relay-a"));
    east.replace_snapshot(0, [first.id]).unwrap();
    west.replace_snapshot(0, [second.id]).unwrap();
    assert_eq!(snapshot.borrow().pools().len(), 2);
    drop(east);
    assert_eq!(snapshot.borrow().pools().len(), 1);
}
