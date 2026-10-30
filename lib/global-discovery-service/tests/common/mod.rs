// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helpers shared by the integration tests, and the `conformance` cases: the
//! behaviour every registration store must show, written once over
//! `(Arc<dyn PoolRegistrar>, Arc<PoolDirectory>, ClockHandle)` and run against
//! the memory store and the DynamoDB store.
//!
//! Nothing here names the DynamoDB store or the AWS SDK, so the memory
//! conformance suite builds without them. The DynamoDB-only helpers are in
//! `common/dynamodb.rs`.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use dynamo_global_common::{PoolKey, PoolLocation};
use dynamo_global_discovery_service::pool_discovery::{
    Capacity, Lease, PoolDirectory, PoolEvent, PoolListing, PoolRegistration, WatchEvent,
    WatchStream,
};
use url::Url;

/// The lease every store under test is configured with: heartbeat every 5 s,
/// 15 s to live.
pub fn lease() -> Lease {
    Lease::new(5, 15).unwrap()
}

pub fn key(cluster: &str, namespace: &str, dgd: &str) -> PoolKey {
    PoolKey::new(cluster, namespace, dgd).unwrap()
}

/// A registration with the capabilities and capacity the memory store's own
/// tests use, deliberately not in sorted order.
pub fn registration(
    cluster: &str,
    namespace: &str,
    dgd: &str,
    model: &str,
    frontends: &[&str],
) -> PoolRegistration {
    PoolRegistration {
        key: key(cluster, namespace, dgd),
        location: PoolLocation {
            region: "us-west-2".to_owned(),
            availability_zone: None,
            datacenter: None,
        },
        model: model.to_owned(),
        capabilities: vec!["prefill".to_owned(), "decode".to_owned()],
        frontends: frontends
            .iter()
            .map(|frontend| Url::parse(frontend).unwrap())
            .collect(),
        capacity: vec![
            Capacity {
                sku: "H100".to_owned(),
                workers: 8,
            },
            Capacity {
                sku: "B200".to_owned(),
                workers: 4,
            },
        ],
    }
}

/// The registration most cases use: pool `dgd` of cluster `west`.
pub fn mocker(dgd: &str, model: &str) -> PoolRegistration {
    registration(
        "west",
        "dynamo",
        dgd,
        model,
        &["http://10.43.1.3:8000/", "http://10.43.1.2:8000/"],
    )
}

/// How a case moves time. Every case advances time only through this handle,
/// so the one place that knows a clock's API is here.
#[derive(Clone)]
pub enum ClockHandle {
    /// The store reads time from Tokio, which the test paused
    /// (`#[tokio::test(start_paused = true)]`): the memory store.
    Tokio,
    /// The store reads time from a manual clock that this closure advances
    /// (see `ClockHandle::manual` in `common/dynamodb.rs`): the DynamoDB store.
    /// A closure, not the `ManualClock` itself, so that this module builds
    /// before that type exists.
    Manual(Arc<dyn Fn(Duration) + Send + Sync>),
}

impl ClockHandle {
    pub async fn advance(&self, by: Duration) {
        match self {
            Self::Tokio => tokio::time::advance(by).await,
            Self::Manual(advance) => advance(by),
        }
    }
}

/// The pool `dgd` of cluster `west`, if it is listed.
pub async fn find(pools: &PoolDirectory, dgd: &str) -> Option<PoolListing> {
    find_key(pools, &key("west", "dynamo", dgd)).await
}

pub async fn find_key(pools: &PoolDirectory, wanted: &PoolKey) -> Option<PoolListing> {
    let snapshot = pools.list().await.unwrap();
    snapshot
        .items
        .into_iter()
        .find(|listing| &listing.key == wanted)
}

/// The next event that is not a `Resync`. Stores differ in when they publish
/// a `Resync` (the memory store once a lease has passed, the DynamoDB store
/// never after the first), so a case that only cares about pool changes
/// skips them.
pub async fn next_change(watch: &mut WatchStream<PoolListing, PoolKey>) -> PoolEvent {
    use futures::StreamExt;
    loop {
        let next = tokio::time::timeout(Duration::from_secs(30), watch.next())
            .await
            .expect("timed out waiting for a watch event");
        match next {
            Some(Ok(WatchEvent::Resync(_))) => continue,
            Some(Ok(event)) => return event,
            Some(Err(error)) => panic!("watch failed: {error:?}"),
            None => panic!("watch ended"),
        }
    }
}

/// The behavioural cases. Each takes the store as its write side and its read
/// side, and a handle to the clock it reads. The stores are configured with
/// [`lease`], and for DynamoDB a 2 s `lease_skew`.
pub mod conformance {
    use super::*;
    use dynamo_global_discovery_service::pool_discovery::{Completeness, PoolRegistrar};
    use futures::StreamExt;

    /// Past the lease plus the DynamoDB store's 2 s skew, so a pool that has
    /// missed its lease is gone in every store.
    const PAST_EXPIRY: Duration = Duration::from_secs(20);

    pub async fn register_answers_the_configured_lease(
        registrar: Arc<dyn PoolRegistrar>,
        _pools: Arc<PoolDirectory>,
        _clock: ClockHandle,
    ) {
        assert_eq!(
            registrar.register(mocker("mocker", "m1")).await.unwrap(),
            lease()
        );
    }

    /// A pool lists with its lists sorted and deduplicated, and no relay.
    pub async fn a_registered_pool_is_listed_normalised(
        registrar: Arc<dyn PoolRegistrar>,
        pools: Arc<PoolDirectory>,
        _clock: ClockHandle,
    ) {
        let mut sent = mocker("mocker", "m1");
        sent.capabilities.push("decode".to_owned());
        sent.frontends
            .push(Url::parse("http://10.43.1.2:8000/").unwrap());
        let _ = registrar.register(sent).await.unwrap();

        let listing = find(&*pools, "mocker").await.expect("listed");
        assert_eq!(listing.model.as_deref(), Some("m1"));
        assert_eq!(listing.capabilities, ["decode", "prefill"]);
        assert_eq!(
            listing.frontends,
            [
                Url::parse("http://10.43.1.2:8000/").unwrap(),
                Url::parse("http://10.43.1.3:8000/").unwrap()
            ]
        );
        assert_eq!(
            listing.capacity,
            [
                Capacity {
                    sku: "B200".to_owned(),
                    workers: 4
                },
                Capacity {
                    sku: "H100".to_owned(),
                    workers: 8
                }
            ]
        );
        assert!(listing.relay_identities.is_empty());
    }

    /// Whoever registers last decides the pool, and it is listed once.
    pub async fn the_last_registration_wins(
        registrar: Arc<dyn PoolRegistrar>,
        pools: Arc<PoolDirectory>,
        _clock: ClockHandle,
    ) {
        let _ = registrar.register(mocker("mocker", "m1")).await.unwrap();
        let _ = registrar.register(mocker("mocker", "m2")).await.unwrap();
        let snapshot = pools.list().await.unwrap();
        assert_eq!(snapshot.items.len(), 1);
        assert_eq!(snapshot.items[0].model.as_deref(), Some("m2"));
    }

    /// Pools are told apart by their whole key, including keys whose parts
    /// would collide if a store joined them with a separator carelessly.
    pub async fn distinct_keys_are_distinct_pools(
        registrar: Arc<dyn PoolRegistrar>,
        pools: Arc<PoolDirectory>,
        _clock: ClockHandle,
    ) {
        let keys = [
            ("west", "dynamo", "a"),
            ("west", "dynamo", "b"),
            ("east", "dynamo", "a"),
            ("a#b", "c", "d"),
            ("a", "b#c", "d"),
            ("a##b", "c", "d"),
            ("a", "b", "c#d"),
            ("a", "b#c", "d#"),
            ("a\\", "b", "c"),
            ("a", "\\b", "c"),
            ("a\\#", "b", "c"),
        ];
        for (index, (cluster, namespace, dgd)) in keys.iter().enumerate() {
            let model = format!("model-{index}");
            let _ = registrar
                .register(registration(cluster, namespace, dgd, &model, &[]))
                .await
                .unwrap();
        }
        let snapshot = pools.list().await.unwrap();
        assert_eq!(snapshot.items.len(), keys.len(), "{:?}", snapshot.items);
        for (index, (cluster, namespace, dgd)) in keys.iter().enumerate() {
            let listing = find_key(&*pools, &key(cluster, namespace, dgd))
                .await
                .unwrap_or_else(|| panic!("{cluster}/{namespace}/{dgd} is not listed"));
            assert_eq!(listing.model, Some(format!("model-{index}")));
        }
    }

    /// `registering_again_only_renews`: heartbeats keep a pool alive past its
    /// lease and publish nothing, and only a change is published.
    pub async fn registering_again_only_renews(
        registrar: Arc<dyn PoolRegistrar>,
        pools: Arc<PoolDirectory>,
        clock: ClockHandle,
    ) {
        assert_eq!(
            registrar.register(mocker("mocker", "m1")).await.unwrap(),
            lease()
        );
        let mut watch = pools.list_and_watch(None).await.unwrap();
        let Some(Ok(WatchEvent::Upserted(first))) = watch.next().await else {
            panic!("expected the pool");
        };
        assert_eq!(first.capabilities, ["decode", "prefill"]);
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));

        // Heartbeats every 10 s keep the 15 s lease alive past 15 s. Lists in
        // another order, or with an item twice, are no change.
        for _ in 0..3 {
            clock.advance(Duration::from_secs(10)).await;
            let mut heartbeat = mocker("mocker", "m1");
            heartbeat.capabilities.reverse();
            heartbeat.frontends.reverse();
            heartbeat.capacity.push(Capacity {
                sku: "H100".to_owned(),
                workers: 8,
            });
            let _ = registrar.register(heartbeat).await.unwrap();
        }
        assert_eq!(find(&*pools, "mocker").await, Some(first));

        // A changed pool is published, and it is the next change: the
        // renewals published nothing.
        let _ = registrar.register(mocker("mocker", "m2")).await.unwrap();
        let PoolEvent::Upserted(changed) = next_change(&mut watch).await else {
            panic!("expected the change");
        };
        assert_eq!(changed.model.as_deref(), Some("m2"));
    }

    /// `a_missed_lease_hides_then_removes_the_pool`.
    pub async fn a_missed_lease_hides_then_removes_the_pool(
        registrar: Arc<dyn PoolRegistrar>,
        pools: Arc<PoolDirectory>,
        clock: ClockHandle,
    ) {
        let _ = registrar.register(mocker("mocker", "m1")).await.unwrap();
        let mut watch = pools.list_and_watch(None).await.unwrap();
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Upserted(_)))
        ));
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));

        clock.advance(PAST_EXPIRY).await;
        // Reads skip it before any sweep or poll runs.
        assert!(find(&*pools, "mocker").await.is_none());
        // Then watchers are told.
        let PoolEvent::Removed(removed) = next_change(&mut watch).await else {
            panic!("expected the pool to be removed");
        };
        assert_eq!(removed, key("west", "dynamo", "mocker"));

        // Registering again brings it back.
        let _ = registrar.register(mocker("mocker", "m1")).await.unwrap();
        let PoolEvent::Upserted(listing) = next_change(&mut watch).await else {
            panic!("expected the pool to come back");
        };
        assert_eq!(listing.key, key("west", "dynamo", "mocker"));
        assert!(find(&*pools, "mocker").await.is_some());
    }

    /// One pool missing its lease does not hide the others.
    pub async fn only_the_pool_that_missed_its_lease_expires(
        registrar: Arc<dyn PoolRegistrar>,
        pools: Arc<PoolDirectory>,
        clock: ClockHandle,
    ) {
        let _ = registrar.register(mocker("old", "m1")).await.unwrap();
        let _ = registrar.register(mocker("kept", "m1")).await.unwrap();
        for _ in 0..4 {
            clock.advance(Duration::from_secs(5)).await;
            let _ = registrar.register(mocker("kept", "m1")).await.unwrap();
        }
        assert!(find(&*pools, "old").await.is_none());
        assert!(find(&*pools, "kept").await.is_some());
        assert_eq!(pools.list().await.unwrap().items.len(), 1);
    }

    /// `lists_are_exact_once_a_lease_has_passed`. `at_first` is what a store
    /// that has just started lists as: `Estimated` for the memory store, which
    /// only knows the registrations it received, and `Exact` for a shared one.
    pub async fn lists_are_exact_once_a_lease_has_passed(
        _registrar: Arc<dyn PoolRegistrar>,
        pools: Arc<PoolDirectory>,
        clock: ClockHandle,
        at_first: Completeness,
    ) {
        let mut watch = pools.list_and_watch(None).await.unwrap();
        let Some(Ok(WatchEvent::Resync(first))) = watch.next().await else {
            panic!("expected the first Resync");
        };
        assert_eq!(first.completeness, at_first);
        assert_eq!(pools.list().await.unwrap().completeness, at_first);

        clock.advance(lease().duration()).await;
        assert_eq!(
            pools.list().await.unwrap().completeness,
            Completeness::Exact
        );
    }
}
