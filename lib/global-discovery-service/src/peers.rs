// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sharing registrations between Global Discovery Service instances.
//!
//! A PoolRelay registers with the instance in its own cluster, but a Global
//! Router may ask the instance in any cluster. So every instance keeps asking
//! the others for the registrations their own PoolRelays made
//! (`GET /registrations`), and keeps a copy of each for the time that
//! registration has left. An instance shares only what its own PoolRelays
//! sent, so a copy is never passed on, and a pool whose PoolRelay stops is
//! gone everywhere about one lease after its last heartbeat.
//!
//! Each instance is asked twice per heartbeat interval, with a timeout of
//! half of one, so each copy is refreshed at least once per heartbeat
//! interval. Copies therefore stay alive while the lease is more than two
//! heartbeat intervals.
//!
//! The instances find each other the way PoolRelays find the routers: from
//! the service's own Service, exported through MCS, so its imported
//! EndpointSlices list every Ready instance in every cluster.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::future::join_all;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use crate::backends::memory::MemoryPoolStore;
use crate::pool_discovery::{ReplicaDirectory, ReplicaEndpoint, ReplicaId};
use crate::service::SharedRegistrations;

pub struct PeerSync {
    /// Every Ready instance. It may list this one, which is skipped.
    pub peers: Arc<ReplicaDirectory>,
    /// Keeps the copies.
    pub store: Arc<MemoryPoolStore>,
    /// This instance's name in its `GET /registrations` replies.
    pub instance: String,
    /// How often to ask each instance, and how long to wait for its reply:
    /// half the heartbeat interval.
    pub every: Duration,
    /// Set once every instance has been asked. Until then this instance may
    /// know only its own PoolRelays, so it is not ready.
    pub synced: Arc<AtomicBool>,
}

impl PeerSync {
    /// Runs until `stop` is cancelled.
    pub async fn run(self, stop: CancellationToken) {
        let client = match reqwest::Client::builder()
            .timeout(self.every)
            .redirect(reqwest::redirect::Policy::none())
            .build()
        {
            Ok(client) => client,
            Err(error) => {
                tracing::error!(%error, "cannot build the HTTP client; registrations stay local");
                return;
            }
        };
        // Logged when they become unreachable and when they recover, not on
        // every round.
        let mut unreachable: HashSet<ReplicaId> = HashSet::new();
        let mut ticks = tokio::time::interval(self.every);
        ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                () = stop.cancelled() => return,
                _ = ticks.tick() => {}
            }
            let peers = match self.peers.list().await {
                Ok(snapshot) => snapshot.items,
                Err(error) => {
                    tracing::warn!(?error, "cannot list the discovery service instances");
                    continue;
                }
            };
            let replies = join_all(peers.iter().map(|peer| fetch(&client, peer))).await;
            for (peer, reply) in peers.iter().zip(replies) {
                match reply {
                    Ok((reply, received)) => {
                        if unreachable.remove(&peer.replica) {
                            tracing::info!(
                                peer = %peer.replica.0,
                                "discovery instance is reachable again"
                            );
                        }
                        self.absorb(peer, reply, received);
                    }
                    Err(error) => {
                        if unreachable.insert(peer.replica.clone()) {
                            tracing::warn!(
                                peer = %peer.replica.0,
                                error = format!("{error:#}"),
                                "cannot read another discovery instance's registrations"
                            );
                        }
                    }
                }
            }
            // Forget instances that left, so one that comes back is logged.
            unreachable.retain(|id| peers.iter().any(|peer| &peer.replica == id));
            self.synced.store(true, Ordering::Release);
        }
    }

    /// Copies each registration in `reply` for the time it had left when the
    /// reply was `received`, so waiting for a slower instance does not
    /// lengthen it. Skips this instance's own reply: copying its own
    /// registrations back would make them copies, which are not shared.
    fn absorb(&self, peer: &ReplicaEndpoint, reply: SharedRegistrations, received: Instant) {
        if reply.instance == self.instance {
            return;
        }
        let waited = received.elapsed();
        for shared in reply.registrations {
            match shared.into_record() {
                Ok((record, remaining)) => {
                    self.store.adopt(record, remaining.saturating_sub(waited))
                }
                Err(error) => tracing::warn!(
                    peer = %peer.replica.0,
                    %error,
                    "skipping an invalid shared registration"
                ),
            }
        }
    }
}

/// Also returns when the reply arrived.
async fn fetch(
    client: &reqwest::Client,
    peer: &ReplicaEndpoint,
) -> anyhow::Result<(SharedRegistrations, Instant)> {
    let url = peer.url.join("registrations")?;
    let reply = client
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok((reply, Instant::now()))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use tokio::net::TcpListener;
    use url::Url;

    use super::*;
    use crate::backends::kubernetes::TrustedNetworkAuthenticator;
    use crate::backends::memory::StaticReplicaDirectory;
    use crate::pool_discovery::{
        Directory, LeasePolicy, PeerCredentials, PoolDirectory, PoolRegistrar, verify,
    };
    use crate::service::{self, DiscoveryStore, Sharing};

    const LEASE: LeasePolicy = LeasePolicy {
        heartbeat_interval: Duration::from_millis(100),
        lease_duration: Duration::from_millis(1000),
    };

    /// Serves one instance that shares its registrations. Returns its store
    /// and URL.
    async fn instance(name: &str, stop: &CancellationToken) -> (Arc<MemoryPoolStore>, Url) {
        let store = Arc::new(MemoryPoolStore::new(LEASE).unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let pools: Arc<PoolDirectory> = store.clone();
        let registrar: Arc<dyn PoolRegistrar> = store.clone();
        tokio::spawn(service::serve(
            listener,
            DiscoveryStore {
                pools,
                routers: Arc::new(StaticReplicaDirectory::new(Vec::new())),
                registrar: Some(registrar),
                authenticator: Arc::new(TrustedNetworkAuthenticator),
                sharing: Some(Sharing {
                    instance: name.to_owned(),
                    registrations: store.clone(),
                    synced: Arc::new(AtomicBool::new(true)),
                }),
            },
            stop.clone(),
        ));
        (store, url)
    }

    fn peer(name: &str, url: &Url) -> ReplicaEndpoint {
        ReplicaEndpoint {
            replica: ReplicaId(name.to_owned()),
            url: url.clone(),
        }
    }

    /// The body `dynamo-global-discovery-client` sends.
    fn registration(model: &str) -> Value {
        json!({
            "key": {"site_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"},
            "location": {"region": "us-west-2", "availability_zone": null, "cluster": "aws-west", "datacenter": null},
            "model": model,
            "capabilities": [],
            "frontends": [],
            "capacity": [{"sku": "H100", "count": 8}]
        })
    }

    async fn register(url: &Url, body: &Value) {
        let response = reqwest::Client::new()
            .post(url.join("pools").unwrap())
            .json(body)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());
    }

    /// Waits up to five seconds for `store` to list exactly `expected` models.
    async fn wait_for_models(store: &MemoryPoolStore, expected: &[&str]) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let mut models: Vec<String> = store
                .list()
                .await
                .unwrap()
                .items
                .into_iter()
                .filter_map(|record| record.model)
                .collect();
            models.sort();
            if models == expected {
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "expected {expected:?}, got {models:?}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    async fn registrations_reach_the_other_instances_and_lapse_there() {
        let stop = CancellationToken::new();
        let (a, a_url) = instance("a", &stop).await;
        let (b, b_url) = instance("b", &stop).await;
        // B asks every instance, itself included.
        tokio::spawn(
            PeerSync {
                peers: Arc::new(StaticReplicaDirectory::new(vec![
                    peer("a", &a_url),
                    peer("b", &b_url),
                ])),
                store: b.clone(),
                instance: "b".to_owned(),
                every: LEASE.heartbeat_interval / 2,
                synced: Arc::default(),
            }
            .run(stop.clone()),
        );

        register(&a_url, &registration("m1")).await;
        wait_for_models(&b, &["m1"]).await;
        // B keeps a copy, which it does not pass on.
        assert_eq!(a.local_registrations().len(), 1);
        assert!(b.local_registrations().is_empty());

        // A changed registration replaces the copy.
        register(&a_url, &registration("m2")).await;
        wait_for_models(&b, &["m2"]).await;

        // With no more heartbeats, the copy lapses with the original.
        wait_for_models(&a, &[]).await;
        wait_for_models(&b, &[]).await;
        stop.cancel();
    }

    #[tokio::test]
    async fn an_unreachable_instance_is_skipped() {
        let stop = CancellationToken::new();
        let (a, a_url) = instance("a", &stop).await;
        let (b, _) = instance("b", &stop).await;
        let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gone = Url::parse(&format!("http://{}/", closed.local_addr().unwrap())).unwrap();
        drop(closed);
        let synced = Arc::new(AtomicBool::new(false));
        tokio::spawn(
            PeerSync {
                peers: Arc::new(StaticReplicaDirectory::new(vec![
                    peer("gone", &gone),
                    peer("a", &a_url),
                ])),
                store: b.clone(),
                instance: "b".to_owned(),
                every: LEASE.heartbeat_interval / 2,
                synced: synced.clone(),
            }
            .run(stop.clone()),
        );
        register(&a_url, &registration("m1")).await;
        wait_for_models(&b, &["m1"]).await;
        assert_eq!(a.local_registrations().len(), 1);
        // A round that could not reach every instance still counts.
        assert!(synced.load(Ordering::Acquire));
        stop.cancel();
    }

    /// A `GET /registrations` reply from `instance`.
    fn reply(instance: &str, model: &str, expires_in_ms: u64) -> SharedRegistrations {
        serde_json::from_value(json!({
            "instance": instance,
            "registrations": [{"registration": registration(model), "expires_in_ms": expires_in_ms}],
        }))
        .unwrap()
    }

    /// Instance "a" copying into `store`.
    fn sync_into(store: &Arc<MemoryPoolStore>) -> PeerSync {
        PeerSync {
            peers: Arc::new(StaticReplicaDirectory::new(Vec::new())),
            store: store.clone(),
            instance: "a".to_owned(),
            every: LEASE.heartbeat_interval / 2,
            synced: Arc::default(),
        }
    }

    fn somewhere(name: &str) -> ReplicaEndpoint {
        peer(name, &Url::parse("http://127.0.0.1:1/").unwrap())
    }

    #[tokio::test(start_paused = true)]
    async fn an_instance_skips_its_own_reply() {
        let store = Arc::new(MemoryPoolStore::new(LEASE).unwrap());
        let sync = sync_into(&store);
        let (record, _) = reply("a", "m1", 0)
            .registrations
            .remove(0)
            .into_record()
            .unwrap();
        let caller = verify(
            &TrustedNetworkAuthenticator,
            &PeerCredentials::PeerAddress("10.0.0.1".parse().unwrap()),
        )
        .await
        .unwrap();
        store.register(record, &caller).await.unwrap();
        tokio::time::advance(Duration::from_millis(10)).await;

        // Its own reply, though it seems to last longer, changes nothing.
        sync.absorb(&somewhere("a"), reply("a", "m1", 60_000), Instant::now());
        assert_eq!(store.local_registrations().len(), 1);
        // The same reply from another instance makes it a copy here.
        sync.absorb(&somewhere("b"), reply("b", "m1", 60_000), Instant::now());
        assert!(store.local_registrations().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn waiting_for_a_slower_instance_does_not_lengthen_copies() {
        let store = Arc::new(MemoryPoolStore::new(LEASE).unwrap());
        let sync = sync_into(&store);
        let received = Instant::now();
        // The reply waited 300 ms for a slower instance's.
        tokio::time::advance(Duration::from_millis(300)).await;
        sync.absorb(&somewhere("b"), reply("b", "m1", 500), received);
        assert_eq!(store.list().await.unwrap().items.len(), 1);
        tokio::time::advance(Duration::from_millis(250)).await;
        assert!(store.list().await.unwrap().items.is_empty());
    }
}
