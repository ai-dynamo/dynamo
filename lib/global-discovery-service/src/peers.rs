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
//! EndpointSlices list every Ready instance in every cluster. They ask each
//! other over plain HTTP, never through a proxy.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::future::join_all;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::pool_discovery::{
    Completeness, Lease, RegistrationExchange, ReplicaDirectory, RouterReplica, RouterReplicaId,
    Snapshot,
};
use crate::service::{SharedRegistrations, Sharing};

/// Asks the other instances for their registrations, and keeps copies.
pub struct PeerSync {
    /// Every Ready instance. It may list this one, which is skipped.
    peers: Arc<ReplicaDirectory>,
    /// Keeps the copies.
    store: Arc<dyn RegistrationExchange>,
    /// This instance's name in its `GET /registrations` replies.
    instance: String,
    /// How often to ask each instance, and how long to wait for its reply:
    /// half the heartbeat interval.
    every: Duration,
    /// Copies from an instance's last answer lapse within this, so one that
    /// has not answered for as long may be missing registrations here.
    lease_duration: Duration,
    /// Set once every instance has been asked. Until then this instance may
    /// know only its own PoolRelays, so it is not ready.
    synced: Arc<AtomicBool>,
    client: reqwest::Client,
}

impl fmt::Debug for PeerSync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PeerSync")
            .field("instance", &self.instance)
            .field("every", &self.every)
            .finish_non_exhaustive()
    }
}

impl PeerSync {
    /// Shares `store`'s registrations with the instances `peers` lists, and
    /// keeps copies of theirs, under the registrations' `lease`. Returns the
    /// task, to start with [`Self::run`], and what the service serves the
    /// others, under one new instance name.
    pub fn new(
        peers: Arc<ReplicaDirectory>,
        store: Arc<dyn RegistrationExchange>,
        lease: Lease,
    ) -> anyhow::Result<(Self, Sharing)> {
        let every = lease.heartbeat_interval() / 2;
        let client = reqwest::Client::builder()
            .timeout(every)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()?;
        let instance = Uuid::new_v4().to_string();
        let synced = Arc::new(AtomicBool::new(false));
        let sharing = Sharing {
            instance: instance.clone(),
            registrations: Arc::clone(&store),
            synced: Arc::clone(&synced),
        };
        let sync = Self {
            peers,
            store,
            instance,
            every,
            lease_duration: lease.duration(),
            synced,
            client,
        };
        Ok((sync, sharing))
    }

    /// Runs until `stop` is cancelled.
    pub async fn run(self, stop: CancellationToken) {
        // Logged when they become unreachable and when they recover, not on
        // every round.
        let mut unreachable: HashSet<RouterReplicaId> = HashSet::new();
        // Logged when it changes. Zero means registrations stay local.
        let mut others_reached = None;
        // When each instance last answered.
        let mut answered: HashMap<RouterReplicaId, Instant> = HashMap::new();
        let mut ticks = tokio::time::interval(self.every);
        ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                () = stop.cancelled() => return,
                _ = ticks.tick() => {}
            }
            let peers = match self.peers.list().await {
                Ok(peers) => peers,
                Err(error) => {
                    tracing::warn!(?error, "cannot list the discovery service instances");
                    self.store.set_copies_missing(true);
                    continue;
                }
            };
            let replies = join_all(peers.items.iter().map(|peer| fetch(&self.client, peer))).await;
            let mut reached = 0;
            for (peer, reply) in peers.items.iter().zip(replies) {
                match reply {
                    Ok((reply, received)) => {
                        answered.insert(peer.replica.clone(), received);
                        if unreachable.remove(&peer.replica) {
                            tracing::info!(
                                peer = %peer.replica.0,
                                "discovery instance is reachable again"
                            );
                        }
                        if reply.instance != self.instance {
                            reached += 1;
                        }
                        self.absorb(reply, received);
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
            unreachable.retain(|id| peers.items.iter().any(|peer| &peer.replica == id));
            answered.retain(|id, _| peers.items.iter().any(|peer| &peer.replica == id));
            self.store.set_copies_missing(copies_missing(
                &peers,
                &answered,
                self.lease_duration,
                Instant::now(),
            ));
            if others_reached != Some(reached) {
                tracing::info!(reached, "other discovery instances reached");
                others_reached = Some(reached);
            }
            self.synced.store(true, Ordering::Release);
        }
    }

    /// Copies each registration in `reply` for the time it had left when the
    /// reply was `received`, so waiting for a slower instance does not
    /// lengthen it. Skips this instance's own reply: copying its own
    /// registrations back would make them copies, which are not shared.
    fn absorb(&self, reply: SharedRegistrations, received: Instant) {
        if reply.instance == self.instance {
            return;
        }
        let waited = received.elapsed();
        for shared in reply.registrations {
            let (registration, remaining) = shared.into_registration();
            self.store
                .adopt(registration, remaining.saturating_sub(waited));
        }
    }
}

/// Whether registrations made with another instance may be missing here: the
/// list of instances may be incomplete, or one has not answered within
/// `lease_duration`, by when the copies from its last answer have lapsed.
fn copies_missing(
    peers: &Snapshot<RouterReplica>,
    answered: &HashMap<RouterReplicaId, Instant>,
    lease_duration: Duration,
    now: Instant,
) -> bool {
    peers.completeness != Completeness::Exact
        || peers.items.iter().any(|peer| {
            answered
                .get(&peer.replica)
                .is_none_or(|at| now.saturating_duration_since(*at) >= lease_duration)
        })
}

/// Also returns when the reply arrived.
async fn fetch(
    client: &reqwest::Client,
    peer: &RouterReplica,
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
    use dynamo_global_common::{Lease, PoolKey, PoolLocation};
    use serde_json::{Value, json};
    use tokio::net::TcpListener;
    use url::Url;

    use super::*;
    use crate::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
    use crate::pool_discovery::{Directory, PoolDirectory, PoolRegistrar, PoolRegistration};
    use crate::service::{self, DiscoveryStore};

    /// A PoolRelay registers every second, and a registration lasts three.
    fn lease() -> Lease {
        Lease::new(1, 3).unwrap()
    }

    fn url_of(listener: &TcpListener) -> Url {
        Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap()
    }

    /// Serves one instance that shares its registrations and asks the
    /// instances `peers` lists, given its own URL. Returns its store, its URL
    /// and the task that asks the others, not yet started.
    async fn instance(
        stop: &CancellationToken,
        peers: impl FnOnce(&Url) -> Vec<RouterReplica>,
    ) -> (Arc<MemoryPoolStore>, Url, PeerSync) {
        let store = Arc::new(MemoryPoolStore::new(lease()));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = url_of(&listener);
        let (sync, sharing) = PeerSync::new(
            Arc::new(StaticReplicaDirectory::new(peers(&url))),
            store.clone(),
            lease(),
        )
        .unwrap();
        let pools: Arc<PoolDirectory> = store.clone();
        let registrar: Arc<dyn PoolRegistrar> = store.clone();
        tokio::spawn(service::serve(
            listener,
            DiscoveryStore {
                pools,
                routers: Arc::new(StaticReplicaDirectory::new(Vec::new())),
                registrar: Some(registrar),
                sharing: Some(sharing),
            },
            None,
            stop.clone(),
        ));
        (store, url, sync)
    }

    fn peer(name: &str, url: &Url) -> RouterReplica {
        RouterReplica {
            replica: RouterReplicaId(name.to_owned()),
            url: url.clone(),
        }
    }

    /// What a PoolRelay registers. Tests register through the store.
    fn pool(model: &str) -> PoolRegistration {
        PoolRegistration {
            key: PoolKey::new("aws-west", "dynamo", "mocker-west-1").unwrap(),
            location: PoolLocation {
                region: "us-west-2".to_owned(),
                availability_zone: None,
                datacenter: None,
            },
            model: model.to_owned(),
            capabilities: Vec::new(),
            frontends: Vec::new(),
            capacity: Vec::new(),
        }
    }

    /// The body `dynamo-global-discovery-client` sends.
    fn registration(model: &str) -> Value {
        json!({
            "key": {"cluster_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"},
            "location": {"region": "us-west-2", "availability_zone": null, "datacenter": null},
            "model": model,
            "capabilities": [],
            "frontends": [],
            "capacity": [{"sku": "H100", "workers": 8}]
        })
    }

    async fn healthz(url: &Url) -> reqwest::StatusCode {
        reqwest::get(url.join("healthz").unwrap())
            .await
            .unwrap()
            .status()
    }

    /// Waits up to five seconds for `url` to report ready.
    async fn wait_for_ready(url: &Url) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while healthz(url).await != reqwest::StatusCode::OK {
            assert!(tokio::time::Instant::now() < deadline, "{url} is not ready");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    async fn models(store: &MemoryPoolStore) -> Vec<String> {
        let mut models: Vec<String> = store
            .list()
            .await
            .unwrap()
            .items
            .into_iter()
            .filter_map(|listing| listing.model)
            .collect();
        models.sort();
        models
    }

    /// Waits up to five seconds for `store` to list exactly `expected` models.
    async fn wait_for_models(store: &MemoryPoolStore, expected: &[&str]) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let models = models(store).await;
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
        let (a, a_url, _) = instance(&stop, |_| Vec::new()).await;
        let _ = a.register(pool("m1")).await.unwrap();
        // B asks every instance: A, itself, one that is gone, and one that
        // never answers.
        let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gone = url_of(&closed);
        drop(closed);
        let silent = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let silent_url = url_of(&silent);
        let (b, b_url, b_sync) = instance(&stop, |own| {
            vec![
                peer("a", &a_url),
                peer("b", own),
                peer("gone", &gone),
                peer("silent", &silent_url),
            ]
        })
        .await;
        // B is not ready until it has asked the other instances once.
        assert_eq!(
            healthz(&b_url).await,
            reqwest::StatusCode::SERVICE_UNAVAILABLE
        );
        tokio::spawn(b_sync.run(stop.clone()));
        // The round ends though two instances did not answer, and by then B
        // has A's registration.
        wait_for_ready(&b_url).await;
        assert_eq!(models(&b).await, ["m1"]);
        // B keeps a copy, which it does not pass on.
        assert_eq!(a.local_registrations().len(), 1);
        assert!(b.local_registrations().is_empty());

        // A changed registration replaces the copy.
        let _ = a.register(pool("m2")).await.unwrap();
        wait_for_models(&b, &["m2"]).await;

        // With no more heartbeats, the copy lapses with the original.
        wait_for_models(&a, &[]).await;
        wait_for_models(&b, &[]).await;

        // A lease has passed. A's lists are complete, but B's are not: two
        // instances never answered it.
        assert_eq!(a.list().await.unwrap().completeness, Completeness::Exact);
        assert_eq!(
            b.list().await.unwrap().completeness,
            Completeness::Estimated
        );
        drop(silent);
        stop.cancel();
    }

    #[test]
    fn copies_may_be_missing_until_every_instance_answers_within_a_lease() {
        let lease = lease().duration();
        let url = Url::parse("http://10.43.0.9:8080/").unwrap();
        let peers = |completeness| Snapshot {
            items: vec![peer("a", &url), peer("b", &url)],
            completeness,
        };
        let then = Instant::now();
        let both = HashMap::from([
            (RouterReplicaId("a".to_owned()), then),
            (RouterReplicaId("b".to_owned()), then),
        ]);
        assert!(!copies_missing(
            &peers(Completeness::Exact),
            &both,
            lease,
            then + lease / 2
        ));
        // A lease after their last answers, their copies have lapsed.
        assert!(copies_missing(
            &peers(Completeness::Exact),
            &both,
            lease,
            then + lease
        ));
        // One never answered.
        let only_a = HashMap::from([(RouterReplicaId("a".to_owned()), then)]);
        assert!(copies_missing(
            &peers(Completeness::Exact),
            &only_a,
            lease,
            then
        ));
        // The list of instances may be incomplete.
        assert!(copies_missing(
            &peers(Completeness::Estimated),
            &both,
            lease,
            then
        ));
    }

    /// A `GET /registrations` reply from `instance`.
    fn reply(instance: &str, model: &str, expires_in_ms: u64) -> SharedRegistrations {
        serde_json::from_value(json!({
            "instance": instance,
            "registrations": [{"registration": registration(model), "expires_in_ms": expires_in_ms}],
        }))
        .unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn absorb_skips_its_own_reply_and_counts_the_wait() {
        let store = Arc::new(MemoryPoolStore::new(lease()));
        let (sync, _) = PeerSync::new(
            Arc::new(StaticReplicaDirectory::new(Vec::new())),
            store.clone(),
            lease(),
        )
        .unwrap();

        // Its own reply is skipped.
        sync.absorb(reply(&sync.instance, "m1", 500), Instant::now());
        assert!(store.list().await.unwrap().items.is_empty());

        // Another instance's reply waited 300 ms for a slower one's, so its
        // copy has 200 ms left, not 500.
        let received = Instant::now();
        tokio::time::advance(Duration::from_millis(300)).await;
        sync.absorb(reply("b", "m1", 500), received);
        assert_eq!(store.list().await.unwrap().items.len(), 1);
        tokio::time::advance(Duration::from_millis(250)).await;
        assert!(store.list().await.unwrap().items.is_empty());
    }
}
