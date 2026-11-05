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
//! interval while the instance answers. Each copy is timed from when it was
//! asked for, so it never outlives its original, and unless its PoolRelay was
//! late, it has at least the lease minus a heartbeat interval left. The lease
//! must be at least three heartbeat intervals, so that copies outlive a round
//! that times out. Every instance must use the same heartbeat interval and
//! lease: each reply says which ones its instance uses, and one that differs
//! does not count as a complete answer. Neither does a reply from an instance
//! that has not heard from all of its own PoolRelays yet, as after a restart.
//!
//! The instances find each other the way PoolRelays find the routers: from
//! the service's own Service, exported through MCS, so its imported
//! EndpointSlices list every Ready instance in every cluster. The import in
//! each cluster must list that cluster's own instances too, or instances
//! behind one Service could not share, so an instance must find itself in the
//! list, which its own reply shows. They ask each other over plain HTTP, never
//! through a proxy, so like `POST /pools`, this assumes a trusted network.
//!
//! Lists are `Estimated` while registrations made with another instance may
//! be missing here: until the first round ends; while this instance does not
//! find itself in the list of instances, or the list may be incomplete; for
//! a lease after an instance leaves the list; once a listed instance has not
//! answered completely for the lease minus two heartbeat intervals, a
//! heartbeat interval before the copies from its last complete answer may
//! lapse; and from when a registration lapses here, a copy or one made here,
//! until every instance has answered completely again, since its PoolRelay
//! may have renewed it late, with any of them, as it does after a failed
//! heartbeat. The store keeps these times, so lists turn `Estimated` on time
//! even if the rounds stop. An instance that stays listed but cannot be
//! reached keeps lists `Estimated` until it answers: the pools its PoolRelays
//! registered may be missing here, and a complete list would let a Global
//! Router drop them. Once it has not answered for a lease, that is logged as
//! an error, and so is an instance that has not found itself for a lease.

use std::collections::hash_map::Entry;
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
    Completeness, ExchangeRound, Lease, RegistrationExchange, ReplicaDirectory, RouterReplica,
    RouterReplicaId, Snapshot,
};
use crate::service::{SharedRegistrations, Sharing};

/// Asks the other instances for their registrations, and keeps copies.
pub struct PeerSync {
    /// Every Ready instance, this one too, which is skipped once its own reply
    /// shows which one it is.
    peers: Arc<ReplicaDirectory>,
    /// Keeps the copies.
    store: Arc<dyn RegistrationExchange>,
    /// This instance's name in its `GET /registrations` replies.
    instance: String,
    /// The registrations' lease, which every instance must share.
    lease: Lease,
    /// How often to ask each instance, and how long to wait for its reply:
    /// half the heartbeat interval.
    every: Duration,
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

/// The instances that do not answer, since when, and whether that has been
/// logged as an error.
type Unreachable = HashMap<RouterReplicaId, (Instant, bool)>;

impl PeerSync {
    /// Shares `store`'s registrations with the instances `peers` lists, and
    /// keeps copies of theirs, under the registrations' `lease`, which must
    /// be at least three heartbeat intervals. Returns the task, to start with
    /// [`Self::run`], and what the service serves the others, under one new
    /// instance name. From now until the task's first round ends, `store`'s
    /// lists are `Estimated`. Call it inside a Tokio runtime.
    pub fn new(
        peers: Arc<ReplicaDirectory>,
        store: Arc<dyn RegistrationExchange>,
        lease: Lease,
    ) -> anyhow::Result<(Self, Sharing)> {
        anyhow::ensure!(
            lease.duration() >= lease.heartbeat_interval() * 3,
            "the lease must be at least three heartbeat intervals, so that copies outlive a \
             round that times out: {lease:?}"
        );
        let every = lease.heartbeat_interval() / 2;
        let client = reqwest::Client::builder()
            .timeout(every)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()?;
        // The other instances have not been asked yet.
        store.end_round(ExchangeRound::default());
        let instance = Uuid::new_v4().to_string();
        let synced = Arc::new(AtomicBool::new(false));
        let sharing = Sharing {
            instance: instance.clone(),
            lease,
            registrations: Arc::clone(&store),
            synced: Arc::clone(&synced),
        };
        let sync = Self {
            peers,
            store,
            instance,
            lease,
            every,
            synced,
            client,
        };
        Ok((sync, sharing))
    }

    /// Runs until `stop` is cancelled.
    pub async fn run(self, stop: CancellationToken) {
        // Logged when they stop answering, after a lease, and when they answer
        // again, not on every round.
        let mut unreachable = Unreachable::new();
        // Logged when they start to differ and when they match again.
        let mut other_lease: HashSet<RouterReplicaId> = HashSet::new();
        // The listed instance that is this one, once its own reply shows it.
        let mut own: Option<RouterReplicaId> = None;
        // Since when this instance has not found itself in the list, and
        // whether that has been logged as an error.
        let mut unfound: Option<(Instant, bool)> = None;
        // Logged when it changes. Zero means registrations stay local.
        let mut others_reached = None;
        let mut answers = Answers::default();
        let missing_after = missing_after(self.lease);
        let mut ticks = tokio::time::interval(self.every);
        ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                () = stop.cancelled() => return,
                _ = ticks.tick() => {}
            }
            let listed = match self.peers.list().await {
                Ok(listed) => listed,
                Err(error) => {
                    tracing::warn!(?error, "cannot list the discovery service instances");
                    self.store.end_round(ExchangeRound::default());
                    continue;
                }
            };
            let listed_ids: Vec<RouterReplicaId> = listed
                .items
                .iter()
                .map(|peer| peer.replica.clone())
                .collect();
            let mut peers = Snapshot {
                items: listed
                    .items
                    .into_iter()
                    .filter(|peer| own.as_ref() != Some(&peer.replica))
                    .collect(),
                completeness: listed.completeness,
            };
            let replies = join_all(peers.items.iter().map(|peer| fetch(&self.client, peer))).await;
            let mut reached = 0;
            for (peer, reply) in peers.items.iter().zip(replies) {
                let (reply, sent) = match reply {
                    Ok(answered) => answered,
                    Err(error) => {
                        self.note_unreachable(&mut unreachable, peer, &error);
                        continue;
                    }
                };
                if unreachable.remove(&peer.replica).is_some() {
                    tracing::info!(
                        peer = %peer.replica.0,
                        "discovery instance is reachable again"
                    );
                }
                if reply.instance == self.instance {
                    // Asked from now on are only the others.
                    answers.forget(&peer.replica);
                    own = Some(peer.replica.clone());
                    continue;
                }
                reached += 1;
                let same_lease = reply.lease == self.lease;
                if !same_lease && other_lease.insert(peer.replica.clone()) {
                    tracing::warn!(
                        peer = %peer.replica.0,
                        theirs = ?reply.lease,
                        ours = ?self.lease,
                        "another discovery instance uses another heartbeat interval or lease, so \
                         its copies may lapse between refreshes; lists stay incomplete until they \
                         match"
                    );
                } else if same_lease && other_lease.remove(&peer.replica) {
                    tracing::info!(peer = %peer.replica.0, "discovery instance uses the same lease again");
                }
                answers.record(&peer.replica, reply.complete && same_lease, sent);
                self.absorb(reply, sent);
            }
            // The instance found to be this one in this round is not another
            // one, and so it does not leave in the next.
            if let Some(own) = &own {
                peers.items.retain(|peer| &peer.replica != own);
            }
            let found_self = own.as_ref().is_some_and(|own| listed_ids.contains(own));
            self.note_unfound(&mut unfound, found_self);
            // Forget instances that left, so one that comes back is logged.
            let listed = |id: &RouterReplicaId| peers.items.iter().any(|peer| &peer.replica == id);
            unreachable.retain(|id, _| listed(id));
            other_lease.retain(|id| listed(id));
            let round =
                answers.end_round(&peers, missing_after, self.lease.duration(), Instant::now());
            self.store.end_round(if found_self {
                round
            } else {
                ExchangeRound::default()
            });
            if others_reached != Some(reached) {
                tracing::info!(reached, "other discovery instances reached");
                others_reached = Some(reached);
            }
            self.synced.store(true, Ordering::Release);
        }
    }

    /// Logs, once a lease has passed, that this instance has not found itself
    /// in the list of instances, and when it does again.
    fn note_unfound(&self, unfound: &mut Option<(Instant, bool)>, found_self: bool) {
        if found_self {
            if unfound.take().is_some_and(|(_, logged)| logged) {
                tracing::info!("this discovery instance is listed again");
            }
            return;
        }
        let (since, logged) = unfound.get_or_insert_with(|| (Instant::now(), false));
        if !*logged && since.elapsed() >= self.lease.duration() {
            *logged = true;
            tracing::error!(
                "this discovery instance has not found itself in the list of instances, so \
                 pool lists stay incomplete; check that its Service is exported through MCS \
                 and imported in this cluster, that the import lists this cluster's own \
                 instances, that --peer-service, --peer-namespace and --peer-port-name name it, \
                 and that this instance answers at its listed address"
            );
        }
    }

    /// Logs that `peer` did not answer: a warning when it stops answering, and
    /// an error once it has not answered for a lease, because lists then stay
    /// incomplete until it does.
    fn note_unreachable(
        &self,
        unreachable: &mut Unreachable,
        peer: &RouterReplica,
        error: &anyhow::Error,
    ) {
        let now = Instant::now();
        match unreachable.entry(peer.replica.clone()) {
            Entry::Vacant(entry) => {
                entry.insert((now, false));
                tracing::warn!(
                    peer = %peer.replica.0,
                    url = %peer.url,
                    error = format!("{error:#}"),
                    "cannot read another discovery instance's registrations"
                );
            }
            Entry::Occupied(mut entry) => {
                let (since, logged) = entry.get_mut();
                if !*logged && now.saturating_duration_since(*since) >= self.lease.duration() {
                    *logged = true;
                    tracing::error!(
                        peer = %peer.replica.0,
                        url = %peer.url,
                        error = format!("{error:#}"),
                        "another discovery instance has not answered for a lease; pool lists stay \
                         incomplete until it does, so Global Routers keep pools that may have left"
                    );
                }
            }
        }
    }

    /// Copies each registration in `reply` for the time it had left, counted
    /// from when the request was `sent`, so neither the reply's transit nor
    /// waiting for a slower instance lengthens it. Skips this instance's own
    /// reply: copying its own registrations back would make them copies,
    /// which are not shared.
    fn absorb(&self, reply: SharedRegistrations, sent: Instant) {
        if reply.instance == self.instance {
            return;
        }
        let waited = sent.elapsed();
        for shared in reply.registrations {
            let (registration, remaining) = shared.into_registration();
            self.store
                .adopt(registration, remaining.saturating_sub(waited));
        }
    }
}

/// How long after another instance was last asked for an answer it gave in
/// full, lists turn `Estimated` unless it has answered in full again. Each
/// copy from that answer has at least the lease minus a heartbeat interval
/// left, unless its PoolRelay was late, so this is a heartbeat interval
/// sooner, which leaves the next round time to end and the store time to tell
/// watchers.
fn missing_after(lease: Lease) -> Duration {
    lease
        .duration()
        .saturating_sub(lease.heartbeat_interval() * 2)
}

/// What the rounds so far say about each instance, to tell whether copies from
/// another instance may be missing here.
#[derive(Default)]
struct Answers {
    /// Each listed instance, and when it last answered with all of its
    /// registrations, by when the request was sent: `None` until it has.
    listed: HashMap<RouterReplicaId, Option<Instant>>,
    /// Instances that left the list, and when, whether or not they ever
    /// answered completely. Copies from them may still lapse, and their
    /// PoolRelays may not have registered with another instance yet.
    departed: HashMap<RouterReplicaId, Instant>,
}

impl Answers {
    /// Notes a reply to a request `sent` then. A reply from an instance that
    /// has not heard from all of its PoolRelays yet, such as one that just
    /// restarted, does not count.
    fn record(&mut self, peer: &RouterReplicaId, complete: bool, sent: Instant) {
        let answered = self.listed.entry(peer.clone()).or_default();
        if complete {
            *answered = Some(sent);
        }
    }

    /// Stops tracking `peer`, which is this instance, without counting it as
    /// one that left.
    fn forget(&mut self, peer: &RouterReplicaId) {
        self.listed.remove(peer);
    }

    /// Ends a round over `peers` at `now`. Copies may be missing, and the
    /// round says nothing, while the list of instances may be incomplete,
    /// while a listed one has never answered completely, and for
    /// `lease_duration` after one leaves. Otherwise the round gives the
    /// oldest of the listed instances' last complete answers, or `now` with
    /// none listed, and vouches for `missing_after` past it.
    fn end_round(
        &mut self,
        peers: &Snapshot<RouterReplica>,
        missing_after: Duration,
        lease_duration: Duration,
        now: Instant,
    ) -> ExchangeRound {
        let listed = |id: &RouterReplicaId| peers.items.iter().any(|peer| &peer.replica == id);
        let left: Vec<RouterReplicaId> = self
            .listed
            .keys()
            .filter(|id| !listed(id))
            .cloned()
            .collect();
        for id in left {
            self.listed.remove(&id);
            self.departed.insert(id, now);
        }
        for peer in &peers.items {
            self.listed.entry(peer.replica.clone()).or_default();
        }
        self.departed.retain(|id, since| {
            !listed(id) && now.saturating_duration_since(*since) < lease_duration
        });
        if peers.completeness != Completeness::Exact || !self.departed.is_empty() {
            return ExchangeRound::default();
        }
        let mut oldest = now;
        for peer in &peers.items {
            let Some(answered) = self.listed.get(&peer.replica).copied().flatten() else {
                return ExchangeRound::default();
            };
            oldest = oldest.min(answered);
        }
        ExchangeRound {
            answered_since: Some(oldest),
            complete_until: Some(oldest + missing_after),
        }
    }
}

/// Also returns when the request was sent.
async fn fetch(
    client: &reqwest::Client,
    peer: &RouterReplica,
) -> anyhow::Result<(SharedRegistrations, Instant)> {
    let url = peer.url.join("registrations")?;
    let sent = Instant::now();
    let reply = client
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok((reply, sent))
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use dynamo_global_common::{Lease, PoolKey, PoolLocation};
    use serde_json::{Value, json};
    use tokio::net::TcpListener;
    use url::Url;

    use super::*;
    use crate::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
    use crate::pool_discovery::{
        Directory, DiscoveryError, PoolDirectory, PoolRegistrar, PoolRegistration, WatchStream,
    };
    use crate::service::{self, PoolDiscoveryBackend};

    /// A PoolRelay registers every second, and a registration lasts three.
    fn lease() -> Lease {
        Lease::new(1, 3).unwrap()
    }

    fn url_of(listener: &TcpListener) -> Url {
        Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap()
    }

    /// The instances a test lists, which it can change.
    #[derive(Default)]
    struct ChangingReplicas(std::sync::Mutex<Vec<RouterReplica>>);

    impl ChangingReplicas {
        fn set(&self, replicas: Vec<RouterReplica>) {
            *self.0.lock().unwrap() = replicas;
        }
    }

    #[async_trait]
    impl Directory<RouterReplica, RouterReplicaId> for ChangingReplicas {
        async fn list(&self) -> Result<Snapshot<RouterReplica>, DiscoveryError> {
            Ok(Snapshot {
                items: self.0.lock().unwrap().clone(),
                completeness: Completeness::Exact,
            })
        }

        async fn list_and_watch(
            &self,
            _cancel_token: Option<CancellationToken>,
        ) -> Result<WatchStream<RouterReplica, RouterReplicaId>, DiscoveryError> {
            unreachable!("PeerSync only lists the instances")
        }
    }

    /// Serves one instance that shares its registrations. Returns its store,
    /// its URL, the task that asks the instances it lists, not yet started,
    /// and that list, empty until the test sets it.
    async fn instance(
        stop: &CancellationToken,
    ) -> (Arc<MemoryPoolStore>, Url, PeerSync, Arc<ChangingReplicas>) {
        instance_with_lease(stop, lease()).await
    }

    /// [`instance`], under `lease`.
    async fn instance_with_lease(
        stop: &CancellationToken,
        lease: Lease,
    ) -> (Arc<MemoryPoolStore>, Url, PeerSync, Arc<ChangingReplicas>) {
        let store = Arc::new(MemoryPoolStore::new(lease));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = url_of(&listener);
        let peers = Arc::new(ChangingReplicas::default());
        let (sync, sharing) = PeerSync::new(peers.clone(), store.clone(), lease).unwrap();
        let pools: Arc<PoolDirectory> = store.clone();
        let registrar: Arc<dyn PoolRegistrar> = store.clone();
        tokio::spawn(service::serve(
            listener,
            PoolDiscoveryBackend {
                pools,
                routers: Arc::new(StaticReplicaDirectory::new(Vec::new())),
                registrar: Some(registrar),
                sharing: Some(sharing),
            },
            None,
            stop.clone(),
        ));
        (store, url, sync, peers)
    }

    /// Serves another instance, under `lease`, whose PoolRelay registers one
    /// pool at `registered`, if given, then not again until `renewed`. It
    /// answers at once until `renewed`, then `delay` after it is asked.
    async fn scripted_instance(
        stop: &CancellationToken,
        lease: Lease,
        registered: Option<Instant>,
        renewed: Instant,
        delay: Duration,
    ) -> Url {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = url_of(&listener);
        let registrations = move || {
            let now = Instant::now();
            let last = if now >= renewed {
                renewed
            } else if let Some(registered) = registered.filter(|registered| now >= *registered) {
                registered
            } else {
                return Vec::new();
            };
            let left = (last + lease.duration()).saturating_duration_since(now);
            if left.is_zero() {
                return Vec::new();
            }
            vec![json!({
                "registration": registration("m1"),
                "expires_in_ms": u64::try_from(left.as_millis()).unwrap(),
            })]
        };
        let app = axum::Router::new().route(
            "/registrations",
            axum::routing::get(move || async move {
                let registrations = registrations();
                if Instant::now() >= renewed {
                    tokio::time::sleep(delay).await;
                }
                axum::Json(json!({
                    "instance": "scripted",
                    "lease": lease,
                    "complete": true,
                    "registrations": registrations,
                }))
            }),
        );
        let stop = stop.clone();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(stop.cancelled_owned())
                .await
        });
        url
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

    /// Waits up to six seconds for `store`'s lists to be `expected`.
    async fn wait_for_completeness(store: &MemoryPoolStore, expected: Completeness) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(6);
        while store.list().await.unwrap().completeness != expected {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the lists are not {expected:?}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    async fn registrations_reach_the_other_instances_and_lapse_there() {
        let stop = CancellationToken::new();
        // A lists only itself.
        let (a, a_url, a_sync, a_peers) = instance(&stop).await;
        a_peers.set(vec![peer("a", &a_url)]);
        tokio::spawn(a_sync.run(stop.clone()));
        let _ = a.register(pool("m1")).await.unwrap();
        // B asks every instance: A, itself, one that is gone, and one that
        // never answers.
        let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gone = url_of(&closed);
        drop(closed);
        let silent = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let silent_url = url_of(&silent);
        let (b, b_url, b_sync, b_peers) = instance(&stop).await;
        b_peers.set(vec![
            peer("a", &a_url),
            peer("b", &b_url),
            peer("gone", &gone),
            peer("silent", &silent_url),
        ]);
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
        wait_for_completeness(&a, Completeness::Exact).await;
        assert_eq!(
            b.list().await.unwrap().completeness,
            Completeness::Estimated
        );
        drop(silent);
        stop.cancel();
    }

    #[tokio::test]
    async fn instances_behind_one_service_keep_sharing_a_pool_whose_heartbeats_alternate() {
        let stop = CancellationToken::new();
        // A and B serve one cluster behind one Service, so a PoolRelay's
        // heartbeats reach either of them. C, in another cluster, asks both.
        let (a, a_url, a_sync, a_peers) = instance(&stop).await;
        let (b, b_url, b_sync, b_peers) = instance(&stop).await;
        let (c, c_url, c_sync, c_peers) = instance(&stop).await;
        a_peers.set(vec![peer("a", &a_url), peer("b", &b_url)]);
        b_peers.set(vec![peer("a", &a_url), peer("b", &b_url)]);
        c_peers.set(vec![
            peer("a", &a_url),
            peer("b", &b_url),
            peer("c", &c_url),
        ]);
        for sync in [a_sync, b_sync, c_sync] {
            tokio::spawn(sync.run(stop.clone()));
        }

        for (heard, other) in [(&a, &b), (&b, &a), (&a, &b), (&b, &a)] {
            let _ = heard.register(pool("m1")).await.unwrap();
            // A heartbeat interval later, the instance that took the last
            // heartbeat shares the pool, the other keeps only a copy, and C
            // still lists the pool.
            tokio::time::sleep(lease().heartbeat_interval()).await;
            assert_eq!(heard.local_registrations().len(), 1);
            assert!(other.local_registrations().is_empty());
            assert_eq!(models(&c).await, ["m1"]);
        }
        stop.cancel();
    }

    #[tokio::test]
    async fn an_instance_that_restarts_or_leaves_keeps_lists_incomplete_until_its_copies_settle() {
        let stop = CancellationToken::new();
        let (_a, a_url, _, _) = instance(&stop).await;
        let (b, b_url, b_sync, b_peers) = instance(&stop).await;
        b_peers.set(vec![peer("a", &a_url), peer("b", &b_url)]);
        tokio::spawn(b_sync.run(stop.clone()));
        // A answers completely a lease after it started, and so B's lists are
        // complete.
        wait_for_completeness(&b, Completeness::Exact).await;

        // A restarts: its first answers are not complete, so B's lists stay
        // incomplete until a lease after A restarted.
        let restarted_at = Instant::now();
        let (_restarted, restarted_url, _, _) = instance(&stop).await;
        b_peers.set(vec![peer("a", &restarted_url), peer("b", &b_url)]);
        wait_for_completeness(&b, Completeness::Estimated).await;
        wait_for_completeness(&b, Completeness::Exact).await;
        assert!(restarted_at.elapsed() >= lease().duration());

        // A leaves: B's lists are incomplete at once, and complete again a
        // lease later, when copies from A have lapsed.
        let left_at = Instant::now();
        b_peers.set(vec![peer("b", &b_url)]);
        wait_for_completeness(&b, Completeness::Estimated).await;
        wait_for_completeness(&b, Completeness::Exact).await;
        assert!(left_at.elapsed() >= lease().duration());
        stop.cancel();
    }

    #[tokio::test]
    async fn an_instance_stops_asking_itself_once_its_own_reply_shows_which_one_it_is() {
        let stop = CancellationToken::new();
        let (b, b_url, b_sync, b_peers) = instance(&stop).await;
        let started = Instant::now();
        b_peers.set(vec![peer("b", &b_url)]);
        let b_stop = CancellationToken::new();
        tokio::spawn(b_sync.run(b_stop.clone()));
        // After one round, B knows which listed instance it is.
        wait_for_ready(&b_url).await;
        // Its entry now has an address where nothing answers, as when it
        // listens on another port than its Service's target port. B does not
        // ask itself, nor count itself as an instance that left, so its lists
        // are complete a lease after it started.
        let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let nobody = url_of(&closed);
        drop(closed);
        b_peers.set(vec![peer("b", &nobody)]);
        wait_for_completeness(&b, Completeness::Exact).await;
        assert!(
            started.elapsed() < lease().duration() + Duration::from_millis(400),
            "complete only {:?} after B started",
            started.elapsed()
        );

        // If B stops asking, its lists turn incomplete on their own once the
        // last round's answer has aged, as it would if another instance had
        // stopped answering.
        b_stop.cancel();
        wait_for_completeness(&b, Completeness::Estimated).await;
        stop.cancel();
    }

    #[tokio::test]
    async fn an_instance_with_another_lease_keeps_lists_incomplete() {
        let stop = CancellationToken::new();
        // A's registrations last four seconds, B's three.
        let (a, a_url, _, _) = instance_with_lease(&stop, Lease::new(1, 4).unwrap()).await;
        let _ = a.register(pool("m1")).await.unwrap();
        let (b, b_url, b_sync, b_peers) = instance(&stop).await;
        b_peers.set(vec![peer("a", &a_url), peer("b", &b_url)]);
        tokio::spawn(b_sync.run(stop.clone()));
        // B copies A's registration, but even once A has answered completely,
        // a lease after it started, its copies may lapse between B's
        // refreshes, so B's lists stay incomplete.
        wait_for_models(&b, &["m1"]).await;
        tokio::time::sleep(Duration::from_secs(5)).await;
        let _ = a.register(pool("m1")).await.unwrap();
        assert_eq!(
            b.list().await.unwrap().completeness,
            Completeness::Estimated
        );
        stop.cancel();
    }

    /// A PoolRelay registers its pool a second after B started, with B or with
    /// another instance, under a 2 s heartbeat and a 6 s lease, which leave
    /// more room for a slow test runner than the 1 s and 3 s of the other
    /// tests. Its later heartbeats fail, and its retry, 5.8 s later, late as a
    /// retry 1 s before the lease ends, reaches the other instance, which from
    /// then on answers 600 ms after it is asked, within B's 1 s timeout. So
    /// the registration lapses at B before any reply with the renewal can
    /// arrive.
    async fn a_late_renewal_keeps_lists_incomplete(registered_with_b: bool) {
        let lease = Lease::new(2, 6).unwrap();
        let stop = CancellationToken::new();
        let (b, b_url, b_sync, b_peers) = instance_with_lease(&stop, lease).await;
        tokio::spawn(b_sync.run(stop.clone()));
        tokio::time::sleep(Duration::from_secs(1)).await;
        let registered = Instant::now();
        if registered_with_b {
            let _ = b.register(pool("m1")).await.unwrap();
        }
        let renewed = registered + Duration::from_millis(5800);
        let other = scripted_instance(
            &stop,
            lease,
            (!registered_with_b).then_some(registered),
            renewed,
            Duration::from_millis(600),
        )
        .await;
        b_peers.set(vec![peer("b", &b_url), peer("other", &other)]);

        // A lease after B started, B lists the pool, completely.
        tokio::time::sleep_until(registered + Duration::from_millis(5400)).await;
        let listed = b.list().await.unwrap();
        assert_eq!(listed.items.len(), 1);
        assert_eq!(listed.completeness, Completeness::Exact);

        // Until the renewal arrives, B's list lacks a pool that is live all
        // along, so it must not say that it is complete: a Global Router
        // would drop the pool.
        let mut lapsed = 0;
        while Instant::now() < renewed + Duration::from_secs(2) {
            let listed = b.list().await.unwrap();
            if listed.items.is_empty() {
                lapsed += 1;
                assert_eq!(listed.completeness, Completeness::Estimated);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(lapsed > 0, "the pool never lapsed at B before the renewal");
        // Then B lists the pool again, completely.
        wait_for_models(&b, &["m1"]).await;
        wait_for_completeness(&b, Completeness::Exact).await;
        stop.cancel();
    }

    #[tokio::test]
    async fn a_late_renewal_keeps_lists_incomplete_while_a_copy_has_lapsed() {
        a_late_renewal_keeps_lists_incomplete(false).await;
    }

    #[tokio::test]
    async fn a_late_renewal_with_another_instance_keeps_lists_incomplete() {
        a_late_renewal_keeps_lists_incomplete(true).await;
    }

    /// Whether lists would be `Estimated` at `at` if a round over `peers`
    /// ended then, with no registration lapsed.
    fn missing(
        answers: &mut Answers,
        peers: &Snapshot<RouterReplica>,
        lease: Lease,
        at: Instant,
    ) -> bool {
        answers
            .end_round(peers, missing_after(lease), lease.duration(), at)
            .complete_until
            .is_none_or(|until| at >= until)
    }

    #[test]
    fn copies_may_be_missing_before_an_instance_misses_its_lease() {
        // A 1 s heartbeat and a 3 s lease: copies from an instance's last
        // answer count as missing 1 s after it.
        let lease = lease();
        assert_eq!(missing_after(lease), Duration::from_secs(1));
        let url = Url::parse("http://10.43.0.9:8080/").unwrap();
        let peers = |completeness, ids: &[&str]| Snapshot {
            items: ids.iter().map(|id| peer(id, &url)).collect(),
            completeness,
        };
        let (a, b) = (
            RouterReplicaId("a".to_owned()),
            RouterReplicaId("b".to_owned()),
        );
        let then = Instant::now();
        let both = || {
            let mut answers = Answers::default();
            answers.record(&a, true, then);
            answers.record(&b, true, then);
            answers
        };
        let half = Duration::from_millis(500);

        assert!(!missing(
            &mut both(),
            &peers(Completeness::Exact, &["a", "b"]),
            lease,
            then + half
        ));
        // Before their copies may lapse, not when the lease ends.
        assert!(missing(
            &mut both(),
            &peers(Completeness::Exact, &["a", "b"]),
            lease,
            then + missing_after(lease)
        ));
        // An answer without all of the instance's registrations, as a
        // restarted instance's first answers are, does not count.
        let mut restarted = Answers::default();
        restarted.record(&a, true, then);
        restarted.record(&b, false, then);
        assert!(missing(
            &mut restarted,
            &peers(Completeness::Exact, &["a", "b"]),
            lease,
            then
        ));
        // The list of instances may be incomplete.
        assert!(missing(
            &mut both(),
            &peers(Completeness::Estimated, &["a", "b"]),
            lease,
            then
        ));

        // An instance that left: copies from it may lapse for a lease after.
        let mut answers = both();
        let only_a = peers(Completeness::Exact, &["a"]);
        assert!(missing(&mut answers, &only_a, lease, then));
        let later = then + half;
        answers.record(&a, true, later);
        assert!(missing(&mut answers, &only_a, lease, later));
        let after_a_lease = then + lease.duration();
        answers.record(&a, true, after_a_lease);
        assert!(!missing(&mut answers, &only_a, lease, after_a_lease));

        // So does one that never answered completely: its PoolRelays may not
        // have registered with another instance yet.
        let mut answers = Answers::default();
        answers.record(&a, true, then);
        assert!(missing(
            &mut answers,
            &peers(Completeness::Exact, &["a", "b"]),
            lease,
            then
        ));
        assert!(missing(&mut answers, &only_a, lease, then));

        // With no other instance, nothing can be missing until the next round
        // is due.
        let mut answers = Answers::default();
        let none = peers(Completeness::Exact, &[]);
        let round = answers.end_round(&none, missing_after(lease), lease.duration(), then);
        assert_eq!(round.answered_since, Some(then));
        assert_eq!(round.complete_until, Some(then + missing_after(lease)));
    }

    #[test]
    fn copies_count_as_missing_a_heartbeat_before_they_may_lapse() {
        // A 5 s heartbeat and a 13 s lease. Copies from an instance's last
        // complete answer, asked for at `then`, may lapse 8 s later, so lists
        // turn `Estimated` 3 s after it, whenever the rounds end: one that
        // ends at once, or one that waits out its 2.5 s timeout.
        let lease = Lease::new(5, 13).unwrap();
        let url = Url::parse("http://10.43.0.9:8080/").unwrap();
        let peers = Snapshot {
            items: vec![peer("a", &url)],
            completeness: Completeness::Exact,
        };
        let mut answers = Answers::default();
        let then = Instant::now();
        answers.record(&RouterReplicaId("a".to_owned()), true, then);
        for ended in [2900, 5400] {
            let at = then + Duration::from_millis(ended);
            let round = answers.end_round(&peers, missing_after(lease), lease.duration(), at);
            assert_eq!(round.answered_since, Some(then));
            assert_eq!(round.complete_until, Some(then + Duration::from_secs(3)));
        }
    }

    /// A `GET /registrations` reply from `instance`.
    fn reply(instance: &str, model: &str, expires_in_ms: u64) -> SharedRegistrations {
        serde_json::from_value(json!({
            "instance": instance,
            "lease": {"heartbeat_interval_s": 1, "lease_duration_s": 3},
            "complete": true,
            "registrations": [{"registration": registration(model), "expires_in_ms": expires_in_ms}],
        }))
        .unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn lists_are_estimated_before_the_first_round_and_until_this_instance_finds_itself() {
        let store = Arc::new(MemoryPoolStore::new(lease()));
        let (sync, _) = PeerSync::new(
            Arc::new(StaticReplicaDirectory::new(Vec::new())),
            store.clone(),
            lease(),
        )
        .unwrap();
        // A lease has passed, but the other instances have not been asked.
        tokio::time::advance(lease().duration()).await;
        assert_eq!(
            store.list().await.unwrap().completeness,
            Completeness::Estimated
        );
        // The rounds list no instance, not even this one, as when this
        // Service is not imported here, so the others' pools may be missing.
        let stop = CancellationToken::new();
        tokio::spawn(sync.run(stop.clone()));
        tokio::time::sleep(lease().duration()).await;
        assert_eq!(
            store.list().await.unwrap().completeness,
            Completeness::Estimated
        );
        stop.cancel();
    }

    #[tokio::test]
    async fn a_lease_under_three_heartbeat_intervals_is_refused() {
        let lease = Lease::new(1, 2).unwrap();
        let error = PeerSync::new(
            Arc::new(StaticReplicaDirectory::new(Vec::new())),
            Arc::new(MemoryPoolStore::new(lease)),
            lease,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("three heartbeat intervals"),
            "{error}"
        );
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
        sync.absorb(reply(&sync.instance, "m1", 500), Instant::now());
        assert!(store.list().await.unwrap().items.is_empty());

        // The request to another instance was sent 300 ms ago, for example
        // as it waited for a slower one's reply, so its copy has 200 ms
        // left, not 500.
        let sent = Instant::now();
        tokio::time::advance(Duration::from_millis(300)).await;
        sync.absorb(reply("b", "m1", 500), sent);
        assert_eq!(store.list().await.unwrap().items.len(), 1);
        tokio::time::advance(Duration::from_millis(250)).await;
        assert!(store.list().await.unwrap().items.is_empty());
    }
}
