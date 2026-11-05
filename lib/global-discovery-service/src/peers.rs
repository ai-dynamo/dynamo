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
//! list, which its own reply shows. They read each other through a
//! [`RegistrationSource`]: over plain HTTP, which like `POST /pools` assumes
//! a trusted network.
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
    Completeness, DiscoveryError, ExchangeRound, Lease, RegistrationExchange, RegistrationSharing,
    RegistrationSource, ReplicaDirectory, RouterReplica, RouterReplicaId, SharedRegistration,
    SharedRegistrations, Snapshot,
};

/// The shortest lease that sharing allows, in heartbeat intervals: a copy
/// must outlive a round that times out.
pub const MIN_LEASE_HEARTBEATS: u32 = 3;

/// What this instance serves the others, and whether it has asked them yet.
#[derive(Clone)]
pub struct Sharing {
    /// Names this process in its replies, so it can tell its own.
    instance: String,
    /// The registrations' lease, which every instance must share.
    lease: Lease,
    registrations: Arc<dyn RegistrationExchange>,
    synced: Arc<AtomicBool>,
}

impl RegistrationSharing for Sharing {
    fn reply(&self) -> SharedRegistrations {
        SharedRegistrations {
            instance: self.instance.clone(),
            lease: self.lease,
            complete: self.registrations.holds_every_local_registration(),
            registrations: self
                .registrations
                .local_registrations()
                .into_iter()
                .map(|(registration, remaining)| SharedRegistration::new(registration, remaining))
                .collect(),
        }
    }

    fn synced(&self) -> bool {
        self.synced.load(Ordering::Acquire)
    }
}

impl fmt::Debug for Sharing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sharing")
            .field("instance", &self.instance)
            .field("synced", &self.synced())
            .finish_non_exhaustive()
    }
}

/// Asks the other instances for their registrations, and keeps copies.
pub struct PeerSync {
    /// Every Ready instance, this one too.
    instances: Arc<ReplicaDirectory>,
    source: Arc<dyn RegistrationSource>,
    /// What this instance shares, and where it keeps the copies.
    sharing: Sharing,
    /// How often to ask each instance, and how long to wait for its reply:
    /// half the heartbeat interval.
    every: Duration,
}

impl fmt::Debug for PeerSync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PeerSync")
            .field("sharing", &self.sharing)
            .field("every", &self.every)
            .finish_non_exhaustive()
    }
}

impl PeerSync {
    /// Shares `store`'s registrations with the instances that `instances`
    /// lists, read through `source`, and keeps copies of theirs, under the
    /// registrations' `lease`, which must be at least
    /// [`MIN_LEASE_HEARTBEATS`] heartbeat intervals. Returns the task, to
    /// start with [`Self::run`], and what the service serves, under one new
    /// instance name. From now until the task's first round ends, `store`'s
    /// lists are `Estimated`. Call it inside a Tokio runtime.
    pub fn new(
        instances: Arc<ReplicaDirectory>,
        source: Arc<dyn RegistrationSource>,
        store: Arc<dyn RegistrationExchange>,
        lease: Lease,
    ) -> anyhow::Result<(Self, Sharing)> {
        anyhow::ensure!(
            lease.duration() >= lease.heartbeat_interval() * MIN_LEASE_HEARTBEATS,
            "the lease must be at least {MIN_LEASE_HEARTBEATS} heartbeat intervals, so that \
             copies outlive a round that times out: {lease:?}"
        );
        // The other instances have not been asked yet.
        store.end_round(ExchangeRound::default());
        let sharing = Sharing {
            instance: Uuid::new_v4().to_string(),
            lease,
            registrations: store,
            synced: Arc::default(),
        };
        let sync = Self {
            instances,
            source,
            sharing: sharing.clone(),
            every: lease.heartbeat_interval() / 2,
        };
        Ok((sync, sharing))
    }

    /// Runs until `stop` is cancelled.
    pub async fn run(self, stop: CancellationToken) {
        let lease = self.sharing.lease;
        let missing_after = missing_after(lease);
        let mut instances = Instances::default();
        // Logged when it changes. Zero means registrations stay local.
        let mut reached = None;
        let mut ticks = tokio::time::interval(self.every);
        ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                () = stop.cancelled() => return,
                _ = ticks.tick() => {}
            }
            let listed = match self.instances.list().await {
                Ok(listed) => listed,
                Err(error) => {
                    tracing::warn!(?error, "cannot list the discovery service instances");
                    self.sharing
                        .registrations
                        .end_round(ExchangeRound::default());
                    continue;
                }
            };
            let others: Vec<&RouterReplica> = listed
                .items
                .iter()
                .filter(|peer| !instances.is_own(&peer.replica))
                .collect();
            let replies = tokio::select! {
                biased;
                () = stop.cancelled() => return,
                replies = join_all(others.iter().map(|peer| self.ask(peer))) => replies,
            };
            let mut answered = 0;
            for (peer, (sent, reply)) in others.into_iter().zip(replies) {
                match reply {
                    Err(error) => instances.unreachable(peer, &error, lease, Instant::now()),
                    Ok(reply) if reply.instance == self.sharing.instance => {
                        instances.own(&peer.replica);
                    }
                    Ok(reply) => {
                        answered += 1;
                        instances.answered(peer, &reply, sent, lease);
                        self.absorb(reply, sent);
                    }
                }
            }
            let round = instances.end_round(&listed, missing_after, lease, Instant::now());
            self.sharing.registrations.end_round(round);
            if reached != Some(answered) {
                tracing::info!(reached = answered, "other discovery instances reached");
                reached = Some(answered);
            }
            self.sharing.synced.store(true, Ordering::Release);
        }
    }

    /// Asks `peer`, waiting at most `every`. Also returns when it was asked.
    async fn ask(&self, peer: &RouterReplica) -> (Instant, anyhow::Result<SharedRegistrations>) {
        let sent = Instant::now();
        let reply = tokio::time::timeout(self.every, self.source.shared_registrations(peer)).await;
        let reply = match reply {
            Ok(reply) => reply.map_err(|error| match error {
                DiscoveryError::Unavailable(cause) => cause,
                other => other.into(),
            }),
            Err(_) => Err(anyhow::anyhow!("no answer within {:?}", self.every)),
        };
        (sent, reply)
    }

    /// Copies each registration in `reply` for the time it had left, counted
    /// from when the request was `sent`, so neither the reply's transit nor
    /// waiting for a slower instance lengthens it.
    fn absorb(&self, reply: SharedRegistrations, sent: Instant) {
        let waited = sent.elapsed();
        for shared in reply.registrations {
            let (registration, remaining) = shared.into_parts();
            self.sharing
                .registrations
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

/// What the rounds so far say about the instances, to tell whether
/// registrations made with another instance may be missing here. Each change
/// is logged once, not every round.
#[derive(Default)]
struct Instances {
    /// Each listed instance other than this one.
    others: HashMap<RouterReplicaId, Instance>,
    /// Instances that left the list, and when. Copies from them may still
    /// lapse, and their PoolRelays may not have registered with another
    /// instance yet.
    departed: HashMap<RouterReplicaId, Instant>,
    /// The names that the list gives this instance, as its own replies showed
    /// them. One listed by address may be listed under two, IPv4 and IPv6.
    own: HashSet<RouterReplicaId>,
    /// Since when this instance has not found itself in the list.
    unfound: Option<Problem>,
}

#[derive(Default)]
struct Instance {
    /// When it was last asked for an answer that it gave with all of its
    /// registrations, under the same lease.
    answered: Option<Instant>,
    /// Since when it has not answered.
    unreachable: Option<Problem>,
    /// It uses another heartbeat interval or lease.
    other_lease: bool,
}

/// A problem that is logged once it has lasted a lease.
struct Problem {
    since: Instant,
    logged: bool,
}

impl Problem {
    fn new(since: Instant) -> Self {
        Self {
            since,
            logged: false,
        }
    }

    /// True once: the first time it is asked after the problem has lasted
    /// `lease`.
    fn lasted(&mut self, lease: Lease, now: Instant) -> bool {
        let lasted = !self.logged && now.saturating_duration_since(self.since) >= lease.duration();
        self.logged |= lasted;
        lasted
    }
}

impl Instances {
    fn is_own(&self, id: &RouterReplicaId) -> bool {
        self.own.contains(id)
    }

    /// The listed instance `id` is this one: it asks only the others from
    /// now on, and does not count itself as one that left.
    fn own(&mut self, id: &RouterReplicaId) {
        self.others.remove(id);
        self.departed.remove(id);
        self.own.insert(id.clone());
    }

    /// `peer` did not answer: logged when it stops answering, and as an error
    /// once it has not answered for a lease, since lists then stay incomplete
    /// until it does.
    fn unreachable(
        &mut self,
        peer: &RouterReplica,
        error: &anyhow::Error,
        lease: Lease,
        now: Instant,
    ) {
        let instance = self.others.entry(peer.replica.clone()).or_default();
        let Some(problem) = &mut instance.unreachable else {
            instance.unreachable = Some(Problem::new(now));
            tracing::warn!(
                peer = %peer.replica.0,
                url = %peer.url,
                error = format!("{error:#}"),
                "cannot read another discovery instance's registrations"
            );
            return;
        };
        if problem.lasted(lease, now) {
            tracing::error!(
                peer = %peer.replica.0,
                url = %peer.url,
                error = format!("{error:#}"),
                "another discovery instance has not answered for a lease; pool lists stay \
                 incomplete until it does, so Global Routers keep pools that may have left"
            );
        }
    }

    /// `peer` answered a request `sent` then. The answer counts only if it
    /// holds all of the instance's registrations, under the same lease.
    fn answered(
        &mut self,
        peer: &RouterReplica,
        reply: &SharedRegistrations,
        sent: Instant,
        lease: Lease,
    ) {
        let instance = self.others.entry(peer.replica.clone()).or_default();
        if instance.unreachable.take().is_some() {
            tracing::info!(peer = %peer.replica.0, "discovery instance is reachable again");
        }
        let other_lease = reply.lease != lease;
        if other_lease != instance.other_lease {
            instance.other_lease = other_lease;
            if other_lease {
                tracing::warn!(
                    peer = %peer.replica.0,
                    theirs = ?reply.lease,
                    ours = ?lease,
                    "another discovery instance uses another heartbeat interval or lease, so its \
                     copies may lapse between refreshes; lists stay incomplete until they match"
                );
            } else {
                tracing::info!(peer = %peer.replica.0, "discovery instance uses the same lease again");
            }
        }
        if reply.complete && !other_lease {
            instance.answered = Some(sent);
        }
    }

    /// Ends a round over the `listed` instances at `now`. The round vouches
    /// for nothing while this instance does not find itself in the list,
    /// while the list may be incomplete, while another listed instance has
    /// never answered completely, and for a lease after one leaves. Otherwise
    /// it gives the oldest of the other instances' last complete answers, or
    /// `now` with none, and vouches for `missing_after` past it.
    fn end_round(
        &mut self,
        listed: &Snapshot<RouterReplica>,
        missing_after: Duration,
        lease: Lease,
        now: Instant,
    ) -> ExchangeRound {
        let found_self = listed
            .items
            .iter()
            .any(|peer| self.own.contains(&peer.replica));
        self.note_found(found_self, lease, now);
        let others: HashSet<&RouterReplicaId> = listed
            .items
            .iter()
            .map(|peer| &peer.replica)
            .filter(|id| !self.own.contains(*id))
            .collect();
        let left: Vec<RouterReplicaId> = self
            .others
            .keys()
            .filter(|id| !others.contains(id))
            .cloned()
            .collect();
        for id in left {
            self.others.remove(&id);
            self.departed.insert(id, now);
        }
        for id in &others {
            self.others.entry((*id).clone()).or_default();
        }
        self.departed.retain(|id, since| {
            !others.contains(id) && now.saturating_duration_since(*since) < lease.duration()
        });
        if !found_self || listed.completeness != Completeness::Exact || !self.departed.is_empty() {
            return ExchangeRound::default();
        }
        let mut oldest = now;
        for id in &others {
            let Some(answered) = self.others.get(*id).and_then(|instance| instance.answered) else {
                return ExchangeRound::default();
            };
            oldest = oldest.min(answered);
        }
        ExchangeRound {
            answered_since: Some(oldest),
            complete_until: Some(oldest + missing_after),
        }
    }

    /// Logs, once a lease has passed, that this instance has not found itself
    /// in the list of instances, and when it does again.
    fn note_found(&mut self, found_self: bool, lease: Lease, now: Instant) {
        if found_self {
            if self.unfound.take().is_some_and(|problem| problem.logged) {
                tracing::info!("this discovery instance is listed again");
            }
        } else if self
            .unfound
            .get_or_insert_with(|| Problem::new(now))
            .lasted(lease, now)
        {
            tracing::error!(
                "this discovery instance has not found itself in the list of instances, so pool \
                 lists stay incomplete; check that its Service is exported through MCS and \
                 imported into this cluster as well (with Karmada, that the ServiceImport is \
                 placed on this cluster too), that --peer-service, --peer-namespace and \
                 --peer-port-name name it, and that this instance answers at its listed address"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use dynamo_global_common::{PoolKey, PoolLocation};
    use tokio::net::TcpListener;
    use url::Url;

    use super::*;
    use crate::backends::http::HttpRegistrationSource;
    use crate::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
    use crate::pool_discovery::{
        Directory, DiscoveryError, PoolDirectory, PoolRegistrar, PoolRegistration, WatchStream,
    };
    use crate::service::{self, PoolDiscoveryBackend};

    /// A PoolRelay registers every second, and a registration lasts three, so
    /// instances ask each other every 500 ms.
    fn lease() -> Lease {
        Lease::new(1, 3).unwrap()
    }

    /// What a PoolRelay registers.
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

    /// The instances that an instance's list shows, which a test changes.
    #[derive(Default)]
    struct Listed(Mutex<Vec<RouterReplica>>);

    impl Listed {
        /// Instances on a [`Network`], by name.
        fn names(&self, names: &[&str]) {
            self.set(
                names
                    .iter()
                    .map(|name| RouterReplica {
                        replica: RouterReplicaId((*name).to_owned()),
                        url: Url::parse(&format!("http://{name}.test/")).unwrap(),
                    })
                    .collect(),
            );
        }

        fn set(&self, instances: Vec<RouterReplica>) {
            *self.0.lock().unwrap() = instances;
        }
    }

    #[async_trait]
    impl Directory<RouterReplica, RouterReplicaId> for Listed {
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

    /// Instances in one process, reached by name, without HTTP. Each answers
    /// with [`Sharing::reply`], as `GET /registrations` does, read when it is
    /// asked and sent `delay` later. One that is down never answers.
    #[derive(Clone, Default)]
    struct Network(Arc<Mutex<HashMap<RouterReplicaId, Node>>>);

    #[derive(Clone)]
    struct Node {
        sharing: Sharing,
        delay: Duration,
        down: bool,
    }

    #[async_trait]
    impl RegistrationSource for Network {
        async fn shared_registrations(
            &self,
            instance: &RouterReplica,
        ) -> Result<SharedRegistrations, DiscoveryError> {
            let node = self.0.lock().unwrap().get(&instance.replica).cloned();
            match node {
                Some(node) if !node.down => {
                    let reply = node.sharing.reply();
                    tokio::time::sleep(node.delay).await;
                    Ok(reply)
                }
                _ => std::future::pending().await,
            }
        }
    }

    impl Network {
        /// Starts an instance named `name`, under `lease`, whose list shows the
        /// instances named in `listed`. Returns its store, its list, and the
        /// token that stops its sharing task.
        fn start(
            &self,
            name: &str,
            lease: Lease,
            listed: &[&str],
        ) -> (Arc<MemoryPoolStore>, Arc<Listed>, CancellationToken) {
            let store = Arc::new(MemoryPoolStore::new(lease));
            let list = Arc::new(Listed::default());
            list.names(listed);
            let sync = self.join(name, &store, lease, &list);
            let stop = CancellationToken::new();
            tokio::spawn(sync.run(stop.clone()));
            (store, list, stop)
        }

        /// Serves an instance named `name` without starting its sharing task.
        fn join(
            &self,
            name: &str,
            store: &Arc<MemoryPoolStore>,
            lease: Lease,
            list: &Arc<Listed>,
        ) -> PeerSync {
            let (sync, sharing) =
                PeerSync::new(list.clone(), Arc::new(self.clone()), store.clone(), lease).unwrap();
            let node = Node {
                sharing,
                delay: Duration::ZERO,
                down: false,
            };
            self.0
                .lock()
                .unwrap()
                .insert(RouterReplicaId(name.to_owned()), node);
            sync
        }

        /// Makes the instance named `alias` answer as `name` does.
        fn alias(&self, alias: &str, name: &str) {
            let mut nodes = self.0.lock().unwrap();
            let node = nodes[&RouterReplicaId(name.to_owned())].clone();
            nodes.insert(RouterReplicaId(alias.to_owned()), node);
        }

        fn change(&self, name: &str, change: impl FnOnce(&mut Node)) {
            change(
                self.0
                    .lock()
                    .unwrap()
                    .get_mut(&RouterReplicaId(name.to_owned()))
                    .unwrap(),
            );
        }
    }

    /// The models `store` lists, and whether the list is complete.
    async fn view(store: &MemoryPoolStore) -> String {
        let snapshot = store.list().await.unwrap();
        let mut models: Vec<String> = snapshot
            .items
            .into_iter()
            .filter_map(|listing| listing.model)
            .collect();
        models.sort();
        let complete = match snapshot.completeness {
            Completeness::Exact => "complete",
            Completeness::Estimated => "incomplete",
        };
        format!("[{}] {complete}", models.join(","))
    }

    /// Waits until `millis` after `start`, on the paused clock.
    async fn at(start: Instant, millis: u64) {
        tokio::time::sleep_until(start + Duration::from_millis(millis)).await;
    }

    #[tokio::test(start_paused = true)]
    async fn registrations_reach_the_other_instances_and_lapse_with_their_originals() {
        let network = Network::default();
        let start = Instant::now();
        let (a, _, _) = network.start("a", lease(), &["a", "b"]);
        let (b, _, _) = network.start("b", lease(), &["a", "b"]);
        let _ = a.register(pool("m1")).await.unwrap();
        // B copies A's registration, and does not share the copy.
        at(start, 250).await;
        assert_eq!(view(&b).await, "[m1] incomplete");
        assert!(b.local_registrations().is_empty());
        // A changed registration replaces the copy.
        let _ = a.register(pool("m2")).await.unwrap();
        at(start, 750).await;
        assert_eq!(view(&b).await, "[m2] incomplete");
        // A lease after they started, the lists are complete.
        at(start, 3100).await;
        assert_eq!(view(&b).await, "[m2] complete");
        // With no more heartbeats, the registration lapses at A at 3.25 s, and
        // the copy at B with it. Each list stays incomplete until every other
        // instance has been asked after the pool may have lapsed there.
        at(start, 3300).await;
        assert_eq!(view(&a).await, "[] incomplete");
        assert_eq!(view(&b).await, "[] incomplete");
        at(start, 4100).await;
        assert_eq!(view(&a).await, "[] complete");
        assert_eq!(view(&b).await, "[] complete");
    }

    #[tokio::test(start_paused = true)]
    async fn instances_behind_one_service_keep_sharing_a_pool_whose_heartbeats_alternate() {
        // A and B serve one cluster behind one Service, so a PoolRelay's
        // heartbeats reach either of them. C, in another cluster, reads both.
        let network = Network::default();
        let start = Instant::now();
        let all = ["a", "b", "c"];
        let (a, _, _) = network.start("a", lease(), &all);
        let (b, _, _) = network.start("b", lease(), &all);
        let (c, _, _) = network.start("c", lease(), &all);
        for beat in 0..6 {
            let (heard, other) = if beat % 2 == 0 { (&a, &b) } else { (&b, &a) };
            at(start, 1000 * beat + 100).await;
            let _ = heard.register(pool("m1")).await.unwrap();
            // A heartbeat interval later, the instance that took the last
            // heartbeat shares the pool, the other keeps only a copy, and C
            // lists the pool, completely once a lease has passed: no copy
            // lapses.
            at(start, 1000 * beat + 1050).await;
            assert_eq!(heard.local_registrations().len(), 1);
            assert!(other.local_registrations().is_empty());
            let complete = if beat >= 2 { "complete" } else { "incomplete" };
            assert_eq!(view(&c).await, format!("[m1] {complete}"));
        }
    }

    /// A PoolRelay registers its pool at 1.1 s, with A, or with B. Its later
    /// heartbeats fail, and its retry, at 3.95 s, late for the 3 s lease, as
    /// the heartbeat client's retry 1 s before the lease ends would be,
    /// reaches A, which answers 300 ms after it is asked from 3.8 s on. So the
    /// registration lapses at B, at 4.1 s, before the reply with the renewal
    /// arrives, at 4.3 s.
    async fn a_late_renewal_keeps_lists_incomplete(registered_with_b: bool) {
        let network = Network::default();
        let start = Instant::now();
        let (a, _, _) = network.start("a", lease(), &["a", "b"]);
        let (b, _, _) = network.start("b", lease(), &["a", "b"]);
        at(start, 1100).await;
        let first = if registered_with_b { &b } else { &a };
        let _ = first.register(pool("m1")).await.unwrap();
        at(start, 3500).await;
        assert_eq!(view(&b).await, "[m1] complete");
        at(start, 3800).await;
        network.change("a", |node| node.delay = Duration::from_millis(300));
        at(start, 3950).await;
        let _ = a.register(pool("m1")).await.unwrap();
        // The pool is live all along, but B lacks it until the renewal
        // arrives, so B must not say that its list is complete: a Global
        // Router would drop the pool.
        at(start, 4150).await;
        assert_eq!(view(&b).await, "[] incomplete");
        at(start, 5400).await;
        assert_eq!(view(&b).await, "[m1] complete");
        // With no more heartbeats, B's copy lapses with A's registration, at
        // 6.95 s: the time A took to answer does not lengthen it.
        at(start, 7000).await;
        assert_eq!(view(&b).await, "[] incomplete");
    }

    #[tokio::test(start_paused = true)]
    async fn a_late_renewal_keeps_lists_incomplete_while_a_copy_has_lapsed() {
        a_late_renewal_keeps_lists_incomplete(false).await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_late_renewal_with_another_instance_keeps_lists_incomplete() {
        a_late_renewal_keeps_lists_incomplete(true).await;
    }

    #[tokio::test(start_paused = true)]
    async fn an_instance_that_restarts_or_leaves_keeps_lists_incomplete_for_a_lease() {
        let network = Network::default();
        let start = Instant::now();
        let (_, _, a_stop) = network.start("a", lease(), &["a", "b"]);
        let (b, b_list, _) = network.start("b", lease(), &["a", "b"]);
        at(start, 3100).await;
        assert_eq!(view(&b).await, "[] complete");
        // A restarts at 4.1 s under the same name. Its answers lack the
        // registrations its PoolRelays have not renewed yet until a lease
        // after that, so B's lists are incomplete from when A's last complete
        // answer, at 4 s, no longer vouches for them, until 7.5 s.
        at(start, 4100).await;
        a_stop.cancel();
        let _ = network.start("a", lease(), &["a", "b"]);
        at(start, 5100).await;
        assert_eq!(view(&b).await, "[] incomplete");
        at(start, 7400).await;
        assert_eq!(view(&b).await, "[] incomplete");
        at(start, 7600).await;
        assert_eq!(view(&b).await, "[] complete");
        // A leaves the list: B's lists are incomplete from the next round, at
        // 8 s, for a lease, until copies from A have lapsed.
        b_list.names(&["b"]);
        at(start, 8100).await;
        assert_eq!(view(&b).await, "[] incomplete");
        at(start, 10900).await;
        assert_eq!(view(&b).await, "[] incomplete");
        at(start, 11100).await;
        assert_eq!(view(&b).await, "[] complete");
    }

    #[tokio::test(start_paused = true)]
    async fn an_instance_that_does_not_answer_completely_keeps_lists_incomplete() {
        let network = Network::default();
        let start = Instant::now();
        let _ = network.start("a", lease(), &["a", "b"]);
        let (b, b_list, _) = network.start("b", lease(), &["a", "b"]);
        at(start, 3100).await;
        assert_eq!(view(&b).await, "[] complete");
        // A stops answering. Its last complete answer, asked for at 3 s,
        // vouches for B's lists until 4 s.
        network.change("a", |node| node.down = true);
        at(start, 4100).await;
        assert_eq!(view(&b).await, "[] incomplete");
        network.change("a", |node| node.down = false);
        at(start, 5100).await;
        assert_eq!(view(&b).await, "[] complete");
        // C uses another lease. B copies its registrations, but its answers
        // do not count as complete, even once C has heard from all of its
        // PoolRelays, a lease of 4 s after it started.
        let (c, _, _) = network.start("c", Lease::new(1, 4).unwrap(), &["c"]);
        b_list.names(&["a", "b", "c"]);
        at(start, 8000).await;
        let _ = c.register(pool("m1")).await.unwrap();
        at(start, 9600).await;
        assert_eq!(view(&b).await, "[m1] incomplete");
    }

    #[tokio::test(start_paused = true)]
    async fn an_instance_finds_itself_in_the_list_and_asks_only_the_others() {
        let network = Network::default();
        let start = Instant::now();
        let store = Arc::new(MemoryPoolStore::new(lease()));
        let list = Arc::new(Listed::default());
        // B is listed twice, as one listed by address may be, under IPv4 and
        // IPv6, and nothing but its replies says that both are B.
        list.names(&["b", "b-ipv6"]);
        let sync = network.join("b", &store, lease(), &list);
        network.alias("b-ipv6", "b");
        // Before its first round, B's lists are incomplete, even a lease after
        // it started.
        at(start, 3000).await;
        assert_eq!(view(&store).await, "[] incomplete");
        // Its first rounds cannot reach it at its listed addresses, so it
        // cannot tell that they are its own yet.
        for name in ["b", "b-ipv6"] {
            network.change(name, |node| node.down = true);
        }
        let stop = CancellationToken::new();
        tokio::spawn(sync.run(stop.clone()));
        at(start, 3600).await;
        assert_eq!(view(&store).await, "[] incomplete");
        for name in ["b", "b-ipv6"] {
            network.change(name, |node| node.down = false);
        }
        // The round at 4 s shows that both names are B, so its lists are
        // complete: it neither waits for itself nor counts itself as an
        // instance that left.
        at(start, 4100).await;
        assert_eq!(view(&store).await, "[] complete");
        // From then on it asks only the others, so its own addresses failing
        // changes nothing.
        for name in ["b", "b-ipv6"] {
            network.change(name, |node| node.down = true);
        }
        at(start, 5600).await;
        assert_eq!(view(&store).await, "[] complete");
        // Not listed, as when its Service is not imported here, B cannot tell
        // what the others hold.
        list.names(&[]);
        at(start, 6100).await;
        assert_eq!(view(&store).await, "[] incomplete");
        list.names(&["b"]);
        at(start, 6600).await;
        assert_eq!(view(&store).await, "[] complete");
        // If the task stops, its last round, at 6.5 s, vouches for the lists
        // only until 7.5 s.
        stop.cancel();
        at(start, 7600).await;
        assert_eq!(view(&store).await, "[] incomplete");
    }

    #[tokio::test]
    async fn a_lease_under_three_heartbeat_intervals_is_refused() {
        let lease = Lease::new(1, 2).unwrap();
        let error = PeerSync::new(
            Arc::new(StaticReplicaDirectory::new(Vec::new())),
            Arc::new(Network::default()),
            Arc::new(MemoryPoolStore::new(lease)),
            lease,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("at least 3 heartbeat intervals"),
            "{error}"
        );
    }

    #[test]
    fn a_round_vouches_until_a_heartbeat_before_the_oldest_answer_may_lapse() {
        // A 5 s heartbeat and a 13 s lease. Copies from an instance's last
        // complete answer, asked for at `then`, may lapse 8 s later, so the
        // round vouches for 3 s after it, whenever it ends.
        let lease = Lease::new(5, 13).unwrap();
        let url = Url::parse("http://10.43.0.9:8080/").unwrap();
        let listed = |completeness, names: &[&str]| Snapshot {
            items: names
                .iter()
                .map(|name| RouterReplica {
                    replica: RouterReplicaId((*name).to_owned()),
                    url: url.clone(),
                })
                .collect(),
            completeness,
        };
        let reply = |complete| SharedRegistrations {
            instance: "a".to_owned(),
            lease,
            complete,
            registrations: Vec::new(),
        };
        let a = listed(Completeness::Exact, &["a"])
            .items
            .into_iter()
            .next()
            .unwrap();
        let then = Instant::now();
        let mut instances = Instances::default();
        instances.own(&RouterReplicaId("b".to_owned()));
        instances.answered(&a, &reply(true), then, lease);
        for ended in [2900, 5400] {
            let round = instances.end_round(
                &listed(Completeness::Exact, &["a", "b"]),
                missing_after(lease),
                lease,
                then + Duration::from_millis(ended),
            );
            assert_eq!(round.answered_since, Some(then));
            assert_eq!(round.complete_until, Some(then + Duration::from_secs(3)));
        }
        // A list that may be incomplete vouches for nothing, and neither does
        // an answer without all of the instance's registrations.
        let round = instances.end_round(
            &listed(Completeness::Estimated, &["a", "b"]),
            missing_after(lease),
            lease,
            then,
        );
        assert!(round.complete_until.is_none());
        let mut restarted = Instances::default();
        restarted.own(&RouterReplicaId("b".to_owned()));
        restarted.answered(&a, &reply(false), then, lease);
        let round = restarted.end_round(
            &listed(Completeness::Exact, &["a", "b"]),
            missing_after(lease),
            lease,
            then,
        );
        assert!(round.complete_until.is_none());
    }

    #[test]
    fn an_instance_that_finds_itself_late_does_not_count_itself_as_one_that_left() {
        let lease = lease();
        let me = RouterReplica {
            replica: RouterReplicaId("b".to_owned()),
            url: Url::parse("http://10.43.0.9:8080/").unwrap(),
        };
        let listed = |items: Vec<RouterReplica>| Snapshot {
            items,
            completeness: Completeness::Exact,
        };
        let error = anyhow::anyhow!("no answer");
        let now = Instant::now();
        // B's first asks to itself fail, so it takes itself for another
        // instance, which then leaves the list, or stays listed, before B's
        // own reply shows that it is B.
        for leaves in [false, true] {
            let mut instances = Instances::default();
            instances.unreachable(&me, &error, lease, now);
            let _ =
                instances.end_round(&listed(vec![me.clone()]), missing_after(lease), lease, now);
            if leaves {
                let _ = instances.end_round(&listed(Vec::new()), missing_after(lease), lease, now);
            }
            instances.own(&me.replica);
            let round =
                instances.end_round(&listed(vec![me.clone()]), missing_after(lease), lease, now);
            assert_eq!(
                round.complete_until,
                Some(now + missing_after(lease)),
                "{leaves}"
            );
        }
    }

    /// Serves an instance over HTTP that reads the others over HTTP. Returns
    /// its store, its URL, its sharing task, not yet started, and its list.
    async fn serve(stop: &CancellationToken) -> (Arc<MemoryPoolStore>, Url, PeerSync, Arc<Listed>) {
        let store = Arc::new(MemoryPoolStore::new(lease()));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let list = Arc::new(Listed::default());
        let (sync, sharing) = PeerSync::new(
            list.clone(),
            Arc::new(HttpRegistrationSource::new().unwrap()),
            store.clone(),
            lease(),
        )
        .unwrap();
        let pools: Arc<PoolDirectory> = store.clone();
        let registrar: Arc<dyn PoolRegistrar> = store.clone();
        tokio::spawn(service::serve(
            listener,
            PoolDiscoveryBackend {
                pools,
                routers: Arc::new(StaticReplicaDirectory::new(Vec::new())),
                registrar: Some(registrar),
                sharing: Some(Arc::new(sharing)),
            },
            None,
            stop.clone(),
        ));
        (store, url, sync, list)
    }

    #[tokio::test]
    async fn instances_share_registrations_over_http() {
        let stop = CancellationToken::new();
        let replica = |name: &str, url: &Url| RouterReplica {
            replica: RouterReplicaId(name.to_owned()),
            url: url.clone(),
        };
        let healthz = |url: &Url| {
            let url = url.join("healthz").unwrap();
            async move { reqwest::get(url).await.unwrap().status() }
        };
        let (a, a_url, _, _) = serve(&stop).await;
        let (b, b_url, b_sync, b_list) = serve(&stop).await;
        let _ = a.register(pool("m1")).await.unwrap();
        // B reads A, itself, and an address where nothing listens.
        let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gone = Url::parse(&format!("http://{}/", closed.local_addr().unwrap())).unwrap();
        drop(closed);
        b_list.set(vec![
            replica("a", &a_url),
            replica("b", &b_url),
            replica("gone", &gone),
        ]);
        // B is not ready until it has asked the others once, and by then it
        // has a copy of A's registration.
        assert_eq!(
            healthz(&b_url).await,
            reqwest::StatusCode::SERVICE_UNAVAILABLE
        );
        tokio::spawn(b_sync.run(stop.clone()));
        let deadline = Instant::now() + Duration::from_secs(5);
        while healthz(&b_url).await != reqwest::StatusCode::OK {
            assert!(Instant::now() < deadline, "B is not ready");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(view(&b).await, "[m1] incomplete");
        stop.cancel();
    }
}
