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
//! a trusted network. Each request names the instance that asks and says
//! whether registrations were made with it, and each reply names the
//! instances that recently asked while holding some. So an instance whose
//! list lags behind, as during an outage of the MCS implementation, learns
//! of an unlisted instance holding registrations when that one asks it, or
//! asks an instance that it reads. Replies do not pass on what they were
//! told, so one further away stays unknown to it.
//!
//! Lists are `Estimated` while registrations made with another instance may
//! be missing here: until the first round ends; while this instance does not
//! find itself in the list of instances, or the list may be incomplete; for
//! a lease after an instance leaves the list; once a listed instance has not
//! answered completely for the lease minus two heartbeat intervals, a
//! heartbeat interval before the copies from its last complete answer may
//! lapse; while an instance holding registrations made with it, that has
//! not answered completely within that time, asks this one or an instance
//! that it reads, and for about a lease and a heartbeat interval after it
//! last did, since it may hold registrations that this one cannot read; and
//! from when a registration lapses here, a copy or one made here, until
//! every instance has answered completely again, since its PoolRelay may
//! have renewed it late, with any of them, as it does after a failed
//! heartbeat. A lapse waits one round more, so that an instance that no one
//! lists and that took the registration over has said so, and a copy's
//! another, since it can lapse up to one request's time before its original.
//! The store keeps these times, so lists turn `Estimated` on time
//! even if the rounds stop. An instance that stays listed but cannot be
//! reached keeps lists `Estimated` until it answers: the pools its PoolRelays
//! registered may be missing here, and a complete list would let a Global
//! Router drop them. Once it has not answered for a lease, that is logged as
//! an error, and so is an instance that asks without answering for a lease,
//! and one that has not found itself for two leases.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use futures::future::join_all;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::pool_discovery::{
    Asker, Completeness, DiscoveryError, ExchangeRound, Lease, RecentAsker, RegistrationExchange,
    RegistrationSharing, RegistrationSource, ReplicaDirectory, RouterReplica, RouterReplicaId,
    SharedRegistration, SharedRegistrations, Snapshot, answer_timeout,
};

/// The shortest lease that sharing allows, in heartbeat intervals: a copy
/// must outlive a round that times out.
pub const MIN_LEASE_HEARTBEATS: u32 = 3;

/// What this instance serves the others, and whether it has asked them yet.
#[derive(Clone)]
pub struct Sharing {
    /// Names this process in its replies, so it can tell its own.
    instance: Uuid,
    /// The registrations' lease, which every instance must share.
    lease: Lease,
    registrations: Arc<dyn RegistrationExchange>,
    synced: Arc<AtomicBool>,
    /// The other instances that asked this one while holding registrations,
    /// by the names their replies carry, and when each last did.
    askers: Arc<Mutex<HashMap<Uuid, Instant>>>,
}

/// How long an asker that holds registrations counts after it last asked: a
/// registration renewed with it up to half a heartbeat interval after its
/// last ask lapses within a lease, and its PoolRelay's registration with
/// another instance reaches this one within another half.
fn asker_kept(lease: Lease) -> Duration {
    lease.duration() + lease.heartbeat_interval()
}

impl Sharing {
    /// The instances that asked this one while holding registrations, within
    /// [`asker_kept`] of `now`, and when each last did. Forgets the others.
    fn askers(&self, lease: Lease, now: Instant) -> Vec<(Uuid, Instant)> {
        let kept = asker_kept(lease);
        let mut askers = self.askers.lock().unwrap_or_else(PoisonError::into_inner);
        askers.retain(|_, asked| now.saturating_duration_since(*asked) < kept);
        askers
            .iter()
            .map(|(asker, asked)| (*asker, *asked))
            .collect()
    }
}

impl RegistrationSharing for Sharing {
    /// Also reports every instance that asked within [`asker_kept`] while
    /// holding registrations, whether or not this one hears from it: a
    /// reader drops those it hears from itself.
    fn reply(&self) -> SharedRegistrations {
        let now = Instant::now();
        let kept = asker_kept(self.lease);
        let askers = self
            .askers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|(asker, asked)| (*asker, now.saturating_duration_since(*asked)))
            .filter(|(_, ago)| *ago < kept)
            .map(|(asker, ago)| RecentAsker::new(asker, ago))
            .collect();
        SharedRegistrations {
            instance: self.instance,
            lease: self.lease,
            complete: self.registrations.holds_every_local_registration(),
            registrations: self
                .registrations
                .local_registrations()
                .into_iter()
                .map(|(registration, remaining)| SharedRegistration::new(registration, remaining))
                .collect(),
            askers,
        }
    }

    /// An asker that holds no registration holds nothing that this instance
    /// may lack, so only one that holds some is noted.
    fn asked_by(&self, asker: Asker) {
        if asker.holding && asker.instance != self.instance {
            self.askers
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(asker.instance, Instant::now());
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
    /// [`answer_timeout`].
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
            instance: Uuid::new_v4(),
            lease,
            registrations: store,
            synced: Arc::default(),
            askers: Arc::default(),
        };
        let sync = Self {
            instances,
            source,
            sharing: sharing.clone(),
            every: answer_timeout(lease),
        };
        Ok((sync, sharing))
    }

    /// Runs until `stop` is cancelled.
    pub async fn run(self, stop: CancellationToken) {
        tracing::info!(instance = %self.sharing.instance, "sharing registrations with the other discovery instances");
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
            let asker = Asker {
                instance: self.sharing.instance,
                holding: self.sharing.registrations.holds_local_registration(),
            };
            let replies = tokio::select! {
                biased;
                () = stop.cancelled() => return,
                replies = join_all(others.iter().map(|peer| self.ask(peer, asker))) => replies,
            };
            // After every reply arrived, so a time that a reply gives as an
            // age is taken no earlier than it was.
            let arrived = Instant::now();
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
                        instances.note_reports(&reply, self.sharing.instance, arrived);
                        self.absorb(reply, sent);
                    }
                }
            }
            let now = Instant::now();
            let askers = self.sharing.askers(lease, now);
            let round = instances.end_round(&listed, &askers, missing_after, lease, now);
            self.sharing.registrations.end_round(round);
            if reached != Some(answered) {
                tracing::info!(reached = answered, "other discovery instances reached");
                reached = Some(answered);
            }
            self.sharing.synced.store(true, Ordering::Release);
        }
    }

    /// Asks `peer` as `asker`, waiting at most `every`. Also returns when it
    /// was asked.
    async fn ask(
        &self,
        peer: &RouterReplica,
        asker: Asker,
    ) -> (Instant, anyhow::Result<SharedRegistrations>) {
        let sent = Instant::now();
        let ask = self.source.shared_registrations(peer, asker);
        let reply = tokio::time::timeout(self.every, ask).await;
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
    /// The instances whose answers counted, by the names their replies
    /// carry, and when each was last asked for one.
    heard: HashMap<Uuid, Instant>,
    /// Askers holding registrations that other instances report, and when
    /// each last asked them, as near as this one can tell. Never reported
    /// on, so a report does not echo between instances.
    reported: HashMap<Uuid, Instant>,
    /// Askers holding registrations that this one has not heard from, and
    /// since when.
    unheard_askers: HashMap<Uuid, Problem>,
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

/// A problem that is logged once it has lasted long enough.
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
    /// `after`.
    fn lasted(&mut self, after: Duration, now: Instant) -> bool {
        let lasted = !self.logged && now.saturating_duration_since(self.since) >= after;
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
        if problem.lasted(lease.duration(), now) {
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
            self.heard.insert(reply.instance, sent);
        }
    }

    /// Ends a round over the `listed` instances at `now`, after the
    /// instances named `askers` asked this one. The round vouches for nothing
    /// while this instance does not find itself in the list, while the list
    /// may be incomplete, while another listed instance has never answered
    /// completely, for a lease after one leaves, and while one asks that has
    /// not answered completely within `missing_after`. Otherwise it gives the
    /// oldest of the other instances' last complete answers, or `now` with
    /// none, and vouches for `missing_after` past it.
    fn end_round(
        &mut self,
        listed: &Snapshot<RouterReplica>,
        askers: &[(Uuid, Instant)],
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
        let unheard_askers = self.note_askers(askers, missing_after, lease, now);
        if !found_self
            || listed.completeness != Completeness::Exact
            || !self.departed.is_empty()
            || unheard_askers
        {
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

    /// Whether any asker holding registrations, of this instance or
    /// reported by one that answered it, has not answered this one completely
    /// within `missing_after` of `now`: as when this instance's list lags
    /// behind, it may hold registrations that this one cannot read. Logs each
    /// such asker once it has lasted a lease, and forgets answers and reports
    /// too old to matter.
    fn note_askers(
        &mut self,
        askers: &[(Uuid, Instant)],
        missing_after: Duration,
        lease: Lease,
        now: Instant,
    ) -> bool {
        self.heard
            .retain(|_, sent| now.saturating_duration_since(*sent) < lease.duration());
        let kept = asker_kept(lease);
        self.reported
            .retain(|_, asked| now.saturating_duration_since(*asked) < kept);
        let unheard: HashSet<Uuid> = askers
            .iter()
            .map(|(asker, _)| *asker)
            .chain(self.reported.keys().copied())
            .filter(|asker| !self.hears(asker, missing_after, now))
            .collect();
        self.unheard_askers.retain(|asker, problem| {
            let still = unheard.contains(asker);
            if !still && problem.logged {
                tracing::info!(
                    asker = %asker,
                    "a discovery instance that held registrations this one could not read is \
                     heard from or gone"
                );
            }
            still
        });
        for asker in &unheard {
            let problem = self
                .unheard_askers
                .entry(*asker)
                .or_insert_with(|| Problem::new(now));
            if problem.lasted(lease.duration(), now) {
                tracing::error!(
                    asker = %asker,
                    "a discovery instance that holds registrations asks this one, or one that \
                     this one reads, but this one has no complete answer from it, so pool lists \
                     stay incomplete: as when an import of the instances lags behind, it may hold \
                     registrations that this one cannot read"
                );
            }
        }
        !unheard.is_empty()
    }

    /// Whether `instance` answered this one completely within
    /// `missing_after` of `now`.
    fn hears(&self, instance: &Uuid, missing_after: Duration, now: Instant) -> bool {
        self.heard
            .get(instance)
            .is_some_and(|sent| now < *sent + missing_after)
    }

    /// Notes the askers that `reply` reports, other than this instance,
    /// `own`. Each last asked when the reply says, counted back from
    /// `arrived`, which is after the reply arrived, so it is kept no less
    /// long than the instance that reports it keeps it.
    fn note_reports(&mut self, reply: &SharedRegistrations, own: Uuid, arrived: Instant) {
        for recent in &reply.askers {
            let (asker, ago) = recent.into_parts();
            let Some(asked) = arrived.checked_sub(ago) else {
                continue;
            };
            if asker != own {
                let last = self.reported.entry(asker).or_insert(asked);
                *last = (*last).max(asked);
            }
        }
    }

    /// Logs, once two leases have passed, that this instance has not found
    /// itself in the list of instances, and when it does again. Its own
    /// endpoint reaches the list only after it is ready and MCS has copied it
    /// back, which can take longer than a lease.
    fn note_found(&mut self, found_self: bool, lease: Lease, now: Instant) {
        if found_self {
            if self.unfound.take().is_some_and(|problem| problem.logged) {
                tracing::info!("this discovery instance is listed again");
            }
        } else if self
            .unfound
            .get_or_insert_with(|| Problem::new(now))
            .lasted(lease.duration() * 2, now)
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

    use futures::StreamExt;

    use super::*;
    use crate::backends::http::HttpRegistrationSource;
    use crate::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
    use crate::backends::registered::RegisteredPoolDirectory;
    use crate::pool_discovery::{
        Directory, DiscoveryError, PoolDirectory, PoolListing, PoolRegistrar, PoolRegistration,
        WatchEvent, WatchStream,
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
            asker: Asker,
        ) -> Result<SharedRegistrations, DiscoveryError> {
            let node = self.0.lock().unwrap().get(&instance.replica).cloned();
            match node {
                Some(node) if !node.down => {
                    node.sharing.asked_by(asker);
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
        // instance has been asked after the pool may have lapsed there, and an
        // instance that no one lists has had a round to say that it took the
        // pool over: at A, until 4 s, and at B, where a copy may lapse half a
        // heartbeat interval before its original, until 4.5 s.
        at(start, 3300).await;
        assert_eq!(view(&a).await, "[] incomplete");
        assert_eq!(view(&b).await, "[] incomplete");
        at(start, 4100).await;
        assert_eq!(view(&a).await, "[] complete");
        assert_eq!(view(&b).await, "[] incomplete");
        at(start, 4600).await;
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
        // answer, at 4 s, no longer vouches for them, until 7.5 s. The A that
        // stopped held no registrations, so none can be missing because of it.
        at(start, 4100).await;
        a_stop.cancel();
        let (_, _, a_stop) = network.start("a", lease(), &["a", "b"]);
        at(start, 5100).await;
        assert_eq!(view(&b).await, "[] incomplete");
        at(start, 7400).await;
        assert_eq!(view(&b).await, "[] incomplete");
        at(start, 7600).await;
        assert_eq!(view(&b).await, "[] complete");
        // A stops and leaves the list: B's lists are incomplete from the next
        // round, at 8 s, for a lease, until copies from A have lapsed.
        a_stop.cancel();
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
    async fn an_instance_asked_by_one_it_does_not_list_keeps_lists_incomplete() {
        // A and A2 serve one cluster, whose list shows both, but B's list lags
        // behind and does not show A2 yet. A pool registers with A, then with
        // A2 from 1.1 s on, so A keeps only a copy, which it does not share,
        // and B's copy lapses at 3.1 s. A answers completely at 4 s, without
        // the pool, but A2 asks B, so B knows that an instance it cannot read
        // may hold registrations it lacks.
        let network = Network::default();
        let start = Instant::now();
        let cluster = ["a", "a2", "b"];
        let (a, _, _) = network.start("a", lease(), &cluster);
        let (a2, _, _) = network.start("a2", lease(), &cluster);
        let (b, b_list, _) = network.start("b", lease(), &["a", "b"]);
        at(start, 100).await;
        let _ = a.register(pool("m1")).await.unwrap();
        for beat in 1..5 {
            at(start, 1000 * beat + 100).await;
            let _ = a2.register(pool("m1")).await.unwrap();
        }
        at(start, 4200).await;
        assert_eq!(view(&b).await, "[] incomplete");
        // Once B's list shows A2, B copies the pool from it, and its lists are
        // complete again.
        b_list.names(&cluster);
        at(start, 5200).await;
        assert_eq!(view(&b).await, "[m1] complete");
    }

    #[tokio::test(start_paused = true)]
    async fn an_instance_that_stopped_asking_keeps_lists_incomplete_until_its_pools_can_return() {
        // B's list lags and does not show A2, which asks B. A pool registers
        // with A2 until A2 stops, after its last ask, at 4 s, and its last
        // renewal, at 4.1 s. The pool's PoolRelay registers late with A, at
        // 7.05 s, before its lease ends at 7.1 s, and B copies it at 7.5 s.
        // Until then, A2's silence must not let B's lists be complete.
        let network = Network::default();
        let start = Instant::now();
        let (a, _, _) = network.start("a", lease(), &["a", "a2", "b"]);
        let (a2, _, a2_stop) = network.start("a2", lease(), &["a2", "b"]);
        let (b, _, _) = network.start("b", lease(), &["a", "b"]);
        for beat in 0..5 {
            at(start, 1000 * beat + 100).await;
            let _ = a2.register(pool("m1")).await.unwrap();
        }
        at(start, 4200).await;
        a2_stop.cancel();
        network.change("a2", |node| node.down = true);
        at(start, 7050).await;
        let _ = a.register(pool("m1")).await.unwrap();
        at(start, 7200).await;
        assert_eq!(view(&b).await, "[] incomplete");
        at(start, 8200).await;
        assert_eq!(view(&b).await, "[m1] complete");
    }

    #[tokio::test(start_paused = true)]
    async fn an_instance_learns_of_one_that_asks_an_instance_it_reads() {
        // A's cluster loses contact with MCS: its list freezes, showing A and
        // an instance that has gone since. A2, started there since, is listed
        // nowhere, and does not list B, started elsewhere since. A pool moves
        // from A to A2, so only A hears from A2, and B learns of A2 from A's
        // answers.
        let network = Network::default();
        let start = Instant::now();
        let frozen = ["a", "gone"];
        let (a, a_list, _) = network.start("a", lease(), &frozen);
        let (a2, a2_list, _) = network.start("a2", lease(), &frozen);
        let (b, b_list, _) = network.start("b", lease(), &["a", "b"]);
        at(start, 100).await;
        let _ = a.register(pool("m1")).await.unwrap();
        for beat in 1..5 {
            at(start, 1000 * beat + 100).await;
            let _ = a2.register(pool("m1")).await.unwrap();
        }
        at(start, 4700).await;
        assert_eq!(view(&b).await, "[] incomplete");
        // A's cluster recovers first: A reads A2, but B still cannot.
        for list in [&a_list, &a2_list] {
            list.names(&["a", "a2"]);
        }
        for beat in 5..10 {
            at(start, 1000 * beat + 100).await;
            let _ = a2.register(pool("m1")).await.unwrap();
        }
        at(start, 9200).await;
        assert_eq!(view(&b).await, "[] incomplete");
        // Once B reads A2 too, it copies the pool from it.
        for list in [&a_list, &a2_list, &b_list] {
            list.names(&["a", "a2", "b"]);
        }
        at(start, 10200).await;
        assert_eq!(view(&b).await, "[m1] complete");
    }

    #[tokio::test(start_paused = true)]
    async fn an_instance_that_holds_no_registrations_keeps_no_list_incomplete() {
        // C asks B, which does not list it, as a test instance given the same
        // Service might. While no PoolRelay registers with C, nothing can be
        // missing because of it.
        let network = Network::default();
        let start = Instant::now();
        let _ = network.start("a", lease(), &["a", "b"]);
        let (b, _, _) = network.start("b", lease(), &["a", "b"]);
        let (c, _, _) = network.start("c", lease(), &["b", "c"]);
        at(start, 3600).await;
        assert_eq!(view(&b).await, "[] complete");
        // Once one does, B cannot read that registration.
        let _ = c.register(pool("m1")).await.unwrap();
        at(start, 4600).await;
        assert_eq!(view(&b).await, "[] incomplete");
        // The registration lapses at 6.6 s, after C last asked while holding
        // it, at 6.5 s. C's later asks, holding nothing, neither renew that
        // ask nor undo it: it counts until a lease and a heartbeat interval
        // after it.
        at(start, 10400).await;
        assert_eq!(view(&b).await, "[] incomplete");
        at(start, 10600).await;
        assert_eq!(view(&b).await, "[] complete");
    }

    #[tokio::test(start_paused = true)]
    async fn a_pool_taken_over_by_an_unlisted_instance_stays_listed_or_lists_incomplete() {
        // A pool registers with B until 2.1 s, so its registration lapses at
        // 5.1 s. Its PoolRelay renews just before that, at 5.07 s, with C,
        // which B does not list and which held no registration until then, so
        // C says that it holds one only when it next asks, at 5.55 s. B's
        // lists must not be complete in between.
        let network = Network::default();
        let start = Instant::now();
        let _ = network.start("a", lease(), &["a", "b"]);
        let (b, _, _) = network.start("b", lease(), &["a", "b"]);
        at(start, 50).await;
        let (c, _, _) = network.start("c", lease(), &["b", "c"]);
        for beat in 0..3 {
            at(start, 1000 * beat + 100).await;
            let _ = b.register(pool("m1")).await.unwrap();
        }
        at(start, 5000).await;
        assert_eq!(view(&b).await, "[m1] complete");
        at(start, 5070).await;
        let _ = c.register(pool("m1")).await.unwrap();
        at(start, 5520).await;
        assert_eq!(view(&b).await, "[] incomplete");
        at(start, 6100).await;
        assert_eq!(view(&b).await, "[] incomplete");
    }

    /// MCS importing one Ready PoolRelay endpoint, for good.
    struct ImportedPool(PoolListing);

    #[async_trait]
    impl Directory<PoolListing, PoolKey> for ImportedPool {
        async fn list(&self) -> Result<Snapshot<PoolListing>, DiscoveryError> {
            Ok(Snapshot {
                items: vec![self.0.clone()],
                completeness: Completeness::Exact,
            })
        }

        async fn list_and_watch(
            &self,
            _cancel_token: Option<CancellationToken>,
        ) -> Result<WatchStream<PoolListing, PoolKey>, DiscoveryError> {
            let initial = [
                Ok(WatchEvent::Upserted(self.0.clone())),
                Ok(WatchEvent::Resync(self.list().await?)),
            ];
            Ok(Box::pin(
                futures::stream::iter(initial).chain(futures::stream::pending()),
            ))
        }
    }

    /// What a `GET /pools` view shows of one pool.
    #[derive(Default)]
    struct Shown {
        listed: bool,
        complete: bool,
        /// The view was complete with the pool.
        complete_with_it: bool,
        /// The pool dropped out of the view.
        dropped: bool,
    }

    impl Shown {
        /// Follows `watch` until `until`. After every event, the pool must not
        /// have dropped out of a view that is still complete, as it may have
        /// been renewed late, and while it is `live`, a complete view must
        /// list it.
        async fn follow(
            &mut self,
            watch: &mut WatchStream<PoolListing, PoolKey>,
            key: &PoolKey,
            until: Instant,
            live: bool,
        ) {
            while let Ok(event) = tokio::time::timeout_at(until, watch.next()).await {
                let was_listed = self.listed;
                match event.expect("the watch goes on").unwrap() {
                    WatchEvent::Upserted(listing) if &listing.key == key => self.listed = true,
                    WatchEvent::Removed(removed) if &removed == key => self.listed = false,
                    WatchEvent::Resync(snapshot) => {
                        self.listed = snapshot.items.iter().any(|listing| &listing.key == key);
                        self.complete = snapshot.completeness == Completeness::Exact;
                    }
                    _ => {}
                }
                let dropped = was_listed && !self.listed;
                assert!(!dropped || !self.complete, "dropped out of a complete view");
                assert!(
                    !live || self.listed || !self.complete,
                    "complete without a live pool"
                );
                self.dropped |= dropped;
                self.complete_with_it |= self.listed && self.complete;
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn get_pools_through_mcs_is_incomplete_whenever_a_pool_drops_out() {
        // MCS imports the pool's PoolRelay, which registers with A. B serves
        // `GET /pools` from the join of MCS with its copies.
        let network = Network::default();
        let start = Instant::now();
        let (a, _, _) = network.start("a", lease(), &["a", "b"]);
        let (b, _, _) = network.start("b", lease(), &["a", "b"]);
        let key = pool("m1").key;
        let imported = PoolListing {
            key: key.clone(),
            location: pool("m1").location,
            model: None,
            capabilities: Vec::new(),
            frontends: Vec::new(),
            capacity: Vec::new(),
            relay_identities: vec!["ip:10.43.0.1".to_owned()],
        };
        let join = RegisteredPoolDirectory::new(Arc::new(ImportedPool(imported)), b, lease());
        let mut watch = join.list_and_watch(None).await.unwrap();
        let relay = tokio::spawn(async move {
            loop {
                let _ = a.register(pool("m1")).await.unwrap();
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
        let mut shown = Shown::default();
        let until = |secs| start + Duration::from_secs(secs);
        shown.follow(&mut watch, &key, until(4), true).await;
        assert!(shown.complete_with_it);
        // While A cannot be reached, B's copy lapses, though the pool is live.
        network.change("a", |node| node.down = true);
        shown.follow(&mut watch, &key, until(8), true).await;
        assert!(shown.dropped);
        network.change("a", |node| node.down = false);
        shown.follow(&mut watch, &key, until(10), true).await;
        assert!(shown.listed && shown.complete);
        // The PoolRelay stops. The copy lapses with its registration, in the
        // sweep that makes B's view incomplete, and the view is complete again
        // once A has been asked after that.
        relay.abort();
        shown.dropped = false;
        shown.follow(&mut watch, &key, until(16), false).await;
        assert!(shown.dropped);
        assert!(!shown.listed && shown.complete);
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
        let a_id = Uuid::from_u128(1);
        let reply = |complete| SharedRegistrations {
            instance: a_id,
            lease,
            complete,
            registrations: Vec::new(),
            askers: Vec::new(),
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
                &[],
                missing_after(lease),
                lease,
                then + Duration::from_millis(ended),
            );
            assert_eq!(round.answered_since, Some(then));
            assert_eq!(round.complete_until, Some(then + Duration::from_secs(3)));
        }
        // A asking changes nothing while its answer vouches, but an instance
        // that asks without having answered may hold registrations that B
        // cannot read, and so may A once its answer no longer vouches.
        let asking = |askers: &[Uuid], ended: u64, instances: &mut Instances| {
            let askers: Vec<(Uuid, Instant)> = askers.iter().map(|asker| (*asker, then)).collect();
            instances
                .end_round(
                    &listed(Completeness::Exact, &["a", "b"]),
                    &askers,
                    missing_after(lease),
                    lease,
                    then + Duration::from_millis(ended),
                )
                .complete_until
        };
        assert!(asking(&[a_id], 2900, &mut instances).is_some());
        assert!(asking(&[a_id, Uuid::from_u128(2)], 2900, &mut instances).is_none());
        assert!(asking(&[a_id], 3100, &mut instances).is_none());
        // A list that may be incomplete vouches for nothing, and neither does
        // an answer without all of the instance's registrations.
        let round = instances.end_round(
            &listed(Completeness::Estimated, &["a", "b"]),
            &[],
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
            &[],
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
            let _ = instances.end_round(
                &listed(vec![me.clone()]),
                &[],
                missing_after(lease),
                lease,
                now,
            );
            if leaves {
                let _ =
                    instances.end_round(&listed(Vec::new()), &[], missing_after(lease), lease, now);
            }
            instances.own(&me.replica);
            let round = instances.end_round(
                &listed(vec![me.clone()]),
                &[],
                missing_after(lease),
                lease,
                now,
            );
            assert_eq!(
                round.complete_until,
                Some(now + missing_after(lease)),
                "{leaves}"
            );
        }
    }

    /// Serves an instance over HTTP that reads the others over HTTP. Returns
    /// its store, its URL, its sharing task, not yet started, its list, and
    /// what it shares.
    async fn serve(
        stop: &CancellationToken,
    ) -> (Arc<MemoryPoolStore>, Url, PeerSync, Arc<Listed>, Sharing) {
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
                sharing: Some(Arc::new(sharing.clone())),
            },
            None,
            stop.clone(),
        ));
        (store, url, sync, list, sharing)
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
        let (a, a_url, _, _, a_sharing) = serve(&stop).await;
        let (b, b_url, b_sync, b_list, b_sharing) = serve(&stop).await;
        let _ = a.register(pool("m1")).await.unwrap();
        let _ = b.register(pool("m1")).await.unwrap();
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
        // B named itself when it asked A, and said that it holds a
        // registration. One that holds nothing is not noted, and one that does
        // not say counts as holding some.
        let (idle, unsure) = (Uuid::new_v4(), Uuid::new_v4());
        for query in [
            format!("asker={idle}&holding=false"),
            format!("asker={unsure}"),
        ] {
            let url = a_url.join(&format!("registrations?{query}")).unwrap();
            assert!(reqwest::get(url).await.unwrap().status().is_success());
        }
        let mut expected = vec![b_sharing.instance, unsure];
        expected.sort();
        let mut askers: Vec<Uuid> = a_sharing
            .askers(lease(), Instant::now())
            .into_iter()
            .map(|(asker, _)| asker)
            .collect();
        askers.sort();
        assert_eq!(askers, expected);
        // A's answers report them, as the other instances read them.
        let url = a_url.join("registrations").unwrap();
        let reply: SharedRegistrations = reqwest::get(url).await.unwrap().json().await.unwrap();
        let mut reported: Vec<Uuid> = reply
            .askers
            .into_iter()
            .map(|asker| asker.into_parts().0)
            .collect();
        reported.sort();
        assert_eq!(reported, expected);
        stop.cancel();
    }
}
