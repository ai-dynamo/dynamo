// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How service instances share registrations (see `crate::peers`): each
//! serves the ones its own PoolRelays made through [`RegistrationSharing`],
//! as [`SharedRegistrations`] (`GET /registrations`), reads the others'
//! through a [`RegistrationSource`], naming itself in each
//! [`RegistrationsRequest`], and keeps copies in a [`RegistrationExchange`].
//! Instances are named as the replicas of the service's own Service.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use uuid::Uuid;

use super::{DiscoveryError, Lease, PoolRegistration, RouterReplica};

/// Where an instance keeps its registrations and the copies of the others'.
pub trait RegistrationExchange: Send + Sync {
    /// The live registrations PoolRelays made with this instance, with the
    /// time each has left. Copies are left out, so each registration is shared
    /// only by the instance that received it.
    fn local_registrations(&self) -> Vec<(PoolRegistration, Duration)>;

    /// Keeps a copy of a registration another instance received, for the
    /// `remaining` time it had when that instance answered, counted from when
    /// it was asked. That instance answers within [`answer_timeout`], so the
    /// copy lapses at most that much before its original.
    fn adopt(&self, registration: PoolRegistration, remaining: Duration);

    /// Takes what a round of asking the other instances showed.
    fn end_round(&self, round: ExchangeRound);

    /// Whether every PoolRelay that registers with this instance has done so
    /// since it started: a lease has passed. Until then, an instance that
    /// copies its registrations may be missing some.
    fn holds_every_local_registration(&self) -> bool;
}

/// What an instance serves the others.
pub trait RegistrationSharing: Send + Sync {
    /// The `GET /registrations` reply: the registrations this instance's
    /// PoolRelays made, with the time each has left.
    fn reply(&self) -> SharedRegistrations;

    /// Notes that the instance named `asker` asked, as its request says. This
    /// is how an instance learns of instances that read it.
    fn asked_by(&self, asker: Uuid);

    /// Whether this instance has asked the others once. Until then it may
    /// know only its own PoolRelays, so it is not ready.
    fn synced(&self) -> bool;
}

/// The other instances, as this one reads them.
#[async_trait]
pub trait RegistrationSource: Send + Sync {
    /// What `instance` shares, asked by `asker`, the name that this
    /// instance's own replies carry. The caller bounds how long it waits.
    async fn shared_registrations(
        &self,
        instance: &RouterReplica,
        asker: Uuid,
    ) -> Result<SharedRegistrations, DiscoveryError>;
}

/// How long an instance waits for another's answer: half a heartbeat
/// interval. A copy therefore lapses at most this much before its original.
pub fn answer_timeout(lease: Lease) -> Duration {
    lease.heartbeat_interval() / 2
}

/// What a round of asking the other instances showed. Lists are `Estimated`
/// while registrations made with another instance may be missing here. The
/// default, before any round, says they may be.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExchangeRound {
    /// The oldest of the times each listed instance was last asked for an
    /// answer that it gave with all of its registrations. A registration
    /// that lapsed here after that may have been renewed with one of them, as
    /// when a PoolRelay renews late, with this instance or another, so lists
    /// stay `Estimated` until every instance has been asked again. `None`
    /// while one may be missing.
    pub answered_since: Option<Instant>,
    /// Until when no registration made with another instance can be missing
    /// here, other than one renewed after `answered_since`. `None` while one
    /// may be missing. The time passes on its own, so lists turn `Estimated`
    /// even if no round ends again.
    pub complete_until: Option<Instant>,
}

/// `GET /registrations` query string.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RegistrationsRequest {
    /// The asking instance, as its replies name it. Instances of earlier
    /// versions do not say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asker: Option<Uuid>,
}

/// `GET /registrations` reply. Only service instances read it, so it is not
/// in `dynamo_global_common`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SharedRegistrations {
    /// Names the answering process, so an instance can tell its own reply.
    pub instance: Uuid,
    /// The answering instance's heartbeat interval and lease. Every instance
    /// must use the same ones.
    pub lease: Lease,
    /// The answering instance has heard from every PoolRelay that registers
    /// with it: a lease has passed since it started. A restarted instance's
    /// first replies may lack registrations it has not heard again yet.
    pub complete: bool,
    pub registrations: Vec<SharedRegistration>,
}

/// A registration as its PoolRelay sent it, and how long it has left.
/// Durations, not times, so the instances' clocks need not agree.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SharedRegistration {
    registration: PoolRegistration,
    expires_in_ms: u64,
}

impl SharedRegistration {
    pub fn new(registration: PoolRegistration, remaining: Duration) -> Self {
        Self {
            registration,
            expires_in_ms: u64::try_from(remaining.as_millis()).unwrap_or(u64::MAX),
        }
    }

    /// The registration, and how long it had left.
    pub fn into_parts(self) -> (PoolRegistration, Duration) {
        (self.registration, Duration::from_millis(self.expires_in_ms))
    }
}
