// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use async_trait::async_trait;
pub use dynamo_global_common::{Lease, PoolRegistration};
use tokio::time::Instant;

use super::DiscoveryError;

/// The write side. A PoolRelay registers its pool on a timer, and each call
/// is also the pool's heartbeat: registering an unchanged pool again only
/// renews it. The [`Lease`] says when to register again, and how long the
/// pool stays listed after its last registration, measured on the backend's
/// clock. PoolRelays get router replicas from a [`super::ReplicaDirectory`],
/// not from this reply.
///
/// V1 assumes a trusted network, where any caller may register any pool, so
/// `register` does not take the caller.
#[async_trait]
pub trait PoolRegistrar: Send + Sync {
    async fn register(&self, registration: PoolRegistration) -> Result<Lease, DiscoveryError>;
}

/// Lets service instances share registrations (see `crate::peers`): each
/// shares the ones its own PoolRelays made, and keeps copies of the others'.
pub trait RegistrationExchange: Send + Sync {
    /// The live registrations PoolRelays made with this instance, with the
    /// time each has left. Copies are left out, so each registration is shared
    /// only by the instance that received it.
    fn local_registrations(&self) -> Vec<(PoolRegistration, Duration)>;

    /// Keeps a copy of a registration another instance received, for the
    /// `remaining` time it had when that instance answered, counted from when
    /// it was asked. That instance answers within half a heartbeat interval,
    /// so the copy lapses at most that much before its original.
    fn adopt(&self, registration: PoolRegistration, remaining: Duration);

    /// Takes what a round of asking the other instances showed.
    fn end_round(&self, round: ExchangeRound);

    /// Whether every PoolRelay that registers with this instance has done so
    /// since it started: a lease has passed. Until then, an instance that
    /// copies its registrations may be missing some.
    fn holds_every_local_registration(&self) -> bool;
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
