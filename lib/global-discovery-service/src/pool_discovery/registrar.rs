// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use async_trait::async_trait;
pub use dynamo_global_common::{Lease, PoolRegistration};

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
    /// `remaining` time it had there.
    fn adopt(&self, registration: PoolRegistration, remaining: Duration);

    /// Whether registrations made with another instance may be missing
    /// here: one has not answered within a lease, or the list of instances
    /// may be incomplete. While they may be, lists are `Estimated`.
    fn set_copies_missing(&self, missing: bool);
}
