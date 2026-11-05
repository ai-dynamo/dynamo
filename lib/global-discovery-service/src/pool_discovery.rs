// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The contract between the Global Discovery Service and its backends.
//! Design: <https://github.com/ai-dynamo/dynamo/issues/17600>.
//!
//! The service reads and writes only through this contract, so a backend
//! can change without changing PoolRelays or routers:
//! - [`PoolDirectory`]: which pools exist.
//! - [`PoolRegistrar`]: a PoolRelay registers its pool. Each registration is
//!   also the pool's heartbeat.
//! - [`ReplicaDirectory`]: the Global Router replicas a PoolRelay connects to.
//! - [`RegistrationSharing`], [`RegistrationSource`] and
//!   [`RegistrationExchange`]: service instances share registrations.
//!
//! The two directories are aliases of one trait, [`Directory`], over pools and
//! over router replicas. What they list, and what a PoolRelay registers, are
//! the service's JSON types from `dynamo_global_common`.
//!
//! # Partial lists
//!
//! A list may be incomplete: its [`Completeness`] is then `Estimated`, and
//! `GET /pools` says `complete: false`. A caller that keeps its own list of
//! pools, such as a router, should do this:
//! - If the list is complete, replace what you have.
//! - If it's partial, merge instead. Add new pools and update changed ones,
//!   but keep known pools that are missing. Removals wait for a complete
//!   list.
//! - A router that's just starting and knows no pools should still use a
//!   partial list. Some pools beat none.
//!
//! With DynamoDB registration storage, a replica lists every pool at once
//! but calls the list partial for one lease after it starts, in case a
//! PoolRelay still registers elsewhere, and while and for one lease after
//! the table cannot be reached, since PoolRelays cannot renew meanwhile.

mod pools;
mod registrar;
mod replicas;
mod sharing;
mod watch;

pub use pools::{Capacity, PoolDirectory, PoolEvent, PoolListing};
pub use registrar::{Lease, PoolRegistrar, PoolRegistration};
pub use replicas::{ReplicaDirectory, RouterReplica, RouterReplicaEvent, RouterReplicaId};
pub use sharing::{
    ExchangeRound, RegistrationExchange, RegistrationSharing, RegistrationSource,
    RegistrationsRequest, SharedRegistration, SharedRegistrations, answer_timeout,
};
pub use watch::{Completeness, Directory, Snapshot, WatchEvent, WatchStream};

use thiserror::Error;

/// Shared by every backend, so the service maps each case to one status code.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DiscoveryError {
    #[error("discovery backend is unavailable")]
    Unavailable(#[source] anyhow::Error),
    /// The backend cannot keep this registration as it is, so sending it
    /// again cannot help. The reason is for the caller.
    #[error("the registration cannot be stored: {0}")]
    Rejected(String),
}
