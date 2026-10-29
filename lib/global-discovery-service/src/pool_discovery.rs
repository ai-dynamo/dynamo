// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The contract between the Global Discovery Service and its stores. Design:
//! <https://github.com/ai-dynamo/enhancements/pull/103>.
//!
//! The service reads and writes only through this contract, so a store can
//! change without changing PoolRelays or routers:
//! - [`PoolDirectory`]: which pools exist.
//! - [`PoolRegistrar`]: a PoolRelay registers its pool. Each registration is
//!   also the pool's heartbeat.
//! - [`ReplicaDirectory`]: the Global Router replicas a PoolRelay connects to.
//! - [`RegistrationExchange`]: service instances share registrations.
//!
//! The two directories are aliases of one trait, [`Directory`], over pools and
//! over router replicas. What they list, and what a PoolRelay registers, are
//! the service's JSON types from `dynamo_global_common`.

mod pools;
mod registrar;
mod replicas;
mod watch;

pub use pools::{Capacity, PoolDirectory, PoolEvent, PoolListing};
pub use registrar::{Lease, PoolRegistrar, PoolRegistration, RegistrationExchange};
pub use replicas::{ReplicaDirectory, RouterReplica, RouterReplicaEvent, RouterReplicaId};
pub use watch::{Completeness, Directory, Snapshot, WatchEvent, WatchStream};

use thiserror::Error;

/// Shared by every store, so the service maps each case to one status code.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DiscoveryError {
    #[error("discovery backend is unavailable")]
    Unavailable(#[source] anyhow::Error),
}
