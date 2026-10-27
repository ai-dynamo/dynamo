// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The contract between the Global Discovery Service and its stores. Design:
//! <https://github.com/ai-dynamo/enhancements/pull/103>.
//!
//! The service reads and writes only through these traits, so a store can
//! change without changing PoolRelays or routers:
//! - [`PoolDirectory`]: which pools exist.
//! - [`PoolRegistrar`]: a PoolRelay registers its pool. Each registration is
//!   also the pool's heartbeat.
//! - [`ReplicaDirectory`]: the Global Router replicas a PoolRelay connects to.

mod record;
mod registrar;
mod replicas;
mod watch;

pub use record::{Capacity, PoolDirectory, PoolEvent, PoolRecord, Revision};
pub use registrar::{PoolRegistrar, Registration};
pub use replicas::{ReplicaDirectory, ReplicaEndpoint, ReplicaEvent, ReplicaId};
pub use watch::{Completeness, Directory, Snapshot, WatchEvent, WatchStream};

use thiserror::Error;

/// Shared by every store, so the service maps each case to one status code.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PoolDiscoveryError {
    #[error("discovery backend is unavailable")]
    Unavailable(#[source] anyhow::Error),
}
