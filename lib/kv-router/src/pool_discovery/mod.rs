// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pluggable pool discovery for the Global Router (DEP-1276).
//!
//! Pools dial into the hub. Discovery answers three questions: which pools
//! exist, who may speak for each pool, and which router replicas a pool must
//! connect to. Pool state still arrives on the Relay's own connections and is
//! assembled by [`crate::global_view::source::PoolObservationAssembler`].
//!
//! | Mode | Pool list stored in | Pools call |
//! |---|---|---|
//! | gRPC | Router memory | `RegisterPool` on the router |
//! | Hybrid | Hub Kubernetes API, written by the router | `RegisterPool` on the router |
//! | Kubernetes (SIG Multicluster) | Imported EndpointSlices, written by the MCS implementation | Nothing: the operator creates a `ServiceExport` |
//! | File | Static file (tests, first deployments) | Nothing |

pub mod admission;
pub mod announce;
pub mod identity;
pub mod mode;
pub mod record;
pub mod registrar;
pub mod replicas;
pub mod watch;

pub use admission::PoolAdmission;
pub use announce::{Announcement, PoolAnnouncer};
pub use identity::{
    PeerCredentials, RelayAuthenticator, RelayIdentity, VerifiedRelayIdentity, verify,
};
pub use mode::{
    BuildPoolDiscovery, McsImplementation, PoolDiscovery, PoolDiscoveryMode, SettlePolicy,
};
pub use record::{PoolDirectory, PoolEvent, PoolRecord, PoolRecordBuilder, Revision};
pub use registrar::{LeasePolicy, PoolRegistrar, Registration};
pub use replicas::{ReplicaDirectory, ReplicaEndpoint, ReplicaEvent, ReplicaId};
pub use watch::{Completeness, Directory, Snapshot, WatchEvent, WatchStream};

use thiserror::Error;

use crate::global_view::PoolKey;

/// Shared by every discovery interface so the gRPC server can map each case
/// to one status code.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PoolDiscoveryError {
    #[error("caller is not authenticated")]
    Unauthenticated,
    #[error("caller may not act for pool {0:?}")]
    PermissionDenied(PoolKey),
    #[error("pool {0:?} is not registered")]
    UnknownPool(PoolKey),
    #[error("discovery backend is unavailable")]
    Unavailable(#[source] anyhow::Error),
}
