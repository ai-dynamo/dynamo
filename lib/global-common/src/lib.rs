// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Types shared by the Global Router components.
//!
//! - A pool's identity and location: [`PoolKey`], [`PoolId`] and its deriver
//!   [`V1PoolIdDeriver`], and [`PoolLocation`].
//! - The JSON of the Global Discovery Service's HTTP API, which the service
//!   and its clients exchange:
//!
//!   | Call | Body | Reply |
//!   |---|---|---|
//!   | `POST /pools` | [`PoolRegistration`] | [`Lease`] |
//!   | `GET /pools` | | [`PoolsResponse`] |
//!   | `GET /routers` | | [`RoutersResponse`] |
//!   | `POST /routers` | [`RouterRegistration`] | [`Lease`] |
//!
//!   A router matches a connecting PoolRelay against a listing's
//!   `relay_identities` with [`address_identity`].
//! - Global Router replicas ([`RouterReplica`]), and list-and-watch events
//!   and snapshots ([`WatchEvent`], [`Snapshot`]), which the service's watch
//!   contract and its clients share.

mod discovery;
/// Relay-to-router receiving transport. Identity fields name routing state;
/// deployment policy decides whether the network needs authentication.
#[cfg(feature = "grpc")]
pub mod grpc {
    tonic::include_proto!("dynamo.globalview.v1");

    /// Four subscription payloads of up to 4 MiB each, plus protobuf framing.
    pub const MAX_POOL_VIEW_FRAME_BYTES: usize = 4 * 4 * 1024 * 1024 + 64;
}
mod pool;
mod replicas;
mod watch;

pub use discovery::{
    Capacity, Lease, LeaseError, PoolListing, PoolRegistration, PoolsResponse, RouterRegistration,
    RouterRegistrationError, RoutersResponse, address_identity,
};
pub use pool::{PoolId, PoolIdDeriver, PoolKey, PoolKeyError, PoolLocation, V1PoolIdDeriver};
pub use replicas::{RouterReplica, RouterReplicaEvent, RouterReplicaId};
pub use watch::{Completeness, Snapshot, WatchEvent};
