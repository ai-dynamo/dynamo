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
//!
//!   A router matches a connecting PoolRelay against a listing's
//!   `relay_identities` with [`address_identity`].
//! - Global Router replicas ([`RouterReplica`]), and list-and-watch events
//!   and snapshots ([`WatchEvent`], [`Snapshot`]), which the service's watch
//!   contract and its clients share.
//! - The gRPC contract replicas sync bookings and session bindings over
//!   ([`replica_sync`]); its generated code needs the `grpc` feature.

mod discovery;
mod pool;
pub mod replica_sync;
mod replicas;
mod watch;

pub use discovery::{
    Capacity, Lease, LeaseError, PoolListing, PoolRegistration, PoolsResponse, RoutersResponse,
    address_identity,
};
pub use pool::{PoolId, PoolIdDeriver, PoolKey, PoolKeyError, PoolLocation, V1PoolIdDeriver};
pub use replicas::{RouterReplica, RouterReplicaEvent, RouterReplicaId};
pub use watch::{Completeness, Snapshot, WatchEvent};
