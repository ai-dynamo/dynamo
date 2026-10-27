// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Types that more than one Global Router component needs, defined once.
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
//!   `relay_identity` with [`address_identity`].
//! - Global Router replicas ([`ReplicaEndpoint`]), and list-and-watch events
//!   and snapshots ([`WatchEvent`], [`Snapshot`]), which the service's watch
//!   contract and its clients share.

mod discovery;
mod pool;
mod replicas;
mod watch;

pub use discovery::{
    Capacity, Lease, LeaseError, PoolListing, PoolRegistration, PoolsResponse, RoutersResponse,
    address_identity,
};
pub use pool::{PoolId, PoolIdDeriver, PoolKey, PoolKeyError, PoolLocation, V1PoolIdDeriver};
pub use replicas::{ReplicaEndpoint, ReplicaEvent, ReplicaId};
pub use watch::{Completeness, Snapshot, WatchEvent};
