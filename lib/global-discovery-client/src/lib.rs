// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client for the Global Discovery Service: listing every Global Router
//! replica ([`GlobalDiscoveryClient::routers`]), listing pools
//! ([`GlobalDiscoveryClient::pools`]) and registering a pool
//! ([`GlobalDiscoveryClient::register_pool`]). [`RouterCache`] and
//! [`PoolCache`] keep the router and pool lists current by reading them on an
//! interval, and [`PoolHeartbeat`] keeps a pool registered by renewing its
//! lease.
//!
//! The types it sends and receives are `dynamo-global-common`'s, re-exported
//! here, so the client reads and writes the JSON as the service does.

mod cache;
mod client;
mod heartbeat;

pub use cache::{Freshness, PoolCache, RouterCache};
pub use client::{DiscoveryClientError, GlobalDiscoveryClient};
pub use dynamo_global_common::{
    Capacity, Lease, LeaseError, PoolKey, PoolKeyError, PoolListing, PoolLocation,
    PoolRegistration, PoolsResponse, RouterReplica, RouterReplicaId,
};
pub use heartbeat::{PoolHeartbeat, RegistrationStatus};
