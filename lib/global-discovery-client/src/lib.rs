// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client for the two calls a PoolRelay makes to the Global Discovery
//! Service: listing every Global Router replica
//! ([`GlobalDiscoveryClient::routers`]) and registering its pool
//! ([`GlobalDiscoveryClient::register_pool`]).
//!
//! The types it sends and receives are `dynamo-global-common`'s, re-exported
//! here, so the client reads and writes the JSON as the service does.

mod client;

pub use client::{DiscoveryClientError, GlobalDiscoveryClient};
pub use dynamo_global_common::{
    Capacity, Lease, LeaseError, PoolKey, PoolKeyError, PoolLocation, PoolRegistration,
    RouterReplica, RouterReplicaId,
};
