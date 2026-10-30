// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client for the Global Discovery Service. PoolRelay lists router replicas
//! ([`GlobalDiscoveryClient::routers`]) and registers its pool
//! ([`GlobalDiscoveryClient::register_pool`]); the Global Router reads candidate
//! pools ([`GlobalDiscoveryClient::pools`]).
//!
//! The types it sends and receives are `dynamo-global-common`'s, re-exported
//! here, so the client reads and writes the JSON as the service does.

mod client;

pub use client::{DiscoveryClientError, GlobalDiscoveryClient};
pub use dynamo_global_common::{
    Capacity, Lease, LeaseError, PoolKey, PoolKeyError, PoolLocation, PoolRegistration,
    PoolsResponse, RouterReplica, RouterReplicaId,
};
