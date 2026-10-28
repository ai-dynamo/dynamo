// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client for the two calls a PoolRelay makes to the Global Discovery
//! Service: listing every Global Router replica
//! ([`GlobalDiscoveryClient::routers`]) and registering its pool
//! ([`GlobalDiscoveryClient::register_pool`]).
//!
//! The crate also defines the router replica types and the list-and-watch
//! events.

mod client;
mod replicas;
mod watch;
mod wire;

pub use client::{DiscoveryClientError, GlobalDiscoveryClient};
pub use replicas::{ReplicaEndpoint, ReplicaEvent, ReplicaId};
pub use watch::{Completeness, Snapshot, WatchEvent};
pub use wire::{Capacity, Lease, PoolKey, PoolLocation, PoolRegistration};
