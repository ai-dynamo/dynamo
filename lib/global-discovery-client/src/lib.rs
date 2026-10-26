// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client for the Global Discovery Service. A PoolRelay lists every Global
//! Router replica ([`GlobalDiscoveryClient::routers`]) and registers its pool
//! ([`GlobalDiscoveryClient::register_pool`]). A Global Router lists every
//! pool ([`GlobalDiscoveryClient::pools`]).
//!
//! The crate also defines the read-side types that the client and the service
//! share: the router replica types and the list-and-watch events.

mod client;
mod replicas;
mod watch;
mod wire;

pub use client::{DiscoveryClientError, GlobalDiscoveryClient};
pub use replicas::{ReplicaEndpoint, ReplicaEvent, ReplicaId};
pub use watch::{Completeness, Snapshot, WatchEvent};
pub use wire::{Capacity, Lease, PoolKey, PoolListing, PoolLocation, PoolRegistration};
