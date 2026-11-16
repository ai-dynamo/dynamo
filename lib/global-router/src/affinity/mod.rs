// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Best-effort session-to-pool affinity, independently of its persistence backend.
//!
//! The contracts here are internal to the Global Router: they are public so a
//! backend can live in its own crate, and they change without a major version
//! while persistent affinity is being built out.

mod cache;
mod memory;
mod selector;
mod stats;
mod store;
mod tenant;
mod tracker;

pub use cache::CachedAffinityStore;
pub use memory::InMemoryAffinityStore;
pub use selector::{AffinityConfig, AffinitySelector, BreakerConfig, StoreTimer};
pub use stats::{AffinityStats, CacheStats};
pub use store::{
    AffinityBinding, AffinityError, AffinityKey, AffinityStore, Expected, WriteOutcome,
};
pub use tenant::{ANONYMOUS_TENANT, tenant_id};
pub use tracker::{AffinitySnapshot, SessionAffinity, StoreBackedSessionAffinity};
