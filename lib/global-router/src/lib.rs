// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool catalog and pool selection for the Global Router.
//!
//! The Global Frontend reads pools from a [`PoolCatalog`] and asks a
//! [`PoolSelector`] which pool serves each request. Neither knows about HTTP
//! or where pools are discovered, so a catalog backed by configuration, the
//! Global Discovery Service, or Pool Relay state plugs in behind the same
//! trait.

mod catalog;
mod discovery;
mod load;
mod placement;
mod selector;
mod session;
mod signals;

pub use catalog::{Pool, PoolCatalog, PoolSnapshot, StaticPoolCatalog};
pub use discovery::DiscoveryPoolCatalog;
pub use load::{InFlightGuard, InFlightTracker};
pub use placement::Overload;
pub use selector::{
    LeastInFlightSelector, PoolSelector, PoolTarget, RoundRobinSelector, RoutingRequest,
};
pub use session::{AffinitySelector, RendezvousSelector};
pub use signals::{PoolLoad, PoolSignals};
