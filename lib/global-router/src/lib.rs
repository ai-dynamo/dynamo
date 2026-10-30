// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool selection for the Global Router.
//!
//! The Global Frontend asks a [`PoolSelector`] which pool, and which of its
//! frontends, serves each request. Selectors that weigh load read it from
//! [`PoolSignals`], which background tasks fill from each pool's Relay and
//! from round trips this process measures, and count this process's own
//! requests in an [`InFlightTracker`].

pub mod affinity;
mod load;
mod pool;
mod selector;
mod signals;

pub use affinity::AffinitySelector;
pub use load::{InFlightGuard, InFlightTracker};
pub use pool::Pool;
pub use selector::{
    LeastInFlightSelector, PoolSelector, PoolTarget, RoundRobinSelector, RoutingRequest,
};
pub use signals::{LoadReport, PoolOccupancy, PoolSignals};
