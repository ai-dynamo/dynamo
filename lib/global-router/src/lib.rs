// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool selection for the Global Router.
//!
//! The Global Frontend asks a [`PoolSelector`] which pool, and which of its
//! frontends, serves each request. Selectors that weigh load read it from
//! [`PoolSignals`], which background tasks fill from each pool's Relay and
//! from round trips this process measures, and count this process's own
//! requests in an [`InFlightTracker`].
//!
//! [`CoreSelector`] is a [`PoolSelector`] too: it selects through the
//! embedded router core, with each pool registered as one of the core's
//! workers and the policies as the core's plugins (see [`core`]).

mod affinity;
pub mod core;
mod frontend_signals;
mod load;
mod placement;
mod pool;
mod selector;
mod signals;

pub use affinity::AffinitySelector;
pub use core::replica_sync;
pub use core::{
    AffinityConfig, CorePolicy, CoreSelector, LeastInFlightConfig, MAX_SESSION_ID_BYTES,
    Reservation,
};
pub use frontend_signals::FrontendSignals;
pub use load::{InFlightGuard, InFlightTracker};
pub use pool::Pool;
pub use selector::{
    LeastInFlightSelector, PoolSelector, PoolTarget, RequestLifecycle, RoundRobinSelector,
    RoutingRequest, Selection,
};
pub use signals::{LoadReport, PoolOccupancy, PoolSignals, RelaySource};
