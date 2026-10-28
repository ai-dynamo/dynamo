// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Frontend: an OpenAI-compatible HTTP entry point that forwards
//! each request, as received, to a pool frontend chosen by its pool
//! selector, and streams the pool's response back as sent (see [`app`]).
//!
//! The Global Frontend does not tokenize or validate. The pool frontend
//! handles a forwarded request exactly as it would one sent to it directly.
//! Selectors that weigh pool load read it from `PoolSignals`, which
//! [`follow_relays`] fills from each pool's KV DC Relay and
//! [`measure_round_trips`] from the round trip to each region.

mod proxy;
mod relay;
mod round_trip;

pub use proxy::{app, serve};
pub use relay::follow_relays;
pub use round_trip::measure_round_trips;

mod catalog;
pub use catalog::{PoolCatalog, PoolSnapshot, StaticPoolCatalog};
pub use dynamo_global_common::Pool;
pub use dynamo_global_router::{PoolSelector, PoolTarget, RoundRobinSelector, RoutingRequest};
