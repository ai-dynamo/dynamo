// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Frontend: Dynamo's OpenAI-compatible HTTP service serving one
//! engine, [`PoolForwardingEngine`], that forwards each request to a pool
//! frontend chosen by its pool selector.
//!
//! The Global Frontend does not tokenize. The pool frontend preprocesses a
//! forwarded request exactly as it would one sent to it directly.
//! [`ModelSync`] keeps the HTTP service's models in step with the pool
//! catalog, so a model is served while at least one pool serves it.
//! Selectors that weigh pool load read it from `PoolSignals`, which
//! [`follow_relays`] fills from each pool's KV DC Relay and
//! [`measure_round_trips`] from the round trip to each region.

mod forward;
mod model_sync;
mod relay;
mod round_trip;

pub use forward::{ForwardedRequest, PoolForwardingEngine};
pub use model_sync::ModelSync;
pub use relay::follow_relays;
pub use round_trip::measure_round_trips;

mod catalog;
pub use catalog::{PoolCatalog, PoolSnapshot, StaticPoolCatalog};
pub use dynamo_global_common::Pool;
pub use dynamo_global_router::{PoolSelector, PoolTarget, RoundRobinSelector, RoutingRequest};
