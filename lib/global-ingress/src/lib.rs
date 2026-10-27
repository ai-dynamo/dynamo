// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Frontend: Dynamo's OpenAI-compatible HTTP service serving one
//! engine, [`PoolForwardingEngine`], that forwards each request to a pool
//! frontend chosen by the Global Router.
//!
//! The Global Frontend does not tokenize. The pool frontend preprocesses a
//! forwarded request exactly as it would one sent to it directly.
//! [`ModelSync`] keeps the HTTP service's models in step with the pool
//! catalog, so a model is served while at least one pool serves it.
//! [`follow_relays`] and [`measure_round_trips`] keep the pool load and
//! round trips that the session-aware selectors read.

mod forward;
mod model_sync;
mod relay;
mod round_trip;
mod stats;

pub use forward::{ForwardedRequest, PoolForwardingEngine};
pub use model_sync::ModelSync;
pub use relay::follow_relays;
pub use round_trip::measure_round_trips;
pub use stats::RoutingStats;
