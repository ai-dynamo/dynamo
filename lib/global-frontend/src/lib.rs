// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Frontend: an OpenAI-compatible HTTP entry point that forwards
//! each request, as received, to a pool frontend chosen by its pool
//! selector, and streams the pool's response back as sent (see [`app`]).
//!
//! The Global Frontend does not tokenize or validate. The pool frontend
//! handles a forwarded request exactly as it would one sent to it directly.
//! Selectors that weigh pool load read it from `PoolSignals`, which
//! [`follow_relays`] or [`serve_relay_views`] fills from each pool's PoolRelay and
//! [`measure_round_trips`] from the round trip to each region.

mod catalog;
mod discovery;
mod metrics;
mod proxy;
mod relay;
mod relay_receiver;
mod relay_state;
mod relay_views;
mod round_trip;

pub use catalog::PoolSnapshot;
pub use discovery::discover_pools;
pub use metrics::Metrics;
pub use proxy::{app, serve};
pub use relay::follow_relays;
pub use relay_receiver::serve_relay_views;
pub(crate) use relay_state::RelayPoolState;
pub use relay_views::RelayGatedPoolCatalog;
pub(crate) use relay_views::{RelaySourceKey, RelayStreamGuard};
pub use round_trip::measure_round_trips;
