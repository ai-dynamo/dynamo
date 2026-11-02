// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Rust JWT authentication + optional FlexPrice usage-billing layer
//! for the Dynamo HTTP frontend.
//!
//! Auth activates only when `DYN_AUTH_ENABLED=true`, gating every inference
//! endpoint (chat completions, completions, embeddings, etc.) while leaving
//! system routes (health/live/metrics/models) unauthenticated. FlexPrice
//! usage-event billing is optional on top of auth (`DYN_FLEXPRICE_ENABLED`)
//! and never adds latency to the request path — events are enqueued
//! fire-and-forget and drained by a background worker. When billing is
//! enabled, a prepaid org whose wallet balance drops below
//! `DYN_FLEXPRICE_MINIMUM_BALANCE` (default: negative) is blocked with 402
//! before reaching the handler; postpaid orgs always bypass this gate. See
//! [`BalanceChecker`].
//!
//! This replaces the two-hop Python `aiohttp` reverse-proxy
//! (`components/src/dynamo/frontend/flexprice/`) with native middleware and
//! an RAII billing guard inside the Rust HTTP service itself.

mod auth;
mod balance;
mod client;
mod config;
mod guard;
mod middleware;
mod revoke;

pub use auth::AuthError;
pub use balance::{BalanceChecker, BalanceStatus};
pub use client::FlexPriceClient;
pub use config::{AuthConfig, FlexPriceConfig};
pub use guard::UsageBillingGuard;
pub use middleware::{OrgUuid, UserUuid, auth_middleware};
pub use revoke::RevokeChecker;

#[cfg(test)]
pub(crate) use auth::tests::sign as sign_test_jwt;
#[cfg(test)]
pub(crate) use revoke::tests as revoke_tests;
