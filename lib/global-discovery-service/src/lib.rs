// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool discovery for the Global Router, served over HTTP.
//!
//! The contract comes first:
//! - [`service`]: the HTTP API, whose JSON types are in `dynamo_global_common`,
//!   and [`PoolDiscoveryBackend`], the backend behind it.
//! - [`pool_discovery`]: the traits that backend is built from.
//!
//! [`backends`] implements those traits: registrations kept in memory or in
//! a DynamoDB table, and the EndpointSlices that SIG Multicluster (MCS)
//! imports, joined with registrations. [`peers`] shares registrations kept in
//! memory between service instances.

pub mod backends;
pub mod peers;
pub mod pool_discovery;
pub mod service;

pub use backends::clock::{Clock, ManualClock, SystemClock};
pub use backends::dynamodb::{ConnectError, DynamoDbConfig, DynamoDbPoolStore};
pub use backends::http::HttpRegistrationSource;
pub use peers::{MIN_LEASE_HEARTBEATS, PeerSync};
pub use service::{PoolDiscoveryBackend, Tls, routes, serve};
