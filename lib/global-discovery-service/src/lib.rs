// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool discovery for the Global Router, served over HTTP.
//!
//! The contract comes first:
//! - [`service`]: the HTTP API (`POST /pools`, `GET /pools`, `GET /routers`,
//!   `GET /healthz`), whose JSON types are in `dynamo_global_common`, and
//!   [`DiscoveryBackend`], the backend behind it.
//! - [`pool_discovery`]: the traits that backend is built from.
//!
//! [`backends`] implements those traits: registrations kept in memory, and
//! the EndpointSlices that SIG Multicluster (MCS) imports.

pub mod backends;
pub mod pool_discovery;
pub mod service;

pub use service::{DiscoveryBackend, Tls, routes, serve};
