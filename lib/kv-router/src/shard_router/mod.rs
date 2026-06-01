// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw-UDS shard router components.
//!
//! This module is gated behind the `uds-raw-bench` feature so it is never
//! compiled into production builds.  It provides:
//!
//! - [`wire`]: Frame codec and request/response types for the UDS protocol.
//! - [`uds_raw`]: [`RawUdsShardClient`] — an [`AsyncShardHandle`] that
//!   communicates with a remote shard over UDS.
//! - [`uds_raw_server`]: [`RawUdsShardServer`] — accepts connections and
//!   dispatches frames to an underlying `S: AsyncShardHandle`.

pub mod uds_raw;
pub mod uds_raw_server;
pub mod wire;

pub use uds_raw::RawUdsShardClient;
pub use uds_raw_server::RawUdsShardServer;
