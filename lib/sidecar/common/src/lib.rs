// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared infrastructure for Rust sidecars.

mod args;
mod endpoint;
mod error;
mod transport;

#[cfg(feature = "tonic-v14")]
pub mod v14 {
    use tonic_v14 as tonic;

    pub use crate::error::status_to_dynamo_v14 as status_to_dynamo;

    // Keep connection policy identical across Tonic versions.
    include!("transport.rs");
}

pub use args::{GrpcTransportArgs, GrpcTransportConfig, SidecarArgs};
pub use endpoint::{GrpcEndpoint, HttpEndpoint};
pub use error::{
    SidecarStartupError, cannot_connect, connection_timeout, engine_shutdown, invalid_argument,
    protocol_error, status_to_dynamo,
};
pub use transport::{DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool};
