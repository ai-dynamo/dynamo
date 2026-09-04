// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared infrastructure for Rust sidecars.

mod args;
mod endpoint;
mod error;
mod json;
mod transport;

pub use args::{GrpcTransportArgs, GrpcTransportConfig, SidecarArgs};
pub use endpoint::GrpcEndpoint;
pub use error::{
    cannot_connect, connection_timeout, engine_shutdown, invalid_argument, protocol_error,
    status_to_dynamo,
};
pub use json::{json_to_struct, struct_to_json};
pub use transport::{DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool};
