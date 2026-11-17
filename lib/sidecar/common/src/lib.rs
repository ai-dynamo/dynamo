// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared infrastructure for Rust sidecars.

#[cfg(feature = "jemalloc")]
#[doc(hidden)]
pub mod allocator;
mod args;
mod endpoint;
mod error;
mod json;
mod run;
mod transport;

/// Startup budget for tests that expect a sidecar to become ready after a CI pause.
#[cfg(any(test, feature = "testing"))]
pub const TEST_STARTUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

#[cfg(feature = "tonic-v14")]
pub mod v14 {
    use tonic_v14 as tonic;

    pub use crate::error::status_to_dynamo_v14 as status_to_dynamo;
    pub use crate::error::{
        is_broken_stream_v14 as is_broken_stream,
        stream_status_to_dynamo_v14 as stream_status_to_dynamo,
    };

    // Keep connection policy identical across Tonic versions.
    include!("transport.rs");
}

pub use args::{GrpcTransportArgs, GrpcTransportConfig, SidecarArgs};
pub use endpoint::{GrpcEndpoint, HttpEndpoint};
pub use error::{
    SidecarStartupError, cancelled, cannot_connect, connection_timeout, engine_shutdown,
    invalid_argument, is_broken_stream, protocol_error, status_to_dynamo, stream_incomplete,
    stream_status_to_dynamo,
};
pub use json::{json_to_struct, struct_to_json};
#[cfg(feature = "tonic-v14")]
pub use json::{json_to_struct_v14, struct_to_json_v14};
pub use transport::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, format_error_chain, startup_deadline,
};

pub use run::{EngineBootstrapResult, run, run_task, run_with_drt};

#[cfg(test)]
#[path = "transport/tests.rs"]
mod transport_tests;
