// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC Transport Module
//!
//! This module provides a gRPC transport implementation using tonic for
//! bidirectional streaming. It wraps the same framing protocol used by
//! TCP/UDS transports in gRPC FramedData messages.

mod server;
mod transport;

/// Generated protobuf types for the VeloStreaming gRPC service.
#[allow(missing_docs)]
pub mod proto {
    tonic::include_proto!("velo.streaming.v1");
}

pub use server::VeloStreamingService;
pub use transport::{GrpcTransport, GrpcTransportBuilder};
