// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC transport implementation
//!
//! This module provides a gRPC-based transport using tonic with minimal protobuf.
//! Messages are wrapped in a simple protobuf FramedData wrapper that contains
//! our existing TCP frame format.
//!
//! The transport reuses the TCP frame format for encoding messages:
//! - 11-byte header: [u16: version][u8: msg_type][u32: header_len][u32: payload_len]
//! - Followed by header bytes and payload bytes
//! - Wrapped in protobuf FramedData { bytes data = 1; }
//!
//! Each peer connection uses a bidirectional gRPC stream over HTTP/2:
//! - Client → Server: Send pre-framed messages
//! - Server → Client: Empty (we only use client->server direction)

mod client;
mod server;
mod transport;

pub use transport::{GrpcTransport, GrpcTransportBuilder};
