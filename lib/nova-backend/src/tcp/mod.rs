// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TCP Transport Module
//!
//! This module provides a high-performance TCP transport implementation with:
//! - Zero-copy frame codec for minimal overhead
//! - CPU pinning support for predictable latency
//! - Frame type routing (Message, Response, Ack, Event)
//! - Graceful shutdown with proper FIN handling
//! - Keep-alive for dead connection detection

mod framing;
mod listener;
mod transport;

pub use framing::TcpFrameCodec;
pub use listener::{RuntimeConfig, TcpListener, TcpListenerBuilder};
pub use transport::{TcpTransport, TcpTransportBuilder};
