// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unix Domain Socket (UDS) Transport Module
//!
//! This module provides a UDS transport implementation that mirrors the TCP transport
//! but uses Unix domain sockets instead of TCP connections. It reuses the same
//! zero-copy frame codec (`TcpFrameCodec`) since the framing protocol is transport-agnostic.
//!
//! Key differences from TCP:
//! - Uses `PathBuf` instead of `SocketAddr`
//! - Uses `UnixStream`/`UnixListener` instead of `TcpStream`/`TcpListener`
//! - No TCP-specific options (nodelay, keepalive, CPU pinning)
//! - Endpoint format: `uds:///path/to/socket`
//!
//! This transport is ideal for same-host communication (e.g., daemon-to-container via
//! bind-mounted sockets), avoiding the overhead of the TCP/IP stack entirely.

mod listener;
mod transport;

pub use listener::{UdsListener, UdsListenerBuilder};
pub use transport::{UdsTransport, UdsTransportBuilder};
