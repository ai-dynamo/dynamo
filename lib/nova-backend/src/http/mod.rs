// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP Transport Module
//!
//! This module provides a simple HTTP transport implementation with:
//! - Fire-and-forget semantics (one-way messaging)
//! - Base64 encoded header bytes in HTTP headers
//! - Raw payload bytes in HTTP body
//! - Three separate routes for Message, Response, and Event types
//! - Simple shutdown (stop accepting requests)

mod server;
mod transport;

pub use server::HttpServer;
pub use transport::{HttpTransport, HttpTransportBuilder};
