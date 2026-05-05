// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared constants for the bidi streaming example.
//!
//! Two binaries:
//! - `server` — registers a bidi handler that uppercases each item from
//!   the client and emits a small trailing summary line after the input
//!   stream ends (demonstrates the half-close: server can keep emitting
//!   after the client has signaled "no more input").
//! - `client` — opens a bidi session, streams a few inputs, prints each
//!   echoed response, and exits cleanly when the server finalizes.
//!
//! Run with the velo request plane:
//!
//! ```text
//! # Terminal 1 (server)
//! DYN_REQUEST_PLANE=velo cargo run --bin bidi_server
//!
//! # Terminal 2 (client)
//! DYN_REQUEST_PLANE=velo cargo run --bin bidi_client
//! ```

pub const DEFAULT_NAMESPACE: &str = "dynamo";
pub const COMPONENT_NAME: &str = "bidi-backend";
pub const ENDPOINT_NAME: &str = "uppercase";
