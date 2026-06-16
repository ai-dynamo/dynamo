// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Bind an ephemeral port and return both the listener and its port.
///
/// The caller keeps the listener alive (and hands it to whatever code accepts on it)
/// to close the TOCTOU gap that `get_random_port` has: there, the listener is dropped
/// before the test's real server binds, letting a parallel test re-grab the same port.
// Allowed-dead-code because this module is included via `#[path]` in several test
// binaries; not every binary uses both helpers.
#[allow(dead_code)]
pub async fn bind_random_port() -> (tokio::net::TcpListener, u16) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind ephemeral port");
    let port = listener
        .local_addr()
        .expect("failed to read local_addr")
        .port();
    (listener, port)
}

/// Get a random available port for testing (prefer to hardcoding port numbers to avoid collisions).
///
/// Note: this drops the listener before returning, so the port can be reused by another
/// caller before the test rebinds. Prefer [`bind_random_port`] for HTTP tests that can
/// hand the listener to the service.
#[allow(dead_code)]
pub async fn get_random_port() -> u16 {
    let (_listener, port) = bind_random_port().await;
    port
}
