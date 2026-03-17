// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Bind to a random available port and return both the listener and port.
///
/// Callers that pass the listener to `HttpService::run_with_listener` (or
/// `spawn_with_listener`) avoid the TOCTOU race where the port is freed and
/// then re-assigned to a different test before the service binds.
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

/// Get a random available port for testing.
///
/// **Prefer [`bind_random_port`] when possible** — it keeps the socket open
/// until the service binds, eliminating port-collision races.  This helper
/// is still useful for services (e.g. gRPC) that do not yet accept a
/// pre-bound listener.
#[allow(dead_code)]
pub async fn get_random_port() -> u16 {
    let (_listener, port) = bind_random_port().await;
    port
}
