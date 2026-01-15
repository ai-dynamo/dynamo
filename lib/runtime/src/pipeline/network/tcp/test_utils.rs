// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test utilities shared across TCP transport tests.

use tokio::net::TcpListener;

/// Creates a connected TCP pair for testing.
///
/// Returns a tuple of (client, server) TcpStream instances that are connected to each other.
/// This is useful for testing functions that operate on TCP streams without needing
/// actual network communication.
pub async fn create_tcp_pair() -> (tokio::net::TcpStream, tokio::net::TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let client = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (server, _) = listener.accept().await.unwrap();

    (client, server)
}
