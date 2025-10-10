// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request plane abstraction for different transport mechanisms

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::Arc;

/// Headers for request plane messages
pub type Headers = HashMap<String, String>;

/// Request plane client trait - sends requests to workers
#[async_trait]
pub trait RequestPlaneClient: Send + Sync {
    /// Send a request to the specified address with headers
    ///
    /// # Arguments
    /// * `address` - The destination address (e.g., NATS subject or HTTP URL)
    /// * `payload` - The request payload (typically TwoPartCodec-encoded)
    /// * `headers` - Request headers for tracing, metadata, etc.
    ///
    /// # Returns
    /// An acknowledgment that the request was received (not the full response)
    async fn send_request(
        &self,
        address: String,
        payload: Bytes,
        headers: Headers,
    ) -> Result<Bytes>;
}

/// Request plane server trait - receives requests from routers
#[async_trait]
pub trait RequestPlaneServer: Send + Sync {
    /// Start the request plane server
    ///
    /// # Arguments
    /// * `bind_address` - The address to bind to (e.g., ":8081" for HTTP)
    async fn start(&mut self, bind_address: String) -> Result<()>;

    /// Stop the request plane server gracefully
    async fn stop(&mut self) -> Result<()>;

    /// Get the public address where this server is accessible
    ///
    /// This is used for service discovery (e.g., registered in etcd)
    fn public_address(&self) -> String;
}

/// Message received from request plane
pub struct RequestMessage {
    /// Request payload (typically TwoPartCodec-encoded)
    pub payload: Bytes,
    /// Request headers for tracing, metadata, etc.
    pub headers: Headers,
}

impl RequestMessage {
    pub fn new(payload: Bytes, headers: Headers) -> Self {
        Self { payload, headers }
    }
}

