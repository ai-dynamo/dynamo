// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NATS Request Plane Client
//!
//! Wraps the NATS client to implement the unified RequestPlaneClient trait,
//! providing a consistent interface across all transport types.

use super::unified_client::{ClientStats, Headers, RequestPlaneClient};
use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;

/// NATS implementation of RequestPlaneClient
///
/// This client wraps the async_nats::Client and adapts it to the
/// unified RequestPlaneClient interface.
pub struct NatsRequestClient {
    client: async_nats::Client,
}

impl NatsRequestClient {
    /// Create a new NATS request client
    ///
    /// # Arguments
    ///
    /// * `client` - The underlying NATS client
    pub fn new(client: async_nats::Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl RequestPlaneClient for NatsRequestClient {
    async fn send_request(
        &self,
        address: String,
        payload: Bytes,
        headers: Headers,
    ) -> Result<Bytes> {
        // Convert generic headers to NATS headers
        let mut nats_headers = async_nats::HeaderMap::new();
        for (key, value) in headers {
            nats_headers.insert(key.as_str(), value.as_str());
        }

        // Send request with headers
        let response = self
            .client
            .request_with_headers(address, nats_headers, payload)
            .await
            .map_err(|e| anyhow::anyhow!("NATS request failed: {}", e))?;

        Ok(response.payload)
    }

    fn transport_name(&self) -> &'static str {
        "nats"
    }

    fn is_healthy(&self) -> bool {
        // Check if NATS client is connected
        // NATS client doesn't expose connection state directly, assume healthy
        true
    }

    fn stats(&self) -> ClientStats {
        // NATS client doesn't expose detailed stats
        // Return basic health indicator
        ClientStats {
            requests_sent: 0,
            responses_received: 0,
            errors: 0,
            bytes_sent: 0,
            bytes_received: 0,
            active_connections: if self.is_healthy() { 1 } else { 0 },
            idle_connections: 0,
            avg_latency_us: 0,
        }
    }

    async fn close(&self) -> Result<()> {
        // NATS client doesn't have an explicit close method
        // Connection is managed by the client lifecycle
        Ok(())
    }
}
