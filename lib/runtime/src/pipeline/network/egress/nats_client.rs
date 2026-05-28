// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NATS Request Plane Client
//!
//! Wraps the NATS client to implement the unified RequestPlaneClient trait,
//! providing a consistent interface across all transport types.

use super::unified_client::{ClientStats, Headers, RequestPlaneClient};
use crate::error::{DynamoError, ErrorType};
use crate::metrics::transport_metrics::NATS_ERRORS_TOTAL;
use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;

/// Client-facing message for oversized request payloads.
///
/// Intentionally generic: it must not leak the NATS subject, byte counts, the
/// `max_payload` limit, or deployment-configuration instructions. Internal
/// diagnostics (subject, payload_bytes, max_payload_bytes) live in logs and
/// the `nats_errors_total` metric instead.
const MAX_PAYLOAD_USER_MESSAGE: &str = "Request payload is too large for this deployment. Reduce the input size or metadata size and retry.";

/// NATS implementation of RequestPlaneClient
///
/// This client wraps the async_nats::Client and adapts it to the
/// unified RequestPlaneClient interface.
pub struct NatsRequestClient {
    client: async_nats::Client,
    max_payload: usize,
}

impl NatsRequestClient {
    /// Create a new NATS request client
    ///
    /// # Arguments
    ///
    /// * `client` - The underlying NATS client
    pub fn new(client: async_nats::Client) -> Self {
        // Snapshot the server-advertised max_payload at construction time. This is
        // not re-read on reconnect, so if the client later reconnects to a server
        // advertising a smaller limit, oversized payloads fall through to the
        // server's own enforcement rather than being caught by the early check below.
        let max_payload = client.server_info().max_payload;
        Self {
            client,
            max_payload,
        }
    }

    fn max_payload_error() -> DynamoError {
        DynamoError::builder()
            .error_type(ErrorType::InvalidArgument)
            .message(MAX_PAYLOAD_USER_MESSAGE)
            .build()
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
        // Best-effort early rejection. This compares the payload body only, not the
        // header/subject framing that also counts toward the server's max_payload for
        // header messages (HPUB). It is therefore a lower bound: it reliably catches
        // oversized payloads without ever false-rejecting a valid request, while
        // borderline messages inflated by large headers are still caught server-side.
        let payload_len = payload.len();
        let max_payload = self.max_payload;
        if max_payload > 0 && payload_len > max_payload {
            NATS_ERRORS_TOTAL
                .with_label_values(&["max_payload_exceeded"])
                .inc();
            tracing::error!(
                address = %address,
                payload_bytes = payload_len,
                max_payload_bytes = max_payload,
                "NATS request payload exceeds server max_payload; rejecting before send"
            );
            return Err(anyhow::anyhow!(Self::max_payload_error()));
        }

        // Convert generic headers to NATS headers
        let mut nats_headers = async_nats::HeaderMap::new();
        for (key, value) in headers {
            nats_headers.insert(key.as_str(), value.as_str());
        }

        // Send request with headers
        let response = self
            .client
            .request_with_headers(address.clone(), nats_headers, payload)
            .await
            .map_err(|e| {
                NATS_ERRORS_TOTAL
                    .with_label_values(&["request_failed"])
                    .inc();
                anyhow::anyhow!(
                    DynamoError::builder()
                        .error_type(ErrorType::CannotConnect)
                        .message(format!("NATS request to {address} failed"))
                        .cause(e)
                        .build()
                )
            })?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_payload_error_is_user_safe() {
        let err = NatsRequestClient::max_payload_error();

        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        assert!(err.message().contains("Request payload is too large"));
        assert!(!err.message().contains("NATS"));
        assert!(!err.message().contains("max_payload"));
        assert!(!err.message().contains("payload_bytes"));
        assert!(!err.message().contains("nats-server"));
    }
}
