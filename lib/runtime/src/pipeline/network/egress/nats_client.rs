// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NATS Request Plane Client
//!
//! Wraps the NATS client to implement the unified RequestPlaneClient trait,
//! providing a consistent interface across all transport types.

use super::unified_client::{ClientStats, Headers, RequestPlaneClient, mixed_request_plane_error};
use crate::error::{DynamoError, ErrorType};
use crate::metrics::transport_metrics::NATS_ERRORS_TOTAL;
use crate::pipeline::network::is_velo_request_address;
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
        if is_velo_request_address(&address) {
            tracing::error!(
                address = %address,
                "NATS request client received a foreign request-plane address; rejecting before publish"
            );
            return Err(mixed_request_plane_error("nats", &address));
        }

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

    #[tokio::test]
    async fn well_formed_velo_address_is_rejected_before_publish() {
        const SENTINEL_SUBJECT: &str = "dynamo.test.sentinel";

        let broker = FakeNats::start().await;
        let client = async_nats::connect(&broker.url).await.unwrap();
        let nats = NatsRequestClient::new(client.clone());
        let err = nats
            .send_request(
                "velo://3e0f0f0c-1a2b-4c3d-8e9f-0123456789ab/AA/1/ns/c/e".to_string(),
                Bytes::from_static(b"request"),
                Headers::new(),
            )
            .await
            .expect_err("a well-formed velo address must not be published to NATS");
        assert!(crate::error::match_error_chain(
            err.as_ref(),
            &[ErrorType::Internal],
            &[ErrorType::WorkerUnavailable]
        ));

        let sentinel = uuid::Uuid::new_v4().to_string();
        let reply = client
            .request_with_headers(
                SENTINEL_SUBJECT,
                async_nats::HeaderMap::new(),
                Bytes::copy_from_slice(sentinel.as_bytes()),
            )
            .await
            .expect("the sentinel request must round-trip through the broker");
        assert_eq!(reply.payload, Bytes::copy_from_slice(sentinel.as_bytes()));

        let commands = broker.commands();
        assert!(
            commands
                .iter()
                .any(|command| command.starts_with(&format!("PUB {SENTINEL_SUBJECT} "))),
            "the broker must see the sentinel request: {commands:?}"
        );
        assert!(
            commands
                .iter()
                .any(|command| command == &format!("BODY {sentinel}")),
            "the broker must echo the sentinel payload: {commands:?}"
        );
        assert!(
            !commands.iter().any(|command| (command.starts_with("PUB ")
                || command.starts_with("HPUB "))
                && command.contains("velo://")),
            "no publish may carry the well-formed velo address: {commands:?}"
        );
    }

    #[tokio::test]
    async fn velo_subject_keeps_its_nats_meaning() {
        let broker = FakeNats::start().await;
        let client = async_nats::connect(&broker.url).await.unwrap();
        let nats = NatsRequestClient::new(client);

        let response = nats
            .send_request(
                "velo://custom-service".to_string(),
                Bytes::from_static(b"request"),
                Headers::new(),
            )
            .await
            .expect("a velo:// prefix alone is a valid NATS subject");
        assert_eq!(response, Bytes::from_static(b"request"));
        assert!(
            broker
                .commands()
                .iter()
                .any(|command| command.starts_with("PUB velo://custom-service ")),
            "{:?}",
            broker.commands()
        );
    }

    /// A minimal NATS broker: enough of the protocol to complete the client
    /// handshake, echo a request, and record every command it reads.
    struct FakeNats {
        url: String,
        commands: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        server: tokio::task::JoinHandle<()>,
    }

    impl FakeNats {
        async fn start() -> Self {
            use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let commands = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let recorded = commands.clone();
            let server = tokio::spawn(async move {
                let (stream, _) = listener.accept().await.unwrap();
                let (read_half, mut write_half) = stream.into_split();
                let mut reader = BufReader::new(read_half);
                write_half
                    .write_all(
                        b"INFO {\"server_id\":\"test\",\"version\":\"2.10.0\",\"proto\":1,\"max_payload\":1048576}\r\n",
                    )
                    .await
                    .unwrap();

                let mut line = String::new();
                loop {
                    line.clear();
                    if reader.read_line(&mut line).await.unwrap() == 0 {
                        break;
                    }
                    recorded.lock().unwrap().push(line.trim_end().to_string());

                    let fields: Vec<&str> = line.split_whitespace().collect();
                    match fields.first().copied() {
                        Some("PING") => {
                            write_half.write_all(b"PONG\r\n").await.unwrap();
                            recorded.lock().unwrap().push("PONG".to_string());
                        }
                        Some("PUB") => {
                            let size: usize = fields.last().unwrap().parse().unwrap();
                            let mut body = vec![0u8; size + 2];
                            reader.read_exact(&mut body).await.unwrap();
                            recorded
                                .lock()
                                .unwrap()
                                .push(format!("BODY {}", String::from_utf8_lossy(&body[..size])));
                            if fields.len() == 4 {
                                let reply = fields[2];
                                write_half
                                    .write_all(format!("MSG {reply} 0 {size}\r\n").as_bytes())
                                    .await
                                    .unwrap();
                                write_half.write_all(&body[..size]).await.unwrap();
                                write_half.write_all(b"\r\n").await.unwrap();
                            }
                        }
                        Some("HPUB") => {
                            let total: usize = fields.last().unwrap().parse().unwrap();
                            let mut body = vec![0u8; total + 2];
                            reader.read_exact(&mut body).await.unwrap();
                            recorded
                                .lock()
                                .unwrap()
                                .push(format!("BODY {}", String::from_utf8_lossy(&body[..total])));
                            if fields.len() == 5 {
                                let reply = fields[2];
                                write_half
                                    .write_all(format!("MSG {reply} 0 {total}\r\n").as_bytes())
                                    .await
                                    .unwrap();
                                write_half.write_all(&body[..total]).await.unwrap();
                                write_half.write_all(b"\r\n").await.unwrap();
                            }
                        }
                        _ => {}
                    }
                }
            });

            Self {
                url: format!("nats://{addr}"),
                commands,
                server,
            }
        }

        fn commands(&self) -> Vec<String> {
            self.commands.lock().unwrap().clone()
        }
    }

    impl Drop for FakeNats {
        fn drop(&mut self) {
            self.server.abort();
        }
    }
}
