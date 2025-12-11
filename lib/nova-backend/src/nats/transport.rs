// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NATS.io transport with publish-subscribe messaging
//!
//! This implementation provides a NATS transport where:
//! - Header bytes are base64-encoded in X-Transport-Header NATS header
//! - Payload bytes are sent as raw bytes in NATS message body
//! - Subjects follow pattern: nova.{instance_id_b58}.{message_type}
//! - All messages use publish() for fire-and-forget semantics
//! - Health checks use request() to detect no responders

use anyhow::{Context, Result};
use base64::Engine;
use bytes::Bytes;
use dashmap::DashMap;
use dynamo_identity::InstanceId;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::transport::{HealthCheckError, TransportError, TransportErrorHandler};
use crate::{MessageType, PeerInfo, Transport, TransportAdapter, TransportKey, WorkerAddress};

#[cfg(feature = "nats")]
use async_nats::{Client, Message as NatsMessage};

/// Encode an InstanceId to Base58 string
fn encode_instance_id_b58(instance_id: InstanceId) -> String {
    bs58::encode(instance_id.as_bytes()).into_string()
}

/// Compute subject prefix for an instance
fn compute_subject_prefix(instance_id: InstanceId) -> String {
    format!("nova.{}", encode_instance_id_b58(instance_id))
}

const TRANSPORT_CONTROL_HEADER: &str = "X-Transport-Header";

/// Task for publishing to NATS
struct PublishTask {
    subject: String,
    headers: async_nats::HeaderMap,
    payload: Bytes,
    on_error: Arc<dyn TransportErrorHandler>,
}

/// NATS transport with publish-subscribe messaging
///
/// This transport uses NATS subjects for routing messages:
/// - Messages: nova.{instance_id_b58}.message
/// - Responses: nova.{instance_id_b58}.response
/// - Events: nova.{instance_id_b58}.event
/// - Health: nova.{instance_id_b58}.health
pub struct NatsTransport {
    key: TransportKey,
    nats_url: String,
    instance_id: OnceLock<InstanceId>,
    local_address: OnceLock<WorkerAddress>,
    client: OnceLock<Client>,
    peers: Arc<DashMap<InstanceId, String>>, // instance_id -> subject prefix
    cancel_token: CancellationToken,

    // Publish queue
    publish_tx: flume::Sender<PublishTask>,
    publish_rx: Arc<parking_lot::Mutex<Option<flume::Receiver<PublishTask>>>>,
}

impl NatsTransport {
    /// Create a new NATS transport
    fn new(key: TransportKey, nats_url: String, nats_client: Option<Client>) -> Self {
        let mut addr_builder = WorkerAddress::builder();
        addr_builder
            .add_entry(key.clone(), nats_url.as_bytes().to_vec())
            .expect("Failed to build WorkerAddress");

        let local_address = addr_builder.build().expect("Failed to build WorkerAddress");

        // Create publish queue channel
        let (publish_tx, publish_rx) = flume::unbounded();

        let local_address_lock = OnceLock::new();
        local_address_lock.set(local_address).ok();

        let client_lock = OnceLock::new();
        if let Some(client) = nats_client {
            client_lock.set(client).ok();
        }

        Self {
            key,
            nats_url,
            instance_id: OnceLock::new(),
            local_address: local_address_lock,
            client: client_lock,
            peers: Arc::new(DashMap::new()),
            cancel_token: CancellationToken::new(),
            publish_tx,
            publish_rx: Arc::new(parking_lot::Mutex::new(Some(publish_rx))),
        }
    }
}

impl Transport for NatsTransport {
    fn key(&self) -> TransportKey {
        self.key.clone()
    }

    fn address(&self) -> WorkerAddress {
        self.local_address
            .get()
            .cloned()
            .expect("local_address should be set during construction")
    }

    fn register(&self, peer_info: PeerInfo) -> Result<(), TransportError> {
        // Get NATS URL from peer's address
        let endpoint = peer_info
            .worker_address()
            .get_entry(&self.key)
            .map_err(|_| TransportError::NoEndpoint)?
            .ok_or(TransportError::NoEndpoint)?;

        // Parse NATS URL
        let nats_url = String::from_utf8(endpoint.to_vec()).map_err(|e| {
            error!("Failed to parse NATS endpoint as UTF-8: {}", e);
            TransportError::InvalidEndpoint
        })?;

        // Validate URL format (basic check)
        if !nats_url.starts_with("nats://") && !nats_url.starts_with("nats://") {
            warn!("NATS URL doesn't start with nats://: {}", nats_url);
        }

        // Compute subject prefix from peer's instance_id
        let subject_prefix = compute_subject_prefix(peer_info.instance_id());
        self.peers
            .insert(peer_info.instance_id(), subject_prefix.clone());

        debug!(
            "Registered peer {} with subject prefix {}",
            peer_info.instance_id(),
            subject_prefix
        );

        Ok(())
    }

    fn send_message(
        &self,
        instance_id: InstanceId,
        header: Vec<u8>,
        payload: Vec<u8>,
        message_type: MessageType,
        on_error: Arc<dyn TransportErrorHandler>,
    ) {
        // Look up peer subject prefix
        let subject_prefix = match self.peers.get(&instance_id) {
            Some(prefix) => prefix.clone(),
            None => {
                let header_bytes = Bytes::from(header);
                let payload_bytes = Bytes::from(payload);
                on_error.on_error(
                    header_bytes,
                    payload_bytes,
                    format!("peer not registered: {}", instance_id),
                );
                return;
            }
        };

        // Determine subject suffix based on message type
        let subject_suffix = match message_type {
            MessageType::Message => "message",
            MessageType::Response => "response",
            MessageType::Ack | MessageType::Event => "event",
        };

        let subject = format!("{}.{}", subject_prefix, subject_suffix);

        // Convert to Bytes early so we can clone if needed
        let header_bytes = Bytes::from(header);
        let payload_bytes = Bytes::from(payload);

        // Base64 encode header for NATS headers
        let header_b64 = base64::engine::general_purpose::STANDARD.encode(&header_bytes);

        // Create headers map
        let mut headers = async_nats::HeaderMap::new();
        headers.insert(TRANSPORT_CONTROL_HEADER, header_b64);

        // Create publish task
        let task = PublishTask {
            subject,
            headers,
            payload: payload_bytes.clone(),
            on_error: on_error.clone(),
        };

        // Send on unbounded channel - if disconnected, report error
        if self.publish_tx.send(task).is_err() {
            on_error.on_error(
                header_bytes,
                payload_bytes,
                "NATS transport is shutting down".to_string(),
            );
        }
    }

    fn start(
        &self,
        instance_id: InstanceId,
        channels: TransportAdapter,
        rt: tokio::runtime::Handle,
    ) -> futures::future::BoxFuture<'_, anyhow::Result<()>> {
        let nats_url = self.nats_url.clone();
        let client_opt = self.client.get().cloned();
        let cancel_token = self.cancel_token.clone();
        let publish_rx = match self.publish_rx.lock().take() {
            Some(rx) => rx,
            None => {
                return Box::pin(async { Err(anyhow::anyhow!("Transport already started")) });
            }
        };

        Box::pin(async move {
            info!("Starting NATS transport on {}", nats_url);

            // Connect to NATS if not provided
            let client = match client_opt {
                Some(c) => c,
                None => {
                    tokio::time::timeout(Duration::from_secs(5), async_nats::connect(&nats_url))
                        .await
                        .context("NATS connection timed out after 5 seconds")??
                }
            };

            // Compute local subject prefix
            let local_prefix = compute_subject_prefix(instance_id);

            // Subscribe to all subjects
            let message_sub = client
                .subscribe(format!("{}.message", local_prefix))
                .await
                .context("Failed to subscribe to message subject")?;

            let response_sub = client
                .subscribe(format!("{}.response", local_prefix))
                .await
                .context("Failed to subscribe to response subject")?;

            let event_sub = client
                .subscribe(format!("{}.event", local_prefix))
                .await
                .context("Failed to subscribe to event subject")?;

            let health_sub = client
                .subscribe(format!("{}.health", local_prefix))
                .await
                .context("Failed to subscribe to health subject")?;

            // Spawn subscription handlers using provided runtime
            rt.spawn(handle_subscription(
                message_sub,
                channels.message_stream.clone(),
                cancel_token.clone(),
            ));

            rt.spawn(handle_subscription(
                response_sub,
                channels.response_stream.clone(),
                cancel_token.clone(),
            ));

            rt.spawn(handle_subscription(
                event_sub,
                channels.event_stream.clone(),
                cancel_token.clone(),
            ));

            rt.spawn(handle_health_subscription(
                health_sub,
                client.clone(),
                cancel_token.clone(),
            ));

            // Spawn sender task using provided runtime
            rt.spawn(sender_task(
                publish_rx,
                client.clone(),
                cancel_token.clone(),
            ));

            // Store instance_id and client
            self.instance_id.set(instance_id).ok();
            self.client.set(client).ok();

            info!(
                "NATS transport started with subject prefix: {}",
                local_prefix
            );

            Ok(())
        })
    }

    fn shutdown(&self) {
        info!("Shutting down NATS transport");

        // Cancel all tasks in the dedicated thread
        self.cancel_token.cancel();

        // Give tasks time to exit gracefully
        std::thread::sleep(std::time::Duration::from_millis(200));

        // NATS client will disconnect when the transport is dropped

        // Clear peers
        self.peers.clear();

        info!("NATS transport shut down");
    }

    fn check_health(
        &self,
        instance_id: InstanceId,
        timeout: Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), HealthCheckError>> + Send + '_>,
    > {
        Box::pin(async move {
            let subject_prefix = self
                .peers
                .get(&instance_id)
                .ok_or(HealthCheckError::PeerNotRegistered)?
                .clone();

            let health_subject = format!("{}.health", subject_prefix);

            let client = self
                .client
                .get()
                .ok_or(HealthCheckError::TransportNotStarted)?;

            // NATS is connectionless/brokered - we can't distinguish between
            // "never connected" and "connection failed" since there's no direct connection.
            // We always use ConnectionFailed for any failure.
            match tokio::time::timeout(timeout, client.request(health_subject, Bytes::new())).await
            {
                Ok(Ok(_)) => Ok(()),
                Ok(Err(_)) => Err(HealthCheckError::ConnectionFailed),
                Err(_) => Err(HealthCheckError::Timeout),
            }
        })
    }
}

/// Sender task that drains the publish queue and sends messages to NATS
async fn sender_task(
    rx: flume::Receiver<PublishTask>,
    client: Client,
    cancel_token: CancellationToken,
) {
    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                debug!("Sender task cancelled");
                break;
            }
            result = rx.recv_async() => {
                match result {
                    Ok(task) => {
                        if let Err(e) = client
                            .publish_with_headers(task.subject, task.headers, task.payload.clone())
                            .await
                        {
                            warn!("NATS publish failed: {}", e);
                            // Call on_error handler
                            task.on_error.on_error(
                                Bytes::new(), // We don't have the original header bytes
                                task.payload,
                                format!("NATS publish failed: {}", e),
                            );
                        }
                    }
                    Err(_) => {
                        debug!("Publish channel closed, sender task exiting");
                        break;
                    }
                }
            }
        }
    }
}

/// Handle subscription for message/response/event subjects
async fn handle_subscription(
    mut subscription: async_nats::Subscriber,
    channel: flume::Sender<(Bytes, Bytes)>,
    cancel_token: CancellationToken,
) {
    use futures::StreamExt;
    loop {
        tokio::select! {
            Some(msg) = subscription.next() => {
                // Extract header from NATS headers
                let header = match extract_header(&msg) {
                    Ok(h) => h,
                    Err(e) => {
                        error!("Failed to extract header from NATS message: {}", e);
                        continue;
                    }
                };

                let payload = Bytes::from(msg.payload.to_vec());

                // Send to channel
                if let Err(e) = channel.send_async((header, payload)).await {
                    error!("Failed to send message to channel: {}", e);
                    break;
                }
            }
            _ = cancel_token.cancelled() => {
                debug!("Subscription cancelled");
                break;
            }
        }
    }

    // Explicitly unsubscribe to clean up server-side state
    if let Err(e) = subscription.unsubscribe().await {
        warn!("Failed to unsubscribe: {}", e);
    }
}

/// Handle health check subscription
async fn handle_health_subscription(
    mut subscription: async_nats::Subscriber,
    client: Client,
    cancel_token: CancellationToken,
) {
    use futures::StreamExt;
    loop {
        tokio::select! {
            Some(msg) = subscription.next() => {
                // Health check request - send empty ack to reply subject
                if let Some(reply) = msg.reply
                    && let Err(e) = client.publish(reply, Bytes::new()).await
                {
                    warn!("Failed to send health check ack: {}", e);
                }
            }
            _ = cancel_token.cancelled() => {
                debug!("Health subscription cancelled");
                break;
            }
        }
    }

    // Explicitly unsubscribe to clean up server-side state
    if let Err(e) = subscription.unsubscribe().await {
        warn!("Failed to unsubscribe health: {}", e);
    }
}

/// Extract header from NATS message headers
fn extract_header(msg: &NatsMessage) -> Result<Bytes, anyhow::Error> {
    let headers = msg
        .headers
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Message has no headers"))?;

    let b64_header = headers
        .get(TRANSPORT_CONTROL_HEADER)
        .ok_or_else(|| anyhow::anyhow!("Missing X-Transport-Header"))?;

    let header_bytes = base64::engine::general_purpose::STANDARD
        .decode(b64_header)
        .context("Failed to decode base64 header")?;

    Ok(Bytes::from(header_bytes))
}

/// Builder for NatsTransport
pub struct NatsTransportBuilder {
    nats_url: Option<String>,
    nats_client: Option<Client>,
    key: Option<TransportKey>,
}

impl NatsTransportBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            nats_url: None,
            nats_client: None,
            key: None,
        }
    }

    /// Set the NATS server URL
    pub fn nats_url(mut self, url: impl Into<String>) -> Self {
        self.nats_url = Some(url.into());
        self
    }

    /// Set a pre-configured NATS client
    pub fn nats_client(mut self, client: Client) -> Self {
        self.nats_client = Some(client);
        self
    }

    /// Set the transport key
    pub fn key(mut self, key: TransportKey) -> Self {
        self.key = Some(key);
        self
    }

    /// Build the NatsTransport
    pub fn build(self) -> Result<NatsTransport> {
        // Either nats_url or nats_client required
        let nats_url = if let Some(client) = self.nats_client {
            // If client provided, use placeholder URL (won't be used)
            return Ok(NatsTransport::new(
                self.key.unwrap_or_else(|| TransportKey::from("nats")),
                "nats://localhost:4222".to_string(),
                Some(client),
            ));
        } else {
            self.nats_url
                .unwrap_or_else(|| "nats://localhost:4222".to_string())
        };

        let key = self.key.unwrap_or_else(|| TransportKey::from("nats"));

        Ok(NatsTransport::new(key, nats_url, None))
    }
}

impl Default for NatsTransportBuilder {
    fn default() -> Self {
        Self::new()
    }
}
