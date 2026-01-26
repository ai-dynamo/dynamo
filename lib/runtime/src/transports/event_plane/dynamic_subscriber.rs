// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamic subscriber that watches discovery and manages connections to multiple publishers.
//!
//! This module enables automatic discovery and connection to new publishers as they come online,
//! and cleanup of disconnected publishers.

use anyhow::Result;
use bytes::Bytes;
use futures::stream::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tokio_util::sync::CancellationToken;

use super::transport::{EventTransportRx, WireStream};
use super::zmq_transport::ZmqSubTransport;
use crate::discovery::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
    EventTransport,
};

/// Manages dynamic subscriptions to multiple publishers.
pub struct DynamicSubscriber {
    discovery: Arc<dyn Discovery>,
    query: DiscoveryQuery,
    topic: String,
    cancel_token: CancellationToken,
}

impl DynamicSubscriber {
    pub fn new(discovery: Arc<dyn Discovery>, query: DiscoveryQuery, topic: String) -> Self {
        Self {
            discovery,
            query,
            topic,
            cancel_token: CancellationToken::new(),
        }
    }

    /// Start watching discovery and create a merged stream of events.
    pub async fn start_zmq(self: Arc<Self>) -> Result<WireStream> {
        let (event_tx, event_rx) = mpsc::unbounded_channel::<Bytes>();

        // Track active endpoint connections with instance ID to endpoint mapping
        let active_endpoints: Arc<RwLock<HashMap<String, (String, CancellationToken)>>> =
            Arc::new(RwLock::new(HashMap::new()));

        // Clone self for the spawned task
        let subscriber_clone = Arc::clone(&self);

        // Spawn background task to watch discovery
        let discovery = Arc::clone(&self.discovery);
        let query = self.query.clone();
        // Use the actual topic for ZMQ native filtering (avoids decoding irrelevant messages)
        let zmq_topic = self.topic.clone();
        let cancel_token = self.cancel_token.clone();
        let endpoints = Arc::clone(&active_endpoints);

        tokio::spawn(async move {
            tracing::debug!(
                ?query,
                cancel_token_cancelled = cancel_token.is_cancelled(),
                "Attempting to start discovery watch"
            );

            // Don't pass the cancel token to list_and_watch - we'll handle cancellation ourselves
            let mut watch_stream = match discovery.list_and_watch(query.clone(), None).await {
                Ok(stream) => {
                    tracing::debug!("Successfully obtained discovery watch stream");
                    stream
                }
                Err(e) => {
                    tracing::error!(error = %e, "Failed to start discovery watch");
                    return;
                }
            };

            tracing::info!(?query, "Started dynamic discovery watch for ZMQ publishers");

            while let Some(event_result) = watch_stream.next().await {
                tracing::debug!("Received discovery event: {:?}", event_result);
                if cancel_token.is_cancelled() {
                    tracing::info!("Dynamic subscriber cancelled, stopping watch");
                    break;
                }

                match event_result {
                    Ok(DiscoveryEvent::Added(instance)) => {
                        tracing::info!(instance = ?instance, "Discovery Added event received");
                        let instance_id = instance.instance_id().to_string();

                        // Extract ZMQ endpoint from the instance
                        if let Some(endpoint) = Self::extract_zmq_endpoint(&instance) {
                            let mut endpoints_guard = endpoints.write().await;

                            // Skip if instance already tracked
                            if endpoints_guard.contains_key(&instance_id) {
                                tracing::debug!(endpoint = %endpoint, instance_id = %instance_id, "Already connected to ZMQ publisher");
                                continue;
                            }

                            tracing::info!(endpoint = %endpoint, instance_id = %instance_id, "Connecting to new ZMQ publisher");

                            // Create cancellation token for this endpoint's stream
                            let endpoint_cancel = CancellationToken::new();
                            endpoints_guard.insert(
                                instance_id.clone(),
                                (endpoint.clone(), endpoint_cancel.clone()),
                            );
                            drop(endpoints_guard);

                            // Spawn task to handle this endpoint's stream
                            let event_tx_clone = event_tx.clone();
                            let zmq_topic_clone = zmq_topic.clone();
                            let endpoint_clone = endpoint.clone();
                            let endpoints_clone = Arc::clone(&endpoints);
                            let instance_id_clone = instance_id.clone();

                            tokio::spawn(async move {
                                if let Err(e) = Self::consume_endpoint_stream(
                                    &endpoint_clone,
                                    &zmq_topic_clone,
                                    event_tx_clone,
                                    endpoint_cancel,
                                )
                                .await
                                {
                                    tracing::warn!(
                                        endpoint = %endpoint_clone,
                                        error = %e,
                                        "Error consuming ZMQ endpoint stream"
                                    );
                                }
                                // Clean up on stream termination
                                endpoints_clone.write().await.remove(&instance_id_clone);
                            });
                        } else {
                            tracing::warn!(
                                instance = ?instance,
                                "Discovery Added event did not contain a ZMQ endpoint"
                            );
                        }
                    }
                    Ok(DiscoveryEvent::Removed(instance_id)) => {
                        let id_str = instance_id.instance_id().to_string();
                        tracing::info!(
                            instance_id = %id_str,
                            "ZMQ publisher removed from discovery, cancelling endpoint stream"
                        );

                        // Cancel the endpoint's stream via its CancellationToken
                        if let Some((_endpoint, cancel)) = endpoints.write().await.remove(&id_str) {
                            cancel.cancel();
                            tracing::info!(instance_id = %id_str, "Cancelled endpoint stream");
                        } else {
                            tracing::warn!(instance_id = %id_str, "No active endpoint found for removed stream instance");
                        }
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "Discovery watch error");
                        break;
                    }
                }
            }

            // Cancel all active endpoints on shutdown
            let endpoints_guard = endpoints.write().await;
            for (_id, (_endpoint, cancel)) in endpoints_guard.iter() {
                cancel.cancel();
            }
            tracing::info!("Discovery watch stream ended");
        });

        // Return a stream that reads from the merged channel
        let stream = async_stream::stream! {
            // Keep subscriber_clone alive by capturing it in the stream
            let _subscriber = subscriber_clone;
            let mut rx = event_rx;
            while let Some(bytes) = rx.recv().await {
                yield Ok(bytes);
            }
        };

        Ok(Box::pin(stream))
    }

    /// Extract ZMQ endpoint from a discovery instance.
    fn extract_zmq_endpoint(instance: &DiscoveryInstance) -> Option<String> {
        if let DiscoveryInstance::EventChannel { transport, .. } = instance
            && let EventTransport::Zmq { endpoint } = transport
        {
            return Some(endpoint.clone());
        }
        None
    }

    /// Consume events from a single endpoint and forward to the merged channel.
    async fn consume_endpoint_stream(
        endpoint: &str,
        zmq_topic: &str,
        event_tx: mpsc::UnboundedSender<Bytes>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        // Connect to the endpoint
        let sub_transport = ZmqSubTransport::connect(endpoint, zmq_topic).await?;
        let mut stream = sub_transport.subscribe(zmq_topic).await?;

        tracing::info!(endpoint = %endpoint, topic = %zmq_topic, "Started consuming ZMQ endpoint stream");

        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    tracing::info!(endpoint = %endpoint, "Endpoint stream cancelled");
                    break;
                }

                event = stream.next() => {
                    match event {
                        Some(Ok(bytes)) => {
                            if event_tx.send(bytes).is_err() {
                                tracing::warn!(endpoint = %endpoint, "Event channel closed, stopping endpoint stream");
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            tracing::error!(
                                endpoint = %endpoint,
                                error = %e,
                                "Error receiving from ZMQ endpoint"
                            );
                            break;
                        }
                        None => {
                            tracing::info!(endpoint = %endpoint, "ZMQ endpoint stream ended");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Stop watching and disconnect from all endpoints.
    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }
}

impl Drop for DynamicSubscriber {
    fn drop(&mut self) {
        self.cancel_token.cancel();
    }
}
