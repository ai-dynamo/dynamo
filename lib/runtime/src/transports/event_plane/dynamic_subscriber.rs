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

        // Track active endpoint connections. A single runtime instance can
        // publish the same topic through multiple ZMQ endpoints, for example
        // one KV-event stream per attention-DP rank. Key by both instance and
        // endpoint so those streams are all consumed.
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
                            let endpoint_key = Self::endpoint_key(&instance_id, &endpoint);
                            let mut endpoints_guard = endpoints.write().await;

                            // Skip if this exact instance/endpoint stream is already tracked.
                            if endpoints_guard.contains_key(&endpoint_key) {
                                tracing::debug!(endpoint = %endpoint, instance_id = %instance_id, "Already connected to ZMQ publisher");
                                continue;
                            }

                            tracing::info!(endpoint = %endpoint, instance_id = %instance_id, "Connecting to new ZMQ publisher");

                            // Create cancellation token for this endpoint's stream
                            let endpoint_cancel = CancellationToken::new();
                            endpoints_guard.insert(
                                endpoint_key.clone(),
                                (endpoint.clone(), endpoint_cancel.clone()),
                            );
                            drop(endpoints_guard);

                            // Spawn task to handle this endpoint's stream
                            let event_tx_clone = event_tx.clone();
                            let zmq_topic_clone = zmq_topic.clone();
                            let endpoint_clone = endpoint.clone();
                            let endpoints_clone = Arc::clone(&endpoints);
                            let endpoint_key_clone = endpoint_key.clone();

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
                                endpoints_clone.write().await.remove(&endpoint_key_clone);
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

                        // Cancel every endpoint stream for this runtime instance.
                        let removed = {
                            let mut endpoints_guard = endpoints.write().await;
                            let keys = endpoints_guard
                                .keys()
                                .filter(|key| Self::key_belongs_to_instance(key, &id_str))
                                .cloned()
                                .collect::<Vec<_>>();

                            let mut removed = Vec::with_capacity(keys.len());
                            for key in keys {
                                if let Some((endpoint, cancel)) = endpoints_guard.remove(&key) {
                                    removed.push((endpoint, cancel));
                                }
                            }
                            removed
                        };

                        if removed.is_empty() {
                            tracing::warn!(instance_id = %id_str, "No active endpoint found for removed stream instance");
                        } else {
                            for (endpoint, cancel) in removed {
                                cancel.cancel();
                                tracing::info!(
                                    endpoint = %endpoint,
                                    instance_id = %id_str,
                                    "Cancelled endpoint stream"
                                );
                            }
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

    fn endpoint_key(instance_id: &str, endpoint: &str) -> String {
        format!("{instance_id}|{endpoint}")
    }

    fn key_belongs_to_instance(endpoint_key: &str, instance_id: &str) -> bool {
        endpoint_key
            .strip_prefix(instance_id)
            .is_some_and(|rest| rest.starts_with('|'))
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
