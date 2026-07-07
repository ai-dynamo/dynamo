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
use crate::config::environment_names::event_plane::DYN_ZMQ_EVENT_SUBSCRIBER_CHANNEL_CAPACITY;
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
        // Bounded merged channel. Many peer publishers (e.g. every other
        // frontend under replica-sync) feed this single-consumer channel; an
        // unbounded channel grows RSS without limit when the consumer can't keep
        // up (observed ~80 GiB/frontend at 168 frontends). Cap it and drop on
        // overflow — the event plane is already best-effort/lossy (ZMQ RCVHWM),
        // so a dropped event costs routing-estimate freshness, not correctness.
        // Configurable via DYN_ZMQ_EVENT_SUBSCRIBER_CHANNEL_CAPACITY (default
        // 100_000, matching ZMQ_RCVHWM).
        let channel_cap = std::env::var(DYN_ZMQ_EVENT_SUBSCRIBER_CHANNEL_CAPACITY)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(100_000);
        let (event_tx, event_rx) = mpsc::channel::<Bytes>(channel_cap);

        // One SUB socket can connect to many PUB endpoints. Keep publisher
        // identity in discovery/envelopes while avoiding one socket and one
        // pump task per publisher/subscriber pair.
        let sub_transport = Arc::new(ZmqSubTransport::dynamic(&self.topic, channel_cap).await?);
        let mut zmq_stream = sub_transport.subscribe(&self.topic).await?;

        // Keep the existing bounded merged-handoff semantics under a slow
        // consumer. The transport itself has a small queue feeding this task.
        let forward_cancel = self.cancel_token.clone();
        let event_tx_for_pump = event_tx.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = forward_cancel.cancelled() => break,
                    event = zmq_stream.next() => {
                        match event {
                            Some(Ok(bytes)) => match event_tx_for_pump.try_send(bytes) {
                                Ok(()) => {}
                                Err(mpsc::error::TrySendError::Full(_)) => {
                                    tracing::trace!("Event subscriber channel full; dropping event");
                                }
                                Err(mpsc::error::TrySendError::Closed(_)) => break,
                            },
                            Some(Err(error)) => {
                                tracing::error!(%error, "Error receiving from dynamic ZMQ socket");
                                break;
                            }
                            None => break,
                        }
                    }
                }
            }
        });

        // Track active publisher identities separately from endpoint ownership.
        // The endpoint refcount check preserves correctness even if discovery
        // ever reports two identities for one transport endpoint.
        let active_endpoints: Arc<RwLock<HashMap<DiscoveryInstanceId, String>>> =
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
        let sub_transport_for_watch = Arc::clone(&sub_transport);

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

            loop {
                let event_result = tokio::select! {
                    _ = cancel_token.cancelled() => {
                        tracing::info!("Dynamic subscriber cancelled, stopping watch");
                        break;
                    }
                    event = watch_stream.next() => {
                        let Some(event) = event else { break };
                        event
                    }
                };
                tracing::debug!("Received discovery event: {:?}", event_result);

                match event_result {
                    Ok(DiscoveryEvent::Added(instance)) => {
                        tracing::info!(instance = ?instance, "Discovery Added event received");
                        let instance_id = instance.id();

                        // Extract ZMQ endpoint from the instance
                        if let Some(endpoint) = Self::extract_zmq_endpoint(&instance, &zmq_topic) {
                            let should_connect = {
                                let mut endpoints_guard = endpoints.write().await;
                                if endpoints_guard.contains_key(&instance_id) {
                                    tracing::debug!(endpoint = %endpoint, ?instance_id, "Already connected to ZMQ publisher");
                                    continue;
                                }
                                let already_connected =
                                    endpoints_guard.values().any(|value| value == &endpoint);
                                endpoints_guard.insert(instance_id.clone(), endpoint.clone());
                                !already_connected
                            };

                            if should_connect {
                                tracing::info!(endpoint = %endpoint, ?instance_id, "Connecting shared ZMQ subscriber to publisher");
                                if let Err(error) =
                                    sub_transport_for_watch.connect_endpoint(&endpoint).await
                                {
                                    endpoints.write().await.remove(&instance_id);
                                    tracing::warn!(%endpoint, %error, "Failed to connect shared ZMQ subscriber");
                                }
                            }
                        } else {
                            tracing::debug!(
                                instance = ?instance,
                                expected_topic = %zmq_topic,
                                "Discovery event is not a matching ZMQ publisher"
                            );
                        }
                    }
                    Ok(DiscoveryEvent::Removed(instance_id)) => {
                        let is_expected_topic = matches!(
                            &instance_id,
                            DiscoveryInstanceId::EventChannel(channel_id)
                                if channel_id.topic == zmq_topic
                        );
                        if !is_expected_topic {
                            tracing::debug!(
                                ?instance_id,
                                expected_topic = %zmq_topic,
                                "Ignoring removal for unrelated event channel"
                            );
                            continue;
                        }

                        let removed = {
                            let mut endpoints_guard = endpoints.write().await;
                            let endpoint = endpoints_guard.remove(&instance_id);
                            endpoint.map(|endpoint| {
                                let still_used =
                                    endpoints_guard.values().any(|value| value == &endpoint);
                                (endpoint, !still_used)
                            })
                        };

                        if let Some((endpoint, should_disconnect)) = removed {
                            if should_disconnect
                                && let Err(error) =
                                    sub_transport_for_watch.disconnect_endpoint(&endpoint).await
                            {
                                tracing::warn!(%endpoint, %error, "Failed to disconnect removed ZMQ publisher");
                            }
                        } else {
                            tracing::debug!(
                                ?instance_id,
                                "No active endpoint found for removed stream instance"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "Discovery watch error");
                        break;
                    }
                }
            }

            // Disconnect each endpoint once on shutdown.
            let remaining: std::collections::HashSet<String> = endpoints
                .write()
                .await
                .drain()
                .map(|(_, endpoint)| endpoint)
                .collect();
            for endpoint in remaining {
                if let Err(error) = sub_transport_for_watch.disconnect_endpoint(&endpoint).await {
                    tracing::debug!(%endpoint, %error, "Failed to disconnect ZMQ endpoint during shutdown");
                }
            }
            tracing::info!("Discovery watch stream ended");
        });

        // Return a stream that reads from the merged channel
        let stream = async_stream::stream! {
            // Keep subscriber_clone alive by capturing it in the stream
            let _subscriber = subscriber_clone;
            let _sub_transport = sub_transport;
            let mut rx = event_rx;
            while let Some(bytes) = rx.recv().await {
                yield Ok(bytes);
            }
        };

        Ok(Box::pin(stream))
    }

    /// Extract ZMQ endpoint from a discovery instance.
    fn extract_zmq_endpoint(instance: &DiscoveryInstance, expected_topic: &str) -> Option<String> {
        if let DiscoveryInstance::EventChannel {
            topic, transport, ..
        } = instance
            && topic == expected_topic
            && let EventTransport::Zmq { endpoint } = transport
        {
            return Some(endpoint.clone());
        }
        None
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

#[cfg(test)]
mod tests {
    use super::*;

    fn event_channel(topic: &str, transport: EventTransport) -> DiscoveryInstance {
        DiscoveryInstance::EventChannel {
            namespace: "test-ns".to_string(),
            component: "test-component".to_string(),
            topic: topic.to_string(),
            instance_id: 1,
            transport,
        }
    }

    #[test]
    fn extracts_only_matching_zmq_topic() {
        let matching = event_channel("kv-events", EventTransport::zmq("tcp://127.0.0.1:1"));
        let wrong_topic = event_channel("kv-metrics", EventTransport::zmq("tcp://127.0.0.1:2"));
        let wrong_transport = event_channel(
            "kv-events",
            EventTransport::nats("namespace.test-ns.component.test-component"),
        );

        assert_eq!(
            DynamicSubscriber::extract_zmq_endpoint(&matching, "kv-events").as_deref(),
            Some("tcp://127.0.0.1:1")
        );
        assert_eq!(
            DynamicSubscriber::extract_zmq_endpoint(&wrong_topic, "kv-events"),
            None
        );
        assert_eq!(
            DynamicSubscriber::extract_zmq_endpoint(&wrong_transport, "kv-events"),
            None
        );
    }
}
