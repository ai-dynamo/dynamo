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
        Self::with_cancel_token(discovery, query, topic, CancellationToken::new())
    }

    pub fn with_cancel_token(
        discovery: Arc<dyn Discovery>,
        query: DiscoveryQuery,
        topic: String,
        cancel_token: CancellationToken,
    ) -> Self {
        Self {
            discovery,
            query,
            topic,
            cancel_token,
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

        // Track active endpoint connections with instance ID to endpoint mapping
        let active_endpoints: Arc<
            RwLock<HashMap<DiscoveryInstanceId, (String, CancellationToken)>>,
        > = Arc::new(RwLock::new(HashMap::new()));

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

            // Pass cancellation through so the discovery backend can stop any
            // task that it owns in addition to the consumer loop below.
            let mut watch_stream = match discovery
                .list_and_watch(query.clone(), Some(cancel_token.clone()))
                .await
            {
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
                    biased;

                    _ = cancel_token.cancelled() => {
                        tracing::info!("Dynamic subscriber cancelled, stopping watch");
                        break;
                    }
                    result = watch_stream.next() => match result {
                        Some(result) => result,
                        None => break,
                    },
                };

                tracing::debug!("Received discovery event: {:?}", event_result);

                match event_result {
                    Ok(DiscoveryEvent::Added(instance)) => {
                        tracing::info!(instance = ?instance, "Discovery Added event received");
                        let instance_id = instance.id();

                        // Extract ZMQ endpoint from the instance
                        if let Some(endpoint) = Self::extract_zmq_endpoint(&instance, &zmq_topic) {
                            let mut endpoints_guard = endpoints.write().await;

                            // Skip if instance already tracked
                            if endpoints_guard.contains_key(&instance_id) {
                                tracing::debug!(endpoint = %endpoint, ?instance_id, "Already connected to ZMQ publisher");
                                continue;
                            }

                            tracing::info!(endpoint = %endpoint, ?instance_id, "Connecting to new ZMQ publisher");

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

                        tracing::info!(
                            ?instance_id,
                            "ZMQ publisher removed from discovery, cancelling endpoint stream"
                        );

                        // Cancel the endpoint's stream via its CancellationToken
                        if let Some((_endpoint, cancel)) =
                            endpoints.write().await.remove(&instance_id)
                        {
                            cancel.cancel();
                            tracing::info!(?instance_id, "Cancelled endpoint stream");
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

    /// Consume events from a single endpoint and forward to the merged channel.
    async fn consume_endpoint_stream(
        endpoint: &str,
        zmq_topic: &str,
        event_tx: mpsc::Sender<Bytes>,
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
                            match event_tx.try_send(bytes) {
                                Ok(()) => {}
                                Err(mpsc::error::TrySendError::Full(_)) => {
                                    // Consumer is behind; drop to bound memory.
                                    // Best-effort plane — a stale estimate
                                    // self-corrects on subsequent events.
                                    tracing::trace!(endpoint = %endpoint, "Event subscriber channel full; dropping event");
                                }
                                Err(mpsc::error::TrySendError::Closed(_)) => {
                                    tracing::warn!(endpoint = %endpoint, "Event channel closed, stopping endpoint stream");
                                    break;
                                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{DiscoverySpec, DiscoveryStream, EventChannelQuery};
    use tokio::sync::Notify;
    use tokio::time::{Duration, timeout};

    struct CancellationAwareDiscovery {
        backend_stopped: Arc<Notify>,
    }

    #[async_trait::async_trait]
    impl Discovery for CancellationAwareDiscovery {
        fn instance_id(&self) -> u64 {
            1
        }

        async fn register_internal(&self, _spec: DiscoverySpec) -> Result<DiscoveryInstance> {
            anyhow::bail!("register is not supported by this test discovery")
        }

        async fn unregister(&self, _instance: DiscoveryInstance) -> Result<()> {
            anyhow::bail!("unregister is not supported by this test discovery")
        }

        async fn list(&self, _query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>> {
            Ok(Vec::new())
        }

        async fn list_and_watch(
            &self,
            _query: DiscoveryQuery,
            cancel_token: Option<CancellationToken>,
        ) -> Result<DiscoveryStream> {
            let cancel_token = cancel_token
                .ok_or_else(|| anyhow::anyhow!("dynamic subscriber must pass cancellation"))?;
            let backend_stopped = Arc::clone(&self.backend_stopped);
            tokio::spawn(async move {
                cancel_token.cancelled().await;
                backend_stopped.notify_one();
            });

            Ok(Box::pin(futures::stream::pending()))
        }
    }

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

    #[tokio::test]
    async fn cancellation_stops_idle_discovery_watch() {
        let backend_stopped = Arc::new(Notify::new());
        let discovery = Arc::new(CancellationAwareDiscovery {
            backend_stopped: Arc::clone(&backend_stopped),
        });
        let query = DiscoveryQuery::EventChannels(EventChannelQuery::topic(
            "test-ns",
            "test-component",
            "kv-events",
        ));
        let subscriber = Arc::new(DynamicSubscriber::new(
            discovery,
            query,
            "kv-events".to_string(),
        ));
        let mut stream = Arc::clone(&subscriber).start_zmq().await.unwrap();

        tokio::task::yield_now().await;
        subscriber.cancel();

        let next = timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("subscriber stream should close promptly after cancellation");
        assert!(next.is_none());

        timeout(Duration::from_secs(1), backend_stopped.notified())
            .await
            .expect("discovery backend should receive cancellation");
    }

    #[tokio::test]
    async fn dropping_returned_stream_cancels_idle_discovery_watch() {
        let backend_stopped = Arc::new(Notify::new());
        let discovery = Arc::new(CancellationAwareDiscovery {
            backend_stopped: Arc::clone(&backend_stopped),
        });
        let query = DiscoveryQuery::EventChannels(EventChannelQuery::topic(
            "test-ns",
            "test-component",
            "kv-events",
        ));
        let parent_token = CancellationToken::new();
        let subscriber = Arc::new(DynamicSubscriber::with_cancel_token(
            discovery,
            query,
            "kv-events".to_string(),
            parent_token.child_token(),
        ));
        let weak_subscriber = Arc::downgrade(&subscriber);
        let stream = subscriber.start_zmq().await.unwrap();

        assert!(
            weak_subscriber.upgrade().is_some(),
            "returned stream should retain the dynamic subscriber"
        );
        drop(stream);
        assert!(
            weak_subscriber.upgrade().is_none(),
            "dropping the returned stream should release the dynamic subscriber"
        );
        assert!(
            !parent_token.is_cancelled(),
            "dropping a subscriber must not cancel its parent token"
        );

        timeout(Duration::from_secs(1), backend_stopped.notified())
            .await
            .expect("dropping the stream should cancel the discovery backend");
    }
}
