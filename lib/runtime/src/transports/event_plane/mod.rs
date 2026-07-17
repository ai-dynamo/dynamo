// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic Event Plane for transport-agnostic pub/sub communication.

mod codec;
mod dynamic_subscriber;
mod frame;
mod nats_transport;
mod traits;
mod transport;
pub mod zmq_transport;

pub use codec::{Codec, MsgpackCodec};
pub use dynamic_subscriber::DynamicSubscriber;
pub use frame::{FRAME_HEADER_SIZE, FRAME_VERSION, Frame, FrameError, FrameHeader};
pub use traits::{EventEnvelope, EventStream, TypedEventStream};
pub use transport::{EventTransportRx, EventTransportTx, WireStream};
pub use zmq_transport::{ZmqPubTransport, ZmqSubTransport};

// Re-export transport kind from discovery for convenience
pub use crate::discovery::{EventScope, EventTransportKind};

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use lru::LruCache;
use rand::TryRngCore;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::DistributedRuntime;
use crate::component::{Component, Endpoint, Namespace};
use crate::discovery::{
    Discovery, DiscoveryInstance, DiscoveryQuery, DiscoverySpec, EventChannelQuery, EventTransport,
};
use crate::protocols::EndpointId;
use crate::traits::DistributedRuntimeProvider;
use crate::utils::local_ip_for_advertise;

// ============================================================================
// Broker Resolution Logic
// ============================================================================

/// Broker endpoints for ZMQ broker mode
#[derive(Debug, Clone)]
struct BrokerEndpoints {
    xsub_endpoints: Vec<String>,
    xpub_endpoints: Vec<String>,
}

/// Resolve ZMQ broker endpoints from environment or discovery
/// Returns None if broker mode is not configured (direct mode)
async fn resolve_zmq_broker(
    drt: &DistributedRuntime,
    scope: &EventScope,
) -> Result<Option<BrokerEndpoints>> {
    // Priority 1: Explicit URL from DYN_ZMQ_BROKER_URL
    if let Ok(broker_url) =
        std::env::var(crate::config::environment_names::zmq_broker::DYN_ZMQ_BROKER_URL)
    {
        let (xsub_endpoints, xpub_endpoints) = parse_broker_url(&broker_url)?;
        tracing::info!(
            num_xsub = xsub_endpoints.len(),
            num_xpub = xpub_endpoints.len(),
            "Using explicit ZMQ broker URL"
        );
        return Ok(Some(BrokerEndpoints {
            xsub_endpoints,
            xpub_endpoints,
        }));
    }

    // Priority 2: Discovery-based lookup if DYN_ZMQ_BROKER_ENABLED is truthy
    if crate::config::env_is_truthy(
        crate::config::environment_names::zmq_broker::DYN_ZMQ_BROKER_ENABLED,
    ) {
        let query = DiscoveryQuery::EventChannels(EventChannelQuery::component(
            scope.namespace().to_string(),
            "zmq_broker".to_string(),
        ));

        let instances = drt.discovery().list(query).await?;

        // Collect all broker instances (multiple brokers for HA)
        let mut xsub_endpoints = Vec::new();
        let mut xpub_endpoints = Vec::new();

        for instance in instances {
            if let DiscoveryInstance::EventChannel { transport, .. } = instance
                && let EventTransport::ZmqBroker {
                    xsub_endpoints: xsubs,
                    xpub_endpoints: xpubs,
                } = transport
            {
                xsub_endpoints.extend(xsubs);
                xpub_endpoints.extend(xpubs);
            }
        }

        if xsub_endpoints.is_empty() {
            anyhow::bail!(
                "DYN_ZMQ_BROKER_ENABLED is set but no broker found in discovery for namespace '{}'",
                scope.namespace()
            );
        }

        tracing::info!(
            num_brokers = xsub_endpoints.len(),
            "Discovered ZMQ brokers from discovery plane"
        );

        return Ok(Some(BrokerEndpoints {
            xsub_endpoints,
            xpub_endpoints,
        }));
    }

    // No broker configured - use direct mode
    Ok(None)
}

/// Parse broker URL format: "xsub=tcp://host1:5555;tcp://host2:5555 , xpub=tcp://host1:5556;tcp://host2:5556"
fn parse_broker_url(url: &str) -> Result<(Vec<String>, Vec<String>)> {
    let parts: Vec<&str> = url.split(',').map(|s| s.trim()).collect();
    if parts.len() != 2 {
        anyhow::bail!(
            "Invalid broker URL format. Expected 'xsub=<urls> , xpub=<urls>', got: {}",
            url
        );
    }

    let mut xsub_endpoints = Vec::new();
    let mut xpub_endpoints = Vec::new();

    for part in parts {
        if let Some(urls_str) = part.strip_prefix("xsub=") {
            xsub_endpoints = urls_str
                .split(';')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        } else if let Some(urls_str) = part.strip_prefix("xpub=") {
            xpub_endpoints = urls_str
                .split(';')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        } else {
            anyhow::bail!(
                "Invalid broker URL part. Expected 'xsub=' or 'xpub=' prefix, got: {}",
                part
            );
        }
    }

    if xsub_endpoints.is_empty() || xpub_endpoints.is_empty() {
        anyhow::bail!(
            "Broker URL must contain at least one xsub and one xpub endpoint. Got xsub={:?}, xpub={:?}",
            xsub_endpoints,
            xpub_endpoints
        );
    }

    Ok((xsub_endpoints, xpub_endpoints))
}

/// Deduplicates events based on (publisher_id, sequence) tuple
/// Required when connecting to multiple brokers in HA mode
struct DeduplicatingStream {
    inner: WireStream,
    codec: Arc<Codec>,
    seen_events: LruCache<(u64, u64), ()>, // (publisher_id, sequence) -> ()
}

impl DeduplicatingStream {
    fn new(inner: WireStream, codec: Arc<Codec>, cache_size: usize) -> Self {
        Self {
            inner,
            codec,
            seen_events: LruCache::new(
                NonZeroUsize::new(cache_size).expect("cache_size must be non-zero"),
            ),
        }
    }
}

impl Stream for DeduplicatingStream {
    type Item = Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    // Decode envelope to extract publisher_id and sequence
                    match self.codec.decode_envelope_identity(&bytes) {
                        Ok(key) => {
                            // Check if we've seen this event before
                            if self.seen_events.contains(&key) {
                                // Duplicate - skip and continue loop
                                tracing::debug!(
                                    publisher_id = key.0,
                                    sequence = key.1,
                                    "Filtered duplicate event from multi-broker setup"
                                );
                                continue;
                            }

                            // New event - record and return
                            self.seen_events.put(key, ());
                            return Poll::Ready(Some(Ok(bytes)));
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Failed to decode envelope for deduplication");
                            return Poll::Ready(Some(Err(e)));
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Event publisher for a specific topic.
pub struct EventPublisher {
    transport_kind: EventTransportKind,
    topic: String,
    subject: String,
    publisher_id: u64,
    sequence: AtomicU64,
    tx: Arc<dyn EventTransportTx>,
    codec: Arc<Codec>,
    runtime_handle: tokio::runtime::Handle,
    // Keeps unregister work in graceful-shutdown Phase 2 when dropped before shutdown.
    graceful_shutdown_tracker: Arc<crate::utils::GracefulShutdownTracker>,
    /// Discovery client and registered instance for unregistration on drop
    discovery_client: Option<Arc<dyn Discovery>>,
    discovery_instance: Option<crate::discovery::DiscoveryInstance>,
}

impl EventPublisher {
    /// Create a publisher for an endpoint-scoped topic.
    pub async fn for_endpoint(endpoint: &Endpoint, topic: impl Into<String>) -> Result<Self> {
        let transport_kind = endpoint.drt().default_event_transport_kind();
        Self::for_endpoint_with_transport(endpoint, topic, transport_kind).await
    }

    /// Create an endpoint-scoped publisher with explicit transport.
    pub async fn for_endpoint_with_transport(
        endpoint: &Endpoint,
        topic: impl Into<String>,
        transport_kind: EventTransportKind,
    ) -> Result<Self> {
        Self::new_internal(
            endpoint.drt(),
            EventScope::Endpoint {
                endpoint: endpoint.id(),
            },
            topic.into(),
            transport_kind,
        )
        .await
    }

    /// Create a publisher for a component-scoped topic.
    ///
    /// The event transport is chosen automatically: if `DYN_EVENT_PLANE` is set that
    /// value is used; otherwise the runtime's default is used (ZMQ for local backends
    /// such as `file`/`mem`, NATS for distributed backends such as `etcd`/`kubernetes`).
    /// Use [`for_component_with_transport`](Self::for_component_with_transport) to
    /// override explicitly.
    pub async fn for_component(comp: &Component, topic: impl Into<String>) -> Result<Self> {
        let transport_kind = comp.drt().default_event_transport_kind();
        Self::for_component_with_transport(comp, topic, transport_kind).await
    }

    /// Create a publisher with explicit transport.
    pub async fn for_component_with_transport(
        comp: &Component,
        topic: impl Into<String>,
        transport_kind: EventTransportKind,
    ) -> Result<Self> {
        let drt = comp.drt();
        let scope = EventScope::Component {
            namespace: comp.namespace().name(),
            component: comp.name().to_string(),
        };
        Self::new_internal(drt, scope, topic.into(), transport_kind).await
    }

    /// Create a publisher for a namespace-scoped topic.
    ///
    /// The event transport is chosen automatically: if `DYN_EVENT_PLANE` is set that
    /// value is used; otherwise the runtime's default is used (ZMQ for local backends
    /// such as `file`/`mem`, NATS for distributed backends such as `etcd`/`kubernetes`).
    /// Use [`for_namespace_with_transport`](Self::for_namespace_with_transport) to
    /// override explicitly.
    pub async fn for_namespace(ns: &Namespace, topic: impl Into<String>) -> Result<Self> {
        let transport_kind = ns.drt().default_event_transport_kind();
        Self::for_namespace_with_transport(ns, topic, transport_kind).await
    }

    /// Create a namespace publisher with explicit transport.
    pub async fn for_namespace_with_transport(
        ns: &Namespace,
        topic: impl Into<String>,
        transport_kind: EventTransportKind,
    ) -> Result<Self> {
        let drt = ns.drt();
        let scope = EventScope::Namespace { name: ns.name() };
        Self::new_internal(drt, scope, topic.into(), transport_kind).await
    }

    async fn new_internal(
        drt: &DistributedRuntime,
        scope: EventScope,
        topic: String,
        transport_kind: EventTransportKind,
    ) -> Result<Self> {
        // Publishers are discovery objects in their own right. A single process
        // can host multiple publishers for the same scope/topic, each with its
        // own ZMQ endpoint and sequence space, so the process ID is not unique
        // enough here.
        let publisher_id = rand::rngs::OsRng
            .try_next_u64()
            .map_err(|error| anyhow::anyhow!("failed to generate publisher ID: {error}"))?;
        let discovery = Some(drt.discovery());
        let runtime_handle = drt.runtime().secondary();
        let subject = scope.subject(&topic);
        let graceful_shutdown_tracker = drt.graceful_shutdown_tracker();

        // Use Msgpack codec for all transports
        enum TransportSetup {
            Nats(Arc<dyn EventTransportTx>, Arc<Codec>),
            ZmqDirect(Arc<dyn EventTransportTx>, Arc<Codec>, String), // includes public endpoint
            ZmqBroker(Arc<dyn EventTransportTx>, Arc<Codec>),
        }

        let transport_setup = match transport_kind {
            EventTransportKind::Nats => {
                let transport = Arc::new(nats_transport::NatsTransport::new_publisher(
                    drt.clone(),
                    subject.clone(),
                ));
                let codec = Arc::new(Codec::Msgpack(MsgpackCodec));
                TransportSetup::Nats(transport as Arc<dyn EventTransportTx>, codec)
            }
            EventTransportKind::Zmq => {
                // Check for broker mode
                if let Some(broker) = resolve_zmq_broker(drt, &scope).await? {
                    // BROKER MODE: Connect to broker (single or multiple endpoints)
                    let pub_transport = if broker.xsub_endpoints.len() == 1 {
                        zmq_transport::ZmqPubTransport::connect(&broker.xsub_endpoints[0], &subject)
                            .await?
                    } else {
                        zmq_transport::ZmqPubTransport::connect_multiple(
                            &broker.xsub_endpoints,
                            &subject,
                        )
                        .await?
                    };

                    let codec = Arc::new(Codec::Msgpack(MsgpackCodec));
                    TransportSetup::ZmqBroker(
                        Arc::new(pub_transport) as Arc<dyn EventTransportTx>,
                        codec,
                    )
                } else {
                    // DIRECT MODE: Bind PUB socket
                    let (pub_transport, actual_bind_endpoint) = std::thread::spawn({
                        let topic = topic.clone();
                        move || {
                            let rt = tokio::runtime::Builder::new_current_thread()
                                .enable_all()
                                .build()
                                .expect("Failed to create Tokio runtime for ZMQ");

                            rt.block_on(async move {
                                zmq_transport::ZmqPubTransport::bind("tcp://0.0.0.0:0", &topic)
                                    .await
                                    .expect("Failed to bind ZMQ publisher")
                            })
                        }
                    })
                    .join()
                    .expect("Failed to join ZMQ initialization thread");

                    // Get local IP for public endpoint
                    let actual_port: u16 = actual_bind_endpoint
                        .rsplit(':')
                        .next()
                        .and_then(|s| s.parse().ok())
                        .expect("Failed to parse port from bind endpoint");
                    let local_ip = local_ip_for_advertise();
                    let public_endpoint = format!("tcp://{}:{}", local_ip, actual_port);

                    let codec = Arc::new(Codec::Msgpack(MsgpackCodec));
                    TransportSetup::ZmqDirect(
                        Arc::new(pub_transport) as Arc<dyn EventTransportTx>,
                        codec,
                        public_endpoint,
                    )
                }
            }
        };

        // Extract transport and codec, and register if needed
        let (tx, codec, discovery_instance) = match transport_setup {
            TransportSetup::Nats(tx, codec) => {
                let transport_config = EventTransport::nats(scope.subject_prefix());
                let spec = DiscoverySpec::EventChannel {
                    scope: scope.clone(),
                    topic: topic.clone(),
                    publisher_id,
                    transport: transport_config,
                };

                let discovery_instance = drt.discovery().register(spec).await?;
                tracing::info!(
                    topic = %topic,
                    transport = ?transport_kind,
                    publisher_id = %publisher_id,
                    "EventPublisher registered with discovery"
                );
                (tx, codec, Some(discovery_instance))
            }
            TransportSetup::ZmqDirect(tx, codec, public_endpoint) => {
                let transport_config = EventTransport::zmq(public_endpoint);
                let spec = DiscoverySpec::EventChannel {
                    scope: scope.clone(),
                    topic: topic.clone(),
                    publisher_id,
                    transport: transport_config,
                };

                let discovery_instance = drt.discovery().register(spec).await?;
                tracing::info!(
                    topic = %topic,
                    transport = ?transport_kind,
                    publisher_id = %publisher_id,
                    "EventPublisher registered with discovery (direct mode)"
                );
                (tx, codec, Some(discovery_instance))
            }
            TransportSetup::ZmqBroker(tx, codec) => {
                tracing::info!(
                    topic = %topic,
                    transport = ?transport_kind,
                    "EventPublisher in broker mode - skipping discovery registration"
                );
                (tx, codec, None)
            }
        };

        Ok(Self {
            transport_kind,
            topic,
            subject,
            publisher_id,
            sequence: AtomicU64::new(0),
            tx,
            codec,
            runtime_handle,
            graceful_shutdown_tracker,
            discovery_client: discovery,
            discovery_instance,
        })
    }

    /// Publish a serializable event.
    pub async fn publish<T: Serialize + Send + Sync>(&self, event: &T) -> Result<()> {
        let payload = self.codec.encode_payload(event)?;
        self.publish_bytes_ref(payload.as_ref()).await
    }

    /// Publish raw bytes.
    pub async fn publish_bytes(&self, bytes: Vec<u8>) -> Result<()> {
        self.publish_bytes_ref(&bytes).await
    }

    /// Publish raw bytes without taking ownership of the payload buffer.
    pub async fn publish_bytes_ref(&self, bytes: &[u8]) -> Result<()> {
        let envelope_bytes = self.codec.encode_envelope_parts(
            self.publisher_id,
            self.sequence.fetch_add(1, Ordering::SeqCst),
            current_timestamp_ms(),
            &self.topic,
            bytes,
        )?;

        self.tx.publish(&self.subject, envelope_bytes).await
    }

    /// Get the publisher ID.
    pub fn publisher_id(&self) -> u64 {
        self.publisher_id
    }

    /// Get the topic.
    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Get the transport kind.
    pub fn transport_kind(&self) -> EventTransportKind {
        self.transport_kind
    }
}

impl Drop for EventPublisher {
    fn drop(&mut self) {
        // Unregister from discovery on drop
        if let (Some(discovery), Some(instance)) =
            (self.discovery_client.take(), self.discovery_instance.take())
        {
            let topic = self.topic.clone();
            let publisher_id = instance.instance_id();
            let runtime_handle = self.runtime_handle.clone();
            let shutdown_guard = self.graceful_shutdown_tracker.register_task();

            // Drop can run outside any Tokio context (notably via PyO3 finalizers), so use
            // the runtime that created the publisher rather than the ambient thread state.
            let spawn_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                runtime_handle.spawn(async move {
                    let _shutdown_guard = shutdown_guard;
                    match discovery.unregister(instance).await {
                        Ok(()) => {
                            tracing::info!(
                                topic = %topic,
                                publisher_id = %publisher_id,
                                "EventPublisher unregistered from discovery"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                topic = %topic,
                                publisher_id = %publisher_id,
                                error = %e,
                                "Failed to unregister EventPublisher from discovery"
                            );
                        }
                    }
                });
            }));

            if spawn_result.is_err() {
                tracing::warn!(
                    topic = %self.topic,
                    publisher_id = %publisher_id,
                    "Skipping EventPublisher unregister during drop because the runtime is unavailable"
                );
            }
        }
    }
}

/// Event subscriber for a specific topic.
pub struct EventSubscriber {
    stream: EventStream,
    #[allow(dead_code)]
    scope: EventScope,
    #[allow(dead_code)]
    topic: String,
    codec: Arc<Codec>,
}

impl EventSubscriber {
    /// Create a subscriber for an endpoint-scoped topic.
    pub async fn for_endpoint(endpoint: &Endpoint, topic: impl Into<String>) -> Result<Self> {
        let transport_kind = endpoint.drt().default_event_transport_kind();
        Self::for_endpoint_with_transport(endpoint, topic, transport_kind).await
    }

    /// Create an endpoint-scoped subscriber with explicit transport.
    pub async fn for_endpoint_with_transport(
        endpoint: &Endpoint,
        topic: impl Into<String>,
        transport_kind: EventTransportKind,
    ) -> Result<Self> {
        Self::for_endpoint_id_with_transport(endpoint.drt(), &endpoint.id(), topic, transport_kind)
            .await
    }

    /// Create a subscriber for an endpoint identity without constructing a
    /// local [`Endpoint`] handle.
    pub async fn for_endpoint_id(
        drt: &DistributedRuntime,
        endpoint: &EndpointId,
        topic: impl Into<String>,
    ) -> Result<Self> {
        let transport_kind = drt.default_event_transport_kind();
        Self::for_endpoint_id_with_transport(drt, endpoint, topic, transport_kind).await
    }

    /// Create an endpoint-identity subscriber with explicit transport.
    pub async fn for_endpoint_id_with_transport(
        drt: &DistributedRuntime,
        endpoint: &EndpointId,
        topic: impl Into<String>,
        transport_kind: EventTransportKind,
    ) -> Result<Self> {
        Self::new_internal(
            drt,
            EventScope::Endpoint {
                endpoint: endpoint.clone(),
            },
            topic.into(),
            transport_kind,
        )
        .await
    }

    /// Create a subscriber for a component-scoped topic.
    ///
    /// The event transport is chosen automatically: if `DYN_EVENT_PLANE` is set that
    /// value is used; otherwise the runtime's default is used (ZMQ for local backends
    /// such as `file`/`mem`, NATS for distributed backends such as `etcd`/`kubernetes`).
    /// Use [`for_component_with_transport`](Self::for_component_with_transport) to
    /// override explicitly.
    pub async fn for_component(comp: &Component, topic: impl Into<String>) -> Result<Self> {
        let transport_kind = comp.drt().default_event_transport_kind();
        Self::for_component_with_transport(comp, topic, transport_kind).await
    }

    /// Create a subscriber with explicit transport.
    pub async fn for_component_with_transport(
        comp: &Component,
        topic: impl Into<String>,
        transport_kind: EventTransportKind,
    ) -> Result<Self> {
        let drt = comp.drt();
        let scope = EventScope::Component {
            namespace: comp.namespace().name(),
            component: comp.name().to_string(),
        };
        Self::new_internal(drt, scope, topic.into(), transport_kind).await
    }

    /// Create a subscriber for a namespace-scoped topic.
    ///
    /// The event transport is chosen automatically: if `DYN_EVENT_PLANE` is set that
    /// value is used; otherwise the runtime's default is used (ZMQ for local backends
    /// such as `file`/`mem`, NATS for distributed backends such as `etcd`/`kubernetes`).
    /// Use [`for_namespace_with_transport`](Self::for_namespace_with_transport) to
    /// override explicitly.
    pub async fn for_namespace(ns: &Namespace, topic: impl Into<String>) -> Result<Self> {
        let transport_kind = ns.drt().default_event_transport_kind();
        Self::for_namespace_with_transport(ns, topic, transport_kind).await
    }

    /// Create a namespace subscriber with explicit transport.
    pub async fn for_namespace_with_transport(
        ns: &Namespace,
        topic: impl Into<String>,
        transport_kind: EventTransportKind,
    ) -> Result<Self> {
        let drt = ns.drt();
        let scope = EventScope::Namespace { name: ns.name() };
        Self::new_internal(drt, scope, topic.into(), transport_kind).await
    }

    async fn new_internal(
        drt: &DistributedRuntime,
        scope: EventScope,
        topic: String,
        transport_kind: EventTransportKind,
    ) -> Result<Self> {
        let discovery = drt.discovery();
        let routing_key = scope.subject(&topic);

        // Use Msgpack codec for all transports
        let (wire_stream, codec): (WireStream, Arc<Codec>) = match transport_kind {
            EventTransportKind::Nats => {
                let transport = nats_transport::NatsTransport::new(drt.clone());
                let stream = transport.subscribe(&routing_key).await?;
                let codec = Arc::new(Codec::Msgpack(MsgpackCodec));
                (stream, codec)
            }
            EventTransportKind::Zmq => {
                // Check for broker mode
                if let Some(broker) = resolve_zmq_broker(drt, &scope).await? {
                    // BROKER MODE: Connect to broker's XPUB (single or multiple endpoints)
                    let codec = Arc::new(Codec::Msgpack(MsgpackCodec));

                    let stream: WireStream = if broker.xpub_endpoints.len() == 1 {
                        // Single broker - no deduplication needed
                        let sub_transport = zmq_transport::ZmqSubTransport::connect_broker(
                            &broker.xpub_endpoints[0],
                            &routing_key,
                        )
                        .await?;
                        sub_transport.subscribe(&routing_key).await?
                    } else {
                        // Multiple brokers - need deduplication
                        let sub_transport =
                            zmq_transport::ZmqSubTransport::connect_broker_multiple(
                                &broker.xpub_endpoints,
                                &routing_key,
                            )
                            .await?;
                        let inner_stream = sub_transport.subscribe(&routing_key).await?;

                        // Wrap with deduplication (default cache size: 100,000 entries)
                        Box::pin(DeduplicatingStream::new(
                            inner_stream,
                            codec.clone(),
                            100_000,
                        ))
                    };

                    (stream, codec)
                } else {
                    // DIRECT MODE: Use dynamic subscriber to discover and connect to publishers
                    let query = match &scope {
                        EventScope::Namespace { name } => {
                            crate::discovery::DiscoveryQuery::EventChannels(
                                crate::discovery::EventChannelQuery::namespace_topic(
                                    name.clone(),
                                    topic.clone(),
                                ),
                            )
                        }
                        EventScope::Component {
                            namespace,
                            component,
                        } => crate::discovery::DiscoveryQuery::EventChannels(
                            crate::discovery::EventChannelQuery::topic(
                                namespace.clone(),
                                component.clone(),
                                topic.clone(),
                            ),
                        ),
                        EventScope::Endpoint { endpoint } => {
                            crate::discovery::DiscoveryQuery::EventChannels(
                                crate::discovery::EventChannelQuery::endpoint_topic(
                                    endpoint.clone(),
                                    topic.clone(),
                                ),
                            )
                        }
                    };

                    let subscriber = Arc::new(DynamicSubscriber::with_cancel_token(
                        discovery,
                        query,
                        topic.clone(),
                        drt.primary_token().child_token(),
                    ));

                    let stream = subscriber.start_zmq().await?;
                    let codec = Arc::new(Codec::Msgpack(MsgpackCodec));
                    (stream, codec)
                }
            }
        };

        // Filter by topic and decode envelopes
        let topic_filter = topic.clone();
        let codec_for_stream = codec.clone();
        let stream = wire_stream.filter_map(move |result| {
            let codec = codec_for_stream.clone();
            let topic_filter = topic_filter.clone();
            async move {
                match result {
                    Ok(bytes) => match codec.decode_envelope(&bytes) {
                        Ok(envelope) => {
                            // Filter by topic for transports that don't support native filtering
                            if envelope.topic == topic_filter {
                                Some(Ok(envelope))
                            } else {
                                None
                            }
                        }
                        Err(e) => Some(Err(e)),
                    },
                    Err(e) => Some(Err(e)),
                }
            }
        });

        tracing::info!(
            topic = %topic,
            transport = ?transport_kind,
            "EventSubscriber created"
        );

        Ok(Self {
            stream: Box::pin(stream),
            scope,
            topic,
            codec,
        })
    }

    /// Get the next event envelope.
    pub async fn next(&mut self) -> Option<Result<EventEnvelope>> {
        self.stream.next().await
    }

    /// Subscribe with automatic deserialization.
    pub fn typed<T: DeserializeOwned + Send + 'static>(self) -> TypedEventSubscriber<T> {
        TypedEventSubscriber {
            stream: self.stream,
            codec: self.codec,
            _marker: std::marker::PhantomData,
        }
    }
}

/// Typed event subscriber that deserializes payloads.
pub struct TypedEventSubscriber<T> {
    stream: EventStream,
    codec: Arc<Codec>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: DeserializeOwned + Send + 'static> TypedEventSubscriber<T> {
    /// Get the next typed event with its envelope.
    pub async fn next(&mut self) -> Option<Result<(EventEnvelope, T)>> {
        std::future::poll_fn(|cx| self.poll_next(cx)).await
    }

    /// Poll for the next typed event.
    pub fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<Option<Result<(EventEnvelope, T)>>> {
        match self.stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(envelope)) => Poll::Ready(Some(match envelope {
                Ok(env) => match self.codec.decode_payload(&env.payload) {
                    Ok(typed) => Ok((env, typed)),
                    Err(e) => Err(e),
                },
                Err(e) => Err(e),
            })),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Get current timestamp in milliseconds since Unix epoch.
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::environment_names::zmq_broker as broker_env;

    #[tokio::test]
    async fn direct_zmq_endpoint_scopes_are_isolated() {
        temp_env::async_with_vars(
            [
                (broker_env::DYN_ZMQ_BROKER_URL, None::<&str>),
                (broker_env::DYN_ZMQ_BROKER_ENABLED, None::<&str>),
            ],
            async {
                let runtime = crate::Runtime::from_current().expect("create runtime handle");
                let drt = DistributedRuntime::new(
                    runtime,
                    crate::distributed::DistributedConfig::process_local(),
                )
                .await
                .expect("create distributed runtime");
                let component = drt
                    .namespace("endpoint-event-isolation-test")
                    .expect("create namespace")
                    .component("worker")
                    .expect("create component");
                let endpoint_a = component.endpoint("a");
                let endpoint_b = component.endpoint("b");

                let publisher_a = EventPublisher::for_endpoint_with_transport(
                    &endpoint_a,
                    "events",
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create endpoint A publisher");
                let publisher_b = EventPublisher::for_endpoint_with_transport(
                    &endpoint_b,
                    "events",
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create endpoint B publisher");
                let mut subscriber_a = EventSubscriber::for_endpoint_with_transport(
                    &endpoint_a,
                    "events",
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create endpoint A subscriber");
                let mut subscriber_b = EventSubscriber::for_endpoint_with_transport(
                    &endpoint_b,
                    "events",
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create endpoint B subscriber");

                let receive = async {
                    loop {
                        publisher_a
                            .publish_bytes(vec![0xa1])
                            .await
                            .expect("publish endpoint A event");
                        publisher_b
                            .publish_bytes(vec![0xb2])
                            .await
                            .expect("publish endpoint B event");

                        let a = tokio::time::timeout(
                            std::time::Duration::from_millis(100),
                            subscriber_a.next(),
                        )
                        .await;
                        let b = tokio::time::timeout(
                            std::time::Duration::from_millis(100),
                            subscriber_b.next(),
                        )
                        .await;
                        if let (Ok(Some(Ok(a))), Ok(Some(Ok(b)))) = (a, b) {
                            assert_eq!(a.publisher_id, publisher_a.publisher_id());
                            assert_eq!(a.payload.as_ref(), &[0xa1]);
                            assert_eq!(b.publisher_id, publisher_b.publisher_id());
                            assert_eq!(b.payload.as_ref(), &[0xb2]);
                            break;
                        }

                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                };

                tokio::time::timeout(std::time::Duration::from_secs(5), receive)
                    .await
                    .expect("each endpoint subscriber should receive only its own publisher");
            },
        )
        .await;
    }

    #[tokio::test]
    async fn direct_zmq_publishers_in_one_endpoint_fan_into_one_subscriber() {
        temp_env::async_with_vars(
            [
                (broker_env::DYN_ZMQ_BROKER_URL, None::<&str>),
                (broker_env::DYN_ZMQ_BROKER_ENABLED, None::<&str>),
            ],
            async {
                let runtime = crate::Runtime::from_current().expect("create runtime handle");
                let drt = DistributedRuntime::new(
                    runtime,
                    crate::distributed::DistributedConfig::process_local(),
                )
                .await
                .expect("create distributed runtime");
                let endpoint = drt
                    .namespace("endpoint-event-fan-in-test")
                    .expect("create namespace")
                    .component("worker")
                    .expect("create component")
                    .endpoint("generate");
                let publisher_a = EventPublisher::for_endpoint_with_transport(
                    &endpoint,
                    "events",
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create first publisher");
                let publisher_b = EventPublisher::for_endpoint_with_transport(
                    &endpoint,
                    "events",
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create second publisher");
                let mut subscriber = EventSubscriber::for_endpoint_with_transport(
                    &endpoint,
                    "events",
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create subscriber");

                let receive = async {
                    let mut publisher_ids = std::collections::HashSet::new();
                    while publisher_ids.len() < 2 {
                        publisher_a.publish_bytes(vec![0xa1]).await.unwrap();
                        publisher_b.publish_bytes(vec![0xb2]).await.unwrap();
                        if let Ok(Some(Ok(envelope))) = tokio::time::timeout(
                            std::time::Duration::from_millis(100),
                            subscriber.next(),
                        )
                        .await
                        {
                            publisher_ids.insert(envelope.publisher_id);
                        }
                    }
                    assert_eq!(
                        publisher_ids,
                        std::collections::HashSet::from([
                            publisher_a.publisher_id(),
                            publisher_b.publisher_id(),
                        ])
                    );
                };
                tokio::time::timeout(std::time::Duration::from_secs(5), receive)
                    .await
                    .expect("subscriber should receive both endpoint publishers");
            },
        )
        .await;
    }

    #[tokio::test]
    async fn same_topic_publishers_are_independent_across_recreation() {
        temp_env::async_with_vars(
            [
                (broker_env::DYN_ZMQ_BROKER_URL, None::<&str>),
                (broker_env::DYN_ZMQ_BROKER_ENABLED, None::<&str>),
            ],
            async {
                let runtime = crate::Runtime::from_current().expect("create runtime handle");
                let drt = DistributedRuntime::new(
                    runtime,
                    crate::distributed::DistributedConfig::process_local(),
                )
                .await
                .expect("create distributed runtime");
                let component = drt
                    .namespace("event-publisher-test")
                    .expect("create namespace")
                    .component("worker")
                    .expect("create component");

                let publisher_a = EventPublisher::for_component_with_transport(
                    &component,
                    "events",
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create first publisher");
                let publisher_b = EventPublisher::for_component_with_transport(
                    &component,
                    "events",
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create second publisher");
                let publisher_a_id = publisher_a.publisher_id();
                let publisher_b_id = publisher_b.publisher_id();

                assert_ne!(publisher_a_id, publisher_b_id);

                let query = DiscoveryQuery::EventChannels(EventChannelQuery::topic(
                    "event-publisher-test",
                    "worker",
                    "events",
                ));
                let instances = drt
                    .discovery()
                    .list(query.clone())
                    .await
                    .expect("list event publishers");
                assert_eq!(instances.len(), 2);
                assert!(
                    instances
                        .iter()
                        .any(|instance| instance.instance_id() == publisher_a_id)
                );
                assert!(
                    instances
                        .iter()
                        .any(|instance| instance.instance_id() == publisher_b_id)
                );

                let mut subscriber = EventSubscriber::for_component_with_transport(
                    &component,
                    "events",
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create subscriber");
                let mut received_a = false;
                let mut received_b = false;

                tokio::time::timeout(std::time::Duration::from_secs(5), async {
                    while !received_a || !received_b {
                        publisher_a
                            .publish_bytes(vec![0xa1])
                            .await
                            .expect("publish from first publisher");
                        publisher_b
                            .publish_bytes(vec![0xb2])
                            .await
                            .expect("publish from second publisher");

                        if let Ok(Some(envelope)) = tokio::time::timeout(
                            std::time::Duration::from_millis(100),
                            subscriber.next(),
                        )
                        .await
                        {
                            let envelope = envelope.expect("receive event envelope");
                            if envelope.publisher_id == publisher_a_id {
                                assert_eq!(envelope.payload.as_ref(), &[0xa1]);
                                received_a = true;
                            } else if envelope.publisher_id == publisher_b_id {
                                assert_eq!(envelope.payload.as_ref(), &[0xb2]);
                                received_b = true;
                            } else {
                                panic!("event from unexpected publisher");
                            }
                        }

                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                })
                .await
                .expect("subscriber should receive events from both publishers");

                drop(publisher_a);
                let publisher_a_recreated = EventPublisher::for_component_with_transport(
                    &component,
                    "events",
                    EventTransportKind::Zmq,
                )
                .await
                .expect("recreate first publisher");
                let publisher_a_recreated_id = publisher_a_recreated.publisher_id();

                assert_ne!(publisher_a_recreated_id, publisher_a_id);
                assert_ne!(publisher_a_recreated_id, publisher_b_id);
                assert_eq!(
                    publisher_a_recreated.sequence.load(Ordering::SeqCst),
                    0,
                    "a recreated publisher starts a new sequence space"
                );

                tokio::time::timeout(std::time::Duration::from_secs(1), async {
                    loop {
                        let instances = drt
                            .discovery()
                            .list(query.clone())
                            .await
                            .expect("list event publishers after recreation");
                        if instances.len() == 2
                            && instances
                                .iter()
                                .any(|instance| instance.instance_id() == publisher_b_id)
                            && instances
                                .iter()
                                .any(|instance| instance.instance_id() == publisher_a_recreated_id)
                        {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .expect("old publisher should unregister without removing current publishers");

                let mut received_b_after_recreation = false;
                let mut received_recreated_a = false;

                tokio::time::timeout(std::time::Duration::from_secs(5), async {
                    while !received_b_after_recreation || !received_recreated_a {
                        publisher_b
                            .publish_bytes(vec![0xb3])
                            .await
                            .expect("publish from second publisher after recreation");
                        publisher_a_recreated
                            .publish_bytes(vec![0xa2])
                            .await
                            .expect("publish from recreated publisher");

                        if let Ok(Some(envelope)) = tokio::time::timeout(
                            std::time::Duration::from_millis(100),
                            subscriber.next(),
                        )
                        .await
                        {
                            let envelope = envelope.expect("receive event envelope after drop");
                            if envelope.publisher_id == publisher_b_id
                                && envelope.payload.as_ref() == [0xb3]
                            {
                                received_b_after_recreation = true;
                            } else if envelope.publisher_id == publisher_a_recreated_id
                                && envelope.payload.as_ref() == [0xa2]
                            {
                                received_recreated_a = true;
                            }
                        }

                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                })
                .await
                .expect("subscriber should receive from surviving and recreated publishers");
            },
        )
        .await;
    }

    #[tokio::test]
    async fn dropped_publisher_unregister_completes_within_graceful_shutdown() {
        temp_env::async_with_vars(
            [
                (broker_env::DYN_ZMQ_BROKER_URL, None::<&str>),
                (broker_env::DYN_ZMQ_BROKER_ENABLED, None::<&str>),
            ],
            async {
                let runtime = crate::Runtime::from_current().expect("create runtime handle");
                let drt = DistributedRuntime::new(
                    runtime,
                    crate::distributed::DistributedConfig::process_local(),
                )
                .await
                .expect("create distributed runtime");
                let component = drt
                    .namespace("event-publisher-shutdown-test")
                    .expect("create namespace")
                    .component("worker")
                    .expect("create component");

                let publisher = EventPublisher::for_component_with_transport(
                    &component,
                    "events",
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create publisher");
                let publisher_id = publisher.publisher_id();

                let query = DiscoveryQuery::EventChannels(EventChannelQuery::topic(
                    "event-publisher-shutdown-test",
                    "worker",
                    "events",
                ));
                let instances = drt
                    .discovery()
                    .list(query.clone())
                    .await
                    .expect("list event publishers");
                assert_eq!(instances.len(), 1);
                assert_eq!(instances[0].instance_id(), publisher_id);

                let tracker = drt.graceful_shutdown_tracker();
                assert_eq!(tracker.get_count(), 0);

                let main_token = drt.runtime().primary_token();
                let endpoint_token = drt.runtime().child_token();

                // Dropping the publisher schedules the async discovery unregister.
                // The drop path must synchronously take a graceful-shutdown guard so
                // that `Runtime::shutdown` Phase 2 waits for the unregister task.
                drop(publisher);
                assert_eq!(
                    tracker.get_count(),
                    1,
                    "dropping a publisher must register its unregister work with the graceful-shutdown tracker"
                );

                drt.runtime().shutdown();

                tokio::time::timeout(std::time::Duration::from_secs(5), main_token.cancelled())
                    .await
                    .expect("graceful shutdown should complete once the unregister task finishes");

                assert!(endpoint_token.is_cancelled());
                assert_eq!(
                    tracker.get_count(),
                    0,
                    "the unregister task must release its graceful-shutdown guard"
                );

                // Phase 3 (main token cancellation) only runs after Phase 2 drained
                // the tracker, so the dropped publisher must already be unregistered.
                let instances = drt
                    .discovery()
                    .list(query)
                    .await
                    .expect("list event publishers after shutdown");
                assert!(
                    instances.is_empty(),
                    "unregister must complete within the graceful-shutdown window"
                );
            },
        )
        .await;
    }

    #[tokio::test]
    async fn runtime_cancellation_stops_retained_direct_zmq_subscriber() {
        temp_env::async_with_vars(
            [
                (broker_env::DYN_ZMQ_BROKER_URL, None::<&str>),
                (broker_env::DYN_ZMQ_BROKER_ENABLED, None::<&str>),
            ],
            async {
                let runtime = crate::Runtime::from_current().expect("create runtime handle");
                let drt = DistributedRuntime::new(
                    runtime,
                    crate::distributed::DistributedConfig::process_local(),
                )
                .await
                .expect("create distributed runtime");
                let component = drt
                    .namespace("event-subscriber-shutdown-test")
                    .expect("create namespace")
                    .component("worker")
                    .expect("create component");

                let mut subscriber = EventSubscriber::for_component_with_transport(
                    &component,
                    "events",
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create subscriber");

                drt.primary_token().cancel();

                let next =
                    tokio::time::timeout(std::time::Duration::from_secs(1), subscriber.next())
                        .await
                        .expect("runtime cancellation should stop the subscriber stream");
                assert!(next.is_none());
            },
        )
        .await;
    }

    #[test]
    fn test_event_scope_subject_prefix() {
        let ns_scope = EventScope::Namespace {
            name: "test-ns".to_string(),
        };
        assert_eq!(ns_scope.subject_prefix(), "namespace.test-ns");

        let comp_scope = EventScope::Component {
            namespace: "test-ns".to_string(),
            component: "test-comp".to_string(),
        };
        assert_eq!(
            comp_scope.subject_prefix(),
            "namespace.test-ns.component.test-comp"
        );
    }

    #[test]
    fn test_event_scope_accessors() {
        let ns_scope = EventScope::Namespace {
            name: "my-ns".to_string(),
        };
        assert_eq!(ns_scope.namespace(), "my-ns");
        assert_eq!(ns_scope.component(), None);

        let comp_scope = EventScope::Component {
            namespace: "my-ns".to_string(),
            component: "my-comp".to_string(),
        };
        assert_eq!(comp_scope.namespace(), "my-ns");
        assert_eq!(comp_scope.component(), Some("my-comp"));
    }

    #[test]
    fn test_timestamp_generation() {
        let ts = current_timestamp_ms();

        // Should be after Jan 1, 2020 (1577836800000) and before Jan 1, 2100 (4102444800000)
        assert!(ts > 1577836800000, "Timestamp should be after 2020");
        assert!(ts < 4102444800000, "Timestamp should be before 2100");
    }

    #[test]
    fn test_event_envelope_serde() {
        let envelope = EventEnvelope {
            publisher_id: 42,
            sequence: 10,
            published_at: 1700000000000,
            topic: "test-topic".to_string(),
            payload: Bytes::from("test data"),
        };

        let json = serde_json::to_string(&envelope).expect("serialize");
        let deserialized: EventEnvelope = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.publisher_id, 42);
        assert_eq!(deserialized.sequence, 10);
        assert_eq!(deserialized.published_at, 1700000000000);
        assert_eq!(deserialized.topic, "test-topic");
        assert_eq!(deserialized.payload, Bytes::from("test data"));
    }
}
