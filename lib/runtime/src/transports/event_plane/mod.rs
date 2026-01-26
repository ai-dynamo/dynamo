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
pub use crate::discovery::EventTransportKind;

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use lru::LruCache;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::DistributedRuntime;
use crate::component::{Component, Namespace};
use crate::discovery::{
    Discovery, DiscoveryInstance, DiscoveryQuery, DiscoverySpec, EventChannelQuery, EventTransport,
};
use crate::traits::DistributedRuntimeProvider;
use crate::utils::ip_resolver::get_local_ip_for_advertise;

/// Scope of the event plane - determines the subject prefix for pub/sub.
#[derive(Debug, Clone)]
pub enum EventScope {
    /// Namespace-level scope: `namespace.{name}`
    Namespace { name: String },
    /// Component-level scope: `namespace.{namespace}.component.{component}`
    Component {
        namespace: String,
        component: String,
    },
}

impl EventScope {
    /// Returns the subject prefix for this scope.
    pub fn subject_prefix(&self) -> String {
        match self {
            EventScope::Namespace { name } => format!("namespace.{}", name),
            EventScope::Component {
                namespace,
                component,
            } => {
                format!("namespace.{}.component.{}", namespace, component)
            }
        }
    }

    /// Get the namespace name
    pub fn namespace(&self) -> &str {
        match self {
            EventScope::Namespace { name } => name,
            EventScope::Component { namespace, .. } => namespace,
        }
    }

    /// Get the component name (if component-scoped)
    pub fn component(&self) -> Option<&str> {
        match self {
            EventScope::Namespace { .. } => None,
            EventScope::Component { component, .. } => Some(component),
        }
    }
}

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

    // Priority 2: Discovery-based lookup if DYN_ZMQ_BROKER_ENABLED=true
    if std::env::var(crate::config::environment_names::zmq_broker::DYN_ZMQ_BROKER_ENABLED)
        .unwrap_or_default()
        == "true"
    {
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
                "DYN_ZMQ_BROKER_ENABLED=true but no broker found in discovery for namespace '{}'",
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
                    match self.codec.decode_envelope(&bytes) {
                        Ok(envelope) => {
                            let key = (envelope.publisher_id, envelope.sequence);

                            // Check if we've seen this event before
                            if self.seen_events.contains(&key) {
                                // Duplicate - skip and continue loop
                                tracing::debug!(
                                    publisher_id = envelope.publisher_id,
                                    sequence = envelope.sequence,
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
    scope: EventScope,
    topic: String,
    publisher_id: u64,
    sequence: AtomicU64,
    tx: Arc<dyn EventTransportTx>,
    codec: Arc<Codec>,
    /// Discovery client and registered instance for unregistration on drop
    discovery_client: Option<Arc<dyn Discovery>>,
    discovery_instance: Option<crate::discovery::DiscoveryInstance>,
}

impl EventPublisher {
    /// Create a publisher for a component-scoped topic.
    pub async fn for_component(comp: &Component, topic: impl Into<String>) -> Result<Self> {
        Self::for_component_with_transport(comp, topic, EventTransportKind::from_env_or_default())
            .await
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
    pub async fn for_namespace(ns: &Namespace, topic: impl Into<String>) -> Result<Self> {
        Self::for_namespace_with_transport(ns, topic, EventTransportKind::from_env_or_default())
            .await
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
        let publisher_id = drt.discovery().instance_id();
        let discovery = Some(drt.discovery());

        // Use Msgpack codec for all transports
        enum TransportSetup {
            Nats(Arc<dyn EventTransportTx>, Arc<Codec>),
            ZmqDirect(Arc<dyn EventTransportTx>, Arc<Codec>, String), // includes public endpoint
            ZmqBroker(Arc<dyn EventTransportTx>, Arc<Codec>),
        }

        let transport_setup = match transport_kind {
            EventTransportKind::Nats => {
                let transport = Arc::new(nats_transport::NatsTransport::new(drt.clone()));
                let codec = Arc::new(Codec::Msgpack(MsgpackCodec));
                TransportSetup::Nats(transport as Arc<dyn EventTransportTx>, codec)
            }
            EventTransportKind::Zmq => {
                // Check for broker mode
                if let Some(broker) = resolve_zmq_broker(drt, &scope).await? {
                    // BROKER MODE: Connect to broker (single or multiple endpoints)
                    let pub_transport = if broker.xsub_endpoints.len() == 1 {
                        zmq_transport::ZmqPubTransport::connect(&broker.xsub_endpoints[0], &topic)
                            .await?
                    } else {
                        zmq_transport::ZmqPubTransport::connect_multiple(
                            &broker.xsub_endpoints,
                            &topic,
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
                    let local_ip = get_local_ip_for_advertise();
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
                    namespace: scope.namespace().to_string(),
                    component: scope.component().unwrap_or("").to_string(),
                    topic: topic.clone(),
                    transport: transport_config,
                };

                let registered_instance = drt.discovery().register(spec).await?;
                tracing::info!(
                    topic = %topic,
                    transport = ?transport_kind,
                    instance_id = %registered_instance.instance_id(),
                    "EventPublisher registered with discovery"
                );
                (tx, codec, Some(registered_instance))
            }
            TransportSetup::ZmqDirect(tx, codec, public_endpoint) => {
                let transport_config = EventTransport::zmq(public_endpoint);
                let spec = DiscoverySpec::EventChannel {
                    namespace: scope.namespace().to_string(),
                    component: scope.component().unwrap_or("").to_string(),
                    topic: topic.clone(),
                    transport: transport_config,
                };

                let registered_instance = drt.discovery().register(spec).await?;
                tracing::info!(
                    topic = %topic,
                    transport = ?transport_kind,
                    instance_id = %registered_instance.instance_id(),
                    "EventPublisher registered with discovery (direct mode)"
                );
                (tx, codec, Some(registered_instance))
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
            scope,
            topic,
            publisher_id,
            sequence: AtomicU64::new(0),
            tx,
            codec,
            discovery_client: discovery,
            discovery_instance,
        })
    }

    /// Publish a serializable event.
    pub async fn publish<T: Serialize + Send + Sync>(&self, event: &T) -> Result<()> {
        let payload = self.codec.encode_payload(event)?;
        self.publish_bytes(payload.to_vec()).await
    }

    /// Publish raw bytes.
    pub async fn publish_bytes(&self, bytes: Vec<u8>) -> Result<()> {
        let envelope = EventEnvelope {
            publisher_id: self.publisher_id,
            sequence: self.sequence.fetch_add(1, Ordering::SeqCst),
            published_at: current_timestamp_ms(),
            topic: self.topic.clone(),
            payload: Bytes::from(bytes),
        };

        let envelope_bytes = self.codec.encode_envelope(&envelope)?;
        let subject = format!("{}.{}", self.scope.subject_prefix(), self.topic);

        self.tx.publish(&subject, envelope_bytes).await
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
            let instance_id = instance.instance_id();

            // Spawn background task for async unregister since Drop is sync
            tokio::spawn(async move {
                match discovery.unregister(instance).await {
                    Ok(()) => {
                        tracing::info!(
                            topic = %topic,
                            instance_id = %instance_id,
                            "EventPublisher unregistered from discovery"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            topic = %topic,
                            instance_id = %instance_id,
                            error = %e,
                            "Failed to unregister EventPublisher from discovery"
                        );
                    }
                }
            });
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
    /// Create a subscriber for a component-scoped topic.
    pub async fn for_component(comp: &Component, topic: impl Into<String>) -> Result<Self> {
        Self::for_component_with_transport(comp, topic, EventTransportKind::from_env_or_default())
            .await
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
    pub async fn for_namespace(ns: &Namespace, topic: impl Into<String>) -> Result<Self> {
        Self::for_namespace_with_transport(ns, topic, EventTransportKind::from_env_or_default())
            .await
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

        // Use Msgpack codec for all transports
        let (wire_stream, codec): (WireStream, Arc<Codec>) = match transport_kind {
            EventTransportKind::Nats => {
                let transport = nats_transport::NatsTransport::new(drt.clone());
                let subject = format!("{}.{}", scope.subject_prefix(), topic);
                let stream = transport.subscribe(&subject).await?;
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
                            &topic,
                        )
                        .await?;
                        sub_transport.subscribe(&topic).await?
                    } else {
                        // Multiple brokers - need deduplication
                        let sub_transport =
                            zmq_transport::ZmqSubTransport::connect_broker_multiple(
                                &broker.xpub_endpoints,
                                &topic,
                            )
                            .await?;
                        let inner_stream = sub_transport.subscribe(&topic).await?;

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
                                crate::discovery::EventChannelQuery::namespace(name.clone()),
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
                    };

                    let subscriber =
                        Arc::new(DynamicSubscriber::new(discovery, query, topic.clone()));

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
        let envelope = self.stream.next().await?;
        match envelope {
            Ok(env) => match self.codec.decode_payload(&env.payload) {
                Ok(typed) => Some(Ok((env, typed))),
                Err(e) => Some(Err(e)),
            },
            Err(e) => Some(Err(e)),
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
    use crate::config::environment_names::event_plane as env;

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
