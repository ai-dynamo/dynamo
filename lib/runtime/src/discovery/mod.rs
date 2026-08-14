// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::protocols::EndpointId;
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use tokio_util::sync::CancellationToken;

mod metadata;
pub use metadata::{DiscoveryMetadata, MetadataSnapshot};

mod registration;
pub use registration::EndpointRegistrationLease;
pub(crate) use registration::EndpointRegistrationManager;

mod mock;
pub use mock::{MockDiscovery, SharedMockRegistry};
mod kv_store;
pub use kv_store::KVStoreDiscovery;

mod kube;
pub use kube::{KubeDiscoveryClient, hash_pod_name};

pub mod utils;
use crate::{
    component::{DeviceType, Instance, TransportType},
    pipeline::network::RequestPlanePayloadCodec,
};
pub use utils::watch_and_extract_field;

/// Largest publisher ID exactly representable by float64-backed JSON metadata.
pub(crate) const MAX_JSON_SAFE_PUBLISHER_ID: u64 = (1 << 53) - 1;

/// Transport kind for event plane - used for configuration and env var selection.
///
/// This enum represents the *type* of transport without connection details.
/// Use `EventTransport` when you need the full transport configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EventTransportKind {
    /// NATS Core pub/sub
    Nats,
    /// ZMQ pub/sub
    #[default]
    Zmq,
}

impl std::fmt::Display for EventTransportKind {
    /// Renders the canonical wire/config spelling — the same values
    /// `DYN_EVENT_PLANE` accepts and `#[serde(rename_all = "snake_case")]`
    /// produces — so logs and diagnostics stay greppable against the value an
    /// operator actually configured. Mirrors `RequestPlaneMode`'s `Display`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nats => write!(f, "nats"),
            Self::Zmq => write!(f, "zmq"),
        }
    }
}

impl EventTransportKind {
    /// Parse from environment variable `DYN_EVENT_PLANE`.
    ///
    /// Returns `Zmq` if the variable is not set or is empty: ZMQ is the default
    /// event plane for all backends. NATS remains available as an explicit opt-in
    /// (`DYN_EVENT_PLANE=nats`). When you have access to a runtime, prefer
    /// `DistributedRuntime::default_event_transport_kind`, which resolves the same
    /// default through the configured discovery backend.
    ///
    /// Returns an error for unrecognised values.
    pub fn from_env() -> Result<Self> {
        match std::env::var(crate::config::environment_names::event_plane::DYN_EVENT_PLANE)
            .as_deref()
        {
            Ok("nats") => Ok(Self::Nats),
            Ok("zmq") | Ok("") | Err(_) => Ok(Self::Zmq),
            Ok(other) => anyhow::bail!(
                "Invalid DYN_EVENT_PLANE value '{}'. Valid values: 'nats', 'zmq'",
                other
            ),
        }
    }

    /// Logs a warning if an invalid value is encountered.
    pub fn from_env_or_default() -> Self {
        Self::from_env().unwrap_or_else(|e| {
            tracing::warn!("{e}, defaulting to ZMQ");
            Self::Zmq
        })
    }

    /// Get the default codec for this transport kind.
    /// NATS defaults to JSON, ZMQ defaults to MsgPack.
    pub fn default_codec(&self) -> EventCodecKind {
        match self {
            Self::Nats => EventCodecKind::Json,
            Self::Zmq => EventCodecKind::Msgpack,
        }
    }
}

/// Codec kind for event plane serialization.
///
/// This enum represents the serialization format for event envelopes and payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventCodecKind {
    /// JSON codec - human-readable, good for debugging
    Json,
    /// MessagePack codec - compact binary format
    Msgpack,
}

impl EventCodecKind {
    /// Parse from environment variable `DYN_EVENT_PLANE_CODEC`.
    /// Returns None if not set, allowing transport to select default.
    /// Returns error for invalid values.
    pub fn from_env() -> Result<Option<Self>> {
        match std::env::var(crate::config::environment_names::event_plane::DYN_EVENT_PLANE_CODEC)
            .as_deref()
        {
            Err(_) => Ok(None), // Not set
            Ok("") => Ok(None), // Empty
            Ok("json") => Ok(Some(Self::Json)),
            Ok("msgpack") => Ok(Some(Self::Msgpack)),
            Ok(other) => anyhow::bail!(
                "Invalid DYN_EVENT_PLANE_CODEC value '{}'. Valid values: 'json', 'msgpack'",
                other
            ),
        }
    }

    /// Parse from environment variable with transport-specific default.
    /// Logs a warning if an invalid value is encountered.
    pub fn from_env_or_transport_default(transport: EventTransportKind) -> Self {
        Self::from_env()
            .unwrap_or_else(|e| {
                tracing::warn!(
                    "{}, defaulting to {:?} for {:?}",
                    e,
                    transport.default_codec(),
                    transport
                );
                None
            })
            .unwrap_or_else(|| transport.default_codec())
    }
}

/// Transport configuration for event plane channels.
///
/// This enum carries both the transport kind and its connection configuration.
/// Kept separate from `TransportType` (request plane) to distinguish event semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "config")]
pub enum EventTransport {
    /// NATS Core pub/sub - subject prefix for the channel
    Nats {
        /// Subject prefix (e.g., "namespace.dynamo.component.backend")
        subject_prefix: String,
    },
    /// ZMQ pub/sub - endpoint address (direct mode)
    Zmq {
        /// ZMQ endpoint (e.g., "tcp://host:port")
        endpoint: String,
    },
    /// ZMQ broker endpoints (broker mode) - for discovery of brokers
    ZmqBroker {
        /// XSUB endpoints (publishers connect here)
        xsub_endpoints: Vec<String>,
        /// XPUB endpoints (subscribers connect here)
        xpub_endpoints: Vec<String>,
    },
}

impl EventTransport {
    /// Get the transport kind
    pub fn kind(&self) -> EventTransportKind {
        match self {
            Self::Nats { .. } => EventTransportKind::Nats,
            Self::Zmq { .. } | Self::ZmqBroker { .. } => EventTransportKind::Zmq,
        }
    }

    /// Create a NATS transport with the given subject prefix
    pub fn nats(subject_prefix: impl Into<String>) -> Self {
        Self::Nats {
            subject_prefix: subject_prefix.into(),
        }
    }

    /// Create a ZMQ transport with the given endpoint
    pub fn zmq(endpoint: impl Into<String>) -> Self {
        Self::Zmq {
            endpoint: endpoint.into(),
        }
    }

    /// Get the subject prefix (NATS) or endpoint (ZMQ)
    /// For ZmqBroker, returns the first XSUB endpoint
    pub fn address(&self) -> &str {
        match self {
            Self::Nats { subject_prefix } => subject_prefix,
            Self::Zmq { endpoint } => endpoint,
            Self::ZmqBroker { xsub_endpoints, .. } => {
                xsub_endpoints.first().map(|s| s.as_str()).unwrap_or("")
            }
        }
    }
}

/// Query key for prefix-based discovery queries
/// Supports hierarchical queries from all endpoints down to specific endpoints
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DiscoveryQuery {
    /// Query all endpoints in the system
    AllEndpoints,
    /// Query all endpoints in a specific namespace
    NamespacedEndpoints {
        namespace: String,
    },
    /// Query all endpoints in a namespace/component
    ComponentEndpoints {
        namespace: String,
        component: String,
    },
    /// Query a specific endpoint
    Endpoint {
        namespace: String,
        component: String,
        endpoint: String,
    },
    AllModels,
    NamespacedModels {
        namespace: String,
    },
    ComponentModels {
        namespace: String,
        component: String,
    },
    EndpointModels {
        namespace: String,
        component: String,
        endpoint: String,
    },
    /// Unified event channel query with optional scope filters
    EventChannels(EventChannelQuery),
    /// Semantic event source query with optional scope filters.
    EventSources(EventSourceQuery),
}

/// Scope of an event channel.
///
/// Event scopes are exact ownership domains. A namespace-scoped query does not
/// match component- or endpoint-scoped publishers in that namespace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventScope {
    Namespace {
        name: String,
    },
    Component {
        namespace: String,
        component: String,
    },
    Endpoint {
        endpoint: EndpointId,
    },
}

impl EventScope {
    pub fn namespace(&self) -> &str {
        match self {
            Self::Namespace { name } => name,
            Self::Component { namespace, .. } => namespace,
            Self::Endpoint { endpoint } => &endpoint.namespace,
        }
    }

    pub fn component(&self) -> Option<&str> {
        match self {
            Self::Namespace { .. } => None,
            Self::Component { component, .. } => Some(component),
            Self::Endpoint { endpoint } => Some(&endpoint.component),
        }
    }

    pub fn endpoint(&self) -> Option<&EndpointId> {
        match self {
            Self::Endpoint { endpoint } => Some(endpoint),
            Self::Namespace { .. } | Self::Component { .. } => None,
        }
    }

    /// Canonical NATS/ZMQ-broker subject prefix for this scope.
    pub fn subject_prefix(&self) -> String {
        match self {
            Self::Namespace { name } => {
                format!("namespace.{}", encode_event_segment(name))
            }
            Self::Component {
                namespace,
                component,
            } => format!(
                "namespace.{}.component.{}",
                encode_event_segment(namespace),
                encode_event_segment(component)
            ),
            Self::Endpoint { endpoint } => format!(
                "namespace.{}.component.{}.endpoint.{}",
                encode_event_segment(&endpoint.namespace),
                encode_event_segment(&endpoint.component),
                encode_event_segment(&endpoint.name)
            ),
        }
    }

    /// Canonical subject/routing key for a topic in this exact scope.
    pub fn subject(&self, topic: &str) -> String {
        format!("{}.{}", self.subject_prefix(), encode_event_segment(topic))
    }

    pub(crate) fn path_prefix(&self) -> String {
        match self {
            Self::Namespace { name } => {
                format!("namespace/{}", encode_event_segment(name))
            }
            Self::Component {
                namespace,
                component,
            } => format!(
                "namespace/{}/component/{}",
                encode_event_segment(namespace),
                encode_event_segment(component)
            ),
            Self::Endpoint { endpoint } => format!(
                "namespace/{}/component/{}/endpoint/{}",
                encode_event_segment(&endpoint.namespace),
                encode_event_segment(&endpoint.component),
                encode_event_segment(&endpoint.name)
            ),
        }
    }
}

/// Percent-encode a subject/path segment while keeping common identifier
/// characters readable. The encoding is reversible and prevents NATS wildcard
/// or discovery path delimiters from changing the channel identity.
pub(crate) fn encode_event_segment(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_') {
            encoded.push(char::from(byte));
        } else {
            use std::fmt::Write as _;
            write!(encoded, "%{byte:02X}").expect("writing to String cannot fail");
        }
    }
    encoded
}

fn decode_event_segment(value: &str) -> Result<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }
        if index + 2 >= bytes.len() {
            anyhow::bail!("invalid percent-encoded event segment: {value}");
        }
        let hex = std::str::from_utf8(&bytes[index + 1..index + 3])?;
        decoded
            .push(u8::from_str_radix(hex, 16).map_err(|error| {
                anyhow::anyhow!("invalid event segment escape %{hex}: {error}")
            })?);
        index += 3;
    }
    String::from_utf8(decoded)
        .map_err(|error| anyhow::anyhow!("event segment is not valid UTF-8: {error}"))
}

/// Unified query for event channels with an optional exact scope and topic.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EventChannelQuery {
    /// Exact scope. `None` is reserved for administrative all-channel queries.
    scope: Option<EventScope>,
    topic: Option<String>,
}

impl EventChannelQuery {
    /// Query all event channels (no filters)
    pub fn all() -> Self {
        Self {
            scope: None,
            topic: None,
        }
    }

    /// Query event channels in a specific namespace
    pub fn namespace(namespace: impl Into<String>) -> Self {
        Self {
            scope: Some(EventScope::Namespace {
                name: namespace.into(),
            }),
            topic: None,
        }
    }

    pub fn namespace_topic(namespace: impl Into<String>, topic: impl Into<String>) -> Self {
        Self {
            scope: Some(EventScope::Namespace {
                name: namespace.into(),
            }),
            topic: Some(topic.into()),
        }
    }

    /// Query event channels for a specific component
    pub fn component(namespace: impl Into<String>, component: impl Into<String>) -> Self {
        Self {
            scope: Some(EventScope::Component {
                namespace: namespace.into(),
                component: component.into(),
            }),
            topic: None,
        }
    }

    /// Query event channels for a specific topic
    pub fn topic(
        namespace: impl Into<String>,
        component: impl Into<String>,
        topic: impl Into<String>,
    ) -> Self {
        Self {
            scope: Some(EventScope::Component {
                namespace: namespace.into(),
                component: component.into(),
            }),
            topic: Some(topic.into()),
        }
    }

    pub fn endpoint(endpoint: EndpointId) -> Self {
        Self {
            scope: Some(EventScope::Endpoint { endpoint }),
            topic: None,
        }
    }

    pub fn endpoint_topic(endpoint: EndpointId, topic: impl Into<String>) -> Self {
        Self {
            scope: Some(EventScope::Endpoint { endpoint }),
            topic: Some(topic.into()),
        }
    }

    /// Get the query specificity (0=all, 1=scope, 2=scope+topic).
    pub fn scope_level(&self) -> u8 {
        if self.topic.is_some() {
            2
        } else if self.scope.is_some() {
            1
        } else {
            0
        }
    }
}

/// Unified query for semantic event sources with an optional exact scope and topic.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EventSourceQuery {
    /// Exact scope. `None` is reserved for administrative all-source queries.
    scope: Option<EventScope>,
    topic: Option<String>,
}

impl EventSourceQuery {
    pub fn all() -> Self {
        Self {
            scope: None,
            topic: None,
        }
    }

    pub fn namespace(namespace: impl Into<String>) -> Self {
        Self {
            scope: Some(EventScope::Namespace {
                name: namespace.into(),
            }),
            topic: None,
        }
    }

    pub fn namespace_topic(namespace: impl Into<String>, topic: impl Into<String>) -> Self {
        Self {
            scope: Some(EventScope::Namespace {
                name: namespace.into(),
            }),
            topic: Some(topic.into()),
        }
    }

    pub fn component(namespace: impl Into<String>, component: impl Into<String>) -> Self {
        Self {
            scope: Some(EventScope::Component {
                namespace: namespace.into(),
                component: component.into(),
            }),
            topic: None,
        }
    }

    pub fn topic(
        namespace: impl Into<String>,
        component: impl Into<String>,
        topic: impl Into<String>,
    ) -> Self {
        Self {
            scope: Some(EventScope::Component {
                namespace: namespace.into(),
                component: component.into(),
            }),
            topic: Some(topic.into()),
        }
    }

    pub fn endpoint(endpoint: EndpointId) -> Self {
        Self {
            scope: Some(EventScope::Endpoint { endpoint }),
            topic: None,
        }
    }

    pub fn endpoint_topic(endpoint: EndpointId, topic: impl Into<String>) -> Self {
        Self {
            scope: Some(EventScope::Endpoint { endpoint }),
            topic: Some(topic.into()),
        }
    }

    /// Get the query specificity (0=all, 1=scope, 2=scope+topic).
    pub fn scope_level(&self) -> u8 {
        if self.topic.is_some() {
            2
        } else if self.scope.is_some() {
            1
        } else {
            0
        }
    }
}

/// Specification for registering objects in the discovery plane
/// Represents the input to the register() operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoverySpec {
    /// Endpoint specification for registration
    Endpoint {
        namespace: String,
        component: String,
        endpoint: String,
        /// Transport type and routing information
        transport: TransportType,
        /// Optional execution device for this endpoint instance.
        /// Used by hetero routing to distinguish CPU and CUDA workers.
        device_type: Option<DeviceType>,
        /// Payload codec accepted by this endpoint's request-plane worker.
        /// `None` represents a legacy JSON-only worker.
        request_plane_codec: Option<RequestPlanePayloadCodec>,
    },
    Model {
        namespace: String,
        component: String,
        endpoint: String,
        /// ModelDeploymentCard serialized as JSON
        /// This allows lib/runtime to remain independent of lib/llm types
        /// DiscoverySpec.from_model() and DiscoveryInstance.deserialize_model() are ergonomic helpers to create and deserialize the model card.
        card_json: serde_json::Value,
        /// Optional suffix appended after instance_id in the key path (e.g., for LoRA adapters)
        /// Key format: {namespace}/{component}/{endpoint}/{instance_id}[/{model_suffix}]
        model_suffix: Option<String>,
    },
    /// Event plane channel specification
    /// Used for registering event publishers/subscribers for discovery
    EventChannel {
        scope: EventScope,
        /// Topic name for this channel (e.g., "kv-events", "kv-metrics")
        topic: String,
        /// Unique identity of this publisher incarnation.
        ///
        /// A process can host multiple publishers for the same topic, so event
        /// channels cannot use the process-level discovery instance ID.
        publisher_id: u64,
        /// Event transport type (NATS subject prefix or ZMQ endpoint)
        transport: EventTransport,
    },
    /// Semantic source of events, independent of event transport discovery.
    EventSource {
        scope: EventScope,
        topic: String,
        /// Unique identity of this source incarnation.
        publisher_id: u64,
        /// Domain-specific source descriptor owned by the consuming crate.
        metadata: serde_json::Value,
    },
}

impl DiscoverySpec {
    /// Creates a Model discovery spec from a serializable type
    /// The card will be serialized to JSON to avoid cross-crate dependencies
    pub fn from_model<T>(
        namespace: String,
        component: String,
        endpoint: String,
        card: &T,
    ) -> Result<Self>
    where
        T: Serialize,
    {
        Self::from_model_with_suffix(namespace, component, endpoint, card, None)
    }

    /// Creates a Model discovery spec with an optional suffix (e.g., for LoRA adapters)
    /// The suffix is appended after the instance_id in the key path
    pub fn from_model_with_suffix<T>(
        namespace: String,
        component: String,
        endpoint: String,
        card: &T,
        model_suffix: Option<String>,
    ) -> Result<Self>
    where
        T: Serialize,
    {
        let card_json = serde_json::to_value(card)?;
        Ok(Self::Model {
            namespace,
            component,
            endpoint,
            card_json,
            model_suffix,
        })
    }

    /// Converts this registration spec into a discovery instance.
    ///
    /// Endpoint and model specs use `default_instance_id`, normally the
    /// discovery client's process-level ID. Event channel and source specs
    /// already carry a publisher-level ID, so they use that instead.
    pub fn into_instance(self, default_instance_id: u64) -> DiscoveryInstance {
        match self {
            Self::Endpoint {
                namespace,
                component,
                endpoint,
                transport,
                device_type,
                request_plane_codec,
            } => DiscoveryInstance::Endpoint(crate::component::Instance {
                namespace,
                component,
                endpoint,
                instance_id: default_instance_id,
                transport,
                device_type,
                request_plane_codec,
            }),
            Self::Model {
                namespace,
                component,
                endpoint,
                card_json,
                model_suffix,
            } => DiscoveryInstance::Model {
                namespace,
                component,
                endpoint,
                instance_id: default_instance_id,
                card_json,
                model_suffix,
            },
            Self::EventChannel {
                scope,
                topic,
                publisher_id,
                transport,
            } => DiscoveryInstance::EventChannel {
                scope,
                topic,
                instance_id: publisher_id,
                transport,
            },
            Self::EventSource {
                scope,
                topic,
                publisher_id,
                metadata,
            } => DiscoveryInstance::EventSource {
                scope,
                topic,
                publisher_id,
                metadata,
            },
        }
    }

    /// Compatibility alias for [`DiscoverySpec::into_instance`].
    pub fn with_instance_id(self, default_instance_id: u64) -> DiscoveryInstance {
        self.into_instance(default_instance_id)
    }
}

/// Registered instances in the discovery plane
/// Represents objects that have been successfully registered with an instance ID
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum DiscoveryInstance {
    /// Registered endpoint instance - wraps the component::Instance directly
    Endpoint(crate::component::Instance),
    Model {
        namespace: String,
        component: String,
        endpoint: String,
        instance_id: u64,
        /// ModelDeploymentCard serialized as JSON
        /// This allows lib/runtime to remain independent of lib/llm types
        card_json: serde_json::Value,
        /// Optional suffix appended after instance_id in the key path (e.g., for LoRA adapters)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model_suffix: Option<String>,
    },
    /// Registered event channel instance for event plane pub/sub
    EventChannel {
        scope: EventScope,
        /// Topic name for this channel (e.g., "kv-events", "kv-metrics")
        topic: String,
        instance_id: u64,
        /// Event transport type (NATS subject prefix or ZMQ endpoint)
        transport: EventTransport,
    },
    /// Registered semantic event source.
    EventSource {
        scope: EventScope,
        topic: String,
        publisher_id: u64,
        metadata: serde_json::Value,
    },
}

/// Validate an idempotent registration for one semantic event-source incarnation.
///
/// NOTE: Descriptor immutability belongs to the generic discovery contract. Backends still
/// perform their own atomic lookup/insert, but all of them use this comparison so an identical
/// registration succeeds and a changed descriptor preserves the original record.
pub(crate) fn validate_event_source_reregistration(
    existing: &DiscoveryInstance,
    candidate: &DiscoveryInstance,
) -> Result<()> {
    let DiscoveryInstanceId::EventSource(existing_id) = existing.id() else {
        anyhow::bail!("existing discovery record is not an event source")
    };
    if candidate.id() != DiscoveryInstanceId::EventSource(existing_id.clone()) {
        anyhow::bail!("event source re-registration changed its identity")
    }
    if existing != candidate {
        anyhow::bail!(
            "Event source incarnation '{}' cannot change its descriptor",
            existing_id.to_path()
        )
    }
    Ok(())
}

impl DiscoveryInstance {
    /// Returns the instance ID for this discovery instance
    pub fn instance_id(&self) -> u64 {
        match self {
            Self::Endpoint(inst) => inst.instance_id,
            Self::Model { instance_id, .. } => *instance_id,
            Self::EventChannel { instance_id, .. } => *instance_id,
            Self::EventSource { publisher_id, .. } => *publisher_id,
        }
    }

    /// Deserializes the model JSON into the specified type T
    /// Returns an error if this is not a Model instance or if deserialization fails
    pub fn deserialize_model<T>(&self) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        match self {
            Self::Model { card_json, .. } => Ok(serde_json::from_value(card_json.clone())?),
            Self::Endpoint(_) => {
                anyhow::bail!("Cannot deserialize model from Endpoint instance")
            }
            Self::EventChannel { .. } => {
                anyhow::bail!("Cannot deserialize model from EventChannel instance")
            }
            Self::EventSource { .. } => {
                anyhow::bail!("Cannot deserialize model from EventSource instance")
            }
        }
    }

    /// Extracts the unique identifier for this discovery instance
    /// Used for tracking, diffing, and removal events
    pub fn id(&self) -> DiscoveryInstanceId {
        match self {
            Self::Endpoint(inst) => DiscoveryInstanceId::Endpoint(EndpointInstanceId {
                namespace: inst.namespace.clone(),
                component: inst.component.clone(),
                endpoint: inst.endpoint.clone(),
                instance_id: inst.instance_id,
            }),
            Self::Model {
                namespace,
                component,
                endpoint,
                instance_id,
                model_suffix,
                ..
            } => DiscoveryInstanceId::Model(ModelCardInstanceId {
                namespace: namespace.clone(),
                component: component.clone(),
                endpoint: endpoint.clone(),
                instance_id: *instance_id,
                model_suffix: model_suffix.clone(),
            }),
            Self::EventChannel {
                scope,
                topic,
                instance_id,
                ..
            } => DiscoveryInstanceId::EventChannel(EventChannelInstanceId {
                scope: scope.clone(),
                topic: topic.clone(),
                instance_id: *instance_id,
            }),
            Self::EventSource {
                scope,
                topic,
                publisher_id,
                ..
            } => DiscoveryInstanceId::EventSource(EventSourceInstanceId {
                scope: scope.clone(),
                topic: topic.clone(),
                publisher_id: *publisher_id,
            }),
        }
    }
}

/// Unique identifier for an endpoint instance
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EndpointInstanceId {
    pub namespace: String,
    pub component: String,
    pub endpoint: String,
    pub instance_id: u64,
}

impl EndpointInstanceId {
    /// Converts to a path string.
    pub fn to_path(&self) -> String {
        format!(
            "{}/{}/{}/{:x}",
            self.namespace, self.component, self.endpoint, self.instance_id
        )
    }

    /// Parses an endpoint instance path.
    pub fn from_path(path: &str) -> Result<Self> {
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() != 4 {
            anyhow::bail!(
                "Invalid EndpointInstanceId path: expected 4 parts, got {}",
                parts.len()
            );
        }
        Ok(Self {
            namespace: parts[0].to_string(),
            component: parts[1].to_string(),
            endpoint: parts[2].to_string(),
            instance_id: u64::from_str_radix(parts[3], 16)
                .map_err(|e| anyhow::anyhow!("Invalid instance_id hex: {}", e))?,
        })
    }
}

/// Unique identifier for a model card instance
/// The combination of (namespace, component, endpoint, instance_id, model_suffix) uniquely identifies a model card
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelCardInstanceId {
    pub namespace: String,
    pub component: String,
    pub endpoint: String,
    pub instance_id: u64,
    /// None for base models, Some(slug) for LoRA adapters
    pub model_suffix: Option<String>,
}

/// Unique identifier for an event channel instance
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventChannelInstanceId {
    pub scope: EventScope,
    /// Topic name for this channel (e.g., "kv-events", "kv-metrics")
    pub topic: String,
    pub instance_id: u64,
}

impl EventChannelInstanceId {
    /// Converts to a delimiter-safe path containing the exact channel scope.
    pub fn to_path(&self) -> String {
        format!(
            "{}/topic/{}/{:x}",
            self.scope.path_prefix(),
            encode_event_segment(&self.topic),
            self.instance_id
        )
    }

    /// Parses a path produced by [`Self::to_path`].
    pub fn from_path(path: &str) -> Result<Self> {
        let parts: Vec<&str> = path.split('/').collect();
        let (scope, topic_index, instance_index) = match parts.as_slice() {
            ["namespace", namespace, "topic", _, _] => (
                EventScope::Namespace {
                    name: decode_event_segment(namespace)?,
                },
                3,
                4,
            ),
            [
                "namespace",
                namespace,
                "component",
                component,
                "topic",
                _,
                _,
            ] => (
                EventScope::Component {
                    namespace: decode_event_segment(namespace)?,
                    component: decode_event_segment(component)?,
                },
                5,
                6,
            ),
            [
                "namespace",
                namespace,
                "component",
                component,
                "endpoint",
                endpoint,
                "topic",
                _,
                _,
            ] => (
                EventScope::Endpoint {
                    endpoint: EndpointId {
                        namespace: decode_event_segment(namespace)?,
                        component: decode_event_segment(component)?,
                        name: decode_event_segment(endpoint)?,
                    },
                },
                7,
                8,
            ),
            _ => anyhow::bail!("invalid EventChannelInstanceId path: {path}"),
        };
        Ok(Self {
            scope,
            topic: decode_event_segment(parts[topic_index])?,
            instance_id: u64::from_str_radix(parts[instance_index], 16)
                .map_err(|e| anyhow::anyhow!("Invalid instance_id hex: {}", e))?,
        })
    }
}

/// Unique identifier for a semantic event source incarnation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventSourceInstanceId {
    pub scope: EventScope,
    pub topic: String,
    pub publisher_id: u64,
}

impl EventSourceInstanceId {
    /// Converts to a delimiter-safe path containing the exact source scope.
    pub fn to_path(&self) -> String {
        format!(
            "{}/topic/{}/{:x}",
            self.scope.path_prefix(),
            encode_event_segment(&self.topic),
            self.publisher_id
        )
    }

    /// Parses a path produced by [`Self::to_path`].
    pub fn from_path(path: &str) -> Result<Self> {
        let channel_id = EventChannelInstanceId::from_path(path)
            .with_context(|| format!("invalid EventSourceInstanceId path: {path}"))?;
        Ok(Self {
            scope: channel_id.scope,
            topic: channel_id.topic,
            publisher_id: channel_id.instance_id,
        })
    }
}

impl ModelCardInstanceId {
    /// Converts to a path string: `{namespace}/{component}/{endpoint}/{instance_id:x}[/{model_suffix}]`
    pub fn to_path(&self) -> String {
        match &self.model_suffix {
            Some(suffix) => format!(
                "{}/{}/{}/{:x}/{}",
                self.namespace, self.component, self.endpoint, self.instance_id, suffix
            ),
            None => format!(
                "{}/{}/{}/{:x}",
                self.namespace, self.component, self.endpoint, self.instance_id
            ),
        }
    }

    /// Parses from a path string: `{namespace}/{component}/{endpoint}/{instance_id:x}[/{model_suffix}]`
    pub fn from_path(path: &str) -> Result<Self> {
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() < 4 || parts.len() > 5 {
            anyhow::bail!(
                "Invalid ModelCardInstanceId path: expected 4 or 5 parts, got {}",
                parts.len()
            );
        }
        Ok(Self {
            namespace: parts[0].to_string(),
            component: parts[1].to_string(),
            endpoint: parts[2].to_string(),
            instance_id: u64::from_str_radix(parts[3], 16)
                .map_err(|e| anyhow::anyhow!("Invalid instance_id hex: {}", e))?,
            model_suffix: parts.get(4).map(|s| s.to_string()),
        })
    }
}

/// Union of instance identifiers for different discovery object types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiscoveryInstanceId {
    Endpoint(EndpointInstanceId),
    Model(ModelCardInstanceId),
    EventChannel(EventChannelInstanceId),
    EventSource(EventSourceInstanceId),
}

impl DiscoveryInstanceId {
    /// Returns the raw instance_id regardless of variant type
    pub fn instance_id(&self) -> u64 {
        match self {
            Self::Endpoint(eid) => eid.instance_id,
            Self::Model(mid) => mid.instance_id,
            Self::EventChannel(ecid) => ecid.instance_id,
            Self::EventSource(esid) => esid.publisher_id,
        }
    }

    /// Extracts the EndpointInstanceId, returning an error if this is a Model or EventChannel variant
    pub fn extract_endpoint_id(&self) -> Result<&EndpointInstanceId> {
        match self {
            Self::Endpoint(eid) => Ok(eid),
            Self::Model(_) => anyhow::bail!("Expected Endpoint variant, got Model"),
            Self::EventChannel(_) => anyhow::bail!("Expected Endpoint variant, got EventChannel"),
            Self::EventSource(_) => anyhow::bail!("Expected Endpoint variant, got EventSource"),
        }
    }

    /// Extracts the ModelCardInstanceId, returning an error if this is an Endpoint or EventChannel variant
    pub fn extract_model_id(&self) -> Result<&ModelCardInstanceId> {
        match self {
            Self::Model(mid) => Ok(mid),
            Self::Endpoint(_) => anyhow::bail!("Expected Model variant, got Endpoint"),
            Self::EventChannel(_) => anyhow::bail!("Expected Model variant, got EventChannel"),
            Self::EventSource(_) => anyhow::bail!("Expected Model variant, got EventSource"),
        }
    }

    /// Extracts the EventChannelInstanceId, returning an error if this is an Endpoint or Model variant
    pub fn extract_event_channel_id(&self) -> Result<&EventChannelInstanceId> {
        match self {
            Self::EventChannel(ecid) => Ok(ecid),
            Self::Endpoint(_) => anyhow::bail!("Expected EventChannel variant, got Endpoint"),
            Self::Model(_) => anyhow::bail!("Expected EventChannel variant, got Model"),
            Self::EventSource(_) => {
                anyhow::bail!("Expected EventChannel variant, got EventSource")
            }
        }
    }

    /// Extracts the EventSourceInstanceId, returning an error for other variants.
    pub fn extract_event_source_id(&self) -> Result<&EventSourceInstanceId> {
        match self {
            Self::EventSource(esid) => Ok(esid),
            Self::Endpoint(_) => anyhow::bail!("Expected EventSource variant, got Endpoint"),
            Self::Model(_) => anyhow::bail!("Expected EventSource variant, got Model"),
            Self::EventChannel(_) => {
                anyhow::bail!("Expected EventSource variant, got EventChannel")
            }
        }
    }
}

/// Events emitted by the discovery watch stream
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryEvent {
    /// A new instance was added.
    ///
    /// Endpoint watches also emit this event when the endpoint data changes without changing its
    /// [`DiscoveryInstanceId`]. Consumers of endpoint watches must replace the previous value.
    Added(DiscoveryInstance),
    /// The complete normalized taint set for an existing model card changed.
    ModelTaintsUpdated(ModelTaintsUpdate),
    /// An instance was removed (identified by its unique ID)
    Removed(DiscoveryInstanceId),
}

/// A scoped, idempotent update to an existing model card's routing taints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelTaintsUpdate {
    pub id: ModelCardInstanceId,
    pub taints: Vec<String>,
}

/// Stream type for discovery events
pub type DiscoveryStream = Pin<Box<dyn Stream<Item = Result<DiscoveryEvent>> + Send>>;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ModelRegistrationIdentity {
    display_name: String,
    source_path: Option<String>,
    is_lora: bool,
}

impl ModelRegistrationIdentity {
    fn base_identity(&self) -> &str {
        self.source_path.as_deref().unwrap_or(&self.display_name)
    }

    fn is_compatible_with(&self, other: &Self) -> bool {
        if self.is_lora || other.is_lora {
            self.base_identity() == other.base_identity()
        } else {
            self.display_name == other.display_name
        }
    }
}

fn extract_model_registration_identity(
    card_json: &serde_json::Value,
    model_suffix: Option<&str>,
) -> Result<ModelRegistrationIdentity> {
    let display_name = card_json
        .get("display_name")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            anyhow::anyhow!("failed to deserialize model display_name from card_json")
        })?;
    let source_path = card_json
        .get("source_path")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let is_lora =
        model_suffix.is_some() || card_json.get("lora").is_some_and(|value| !value.is_null());

    Ok(ModelRegistrationIdentity {
        display_name,
        source_path,
        is_lora,
    })
}

fn find_conflicting_model_name(
    instances: &[DiscoveryInstance],
    requested_identity: &ModelRegistrationIdentity,
) -> Result<Option<String>> {
    for instance in instances {
        if let DiscoveryInstance::Model {
            card_json,
            model_suffix,
            ..
        } = instance
        {
            let existing_identity =
                extract_model_registration_identity(card_json, model_suffix.as_deref())?;
            if !requested_identity.is_compatible_with(&existing_identity) {
                return Ok(Some(existing_identity.display_name));
            }
        }
    }

    Ok(None)
}

const TOPOLOGY_TAINT_PREFIX: &str = "dynamo.topology/";

fn model_card_without_taints(
    instance: &DiscoveryInstance,
) -> Result<(serde_json::Value, HashSet<String>)> {
    let DiscoveryInstance::Model { card_json, .. } = instance else {
        anyhow::bail!("model update requires a model discovery instance")
    };

    let mut card = card_json.clone();
    let runtime_config = card
        .get_mut("runtime_config")
        .and_then(serde_json::Value::as_object_mut)
        .context("model card is missing runtime_config")?;
    let taints = runtime_config
        .remove("taints")
        .unwrap_or_else(|| serde_json::Value::Array(Vec::new()));
    let taints = taints
        .as_array()
        .context("model card runtime_config.taints must be an array")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .context("model card runtime_config.taints entries must be strings")
        })
        .collect::<Result<HashSet<_>>>()?;

    Ok((card, taints))
}

fn expected_topology_taints(card: &serde_json::Value) -> Result<HashSet<String>> {
    let Some(domains) = card
        .pointer("/runtime_config/topology_domains")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(HashSet::new());
    };

    domains
        .iter()
        .map(|(domain, value)| {
            let value = value
                .as_str()
                .context("model card runtime_config.topology_domains values must be strings")?;
            Ok(format!(
                "{TOPOLOGY_TAINT_PREFIX}{}={}",
                domain.trim(),
                value.trim()
            ))
        })
        .collect()
}

/// Validate the discovery-layer mutable boundary for model-card updates.
///
/// Model cards remain immutable after registration except for worker-managed
/// `runtime_config.taints`. Reserved topology taints are derived from the
/// immutable `topology_domains` map and must stay canonical.
#[derive(Debug)]
struct ValidatedModelTaintUpdate {
    existing_taints: HashSet<String>,
    candidate_taints: HashSet<String>,
}

fn validate_model_taint_update(
    existing: &DiscoveryInstance,
    candidate: &DiscoveryInstance,
) -> Result<ValidatedModelTaintUpdate> {
    if existing.id() != candidate.id() {
        anyhow::bail!("model update cannot change discovery identity")
    }

    let (existing_card, existing_taints) = model_card_without_taints(existing)?;
    let (candidate_card, candidate_taints) = model_card_without_taints(candidate)?;
    if existing_card != candidate_card {
        anyhow::bail!("model update can only change runtime_config.taints")
    }

    let expected_topology = expected_topology_taints(&candidate_card)?;
    let actual_topology = candidate_taints
        .iter()
        .filter(|taint| taint.starts_with(TOPOLOGY_TAINT_PREFIX))
        .cloned()
        .collect::<HashSet<_>>();
    if actual_topology != expected_topology {
        anyhow::bail!(
            "reserved {TOPOLOGY_TAINT_PREFIX} taints must match runtime_config.topology_domains"
        )
    }

    Ok(ValidatedModelTaintUpdate {
        existing_taints,
        candidate_taints,
    })
}

/// Validate a same-ID model registration replay without replacing authoritative taints.
pub(crate) fn validate_model_reregistration(
    existing: &DiscoveryInstance,
    candidate: &DiscoveryInstance,
) -> Result<()> {
    validate_model_taint_update(existing, candidate).map(|_| ())
}

fn sorted_taints(taints: HashSet<String>) -> Vec<String> {
    let mut taints = taints.into_iter().collect::<Vec<_>>();
    taints.sort_unstable();
    taints
}

/// Classify a discovery value transition without widening `Added` into an upsert.
///
/// Model cards are immutable after registration except for `runtime_config.taints`.
/// Endpoints retain their existing replacement-as-Added behavior. Same-ID changes
/// to other discovery object types remain ignored.
pub(crate) fn classify_discovery_change(
    existing: Option<&DiscoveryInstance>,
    candidate: &DiscoveryInstance,
) -> Result<Option<DiscoveryEvent>> {
    let Some(existing) = existing else {
        return Ok(Some(DiscoveryEvent::Added(candidate.clone())));
    };

    if existing == candidate {
        return Ok(None);
    }

    if matches!(existing, DiscoveryInstance::Model { .. })
        && matches!(candidate, DiscoveryInstance::Model { .. })
    {
        let ValidatedModelTaintUpdate {
            existing_taints,
            candidate_taints,
        } = validate_model_taint_update(existing, candidate)?;
        if existing_taints == candidate_taints {
            return Ok(None);
        }

        let DiscoveryInstanceId::Model(id) = candidate.id() else {
            unreachable!("model discovery instance must have a model id")
        };
        return Ok(Some(DiscoveryEvent::ModelTaintsUpdated(
            ModelTaintsUpdate {
                id,
                taints: sorted_taints(candidate_taints),
            },
        )));
    }

    if matches!(candidate, DiscoveryInstance::Endpoint(_)) {
        Ok(Some(DiscoveryEvent::Added(candidate.clone())))
    } else {
        Ok(None)
    }
}

/// Reconcile an authoritative snapshot while retaining the last valid value for
/// any model card that attempts an immutable mutation.
pub(crate) fn reconcile_discovery_snapshot(
    known: &HashMap<DiscoveryInstanceId, DiscoveryInstance>,
    current: HashMap<DiscoveryInstanceId, DiscoveryInstance>,
) -> (
    Vec<DiscoveryEvent>,
    HashMap<DiscoveryInstanceId, DiscoveryInstance>,
) {
    let mut events = known
        .keys()
        .filter(|id| !current.contains_key(*id))
        .cloned()
        .map(DiscoveryEvent::Removed)
        .collect::<Vec<_>>();
    let mut next = HashMap::with_capacity(current.len());

    for (id, candidate) in current {
        match classify_discovery_change(known.get(&id), &candidate) {
            Ok(Some(event)) => {
                events.push(event);
                next.insert(id, candidate);
            }
            Ok(None) => {
                let retained = known.get(&id).cloned().unwrap_or(candidate);
                next.insert(id, retained);
            }
            Err(error) => {
                tracing::error!(
                    ?id,
                    %error,
                    "Rejecting immutable discovery model-card mutation"
                );
                if let Some(existing) = known.get(&id) {
                    next.insert(id, existing.clone());
                }
            }
        }
    }

    (events, next)
}

fn model_with_updated_taints(
    existing: &DiscoveryInstance,
    mut taints: HashSet<String>,
) -> Result<DiscoveryInstance> {
    if let Some(taint) = taints
        .iter()
        .find(|taint| taint.starts_with(TOPOLOGY_TAINT_PREFIX))
    {
        anyhow::bail!("taint '{taint}' uses reserved prefix '{TOPOLOGY_TAINT_PREFIX}'")
    }

    let (card_without_taints, existing_taints) = model_card_without_taints(existing)?;
    taints.extend(expected_topology_taints(&card_without_taints)?);
    if taints == existing_taints {
        return Ok(existing.clone());
    }

    let mut candidate = existing.clone();
    let DiscoveryInstance::Model { card_json, .. } = &mut candidate else {
        anyhow::bail!("model taint update requires a model discovery instance")
    };
    let runtime_config = card_json
        .get_mut("runtime_config")
        .and_then(serde_json::Value::as_object_mut)
        .context("model card is missing runtime_config")?;
    runtime_config.insert(
        "taints".to_string(),
        serde_json::Value::Array(
            sorted_taints(taints)
                .into_iter()
                .map(serde_json::Value::String)
                .collect(),
        ),
    );
    Ok(candidate)
}

/// Discovery trait for service discovery across different backends
#[async_trait]
pub trait Discovery: Send + Sync {
    /// Returns a unique identifier for this worker (e.g lease id if using etcd or generated id for memory store)
    /// Endpoint and model objects created by this worker use this ID. Event
    /// channels and sources use a publisher-level ID because a worker can own
    /// more than one publisher for the same topic.
    fn instance_id(&self) -> u64;

    /// Registers an object in the discovery plane with the instance id
    async fn register(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
        let (namespace, component, endpoint, requested_identity) = match &spec {
            DiscoverySpec::Model {
                namespace,
                component,
                endpoint,
                card_json,
                model_suffix,
                ..
            } => (
                namespace.clone(),
                component.clone(),
                endpoint.clone(),
                extract_model_registration_identity(card_json, model_suffix.as_deref())?,
            ),
            _ => return self.register_internal(spec).await,
        };

        let query = DiscoveryQuery::EndpointModels {
            namespace: namespace.clone(),
            component: component.clone(),
            endpoint: endpoint.clone(),
        };

        if let Some(conflicting_name) =
            find_conflicting_model_name(&self.list(query.clone()).await?, &requested_identity)?
        {
            let requested_name = &requested_identity.display_name;
            anyhow::bail!(
                "Cannot register model '{requested_name}' on endpoint '{namespace}/{component}/{endpoint}': a different model '{conflicting_name}' is already registered there"
            );
        }

        let instance = self.register_internal(spec).await?;

        if let Some(conflicting_name) =
            find_conflicting_model_name(&self.list(query).await?, &requested_identity)?
        {
            let requested_name = &requested_identity.display_name;
            if let Err(unregister_err) = self.unregister(instance.clone()).await {
                return Err(anyhow::anyhow!(
                    "Cannot register model '{requested_name}' on endpoint '{namespace}/{component}/{endpoint}': a different model '{conflicting_name}' is already registered there"
                ))
                .context(format!(
                    "failed to roll back conflicting model registration for instance {instance_id}: {unregister_err}",
                    instance_id = instance.instance_id()
                ));
            }

            anyhow::bail!(
                "Cannot register model '{requested_name}' on endpoint '{namespace}/{component}/{endpoint}': a different model '{conflicting_name}' is already registered there"
            );
        }

        Ok(instance)
    }

    /// Backend-specific raw registration implementation.
    async fn register_internal(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance>;

    /// Replace the caller-managed taints of this worker's existing base model card.
    async fn update_model_taints(
        &self,
        id: ModelCardInstanceId,
        taints: HashSet<String>,
    ) -> Result<()> {
        if id.instance_id != self.instance_id() {
            anyhow::bail!(
                "cannot update model taints for worker {}; this discovery client owns worker {}",
                id.instance_id,
                self.instance_id()
            )
        }
        if id.model_suffix.is_some() {
            anyhow::bail!("model taint updates are supported only for base model cards")
        }
        self.update_model_taints_internal(id, taints).await
    }

    /// Backend-specific authoritative read, taint-only mutation, and persistence.
    async fn update_model_taints_internal(
        &self,
        _id: ModelCardInstanceId,
        _taints: HashSet<String>,
    ) -> Result<()> {
        anyhow::bail!("model taint updates are not supported by this discovery backend")
    }

    /// Unregisters an instance from the discovery plane
    async fn unregister(&self, instance: DiscoveryInstance) -> Result<()>;

    /// Returns a list of currently registered instances for the given discovery query
    /// This is a one-time snapshot without watching for changes
    async fn list(&self, query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>>;

    /// Returns a stream of discovery events (Added/Removed) for the given discovery query
    /// The optional cancellation token can be used to stop the watch stream
    async fn list_and_watch(
        &self,
        query: DiscoveryQuery,
        cancel_token: Option<CancellationToken>,
    ) -> Result<DiscoveryStream>;

    /// Clean up resources held by this discovery backend.
    /// For KV store backends, this deletes owned registrations immediately rather than
    /// waiting for TTL expiry. Default is a no-op for backends that don't need cleanup.
    fn shutdown(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_channel_id_path_round_trips_reserved_segments() {
        let id = EventChannelInstanceId {
            scope: EventScope::Endpoint {
                endpoint: EndpointId {
                    namespace: "ns.with/slash".to_string(),
                    component: "component.*".to_string(),
                    name: "endpoint.>/%".to_string(),
                },
            },
            topic: "kv.events/>".to_string(),
            instance_id: 0xfeed,
        };

        let path = id.to_path();
        assert!(!path.contains("ns.with/slash"));
        assert_eq!(EventChannelInstanceId::from_path(&path).unwrap(), id);
    }

    #[test]
    fn endpoint_source_id_path_round_trips_reserved_segments() {
        let id = EventSourceInstanceId {
            scope: EventScope::Endpoint {
                endpoint: EndpointId {
                    namespace: "ns.with/slash".to_string(),
                    component: "component.*".to_string(),
                    name: "endpoint.>/%".to_string(),
                },
            },
            topic: "kv.events/>".to_string(),
            publisher_id: 0xfeed,
        };

        let path = id.to_path();
        assert!(!path.contains("ns.with/slash"));
        assert_eq!(EventSourceInstanceId::from_path(&path).unwrap(), id);
    }

    #[test]
    fn endpoint_codec_metadata_round_trips_and_defaults_when_omitted() {
        let instance = DiscoverySpec::Endpoint {
            namespace: "default".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            transport: TransportType::Nats("worker.generate".to_string()),
            device_type: None,
            request_plane_codec: Some(RequestPlanePayloadCodec::Msgpack),
        }
        .into_instance(42);

        let mut metadata = serde_json::to_value(&instance).unwrap();
        assert_eq!(metadata["request_plane_codec"], "msgpack");
        let round_trip: DiscoveryInstance = serde_json::from_value(metadata.clone()).unwrap();
        match round_trip {
            DiscoveryInstance::Endpoint(instance) => assert_eq!(
                instance.request_plane_codec,
                Some(RequestPlanePayloadCodec::Msgpack)
            ),
            _ => panic!("expected endpoint discovery metadata"),
        }

        metadata
            .as_object_mut()
            .unwrap()
            .remove("request_plane_codec");
        let legacy: DiscoveryInstance = serde_json::from_value(metadata).unwrap();
        match legacy {
            DiscoveryInstance::Endpoint(instance) => {
                assert_eq!(instance.request_plane_codec, None)
            }
            _ => panic!("expected endpoint discovery metadata"),
        }
    }
}

#[cfg(test)]
mod model_taint_update_tests {
    use super::*;

    fn model_instance(taints: &[&str]) -> DiscoveryInstance {
        DiscoveryInstance::Model {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            instance_id: 7,
            card_json: serde_json::json!({
                "display_name": "model",
                "runtime_config": {
                    "taints": taints,
                    "topology_domains": {"zone": "west"}
                }
            }),
            model_suffix: None,
        }
    }

    #[test]
    fn model_update_accepts_only_caller_managed_taint_changes() {
        let existing = model_instance(&["old", "dynamo.topology/zone=west"]);
        let candidate = model_instance(&["new", "dynamo.topology/zone=west"]);

        validate_model_taint_update(&existing, &candidate).unwrap();
    }

    #[test]
    fn model_update_rejects_immutable_card_changes() {
        let existing = model_instance(&["dynamo.topology/zone=west"]);
        let mut candidate = model_instance(&["dynamo.topology/zone=west"]);
        let DiscoveryInstance::Model { card_json, .. } = &mut candidate else {
            unreachable!()
        };
        card_json["display_name"] = serde_json::json!("other-model");

        let error = validate_model_taint_update(&existing, &candidate).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("can only change runtime_config.taints")
        );
    }

    #[test]
    fn model_update_rejects_reserved_topology_taint_changes() {
        let existing = model_instance(&["dynamo.topology/zone=west"]);
        let candidate = model_instance(&["dynamo.topology/zone=east"]);

        let error = validate_model_taint_update(&existing, &candidate).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must match runtime_config.topology_domains")
        );
    }

    #[test]
    fn changed_taints_are_classified_as_a_scoped_normalized_event() {
        let existing = model_instance(&["old", "dynamo.topology/zone=west"]);
        let candidate = model_instance(&["gpu", "blue", "dynamo.topology/zone=west"]);
        let DiscoveryInstanceId::Model(id) = candidate.id() else {
            unreachable!()
        };

        assert_eq!(
            classify_discovery_change(Some(&existing), &candidate).unwrap(),
            Some(DiscoveryEvent::ModelTaintsUpdated(ModelTaintsUpdate {
                id,
                taints: vec![
                    "blue".to_string(),
                    "dynamo.topology/zone=west".to_string(),
                    "gpu".to_string(),
                ],
            }))
        );
    }

    #[test]
    fn taint_order_only_changes_are_no_ops() {
        let existing = model_instance(&["gpu", "dynamo.topology/zone=west"]);
        let candidate = model_instance(&["dynamo.topology/zone=west", "gpu"]);

        assert_eq!(
            classify_discovery_change(Some(&existing), &candidate).unwrap(),
            None
        );
    }

    #[test]
    fn update_api_derives_topology_taints_and_rejects_reserved_input() {
        let existing = model_instance(&["old", "dynamo.topology/zone=west"]);
        let updated =
            model_with_updated_taints(&existing, HashSet::from(["new".to_string()])).unwrap();
        let (_, taints) = model_card_without_taints(&updated).unwrap();
        assert_eq!(
            taints,
            HashSet::from(["new".to_string(), "dynamo.topology/zone=west".to_string()])
        );
        assert!(
            model_with_updated_taints(
                &existing,
                HashSet::from(["dynamo.topology/zone=east".to_string()])
            )
            .is_err()
        );
    }
}
